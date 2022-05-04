import sys
assert sys.version_info >= (3, 7), "This code requires python version >= 3.7"
import functools
import time
import click
import torch
import pprint
import tops
import tqdm
from pathlib import Path
from ssd.evaluate import evaluate
from ssd import utils
from tops.config import instantiate
from tops import logger, checkpointer
from torch.optim.lr_scheduler import ChainedScheduler
from omegaconf import OmegaConf
torch.backends.cudnn.benchmark = True

def train_epoch(
        model, scaler: torch.cuda.amp.GradScaler,
        optim, dataloader_train, scheduler,
        gpu_transform: torch.nn.Module,
        log_interval: int):
    """
    Train one epoch.
    Args:
        model: model to be trained
        scaler: scaler for gradient accumulation
        optim: optimizer
        dataloader_train: dataloader for training
        scheduler: scheduler
        gpu_transform: gpu transform
        log_interval: interval for logging
    """

    grad_scale = scaler.get_scale()
    for batch in tqdm.tqdm(dataloader_train, f"Epoch {logger.epoch()}"):

        # Transform batch to cuda if available
        batch = tops.to_cuda(batch)

        # Get labels
        batch["labels"] = batch["labels"].long()
        batch = gpu_transform(batch)

        # Autocast to improve performance, if available
        with torch.cuda.amp.autocast(enabled=tops.amp()):

            # Perform forward pass
            # In training this returns bbox_delta and confidences
            bbox_delta, confs = model(batch["image"])

            # Compute loss
            loss, to_log = model.loss_func(bbox_delta, confs, batch["boxes"], batch["labels"])

        # Perform backward pass
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

        # If gradient scaling is same, get scheduler step
        if grad_scale == scaler.get_scale():
            scheduler.step()
            if logger.global_step() % log_interval:

                # Log learning rate
                logger.add_scalar("stats/learning_rate", scheduler._schedulers[-1].get_last_lr()[-1])
        else:
            # Get gradient scaler
            grad_scale = scaler.get_scale()

            # Log gradient scale
            logger.add_scalar("amp/grad_scale", scaler.get_scale())

        if logger.global_step() % log_interval == 0:
            # Log loss
            to_log = {f"loss/{k}": v.mean().cpu().item() for k, v in to_log.items()}
            logger.add_dict(to_log)

        # torch.cuda.amp skips gradient steps if backward pass produces NaNs/infs.
        # If it happens in the first iteration, scheduler.step() will throw exception
        logger.step()

    #return


def print_config(cfg):
    container = OmegaConf.to_container(cfg)
    pp = pprint.PrettyPrinter(indent=2, compact=False)
    print("--------------------Config file below--------------------")
    pp.pprint(container)
    print("--------------------End of config file--------------------")


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--evaluate-only", default=False, is_flag=True, help="Only run evaluation, no training.")
@click.option("--verbose", default=False, is_flag=True, help="Print config.")
@click.option("--save_graph", default=False, is_flag=True, help="Save graph to tensorboard.")
def train(config_path: Path, evaluate_only: bool, verbose=False, save_graph=False):
    logger.logger.DEFAULT_SCALAR_LEVEL = logger.logger.DEBUG
    cfg = utils.load_config(config_path)

    if verbose:
        print_config(cfg)

    tops.init(cfg.output_dir)
    tops.set_amp(cfg.train.amp)  # Set AMP mode
    tops.set_seed(cfg.train.seed)
    dataloader_train = instantiate(cfg.data_train.dataloader)
    dataloader_val = instantiate(cfg.data_val.dataloader)

    # Load gt boxes in validation set
    coco_gt = dataloader_val.dataset.get_annotations_as_coco()

    # Model to cuda if available
    model = tops.to_cuda(instantiate(cfg.model))
    optimizer = instantiate(cfg.optimizer, params=utils.tencent_trick(model))
    scheduler = ChainedScheduler(instantiate(list(cfg.schedulers.values()), optimizer=optimizer))

    # Register model to checkpointer
    checkpointer.register_models(
        dict(model=model, optimizer=optimizer, scheduler=scheduler))

    # Global time counter
    total_time = 0

    # Load checkpoint if exists
    if checkpointer.has_checkpoint():
        train_state = checkpointer.load_registered_models(load_best=False)
        total_time = train_state["total_time"]
        logger.log(f"Resuming train from: epoch: {logger.epoch()}, global step: {logger.global_step()}")

    # Instantiate gpu transform for train and validation set
    gpu_transform_val = instantiate(cfg.data_val.gpu_transform)
    gpu_transform_train = instantiate(cfg.data_train.gpu_transform)

    # Wrap torch evaluate in a function to be able to use it in a tqdm loop
    evaluation_fn = functools.partial(
        evaluate,
        model=model,
        dataloader=dataloader_val,
        cocoGt=coco_gt,
        gpu_transform=gpu_transform_val,
        label_map=cfg.label_map
    )

    # If only evaluation is requested, run evaluation and exit
    if evaluate_only:
        evaluation_fn()
        exit()

    # Initialize gradient scaler
    scaler = torch.cuda.amp.GradScaler(enabled=tops.amp())

    # Create dummy input to print model summary
    dummy_input = tops.to_cuda(torch.randn(1, cfg.train.image_channels, *cfg.train.imshape))
    tops.print_module_summary(model, (dummy_input,))

    # Save graph for tensorboard, if requested
    if save_graph:
        logger.add_graph(model, dummy_input)

    # Start training
    start_epoch = logger.epoch()
    for _ in range(start_epoch, cfg.train.epochs):
        start_epoch_time = time.time()

        # Perform training epoch
        train_epoch(model, scaler, optimizer, dataloader_train, scheduler, gpu_transform_train, cfg.train.log_interval)
        end_epoch_time = time.time() - start_epoch_time
        total_time += end_epoch_time

        # Log end of epoch time
        logger.add_scalar("stats/epoch_time", end_epoch_time)

        eval_stats = evaluation_fn()
        eval_stats = {f"metrics/{key}": val for key, val in eval_stats.items()}

        # Log evaluation stats
        logger.add_dict(eval_stats, level=logger.logger.INFO)

        # Save checkpoint
        train_state = dict(total_time=total_time)
        checkpointer.save_registered_models(train_state)
        logger.step_epoch()
    logger.add_scalar("stats/total_time", total_time)


if __name__ == "__main__":
    train()

    # TODO: this seems rad
    # import os
    # os.system('say "your program has finished"')
