import torch
import matplotlib.pyplot as plt

from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm


def get_config(config_path):
    cfg = LazyConfig.load(config_path)
    cfg.train.batch_size = 1
    return cfg


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader


def analyze_something(dataloader, cfg):
    for batch in tqdm(dataloader):
        # Remove the two lines below and start analyzing :D
        print("The keys in the batch are:", batch.keys())
        # exit()


def statistics(dataloader, cfg):
    rgb_tensor = torch.tensor([0.0, 0.0, 0.0])
    nr_of_pixels = len(dataloader) * cfg.train.imshape[0] * cfg.train.imshape[1]
    for (images, _, _) in dataloader:
        rgb_tensor += torch.sum(images, (0, 2, 3))

    mu_tensor = rgb_tensor / nr_of_pixels

    sse_tensor = torch.tensor([0.0, 0.0, 0.0])
    for (images, _, _) in dataloader:
        for image in images:
            assert len(sse_tensor) == 3
            for i in range(len(sse_tensor)):
                sse_tensor[i] += torch.sum((image[i] - mu_tensor[i]) ** 2)

    sigmas = torch.sqrt(sse_tensor / nr_of_pixels)

    return mu_tensor, sigmas


def analyze_distribution(dataloader, cfg):
    labels = cfg.label_map
    aspect_ratios = dict.fromkeys(labels.keys())
    areas = dict.fromkeys(labels.keys())


def analyze_bounding_boxes(dataloader):
    pass


def get_next_image(dataloader):
    return next(iter(dataloader))


def plot_and_save_single_image(tensor, plot=False, save=False):
    img = tensor['image'][0]
    if plot:
        fig, ax = plt.figure(figsize=(18, 2))
        ax.plt(img.permute(1, 2, 0), cmap='grey')
        plt.show()
    if save:
        pass



def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)

    statistics(dataloader)


if __name__ == '__main__':
    main()
