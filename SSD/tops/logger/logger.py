import atexit
import json
import logging
from abc import ABC, abstractmethod
from argparse import ArgumentError
from pathlib import Path
import pathlib
from typing import List
from torch.utils import tensorboard

_global_step = 0
_epoch = 0


INFO = logging.INFO
WARN = logging.WARN
DEBUG = logging.DEBUG
supported_backends = ["stdout", "json", "tensorboard"]
_output_dir = None

DEFAULT_SCALAR_LEVEL = DEBUG
DEFAULT_LOG_LEVEL = INFO
DEFAULT_LOGGER_LEVEL = INFO
DEFAULT_GRAPH_LEVEL = INFO

class Backend(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def add_scalar(self, tag, value, **kwargs):
        pass

    @abstractmethod
    def add_graph(self, model, input_to_model):
        """
        Add model graph to the backend.
        Args:
            model: model to be added
            input_to_model: input to the model (can be dummy data)
        """
        pass

    def add_dict(self, values, **kwargs):
        for tag, value in values.items():
            self.add_scalar(tag, value, **kwargs)

    def log(self, msg, level):
        """
        Log a message.
        Args:
            msg: message to be logged
            level: logging level
        """
        pass

    def finish(self):
        """
        Finish logging by closing IO streams, etc.
        """
        pass


class TensorBoardBackend(Backend):

    def __init__(self, output_dir: Path):
        output_dir.mkdir(exist_ok=True, parents=True)
        self.writer = tensorboard.SummaryWriter(log_dir=output_dir)
        self.closed = False
    
    def add_scalar(self, tag, value, **kwargs):
        self.writer.add_scalar(tag, value, new_style=True, global_step=_global_step)


    def add_graph(self, model, input_to_model, **kwargs):
        # Note: all must produce traceable outputs for this to work.
        self.writer.add_graph(model, input_to_model, verbose=False, use_strict_trace=True)

    def finish(self):
        if self.closed:
            return
        self.closed = True
        self.writer.flush()
        self.writer.close()


class StdOutBackend(Backend):

    def __init__(self, filepath: Path, print_to_file=True) -> None:
        log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
        self.rootLogger = logging.getLogger()
        self.rootLogger.setLevel(DEFAULT_LOGGER_LEVEL)

        self.consoleHandler = logging.StreamHandler()
        self.consoleHandler.setFormatter(log_formatter)
        self.print_to_file = print_to_file
        if self.print_to_file:
            self.file_handler = logging.FileHandler(filepath)
            self.file_handler.setFormatter(log_formatter)
            self.rootLogger.addHandler(self.file_handler)
        self.rootLogger.addHandler(self.consoleHandler)
        self.closed = False
    
    def add_scalar(self, tag, value, level):
        msg = f"{tag}: {value}"
        self.rootLogger.log(level, msg)
    
    def add_dict(self, values, level):
        msg = ""
        for tag, value in values.items():
            msg += f"{tag}: {value:.3f}, "
        self.rootLogger.log(level, msg)

    def add_graph(self, model, input_to_model, level):
        msg = f"Added graph of model."
        self.rootLogger.log(level, msg)
    
    def log(self, msg, level):
        self.rootLogger.log(level, msg)
    
    def finish(self):
        if self.closed:
            return
        self.closed = True
        if self.print_to_file:
            self.file_handler.flush()
            self.file_handler.close()
            self.rootLogger.removeHandler(self.file_handler)
        self.rootLogger.removeHandler(self.consoleHandler)
        self.consoleHandler.close()

class JSONBackend(Backend):

    def __init__(self, filepath: Path) -> None:
        self.file = open(filepath, "a")
        self.closed = False
    
    def add_scalar(self, tag, value, **kwargs):
        self.add_dict({tag: value})
    
    def add_dict(self, values, **kwargs):
        values = {**values, "global_step":_global_step}
        values_str = json.dumps(values) + "\n"
        self.file.write(values_str)

    def add_graph(self, model, input_to_model, **kwargs):
        """JSON format logging does not support graphs."""
        pass
    
    def finish(self):
        if self.closed:
            return
        self.closed = True
        self.file.flush()
        self.file.close()


_backends: List[Backend] = [StdOutBackend(None, False)]

def init(output_dir, backends):
    """
    Initialize logging.
    Args:
        output_dir: directory to save logs
        backends: list of backends to use
    """
    global _backends, _output_dir
    for backend in _backends:
        backend.finish()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    _output_dir = output_dir
    _resume()
    _write_metadata()
    _backends = []
    for backend in backends:
        if backend not in supported_backends:
             raise ArgumentError(f"{backend} not in supported. Has to be one of: {', '.join(backends)}")
        if backend == "stdout":
            _backends.append(StdOutBackend(output_dir.joinpath("log.txt")))
        if backend == "tensorboard":
            _backends.append(TensorBoardBackend(output_dir.joinpath("tensorboard")))
        if backend == "json":
            _backends.append(JSONBackend(output_dir.joinpath("scalars.json")))
    atexit.register(finish)


def log(msg, level=DEFAULT_LOG_LEVEL):
    """
    Log a message.
    Args:
        msg: message to be logged
        level: logging level
    """
    for backend in _backends:
        backend.log(msg, level)

def add_scalar(tag, value, level=DEFAULT_SCALAR_LEVEL):
    """
    Add scalar to all backends.
    Args:
        tag: tag of the scalar
        value: value of the scalar
        level: logging level
    """
    for backend in _backends:
        backend.add_scalar(tag, value, level=level)

def add_dict(values: dict, level=DEFAULT_SCALAR_LEVEL):
    """
    Add dictionary of scalars to all backends.
    Args:
        values: dictionary of scalars
        level: logging level
    """
    for backend in _backends:
        backend.add_dict(values, level=level)


def add_graph(model, input_example, level=DEFAULT_GRAPH_LEVEL):
    """
    Add graph of model to all backends.
    Args:
        model: model to be plotted
        input_example: example input to model
    """
    for backend in _backends:
        backend.add_graph(model, input_example, level=level)


def finish():
    """
    Finish all backends.
    """
    _write_metadata()
    for backend in _backends:
        backend.finish()


def step():
    global _global_step
    _global_step += 1

def step_epoch():
    global _epoch
    _epoch += 1

def _write_metadata():
    with open(_output_dir.joinpath("metadata.json"), "w") as fp:
        json.dump(dict(global_step=_global_step, epoch=_epoch), fp)

def _resume():
    global _epoch, _global_step
    metadata_path = _output_dir.joinpath("metadata.json")
    if not metadata_path.is_file():
        return
    with open(metadata_path, "r") as fp:
        data = json.load(fp)
    _epoch = data["epoch"]
    _global_step = data["global_step"]

def epoch():
    return _epoch

def global_step():
    return _global_step


def read_logs(output_dir: pathlib.Path):
    log_path = output_dir.joinpath("logs","scalars.json")
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing log file: {log_path}")
    with open(log_path, "r") as fp:
        log_entries = fp.readlines()
    return [json.loads(s) for s in log_entries]
