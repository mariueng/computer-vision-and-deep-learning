import random
import numpy as np
import torch

AMP_enabled = False

def set_amp(value: bool):
    """
    Set whether to use AMP or not.
    """
    global AMP_enabled
    AMP_enabled = value


def amp():
    """
    Return whether to use AMP or not.
    """
    return AMP_enabled


def _to_cuda(element):
    """
    Convert element to CUDA tensor.
    Args:
        element: element to convert
    Returns:
        element converted to CUDA tensor in a background thread."""
    return element.to(get_device(), non_blocking=True)


def to_cuda(elements):
    """
    Convert elements to CUDA tensors.
    Args:
        elements: elements to convert
    Returns:
        elements converted to CUDA tensors in a background thread.
    """
    if isinstance(elements, tuple) or isinstance(elements, list):
        return [_to_cuda(x) for x in elements]
    if isinstance(elements, dict):
        return {k: _to_cuda(v) for k,v in elements.items()}
    return _to_cuda(elements)


def get_device() -> torch.device:
    """
    Return dec
    """
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
