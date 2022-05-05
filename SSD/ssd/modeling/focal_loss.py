import warnings
from typing import Optional

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import SmoothL1Loss

from typing import List


"""
Implementation based on: https://github.com/clcarwin/focal_loss_pytorch/blob/e11e75bad957aecf641db6998a1016204722c1bb/focalloss.py#L6
"""

# Code based on: https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py

def validate_input(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: Optional[float] = None,
    ) -> torch.Tensor:
    if eps is not None and not torch.jit.is_scripting():
        warnings.warn(
            "`focal_loss` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if len(input.shape) < 2:
        raise ValueError(f"Invalid input shape, we expect B x C x H x W. Got {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f"Input and target must have the same batch size. Got {input.size(0)} and {target.size(0)}")

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f"Expected target size {out_size}. Got {target.size()}")

    if input.device != target.device:
        raise ValueError(f"Input and target must be on the same device. Got {input.device} and {target.device}")


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: Optional[float] = None,
    ) -> torch.Tensor:
    """
    Compute the focal loss between `input` and `target`.
    Args:
        input: A tensor of shape (N, C, H, W) representing the predictions
            of the network.
        target: A tensor of shape (N, H, W) representing the ground truth
            labels.
        alpha: A scalar multiplying alpha to the loss from positive examples.
        gamma: A scalar modulating the loss from hard and easy examples.
        reduction: The reduction to apply to the output.
        eps: A scalar added to the denominator for numerical stability.
    Returns:
        A tensor containing the loss.
    """
    input_soft: torch.Tensor = F.softmax(input, dim=1)
    log_input_soft: torch.Tensor = F.log_softmax(input, dim=1)

    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * log_input_soft
    loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = loss_tmp.mean()
    elif reduction == 'sum':
        loss = loss_tmp.sum()
    else:
        raise NotImplementedError(f"Reduction {reduction} is not implemented")

    return loss


class FocalLoss(nn.Module):
    def __init__(self,
            alpha: float = 2,
            gamma: float = 0.25,
            reduction: str = 'none',
            eps: Optional[float] = None
        ):
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: Optional[float] = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Perform focal loss on the input and target tensors.
        Args:
            input: A tensor of shape (N, C, H, W) representing the predictions
                of the network.
            target: A tensor of shape (N, H, W) representing the ground truth
                labels.
        Returns:
            A tensor of shape (N, C, H, W) representing the loss.
        """

        validate_input(input, target, self.alpha, self.gamma, self.reduction, self.eps)

        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)


def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels: tensor with labels of shape :math:`(N, H, W)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.
    Returns:
        the labels in one hot tensor of shape :math:`(N, C, H, W)`
    """
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"Input labels type is not a torch.Tensor. Got {type(labels)}")

    if labels.dtype != torch.int64:
        raise ValueError(f"labels must be of the same dtype torch.int64. Got: {labels.dtype}")

    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one." " Got: {}".format(num_classes))

    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
        
