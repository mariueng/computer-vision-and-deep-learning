import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import SmoothL1Loss, Parameter

from typing import List


"""
Implementation originally based on: https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py

Originally uses input and target, adapted to work with bbox_delta and confidences.

"""


def focal_loss(
    confs: torch.FloatTensor,
    gt_labels: torch.FloatTensor,
    alpha: float,
    gamma: float = 2.0,
    ) -> torch.Tensor:
    """
    Compute the focal loss between `input` and `target`.
    Args:
        bbox_delta: [batch_size, 4, num_anchors]
        confs: [batch_size, num_classes, num_anchors]
        gt_bbox: [batch_size, num_gt, 4]
        gt_labels: [batch_size, num_gt]
        alpha: focal loss's alpha
        gamma: focal loss's gamma
        reduction: 'none' | 'mean' | 'sum'
        eps: epsilon
    Returns:
        loss: [batch_size, num_classes, num_anchors]
    """

    print(f'Number of classes: {confs.shape[1]}')

    # Calculate softmax and log softmax of confidences
    input_soft: torch.Tensor = F.softmax(confs, dim=1)
    print(f'Input softmax shape: {input_soft.shape}')
    log_input_soft: torch.Tensor = F.log_softmax(confs, dim=1)
    print(f'Log input softmax shape: {log_input_soft.shape}')

    # One-hot encode ground truth labels
    target_one_hot: torch.Tensor = one_hot(gt_labels, num_classes=confs.shape[1], device=confs.device, dtype=confs.dtype)
    print(f'Target one-hot shape: {target_one_hot.shape}')
    target_one_hot = torch.transpose(target_one_hot, -1, -2)
    print(f'Target one-hot shape: {target_one_hot.shape}')

    # Calculate weights for each anchor
    weight = torch.pow(-input_soft + 1.0, gamma)

    # Calculate focal losses
    focal = -alpha * weight * log_input_soft
    loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

    loss = loss_tmp.mean()

    # Check that output is correct shape: [batch_size, num_classes, num_anchors]
    print(f'Loss shape: {loss.shape}')
    assert loss.shape == (confs.shape[0], confs.shape[1], confs.shape[2])

    return loss


class FocalLoss(torch.nn.Module):
    def __init__(self,
                anchors,
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

        # Set scales for later calculations
        self.scale_xy = 1.0 / anchors.scale_xy
        self.scale_wh = 1.0 / anchors.scale_wh

        # Get Smooth L1 Loss function
        self.sl1_loss = SmoothL1Loss(reduction=self.reduction)
        self.anchors = Parameter(anchors(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
                                 requires_grad=False)

    def _loc_vec(self, loc):
        """
        Generate Location Vectors
        Args:
            loc: [batch_size, num_anchors, 4]
        Returns:
            loc_vec: [batch_size, num_anchors, 4]
        """
        gxy = self.scale_xy * (loc[:, :2, :] - self.anchors[:, :2, :]) / self.anchors[:, 2:, ]
        gwh = self.scale_wh * (loc[:, 2:, :] / self.anchors[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def regression_loss(self, bbox_delta, gt_bbox, gt_labels):

        pos_mask = (gt_labels > 0).unsqueeze(1).repeat(1, 4, 1)
        bbox_delta = bbox_delta[pos_mask]
        gt_locations = self._loc_vec(gt_bbox)
        gt_locations = gt_locations[pos_mask]
        num_pos = gt_locations.shape[0]/4
        return F.smooth_l1_loss(bbox_delta, gt_locations, reduction="sum"), num_pos


    def forward(self,
                bbox_delta: torch.Tensor,
                confs: torch.Tensor,
                gt_bbox: torch.FloatTensor,
                gt_labels: torch.LongTensor
            ) -> torch.Tensor:
        """
        Perform loss calculations on the input and target tensors.
        Args:
            bbox_delta: [batch_size, 4, num_anchors]
            confs: [batch_size, num_classes, num_anchors]
            gt_bbox: [batch_size, num_anchors, 4]
            gt_label = [batch_size, num_anchors]
        Returns:
            loss: [batch_size, num_classes, num_anchors]
        """

        # Reshape to match bbox_delta
        gt_bbox = gt_bbox.transpose(1, 2).contiguous()

        # Compute same regression loss as before
        regression_loss, num_pos = self.regression_loss(bbox_delta, gt_bbox, gt_labels)

        # Compute focal loss (new classification loss)
        classification_loss = focal_loss(confs, gt_labels, self.alpha, self.gamma)

        # Compute total loss
        total_loss = regression_loss / num_pos + classification_loss / num_pos

        # Log to tensorboard
        to_log = dict(
            regression_loss=regression_loss / num_pos,
            classification_loss=classification_loss,
            total_loss=total_loss
        )

        return total_loss, to_log


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
        
