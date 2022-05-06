import warnings
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.nn import SmoothL1Loss, Parameter
from torchvision.ops.focal_loss import sigmoid_focal_loss

"""
Implementation originally based on: https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py

Originally uses input and target, adapted to work with bbox_delta and confidences as in the starter code.

"""


def focal_loss(
    confs: torch.FloatTensor,
    gt_labels: torch.FloatTensor,
    alphas: torch.FloatTensor,
    gamma: float = 2.0,
    ) -> torch.Tensor:
    """
    Compute the focal loss between `input` and `target`.
    Args:
        confs: [batch_size, num_classes, num_anchors]
        gt_labels: [batch_size, num_anchors]
        alpha: focal loss's alpha
        gamma: focal loss's gamma
    Returns:
        loss: [batch_size, num_classes, num_anchors]
    """

    num_classes = confs.shape[1]

    # Create alpha that matches the shape of the input tensor
    alpha = alphas.repeat(confs.shape[2], 1).T  # [N, num_classes]
    alpha = alpha.repeat(confs.shape[0], 1, 1)  # [N, num_classes, num_anchors]

    # From here on, all shapes should be # [N, num_classes, num_anchors] to allow
    # for element wise matrix multiplication. 

    # Calculate softmax and log softmax of confidences
    p: torch.Tensor = F.softmax(confs, dim=1)
    log_p: torch.Tensor = F.log_softmax(confs, dim=1)

    # One-hot encode ground truth labels: 
    y: torch.Tensor = one_hot_encoder(
        gt_labels,
        num_classes=num_classes,
        device=confs.device,
        dtype=confs.dtype
    )

    # Calculate focal losses
    focal_losses = -alpha * torch.pow(-p + 1.0, gamma) * log_p * y

    return torch.sum(focal_losses)



class FocalLoss(torch.nn.Module):
    def __init__(self,
                anchors,
                alpha: torch.Tensor = torch.tensor([[0.01, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
                gamma: float = 0.25,
                reduction: str = 'none',
                eps: Optional[float] = None
        ):
        super().__init__()
        self.alpha: List[float] = alpha
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
        """
        Compute the regression loss between `bbox_delta` and `gt_bbox`.
        Args:
            bbox_delta: [batch_size, num_anchors, 4]
            gt_bbox: [batch_size, num_anchors, 4]
            gt_labels: [batch_size, num_anchors]
        Returns:
            loss: [batch_size, num_anchors]
            num_pos: Number of positive anchors
        """
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

        # Reshape to match bbox_delta: [batch_size, 4, num_ancors]
        gt_bbox = gt_bbox.transpose(1, 2).contiguous()

        # Compute same regression loss as before
        regr_loss, num_pos = self.regression_loss(bbox_delta, gt_bbox, gt_labels)

        # Compute focal loss (new classification loss)
        cls_loss = focal_loss(confs, gt_labels, self.alpha, self.gamma)

        # Compute total loss
        total_loss = regr_loss / num_pos + cls_loss / num_pos

        # Log to tensorboard
        to_log = dict(
            regression_loss=regr_loss / num_pos,
            classification_loss=cls_loss / num_pos,
            total_loss=total_loss
        )

        return total_loss, to_log

def one_hot_encoder(
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
    Returns:
        the labels in one hot tensor of shape :math:`(N, C, H, W)`
    """

    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
