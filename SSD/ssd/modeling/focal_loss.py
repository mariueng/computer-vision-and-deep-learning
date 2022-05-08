import torch
import torch.nn.functional as F
from torch.nn import SmoothL1Loss, Parameter
from typing import List

"""
Implementation originally based on: https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py

Originally uses input and target, adapted to work with bbox_delta and confidences as in the starter code.

"""

class FocalLoss(torch.nn.Module):
    def __init__(self,
                anchors,
                alphas: List[float] = [0.01, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                gamma: float = 2.0,
        ):
        super().__init__()
        self.scale_xy = 1.0 / anchors.scale_xy
        self.scale_wh = 1.0 / anchors.scale_wh
        self.alphas = torch.tensor(alphas).cuda() if torch.cuda.is_available() else torch.tensor(alphas)
        self.gamma = gamma
        self.anchors = Parameter(anchors(order="xywh").transpose(0, 1).unsqueeze(dim = 0), requires_grad=False)

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

    def focal_loss(self, confs, gt_labels):
        """
        Compute the focal loss between `confs` and `gt_labels`.
        Args:
            confs: [batch_size, num_classes, num_anchors]
            gt_labels: [batch_size, num_anchors]
            alpha: focal loss's alpha
            gamma: focal loss's gamma
        Returns:
            loss: focal loss
        """
        num_classes = confs.shape[1]

        # Repeat and reshape alpha values to match confs shape
        alphas: torch.Tensor = self.alphas.repeat(confs.shape[2], 1).T  # [batch_size, num_classes]
        alphas: torch.Tensor = self.repeat(confs.shape[0], 1, 1)        # [batch_size, num_classes, num_anchors]

        # Calculate softmax and log softmax of confidences
        p: torch.Tensor     = F.softmax(confs, dim=1)
        log_p: torch.Tensor = F.log_softmax(confs, dim=1)

        # One-hot encode ground truth labels: 
        y: torch.Tensor = F.one_hot(
            gt_labels,
            num_classes=num_classes,
            device=confs.device,
            dtype=confs.dtype
        )
        y = torch.transpose(y, -1, -2)

        # Calculate focal losses
        losses: torch.Tensor = -alphas * torch.pow(-p + 1.0, self.gamma) * y * log_p

        return torch.sum(losses)
        

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
        pos_mask = (gt_labels > 0).unsqueeze(1).repeat(1, 4, 1)  # Get positive gt mask
        bbox_delta = bbox_delta[pos_mask]                        # Filter positive bbox_delta
        gt_locations = self._loc_vec(gt_bbox)                    # Find all ground truth locations
        gt_locations = gt_locations[pos_mask]                    # Get only positive ground truth locations
        num_pos = gt_locations.shape[0] / 4                      # Number of positive ground truth anchors
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
            total_loss: [batch_size, num_classes, num_anchors]
            to_log: tensorboard logs
        """

        # Reshape to match bbox_delta: [batch_size, 4, num_ancors]
        gt_bbox = gt_bbox.transpose(1, 2).contiguous()
    
        # Compute focal loss
        cls_loss = self.focal_loss(confs, gt_labels)

        # Compute same regression loss (same as before)
        regr_loss, num_pos = self.regression_loss(bbox_delta, gt_bbox, gt_labels)

        # Compute total loss
        total_loss = regr_loss / num_pos + cls_loss / num_pos

        # Log to tensorboard
        to_log = dict(
            regression_loss=regr_loss / num_pos,
            classification_loss=cls_loss / num_pos,
            total_loss=total_loss
        )
        return total_loss, to_log
