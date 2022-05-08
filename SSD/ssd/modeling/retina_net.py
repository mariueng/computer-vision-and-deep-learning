import torch
import torch.nn as nn
from .anchor_encoder import AnchorEncoder
from .ssd import filter_predictions

"""
RetinaNet Head implemented into SSD model.

Contains code for Task 2.3.3 and 2.3.4. Only difference is running with or without
the improved weight initalization.

"""


class RetinaNet(nn.Module):
    def __init__(self, 
            feature_extractor: nn.Module,
            anchors,
            loss_objective,
            num_classes: int,
            improved_weight_init: bool = False):
        super().__init__()
        """
        Implements the SSD network with shared deep convolutional nets as 
        regression/classification heads.
        """

        self.feature_extractor = feature_extractor
        self.num_boxes_per_fmap = anchors.num_boxes_per_fmap
        self.loss_func = loss_objective
        self.num_classes = num_classes
        self.improved_weight_init = improved_weight_init
        self.regression_heads = self._make_head(self.num_boxes_per_fmap, 4)
        self.classification_heads = self._make_head(self.num_boxes_per_fmap, self.num_classes)

        self.anchor_encoder = AnchorEncoder(anchors)
        self._init_weights()

    def _init_weights(self):
        layers = [*self.regression_heads, *self.classification_heads]
        if not self.improved_weight_init:
            # Normal weight init for Task 2.3.3.
            for layer in layers:
                for param in layer.parameters():
                    if param.dim() > 1: nn.init.xavier_uniform_(param)
        # else:
        # Improved weight initialization for Task 2.3.4.
        π = torch.tensor(.99)
        # Initialize all new conv layers except for the final one with bias=0 and a Gaussian weight fill with sigma=0.01.
        # TODO: If results poor, try removing this one and else condition.
        # for layer in layers[:-1]:
        #     for param in layer.parameters():
        #         if param.dim() > 1: nn.init.normal_(param, 0, 0.01)

        # For the final conv layer of the regression_heads and classification_heads, we set the bias initialization to b = - log ((1 - π) / π).
        n = torch.tensor(self.num_boxes_per_fmap[-1])
        b = torch.ones(n, 1) * torch.log(π * (self.num_classes - 1)/(1 - π))  #TODO: test with 1 instead of self.num_classes
        self.classification_heads[-1][-1].bias.data[:n] = b.flatten().clone()
        self.regression_heads[-1][-1].bias.data[:n] = b.flatten().clone()


    def _make_head(self, num_boxes_per_fmap, k):
        """
        Creates convolutional heads for RetinaNet.
        Args:
            num_boxes_per_fmap: Number of boxes per feature map.
            k: Factor to mulitply linear output with.
        Returns:
            List of convolutional heads.
        """
        layers = []
        fpn_outputs = 256

        # Matching number of anchor boxes to the number of feature maps.
        for num_boxes in num_boxes_per_fmap:
            layer = []
            for i in range(4):
                layer.append(nn.Conv2d(fpn_outputs, fpn_outputs, kernel_size=3, stride=1, padding=1))
                layer.append(nn.ReLU())
            layer.append(nn.Conv2d(fpn_outputs, num_boxes * k, kernel_size=3, stride=1, padding=1))
            layers.append(nn.Sequential(*layer))
        return nn.ModuleList(layers)

    def regress_boxes(self, features):
        locations = []
        confidences = []
        for idx, x in enumerate(features):
            bbox_delta = self.regression_heads[idx](x).view(x.shape[0], 4, -1)
            bbox_conf = self.classification_heads[idx](x).view(x.shape[0], self.num_classes, -1)
            locations.append(bbox_delta)
            confidences.append(bbox_conf)
        bbox_delta = torch.cat(locations, 2).contiguous()
        confidences = torch.cat(confidences, 2).contiguous()
        return bbox_delta, confidences

    
    def forward(self, img: torch.Tensor, **kwargs):
        """
            img: shape: NCHW
        """
        if not self.training:
            # If not training, return predictions.
            return self.forward_test(img, **kwargs)

        # If training, return loss.
        features = self.feature_extractor(img)
        return self.regress_boxes(features)
    
    def forward_test(self,
            img: torch.Tensor,
            imshape=None,
            nms_iou_threshold=0.5, max_output=200, score_threshold=0.05):
        """
            img: shape: NCHW
            nms_iou_threshold, max_output is only used for inference/evaluation, not for training
        """
        features = self.feature_extractor(img)
        bbox_delta, confs = self.regress_boxes(features)
        boxes_ltrb, confs = self.anchor_encoder.decode_output(bbox_delta, confs)
        predictions = []
        for img_idx in range(boxes_ltrb.shape[0]):
            boxes, categories, scores = filter_predictions(
                boxes_ltrb[img_idx], confs[img_idx],
                nms_iou_threshold, max_output, score_threshold)
            if imshape is not None:
                H, W = imshape
                boxes[:, [0, 2]] *= H
                boxes[:, [1, 3]] *= W
            predictions.append((boxes, categories, scores))

        # Convert to tuple for traceability.
        return tuple(predictions)
