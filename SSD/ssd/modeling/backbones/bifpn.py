import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from typing import List, Tuple

from .resnet_fpn import load_feature_extractor

"""

Original code base: https://github.com/tristandb/EfficientDet-PyTorch/blob/master/bifpn.py

Adapted to match feature_sizes

"""


class DepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution. 
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                               padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)
    
class ConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.act(x)

class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network backbone.
    """
    def __init__(self, feature_size=64, epsilon=0.0001):
        super().__init__()
        self.epsilon = epsilon

        self.p3_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p7_td = DepthwiseConvBlock(feature_size, feature_size)

        self.p4_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p7_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p8_out = DepthwiseConvBlock(feature_size, feature_size)

        # TODO: Init weights
        self.w1 = nn.Parameter(torch.Tensor(2, 4))
        self.w1.requires_grad = True
        self.w1_relu = nn.ReLU(inplace=True)
        self.w2 = nn.Parameter(torch.Tensor(3, 4))
        self.w2.requires_grad = True
        self.w2_relu = nn.ReLU(inplace=True)
    
    def forward(self, inputs):
        # Inputs here have been passed through the feature extractor.
        p3_x, p4_x, p5_x, p6_x, p7_x, p8_x = inputs

        # Calculate Top-Down Pathway
        with torch.no_grad():
            w1 = self.w1_relu(self.w1)
            w1 /= torch.sum(w1, dim=0) + self.epsilon
            w2 = self.w2_relu(self.w2)
            w2 /= torch.sum(w2, dim=0) + self.epsilon

        p8_td = p8_x
        p7_td = self.p7_td(w1[0, 0] * p7_x + w1[1, 0] * nn.Upsample(scale_factor=2)(p8_td))
        p6_td = self.p6_td(w1[0, 1] * p6_x + w1[1, 1] * nn.Upsample(scale_factor=2)(p7_td))
        p5_td = self.p5_td(w1[0, 1] * p5_x + w1[1, 1] * nn.Upsample(scale_factor=2)(p6_td))
        p4_td = self.p4_td(w1[0, 2] * p4_x + w1[1, 2] * nn.Upsample(scale_factor=2)(p5_td))
        p3_td = self.p3_td(w1[0, 3] * p3_x + w1[1, 3] * nn.Upsample(scale_factor=2)(p4_td))

        # Calculate Bottom-Up Pathway
        p3_out = p3_td
        p4_out = self.p4_out(w2[0, 0] * p4_x + w2[1, 0] * p4_td + w2[2, 0] * nn.MaxPool2d(kernel_size=2)(p3_out))
        p5_out = self.p5_out(w2[0, 1] * p5_x + w2[1, 1] * p5_td + w2[2, 1] * nn.MaxPool2d(kernel_size=2)(p4_out))
        p6_out = self.p6_out(w2[0, 2] * p6_x + w2[1, 2] * p6_td + w2[2, 2] * nn.MaxPool2d(kernel_size=2)(p5_out))
        p7_out = self.p7_out(w2[0, 3] * p7_x + w2[1, 3] * p7_td + w2[2, 3] * nn.MaxPool2d(kernel_size=2)(p6_out))
        p8_out = self.p8_out(w2[0, 3] * p8_x + w2[1, 3] * p8_td + w2[2, 3] * nn.MaxPool2d(kernel_size=2)(p7_out))

        out = [p3_out, p4_out, p5_out, p6_out, p7_out, p8_out]

        # print("Final out:")
        # for feat in out:
        #      print(feat.shape)
        return tuple(out)
    
class BiFPN(nn.Module):
    def __init__(self,
            input_channels: List[int],
            feature_size: int,
            output_feature_sizes: List[Tuple[int]],
        ):
        super().__init__()
        # print("Input channels:")
        # print(input_channels)
        self.input_channels = input_channels
        self.feature_size = feature_size
        self.output_feature_shape = output_feature_sizes
        # print("Output feature shapes:")
        # print(output_feature_sizes)

        # Feature extractor contains layers [c0, bn0, rl0, mp0 c1, c2, c3, c4, c5, c6]
        self.feature_extractor = load_feature_extractor(self.input_channels)
        assert len(self.feature_extractor) == 10, f"Expected Feat_extr to have 10 layers, but got {len(self.feature_extractor)}"

        # Structure of BiFPN
        self.num_layers = 2
        self.epsilon = 0.0001

        self.p3 = nn.Conv2d(input_channels[0], self.feature_size, kernel_size=3, stride=1, padding=1)
        self.p4 = nn.Conv2d(input_channels[1], self.feature_size, kernel_size=3, stride=1, padding=1)
        self.p5 = nn.Conv2d(input_channels[2], self.feature_size, kernel_size=3, stride=1, padding=1)

        # p6 is obtained via a 3x3 stride-2 conv on C5
        self.p6 = nn.Conv2d(input_channels[3], self.feature_size, kernel_size=3, stride=1, padding=1)

        # p7 is computed by applying ReLU followed by a 3x3 stride-2 conv on p6
        self.p7 = ConvBlock(input_channels[4], self.feature_size, kernel_size=3, stride=1, padding=1)

        # p8 is a custom layer added to the end of the network
        self.p8 = ConvBlock(self.feature_size, self.feature_size, kernel_size=3, stride=1, padding=1)

        bifpns = []
        for _ in range(self.num_layers):
            bifpns.append(BiFPNBlock(self.feature_size))
        self.bifpn = nn.Sequential(*bifpns)

    def forward(self, x):
        # print("Shape after resnet")
        for feature in self.feature_extractor[:4]:
            x = feature(x)
            # print(x.shape)

        out_features = dict()

        for idx, feature in enumerate(self.feature_extractor[4:]):
            x = feature(x)
            out_features[f"c{idx + 1}"] = x

        # Calculate the input column of BiFPN
        p3 = self.p3(out_features["c1"])
        p4 = self.p4(out_features["c2"])
        p5 = self.p5(out_features["c3"])
        p6 = self.p6(out_features["c4"])
        p7 = self.p7(out_features["c5"])
        p8 = self.p8(out_features["c6"])

        out_features = [p3, p4, p5, p6, p7, p8]

        # print("Shape after Deptwise Conv Blocks: ")
        # for feature in out_features:
        #     print(feature.shape)

        out_features = self.bifpn(out_features)

        # Check that the output features are correct
        for idx, feature in enumerate(out_features):
            # print(feature.shape)
            out_channel = self.feature_size
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"

        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"

        return tuple(out_features) #self.bifpn(out_features)


def load_feature_extractor(input_channels):
    model = models.resnet50(pretrained=True)
    pretrained_layers = nn.Sequential(*list(model.children())[:-2])
    additional_layers = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(input_channels[3], input_channels[4], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels[4], input_channels[4], kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
        ),
        nn.Sequential(
            nn.Conv2d(input_channels[4], input_channels[5], kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels[5], 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
        ),
    ])
    return nn.Sequential(*pretrained_layers, *additional_layers)