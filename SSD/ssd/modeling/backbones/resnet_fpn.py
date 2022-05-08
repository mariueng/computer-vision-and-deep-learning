import torch.nn as nn
import math, pdb
import torch.nn.functional as F
import torch
import torchvision.models as models
from torch.nn.parameter import Parameter
from typing import List, Tuple

"""

Implementation based on original code from: https://github.com/gurkirt/RetinaNet.pytorch.1.x/blob/master/models/resnetFPN.py

"""


# Helper methods to construct FeaturePyramidNetwork backbone structure.

def conv3x3(in_planes, out_planes, stride=1, padding=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=bias)

def conv1x1(in_channel, out_channel, **kwargs):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, **kwargs)


class BasicBlock(nn.Module):
    """
    BasicBlock is a standard ResNet block.
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetFPNFeatureExtractor(nn.Module):
    """
    Feature Pyramid Network (FPN) using ResNet.

    Constructs the FPN manually using the pretrained parameters from the ResNet.
    """
    def __init__(self, block, layers, use_bias, seq_len):
        self.inplanes = 64
        super().__init__()

        # Input layer
        self.conv1 = nn.Conv2d(3*seq_len, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # c3
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # c4
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # c5

        # FPN layers
        self.conv6 = conv3x3(512 * block.expansion, 256, stride=2, padding=1, bias=use_bias)  # P6
        self.conv7 = conv3x3(256, 256, stride=2, padding=1, bias=use_bias)  # P7

        self.lateral_layer1 = conv1x1(512 * block.expansion, 256, bias=use_bias)
        self.lateral_layer2 = conv1x1(256 * block.expansion, 256, bias=use_bias)
        self.lateral_layer3 = conv1x1(128 * block.expansion, 256, bias=use_bias)
        self.lateral_layer4 = conv1x1(64 * block.expansion, 256, bias=use_bias)

        self.corr_layer1 = conv3x3(256, 256, stride=1, padding=1, bias=use_bias)  # P4
        self.corr_layer2 = conv3x3(256, 256, stride=1, padding=1, bias=use_bias)  # P4
        self.corr_layer3 = conv3x3(256, 256, stride=1, padding=1, bias=use_bias)  # P3
        self.corr_layer4 = conv3x3(256, 256, stride=1, padding=1, bias=use_bias)  # P2

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, a=1)
                if hasattr(m.bias, 'data'):
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample(self, x, y):
        """
        Upsample `x` to the size of `y` using nearest neighbor interpolation.
        """
        _, _, h, w = y.size()
        x_upsampled = F.interpolate(x, [h, w], mode='nearest')

        return x_upsampled

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Forward input layer
        x = self._forward_input_layer(x)

        # Forward ResNet layers
        c2 = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down FPN lateral layers
        p5 = self.lateral_layer1(c5)
        p5_upsampled = self._upsample(p5, c4)
        p5 = self.corr_layer1(p5)

        p4 = self.lateral_layer2(c4)
        p4 = p5_upsampled + p4
        p4_upsampled = self._upsample(p4, c3)
        p4 = self.corr_layer2(p4)

        p3 = self.lateral_layer3(c3)
        p3 = p4_upsampled + p3
        p3_upsampled = self._upsample(p3, c2)
        p3 = self.corr_layer3(p3)

         # Custom FPN layer
        p2 = self.lateral_layer4(c2)
        p2 = p3_upsampled + p2
        p2 = self.corr_layer4(p2)

        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))

        out_features = [p2, p3, p4, p5, p6, p7]

        return tuple(out_features)

    def _forward_input_layer(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        return x

    def load_state_from_dict(self, state_dict, seq_len=1):
        """
        Loads the state_dict into the model, i.e. pretrained parameters.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state.keys():
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                if name == 'conv1.weight':
                    print(name, 'is being filled with {:d} seq_len\n'.format(seq_len))
                    param = param.repeat(1, seq_len, 1, 1)
                    param = param / float(seq_len)
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            else:
                #print('NAME IS NOT IN OWN STATE::>' + name)
                pass


class ResNetFPN(nn.Module):
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]],
            model: str,
            pretrained: bool,
        ):
        super().__init__()

        self.out_channels = output_channels
        self.image_channels = image_channels
        self.output_feature_shape = output_feature_sizes
        self.fpn_output_channels = 256

        self.feature_extractor = load_feature_extractor(model, pretrained)

    def forward(self, x):
        return self.feature_extractor(x)


def load_feature_extractor(model_name: str, pretrained: bool, use_bias=True, seq_len=1):
    """
    Returns a FPN feature extractor on top of a pretrained ResNet model
    """
    f = list(filter(lambda x: x.startswith('resnet'), dir(models)))

    if model_name[:6] not in f or int(model_name[-2:]) > 50:
        raise ValueError(f'Model {model_name} is not supported.')

    resnet_models = {'resnet18': [2, 2, 2, 2], 'resnet34': [3, 4, 6, 3]}

    feature_extractor = ResNetFPNFeatureExtractor(BasicBlock, resnet_models[model_name], use_bias, seq_len)

    if pretrained:
        if model_name == 'resnet18':
            pretrained_model = models.resnet18(pretrained=True)
        elif model_name == 'resnet34':
            pretrained_model = models.resnet34(pretrained=True)
        else:
            raise ValueError(f'Model {model_name} is not supported.')

    feature_extractor.load_state_from_dict(pretrained_model.state_dict())

    return feature_extractor
