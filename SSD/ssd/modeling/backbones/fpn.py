import torch
import torchvision.models as models

from typing import Tuple, List, OrderedDict
from torchvision.ops import FeaturePyramidNetwork


def load_resnet_model(model_name: str, pretrained: bool, **kwargs) -> torch.nn.Module:
    """
    Loads a resnet model from the list of available models.
    :param model_name: Name of the model to load.
    :return: The loaded model.
    """

    f = list(filter(lambda x: x.startswith('resnet'), dir(models)))

    if model_name not in f:
        raise ValueError(f'Model {model_name} is not supported.')

    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
    else:
        print("Come on, give me a break")
        raise ValueError(f'Model {model_name} is overkill and you know it.')

    return model

class FPN(torch.nn.Module):
    """
    Feature Pyramid Network backbone for SSD network.
    """
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
        self.fpn_resnet_channels = [64, 128, 256, 512, 256, 64]
        self.output_feature_shape = output_feature_sizes
        self.fpn_output_channels = 256

        # Implement FPN on top of a pre-trained backbone, e.g. ResNet-34
        self.feature_extractor = load_resnet_model(model_name=model, pretrained=pretrained)

        # Additional layers
        self.additional_layers = torch.nn.ModuleList([
            torch.nn.Sequential(  # Custom input layer to match ResNet input
                torch.nn.Conv2d(in_channels=self.image_channels,
                                out_channels=self.fpn_resnet_channels[0],
                                kernel_size=3,
                                stride=4,
                                padding=1
                ),  # 64 x 32 x 256
            ),

            torch.nn.Sequential(  # p6
                torch.nn.Conv2d(in_channels=self.fpn_resnet_channels[3],
                                out_channels=self.fpn_resnet_channels[4],
                                kernel_size=3,
                                stride=2,
                                padding=1
                ),
            ),

            torch.nn.Sequential(  # p7
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=self.fpn_resnet_channels[4],
                                out_channels=self.fpn_resnet_channels[5],
                                kernel_size=3,
                                stride=2,
                                padding=1
                ),
            ),
        ])

        # Use torchvision's FPN implementation
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.fpn_resnet_channels,  # Must match ResNet output features
            out_channels=self.fpn_output_channels  # Outputs same number of channels for all feature maps
        )

    def forward(self, x):

        out_features = []

        # NOTE! The input in_channels_list in FPN has to match the number of 
        # channels in each of the feature maps that ResNet outputs.

        # Custom input layer
        input_feat = self.additional_layers[0](x)  # 64 x 32 x 256
        out_features.append(input_feat)

        # ResNet layers
        out_feature = self.feature_extractor.layer2(input_feat)  # 128 x 16 x 128, p3 in RetinaNet
        out_features.append(out_feature)

        out_feature = self.feature_extractor.layer3(out_feature)  # 256 x 8 x 64, p4 in RetinaNet
        out_features.append(out_feature)

        out_feature = self.feature_extractor.layer4(out_feature)  # 512 x 4 x 32, p5 in RetinaNet
        out_features.append(out_feature)

        # Additional layers
        out_feature = self.additional_layers[1](out_feature)  # 256 x 2 x 16, p6 in RetinaNet
        out_features.append(out_feature)

        out_feature = self.additional_layers[2](out_feature)  # 64 x 1 x 8, p7 in RetinaNet
        out_features.append(out_feature)

        # Check that the output features are correct
        for idx, feature in enumerate(out_features):
            out_channel = self.fpn_resnet_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"

        # Assert that the output features shape is correct
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"

    
        # FPN expects OrderedDict of Tensors
        output = OrderedDict()

        for idx, feature in enumerate(out_features):
            output[f"p{idx}"] = feature

        output = self.fpn(output)

        # Convert to tuple of tensors before returning for traceability
        return tuple(output.values())