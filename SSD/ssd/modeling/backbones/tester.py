import torch
from typing import Tuple, List


class TestModel(torch.nn.Module):
    """
    This is a tester backbone for SSD.

    ONLY to be used to test other functionality of the program.
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()

        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        self.feature_extractor = torch.nn.Sequential( # 128 x 32 x 256
            torch.nn.Conv2d(image_channels, output_channels[0], kernel_size=3, padding=1, stride=4),
            # torch.nn.Linear(image_channels, output_channels[0]),
        )

        self.additional_layers = torch.nn.ModuleList([
            torch.nn.Sequential( # 256 x 16 x 128
                torch.nn.Conv2d(output_channels[0], output_channels[1], kernel_size=3, padding=1, stride=2),
                # torch.nn.Linear(output_channels[0], output_channels[1]),
            ),
            torch.nn.Sequential( # 128 x 8 x 64
                torch.nn.Conv2d(output_channels[1], output_channels[2], kernel_size=3, padding=1, stride=2),
                # torch.nn.Linear(output_channels[1], output_channels[2]),
            ),
            torch.nn.Sequential( # 128 x 4 x 32
                torch.nn.Conv2d(output_channels[2], output_channels[3], kernel_size=3, padding=1, stride=2),
                # torch.nn.Linear(output_channels[2], output_channels[3]),
            ),
            torch.nn.Sequential( # 64 x 2 x 16
                torch.nn.Conv2d(output_channels[3], output_channels[4], kernel_size=3, padding=1, stride=2),
                # torch.nn.Linear(output_channels[3], output_channels[4]),
            ),
            torch.nn.Sequential( # 64 x 1 x 8
                torch.nn.Conv2d(output_channels[4], output_channels[5], kernel_size=3, padding=1, stride=2),
                # torch.nn.Linear(output_channels[0], output_channels[1]),
            ),
        ])

    def forward(self, x):
        # Stores output features
        out_features = []

        # Pass through the first layer
        x = self.feature_extractor(x)
        out_features.append(x)

        # Pass through the additional layers
        for additional_layer in self.additional_layers.children():
            x = additional_layer(x)
            out_features.append(x)

        # Check that the output features are correct
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"

        # Assert that the output features shape is correct
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"

        # Convert to tuple of tensors before returning for traceability
        return tuple(out_features)
