import torch
from typing import Tuple, List


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(image_channels, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2), # 150x150 out

            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2), # 75 x 75 out

            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            # Use ceil mode to get 75/2 to ouput 38
            torch.nn.Conv2d(64, output_channels[0], kernel_size=3, padding=1, stride=2),
            torch.nn.ReLU(),
        )

        self.additional_layers = torch.nn.ModuleList([
            torch.nn.Sequential( # 19 x 19 out
                torch.nn.Conv2d(output_channels[0], 128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[1], kernel_size=3, padding=1, stride=2),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential( # 10x10 out
                torch.nn.Conv2d(output_channels[1], 256, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, output_channels[2], kernel_size=3, padding=1, stride=2),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential( # 5 x 5 out
                torch.nn.Conv2d(output_channels[2], 128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[3], kernel_size=3, padding=1, stride=2),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential( # 3 x 3 out
                torch.nn.Conv2d(output_channels[3], 128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[4], kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential( # 1 x 1 out
                torch.nn.Conv2d(output_channels[4], 128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[5], kernel_size=3),
                torch.nn.ReLU(),
            ),
        ])

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        x = self.feature_extractor(x)
        out_features.append(x)
        for additional_layer in self.additional_layers.children():
            x = additional_layer(x)
            out_features.append(x)
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        #print(f'Type in basic: {type(out_features)}')
        return tuple(out_features)


class BasicModelExtended(BasicModel): 

    def __init__(self, 
            output_channels: List[int], 
            image_channels: int, 
            output_feature_sizes: List[Tuple[int]]):
        super().__init__(output_channels, image_channels, output_feature_sizes)
        self.output_channels = output_channels

        # Leaky ReLU slope
        self.slope = 0.2

        # First layer
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=image_channels,
                            out_channels=32,
                            kernel_size=3,
                            padding=1
            ),  # 32 x 128 x 1024
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.LeakyReLU(negative_slope=self.slope),
            torch.nn.Conv2d(in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            padding=1
            ),  # 64 x 128 x 1024

            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.LeakyReLU(negative_slope=self.slope),     
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 64 x 64 x 512
            torch.nn.Conv2d(in_channels=64,
                            out_channels=128,
                            kernel_size=3,
                            padding=1
            ),  # 128 x 64 x 512

            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=self.slope),
            torch.nn.Conv2d(in_channels=128,
                            out_channels=128,
                            kernel_size=3,
                            padding=1
            ),  # 128 x 64 x 512

            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.LeakyReLU(negative_slope=self.slope),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # 128 x 32 x 256
            torch.nn.Conv2d(in_channels=128,
                            out_channels=256,
                            kernel_size=3,
                            padding=1
            ),  # 256 x 32 x 256

            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.LeakyReLU(negative_slope=self.slope),
            torch.nn.Conv2d(in_channels=256,
                            out_channels=512,
                            kernel_size=3,
                            padding=1
            ),  # 512 x 32 x 256

            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.LeakyReLU(negative_slope=self.slope),
            torch.nn.Conv2d(in_channels=512,
                            out_channels=output_channels[0],
                            kernel_size=3,
                            padding=1,
            ),  # 128 x 32 x 256
        )

        self.additional_layers = torch.nn.ModuleList([
            # Second layer
            torch.nn.Sequential( # 16 x 128 out
                torch.nn.BatchNorm2d(num_features=output_channels[0]),
                torch.nn.LeakyReLU(negative_slope=self.slope),
                torch.nn.Conv2d(in_channels=output_channels[0],
                                out_channels=512,
                                kernel_size=3,
                                padding=1
                ),  # 512 x 32 x 256

                torch.nn.BatchNorm2d(num_features=512),
                torch.nn.LeakyReLU(negative_slope=self.slope),
                torch.nn.Conv2d(in_channels=512,
                                out_channels=output_channels[1],
                                kernel_size=3,
                                padding=1,
                                stride=2
                ),  # 256 x 16 x 128
            ),

            # Third layer
            torch.nn.Sequential( # 8 x 64 out
                torch.nn.BatchNorm2d(num_features=output_channels[1]),
                torch.nn.LeakyReLU(negative_slope=self.slope),
                torch.nn.Conv2d(in_channels=output_channels[1], 
                                out_channels=256,
                                kernel_size=3,
                                padding=1
                ),  # 256 x 16 x 128

                torch.nn.BatchNorm2d(num_features=256),
                torch.nn.LeakyReLU(negative_slope=self.slope),
                torch.nn.Conv2d(in_channels=256,
                                out_channels=output_channels[2],
                                kernel_size=3,
                                padding=1,
                                stride=2
                ), # 128 x 8 x 64
            ),

            # Fourth layer
            torch.nn.Sequential( # 4 x 32 out
                torch.nn.BatchNorm2d(num_features=output_channels[2]),
                torch.nn.LeakyReLU(negative_slope=self.slope),
                torch.nn.Conv2d(in_channels=output_channels[2],
                                out_channels=128,
                                kernel_size=3,
                                padding=1
                ),  # 128 x 8 x 64

                torch.nn.BatchNorm2d(num_features=128),
                torch.nn.LeakyReLU(negative_slope=self.slope),
                torch.nn.Conv2d(in_channels=128,
                                out_channels=output_channels[3],
                                kernel_size=3,
                                padding=1,
                                stride=2
                ),  # 128 x 4 x 32
            ),

            # Fifth layer
            torch.nn.Sequential( # 2 x 16 out
                torch.nn.BatchNorm2d(num_features=output_channels[3]),
                torch.nn.LeakyReLU(negative_slope=self.slope),
                torch.nn.Conv2d(in_channels=output_channels[3],
                                out_channels=128,
                                kernel_size=3,
                                padding=1
                ),  # 128 x 4 x 32

                torch.nn.BatchNorm2d(num_features=128),
                torch.nn.LeakyReLU(negative_slope=self.slope),
                torch.nn.Conv2d(in_channels=128,
                                out_channels=output_channels[4],
                                kernel_size=3,
                                padding=1,
                                stride=2
                ),  # 64 x 2 x 16
            ),

            # Sixth layer
            torch.nn.Sequential( # 1 x 8 out
                torch.nn.BatchNorm2d(num_features=output_channels[4]),
                torch.nn.LeakyReLU(negative_slope=self.slope),
                torch.nn.Conv2d(in_channels=output_channels[4],
                                out_channels=128,
                                kernel_size=3,
                                padding=1
                ),  # 128 x 2 x 16

                torch.nn.BatchNorm2d(num_features=128),
                torch.nn.LeakyReLU(negative_slope=self.slope),
                torch.nn.Conv2d(in_channels=128,
                                out_channels=output_channels[5],
                                kernel_size=2,
                                stride=2
                ),  # 64 x 1 x 8
            ),
        ])

