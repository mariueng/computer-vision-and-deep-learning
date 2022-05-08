import torch
from torch import nn
from typing import Tuple, List
import torchvision.models as tvm


class BiFPN(torch.nn.Module):
    """
    This is a resnet backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            input_channels: List[int],
            output_channels: List[int],
            output_feature_sizes: List[Tuple[int]]
        ):
        super().__init__()
        self.in_channels = input_channels
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        model = tvm.resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        backbone = nn.Sequential(*modules)

        self.conv = backbone[0]
        self.bn1 = backbone[1]
        self.relu = backbone[2]
        self.maxpool = backbone[3]
        self.conv1 = backbone[4]
        self.conv2 = backbone[5]
        self.conv3 = backbone[6]
        self.conv4 = backbone[7]
        
        self.conv5 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=input_channels[3],
                out_channels=256,
                kernel_size=3,
                padding=1,
                stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=input_channels[4],
                kernel_size=3,
                padding=1,
                stride=2),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=input_channels[4],
                out_channels=128,
                kernel_size=2,
                padding=1,
                stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=input_channels[5],
                kernel_size=2,
                padding=0,
                stride=2),  # This was 1, changed to make the model run, not sure if correct
            nn.ReLU(),
        )
        self.feature_extractor = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]

        P3_channels, P4_channels, P5_channels, P6_channels, P7_channels, P8_channels = input_channels
        self.channels_bifpn = 64

        self.p7_td_conv  = nn.Conv2d(P7_channels, self.channels_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p7_td_conv_2  = nn.Conv2d(self.channels_bifpn, self.channels_bifpn, kernel_size=3, stride=1, groups=self.channels_bifpn, bias=True, padding=1)
        self.p7_td_act   = nn.ReLU()
        self.p7_td_conv_bn = nn.BatchNorm2d(self.channels_bifpn)
        self.p7_td_w1    = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p7_td_w2    = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p8_upsample   = nn.Upsample(scale_factor=2, mode='nearest')

        self.p6_td_conv  = nn.Conv2d(P6_channels, self.channels_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p6_td_conv_2  = nn.Conv2d(self.channels_bifpn, self.channels_bifpn, kernel_size=3, stride=1, groups=self.channels_bifpn, bias=True, padding=1)
        self.p6_td_act   = nn.ReLU()
        self.p6_td_conv_bn = nn.BatchNorm2d(self.channels_bifpn)
        self.p6_td_w1    = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p6_td_w2    = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p7_upsample   = nn.Upsample(scale_factor=2, mode='nearest')

        self.p5_td_conv  = nn.Conv2d(P5_channels,self.channels_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p5_td_conv_2  = nn.Conv2d(self.channels_bifpn,self.channels_bifpn, kernel_size=3, stride=1, groups=self.channels_bifpn, bias=True, padding=1)
        self.p5_td_act   = nn.ReLU()
        self.p5_td_conv_bn = nn.BatchNorm2d(self.channels_bifpn)
        self.p5_td_w1    = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_td_w2    = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p6_upsample   = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_td_conv  = nn.Conv2d(P4_channels, self.channels_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p4_td_conv_2  = nn.Conv2d(self.channels_bifpn, self.channels_bifpn, kernel_size=3, stride=1, groups=self.channels_bifpn, bias=True, padding=1)
        self.p4_td_act   = nn.ReLU()
        self.p4_td_conv_bn = nn.BatchNorm2d(self.channels_bifpn)
        self.p4_td_w1    = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_td_w2    = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_upsample   = nn.Upsample(scale_factor=2, mode='nearest')


        self.p3_out_conv = nn.Conv2d(P3_channels, self.channels_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p3_out_conv_2 = nn.Conv2d(self.channels_bifpn, self.channels_bifpn, kernel_size=3, stride=1, groups=self.channels_bifpn, bias=True, padding=1)
        self.p3_out_act   = nn.ReLU()
        self.p3_out_conv_bn = nn.BatchNorm2d(self.channels_bifpn)
        self.p3_out_w1   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p3_out_w2   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_upsample  = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_out_conv = nn.Conv2d(self.channels_bifpn, self.channels_bifpn, kernel_size=3, stride=1, groups=self.channels_bifpn, bias=True, padding=1)
        self.p4_out_act   = nn.ReLU()
        self.p4_out_conv_bn = nn.BatchNorm2d(self.channels_bifpn)
        self.p4_out_w1   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w2   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w3   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p3_downsample= nn.MaxPool2d(kernel_size=2)

        self.p5_out_conv = nn.Conv2d(self.channels_bifpn,self.channels_bifpn, kernel_size=3, stride=1, groups=self.channels_bifpn, bias=True, padding=1)
        self.p5_out_act   = nn.ReLU()
        self.p5_out_conv_bn = nn.BatchNorm2d(self.channels_bifpn)
        self.p5_out_w1   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_out_w2   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_out_w3   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_downsample= nn.MaxPool2d(kernel_size=2)

        self.p6_out_conv = nn.Conv2d(self.channels_bifpn, self.channels_bifpn, kernel_size=3, stride=1, groups=self.channels_bifpn, bias=True, padding=1)
        self.p6_out_act   = nn.ReLU()
        self.p6_out_conv_bn = nn.BatchNorm2d(self.channels_bifpn)
        self.p6_out_w1   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p6_out_w2   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p6_out_w3   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_downsample= nn.MaxPool2d(kernel_size=2)

        self.p7_out_conv = nn.Conv2d(self.channels_bifpn, self.channels_bifpn, kernel_size=3, stride=1, groups=self.channels_bifpn, bias=True, padding=1)
        self.p7_out_act   = nn.ReLU()
        self.p7_out_conv_bn = nn.BatchNorm2d(self.channels_bifpn)
        self.p7_out_w1   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p7_out_w2   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p7_out_w3   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p6_downsample= nn.MaxPool2d(kernel_size=2)


        self.p8_out_conv = nn.Conv2d(P8_channels,self.channels_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p8_out_conv_2 = nn.Conv2d(self.channels_bifpn,self.channels_bifpn, kernel_size=3, stride=1, groups=self.channels_bifpn, bias=True, padding=1)
        self.p8_out_act  = nn.ReLU()
        self.p8_out_conv_bn = nn.BatchNorm2d(self.channels_bifpn)
        self.p8_out_w1   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p8_out_w2   = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p7_downsample= nn.MaxPool2d(kernel_size=2)

        

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
        #print("x: ", x.shape)
        out_features = []
        #out_features = nn.ModuleList
        out_features_keys = ["c1","c2","c3","c4","c5","c6"]
        
        #First layers of ResNet50 
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.conv1(x)
        out_features.append(c1)
        #print("c1: ", c1.shape)
        c2 = self.conv2(c1)
        out_features.append(c2)
        #print("c2: ", c2.shape)
        c3 = self.conv3(c2)
        out_features.append(c3)
        #print("c3: ", c3.shape)
        c4 = self.conv4(c3)
        out_features.append(c4)
        #print("c4: ", c4.shape)
        c5 = self.conv5(c4)
        out_features.append(c5)
        #print("c5: ", c5.shape)
        c6 = self.conv6(c5)
        out_features.append(c6)
        #print("c6: ", c6.shape)
        output_dict = dict(zip(out_features_keys, out_features))
        
        # print("Out 0: ", out_features[0].shape)
        # print("Out 1: ", out_features[1].shape)
        # print("Out 2: ", out_features[2].shape)
        # print("Out 3: ", out_features[3].shape)
        # print("Out 4: ", out_features[4].shape)
        # print("Out 5: ", out_features[5].shape)
        epsilon = 0.001
        P3, P4, P5, P6, P7, P8 = out_features
        #P8, P7, P6, P5, P4, P3 = out_features

        P8_td  = self.p8_out_conv(P8)
        
        P7_td_inp = self.p7_td_conv(P7)
        P7_td = self.p7_td_conv_2((self.p7_td_w1 * P7_td_inp + self.p7_td_w2 * self.p8_upsample(P8_td)) /
                                 (self.p7_td_w1 + self.p7_td_w2 + epsilon))
    
        P7_td = self.p7_td_act(P7_td)
        P7_td = self.p7_td_conv_bn(P7_td)

        P6_td_inp = self.p6_td_conv(P6)
        P6_td = self.p6_td_conv_2((self.p6_td_w1 * P6_td_inp + self.p6_td_w2 * self.p7_upsample(P7_td)) /
                                 (self.p6_td_w1 + self.p6_td_w2 + epsilon))
        P6_td = self.p6_td_act(P6_td)
        P6_td = self.p6_td_conv_bn(P6_td)
         
        P5_td_inp = self.p5_td_conv(P5)
        P5_td = self.p5_td_conv_2((self.p5_td_w1 * P5_td_inp + self.p5_td_w2 * self.p6_upsample(P6_td)) /
                                 (self.p5_td_w1 + self.p5_td_w2 + epsilon))
        P5_td = self.p5_td_act(P5_td)
        P5_td = self.p5_td_conv_bn(P5_td)
        
        P4_td_inp = self.p4_td_conv(P4)
        P4_td = self.p4_td_conv_2((self.p4_td_w1 * P4_td_inp + self.p4_td_w2 * self.p5_upsample(P5_td)) /
                                 (self.p4_td_w1 + self.p4_td_w2 + epsilon))
        P4_td = self.p4_td_act(P4_td)
        P4_td = self.p4_td_conv_bn(P4_td)


        P3_td  = self.p3_out_conv(P3)
        P3_out = self.p3_out_conv_2((self.p3_out_w1 * P3_td + self.p3_out_w2 * self.p4_upsample(P4_td)) /
                                 (self.p3_out_w1 + self.p3_out_w2 + epsilon))
        P3_out = self.p3_out_act(P3_out)
        P3_out = self.p3_out_conv_bn(P3_out)

        #print ("SHAPS43: ", P4_td.shape, P3_out.shape)

        P4_out = self.p4_out_conv((self.p4_out_w1 * P4_td_inp  + self.p4_out_w2 * P4_td + self.p4_out_w3 * self.p3_downsample(P3_out) )
                                    / (self.p4_out_w1 + self.p4_out_w2 + self.p4_out_w3 + epsilon))
        P4_out = self.p4_out_act(P4_out)
        P4_out = self.p4_out_conv_bn(P4_out)

        #print ("SHAPS54: ", P5_td.shape, P4_out.shape)
        P5_out = self.p5_out_conv(( self.p5_out_w1 * P5_td_inp + self.p5_out_w2 * P5_td + self.p5_out_w3 * self.p4_downsample(P4_out) )
                                    / (self.p5_out_w2 + self.p5_out_w3 + epsilon))
        P5_out = self.p5_out_act(P5_out)
        P5_out = self.p5_out_conv_bn(P5_out)

        #print ("SHAPS65: ", P6_td.shape, P6_td_inp.shape, P5_out.shape, self.p5_downsample(P5_out).shape)
        P6_out = self.p6_out_conv((self.p6_out_w1 * P6_td_inp + self.p6_out_w2 * P6_td + self.p6_out_w3 * self.p5_downsample(P5_out) )
                                    / (self.p6_out_w1 + self.p6_out_w2 + self.p6_out_w3 + epsilon))
        P6_out = self.p6_out_act(P6_out)
        P6_out = self.p6_out_conv_bn(P6_out)

        #print ("SHAPS76: ", P7_td.shape, P7_td_inp.shape, P6_out.shape)
        P7_out = self.p7_out_conv((self.p7_out_w1 * P7_td_inp + self.p7_out_w2 * P7_td + self.p7_out_w3 * self.p6_downsample(P6_out) )
                                    / (self.p7_out_w1 + self.p7_out_w2 + self.p7_out_w3 + epsilon))
        P7_out = self.p7_out_act(P7_out)
        P7_out = self.p7_out_conv_bn(P7_out)


        P8_out = self.p8_out_conv_2((self.p8_out_w1 * P8_td + self.p8_out_w2 * self.p7_downsample(P7_out)) /
                                 (self.p8_out_w1 + self.p8_out_w2 + epsilon))
        P8_out = self.p8_out_act(P8_out)
        P8_out = self.p8_out_conv_bn(P8_out)
        

        out_features = [P3_out, P4_out, P5_out, P6_out, P7_out, P8_out]
        
        # for feature in out_features:
        #     print("feature size: ", feature.shape)
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)