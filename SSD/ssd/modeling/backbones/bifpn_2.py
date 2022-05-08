from turtle import forward

from pyrsistent import freeze
import torchvision
import torch
from torch import nn
from typing import Tuple, List
from collections import OrderedDict
from torch.autograd import Variable

class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False) -> None:
        super(DepthwiseConvBlock,self).__init__()
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                               padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                   stride, padding, dilation, groups=1, bias=False)
        
        self.bn = nn.BatchNorm2d(256, momentum=0.9997, eps=4e-5)
        self.act = nn.GELU()
        
    def forward(self, inputs):
                
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class BiFPNBlock(torch.nn.Module):
    def __init__(self):
        super(BiFPNBlock, self).__init__()
        self.epsilon = 1e-4
        
        self.convDP6 = DepthwiseConvBlock(in_channels=256, out_channels=256)
        self.convDP5 = DepthwiseConvBlock(in_channels=256, out_channels=256)
        self.convDP4 = DepthwiseConvBlock(in_channels=256, out_channels=256)
        self.convDP3 = DepthwiseConvBlock(in_channels=256, out_channels=256)
        self.convDP2 = DepthwiseConvBlock(in_channels=256, out_channels=256)

        self.convUP7 = DepthwiseConvBlock(in_channels=256, out_channels=256)
        self.convUP6 = DepthwiseConvBlock(in_channels=256, out_channels=256)
        self.convUP5 = DepthwiseConvBlock(in_channels=256, out_channels=256)
        self.convUP4 = DepthwiseConvBlock(in_channels=256, out_channels=256)
        self.convUP3 = DepthwiseConvBlock(in_channels=256, out_channels=256)
        
        self.w1 = torch.nn.Parameter(torch.ones(2, 5))
        self.w1_relu = nn.GELU()
        self.w2 = torch.nn.Parameter(torch.ones(3, 5))
        self.w2_relu = nn.GELU()
        
    def forward(self, inputs):        
        P2, P3, P4, P5, P6, P7 = inputs

        # Top-down pathway
        w1 = self.w1_relu(self.w1)
        w1 = w1 / torch.sum(w1, dim=0) + self.epsilon
        w2 = self.w2_relu(self.w2)
        w2 = w2 / torch.sum(w2, dim=0) + self.epsilon
        
        DP7 = P7
        DP6 = self.convDP6(w1[0, 0] * P6 + w1[1, 0] * nn.functional.interpolate(DP7, scale_factor=2))
        DP5 = self.convDP5(w1[0, 1] * P5 + w1[1, 1] * nn.functional.interpolate(DP6, scale_factor=2))
        DP4 = self.convDP4(w1[0, 2] * P4 + w1[1, 2] * nn.functional.interpolate(DP5, scale_factor=2))
        DP3 = self.convDP3(w1[0, 3] * P3 + w1[1, 3] * nn.functional.interpolate(DP4, scale_factor=2))
        DP2 = self.convDP2(w1[0, 4] * P2 + w1[1, 4] * nn.functional.interpolate(DP3, scale_factor=2))
            
        # DP7 = P7
        # DP6 = self.convDP6(P6 + nn.functional.interpolate(DP7, scale_factor=2))
        # DP5 = self.convDP5(P5 + nn.functional.interpolate(DP6, scale_factor=2))
        # DP4 = self.convDP4(P4 + nn.functional.interpolate(DP5, scale_factor=2))
        # DP3 = self.convDP3(P3 + nn.functional.interpolate(DP4, scale_factor=2))
        # DP2 = self.convDP2(P2 + nn.functional.interpolate(DP3, scale_factor=2))    
                
        # Bottom-up pathway
        UP2 = DP2
        UP3 = self.convUP3(w2[0, 0] * DP3 + w2[1, 0] * P3 + w2[2, 0] * nn.Upsample(scale_factor=0.5)(UP2))
        UP4 = self.convUP4(w2[0, 1] * DP4 + w2[1, 1] * P4 + w2[2, 1] * nn.Upsample(scale_factor=0.5)(UP3))
        UP5 = self.convUP5(w2[0, 2] * DP5 + w2[1, 2] * P5 + w2[2, 2] * nn.Upsample(scale_factor=0.5)(UP4))
        UP6 = self.convUP6(w2[0, 3] * DP6 + w2[1, 3] * P6 + w2[2, 3] * nn.Upsample(scale_factor=0.5)(UP5))
        UP7 = self.convUP7(w2[0, 4] * DP7 + w2[1, 4] * P7 + w2[2, 4] * nn.Upsample(scale_factor=0.5)(UP6))
        
        # UP2 = DP2
        # UP3 = self.convUP3(DP3 + P3 + nn.Upsample(scale_factor=0.5)(UP2))
        # UP4 = self.convUP4(DP4 + P4 + nn.Upsample(scale_factor=0.5)(UP3))
        # UP5 = self.convUP5(DP5 + P5 + nn.Upsample(scale_factor=0.5)(UP4))
        # UP6 = self.convUP6(DP6 + P6 + w2[2, 3] * nn.Upsample(scale_factor=0.5)(UP5))
        # UP7 = self.convUP7(w2[0, 4] * DP7 + w2[1, 4] * P7 + w2[2, 4] * nn.Upsample(scale_factor=0.5)(UP6))     
                
        return [UP2, UP3, UP4, UP5, UP6, UP7]
        

class FPN(torch.nn.Module):
    def __init__(self, output_channels: List[int],
            #image_channels: int,
            #output_feature_sizes: List[Tuple[int]]
            ):
        super().__init__()
        self.out_channels = output_channels
        # self.output_feature_shape = output_feature_sizes

        self.oldModel = torchvision.models.resnet34(pretrained=True)
        # self.model = torch.nn.Sequential(*(list(self.oldmodel.children())[4:-2]))
        
        ## Add feature extractors
        ###############################################################################################################
        self.feature_extractorP2 = torch.nn.Sequential(                                     # 32x256
            self.oldModel.conv1,
            self.oldModel.bn1,
            self.oldModel.relu,
            self.oldModel.maxpool,
            self.oldModel.layer1,
        )
        self.feature_extractorP3 = torch.nn.Sequential(self.oldModel.layer2)                # 16x128
        self.feature_extractorP4 = torch.nn.Sequential(self.oldModel.layer3)                # 8x64
        self.feature_extractorP5 = torch.nn.Sequential(self.oldModel.layer4)                # 4x32
        self.feature_extractorP6 = torch.nn.Sequential(                                     # 2x16
            nn.Conv2d(
                in_channels=512,
                #out_channels=64,
                out_channels=256,  
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        self.feature_extractorP7 = torch.nn.Sequential(                                     # 1x8
            nn.Conv2d(
                #in_channels=64,
                #out_channels=64,
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        
        self.convP7 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)
        self.convP6 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)
        self.convP5 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        self.convP4 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1)
        self.convP3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1)
        self.convP2 = torch.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1)
        
        bifpns = []
        for _ in range(3):
            bifpns.append(BiFPNBlock())
       
        self.biFPNs = nn.Sequential(*bifpns)
        
        # Extract features from P2-P7
        #self.feature_extractorFPN = torchvision.ops.FeaturePyramidNetwork([64, 128, 256, 512, 64, 64], 64)
        self.feature_extractorFPN = torchvision.ops.FeaturePyramidNetwork([64, 128, 256, 512, 256, 256], 256)

    def forward(self, x):

        ## Custom Backbone
        P2 = self.feature_extractorP2(x)
        P3 = self.feature_extractorP3(P2)
        P4 = self.feature_extractorP4(P3)
        P5 = self.feature_extractorP5(P4)
        P6 = self.feature_extractorP6(P5)
        P7 = self.feature_extractorP7(P6)
        
        # After this 256 feature maps
        P2 = self.convP2(P2)
        P3 = self.convP3(P3)
        P4 = self.convP4(P4)
        P5 = self.convP5(P5)
        P6 = self.convP6(P6)
        P7 = self.convP7(P7)
        
            
        #BiFPN        
        BiFPNout = self.biFPNs([P2, P3, P4, P5, P6, P7])        
        
        ## CAM
        # from pytorch_grad_cam import GradCAM
        # from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        # from pytorch_grad_cam.utils.image import show_cam_on_image
        #
        # targets = [Class(9)]
        # cam = GradCAM(self.feature_extractorFPN)
        # cam_output = cam(input_tensor=x, targets=targets)
        # cam_output = cam_output[0, :]
        # viz = show_cam_on_image(rgb_img, cam_output, use_rgb=True)
                
        return tuple(BiFPNout)
        
        ## FPN
        FeatureMaps = OrderedDict()
        FeatureMaps['P2'] = P2
        FeatureMaps['P3'] = P3
        FeatureMaps['P4'] = P4
        FeatureMaps['P5'] = P5
        FeatureMaps['P6'] = P6
        FeatureMaps['P7'] = P7
        
        outFeatures = []
        FPNout = self.feature_extractorFPN(FeatureMaps)

        for _, v in FPNout.items():
            outFeatures.append(v)
        
        return tuple(outFeatures)