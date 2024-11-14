import copy
import os
import random
import shutil
import zipfile
from math import atan2, cos, sin, sqrt, pi, log
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pylab as p
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from numpy import linalg as LA
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm
from torch.nn import Module, MaxPool2d
from torchvision.transforms import CenterCrop
from torchvision import models


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # return self.sigmoid(self.conv(x))
        return self.conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)

class DecoderBlockv2(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=1):
        super().__init__()
        if upsample:
            self.upconv = nn.ConvTranspose2d(in_channels * 2, in_channels * 2, kernel_size=2, stride=2)
        else:
            self.upconv = nn.Identity()
        self.layers = DoubleConv(in_channels * 2, out_channels)

    def forward(self, x, skip_connection):

        target_height = x.size(2)
        target_width = x.size(3)
        skip_interp = F.interpolate(
            skip_connection, size=(target_height, target_width), mode='bilinear', align_corners=False)

        concatenated = torch.cat([skip_interp, x], dim=1)

        concatenated = self.upconv(concatenated)

        output = self.layers(concatenated)
        return output


class UNetFT(nn.Module):
    def __init__(self, n_classes, pretrained=True,
                 in_channels =1, layer1_features=32, layer2_features=16,
                 layer3_features=24, layer4_features=40, layer5_features=80):
        super(UNetFT, self).__init__()
        self.effnet = models.efficientnet_b0(pretrained=pretrained)

        self.n_classes = n_classes

        self.input_features = in_channels
        self.layer1_features = layer1_features
        self.layer2_features = layer2_features
        self.layer3_features = layer3_features
        self.layer4_features = layer4_features
        self.layer5_features = layer5_features


        self.effnet.features[0][0] = nn.Conv2d(in_channels, layer1_features, kernel_size=3, stride=2, padding=1,
                                               bias=False)
        self.encoder1 = nn.Sequential(*list(self.effnet.features.children())[0])
        self.encoder2 = nn.Sequential(*list(self.effnet.features.children())[1])
        self.encoder3 = nn.Sequential(*list(self.effnet.features.children())[2])
        self.encoder4 = nn.Sequential(*list(self.effnet.features.children())[3])
        self.encoder5 = nn.Sequential(*list(self.effnet.features.children())[4])

        del self.effnet

        for param in self.encoder1.parameters():
            param.requires_grad = False
        for param in self.encoder2.parameters():
            param.requires_grad = False

        self.bottleneck = DoubleConv(self.layer5_features, self.layer5_features)


        self.decoder1 = DecoderBlockv2(self.layer5_features, self.layer4_features)
        self.decoder2 = DecoderBlockv2(self.layer4_features, self.layer3_features)
        self.decoder3 = DecoderBlockv2(self.layer3_features, self.layer2_features)
        self.decoder4 = DecoderBlockv2(self.layer2_features, self.layer1_features, upsample=0)
        self.decoder5 = DecoderBlockv2(self.layer1_features, self.layer1_features)


        self.final_conv = OutConv(self.layer1_features, self.n_classes)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
        x = self.bottleneck(x5)
        x = self.decoder1(x, x5)
        x = self.decoder2(x, x4)
        x = self.decoder3(x, x3)
        x = self.decoder4(x, x2)
        x = self.decoder5(x, x1)
        logits = self.final_conv(x)
        return logits





