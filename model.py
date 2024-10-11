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


class Block(Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 5)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 5)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(Module):
    def __init__(self, channels=(1, 16, 32, 64)):
        super(Encoder, self).__init__()
        self.enBlocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1])
             for i in range(len(channels) - 1)]
        )
        self.pool = MaxPool2d(2)

    def forward(self, x):
        # empty list that stores intermediate output
        block_output = []
        for block in self.enBlocks:
            x = block(x)
            block_output.append(x)
            x = self.pool(x)
        return block_output


class Decoder(Module):
    def __init__(self, channels=(1, 16, 32, 64)):
        super(Decoder, self).__init__()
        self.channels = channels
        self.upconvs = nn.ModuleList(
            nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
            for i in range(len(channels) - 1)
        )
        self.dec_blocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1])
             for i in range(len(channels) - 1)])

    def forward(self, x, enc_features):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            enc_feat = self.crop(enc_features[i], x)
            x = torch.cat([x, enc_feat], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_features, x):
        (_, _, H, W) = x.shape
        enc_features = CenterCrop([H, W])(enc_features)
        return enc_features


class UNet(Module):
    def __init__(self, enc_channels=(1, 16, 32, 64),
                 dec_channels=(64, 32, 16),
                 nb_classes=1, retainDim=True,
                 outSize=(256, 256)):
        super(UNet, self).__init__()
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)

        self.head = nn.Conv2d(dec_channels[-1], nb_classes, 1)
        self.retainDim = retainDim
        self.outSize = outSize

    def forward(self, x):

        enc_features = self.encoder(x)

        dec_features = self.decoder(enc_features[::-1][0],
                                   enc_features[::-1][1:])

        map = self.head(dec_features)

        if self.retainDim:
            map = F.interpolate(map, self.outSize, mode='bilinear')
        return map