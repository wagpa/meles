import logging
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torchvision import models
import torch.nn.functional as fun

from meles.common import Image, Images
from meles.frame_aggregation.aggregator import FrameAggregator
from meles.frame_aggregation.iresnet import iresnet34

logger = logging.getLogger(__name__)


class FacialEncoder(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.model = iresnet34() # Hidden state: B x 512 x 7 x 7
        if weights:
            sd = torch.load(weights, map_location="cpu")
            sd = {k.removeprefix("module."): v for k, v in sd.items()}
            self.model.load_state_dict(sd, strict=True)

    def forward(self, x):
        return self.model(x)[1]


class FacialDecoderBlock(nn.Module):
    def __init__(self, cin, cout, groups=8):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.norm1 = nn.GroupNorm(groups, cout)
        self.norm2 = nn.GroupNorm(groups, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        x = self.up(x)
        h = fun.silu(self.norm1(self.conv1(x)))
        h = fun.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class FacialDecoder(nn.Module):
    def __init__(self, cin=256, chans=(256, 192, 128, 64)):
        super().__init__()
        self.blocks = nn.ModuleList()
        for c in chans:
            self.blocks.append(FacialDecoderBlock(cin, c))
            cin = c
        self.out = nn.Sequential(
            nn.GroupNorm(8, cin),
            nn.SiLU(),
            nn.Conv2d(cin, 3, 3, 1, 1)
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return torch.tanh(self.out(x))


class FacialAggregator(nn.Module):
    def __int__(self, cin=512, heads=8, groups=8):
        self.attn = nn.MultiheadAttention(cin, heads)
        self.norm1 = nn.GroupNorm(groups, cin)

    def forward(self, x):
        return x # implement me!



class Facial(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FacialEncoder(weights=None) # TODO add weights
        self.aggregator = FacialAggregator()
        self.decoder = FacialDecoder()
