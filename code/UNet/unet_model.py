import torch
import torch.nn as nn
from unet_parts import *
from class_lib import SEBlock

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, se_mode='avg', se_ratio=16):
        super().__init__()
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        #self.se1 = SEBlock(mode=se_mode, channels=128, ratio=se_ratio)

        self.down2 = Down(128, 256)
        #self.se2 = SEBlock(mode=se_mode, channels=256, ratio=se_ratio)

        self.down3 = Down(256, 512)
        #self.se3 = SEBlock(mode=se_mode, channels=512, ratio=se_ratio)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        #self.se4 = SEBlock(mode=se_mode, channels=1024 // factor, ratio=se_ratio)

        self.up1 = Up(1024, 512 // factor, bilinear)
        #self.se5 = SEBlock(mode=se_mode, channels=512 // factor, ratio=se_ratio)

        self.up2 = Up(512, 256 // factor, bilinear)
        #self.se6 = SEBlock(mode=se_mode, channels=256 // factor, ratio=se_ratio)

        self.up3 = Up(256, 128 // factor, bilinear)
        #self.se7 = SEBlock(mode=se_mode, channels=128 // factor, ratio=se_ratio)

        self.up4 = Up(128, 64, bilinear)
        #self.se8 = SEBlock(mode=se_mode, channels=64, ratio=se_ratio)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        #x2 = self.se1(x2)

        x3 = self.down2(x2)
        #x3 = self.se2(x3)

        x4 = self.down3(x3)
        #x4 = self.se3(x4)

        x5 = self.down4(x4)
        #x5 = self.se4(x5)

        x = self.up1(x5, x4)
        #x = self.se5(x)

        x = self.up2(x, x3)

        x = self.up3(x, x2)
        #x = self.se7(x)

        x = self.up4(x, x1)
        #x = self.se8(x)

        logits = self.outc(x)
        return logits
