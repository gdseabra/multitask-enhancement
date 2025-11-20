""" Parts of the U-Net model with scSE Blocks added """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class SCSEBlock(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation Block
    Based on Roy et al., MICCAI 2018
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        
        # Channel Squeeze and Excitation (cSE)
        # Squeeze: Global Average Pooling
        # Excitation: Fully Connected Layers
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(1, in_channels // reduction), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // reduction), in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Spatial Squeeze and Excitation (sSE)
        # Squeeze: 1x1 Convolution to project to 1 channel
        # Excitation: Sigmoid to generate spatial mask
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Path 1: Channel Attention
        # Recalibrates the "what" (feature importance)
        x_cse = x * self.cSE(x)
        
        # Path 2: Spatial Attention
        # Recalibrates the "where" (spatial importance / noise suppression)
        x_sse = x * self.sSE(x)
        
        # Concurrent: Add the two paths together
        return x_cse + x_sse


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 => SCSEBlock"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Add the scSE block after the convolutions
        self.scse = SCSEBlock(out_channels)

    def forward(self, x):
        x = self.double_conv(x)
        x = self.scse(x) # Apply attention here
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=90, ndim=2, chs: tuple[int, ...] = (64, 128, 256, 512, 1024)):
        super(UNet, self).__init__()
        self.n_channels = in_ch
        self.n_classes = out_ch

        bilinear = True
        self.bilinear = bilinear

        self.inc = (DoubleConv(self.n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, out_ch))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


if __name__ == '__main__':
    # Assuming input is 1 channel (grayscale fingerprint) and output is 2 (e.g., Sin/Cos of orientation)
    # or 90 (classification bins), based on your 'out_ch' default.
    model         = UNet(in_ch=1, out_ch=90)

    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model         = model.to(device)
    
    print(f"Model with scSE blocks initialized on {device}")
    
    # Testing with a standard fingerprint resolution
    summary(model, (1, 512, 512))