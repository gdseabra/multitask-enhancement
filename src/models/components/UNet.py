""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# --- Helper module for ASPP ---
class ASPPConv(nn.Sequential):
    """Convolutional block for ASPP: Conv2d -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

# --- Helper module for ASPP ---
class ASPPPooling(nn.Sequential):
    """Image-level pooling branch for ASPP"""
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:] # Get H, W
        x = super(ASPPPooling, self).forward(x)
        # Upsample back to the original H, W
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

# --- Main ASPP Module ---
class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP)
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        rates (list[int]): List of dilation rates for the atrous convolutions.
    """
    def __init__(self, in_channels, out_channels, rates=(3, 6, 9)):
        super(ASPP, self).__init__()

        # --- Dilation Rate Selection ---
        # With a 512x512 input, 4 max-pools (2^4) result in a 32x32
        # feature map at the bottleneck.
        # Standard rates like [6, 12, 18] are too large for a 32x32 map
        # (e.g., rate=18 has a 37-pixel field).
        # We choose [3, 6, 9], which have receptive fields of [7, 13, 19].
        # This captures local, medium, and broad context within the 32x32
        # map, ideal for noisy fingerprint features.

        inter_channels = 256 # Intermediate channels for each branch

        # 1. 1x1 Convolution Branch
        self.conv1 = ASPPConv(in_channels, inter_channels, kernel_size=1, padding=0, dilation=1)

        # 2. Atrous Convolution Branches
        self.convs = nn.ModuleList()
        for rate in rates:
            self.convs.append(
                ASPPConv(in_channels, inter_channels, kernel_size=3, 
                         padding=rate, dilation=rate)
            )

        # 3. Image Pooling Branch
        self.pool = ASPPPooling(in_channels, inter_channels)

        # --- Final Projection Layer ---
        # Total channels = (1 (1x1) + len(rates) + 1 (pool)) * inter_channels
        total_channels = (1 + len(rates) + 1) * inter_channels
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5) # Dropout is common in ASPP
        )

    def forward(self, x):
        branches = [self.conv1(x)] # 1x1 conv
        branches.extend([conv(x) for conv in self.convs]) # Atrous convs
        branches.append(self.pool(x)) # Image pooling

        # Concatenate all branches
        x_cat = torch.cat(branches, dim=1)
        
        # Apply final 1x1 projection
        return self.project(x_cat)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

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

    def forward(self, x):
        return self.double_conv(x)


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
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        # return F.interpolate(x, scale_factor=1/4, mode='bilinear')
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
        
        # Instead of the original self.down4, we split the max-pooling
        # and the convolution. The convolution is replaced by ASPP.
        self.down4_pool = nn.MaxPool2d(2)
        self.bottleneck_aspp = ASPP(512, 1024 // factor, rates=(3, 6, 9))
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
        x_pool = self.down4_pool(x4)
        x5 = self.bottleneck_aspp(x_pool)

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
        self.down4_pool = torch.utils.checkpoint(self.down4_pool)
        self.bottleneck_aspp = torch.utils.checkpoint(self.bottleneck_aspp)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


if __name__ == '__main__':
    model         = UNet(in_ch=1)

    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model         = model.to(device)

    summary(model, (1, 512, 512))