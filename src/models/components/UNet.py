""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


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


class DoubleConvWithPooling(nn.Module):
    """(convolution => [BN] => ReLU => MaxPool2d) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        # Using nn.Sequential for each block for clarity
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # Added Max Pooling
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # Added Max Pooling
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
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

    def __init__(self, in_channels, out_channels, bilinear=True, conv_block=DoubleConv):
        """
        Added conv_block argument to flexibly change the convolution type.
        """
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = conv_block(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = conv_block(in_channels, out_channels)

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
        return x


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=90, ndim=2, chs: tuple[int, ...] = (32, 64, 128, 256, 512, 1024)):
        super(UNet, self).__init__()
        
        # --- REFACTORED SECTION ---
        # This init method now uses the `chs` tuple to define layer channels.
        # The default `chs` (64, 128, 256, 512, 1024) replicates the original
        # hardcoded behavior.
        
        self.n_channels = in_ch
        self.n_classes = out_ch
        
        # Note: The original code hardcoded bilinear=True. We preserve this behavior.
        # The `ndim` parameter was unused, so it is ignored.
        bilinear = True 
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        # Encoder (Down-sampling path)
        self.inc = (DoubleConv(self.n_channels, chs[0]))
        self.down1 = (Down(chs[0], chs[1])) # 64 -> 128 (H/2, W/2)
        self.down2 = (Down(chs[1], chs[2])) # 128 -> 256 (H/4, W/4)
        self.down3 = (Down(chs[2], chs[3])) # 256 -> 512 (H/8, W/8)
        self.down4 = (Down(chs[3], chs[4])) # 512 -> 1024 (H/16, W/16)

        # Bottleneck layer
        self.down5 = (Down(chs[4], chs[5] // factor)) # 1024 -> 1024 (H/32, W/32)

        # Decoder (Up-sampling path)
        self.up1 = (Up(chs[5], chs[4] // factor, bilinear)) # 1048+1048 -> 256  
        self.up2 = (Up(chs[4], chs[3] // factor, bilinear))
        
        self.outc = (OutConv(chs[3] // factor, out_ch))
        # --- END OF REFACTORED SECTION ---

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        logits = self.outc(x)
        return logits




if __name__ == '__main__':
    model         = UNet(in_ch=1, out_ch=90)

    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model         = model.to(device)

    print("--- Model Summary (with changes) ---")
    summary(model, (1, 512, 512))