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
        self.pool = nn.AvgPool2d(2)  # Added 2x2 Average Pooling

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)  # Apply pooling after convolution
        return x


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=91, ndim=2, chs: tuple[int, ...] = (64, 128, 256, 512, 1024)):
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
        
        # Apply the DoubleConvWithPooling only to the last Up block (up4)
        self.up4 = (Up(128, 64, bilinear, conv_block=DoubleConvWithPooling))
        
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
    model         = UNet(in_ch=3)

    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model         = model.to(device)

    print("--- Model Summary (with changes) ---")
    summary(model, (3, 256, 256))
    
    # Example to trace tensor shapes:
    print("\n--- Tracing Tensor Shapes ---")
    test_tensor = torch.randn(1, 3, 256, 256).to(device)
    x1 = model.inc(test_tensor)
    print(f'x1 (inc): \t\t{x1.shape}')
    x2 = model.down1(x1)
    print(f'x2 (down1): \t\t{x2.shape}')
    x3 = model.down2(x2)
    print(f'x3 (down2): \t\t{x3.shape}')
    x4 = model.down3(x3)
    print(f'x4 (down3): \t\t{x4.shape}')
    x5 = model.down4(x4)
    print(f'x5 (down4): \t\t{x5.shape}')
    x = model.up1(x5, x4)
    print(f'up1 output: \t\t{x.shape}')
    x = model.up2(x, x3)
    print(f'up2 output: \t\t{x.shape}')
    x = model.up3(x, x2)
    print(f'up3 output: \t\t{x.shape}')
    x = model.up4(x, x1)
    print(f'up4 (with pooling): \t{x.shape}') # Notice the size reduction here
    logits = model.outc(x)
    print(f'logits (outc): \t\t{logits.shape}') # And the final size reduction here