import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import torch
import torch.nn as nn
import numpy as np
import cv2

def generate_gabor_kernels(num_orientations, ksize, sigma, lambd, gamma=0.5):
    """
    Generates a stack of Gabor filter kernels.

    Args:
        num_orientations (int): Number of orientations (e.g., 90).
        ksize (int): The size of the Gabor kernel (e.g., 31).
        sigma (float): Standard deviation of the Gaussian envelope.
        lambd (float): Wavelength of the sinusoidal factor.
        gamma (float): Spatial aspect ratio.

    Returns:
        torch.Tensor: A tensor of shape (num_orientations, 1, ksize, ksize)
                      containing the Gabor kernels.
    """
    kernels = []
    # Orientations from 0 to 178 degrees, matching your U-Net output
    for i in range(num_orientations):
        theta = i * np.pi / num_orientations # Angle in radians
        kernel = cv2.getGaborKernel(
            (ksize, ksize), 
            sigma, 
            theta, 
            lambd, 
            gamma, 
            psi=0, # Phase offset, 0 and pi/2 are common
            ktype=cv2.CV_32F
        )
        # Add a channel dimension for PyTorch compatibility
        kernels.append(kernel)
    
    # Stack kernels into a single tensor
    gabor_kernels = np.stack(kernels, axis=0)
    # Add the 'in_channels' dimension
    gabor_kernels = torch.from_numpy(gabor_kernels).unsqueeze(1)
    
    return gabor_kernels

class GaborConvLayer(nn.Module):
    def __init__(self, num_orientations=90, ksize=31, sigma=4.0, lambd=10.0):
        super(GaborConvLayer, self).__init__()
        
        # Generate the fixed Gabor kernels
        gabor_weights = generate_gabor_kernels(num_orientations, ksize, sigma, lambd)
        
        # Create a non-trainable Conv2d layer
        self.conv = nn.Conv2d(
            in_channels=1, 
            out_channels=num_orientations, 
            kernel_size=ksize, 
            padding='same', # Preserves input spatial dimensions
            bias=False
        )
        
        # Assign the fixed Gabor weights and make them non-trainable
        self.conv.weight = nn.Parameter(gabor_weights, requires_grad=False)

    def forward(self, x):
        # Apply the convolution
        return self.conv(x)

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
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return F.interpolate(x, scale_factor=1/4, mode='bilinear')


# ---------------------- MULTITASK U-NET ---------------------- #

class UNetEncoder(nn.Module):
    """Shared encoder"""
    def __init__(self, in_ch=1, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class UNetDecoder(nn.Module):
    """One decoder head"""
    def __init__(self, out_ch, bilinear=True):
        super().__init__()
        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_ch)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class MultiTaskUNet(nn.Module):
    """Shared encoder with two task-specific decoders"""
    def __init__(self, in_ch=1, out_ch=91, ndim=2, chs: tuple[int, ...] = (64, 128, 256, 512, 1024)):
        super().__init__()
        bilinear = True
        self.encoder = UNetEncoder(in_ch=in_ch, bilinear=bilinear)
        self.decoder_orient = UNetDecoder(90, bilinear)
        self.decoder_seg = UNetDecoder(1, bilinear)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        out_orient = self.decoder_orient(x1, x2, x3, x4, x5)
        out_seg = self.decoder_seg(x1, x2, x3, x4, x5)
        return out_orient, out_seg


if __name__ == '__main__':
    model = MultiTaskUNet(in_ch=3, out_ch_orient=90, out_ch_seg=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    summary(model, (3, 256, 256))
