import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from thop import clever_format, profile
from torchsummary import summary

def autopad(k, p=None, d=1):  
    '''
    k: kernel
    p: padding
    d: dilation
    '''
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k] # actual kernel-size
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, paddin g, groups, dilation, activation)."""
    default_act = nn.GELU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depth-wise convolution with args(ch_in, ch_out, kernel, stride, dilation, activation)."""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

    
# Lightweight Cascade Multi-Receptive Fields Module
class CMRF(nn.Module):
    """CMRF Module with args(ch_in, ch_out, number, shortcut, groups, expansion)."""
    def __init__(self, c1, c2, N=8, shortcut=True, g=1, e=0.5):
        super().__init__()
        
        self.N         = N
        self.c         = int(c2 * e / self.N)
        self.add       = shortcut and c1 == c2
        
        self.pwconv1   = Conv(c1, c2//self.N, 1, 1)
        self.pwconv2   = Conv(c2//2, c2, 1, 1)
        self.m         = nn.ModuleList(DWConv(self.c, self.c, k=3, act=False) for _ in range(N-1))

    def forward(self, x):
        """Forward pass through CMRF Module."""
        x_residual = x
        x          = self.pwconv1(x)

        x          = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
        x.extend(m(x[-1]) for m in self.m)
        x[0]       = x[0] +  x[1] 
        x.pop(1)
        
        y          = torch.cat(x, dim=1) 
        y          = self.pwconv2(y)
        return x_residual + y if self.add else y


'''
U-shape/U-like Model
'''
# Encoder in TinyResUNet
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.cmrf       = CMRF(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        # Residual projection if in/out channels differ
        self.res_proj   = nn.Identity()
        if in_channels != out_channels:
            self.res_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.res_proj(x)
        x = self.cmrf(x)
        x = x + residual  # residual connection
        return self.downsample(x), x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.cmrf     = CMRF(in_channels, out_channels)
        self.upsample = F.interpolate

        # Residual projection for decoder 
        self.res_proj = nn.Identity()
        if in_channels != out_channels:
            self.res_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, skip_connection):
        x = self.upsample(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = torch.cat([x, skip_connection], dim=1)
        residual = self.res_proj(x)
        x = self.cmrf(x)
        x = x + residual  # residual connection
        return x


# TinyU-Net
class TinyResUNet(nn.Module):
    """TinyU-Net with args(in_channels, num_classes)."""
    '''
    in_channels: The number of input channels
    num_classes: The number of segmentation classes
    '''
    def __init__(self, in_ch=3, out_ch=1, ndim=2, chs: tuple[int, ...] = (32, 64, 128, 256, 512)):
        super(TinyResUNet, self).__init__()
        # in_filters      = [192, 384, 768, 1024]
        # out_filters     = [64, 128, 256, 512]
        in_filters      = [96, 192, 384, 512]
        out_filters     = [32, 64, 128, 256]
        self.encoder1   = EncoderBlock(in_ch, chs[0])
        self.encoder2   = EncoderBlock(chs[0], chs[1])
        self.encoder3   = EncoderBlock(chs[1], chs[2])
        self.encoder4   = EncoderBlock(chs[2], chs[3])

        self.enh_decoder4   = DecoderBlock(in_filters[3], out_filters[3])
        self.enh_decoder3   = DecoderBlock(in_filters[2], out_filters[2])
        self.enh_decoder2   = DecoderBlock(in_filters[1], out_filters[1])
        self.enh_decoder1   = DecoderBlock(in_filters[0], out_filters[0])
        self.enh_head = nn.Conv2d(out_filters[0], out_ch, kernel_size=1)

        self.mnt_decoder4   = DecoderBlock(in_filters[3], out_filters[3])
        self.mnt_decoder3   = DecoderBlock(in_filters[2], out_filters[2])
        self.mnt_decoder2   = DecoderBlock(in_filters[1], out_filters[1])
        self.mnt_decoder1   = DecoderBlock(in_filters[0], out_filters[0])
        self.mnt_head = nn.Sequential(
            nn.Conv2d(out_filters[0], 12, kernel_size=1),
            nn.Tanh()
        )


        
    def forward(self, x):
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)

        x_enh        = self.enh_decoder4(x, skip4)
        x_enh        = self.enh_decoder3(x_enh, skip3)
        x_enh        = self.enh_decoder2(x_enh, skip2)
        x_enh        = self.enh_decoder1(x_enh, skip1)
        enh        = self.enh_head(x_enh)


        x_mnt        = self.mnt_decoder4(x, skip4)
        x_mnt        = self.mnt_decoder3(x_mnt, skip3)
        x_mnt        = self.mnt_decoder2(x_mnt, skip2)
        x_mnt        = self.mnt_decoder1(x_mnt, skip1)
        mnt        = self.mnt_head(x_mnt)

        return enh, mnt


if __name__ == '__main__':
    model         = TinyResUNet(in_ch=1)

    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model         = model.to(device)

    summary(model, (1, 128, 128))
        
    dummy_input   = torch.randn(1, 1, 128, 128).to(device)
    flops, params = profile(model, (dummy_input, ), verbose=False)
    #-------------------------------------------------------------------------------#
    #   flops * 2 because profile does not consider convolution as two operations.
    #-------------------------------------------------------------------------------#
    flops         = flops * 2
    flops, params = clever_format([flops, params], "%.4f")
    print(f'Total GFLOPs: {flops}')
    print(f'Total Params: {params}')