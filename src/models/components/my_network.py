""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class InitBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitBlock, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.apply(weights_init)

    def forward(self, _x):
        return self.double_conv(_x) + self.res_conv(_x)


class DownBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(DownBlock, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )

        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        )


        self.apply(weights_init)

    def forward(self, _x):
        h = self.double_conv(_x)
        res = self.res_conv(_x)
        return h + res


class BottleBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(BottleBlock, self).__init__()

        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )
        self.bottle_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )

        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )

        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

        self.apply(weights_init)

    def forward(self, _x):
        res = self.res_conv(_x)

        h = self.down_conv(_x) 
        h = self.bottle_conv(h)
        h = self.up_conv(h)
        return h + res


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()

        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )

        self.res_conv = (
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

        self.apply(weights_init)

    def forward(self, x_current, x_previous):
        out_res = self.res_conv(x_current)

        _x_current = self.up_conv1(x_current)
        _x = torch.cat([x_previous, _x_current], dim=1)

        out_conv2 = self.up_conv2(_x)

        # out_res = F.interpolate(out_res, size=out_conv2.shape[-2:], mode='bilinear', align_corners=False)

        _x = out_conv2 + out_res

        return _x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


class FingerGAN(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        ndim: int = 2,
        chs: tuple[int, ...] = (64, 128, 256, 512, 1024),
        out_ch: int = 1
    ):
        """Initialize a `FingerGAN` module.

        :param input_channels: The number of input channels.
        :param lin1_size: The number of output features of the first downblock.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param lin4_size: The number of output features of the third linear layer.
        :param lin5_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        # 64,128,256,512,1024
        super(FingerGAN, self).__init__()

        # x = 192 x 192
        self.init_block = InitBlock(in_ch, chs[0]) # 190 x 190, 188 x 188

        self.down1 = DownBlock(chs[0], chs[1])     # 94 x 94,   92 x 92
        self.down2 = DownBlock(chs[1], chs[2])    # 46 x 46,   44 x 44
        self.down3 = DownBlock(chs[2], chs[3])    # 22 x 22,   20 x 20

        self.bottle = BottleBlock(chs[3], chs[4])   # 10 x 10,   8  x 8,  10 x 10

        self.up1 = UpBlock(chs[4], chs[3])       # 20 x 20, 22 x 22
        self.up2 = UpBlock(chs[3], chs[2])        # 44 x 44, 46 x 46
        self.up3 = UpBlock(chs[2], chs[1])        # 92 x 92, 94 x 94
        self.up4 = UpBlock(chs[1], chs[0])         # 188 x 188, 190 x 190

        self.out_block = OutConv(chs[0], in_ch)     # 192 x 192

    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        x1 = self.init_block(_x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.bottle(x4)

        _x = self.up1(x5, x4)
        _x = self.up2(_x, x3)
        _x = self.up3(_x, x2)
        _x = self.up4(_x, x1)
        logits_sigmoid = self.out_block(_x)
        return logits_sigmoid


if __name__ == "__main__":
    # x = torch.randn(1, 2, 192, 192)
    _ = FingerGAN(2)

    # fD = Discriminator(2)

    # x_ = fG(x)
    # y = fD(x_)
    pass

