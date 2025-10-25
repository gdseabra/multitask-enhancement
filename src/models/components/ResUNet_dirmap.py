
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from thop import clever_format, profile
from torchsummary import summary

def conv_nd(
    in_channels: int, out_channels: int, kernel_size: int, ndim: int, **kwargs: Any
) -> nn.Module:
    """Convolutional layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Kernel size. Must be odd.
    ndim : int
        Number of dimensions. Must be 2 or 3.
    kwargs : Any
        Additional arguments.

    Returns
    -------
    nn.Module
        Convolutional layer.
    """
    if ndim == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
    elif ndim == 3:
        return nn.Conv3d(in_channels, out_channels, kernel_size, **kwargs)
    else:
        raise ValueError("ndim must be 2 or 3")


def upsample_nd(scale_factor: float, ndim: int, **kwargs: Any) -> nn.Module:
    """Upsampling layer.

    Parameters
    ----------
    scale_factor : float
        Scale factor.
    ndim : int
        Number of dimensions. Must be 2 or 3.
    kwargs : Any
        Additional arguments.

    Returns
    -------
    nn.Module
        Upsampling layer.
    """
    if ndim == 2:
        return nn.Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=False, **kwargs
        )
    elif ndim == 3:
        return nn.Upsample(
            scale_factor=scale_factor, mode="trilinear", align_corners=False, **kwargs
        )
    else:
        raise ValueError("ndim must be 2 or 3")


def maxpool_nd(ndim: int, **kwargs: Any) -> nn.Module:
    """Max pooling layer.

    Parameters
    ----------
    ndim : int
        Number of dimensions. Must be 2 or 3.
    kwargs : Any
        Additional arguments.

    Returns
    -------
    nn.Module
        Max pooling layer.
    """
    if ndim == 2:
        return nn.MaxPool2d(**kwargs)
    elif ndim == 3:
        return nn.MaxPool3d(**kwargs)
    else:
        raise ValueError("ndim must be 2 or 3")

class Block(nn.Module):
    """UNet block.

    It is a sequence of a convolutional layer, a group normalization layer, and a SiLU
    activation function.
    """

    def __init__(self, dim: int, dim_out: int, groups: int = 8) -> None:
        """Initialize the ResUNet block.

        Parameters
        ----------
        dim : int
            Input dimension.
        dim_out : int
            Output dimension.
        groups : int, optional
            Number of groups for group normalization, by default 8.

        """
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UNet block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.

        """
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """ResNet block.

    It is a sequence of two UNet blocks and a residual convolutional layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        groups: int = 8,
    ) -> None:
        """Initialize the ResnetBlock module.

        Parameters
        ----------
        dim : int
            Input dimension.
        dim_out : int
            Output dimension.
        groups : int, optional
            Number of groups for group normalization, by default 8.

        """
        super().__init__()

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        self.res_conv = (
            nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the ResnetBlock module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.

        """

        h = self.block1(x)

        h = self.block2(h)

        return h + self.res_conv(x)


class Encoder(nn.Module):
    """UNet encoder.

    It consists of a series of UNet blocks and max pooling layers.
    """

    def __init__(self, chs: list[int], ndim: int):
        """Initialize the UNet encoder.

        Parameters
        ----------
        chs : list[int]
            List with the number of channels in each layer. The second element and onwards
            double the number of channels in the previous layer.
        ndim : int
            Number of dimensions. Must be 2 or 3.
        """
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [ResnetBlock(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.pool = maxpool_nd(ndim, kernel_size=2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass of the UNet encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        list[torch.Tensor]
            List with the features of each layer.
        """
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    """UNet decoder.

    It consists of a series of upconvolutions and UNet blocks.
    """

    def __init__(self, chs: list[int], ndim: int):
        """Initialize the UNet decoder.

        Parameters
        ----------
        chs : list[int]
            List with the number of channels in each layer. The second element and onwards
            halve the number of channels in the previous layer.
        ndim : int
            Number of dimensions. Must be 2 or 3.
        """
        super().__init__()
        self.upconvs = nn.ModuleList(
            [
                nn.Sequential(
                    upsample_nd(2, ndim), 
                    conv_nd(chs[i], chs[i + 1], 1, ndim)
                )
                for i in range(len(chs) - 1)
            ]
        )
        self.dec_blocks = nn.ModuleList(
            [ResnetBlock(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )

    def forward(self, x: torch.Tensor, encoder_features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass of the UNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        encoder_features : list[torch.Tensor]
            List with the features of each layer of the encoder.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        # Iterate using indices
        for i, upconv in enumerate(self.upconvs):
            for j, dec_block in enumerate(self.dec_blocks):
                if i == j:
                    enc_ftrs = encoder_features[i]
                    x = upconv(x)
                    # Ensure that x and enc_ftrs are of the same size for concatenation
                    if x.shape != enc_ftrs.shape:
                        x = F.interpolate(x, size=enc_ftrs.shape[2:], mode="nearest")
                    x = torch.cat([x, enc_ftrs], dim=1)
                    x = dec_block(x)
        return x


class ResUNet_dirmap(nn.Module):
    """UNet model.

    It follows the architecture proposed in [1]_. Some modifications were made to the
    original architecture. One of the main differences is that we replaced the transposed
    convolutions with upsampling followed by convolutions. This change was made to avoid
    checkerboard artifacts [2]_.

    References
    ----------
    .. [1] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional \
           networks for biomedical image segmentation." Medical image computing and \
           computer-assisted intervention–MICCAI 2015: 18th international conference, \
           Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer \
           International Publishing, 2015.
    .. [2] Odena, Augustus, Vincent Dumoulin, and Chris Olah. "Deconvolution and \
           checkerboard artifacts." Distill 1.10 (2016): e3.
    """

    def __init__(
        self,
        in_ch: int = 3,
        ndim: int = 2,
        chs: tuple[int, ...] = (32, 64, 128, 256, 512),
        out_ch: int = 1,
    ):
        """Initialize the UNet model.

        Parameters
        ----------
        in_ch : int
            Number of input channels. Defaults to 3.
        ndim : int
            Number of dimensions. Must be 2 or 3. Defaults to 2.
        chs : tuple[int, ...]
            Number of channels in each layer. Defaults to (64, 128, 256, 512, 1024). It
            must have at least two elements and the from the second element onwards, the
            number of channels must double the previous one.
        out_ch : int
            Number of output channels. Defaults to 1.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        super().__init__()
        enc_chs = [in_ch] + list(chs)
        dec_chs = list(reversed(chs)) + [out_ch]
        self.encoder = Encoder(enc_chs, ndim)
        self.decoder = Decoder(dec_chs[:-1], ndim)
        self.head = conv_nd(dec_chs[-2], out_ch, 1, ndim, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        z = self.encoder(x)
        z = self.decoder(z[-1], z[:-1][::-1])
        z = self.head(z)
        z = F.interpolate(z, scale_factor=1/4, mode="bilinear")
        x_downsampled = F.interpolate(x, scale_factor=1/8, mode="bilinear")
        z = z + x_downsampled

        return z


if __name__ == '__main__':
    model         = ResUNet_dirmap(in_ch=1)

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