import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile, clever_format

# ------------------- Basic Blocks (unchanged) -------------------

def conv_nd(in_channels, out_channels, kernel_size, ndim, **kwargs):
    if ndim == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
    elif ndim == 3:
        return nn.Conv3d(in_channels, out_channels, kernel_size, **kwargs)
    else:
        raise ValueError("ndim must be 2 or 3")

def maxpool_nd(ndim: int, **kwargs) -> nn.Module:
    if ndim == 2:
        return nn.MaxPool2d(**kwargs)
    elif ndim == 3:
        return nn.MaxPool3d(**kwargs)
    else:
        raise ValueError("ndim must be 2 or 3")

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups)
        self.block2 = Block(dim_out, dim_out, groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        return self.block2(self.block1(x)) + self.res_conv(x)

# ------------------- Encoder-Only Network -------------------

class DirmapEncoder(nn.Module):
    """
    Encoder that downsamples by 8× and outputs 90 channels.
    """

    def __init__(self, in_ch=1, ndim=2, chs=(64, 128, 256), out_ch=90):
        """
        Parameters
        ----------
        in_ch : int
            Number of input channels.
        ndim : int
            Number of dimensions (2 or 3).
        chs : tuple[int]
            Channel sizes of encoder stages.
        out_ch : int
            Number of output channels (default 90).
        """
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.pool = maxpool_nd(ndim, kernel_size=2)

        # Build encoder
        enc_chs = [in_ch] + list(chs)
        for i in range(len(enc_chs) - 1):
            self.encoder_blocks.append(ResnetBlock(enc_chs[i], enc_chs[i + 1]))

        # Final conv to 90 channels
        self.head = conv_nd(enc_chs[-1], out_ch, 1, ndim)

    def forward(self, x):
        # 3 downsamples → output is W/8 × H/8
        for block in self.encoder_blocks:
            x = block(x)
            x = self.pool(x)
        x = self.head(x)
        return x

# ------------------- Test the model -------------------

if __name__ == '__main__':
    model = DirmapEncoder(in_ch=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    summary(model, (1, 512, 512))

    dummy_input = torch.randn(1, 1, 512, 512).to(device)
    flops, params = profile(model, (dummy_input,), verbose=False)
    flops = flops * 2  # count MACs as 2 ops
    flops, params = clever_format([flops, params], "%.4f")
    print(f"Total GFLOPs: {flops}")
    print(f"Total Params: {params}")

    out = model(dummy_input)
    print("Output shape:", out.shape)  # expected: [1, 90, 16, 16] for 128×128 input
