# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""C2-STR with Dynamic Tanh, TSSA, Mona adaptation, and spatial gating."""

import torch
from torch import nn

from ultralytics.nn.modules.block import C2PSA, PSABlock

from .attention import TSSA
from .mona import Mona
from .semnet import SEFN
from .transMamba import SpectralEnhancedFFN


class DynamicTanh(nn.Module):
    """DynamicTanh implementation retained from the track slab experiments."""

    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        """Initialize layers and parameters."""
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        """Apply the feature transformation."""
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        """Return the module configuration."""
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


class STRBlock(PSABlock):
    """STRBlock implementation retained from the track slab experiments."""

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initialize layers and parameters."""
        super().__init__(c, attn_ratio, num_heads, shortcut)
        self.ffn = SEFN(c, ffn_expansion_factor=2, bias=False)
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.mona1 = Mona(c)
        self.mona2 = Mona(c)
        self.attn = TSSA(c, num_heads=num_heads)

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x_spatial = x
        _, C, H, W = x.size()
        x = (
            x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
            if self.add
            else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        )
        x = self.mona1(x)
        x = x + self.ffn(self.dyt2(x), x_spatial) if self.add else self.ffn(self.dyt2(x), x_spatial)
        x = self.mona2(x)
        return x


class C2STR(C2PSA):
    """C2STR implementation retained from the track slab experiments."""

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initialize layers and parameters."""
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(
            *(STRBlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))
        )


class STRBlockSpectral(PSABlock):
    """Historical spectral implementation recovered from the archived YOLO11 project."""

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initialize the spectral transformation."""
        super().__init__(c, attn_ratio, num_heads, shortcut)
        self.ffn = SpectralEnhancedFFN(c, ffn_expansion_factor=2, bias=False)
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.mona1 = Mona(c)
        self.mona2 = Mona(c)
        self.attn = TSSA(c, num_heads=num_heads)

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        _, C, H, W = x.size()
        x = (
            x + self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
            if self.add
            else self.attn(self.dyt1(x).flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        )
        x = self.mona1(x)
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        x = self.mona2(x)
        return x


class C2STRSpectral(C2PSA):
    """Historical spectral implementation recovered from the archived YOLO11 project."""

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initialize the spectral transformation."""
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(
            *(STRBlockSpectral(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))
        )


# Legacy pickle names: old Releases resolve to the paper-named implementations.
C2TSSA_DYT_Mona_SEFFN = C2STRSpectral
C2TSSA_DYT_Mona_SEFN = C2STR
TSSAlock_DYT_Mona_SEFFN = STRBlockSpectral
TSSAlock_DYT_Mona_SEFN = STRBlock
