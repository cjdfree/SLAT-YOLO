# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# ruff: noqa: N999
# Keep the historical module path so archived checkpoints remain loadable.
"""Spectral FFN used in the recovered experimental checkpoints; no Mamba dependency."""

import torch
from torch import nn
from torch.nn import functional as F


class SpectralEnhancedFFN(nn.Module):
    """Historical spectral implementation recovered from the archived YOLO11 project."""

    def __init__(self, dim, ffn_expansion_factor, bias):
        """Initialize the spectral transformation."""
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=2,
            groups=hidden_features * 2,
            bias=bias,
            dilation=2,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.fft_channel_weight = nn.Parameter(torch.randn((1, hidden_features * 2, 1, 1)))
        self.fft_channel_bias = nn.Parameter(torch.randn((1, hidden_features * 2, 1, 1)))

    def pad(self, x, factor):
        """Pad width to a multiple of the FFT factor."""
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, "constant", 0)
        return (x, t_pad)

    def unpad(self, x, t_pad):
        """Restore the input width after the inverse FFT."""
        hw = x.shape[-1]
        return x[..., t_pad[0] : hw - t_pad[1]]

    def forward(self, x):
        """Refine features with dilated filtering and spectral modulation."""
        x_dtype = x.dtype
        x = self.project_in(x)
        x = self.dwconv(x)
        x, pad_w = self.pad(x, 2)
        x = torch.fft.rfft2(x.float())
        x = self.fft_channel_weight * x + self.fft_channel_bias
        x = torch.fft.irfft2(x)
        x = self.unpad(x, pad_w)
        x1, x2 = x.chunk(2, dim=1)
        x = F.silu(x1) * x2
        x = self.project_out(x.to(x_dtype))
        return x
