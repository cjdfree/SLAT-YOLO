# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Coordinate attention and token statistics self-attention for SLAT-YOLO."""

import torch
from einops import rearrange
from torch import nn


class h_sigmoid(nn.Module):
    """h_sigmoid implementation retained from the track slab experiments."""

    def __init__(self, inplace=True):
        """Initialize layers and parameters."""
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        """Apply the feature transformation."""
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """h_swish implementation retained from the track slab experiments."""

    def __init__(self, inplace=True):
        """Initialize layers and parameters."""
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        """Apply the feature transformation."""
        return x * self.sigmoid(x)


class CA(nn.Module):
    """CA implementation retained from the track slab experiments."""

    def __init__(self, inp, reduction=32):
        """Initialize layers and parameters."""
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """Apply the feature transformation."""
        identity = x
        h, w = x.shape[-2:]
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


class TSSA(nn.Module):
    """TSSA implementation retained from the track slab experiments."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, **kwargs):
        """Initialize layers and parameters."""
        super().__init__()
        self.heads = num_heads
        self.attend = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.temp = nn.Parameter(torch.ones(num_heads, 1))
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(proj_drop))

    def forward(self, x):
        """Apply the feature transformation."""
        w = rearrange(self.qkv(x), "b n (h d) -> b h n d", h=self.heads)
        w_normed = torch.nn.functional.normalize(w, dim=-2)
        w_sq = w_normed**2
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp)
        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-08)).unsqueeze(-2), w**2)
        attn = 1.0 / (1 + dots)
        attn = self.attn_drop(attn)
        out = -torch.mul(w.mul(Pi.unsqueeze(-1)), attn)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# Legacy pickle names: old Releases resolve to the paper-named implementations.
CoordAtt = CA
AttentionTSSA = TSSA
