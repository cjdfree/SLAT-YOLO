# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Slim-neck aggregation and multi-patch channel recalibration."""

import torch
from torch import nn

from ultralytics.nn.modules import Conv


class GSConv(nn.Module):
    """GSConv implementation retained from the track slab experiments."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize layers and parameters."""
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, Conv.default_act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, Conv.default_act)

    def forward(self, x):
        """Apply the feature transformation."""
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        return torch.cat((y[0], y[1]), 1)


class GSBottleneck(nn.Module):
    """GSBottleneck implementation retained from the track slab experiments."""

    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        """Initialize layers and parameters."""
        super().__init__()
        c_ = int(c2 * e)
        self.conv_lighting = nn.Sequential(GSConv(c1, c_, 1, 1), GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        """Apply the feature transformation."""
        return self.conv_lighting(x) + self.shortcut(x)


class VoVGSCSP(nn.Module):
    """VoVGSCSP implementation retained from the track slab experiments."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize layers and parameters."""
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        """Apply the feature transformation."""
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))


class Residual(nn.Module):
    """Residual implementation retained from the track slab experiments."""

    def __init__(self, fn):
        """Initialize layers and parameters."""
        super().__init__()
        self.fn = fn

    def forward(self, x):
        """Apply the feature transformation."""
        return self.fn(x) + x


def DcovN(c1, c2, depth, kernel_size=3, patch_size=3):
    """DcovN implementation retained from the track slab experiments."""
    dcovn = nn.Sequential(
        nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_size),
        nn.SiLU(),
        nn.BatchNorm2d(c2),
        *[
            nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=c2, out_channels=c2, kernel_size=kernel_size, stride=1, padding=1, groups=c2
                        ),
                        nn.SiLU(),
                        nn.BatchNorm2d(c2),
                    )
                ),
                nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=1, stride=1, padding=0, groups=1),
                nn.SiLU(),
                nn.BatchNorm2d(c2),
            )
            for i in range(depth)
        ],
    )
    return dcovn


class MPCR(nn.Module):
    """MPCR implementation retained from the track slab experiments."""

    def __init__(self, c1, c2, depth, kernel_size=3, patch_size=(3, 5, 7), reduction=16):
        """Initialize layers and parameters."""
        super().__init__()
        if c1 != c2:
            c2 = c1
        self.DCovN0 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[0])
        self.DCovN1 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[1])
        self.DCovN2 = DcovN(c1, c2, depth, kernel_size=kernel_size, patch_size=patch_size[2])
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c2, c2 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c2 // reduction, c2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Apply the feature transformation."""
        b, c, _, _ = x.size()
        y0 = self.DCovN0(x)
        y1 = self.DCovN1(x)
        y2 = self.DCovN2(x)
        y0 = self.avg_pool(y0).view(b, c)
        y1 = self.avg_pool(y1).view(b, c)
        y2 = self.avg_pool(y2).view(b, c)
        y4 = self.avg_pool(x).view(b, c)
        y = (y0 + y1 + y2 + y4) / 4
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.exp(y)
        return x * y.expand_as(x)


# Legacy pickle names: old Releases resolve to the paper-named implementations.
MultiSEAM = MPCR
