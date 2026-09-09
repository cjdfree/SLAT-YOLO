# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""MPCR detection head using the current Ultralytics training and inference interface."""

from torch import nn

from ultralytics.nn.modules import Conv, Detect, DWConv

from .block import MPCR


class MPCRHead(Detect):
    """Recalibrate box and class features with independent multi-patch context units."""

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        """Build the experimental prediction branches on the upstream Detect interface."""
        if end2end:
            raise ValueError("MPCR-Head uses YOLO11 one-to-many detection; set end2end=False.")
        super().__init__(nc, reg_max, end2end, ch)
        c2, c3 = max(16, ch[0] // 4, reg_max * 4), max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(c, c2, 3), MPCR(c2, c2, 1), nn.Conv2d(c2, 4 * reg_max, 1)) for c in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(c, c, 3), Conv(c, c3, 1)),
                MPCR(c3, c3, 1),
                nn.Conv2d(c3, nc, 1),
            )
            for c in ch
        )


# Legacy pickle names: old Releases resolve to the paper-named implementations.
Detect_MultiSEAM = MPCRHead
