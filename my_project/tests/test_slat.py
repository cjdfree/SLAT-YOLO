# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Regression coverage for model integration, spectral and spatial paths, and regression loss."""

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.slat_loss import WiseShapeIoULoss

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(2)


@pytest.mark.parametrize("config", sorted((ROOT / "my_project/yaml").glob("*.yaml")), ids=lambda p: p.stem)
def test_config_forward_backward(config):
    """Build every real architecture, check decoded boxes, and backpropagate through its head."""
    model = YOLO(str(config), task="detect", verbose=False).model
    image = torch.randn(2, 3, 256, 256)
    model.eval()
    with torch.no_grad():
        prediction = model(image)[0]
    assert prediction.shape == (2, 5, 1344)
    assert torch.isfinite(prediction).all()
    model.train()
    raw = model(image)
    (raw["boxes"].square().mean() + raw["scores"].square().mean()).backward()
    assert torch.isfinite(model.model[0].conv.weight.grad).all()


def test_wise_shape_geometry_and_running_mean():
    """Check exact matches, disjoint boxes, finite gradients, and inference mean preservation."""
    loss = WiseShapeIoULoss()
    target = torch.tensor([[0.0, 0.0, 2.0, 20.0], [10.0, 10.0, 12.0, 30.0]])
    pred = torch.tensor([[0.0, 0.0, 2.0, 20.0], [20.0, 20.0, 23.0, 40.0]], requires_grad=True)
    value = loss(pred, target)
    assert value.shape == (2,)
    assert value[0].item() == 0.0 and value[1] > 0
    assert loss.iou_mean < 1.0
    value.sum().backward()
    assert torch.isfinite(pred.grad).all()
    before = loss.iou_mean.clone()
    with torch.no_grad():
        loss(pred, target)
    torch.testing.assert_close(before, loss.iou_mean)


def test_loss_selection_and_training_step():
    """Exercise actual detector assignment and loss while checking upstream CIoU stays the default."""
    for name, custom in [("yolo11n.yaml", False), ("slat-yolo.yaml", True)]:
        wrapper = YOLO(str(ROOT / "my_project/yaml" / name), task="detect", verbose=False)
        model = wrapper.model.train()
        model.args = get_cfg(overrides=model.args)
        batch = {
            "img": torch.rand(2, 3, 256, 256),
            "batch_idx": torch.tensor([0.0, 1.0]),
            "cls": torch.zeros(2, 1),
            "bboxes": torch.tensor([[0.5, 0.5, 0.05, 0.7], [0.5, 0.5, 0.1, 0.5]]),
        }
        value, _ = model(batch)
        assert isinstance(model.criterion.bbox_loss.iou_loss, WiseShapeIoULoss) == custom
        assert torch.isfinite(value).all()
        value.sum().backward()
        assert torch.isfinite(model.model[0].conv.weight.grad).all()
