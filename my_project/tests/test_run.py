# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Check the public runner's private-data interface and training defaults."""

import sys

import pytest

from my_project import run


@pytest.mark.parametrize("mode", ("train", "val"))
def test_data_required_before_model_loading(monkeypatch, mode):
    """Fail early without downloading or building a model when no dataset YAML is supplied."""
    monkeypatch.setattr(sys, "argv", ["run", mode])

    def unexpected_model(*args, **kwargs):
        pytest.fail("Missing --data must be rejected before model loading")

    monkeypatch.setattr(run, "YOLO", unexpected_model)
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2


def test_train_uses_user_yaml_and_1024_default(monkeypatch, tmp_path):
    """Pass the user's YAML to Ultralytics unchanged and use the documented image size."""
    config = tmp_path / "private.yaml"
    content = "path: /private/images\ntrain: train\nval: val\nnames: [crack]\n"
    config.write_text(content, encoding="utf-8")
    captured = {}

    class Model:
        def train(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(run, "YOLO", lambda *args, **kwargs: Model())
    monkeypatch.setattr(sys, "argv", ["run", "train", "--data", str(config)])
    run.main()
    assert captured["data"] == str(config.resolve())
    assert captured["imgsz"] == 1024
    assert captured["epochs"] == 1200 and captured["batch"] == 16
    assert config.read_text(encoding="utf-8") == content


def test_predict_does_not_require_data(monkeypatch):
    """Keep prediction available without a dataset configuration."""
    captured = {}

    class Model:
        def predict(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(run, "YOLO", lambda *args, **kwargs: Model())
    monkeypatch.setattr(sys, "argv", ["run", "predict", "--model", "own.pt", "--source", "images"])
    run.main()
    assert captured["source"] == "images" and captured["imgsz"] == 1024
