# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train, validate, or predict with the curated SLAT-YOLO models."""

import argparse
from pathlib import Path

import yaml

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Resolve project paths and dispatch to the standard Ultralytics Python API."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "val", "predict"))
    parser.add_argument("--model", default="my_project/yaml/slat-yolo.yaml")
    parser.add_argument("--data", default="my_project/datasets/cracks.yaml")
    parser.add_argument("--source", help="Image, directory, or video for prediction.")
    parser.add_argument("--device", default="cpu", help="Use 0 for the first CUDA GPU.")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="slat-yolo")
    args = parser.parse_args()
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if args.mode != "train" and model_path.suffix.lower() != ".pt":
        parser.error("Validation and prediction require a trained .pt checkpoint, supplied with --model.")
    model = YOLO(str(model_path), task="detect")
    common = {"device": args.device, "imgsz": args.imgsz, "project": str(ROOT / "my_project/outputs"), "name": args.name}
    if args.mode == "predict":
        if args.source is None:
            parser.error("Prediction requires --source.")
        model.predict(source=args.source, save=True, **common)
        return
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    dataset_root = Path(data["path"])
    if not dataset_root.is_absolute():
        dataset_root = (ROOT / dataset_root).resolve()
    if not dataset_root.is_dir():
        parser.error("Dataset is missing. Run python -m my_project.download_assets datasets first.")
    data["path"] = str(dataset_root)
    resolved = ROOT / "my_project/outputs/resolved-data.yaml"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    if args.mode == "train":
        # These values come from archived args.yaml; optimizer=auto is recorded, not inferred as AdamW.
        model.train(
            data=str(resolved),
            epochs=args.epochs,
            batch=args.batch,
            workers=args.workers,
            patience=0,
            pretrained=False,
            amp=False,
            optimizer="auto",
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            seed=0,
            deterministic=True,
            **common,
        )
    else:
        model.val(data=str(resolved), batch=args.batch, workers=args.workers, split="val", **common)


if __name__ == "__main__":
    main()
