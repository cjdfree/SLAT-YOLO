# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train, validate, or predict with the curated SLAT-YOLO models."""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Resolve project paths and dispatch to the standard Ultralytics Python API."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "val", "predict"))
    parser.add_argument("--model", default="my_project/yaml/slat-yolo.yaml")
    parser.add_argument("--data", help="Path to your own YOLO dataset YAML; required for train and val.")
    parser.add_argument("--source", help="Image, directory, or video for prediction.")
    parser.add_argument("--device", default="cpu", help="Use 0 for the first CUDA GPU.")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="slat-yolo")
    args = parser.parse_args()
    if args.mode in {"train", "val"} and not args.data:
        parser.error("Training and validation require --data pointing to your own YOLO dataset YAML.")
    if args.mode == "predict" and not args.source:
        parser.error("Prediction requires --source.")
    data_path = Path(args.data).expanduser().resolve() if args.data else None
    if data_path is not None and not data_path.is_file():
        parser.error(f"Dataset YAML not found: {data_path}")
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if args.mode != "train" and model_path.suffix.lower() != ".pt":
        parser.error("Validation and prediction require a trained .pt checkpoint, supplied with --model.")
    model = YOLO(str(model_path), task="detect")
    common = {"device": args.device, "imgsz": args.imgsz, "project": str(ROOT / "my_project/outputs"), "name": args.name}
    if args.mode == "predict":
        model.predict(source=args.source, save=True, **common)
        return
    if args.mode == "train":
        # Project training defaults: 1024-pixel input; remaining unspecified options use Ultralytics defaults.
        model.train(
            data=str(data_path),
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
        model.val(data=str(data_path), batch=args.batch, workers=args.workers, split="val", **common)


if __name__ == "__main__":
    main()
