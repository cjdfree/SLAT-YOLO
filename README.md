# SLAT-YOLO

[English](README.md) | [简体中文](README.zh-CN.md)

SLAT-YOLO is a YOLO11-based detector for track slab cracks. It combines attention-enhanced backbone features,
token refinement, lightweight feature fusion and context-aware detection. This repository provides model code,
YAML configurations, training scripts, experimental curves and baseline/ablation checkpoints, built on Ultralytics 8.4.143.

Associated paper: **SLAT-YOLO: Shape-aware attention and spatially enhanced token refinement for robust track slab crack detection**.

## Modules

| Module | Role | Implementation |
| --- | --- | --- |
| C3k2-CA | Adds coordinate attention to backbone features to retain directional and positional information. | [C3k2](ultralytics/nn/modules/block.py), [CA](ultralytics/nn/extra_modules/attention.py) |
| C2-STR | Refines feature tokens through token statistics attention, Dynamic Tanh, Mona adaptation and gated feed-forward processing. | [transformer.py](ultralytics/nn/extra_modules/transformer.py) |
| VoV-GSCSP / GSConv | Combines lightweight convolution and cross-stage feature fusion in the neck. | [block.py](ultralytics/nn/extra_modules/block.py) |
| MPCR-Head | Recalibrates box and classification features using context from multiple patch scales. | [head.py](ultralytics/nn/extra_modules/head.py) |
| Wise-ShapeIoU | Combines shape-aware box regression with adaptive loss weighting. | [slat_loss.py](ultralytics/utils/slat_loss.py) |

Model and ablation configurations are in [my_project/yaml](my_project/yaml). The default entry is
[slat-yolo.yaml](my_project/yaml/slat-yolo.yaml).

## Installation and training

Use Python 3.10+ and a PyTorch installation appropriate for your CPU/CUDA environment.

```bash
git clone https://github.com/cjdfree/SLAT-YOLO.git
cd SLAT-YOLO
pip install -e .
```

The research dataset is not distributed. Supply your own YOLO dataset YAML using `--data`.

```bash
python -m my_project.run train --data /path/to/your/data.yaml --device 0 --imgsz 1024

# YOLO11n baseline
python -m my_project.run train --model my_project/yaml/yolo11n.yaml --data /path/to/your/data.yaml --device 0 --name baseline
```

The [training script](my_project/run.py) sets the following project defaults; unspecified options use Ultralytics defaults.
The image input size is **1024**. Outputs are saved under `my_project/outputs/`.

| Parameter | Default |
| --- | --- |
| `imgsz` | **1024** |
| `epochs` / `batch` | 1200 / 16 |
| `optimizer` / `lr0` / `lrf` | auto / 0.01 / 0.01 |
| `momentum` / `weight_decay` | 0.937 / 0.0005 |
| `pretrained` / `amp` | False / False |
| `seed` / `deterministic` | 0 / True |
| `patience` / `workers` | 0 / 4 |

## Experimental results

Values below are percentages from the epoch with the highest recorded validation AP50 in each training CSV;
precision, recall and AP50–95 come from the same epoch. These are archived validation results, not independent test scores.
The historical split contains augmented images derived from shared originals across training and validation.

### Ablation study

Rows denote additions to YOLO11n; SLAT-YOLO denotes the full configuration.

| Configuration | Precision (%) | Recall (%) | AP50 (%) | AP50–95 (%) |
| --- | ---: | ---: | ---: | ---: |
| [C2-STR](my_project/results/ablation/yolo11n-C2-STR) | 94.31 | 88.09 | 93.78 | 68.48 |
| [C3k2-CA](my_project/results/ablation/yolo11n-C3k2-CA) | 95.06 | 83.62 | 92.81 | 72.93 |
| [C3k2-CA-C2-STR](my_project/results/ablation/yolo11n-C3k2-CA-C2-STR) | 97.17 | 85.11 | 94.71 | 69.50 |
| [C3k2-CA-C2-STR-VoV-GSCSP](my_project/results/ablation/yolo11n-C3k2-CA-C2-STR-VoV-GSCSP) | 95.03 | 89.58 | 95.75 | 78.69 |
| [C3k2-CA-C2-STR-VoV-GSCSP-MPCR-Head](my_project/results/ablation/yolo11n-C3k2-CA-C2-STR-VoV-GSCSP-MPCR-Head) | 96.79 | 89.57 | 96.10 | 78.12 |
| [SLAT-YOLO](my_project/results/ablation/slat-yolo) | 97.52 | 91.98 | 96.62 | 79.62 |
| [C3k2-CA-MPCR-Head](my_project/results/ablation/yolo11n-C3k2-CA-MPCR-Head) | 95.69 | 87.45 | 94.47 | 73.95 |
| [C3k2-CA-VoV-GSCSP](my_project/results/ablation/yolo11n-C3k2-CA-VoV-GSCSP) | 95.21 | 89.79 | 95.38 | 71.43 |
| [C3k2-CA-VoV-GSCSP-MPCR-Head](my_project/results/ablation/yolo11n-C3k2-CA-VoV-GSCSP-MPCR-Head) | 94.25 | 91.70 | 96.87 | 79.25 |
| [C3k2-CA-VoV-GSCSP-MPCR-Head-CIoU](my_project/results/ablation/yolo11n-C3k2-CA-VoV-GSCSP-MPCR-Head-CIoU) | 97.50 | 91.28 | 95.84 | 77.74 |
| [C3k2-CA-VoV-GSCSP-MPCR-Head-GIoU](my_project/results/ablation/yolo11n-C3k2-CA-VoV-GSCSP-MPCR-Head-GIoU) | 95.45 | 91.28 | 95.93 | 80.36 |
| [C3k2-CA-VoV-GSCSP-MPCR-Head-SIoU](my_project/results/ablation/yolo11n-C3k2-CA-VoV-GSCSP-MPCR-Head-SIoU) | 95.44 | 88.99 | 95.52 | 73.93 |
| [C3k2-CA-VoV-GSCSP-MPCR-Head-Wise-ShapeIoU](my_project/results/ablation/yolo11n-C3k2-CA-VoV-GSCSP-MPCR-Head-Wise-ShapeIoU) | 94.35 | 92.55 | 96.14 | 78.36 |
| [MPCR-Head](my_project/results/ablation/yolo11n-MPCR-Head) | 95.83 | 81.70 | 91.87 | 69.48 |
| [VoV-GSCSP](my_project/results/ablation/yolo11n-VoV-GSCSP) | 89.74 | 84.68 | 91.79 | 67.02 |
| [VoV-GSCSP-MPCR-Head](my_project/results/ablation/yolo11n-VoV-GSCSP-MPCR-Head) | 94.84 | 87.66 | 94.15 | 73.66 |
| [Wise-ShapeIoU](my_project/results/ablation/yolo11n-Wise-ShapeIoU) | 92.77 | 86.81 | 93.42 | 72.50 |

### Loss comparison

| Configuration | Precision (%) | Recall (%) | AP50 (%) | AP50–95 (%) |
| --- | ---: | ---: | ---: | ---: |
| [CIoU](my_project/results/loss/yolo11n-CIoU) | 94.04 | 82.13 | 90.26 | 69.53 |
| [DIoU](my_project/results/loss/yolo11n-DIoU) | 94.21 | 72.69 | 85.50 | 61.60 |
| [EIoU](my_project/results/loss/yolo11n-EIoU) | 94.59 | 81.78 | 89.43 | 64.98 |
| [GIoU](my_project/results/loss/yolo11n-GIoU) | 93.87 | 79.36 | 86.71 | 66.72 |
| [Inner-CIoU](my_project/results/loss/yolo11n-Inner-CIoU) | 93.73 | 85.11 | 90.46 | 69.74 |
| [Inner-PIoU](my_project/results/loss/yolo11n-Inner-PIoU) | 95.57 | 82.13 | 90.74 | 69.52 |
| [Inner-ShapeIoU](my_project/results/loss/yolo11n-Inner-ShapeIoU) | 91.23 | 75.96 | 85.03 | 62.70 |
| [PIoU](my_project/results/loss/yolo11n-PIoU) | 94.42 | 84.89 | 92.23 | 69.51 |
| [ShapeIoU](my_project/results/loss/yolo11n-ShapeIoU) | 92.39 | 84.47 | 90.76 | 64.66 |
| [SIoU](my_project/results/loss/yolo11n-SIoU) | 95.04 | 81.55 | 90.52 | 70.33 |
| [Wise-CIoU](my_project/results/loss/yolo11n-Wise-CIoU) | 94.07 | 85.32 | 92.00 | 69.63 |
| [Wise-Inner-MPDIoU-v3](my_project/results/loss/yolo11n-Wise-Inner-MPDIoU-v3) | 93.81 | 85.11 | 91.52 | 70.90 |
| [Wise-ShapeIoU](my_project/results/loss/yolo11n-Wise-ShapeIoU) | 92.77 | 86.81 | 93.42 | 72.50 |
| [Wise-WIoU](my_project/results/loss/yolo11n-Wise-WIoU) | 93.77 | 83.83 | 92.16 | 70.29 |

### Attention comparison

| Configuration | Precision (%) | Recall (%) | AP50 (%) | AP50–95 (%) |
| --- | ---: | ---: | ---: | ---: |
| [C3k2-CA](my_project/results/attention/yolo11n-C3k2-CA) | 95.06 | 83.62 | 92.81 | 72.93 |
| [CBAM](my_project/results/attention/yolo11n-CBAM) | 95.03 | 81.39 | 91.02 | 67.10 |
| [EMA](my_project/results/attention/yolo11n-EMA) | 93.54 | 78.51 | 88.97 | 61.67 |
| [SEAttention](my_project/results/attention/yolo11n-SEAttention) | 92.90 | 83.47 | 91.50 | 71.59 |

### Detector comparison

| Configuration | Precision (%) | Recall (%) | AP50 (%) | AP50–95 (%) |
| --- | ---: | ---: | ---: | ---: |
| [RT-DETR-L](my_project/results/comparison/rtdetr-l) | 95.05 | 75.53 | 82.93 | 54.83 |
| [YOLO11n](my_project/results/comparison/yolo11n) | 96.98 | 82.12 | 91.30 | 70.02 |
| [YOLO12n](my_project/results/comparison/yolo12n) | 98.56 | 80.00 | 89.94 | 68.44 |
| [YOLO13n](my_project/results/comparison/yolo13n) | 94.96 | 83.40 | 90.12 | 71.35 |
| [YOLOv10n](my_project/results/comparison/yolov10n) | 90.38 | 79.95 | 89.51 | 59.56 |
| [YOLOv5n](my_project/results/comparison/yolov5n) | 93.74 | 73.40 | 82.04 | 61.88 |
| [YOLOv8n](my_project/results/comparison/yolov8n) | 95.31 | 82.12 | 89.75 | 69.46 |
| [YOLOv9t](my_project/results/comparison/yolov9t) | 85.99 | 73.15 | 79.91 | 51.18 |

Training durations differ: YOLOv5n has 2500 recorded epochs, RT-DETR-L has 515, and the other comparison runs have 1200.
All tables link to their training logs and curves. Full summaries are available in
[ablation_summary.csv](my_project/results/ablation_summary.csv) and
[comparison_summary.csv](my_project/results/comparison_summary.csv). Image previews and annotations are excluded.

![SLAT-YOLO training curves](my_project/results/ablation/slat-yolo/results.png)

## Checkpoints and prediction

The downloader provides 17 baseline/ablation inference checkpoints. The final full-model experiment has training logs
but no released checkpoint; the example below uses the architecture ablation without the final Wise-ShapeIoU stage.

```bash
python -m my_project.download_assets checkpoints
python -m my_project.run predict --model my_project/checkpoints/yolo11n-C3k2-CA-C2-STR-VoV-GSCSP-MPCR-Head.pt --source /path/to/images --device 0 --imgsz 1024
```

## License and acknowledgments

This repository uses [AGPL-3.0](LICENSE). Citation metadata is provided in [CITATION.cff](CITATION.cff).
Building blocks were adapted from the authors' experiment archives and the following projects:

[Ultralytics](https://github.com/ultralytics/ultralytics),
[Coordinate Attention](https://github.com/Andrew-Qibin/CoordAttention),
[ToST](https://github.com/RobinWu218/ToST), [Mona](https://github.com/Leiyi-Hu/mona),
[GSConv](https://github.com/AlanLi1997/slim-neck-by-gsconv),
[SEAM / MultiSEAM](https://github.com/Krasjet-Yu/YOLO-FaceV2).

Original third-party notices are retained in [docs/upstream/licenses](docs/upstream/licenses).
SLAT-YOLO is a research project and is not an official Ultralytics product.
