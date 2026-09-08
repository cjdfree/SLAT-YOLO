# SLAT-YOLO development

This repository curates the author's crack-detection experiments on Ultralytics 8.4.143.
Keep upstream core changes focused on the model parser and opt-in loss integration. Use 120-column formatting,
Google-style docstrings, explicit imports and the existing Ultralytics package structure.

Read `my_project/REPRODUCIBILITY.md` before changing model names, architecture, data splits, or claims.
Spectral SEFFN and spatial SEFN are distinct implementations. Preserve checkpoint parameter keys and archived results.
Do not replace or fabricate historical experiment results. Keep new run outputs under `my_project/outputs`.
Large data and checkpoints belong in Releases; their hashes are versioned in `my_project/provenance`.

Run `python -m pytest my_project/tests -o addopts='' -q` after module, parser or loss changes.
Use a feature branch for further changes; do not force-push or publish packages to the upstream Ultralytics project.
