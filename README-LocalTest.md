# Local Test Guide

This guide explains how to run `localtest.sh` locally to evaluate a checkpoint with `TestResult.py`.

## What this script does

`localtest.sh`:
- validates your checkpoint path,
- calls `mainTEST.sh`, which holds the editable test settings and runs `TestResult.py`,
- runs evaluation and prints:
  - AUROC, AUPRC,
  - Precision, Recall, F1,
  - TP, FP, FN, TN,
  - confusion matrix.

## Quick start

From the repository root:

```bash
sbatch localtest.sh ./checkpoints/<your_model>.pt
```

Equivalent using environment variable:

```bash
MODEL_PATH=./checkpoints/<your_model>.pt sbatch localtest.sh
```

## Standard settings

If you do not pass any overrides, these defaults are used:

- `IMAGES_DIR=./data/EVC_Barretts_FullSet/images`
- `IMAGE_SIZE=224`
- `BATCH_SIZE=32`
- `BACKBONE_NAME=vit_base_patch16_dinov3`
- `THRESHOLD=0.5`
- `PRETRAINED_FLAG=0`

So the most standard run is:

```bash
sbatch localtest.sh ./checkpoints/<your_model>.pt
```

## Optional overrides

You can override defaults without editing files:

```bash
IMAGES_DIR=./data/EVC_Barretts_FullSet/images \
IMAGE_SIZE=224 \
BATCH_SIZE=32 \
BACKBONE_NAME=vit_base_patch16_dinov3 \
THRESHOLD=0.5 \
PRETRAINED_FLAG=0 \
sbatch localtest.sh ./checkpoints/<your_model>.pt
```

These defaults are defined in `mainTEST.sh` under the `Test settings` block.

### How to override exactly

Use this pattern:

```bash
VAR1=value1 VAR2=value2 sbatch localtest.sh ./checkpoints/<your_model>.pt
```

Rules:
- Put overrides before `/bin/bash ...` on the same line.
- Overrides only affect that single command run.
- If you do not set a variable, the script uses its default.

Examples:
Override only backbone:

```bash
BACKBONE_NAME=vit_large_patch14_dinov2 sbatch localtest.sh ./checkpoints/<your_model>.pt
```

Meaning of overrides:
- `IMAGES_DIR`: folder with `.png` images.
- `IMAGE_SIZE`: resize size used before inference.
- `BATCH_SIZE`: inference batch size.
- `BACKBONE_NAME`: timm backbone used to build model before loading checkpoint.
- `THRESHOLD`: probability cutoff for positive class (ACHD).
- `PRETRAINED_FLAG`: `1` passes `--pretrained`, `0` passes `--no-pretrained`.

## Common issues

- "Model checkpoint not found": verify `MODEL_PATH` points to an existing `.pt`/`.pth` file.
- "Images directory not found": set `IMAGES_DIR` to a valid folder.
- State dict/key mismatch errors: ensure `BACKBONE_NAME` and model architecture match the checkpoint used during training.

## Tip

If your shell is not in the repo root, first `cd` into the repo, then run the commands above.
