# Clean Baseline — RARE26 Team Internship

Minimal, reproducible code for the best run: SupPro pretrain on Gastronet
DINOv2 with a 20/80 balanced sampler, then post-hoc-fit sklearn KNN / SVM
heads under LOCO + ensemble averaging.

## Layout

```
.
├── data.py                      # Datasets, augmentation presets, LOCO/k-fold splits
├── model.py                     # Backbone + projection MLP + SklearnKNNHead / SklearnSVMHead
├── metrics.py                   # AUROC / AUPRC / PPV@90R / threshold + W&B log helper
├── train.py                     # Single entry point: --stage pretrain | finetune
├── requirements.txt
├── README.md
├── runscripts/
│   ├── jobscript_slurm_pretrain.sh   # Two-fold LOCO SupPro pretrain
│   └── jobscript_slurm_finetune.sh   # Per-fold head fit + ensemble bundle
└── Submission_files/
    ├── Dockerfile
    └── predict.py               # Container entry point — handles single + ensemble checkpoints
```

## What's in / out

**In** — only the knobs that actually moved the needle on the best run:

- Backbones: Gastronet (DINOv2 ViT-B/14 reg4) + DINOv3 + SimCLR / MoCo-v2 / ResNet-50.
- SupPro contrastive pretrain on the projection head.
- BalancedBatchSampler with configurable `--pos-ratio` (default `0.2` = 20/80).
- Augmentation intensity presets `1..4` (`--augmentation-intensity`).
- LOCO leave-one-center-out CV with automatic ensemble bundling.
- Post-hoc sklearn heads: KNN with `--knn-neighbors 5,25,51` and SVM with `--svm-C 0.5,2,10`.
  One finetune call fits every head × hyperparameter and writes ensemble bundles.

**Out** — explicitly removed:

- All ROI / Grad-CAM-guided sampling machinery.
- Linear, MLP, cosine, residual, LN-linear heads (the differentiable heads).
- Baseline (frozen-backbone + supervised CE) stage.
- Supervised contrastive ablations (`global-only`, `roi-aux`, …).
- Top-block unfreeze + finetune backbone LR.
- `--init-encoder-ckpt` warm-start.
- Grad-CAM training/eval, hard-negative ROI, external testset code path.

## End-to-end run

### 1. Pretrain (one job → two LOCO encoders)

```bash
sbatch runscripts/jobscript_slurm_pretrain.sh
```

Defaults reproduce the best run: Gastronet, T=0.1, backbone LR 1e-5, projection
LR 3e-4, batch 32, pos_ratio 0.2, aug intensity 3, 50 epochs, 2-fold LOCO.

Override with env vars:
```bash
EXPERIMENT_ID=my_run BACKBONE_PRESET=dinov3 EPOCHS=30 \
    sbatch runscripts/jobscript_slurm_pretrain.sh
```

Encoder checkpoints land at
`checkpoints/<EXPERIMENT_ID>/<EXPERIMENT_ID>_pretrain_fold{0,1}/<...>_encoder.pt`.

### 2. Finetune (one job → all heads + ensembles)

```bash
ENCODER_CKPTS=checkpoints/my_run/my_run_pretrain_fold0/my_run_pretrain_fold0_encoder.pt,checkpoints/my_run/my_run_pretrain_fold1/my_run_pretrain_fold1_encoder.pt \
HEAD_TYPES=knn,svm \
KNN_NEIGHBORS=5,25,51 \
SVM_C=0.5,2,10 \
EXPERIMENT_ID=my_run \
    sbatch runscripts/jobscript_slurm_finetune.sh
```

Outputs:

```
checkpoints/my_run/my_run_finetune/knn5/my_run_finetune_knn5_fold0.pt
checkpoints/my_run/my_run_finetune/knn5/my_run_finetune_knn5_fold1.pt
...
checkpoints/my_run/my_run_ensembles/ensemble_knn5.pt          ← what you submit
checkpoints/my_run/my_run_ensembles/ensemble_svmC2.0.pt
```

Per-fold W&B run gets one summary point with `val/AUROC`, `val/AUPRC`,
`val/PPV@90RECALL`, `val/Threshold`. The pooled cross-LOCO-val metrics for
each ensemble are printed at the end of the finetune job.

### 3. Submit

```bash
cp checkpoints/my_run/my_run_ensembles/ensemble_knn5.pt model.pt
docker build -t team-internship:latest -f Submission_files/Dockerfile .
docker save -o RARE26-submission.tar team-internship:latest
```

`predict.py` auto-detects whether the checkpoint is a single fold or an
ensemble bundle. The ensemble path averages each fold's positive-class
probability per sample.

## Configuration cheatsheet

All knobs are pure CLI flags on `train.py`. Defaults match the best run.

| Flag                          | Default | Notes                                    |
| ----------------------------- | ------- | ---------------------------------------- |
| `--stage`                     | —       | `pretrain` or `finetune` (required)      |
| `--backbone-preset`           | `gastronet` | see `BACKBONE_PRESETS` in train.py   |
| `--backbone-weights-path`     | None    | required for Gastronet/SimCLR/MoCo       |
| `--input-size`                | 336     |                                          |
| `--loco`                      | off     | enables leave-one-center-out             |
| `--num-folds`                 | 1       | with `--loco`: must equal #centers       |
| `--fold-index`                | 0       | starting fold (or only fold)             |
| `--epochs`                    | 50      |                                          |
| `--batch-size`                | 32      |                                          |
| `--pretrain-backbone-lr`      | 1e-5    | dominant lever for SupPro                |
| `--pretrain-proj-lr`          | 3e-4    |                                          |
| `--warmup-epochs`             | 3       |                                          |
| `--temperature`               | 0.1     |                                          |
| `--base-temperature`          | 0.07    |                                          |
| `--balanced-sampler`          | off     | enable BalancedBatchSampler              |
| `--pos-ratio`                 | 0.2     | 20/80 in the best run                    |
| `--augmentation-intensity`    | 3       | 1 (low) … 4 (extreme)                    |
| `--head-types`                | `knn`   | subset of `{knn, svm}`                   |
| `--knn-neighbors`             | `5`     | CSV — fits one head per value            |
| `--svm-C`                     | `2.0`   | CSV — fits one head per value            |
| `--encoder-ckpt`              | None    | CSV: one per fold for `--loco`           |

## Determinism

`seed_everything` sets PYTHONHASHSEED / numpy / torch / cudnn deterministic
mode and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Dataset row order is sorted by
(center, label, img), so the same `--seed` yields the same LOCO/k-fold split.
