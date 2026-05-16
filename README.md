# Team Internship - AIES Master Program

This repository contains code and resources for the Team Internship project as part of the Master's program in Artificial Intelligence and Engineering Systems (AIES) at TU Eindhoven.

---

## Training pipeline overview

Training is split into two stages: **pretrain** (contrastive, SupMin or SupPro loss) and **finetune** (supervised classifier on top of the frozen or fine-tuned backbone). A **baseline** stage (end-to-end supervised) is also available.

### Example run

```bash
STAGES_CSV='pretrain,finetune' \
sbatch --export=ALL,\
EXPERIMENT_ID=P1_BB_GastronetDinoV2_t1,\
BACKBONES_CSV=gastronet,\
TEMPERATURE=0.07,\
PRETRAIN_BACKBONE_LR=1e-5,\
PRETRAIN_PROJ_LR=3e-4,\
FINETUNE_LR=3e-4,\
BATCH_SIZE=32,\
PRETRAIN_LOSS=suppro,\
WANDB_GROUP=backbonsuppo,\
EXPERIMENT_SAVE_SUBDIR=report_t1 \
jobscript_slurm.sh
```

Or locally:

```bash
python train.py \
  --stage pretrain \
  --loss-name suppro \
  --experiment-id roi-suppro-balanced-v1 \
  --backbone-preset gastronet \
  --epochs 30 \
  --batch-size 32 \
  --balanced-sampler \
  --roi-records-path ./checkpoints/roi_records/roiscroptrain.json \
  --roi-min-crop-scale 0.6 \
  --roi-max-crop-scale 1.0
```

---

## Batch sampling: balanced 50/50 per batch

Pass `--balanced-sampler` to enforce exactly half positives and half negatives in every mini-batch. This is especially important for SupPro pretraining where the contrastive loss is sensitive to class balance.

### How it works (`BalancedBatchSampler` in `data.py`)

At the **start of each epoch** the positive index pool and the negative index pool are each independently shuffled. Batches are then filled by walking through these shuffled pools sequentially — wrapping around and reshuffling when a pool is exhausted.

This guarantees that every positive sample is seen either `floor(B·h / N_pos)` or `ceil(B·h / N_pos)` times per epoch (at most one apart), where `B` is the number of batches, `h = batch_size / 2`, and `N_pos` is the number of positive samples.

**Why not just sample with replacement per batch?**  
Sampling with replacement independently for every batch produces high variance: some positives may appear many times while others are never seen in an epoch. For contrastive losses this matters — stale or over-represented positives bias the embedding space.

---

## Two-view crop augmentation

During contrastive pretraining each image produces two views. Both views go through the same light augmentation pipeline (flip, rotate, colour jitter). On top of that, a **random crop** is applied to every image — positive or negative — before the augmentation transform.

### Crop scale range

The crop scale is sampled **independently for each view** from a uniform distribution:

```
scale ~ Uniform(roi_min_crop_scale, roi_max_crop_scale)
```

| Argument | Default | Meaning |
|---|---|---|
| `--roi-min-crop-scale` | `0.6` | Smallest crop as a fraction of the full image |
| `--roi-max-crop-scale` | `1.0` | Largest crop (1.0 = full image) |

Because each view draws its own scale, the two views of the same image see it at **different zoom levels** on every iteration, adding scale invariance to the contrastive objective.

### Positives: ROI-guided crops

When ROI guidance is active (via `--roi-records-path` or `--roi-guided-training`) and a record exists for a positive sample, both views are cropped to a region **centred on the annotated lesion ROI** at the randomly sampled scale. Each view also gets its own independent center jitter (`--roi-center-jitter`).

- View 1: ROI-centred crop at scale₁, passed through `transform1`
- View 2: ROI-centred crop at scale₂, passed through `roi_transform2`

When ROI guidance is inactive, or for positives without a record, both views fall through to the same random full-image crop described below.

### Negatives: consistent random crops

Negatives have no ROI record but receive the **same random crop treatment** as positives: each view independently samples a scale from `[roi_min_crop_scale, roi_max_crop_scale]` and takes a randomly placed crop of that size from the full image. This prevents the model from using crop scale or field-of-view as a proxy for class identity.

### ROI context and aspect ratio

| Argument | Default | Meaning |
|---|---|---|
| `--roi-context-scale` | `2.0` | Multiplier on the ROI bounding box before cropping (adds surrounding context) |
| `--roi-max-aspect-ratio` | `1.5` | Maximum allowed aspect ratio of the ROI box; shorter side is expanded beyond this |
| `--roi-center-jitter` | `0.05` | Random shift of the crop centre as a fraction of crop size |
| `--roi-focus-prob` | `1.0` | Probability of applying ROI guidance to a positive that has a record |

---

## Hard negative mining (off by default)

Hard negatives are NDBE (label=0) images that a previously trained model incorrectly classified as neoplastic. When provided, they act as additional near-decision-boundary anchors in the SupPro loss.

Hard negative mining is **disabled by default**. It activates only when `--hard-neg-roi-records-path` is explicitly provided and requires `--stage pretrain` with `--loss-name suppro`.

| Argument | Default | Meaning |
|---|---|---|
| `--hard-neg-roi-records-path` | `None` (off) | Path to hard-negative ROI JSON |
| `--hard-neg-roi-weight` | `0.2` | Weight of the hard-negative loss term |
| `--hard-neg-roi-warmup-epochs` | `0` | Epochs before the hard-negative loss is switched on |
