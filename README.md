# RARE26 Challenge — Team A

Code for our submission to the RARE26 challenge (Recognition of Abnormalities in low-pREvalence cancer), developed as part of the AI&ES Master's Team Internship at TU Eindhoven.

> **Branches**
> - `main` — full research codebase with all ablation tooling, ROI helpers, and analysis scripts.
> - `clean-baseline` — a minimal, self-contained branch that reproduces only the best-performing run. Intended as a clean starting point for future research: no ablation scaffolding, just the configuration that scored highest on the RARE26 challenge.

---

## Project overview

Barrett's Esophagus (BE) is a precancerous condition of the lower oesophagus. Neoplastic lesions appear in fewer than 1% of routine surveillance cases, creating extreme class imbalance that causes standard classifiers to produce high false-positive rates. This repository implements a **two-stage supervised contrastive learning pipeline** to address this:

1. **Pretraining** — a backbone is optimised with the **Supervised Prototypes (SupPro)** contrastive loss, which explicitly organises the embedding space by pulling samples toward fixed class prototypes. This produces a well-separated feature space even under severe class imbalance.
2. **Finetuning** — the backbone is frozen and a lightweight classification head (k-NN, SVM, MLP, or linear) is trained on top of the learned embeddings.

The best single-model configuration uses a **GastroNet-5M DINOv2** backbone (domain-specific pretraining on gastrointestinal endoscopy images) combined with the SupPro loss and a k-NN classification head, achieving a PPV@90Recall of 0.0223 on the hidden RARE26 test set — higher than the best model of the previous RARE25 challenge.

---

## Repository structure

```
RARE26-Team-internship/
│
├── train.py                  # Main training entry point
├── model.py                  # Backbone + projection/classification head definitions
├── data.py                   # Dataset, transforms, and balanced batch sampler
├── gradcam.py                # Grad-CAM computation and consensus-mass evaluation
├── metrics.py                # PPV@90Recall, AUROC, AUPRC, and related metrics
├── testdata.py               # Loaders for the EVC Barrett's segmentation test set
│
├── ROI_helpers/
│   ├── roi_guidance.py       # ROI crop logic (centred crops from bounding boxes)
│   ├── build_gradcam_cache.py# Pre-compute and cache Grad-CAM ROI records for training images
│   └── export_train_rois.py  # Export per-image ROI JSON from a trained model checkpoint
│
├── runscripts/
│   ├── jobscript_slurm.sh          # Main SLURM job script (pretrain + finetune)
│   ├── jobscript_slurm_pretrain.sh # Pretrain-only SLURM script
│   ├── jobscript_slurm_loco_final.sh # LOCO (leave-one-center-out) final runs
│   ├── main.sh / main_pretrain.sh  # Local convenience wrappers
│   └── submit_best_model.sh        # Script to run the final best-model configuration
│
├── Submission_files/
│   └── predict.py            # Challenge submission: loads a checkpoint and runs inference
│
├── Interactive_analysis/
│   ├── extract_features.py   # Batch-extract pooled + projected embeddings from checkpoints
│   ├── model.py              # Thin model wrapper used by the analysis notebooks
│   ├── data.py               # Data utilities used by the analysis notebooks
│   ├── notebook.py           # Interactive PCA / embedding visualisation (live)
│   ├── notebook_data.py      # Pre-loads feature data for notebook visualisations
│   └── notebook_static.py    # Generates static PCA plots for reporting
│
├── visualizationscripts/
│   ├── visualize_roi_sampling.py   # Visualise ROI-guided vs. random crop pairs
│   ├── visualize_suppro_batch.py   # Visualise a SupPro training batch
│   └── visualize_suppro_views.py   # Visualise the two augmented views for a single image
│
└── data/
    └── Challenge_train_data/ # RARE26 training images, organised by center and class
```

---

## How the files relate

`train.py` is the central script. It imports from all other root-level modules:

- **`model.py`** defines the `Model` class, which wraps a backbone (loaded from a GastroNet or DINOv3 checkpoint via `timm`) with a projection head (used during SupPro pretraining) or a classification head (used during finetuning). It also implements k-NN and SVM heads as non-parametric alternatives.
- **`data.py`** handles everything data-related: building train/validation dataframes with stratified splits, constructing `SimpleDataset` objects with the appropriate augmentation transforms, and providing the `BalancedBatchSampler` for controlling class proportions in each mini-batch.
- **`gradcam.py`** computes Grad-CAM activations over ViT backbones and evaluates the **consensus mass** metric — the fraction of a model's CAM signal that falls inside clinician-annotated lesion regions.
- **`metrics.py`** centralises all evaluation logic: PPV@90Recall, AUROC, AUPRC, and threshold selection.
- **`testdata.py`** loads the EVC Barrett's segmentation dataset used for Grad-CAM evaluation.
- **`ROI_helpers/roi_guidance.py`** is called both from `data.py` (to perform ROI-centred crops during training) and from `train.py` (to update ROI records from live Grad-CAM signals).

`Interactive_analysis/extract_features.py` reads a `runs.csv` file listing experiment checkpoints, loads each model, and saves pooled embeddings and logits to `features_out/`. These are then consumed by the notebook scripts to generate PCA projections of the embedding space — the main tool used in the paper to understand class and center separation.

`Submission_files/predict.py` is a self-contained script that loads a saved model checkpoint and produces the CSV output required by the RARE26 submission server.

---

## Training pipeline

Training proceeds in two stages, controlled by the `--stage` argument to `train.py`.

### Stage 1 — Pretraining (SupPro contrastive loss)

The backbone is initialised from a pretrained checkpoint (GastroNet-5M DINOv2 by default). During pretraining, each image in the batch is passed through a stochastic augmentation pipeline **twice**, producing two differently-augmented views of the same image. The SupPro loss then optimises the backbone and a projection head to pull both views toward fixed class prototypes while separating the two classes in embedding space.

The loss combines NT-Xent (standard unsupervised contrastive) with a prototype attraction term: if a sample's similarity to its class prototype is below 0.5, the prototype term is added; once the sample is close enough, only the NT-Xent term acts. This conditional design prevents the prototypes from collapsing all embeddings to a single point, which is the failure mode of standard SupCon on severely imbalanced data.

Training uses SGD with momentum 0.9, weight decay 1×10⁻⁴, a 3-epoch linear warm-up followed by cosine annealing, a backbone learning rate of 1×10⁻⁵, and a projection-head learning rate of 3×10⁻⁴.

### Stage 2 — Finetuning (classification head)

After pretraining, the backbone weights are frozen and a new classification head is attached. Four head types are supported: `linear`, `knn` (k=5), `svm` (RBF kernel), and `mlp`. The k-NN head performed best overall because it directly exploits the cluster geometry created by SupPro without introducing additional learned weights that could disrupt the feature structure on this small dataset.

A constant learning rate of 3×10⁻⁴ is used for the classification head, trained for 30 epochs.

### Baseline

Passing `--stage baseline` runs end-to-end supervised training with cross-entropy loss, without the contrastive pretraining stage. This served as the performance baseline in the ablation studies.

---

## Key training concepts

### Balanced batch sampling

Contrastive learning requires positive (neoplastic) samples to be present in every mini-batch. With only ~5% positives in the dataset, random sampling produces many batches with zero positives and no useful training signal for the SupPro loss.

The `BalancedBatchSampler` in `data.py` addresses this by enforcing a fixed proportion of positive images per batch. The positive and negative index pools are independently shuffled at the start of each epoch, then batches are filled by walking sequentially through these pools — wrapping around and reshuffling when a pool is exhausted. This ensures every sample is seen approximately the same number of times per epoch, avoiding the bias of per-batch sampling with replacement.

The proportion is controlled by `--positive-share` (e.g. `0.25` for 25% positives). In the ablation studies, a 25% positive share produced the best class separation in embedding space and the best hidden-server performance. The `--balanced-sampler` flag activates the sampler; without it, natural dataset sampling is used.

### Two-view crop augmentation

During contrastive pretraining, every image produces **two independently augmented views**. Both views pass through the same base augmentation pipeline (random horizontal flip, vertical flip, rotation, and colour jitter). On top of this, a **random crop** is applied independently to each view:

```
scale ~ Uniform(roi_min_crop_scale, roi_max_crop_scale)
```

Because each view samples its own crop scale, the two views of the same image see different zoom levels on every iteration. This teaches the model scale invariance: embeddings for the same lesion should be similar regardless of how much surrounding context is visible.

| Argument | Default | Meaning |
|---|---|---|
| `--roi-min-crop-scale` | `0.6` | Smallest crop as a fraction of the full image |
| `--roi-max-crop-scale` | `1.0` | Largest crop (1.0 = full image) |
| `--augmentation-intensity` | `3` | Preset for flip/rotation/colour jitter strength (1–4) |

For positive (neoplastic) samples with a Grad-CAM ROI record available, both crops are additionally **centred on the annotated lesion region** at their respective scales, ensuring the lesion is visible in both views. Negative samples receive the same random crop treatment as positives, preventing the model from using crop field-of-view as a shortcut for class identity.

### ROI-guided sampling

Standard small random crops can occasionally exclude the visible lesion entirely, removing the diagnostic signal from the positive view. ROI-guided sampling counters this by cropping neoplastic images around the lesion ROI, derived from Grad-CAM activations of a previously trained model.

The ROI records are stored as JSON (default path: `./checkpoints/roi_records/`) and loaded via `--roi-records-path`. When an ROI record exists for a positive sample, both views are centred on that region with a configurable context multiplier (`--roi-context-scale`, default 2.0) and per-view centre jitter (`--roi-center-jitter`, default 0.05). The probability of applying ROI guidance is controlled by `--roi-focus-prob` (default 1.0 for positive samples).

---

## Elaborate example run (SLURM / HPC)

The following reproduces the best configuration from the paper: GastroNet-5M DINOv2 backbone, SupPro loss, ROI-guided crops, and a LOCO (leave-one-center-out) split to prevent patient-level data leakage.

### How LOCO works

LOCO is orchestrated entirely inside `train.py` when the `--loco` flag is set. A single job runs the full pipeline:

1. **Pretrain loop** — one encoder is trained per held-out center (center 1 held out → encoder trained on center 2, and vice versa). Encoder checkpoints are saved under `<save-dir>/pretrain/fold{i}_val_{center}/`.
2. **Finetune loop** — for each fold, the matching encoder is auto-loaded via the `--encoder-ckpt` template, and a classification head is trained on the remaining data.
3. **Ensemble + submission artifact** — after all folds, `train.py` averages the fold predictions and writes a single merged `_submission.pt` file ready to be placed in the challenge container.

No manual looping over centers is needed.

### Step 1 — Submit the LOCO job

```bash
sbatch runscripts/jobscript_slurm_loco_final.sh
```

The script runs both stages in sequence in a single SLURM job (up to 24 h on a single A100). Progress is logged to `slurm_trainmodel/slurm_loco_final-<jobid>.out` and to Weights & Biases.

Key hyperparameters used (set inside the script):

| Parameter | Value |
|---|---|
| Backbone | GastroNet-5M DINOv2 (`gastronet`) |
| Pretrain loss | `suppro`, τ = 0.10 |
| Backbone LR | 1×10⁻⁵ |
| Projection-head LR | 3×10⁻⁴ |
| Finetune LR | 2×10⁻⁴ |
| Batch size | 32 |
| Epochs (pretrain / finetune) | 30 / 30 |
| Augmentation intensity | 3 (default) |
| ROI focus probability | 0.5 |
| Crop scale range | [0.4, 1.0] |
| Seed | 42 (deterministic) |

When the job finishes, the submission artifact is at:

```
./checkpoints/<run_tag>/finetune/<run_tag>_finetune_submission.pt
```

with calibration metadata in the matching `_submission.json`.

### Step 2 — Extract embeddings for analysis (optional)

After training, you can extract embeddings for all runs listed in `Interactive_analysis/runs.csv`:

```bash
cd Interactive_analysis
python extract_features.py \
  --runs-csv runs.csv \
  --datasets train val evc \
  --output-dir features_out
```

This saves pooled backbone features, projection-head features, and deployed logits to `features_out/<experiment_id>__<dataset>/`.

### Step 3 — Generate PCA projections (optional)

```bash
cd Interactive_analysis
python notebook_static.py \
  --features-dir features_out \
  --experiment-id <run_tag>_finetune \
  --output-dir pca_plots
```

### Step 4 — Challenge submission

`Submission_files/predict.py` is designed to run inside the challenge Docker container. Paths are hardcoded to the container layout:

| Path | Role |
|---|---|
| `/app/model.pt` | Model checkpoint to load |
| `/data/test` | Directory of test images |
| `/output/predictions.csv` | Output file written by the script |

To use it, copy the `_submission.pt` artifact to `/app/model.pt` inside the container image and run:

```bash
python Submission_files/predict.py
```

The script loads the single checkpoint (which already contains the LOCO-ensembled weights), runs inference over all test images, and writes `sample_id, prediction` rows to `/output/predictions.csv`.

---

## Evaluation metrics

| Metric | Description |
|---|---|
| **PPV@90Recall** | Positive Predictive Value when the decision threshold is set so that 90% of all neoplastic cases are recalled. The primary challenge metric. |
| **AUROC** | Area under the ROC curve — aggregate discrimination across all thresholds. |
| **AUPRC** | Area under the Precision-Recall curve — particularly informative under class imbalance. |
| **Consensus Mass** | Fraction of total Grad-CAM activation that falls inside clinician-annotated lesion regions. Measures whether the model attends to the actual pathology. |

---

## Data

The training data is provided by the RARE26 challenge under a non-commercial CC-BY-NC-SA licence. Place it under `data/Challenge_train_data/` organised as:

```
data/Challenge_train_data/
  center_1/
    ndbe/   ← non-dysplastic Barrett's Esophagus images
    neo/    ← neoplasia images
  center_2/
    ndbe/
    neo/
```

The internal segmentation test set (EVC Barrett's, 100 images with expert masks) should be placed under `data/EVC_Barretts_FullSet/`.
