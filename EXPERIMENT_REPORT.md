# RARE26 — Experiment Report

Barrett's esophagus neoplasia classification (RARE25/26). Two-stage SupPro pipeline:
frozen/contrastively-tuned backbone → sklearn head on frozen features. LOCO (leave-one-
center-out) CV over 2 centers. Branch `clean-baseline`. W&B: `ngmtue/rare26`.

## Data
- **center_1**: 2218 ndbe (neg) / 61 neo (pos) = 2279, **2.7% positive**
- **center_2**:  719 ndbe (neg) / 97 neo (pos) =  816, **11.9% positive**
- Total 3095, ~5.1% positive. **4× prevalence gap between centers + heavy imbalance.**
- Deployment target prevalence ≈ **1/100**.

## Metric
- Operating-point metric: **PPV@90RECALL** (challenge). It is **prevalence-sensitive** and at
  low prevalence is **FPR-limited**: `PPV@90R ≈ 0.9·π / (0.9·π + FPR·(1−π))`.
- AUROC / AUPRC for ranking context (with caveats below).

---

## The journey & key findings

### 1. Baseline + the missing crop (regression fixed)
`clean-baseline` had dropped **all crop augmentation** — the two SupPro views differed only by
flip/colour, so the contrastive task collapsed (train loss fell, val loss flat). main built each
view with a random scale-0.95 crop. Restored as `--crop-min-scale` (default 0.95; `RandomFixedScaleCrop`).
**Crop scale is settled: larger crop (milder, ~0.95) consistently wins; 1.0 disables cropping and re-collapses the views.**

### 2. Multi-backbone fusion — combining helps (late fusion only)
Three Gastronet backbones: **dino** (DINOv2 ViT-B/14), **simclr** (ResNet50), **moco** (ResNet50).
- **Late fusion (average probabilities) beats the best single backbone**, verified per-center (not a pooling artifact). Triple > dino alone on both centers.
- **Early fusion (feature concat) is worse** — the weaker backbones pollute the joint kNN metric. Dropped.
- "Count k neighbours across backbones" == "average the scores" for equal k (identical operation).

### 3. Three metric traps (all from coarse kNN scores)
- **PPV@90RECALL cliff:** kNN vote-fractions give ~10–15% of positives a score of **exactly 0**
  (no positive among k neighbours, cross-center), tied with thousands of score-0 negatives.
  Reaching 90% recall forces the threshold to 0 → admits everything → **PPV = prevalence (0.0511) for every model.** Models are excellent ≤80% recall (95% precision) then cliff.
- **AUPRC granularity bias:** `sklearn.average_precision_score` is biased *up* by tie-breaking.
  Jittering scores with pure noise (no info) lifts AUPRC +0.018. **AUROC is immune** (ranking metric, ties = 0.5). ⇒ rank fusion by AUROC; AUPRC only fair once scores are continuous.
- **kNN probe blindness:** the per-epoch cross-center kNN-5 probe was flat over epochs — *not*
  because the encoder wasn't improving, but because kNN couldn't *see* it. This misled us into
  thinking pretraining didn't help (it does — see §5).

### 4. Continuous head (logistic) — the big win
Replacing the kNN head with a **continuous-score** head (sklearn on StandardScaler'd frozen
features, no backprop / no MLP) removes the ties → un-cliffs PPV@90R and unbiases AUPRC.

| Head (triple fusion, k/where applicable) | PPV@90R | AUROC | AUPRC |
|---|---|---|---|
| kNN-5 (original baseline) | 0.0511 | 0.90 | 0.81 |
| distance-weighted kNN | 0.0511 (still cliffs) | 0.926 | 0.851 |
| linear SVM (Platt) | 0.556 | 0.979 | 0.904 |
| **logistic regression** | **0.730** | **0.979** | **0.925** |

Ranking: **logistic > linSVM > dwknn**. dwknn still cliffs (confirms root cause = hard score-0
zeros, not granularity). **Current best model = logistic regression (class_weight=balanced) on the triple late-fusion.**

### 5. Pretrain matters a lot (frozen vs trained)
Epoch-0 frozen Gastronet vs 50-epoch SupPro, same logistic+triple head:

| | PPV@90R | AUROC | AUPRC |
|---|---|---|---|
| epoch-0 (frozen) | 0.238 | 0.946 | 0.784 |
| 50-epoch SupPro | **0.730** | 0.979 | 0.925 |

**SupPro pretraining ~triples PPV@90R.** It is a major lever, not a dead end (the earlier "pretrain doesn't help" read was the kNN-probe trap). The probe is now **logistic** and saves the **best-probe-epoch** encoder.

### 6. Prevalence & threshold stress test (deployment reality)
Best model (logistic+triple), pooled operating point TPR 0.905 / FPR 0.018:

| Prevalence | PPV@90R |  | Per-center (FPR@90R) | PPV@1% |
|---|---|---|---|---|
| 5.1% (obs) | 0.730 | | center_1 (FPR 0.052) | 0.148 |
| 2% | 0.506 | | center_2 (FPR 0.004) | 0.687 |
| **1% (target)** | **0.336** | | | |
| 0.5% | 0.201 | | | |

- **At 1/100, PPV@90R ≈ 0.34, not 0.73.** Driven entirely by FPR@90recall, not AUROC.
- **center_1 is the bottleneck** (FPR 5.2% → collapses at 1%); center_2 is strong (FPR 0.4%).
- The pooled 0.73 is **optimistic**: a single threshold hits 90% pooled recall mostly via the
  easy center_2 while under-recalling center_1.
- **⇒ The lever for deployment is lowering FPR@90recall on center_1, not raising AUROC.**

---

## Current best model
**Logistic regression (sklearn, StandardScaler'd frozen pooled features, class_weight=balanced),
late-fused (mean probability) over dino+simclr+moco, hflip TTA.**
Pooled LOCO: PPV@90R 0.73 (@5%), AUROC 0.979, AUPRC 0.925. PPV@1% ≈ 0.34 (center_1-limited).

## Hyperparameters
| Hyperparameter | Status | Note |
|---|---|---|
| crop_min_scale | **done** = 0.95 | larger crop wins; 1.0 collapses views |
| head type | **done** = logistic | continuous scores; un-cliffs PPV@90R |
| dino backbone-LR & T | **done** (pre-optimized) | ViT left as-is |
| ResNet backbone-LR | **in progress** | simclr/moco × {1e-5,1e-4,3e-4,1e-3}; ResNets likely under-tuned at the ViT's 1e-5 |
| temperature T (ResNets) | **next** | sweep at best LR |
| **batch size** | **TODO (high)** | contrastive learning is # -negatives sensitive; 32 is small for SupCon |
| **pos-ratio** (balanced sampler) | **TODO (high)** | positive fraction/batch (0.2) — directly shapes the minority contrastive signal |
| logistic C / class_weight | **TODO (cheap)** | head regularisation; no re-extraction needed |
| proj-lr / proj-dim | TODO (low) | projection head is discarded downstream |
| warmup, base-T | TODO (low) | minor |
| epochs | handled | best-probe-epoch encoder saved automatically |

## Methodology guardrails
- **Rank by the cross-center logistic probe** (FPR@90recall / AUPRC on the held-out center), not
  intra-center `sep/*` or AUROC alone.
- **Equal-weight fusion only** — no val-performance weighting (leakage with 2 LOCO centers).
- **Best-epoch by probe AUPRC**, saved per run.
- ROI-guided sampling is **off the table** (broken/biased ROI JSON).

## Tooling
- `train.py` — pretrain (logistic probe, best-epoch save) + finetune (sklearn head).
- `combine_backbones.py` — late fusion (avg probs) from ensemble bundles.
- `concat_fusion.py` — early fusion (concat features, one kNN). [worse; kept for reference]
- `eval_heads.py` / `runscripts/eval_heads.sh` — continuous-head ablation (logistic/linSVM/dwknn).
- `prevalence_analysis.py` / `runscripts/prevalence.sh` — per-center vs pooled threshold + prevalence sweep.
- Snellius: account `scur2421`, alias `snellius-proj`, repo `~/RARE26`. Submit with
  `source /etc/profile; sbatch --export=ALL` (see CLAUDE.md).

## Next steps
1. Finish ResNet LR sweep → pick best LR per ResNet by probe FPR@90R/AUPRC (esp. center_1).
2. Temperature sweep at best LR (per ResNet).
3. batch-size & pos-ratio sweeps (likely high-impact, untouched).
4. Logistic C / class_weight tuning (cheap).
5. Re-fuse best per-backbone encoders; re-check PPV@1% on center_1 (the deployment number).
