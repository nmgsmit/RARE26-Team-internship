# RARE26 — project standards

Barrett's esophagus classification (RARE25/26 challenge). Two-stage SupPro pipeline.
Work on branch **`clean-baseline`**. GitHub pushes are done by the user — commit locally, don't push.

## W&B logging (always)
- **Entity `ngmtue`, project `rare26`.** (The account's only writable entity is `ngmtue` — note g-m, not `nmgtue`.)
- Credentials + routing live in gitignored **`.env`** at repo root:
  `WANDB_API_KEY=…`, `WANDB_ENTITY=ngmtue`, `WANDB_PROJECT=rare26`.
- Both jobscripts auto-export `WANDB_ENTITY`/`WANDB_PROJECT` from `.env`, so a bare
  `sbatch` already logs to the right place. A value passed via `--export` overrides it.
- Groups: pretrain/finetune of a baseline → `clean-baseline`; smoke tests → `clean-baseline-smoke`.

## Compute (Snellius)
- Run everything as account **`scur2421`** (NOT `nsmit2`). SSH alias **`snellius-proj`**.
- Repo on Snellius: `~/RARE26`. Dataset: `~/data/Challenge_train_data`. Backbones: `~/Gastronet/`.

## Submitting jobs (Snellius gotchas)
- Snellius defaults `SBATCH_EXPORT=NONE`, and `module` is a shell function. To pass env
  overrides AND keep `module load` working, submit as:
  ```
  ssh snellius-proj 'source /etc/profile; cd ~/RARE26; \
    EXPERIMENT_ID=x EPOCHS=50 sbatch --export=ALL runscripts/jobscript_slurm_pretrain.sh'
  ```
- Comma-valued vars (e.g. `ENCODER_CKPTS=a,b`) must be **prefix** vars, never inline in
  `--export` (comma is a separator there).
- Deploy edited `.sh` files as **LF** — Windows edits add CRLF that `sbatch` rejects
  (`sed -i 's/\r$//' file.sh`).

## Pipeline
- `--stage pretrain`: SupPro encoder. Logs `train_loss`/`valid_loss` + `sep/*` (class
  separation in projection space). Saves one encoder per LOCO fold.
- `--stage finetune`: fits a sklearn **KNN/SVM head on frozen features** (no backprop),
  logs per-fold `val/*`, builds a LOCO ensemble (mean of fold probabilities), and logs the
  pooled cross-center result as one `run_type=ensemble` run with `pooled/*` in its summary.
  → To compare head/aug ablations, sort the `pooled/*` (ensemble) runs.
- **Primary metric to optimize: `pooled/PPV@90RECALL`** (the challenge's operating-point
  metric). `pooled/AUROC`/`AUPRC` are secondary/stabler context. Note PPV@90R is noisy here
  and dominated by center_1's ~2.7% prevalence — read it alongside AUPRC.
- `sep/*` (pooled backbone space, what the head uses) vs `sep_proj/*` (projection space the
  loss optimises): projection sep rises ~tautologically; only pooled sep + downstream `val/*`
  reflect transferable quality. Both are intra-center on the held-out fold, so they don't
  capture cross-center transfer the way the LOCO kNN does.
- **`probe/*` (pretrain, per-epoch): the metric to actually watch.** A cross-center kNN-5
  probe — fit on the train center's pooled features, score the held-out center → a live
  per-epoch preview of LOCO `probe/PPV@90RECALL`/`AUROC`/`AUPRC`. Unlike `sep/*` (intra-
  center), it measures transfer. No TTA, so it reads slightly below the final finetune
  number, but the trend is the honest signal of whether more pretraining is helping.
- Encoder path per fold:
  `checkpoints/<EID>/<EID>_pretrain_fold<i>/<EID>_pretrain_fold<i>_encoder.pt`.

## Augmentation
- **`--crop-min-scale` (default 0.95)** controls the random crop that builds the two SupPro
  views; `1.0` disables cropping. 0.95 reproduces the best historical run. This is the knob
  for crop-size ablations. (`CROP_MIN_SCALE` env in the pretrain jobscript.)
