"""Average per-fold KNN positive-class probabilities across multiple backbone
ensemble bundles, then pool across LOCO folds. Tests whether a multi-backbone
ensemble (e.g. Gastronet-SimCLR + Gastronet-DINOv2, 2 folds each = 4 models)
beats a single backbone. Ranks by PPV@90RECALL (the challenge metric).

Each held-out center is scored by the mean of that fold's models across backbones;
folds are then pooled (LOCO folds don't overlap), exactly like train.py's ensemble.

Usage:
  python combine_backbones.py simclr=path/ensemble_knn5.pt dinov2=path2/ensemble_knn5.pt ...
Each path is an ensemble_*.pt from `train.py --stage finetune --loco`.
"""
import os
import sys

import numpy as np
import torch

from metrics import compute_group_eval_metrics


def folds_of(path):
    # map_location=cpu: this runs on the login node (no GPU); bundles hold
    # GPU-saved state_dict tensors we don't use, but torch.load must place them.
    bundle = torch.load(path, weights_only=False, map_location="cpu")
    return {f["fold_index"]: f for f in bundle["folds"]}


def pooled(folds_list):
    """Mean of per-fold positive-class probs across backbones, pooled over folds."""
    all_t, all_s = [], []
    ref = folds_list[0]
    for fi in sorted(ref):
        t = np.asarray(ref[fi]["val_targets"], dtype=int)
        stack = []
        for fl in folds_list:
            assert np.array_equal(np.asarray(fl[fi]["val_targets"], dtype=int), t), (
                f"val targets misaligned at fold {fi} — different split/seed across runs?"
            )
            stack.append(np.asarray(fl[fi]["val_scores"], dtype=float))
        all_t.extend(t.tolist())
        all_s.extend(np.mean(stack, axis=0).tolist())  # average probs across backbones
    return compute_group_eval_metrics(all_t, all_s)


def _row(label, m):
    print(f"  {label:28s} PPV@90R={m['PPV@90RECALL']:.4f}  "
          f"AUROC={m['AUROC']:.4f}  AUPRC={m['AUPRC']:.4f}")


def main(items):
    names = [it.split("=", 1)[0] for it in items]
    paths = [it.split("=", 1)[1] for it in items]
    folds = [folds_of(p) for p in paths]

    print("=== single backbones (pooled LOCO) ===")
    for n, fl in zip(names, folds):
        _row(n, pooled([fl]))
    print(f"=== combined: {'+'.join(names)} ({2 * len(names)} models) ===")
    combined = pooled(folds)
    _row("COMBINED", combined)

    if os.environ.get("WANDB_LOG", "1") == "1":
        try:
            import wandb
            run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "rare26"),
                entity=os.environ.get("WANDB_ENTITY"),
                group="multibackbone",
                name="combined_" + "_".join(names),
                job_type="ensemble",
                config={"run_type": "ensemble_multibackbone",
                        "backbones": names, "n_models": 2 * len(names)},
            )
            wandb.log({f"pooled/{k}": v for k, v in combined.items()})
            for k, v in combined.items():
                wandb.summary[f"pooled/{k}"] = v
            run.finish()
            print("logged combined -> W&B (group=multibackbone)")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] wandb logging skipped: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python combine_backbones.py name=ensemble.pt name2=ensemble2.pt ...")
    main(sys.argv[1:])
