"""Early-fusion (feature concatenation) multi-backbone ensemble for LOCO.

For each LOCO fold: load each backbone's encoder, extract pooled features for the
train center and the held-out val center, L2-normalise PER backbone, concatenate
across backbones, fit one KNN on the joint space, score the held-out center
(horizontal-flip TTA at the probability level), then pool across folds.

This is the early-fusion counterpart to combine_backbones.py (late fusion) -- the
only scheme where the backbones' geometries actually interact (a single metric over
the joint space). Run with ONE eid to get the per-backbone normalised-kNN baseline
(apples-to-apples with the concat, same metric); run with several to fuse.

Needs a GPU (feature extraction). Usage (venv active):
  python concat_fusion.py --eids EID_A,EID_B[,EID_C] --knn 20 --tag dino_simclr
Encoders are read from the standard path:
  checkpoints/<EID>/<EID>_pretrain_fold<i>/<EID>_pretrain_fold<i>_encoder.pt
"""
import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from data import (SimpleDataset, build_dataset_dataframe, build_eval_transform,
                  seed_worker, split_dataframe)
from metrics import compute_group_eval_metrics
from model import Model, load_encoder_checkpoint


def _encoder_path(eid, fold):
    return f"checkpoints/{eid}/{eid}_pretrain_fold{fold}/{eid}_pretrain_fold{fold}_encoder.pt"


def _load_encoder(path, n_classes, input_size, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("model_config", {}))
    cfg["n_classes"] = n_classes
    cfg["pretrained"] = False
    cfg["head_type"] = "knn"
    cfg["knn_neighbors"] = 5
    cfg.setdefault("input_size", input_size)
    for meta in ("backbone_preset", "num_folds", "fold_index", "loss_name",
                 "backbone_weights_path"):
        cfg.pop(meta, None)
    model = Model(**cfg).to(device)
    load_encoder_checkpoint(model, path, strict=False)
    model.eval()
    return model


@torch.no_grad()
def _pooled(model, loader, device, flip):
    """L2-normalised pooled features (N, D); optionally horizontal-flipped input."""
    out = []
    for images, _ in loader:
        images = images.to(device)
        if flip:
            images = torch.flip(images, dims=[-1])
        out.append(F.normalize(model.encode(images), dim=-1).cpu().numpy())
    return np.concatenate(out, 0)


def _labels(loader):
    return np.concatenate([y.numpy() for _, y in loader], 0)


def _concat(models, loader, device, flip):
    return np.concatenate([_pooled(m, loader, device, flip) for m in models], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eids", required=True, help="Comma-separated experiment ids (backbones).")
    ap.add_argument("--knn", type=int, default=20)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--data-dir", default="../data/Challenge_train_data")
    ap.add_argument("--input-size", type=int, default=336)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-folds", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    eids = [e.strip() for e in args.eids.split(",") if e.strip()]
    tag = args.tag or "+".join(eids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, class_names = build_dataset_dataframe(args.data_dir)
    n_classes = len(class_names)
    eval_tf = build_eval_transform(args.input_size)

    all_t, all_s = [], []
    for fold in range(args.num_folds):
        ns = SimpleNamespace(loco=True, fold_index=fold, num_folds=args.num_folds, seed=args.seed)
        train_df, val_df, holdout = split_dataframe(df, ns)
        mk = lambda d: DataLoader(SimpleDataset(d, eval_tf), batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers,
                                  pin_memory=True, worker_init_fn=seed_worker)
        train_loader, val_loader = mk(train_df), mk(val_df)

        models = [_load_encoder(_encoder_path(e, fold), n_classes, args.input_size, device)
                  for e in eids]
        Xtr = _concat(models, train_loader, device, flip=False)
        ytr = _labels(train_loader)
        knn = KNeighborsClassifier(n_neighbors=min(args.knn, len(ytr)))
        knn.fit(Xtr, ytr)
        pos = list(knn.classes_).index(1)
        # probability-level h-flip TTA, matching the finetune's collect_scores
        s_id = knn.predict_proba(_concat(models, val_loader, device, flip=False))[:, pos]
        s_fl = knn.predict_proba(_concat(models, val_loader, device, flip=True))[:, pos]
        scores = 0.5 * (s_id + s_fl)
        yval = _labels(val_loader)
        m = compute_group_eval_metrics(yval, scores)
        print(f"  fold {fold} (holdout {holdout}) | PPV@90R={m['PPV@90RECALL']:.4f} "
              f"AUROC={m['AUROC']:.4f} AUPRC={m['AUPRC']:.4f}")
        all_t.extend(np.asarray(yval, int).tolist())
        all_s.extend(np.asarray(scores, float).tolist())

    pooled = compute_group_eval_metrics(all_t, all_s)
    print(f"=== CONCAT [{tag}] k={args.knn} | pooled LOCO ===")
    print(f"  PPV@90R={pooled['PPV@90RECALL']:.4f}  AUROC={pooled['AUROC']:.4f}  "
          f"AUPRC={pooled['AUPRC']:.4f}")

    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(project=os.environ.get("WANDB_PROJECT", "rare26"),
                             entity=os.environ.get("WANDB_ENTITY"),
                             group="multibackbone",
                             name=f"concat_{tag}_k{args.knn}",
                             job_type="ensemble",
                             config={"run_type": "ensemble_concat", "backbones": eids,
                                     "knn": args.knn, "n_models": len(eids) * args.num_folds})
            for k, v in pooled.items():
                wandb.summary[f"pooled/{k}"] = v
            wandb.log({f"pooled/{k}": v for k, v in pooled.items()})
            run.finish()
            print("logged -> W&B (group=multibackbone)")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] wandb logging skipped: {e}")


if __name__ == "__main__":
    main()
