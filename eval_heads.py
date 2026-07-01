"""Continuous-score head ablation on FROZEN pooled features (no backprop, no MLP /
trained nn.Linear). Heads (all sklearn, fit on frozen features):
  - logistic   : LogisticRegression (continuous predict_proba)
  - linsvm     : linear SVC + Platt probability (continuous)
  - dwknn      : distance-weighted KNN (finer-grained than uniform vote fraction)

Continuous scores remove the kNN score-0 ties that (a) cliff PPV@90RECALL by forcing
the threshold to 0 and (b) bias AUPRC via granularity. For each head we evaluate each
single backbone AND the triple late-fusion (avg of per-backbone probabilities), pooled
over LOCO folds. Ranks by PPV@90RECALL (the challenge metric).

Features are L2-row then StandardScaler-standardised per backbone (linear heads are
scale-sensitive). h-flip TTA at the probability level. Needs a GPU.

Usage: python eval_heads.py --backbones dino=EID,simclr=EID,moco=EID --heads logistic,linsvm,dwknn --knn 20
"""
import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from data import (SimpleDataset, build_dataset_dataframe, build_eval_transform,
                  seed_worker, split_dataframe)
from metrics import compute_group_eval_metrics
from model import Model, load_encoder_checkpoint


def _enc_path(eid, fold):
    return f"checkpoints/{eid}/{eid}_pretrain_fold{fold}/{eid}_pretrain_fold{fold}_encoder.pt"


def _load(path, n_classes, input_size, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("model_config", {}))
    cfg.update(n_classes=n_classes, pretrained=False, head_type="knn", knn_neighbors=5)
    cfg.setdefault("input_size", input_size)
    for m in ("backbone_preset", "num_folds", "fold_index", "loss_name", "backbone_weights_path"):
        cfg.pop(m, None)
    model = Model(**cfg).to(device)
    load_encoder_checkpoint(model, path, strict=False)
    model.eval()
    return model


@torch.no_grad()
def _feats(model, loader, device, flip):
    out = []
    for images, _ in loader:
        images = images.to(device)
        if flip:
            images = torch.flip(images, dims=[-1])
        out.append(model.encode(images).cpu().numpy())
    return np.concatenate(out, 0)


def _labels(loader):
    return np.concatenate([y.numpy() for _, y in loader], 0)


def _make_head(name, knn, seed):
    if name == "logistic":
        return LogisticRegression(max_iter=3000, class_weight="balanced")
    if name == "linsvm":
        return SVC(kernel="linear", probability=True, class_weight="balanced", random_state=seed)
    if name == "dwknn":
        return KNeighborsClassifier(n_neighbors=knn, weights="distance")
    raise ValueError(f"unknown head {name}")


def _pos_proba(head, X):
    return head.predict_proba(X)[:, list(head.classes_).index(1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", required=True, help="name=eid,name=eid,...")
    ap.add_argument("--heads", default="logistic,linsvm,dwknn")
    ap.add_argument("--knn", type=int, default=20)
    ap.add_argument("--data-dir", default="../data/Challenge_train_data")
    ap.add_argument("--input-size", type=int, default=336)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-folds", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    names, eids = [], {}
    for tok in args.backbones.split(","):
        nm, eid = tok.split("=", 1)
        names.append(nm.strip()); eids[nm.strip()] = eid.strip()
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, class_names = build_dataset_dataframe(args.data_dir)
    n_classes = len(class_names)
    eval_tf = build_eval_transform(args.input_size)

    # cache[name][fold] = (Xtr, ytr, Xval_id, Xval_flip, yval)  -- standardised
    cache = {n: {} for n in names}
    for fold in range(args.num_folds):
        ns = SimpleNamespace(loco=True, fold_index=fold, num_folds=args.num_folds, seed=args.seed)
        tr_df, val_df, _ = split_dataframe(df, ns)
        mk = lambda d: DataLoader(SimpleDataset(d, eval_tf), batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker)
        tr_l, val_l = mk(tr_df), mk(val_df)
        ytr, yval = _labels(tr_l), _labels(val_l)
        for n in names:
            mdl = _load(_enc_path(eids[n], fold), n_classes, args.input_size, device)
            Xtr = _feats(mdl, tr_l, device, False)
            Xvi = _feats(mdl, val_l, device, False)
            Xvf = _feats(mdl, val_l, device, True)
            sc = StandardScaler().fit(Xtr)
            cache[n][fold] = (sc.transform(Xtr), ytr, sc.transform(Xvi), sc.transform(Xvf), yval)
            del mdl
            torch.cuda.empty_cache()

    results = []
    for head in heads:
        configs = [(n, [n]) for n in names] + [("triple", names)]
        for label, members in configs:
            all_t, all_s = [], []
            for fold in range(args.num_folds):
                probs = []
                for n in members:
                    Xtr, ytr, Xvi, Xvf, yval = cache[n][fold]
                    clf = _make_head(head, args.knn, args.seed).fit(Xtr, ytr)
                    probs.append(0.5 * (_pos_proba(clf, Xvi) + _pos_proba(clf, Xvf)))
                s = np.mean(probs, axis=0)
                all_t.extend(yval.tolist()); all_s.extend(s.tolist())
            m = compute_group_eval_metrics(all_t, all_s)
            results.append((head, label, m))
            print(f"  {head:9s} {label:8s} | PPV@90R={m['PPV@90RECALL']:.4f}  "
                  f"AUROC={m['AUROC']:.4f}  AUPRC={m['AUPRC']:.4f}  thr={m['Threshold']:.4f}")

            if not args.no_wandb:
                try:
                    import wandb
                    run = wandb.init(project=os.environ.get("WANDB_PROJECT", "rare26"),
                                     entity=os.environ.get("WANDB_ENTITY"), group="head-ablation",
                                     name=f"{head}_{label}_k{args.knn}", job_type="head_ablation",
                                     config={"run_type": "head_ablation", "head": head,
                                             "backbones": members, "knn": args.knn})
                    for k, v in m.items():
                        wandb.summary[f"pooled/{k}"] = v
                    wandb.log({f"pooled/{k}": v for k, v in m.items()})
                    run.finish()
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] wandb skipped: {e}")

    print("\n=== RANKED by PPV@90RECALL ===")
    for head, label, m in sorted(results, key=lambda r: -(r[2]["PPV@90RECALL"] or 0)):
        print(f"  {head:9s} {label:8s} | PPV@90R={m['PPV@90RECALL']:.4f}  "
              f"AUROC={m['AUROC']:.4f}  AUPRC={m['AUPRC']:.4f}")


if __name__ == "__main__":
    main()
