"""Stress-test the best model (logistic + triple fusion) against:
  (1) per-center vs pooled threshold for PPV@90RECALL, and
  (2) prevalence shift -- how PPV@90R holds as positives drop toward ~1/100.

PPV at a fixed operating point (TPR, FPR) is prevalence-dependent:
    PPV(pi) = TPR*pi / (TPR*pi + FPR*(1-pi))
so we read TPR/FPR at the 90%-recall threshold from the held-out scores and
recompute PPV across prevalences. No model retraining needed for the sweep.

Reuses eval_heads feature extraction + logistic head (GPU). Triple = mean of the
three backbones' per-fold probabilities (hflip TTA).
"""
import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from types import SimpleNamespace

from data import (SimpleDataset, build_dataset_dataframe, build_eval_transform,
                  seed_worker, split_dataframe)
from eval_heads import _enc_path, _load, _feats, _labels, _make_head, _pos_proba

# Override with env BACKBONES="name=eid,name=eid,..."
_BB = os.environ.get("BACKBONES",
                     "dino=clean_baseline_crop095_knn5,simclr=simclr_crop095_knn5,moco=moco_crop095_knn5")
BACKBONES = dict(t.split("=", 1) for t in _BB.split(","))
print("BACKBONES:", BACKBONES)
INPUT, BS, NW, SEED = 336, 32, 10, 42


def operating_point(y, s, recall=0.90):
    """Highest-threshold point with TPR>=recall. Returns (thr,TPR,FPR,PPV,P,N)."""
    y = np.asarray(y, int); s = np.asarray(s, float)
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    P, N = int(y.sum()), int((y == 0).sum())
    cum_tp = np.cumsum(ys == 1); cum_fp = np.cumsum(ys == 0)
    k = int(np.argmax((cum_tp / P) >= recall))   # first index reaching target recall
    tp, fp = int(cum_tp[k]), int(cum_fp[k])
    return float(s[order][k]), tp / P, fp / N, tp / (tp + fp), P, N


def ppv_at(tpr, fpr, pi):
    return tpr * pi / (tpr * pi + fpr * (1 - pi)) if (tpr * pi + fpr * (1 - pi)) > 0 else float("nan")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, class_names = build_dataset_dataframe("../data/Challenge_train_data")
    n_classes = len(class_names)
    eval_tf = build_eval_transform(INPUT)

    per_center = {}   # holdout_center -> (y, triple_score)
    for fold in (0, 1):
        ns = SimpleNamespace(loco=True, fold_index=fold, num_folds=2, seed=SEED)
        tr_df, val_df, holdout = split_dataframe(df, ns)
        mk = lambda d: DataLoader(SimpleDataset(d, eval_tf), batch_size=BS, shuffle=False,
                                  num_workers=NW, pin_memory=True, worker_init_fn=seed_worker)
        tr_l, val_l = mk(tr_df), mk(val_df)
        ytr, yval = _labels(tr_l), _labels(val_l)
        probs = []
        for eid in BACKBONES.values():
            mdl = _load(_enc_path(eid, fold), n_classes, INPUT, device)
            sc = StandardScaler().fit(_feats(mdl, tr_l, device, False))
            Xtr = sc.transform(_feats(mdl, tr_l, device, False))   # re-extract train (cheap vs val)
            clf = _make_head("logistic", 20, SEED).fit(Xtr, ytr)
            pid = _pos_proba(clf, sc.transform(_feats(mdl, val_l, device, False)))
            pfl = _pos_proba(clf, sc.transform(_feats(mdl, val_l, device, True)))
            probs.append(0.5 * (pid + pfl))
            del mdl; torch.cuda.empty_cache()
        per_center[holdout] = (yval, np.mean(probs, axis=0))

    # ---- 1) per-center vs pooled threshold ----
    print("\n=== PPV@90RECALL: per-center threshold vs single pooled threshold ===")
    yp = np.concatenate([per_center[c][0] for c in sorted(per_center)])
    sp = np.concatenate([per_center[c][1] for c in sorted(per_center)])
    pth, ptpr, pfpr, pppv, P, N = operating_point(yp, sp)
    print(f"POOLED  thr={pth:.3f}  TPR={ptpr:.3f} FPR={pfpr:.4f}  PPV={pppv:.4f}  (prev={P/(P+N):.4f})")
    tp_sum = fp_sum = 0
    for c in sorted(per_center):
        y, s = per_center[c]
        th, tpr, fpr, ppv, p, n = operating_point(y, s)
        # own-threshold contribution (per-center calibration), summed across centers
        own = s >= th
        tp_sum += int((own & (y == 1)).sum()); fp_sum += int((own & (y == 0)).sum())
        # pooled-threshold PPV on this center (for the comparison column)
        pp = s >= pth
        tpc, fpc = int((pp & (y == 1)).sum()), int((pp & (y == 0)).sum())
        print(f"  {c}: prev={p/(p+n):.4f} | OWN thr={th:.3f} PPV={ppv:.4f} TPR={tpr:.3f} FPR={fpr:.4f} "
              f"| at POOLED thr: PPV={tpc/(tpc+fpc) if tpc+fpc else float('nan'):.4f}")
    print(f"per-center-threshold combined PPV = {tp_sum/(tp_sum+fp_sum):.4f}  vs pooled {pppv:.4f}")

    # ---- 2) prevalence sweep (pooled operating point) ----
    print("\n=== PPV@90RECALL vs prevalence (pooled operating point: TPR=%.3f FPR=%.4f) ===" % (ptpr, pfpr))
    print("  prevalence   PPV@90R")
    for pi in [P/(P+N), 0.05, 0.03, 0.02, 0.01, 0.005]:
        print(f"  1/{1/pi:6.0f} = {pi:.4f}   {ppv_at(ptpr, pfpr, pi):.4f}")
    print("\nper-center FPR@90recall (lower = holds up better at low prevalence):")
    for c in sorted(per_center):
        _, tpr, fpr, _, _, _ = operating_point(*per_center[c])
        print(f"  {c}: FPR={fpr:.4f} -> PPV@1%% = {ppv_at(tpr, fpr, 0.01):.4f}")


if __name__ == "__main__":
    main()
