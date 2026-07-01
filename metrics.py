"""Validation / ensemble metrics + W&B logging for the clean baseline.

Only what train.py and the ensemble path need:
  - PPV at fixed recall (the operating-point metric we report)
  - the threshold that achieves that recall
  - AUROC, AUPRC
  - score collection from a DataLoader with horizontal-flip TTA
"""

from collections import OrderedDict

import numpy as np
import torch
import wandb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    silhouette_score,
)


def _safe_div(numerator, denominator):
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _to_numpy(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")
    return y_true, y_score


def compute_ppv_at_recall_target(y_true, y_score, recall_target=0.90):
    y_true, y_score = _to_numpy(y_true, y_score)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    idx = np.where(recall >= recall_target)[0]
    if len(idx) == 0:
        return float("nan")
    return float(precision[idx[-1]])


def select_highest_threshold_for_target_recall(y_true, y_score, recall_target=0.90):
    """Highest score threshold such that recall >= recall_target."""
    y_true, y_score = _to_numpy(y_true, y_score)
    if len(y_true) == 0:
        return float("nan")
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return float("nan")

    ranked_indices = np.argsort(-y_score, kind="mergesort")
    ranked_scores = y_score[ranked_indices]
    ranked_labels = y_true[ranked_indices]

    tp = 0
    index = 0
    while index < len(ranked_scores):
        threshold = ranked_scores[index]
        while index < len(ranked_scores) and ranked_scores[index] == threshold:
            if ranked_labels[index] == 1:
                tp += 1
            index += 1
        recall = tp / positives
        if recall >= recall_target:
            return float(threshold)

    min_score = float(np.min(ranked_scores))
    if np.isfinite(min_score):
        return float(np.nextafter(min_score, -np.inf))
    return float("nan")


def compute_operating_metrics(y_true, y_score, threshold):
    y_true, y_score = _to_numpy(y_true, y_score)
    nan = float("nan")
    metrics = OrderedDict([
        ("Threshold", float(threshold) if np.isfinite(threshold) else nan),
        ("PPV", nan),
        ("TPR", nan),
        ("TNR", nan),
        ("FPR", nan),
        ("NPV", nan),
        ("F1", nan),
        ("Alert Rate", nan),
    ])

    if len(y_true) == 0 or not np.isfinite(threshold):
        return metrics

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = len(y_true)

    metrics.update({
        "PPV": _safe_div(tp, tp + fp),
        "TPR": _safe_div(tp, tp + fn),
        "TNR": _safe_div(tn, tn + fp),
        "FPR": _safe_div(fp, fp + tn),
        "NPV": _safe_div(tn, tn + fn),
        "F1": _safe_div(2 * tp, (2 * tp) + fp + fn),
        "Alert Rate": _safe_div(tp + fp, total),
    })
    return metrics


def ppv_at_prevalence(tpr, fpr, prevalence):
    """Rescale a fixed (TPR, FPR) operating point to a different positive
    prevalence: PPV(pi) = TPR*pi / (TPR*pi + FPR*(1-pi)). Lets us read PPV at
    the ~1% deployment prevalence from the dataset's natural (much higher) one,
    with no retraining -- see prevalence_analysis.py for the sweep this mirrors.
    """
    if not (np.isfinite(tpr) and np.isfinite(fpr)):
        return float("nan")
    denom = tpr * prevalence + fpr * (1 - prevalence)
    return float(tpr * prevalence / denom) if denom > 0 else float("nan")


def compute_group_eval_metrics(y_true, y_score, recall_target=0.90, threshold=None, target_prevalence=0.01):
    """Returns AUROC/AUPRC/PPV@90R + the operating-point metrics at the chosen threshold.

    Also reports PPV@90RECALL@<target_prevalence>PREV: the PPV@90R operating
    point (TPR/FPR) rescaled to target_prevalence, since raw PPV@90RECALL is
    computed at this dataset's (much higher) natural prevalence and is not
    representative of the ~1% deployment target.
    """
    y_true, y_score = _to_numpy(y_true, y_score)
    nan = float("nan")

    metrics = OrderedDict([
        ("PPV@90RECALL", nan),
        (f"PPV@90RECALL@{target_prevalence:g}PREV", nan),
        ("AUROC", nan),
        ("AUPRC", nan),
        ("Total Samples", int(len(y_true))),
        ("Positive Samples (label 1)", int(np.sum(y_true == 1))),
        ("Negative Samples (label 0)", int(np.sum(y_true == 0))),
    ])

    if len(y_true) == 0:
        return metrics

    if len(np.unique(y_true)) >= 2:
        metrics["PPV@90RECALL"] = compute_ppv_at_recall_target(y_true, y_score, recall_target)
        metrics["AUROC"] = float(roc_auc_score(y_true, y_score))
        metrics["AUPRC"] = float(average_precision_score(y_true, y_score))

    if threshold is None:
        threshold = select_highest_threshold_for_target_recall(y_true, y_score, recall_target)
    op = compute_operating_metrics(y_true, y_score, threshold)
    for key, value in op.items():
        metrics[key] = value

    metrics[f"PPV@90RECALL@{target_prevalence:g}PREV"] = ppv_at_prevalence(
        metrics["TPR"], metrics["FPR"], target_prevalence)
    return metrics


def compute_separation_metrics(features, labels, max_samples=2000, seed=0):
    """How well the two classes separate in the pooled feature space the KNN/SVM
    head classifies in. Raw (unnormalised) features with euclidean geometry, to
    match the KNN's actual decision rule.

    Returns (higher = better separated):
      sep/silhouette       silhouette score (euclidean), in [-1, 1]
      sep/fisher_ratio     between-class scatter / within-class scatter
      sep/sep_ratio        inter-class / intra-class euclidean distance (~CAD/SAD)
    """
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    nan = float("nan")
    out = OrderedDict([
        ("sep/silhouette", nan),
        ("sep/fisher_ratio", nan),
        ("sep/sep_ratio", nan),
    ])
    if X.ndim != 2 or len(y) != len(X) or len(np.unique(y)) < 2:
        return out
    if min(np.bincount(y, minlength=2)[:2]) < 2:  # need >=2 per class
        return out

    # ponytail: silhouette is O(n^2); subsample above max_samples (stratified-ish).
    if len(X) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, y = X[idx], y[idx]

    mu = {c: X[y == c].mean(axis=0) for c in (0, 1)}
    # within-class scatter = mean squared euclidean distance to own centroid
    within = {c: float(np.mean(np.sum((X[y == c] - mu[c]) ** 2, axis=1))) for c in (0, 1)}
    between = float(np.sum((mu[0] - mu[1]) ** 2))
    sw = within[0] + within[1]

    intra = {c: float(np.mean(np.linalg.norm(X[y == c] - mu[c], axis=1))) for c in (0, 1)}
    intra_eucl = 0.5 * (intra[0] + intra[1])
    centroid_eucl = float(np.linalg.norm(mu[0] - mu[1]))

    out["sep/silhouette"] = float(silhouette_score(X, y, metric="euclidean"))
    out["sep/fisher_ratio"] = _safe_div(between, sw)
    out["sep/sep_ratio"] = _safe_div(centroid_eucl, intra_eucl)
    return out


@torch.no_grad()
def collect_scores(model, loader, device, tta=True):
    """(y_true, y_score) where y_score is the positive-class probability.

    With ``tta=True`` averages identity + horizontal-flip softmax probabilities.
    Endoscopy frames are L/R-symmetric, so this is a safe and cheap TTA.
    """
    y_true, y_score = [], []
    model.eval()
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]
        if tta:
            flipped = torch.flip(images, dims=[-1])
            logits_flip = model(flipped)
            probs = 0.5 * (probs + torch.softmax(logits_flip, dim=1)[:, 1])
        y_score.extend(probs.detach().cpu().tolist())
        y_true.extend(labels.detach().cpu().tolist())
    return y_true, y_score


def log_val_metrics(epoch, optimizer, train_loss, valid_loss, valid_metrics, extra=None):
    """Single log point per epoch. Test/ensemble metrics are logged separately
    after head-fit / ensemble-build in train.py."""
    learning_rate = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
    payload = OrderedDict([
        ("epoch", epoch + 1),
        ("learning_rate", learning_rate),
        ("train_loss", train_loss),
        ("valid_loss", valid_loss),
        ("val/Threshold", valid_metrics.get("Threshold", float("nan"))),
        ("val/PPV@90RECALL", valid_metrics.get("PPV@90RECALL", float("nan"))),
        ("val/AUROC", valid_metrics.get("AUROC", float("nan"))),
        ("val/AUPRC", valid_metrics.get("AUPRC", float("nan"))),
        ("val/PPV", valid_metrics.get("PPV", float("nan"))),
        ("val/TPR", valid_metrics.get("TPR", float("nan"))),
        ("val/FPR", valid_metrics.get("FPR", float("nan"))),
    ])
    if extra:
        payload.update(extra)
    wandb.log(payload)


def _demo():
    """Separated classes must score better than overlapping ones."""
    rng = np.random.default_rng(0)
    y = np.array([0] * 100 + [1] * 100)
    far = np.vstack([rng.normal(0, 0.1, (100, 8)) + 5, rng.normal(0, 0.1, (100, 8)) - 5])
    near = rng.normal(0, 1.0, (200, 8))  # both classes drawn from the same blob
    s_far = compute_separation_metrics(far, y)
    s_near = compute_separation_metrics(near, y)
    assert s_far["sep/silhouette"] > s_near["sep/silhouette"]
    assert s_far["sep/fisher_ratio"] > s_near["sep/fisher_ratio"]
    assert s_far["sep/sep_ratio"] > s_near["sep/sep_ratio"]

    # PPV should shrink as prevalence drops, and match the natural-prevalence PPV at pi=P/(P+N).
    assert ppv_at_prevalence(0.9, 0.05, 0.5) > ppv_at_prevalence(0.9, 0.05, 0.01)
    natural = ppv_at_prevalence(0.9, 0.05, 100 / (100 + 100))
    assert abs(natural - (0.9 * 100) / (0.9 * 100 + 0.05 * 100)) < 1e-9
    # degenerate inputs return NaNs, not crashes
    bad = compute_separation_metrics(np.zeros((3, 4)), np.array([0, 0, 0]))
    assert all(np.isnan(v) for v in bad.values())
    print("ok:", {k: round(v, 3) for k, v in s_far.items()})


if __name__ == "__main__":
    _demo()
