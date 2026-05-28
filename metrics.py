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


def compute_group_eval_metrics(y_true, y_score, recall_target=0.90, threshold=None):
    """Returns AUROC/AUPRC/PPV@90R + the operating-point metrics at the chosen threshold."""
    y_true, y_score = _to_numpy(y_true, y_score)
    nan = float("nan")

    metrics = OrderedDict([
        ("PPV@90RECALL", nan),
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
    return metrics


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
