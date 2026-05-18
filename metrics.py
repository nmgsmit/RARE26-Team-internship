from collections import OrderedDict

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score
import torch
import wandb


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


# train.py uses this threshold to define the epoch operating point. Changing it changes both the
# logged validation/test metrics and the best-checkpoint policy.
def select_highest_threshold_for_target_recall(y_true, y_score, recall_target=0.90):
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
        ("Alert Rate", nan),
        ("FPR", nan),
        ("NPV", nan),
        ("TPR", nan),
        ("TNR", nan),
        ("PPV", nan),
        ("Threshold", float(threshold) if np.isfinite(threshold) else nan),
        ("F1", nan),
    ])

    if len(y_true) == 0 or not np.isfinite(threshold):
        return metrics

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_samples = len(y_true)

    ppv = _safe_div(tp, tp + fp)
    tpr = _safe_div(tp, tp + fn)
    tnr = _safe_div(tn, tn + fp)
    fpr = _safe_div(fp, fp + tn)
    npv = _safe_div(tn, tn + fn)
    alert_rate = _safe_div(tp + fp, total_samples)
    f1 = _safe_div(2 * tp, (2 * tp) + fp + fn)

    metrics.update({
        "Alert Rate": alert_rate,
        "FPR": fpr,
        "NPV": npv,
        "TPR": tpr,
        "TNR": tnr,
        "PPV": ppv,
        "F1": f1,
    })
    return metrics


# This is the only place where the 1% deployment assumption is applied. Keep train.py at
# prevalence=0.01 if you want identical projected metrics, W&B curves, and checkpoint ranking.
def project_operating_metrics_to_prevalence(metrics, prevalence=0.01, population_size=1000):
    prevalence = float(prevalence)
    population_size = float(population_size)
    nan = float("nan")
    projected = OrderedDict([
        ("Target Prevalence", prevalence),
        ("Projected PPV", nan),
        ("Projected NPV", nan),
        ("Projected Alert Rate", nan),
        ("Projected TP per 1000", nan),
        ("Projected FP per 1000", nan),
        ("Projected FN per 1000", nan),
        ("Projected TN per 1000", nan),
        ("Threshold", nan),
        ("TPR", nan),
        ("FPR", nan),
        ("TNR", nan),
    ])

    if prevalence < 0.0 or prevalence > 1.0:
        raise ValueError(f"Prevalence must be between 0 and 1, got {prevalence}.")

    tpr = float(metrics.get("TPR", nan))
    fpr = float(metrics.get("FPR", nan))
    tnr = float(metrics.get("TNR", nan))
    threshold = float(metrics.get("Threshold", nan))

    projected["Threshold"] = threshold if np.isfinite(threshold) else nan
    projected["TPR"] = tpr
    projected["FPR"] = fpr
    projected["TNR"] = tnr

    if not (np.isfinite(tpr) and np.isfinite(fpr) and np.isfinite(tnr)):
        return projected

    negative_prevalence = 1.0 - prevalence
    fnr = 1.0 - tpr

    tp_rate = tpr * prevalence
    fp_rate = fpr * negative_prevalence
    fn_rate = fnr * prevalence
    tn_rate = tnr * negative_prevalence

    projected.update({
        "Projected PPV": _safe_div(tp_rate, tp_rate + fp_rate),
        "Projected NPV": _safe_div(tn_rate, tn_rate + fn_rate),
        "Projected Alert Rate": tp_rate + fp_rate,
        "Projected TP per 1000": tp_rate * population_size,
        "Projected FP per 1000": fp_rate * population_size,
        "Projected FN per 1000": fn_rate * population_size,
        "Projected TN per 1000": tn_rate * population_size,
    })
    return projected


def compute_group_eval_metrics(y_true, y_score, recall_target=0.90, threshold=None):
    y_true, y_score = _to_numpy(y_true, y_score)
    nan = float("nan")

    metrics = OrderedDict([
        ("PPV@90RECALL", nan),
        ("AUROC", nan),
        ("AUPRC", nan),
        ("Total Samples", int(len(y_true))),
        ("Negative Samples (label 0)", int(np.sum(y_true == 0))),
        ("Positive Samples (label 1)", int(np.sum(y_true == 1))),
        ("Alert Rate", nan),
        ("FPR", nan),
        ("NPV", nan),
        ("TPR", nan),
        ("TNR", nan),
        ("PPV", nan),
        ("Threshold", nan),
        ("F1", nan),
    ])

    if len(y_true) == 0:
        return metrics

    if len(np.unique(y_true)) >= 2:
        metrics["PPV@90RECALL"] = compute_ppv_at_recall_target(y_true, y_score, recall_target=recall_target)
        metrics["AUROC"] = float(roc_auc_score(y_true, y_score))
        metrics["AUPRC"] = float(average_precision_score(y_true, y_score))

    if threshold is None:
        threshold = select_highest_threshold_for_target_recall(y_true, y_score, recall_target=recall_target)

    operating_metrics = compute_operating_metrics(y_true, y_score, threshold)
    for key, value in operating_metrics.items():
        metrics[key] = value

    return metrics


def collect_scores(model, loader, device):
    y_true = []
    y_score = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs_neo = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
            y_score.extend(probs_neo)
            y_true.extend(labels.detach().cpu().tolist())
    return y_true, y_score


def compute_binary_dice_score(pred_mask, target_mask, eps=1e-8):
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.detach().cpu().numpy()
    if torch.is_tensor(target_mask):
        target_mask = target_mask.detach().cpu().numpy()

    pred_mask = np.asarray(pred_mask).astype(bool)
    target_mask = np.asarray(target_mask).astype(bool)
    if pred_mask.shape != target_mask.shape:
        raise ValueError(
            f"Dice masks must share the same shape, got {pred_mask.shape} and {target_mask.shape}."
        )

    intersection = float(np.logical_and(pred_mask, target_mask).sum())
    denominator = float(pred_mask.sum() + target_mask.sum())
    if denominator == 0.0:
        return 1.0
    return float((2.0 * intersection + eps) / (denominator + eps))


def compute_batch_binary_dice_scores(pred_masks, target_masks, ignore_empty_targets=True, eps=1e-8):
    if torch.is_tensor(pred_masks):
        pred_masks = pred_masks.detach().cpu().numpy()
    if torch.is_tensor(target_masks):
        target_masks = target_masks.detach().cpu().numpy()

    pred_masks = np.asarray(pred_masks)
    target_masks = np.asarray(target_masks)
    if pred_masks.shape != target_masks.shape:
        raise ValueError(
            f"Dice mask batches must share the same shape, got {pred_masks.shape} and {target_masks.shape}."
        )

    scores = []
    skipped = 0
    for pred_mask, target_mask in zip(pred_masks, target_masks):
        if ignore_empty_targets and not np.any(target_mask):
            scores.append(float("nan"))
            skipped += 1
            continue
        scores.append(compute_binary_dice_score(pred_mask, target_mask, eps=eps))
    return scores, skipped


def log_metrics(
    epoch,
    optimizer,
    avg_train_loss,
    train_accuracy,
    avg_valid_loss,
    valid_accuracy,
    valid_metrics,
    test_metrics,
    val_projected_metrics,
    test_projected_metrics,
    extra_payload=None,
):
    learning_rate = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
    # Keep these keys stable if you want W&B dashboards from main and test-model to stay directly comparable.
    payload = OrderedDict([
        ("epoch", epoch + 1),
        ("learning_rate", learning_rate),
        ("train_loss", avg_train_loss),
        ("valid_loss", avg_valid_loss),
        ("selected_treshold", valid_metrics["Threshold"]),
        ("val/Positive Samples", valid_metrics["Positive Samples (label 1)"]),
        ("val/Negative Samples", valid_metrics["Negative Samples (label 0)"]),
        ("val/PPV", valid_metrics["PPV"]),
        ("val/AUROC", valid_metrics["AUROC"]),
        ("val/AUPRC", valid_metrics["AUPRC"]),
        ("val/TPR (Recall)", valid_metrics["TPR"]),
        ("val/FPR (False Alarm Rate)", valid_metrics["FPR"]),
        ("1%val/Projected PPV", val_projected_metrics["Projected PPV"]),
        ("1%val/Projected Alert Rate", val_projected_metrics["Projected Alert Rate"]),
        ("1%val/Projected TP per 1000", val_projected_metrics["Projected TP per 1000"]),
        ("1%val/Projected FP per 1000", val_projected_metrics["Projected FP per 1000"]),
        ("1%val/Projected FN per 1000", val_projected_metrics["Projected FN per 1000"]),
        ("test/Positive Samples", test_metrics["Positive Samples (label 1)"]),
        ("test/Negative Samples", test_metrics["Negative Samples (label 0)"]),
        ("test/PPV", test_metrics["PPV"]),
        ("test/AUROC", test_metrics["AUROC"]),
        ("test/AUPRC", test_metrics["AUPRC"]),
        ("test/TPR (Recall)", test_metrics["TPR"]),
        ("test/FPR (False Alarm Rate)", test_metrics["FPR"]),
        ("1%test/Projected PPV", test_projected_metrics["Projected PPV"]),
        ("1%test/Projected Alert Rate", test_projected_metrics["Projected Alert Rate"]),
        ("1%test/Projected TP per 1000", test_projected_metrics["Projected TP per 1000"]),
        ("1%test/Projected FP per 1000", test_projected_metrics["Projected FP per 1000"]),
        ("1%test/Projected FN per 1000", test_projected_metrics["Projected FN per 1000"]),
    ])
    if extra_payload:
        payload.update(extra_payload)
    wandb.log(payload)
