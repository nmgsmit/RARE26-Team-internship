import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, f1_score, recall_score, accuracy_score, confusion_matrix
import wandb
import torch

def compute_group_eval_metrics(y_true, y_score, recall_target=0.90):
    nan = float("nan")
    try:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        if len(y_true) == 0 or len(y_score) == 0 or len(np.unique(y_true)) < 2:
            return nan, nan, nan, nan, nan, nan, nan, nan

        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)

        y_pred = (y_score >= 0.5).astype(int)
        f1 = f1_score(y_true, y_pred)
        sensitivity = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else nan
        total_samples = len(y_true)

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        idx = np.where(recall >= recall_target)[0]
        if len(idx) > 0:
            ppv_at_recall = float(precision[idx[-1]])
        else:
            ppv_at_recall = nan

        return float(auroc), float(auprc), ppv_at_recall, f1, specificity, sensitivity, accuracy, total_samples
    except Exception:
        return nan, nan, nan, nan, nan, nan, nan, nan

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

def log_metrics(epoch, optimizer, avg_train_loss, train_accuracy, avg_valid_loss, valid_accuracy,
                valid_auprc, valid_auroc, valid_ppv_at_90_recall, valid_f1, valid_specificity, valid_sensitivity, valid_total,
                test_auprc, test_auroc, test_ppv_at_90_recall, test_f1, test_specificity, test_sensitivity, test_accuracy, test_total):
    wandb.log({
        "epoch": epoch + 1,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "train_loss": avg_train_loss,
        "train_accuracy": train_accuracy,
        "valid_loss": avg_valid_loss,
        "valid_accuracy": valid_accuracy,
        "val/epoch": epoch + 1,
        "val/train_loss": avg_train_loss,
        "val/valid_loss": avg_valid_loss,
        "val/AUPRC": valid_auprc,
        "val/AUROC": valid_auroc,
        "val/PPV@90RECALL": valid_ppv_at_90_recall,
        "val/F1": valid_f1,
        "val/Specificity": valid_specificity,
        "val/Sensitivity": valid_sensitivity,
        "val/Accuracy": valid_accuracy,
        "val/TotalSamples": valid_total,
        "test/epoch": epoch + 1,
        "test/train_loss": avg_train_loss,
        "test/valid_loss": avg_valid_loss,
        "test/AUPRC": test_auprc,
        "test/AUROC": test_auroc,
        "test/PPV@90RECALL": test_ppv_at_90_recall,
        "test/F1": test_f1,
        "test/Specificity": test_specificity,
        "test/Sensitivity": test_sensitivity,
        "test/Accuracy": test_accuracy,
        "test/TotalSamples": test_total,
    })
