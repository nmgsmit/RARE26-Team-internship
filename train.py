import os
import inspect
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from argparse import ArgumentParser
from torch.optim import AdamW
<<<<<<< Updated upstream
<<<<<<< Updated upstream
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Subset
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
=======

from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, f1_score, recall_score, accuracy_score, confusion_matrix
from metrics import compute_group_eval_metrics, collect_scores, log_metrics
>>>>>>> Stashed changes
=======

from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, f1_score, recall_score, accuracy_score, confusion_matrix
from metrics import compute_group_eval_metrics, collect_scores, log_metrics
>>>>>>> Stashed changes
import wandb

from data import prepare_datasets
from testdata import load_external_testset
 
# Dataset structure:
# data/
# ├── center_1/
# │   ├── ndbe/  
# │   └── neo/  
# └── center_2/
#     ├── ndbe/
#     └── neo/ 
### 

def get_args_parser():
    parser = ArgumentParser("RARE25 Classification Training")
    # Change the default path to match your folder name!
    parser.add_argument("--data-dir", type=str, default="./data", help="Where you put center_1, center_2, etc.")
    parser.add_argument("--DatasetSplit", type=int, default=80, help="Percentage of images for training (rest for validation)")
    parser.add_argument("--batch-size", type=int, default=32, help="How many images to look at once")
    parser.add_argument("--epochs", type=int, default=20, help="How many times to loop over the whole dataset")
    parser.add_argument("--lr", type=float, default=1e-4, help="How fast the model 'learns'")
    parser.add_argument("--num-workers", type=int, default=4, help="CPU power for loading images")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="rare25-test-run")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Where to save the trained model")
    parser.add_argument("--centers", nargs="+", default=None, help="Optional list of centers to use, e.g. --centers center_1")
    parser.add_argument("--backbone-name", type=str, default="vit_base_patch16_dinov3", help="timm DinoV3 backbone name")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained DinoV3 weights")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained", help="Disable pretrained DinoV3 weights")
    parser.add_argument(
        "--testset-images-dir",
        type=str,
        default="./data/EVC_Barretts_FullSet/images",
        help="Path to external testset images used for per-epoch testset metrics",
    )
    parser.add_argument(
        "--debug-center1-balanced",
        action="store_true",
        help="Use only center_1 and cap both classes to --debug-class-count samples for quick sanity checks",
    )
    parser.add_argument(
        "--debug-class-count",
        type=int,
        default=61,
        help="Number of samples per class used when --debug-center1-balanced is enabled",
    )
<<<<<<< Updated upstream
<<<<<<< Updated upstream
    parser.set_defaults(pretrained=True)
    return parser


def compute_group_eval_metrics(y_true, y_score, recall_target=0.90):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    valid_points = recall >= recall_target
    if np.any(valid_points):
        ppv_at_recall = float(np.max(precision[valid_points]))
    else:
        ppv_at_recall = float("nan")

    return float(auroc), float(auprc), ppv_at_recall


def infer_testset_label_from_filename(image_path):
    stem = image_path.stem.upper()
    if stem.endswith("_ACHD"):
        return 1
    if stem.endswith("_NDBT"):
        return 0
    raise ValueError(
        f"Could not infer class from filename '{image_path.name}'. "
        "Expected suffix _ACHD or _NDBT before extension."
    )


class ExternalTestsetDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
        self.labels = [infer_testset_label_from_filename(p) for p in image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), self.labels[idx]


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


def build_model_compat(Model, args, n_classes):
    sig = inspect.signature(Model.__init__)
    params = sig.parameters
=======
=======




>>>>>>> Stashed changes




<<<<<<< Updated upstream
>>>>>>> Stashed changes




=======
>>>>>>> Stashed changes

def main(args):
    # Log into Weights & Biases so we can see the graphs later
    wandb.init(project="RARE25-Project", name=args.experiment_id, config=vars(args))
    
    # Setup directories and devices
    os.makedirs(args.save_dir, exist_ok=True) # Ensure save directory exists
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_loader, valid_loader, train_datasets, valid_datasets = prepare_datasets(args, device)
    testset_loader, testset_ds, testset_image_paths = load_external_testset(
        args.testset_images_dir, args.batch_size, args.num_workers, device
    )
    print(f"Using testset images from {args.testset_images_dir} ({len(testset_image_paths)} samples)")
    
    # MODEL SETUP ----------------------------------------------------------------------------------------------------------
    from model import Model

    class_names = train_datasets[0].dataset.classes
    n_classes = len(class_names)

    model = Model(in_channels=3, n_classes=n_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # TRAINING LOOP --------------------------------------------------------------------------------------------------------
    best_valid_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_loader, valid_loader, testset_loader, train_datasets, valid_datasets, testset_ds, testset_image_paths = prepare_datasets(args, device)
            print(f"Using testset images from {args.testset_images_dir} ({len(testset_image_paths)} samples)")

<<<<<<< Updated upstream
<<<<<<< Updated upstream
        avg_train_loss = train_loss / train_total
        train_accuracy = train_correct / train_total

        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        valid_targets = []
        valid_scores = []

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                valid_loss += loss.item() * images.size(0)
                predictions = torch.argmax(outputs, dim=1)
                valid_correct += (predictions == labels).sum().item()
                valid_total += labels.size(0)

                probs = torch.softmax(outputs, dim=1)[:, 1]
                valid_scores.extend(probs.detach().cpu().tolist())
                valid_targets.extend(labels.detach().cpu().tolist())

        avg_valid_loss = valid_loss / valid_total
        valid_accuracy = valid_correct / valid_total
        valid_auroc, valid_auprc, valid_ppv_at_90_recall = compute_group_eval_metrics(valid_targets, valid_scores)
        test_targets, test_scores = collect_scores(model, testset_loader, device)
        test_auroc, test_auprc, test_ppv_at_90_recall = compute_group_eval_metrics(test_targets, test_scores)
        test_predictions = (np.asarray(test_scores) >= 0.5).astype(int).tolist()
        test_confusion_matrix = wandb.plot.confusion_matrix(
            probs=None,
            y_true=test_targets,
            preds=test_predictions,
            class_names=class_names,
        )

        wandb.log({
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "valid_loss": avg_valid_loss,
            "valid_accuracy": valid_accuracy,
            "epoch (validation)": epoch + 1,
            "learning rate (validation)": optimizer.param_groups[0]["lr"],
            "train loss (validation)": avg_train_loss,
            "valid loss (validation)": avg_valid_loss,
            "AUPRC (validation)": valid_auprc,
            "AUROC (validation)": valid_auroc,
            "PPV@90RECALL (validation)": valid_ppv_at_90_recall,
            "epoch (test)": epoch + 1,
            "learning rate (test)": optimizer.param_groups[0]["lr"],
            "train loss (test)": avg_train_loss,
            "valid loss (test)": avg_valid_loss,
            "AUPRC (test)": test_auprc,
            "AUROC (test)": test_auroc,
            "PPV@90RECALL (test)": test_ppv_at_90_recall,
            "confusion matrix (test)": test_confusion_matrix,
        })

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {avg_valid_loss:.4f} | Val Acc: {valid_accuracy:.4f} | "
            f"Val AUPRC: {valid_auprc:.4f} | Val AUROC: {valid_auroc:.4f} | Val PPV@90R: {valid_ppv_at_90_recall:.4f} | "
            f"Test AUPRC: {test_auprc:.4f} | Test AUROC: {test_auroc:.4f} | Test PPV@90R: {test_ppv_at_90_recall:.4f}"
        )

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            save_path = os.path.join(args.save_dir, f"{args.experiment_id}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"   -> Saved new best model to {save_path}")

    final_save_path = os.path.join(args.save_dir, f"{args.experiment_id}_final.pt")
    torch.save(model.state_dict(), final_save_path)

    print(f"Class mapping: {class_names}")
    print("Training finished! Check your WandB dashboard.")
    wandb.finish()

if __name__ == "__main__":
    main(get_args_parser().parse_args())
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
