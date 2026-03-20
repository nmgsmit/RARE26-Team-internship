import os
import torch
import torch.nn as nn
from argparse import ArgumentParser
from torch.optim import AdamW
from metrics import compute_group_eval_metrics, collect_scores, log_metrics
from data import prepare_datasets
from testdata import load_external_testset
import wandb

# Dataset structure:
# data/
# ├── center_1/
# │   ├── ndbe/  
# │   └── neo/  
# └── center_2/
#     ├── ndbe/
#     └── neo/ 
### 

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def get_args_parser():
    parser = ArgumentParser("RARE25 Classification Training")
    parser.add_argument("--data-dir", type=str, default="./data", help="Where you put center_1, center_2, etc.")
    parser.add_argument("--batch-size", type=int, default=32, help="How many images to look at once")
    parser.add_argument("--epochs", type=int, default=20, help="How many times to loop over the whole dataset")
    parser.add_argument("--lr", type=float, default=1e-4, help="How fast the model 'learns'")
    parser.add_argument("--num-workers", type=int, default=4, help="CPU power for loading images")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="rare25-test-run")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Where to save the trained model")
    parser.add_argument("--backbone-name", type=str, default="vit_base_patch16_dinov3.lvd1689m", help="timm DinoV3 backbone name")
    parser.add_argument(
        "--testset-images-dir",
        type=str,
        default="./data/EVC_Barretts_FullSet/images",
        help="Path to external testset images used for per-epoch testset metrics",
    )
    return parser


def main(args):
    experiment_suffix = "focal_wrs"
    if experiment_suffix not in args.experiment_id:
        args.experiment_id = f"{args.experiment_id}_{experiment_suffix}"

    # Log into Weights & Biases so we can see the graphs later
    wandb.init(project="RARE25-Project", name=args.experiment_id, config=vars(args))
    
    # Setup directories and devices
    os.makedirs(args.save_dir, exist_ok=True) # Ensure save directory exists
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_loader, valid_loader, train_ds, _, class_names = prepare_datasets(args, device)
    testset_loader, _, testset_image_paths = load_external_testset(
        args.testset_images_dir, args.batch_size, args.num_workers, device
    )
    print(f"Using testset images from {args.testset_images_dir} ({len(testset_image_paths)} samples)")
    
    # MODEL SETUP ----------------------------------------------------------------------------------------------------------
    from model import Model

    n_classes = len(class_names)
    model = Model(
        in_channels=3,
        n_classes=n_classes,
        backbone_name=args.backbone_name,
    ).to(device)

    train_labels = torch.tensor(train_ds.df["label"].tolist(), dtype=torch.long)
    class_counts = torch.bincount(train_labels, minlength=n_classes).float()
    if torch.any(class_counts == 0):
        raise ValueError(f"At least one class has zero training samples: {class_counts.tolist()}")
    class_weights = class_counts.sum() / (n_classes * class_counts)
    class_weights = class_weights.to(device)
    print(f"Using focal loss with class weights: {class_weights.tolist()}")

    criterion = FocalLoss(alpha=class_weights)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # TRAINING LOOP --------------------------------------------------------------------------------------------------------
    best_valid_ppv_at_90_recall = float('-inf')

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
            predictions = torch.argmax(outputs, dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)


        if train_total == 0:
            raise ValueError(
                "Training loader produced zero samples. Check the dataset split, filters, and batch configuration."
            )
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

        if valid_total == 0:
            raise ValueError(
                "Validation loader produced zero samples. Check the dataset split, filters, and batch configuration."
            )
        avg_valid_loss = valid_loss / valid_total
        valid_accuracy = valid_correct / valid_total
        (
            valid_auroc,
            valid_auprc,
            valid_ppv_at_90_recall,
            valid_f1,
            valid_specificity,
            valid_sensitivity,
            _valid_metric_accuracy,
            _valid_metric_total,
        ) = compute_group_eval_metrics(valid_targets, valid_scores)
        test_targets, test_scores = collect_scores(model, testset_loader, device)
        (
            test_auroc,
            test_auprc,
            test_ppv_at_90_recall,
            test_f1,
            test_specificity,
            test_sensitivity,
            test_accuracy,
            test_total,
        ) = compute_group_eval_metrics(test_targets, test_scores)

    
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {avg_valid_loss:.4f} | Val Acc: {valid_accuracy:.4f} | "
            f"Val AUPRC: {valid_auprc:.4f} | Val AUROC: {valid_auroc:.4f} | Val PPV@90R: {valid_ppv_at_90_recall:.4f} | "
            f"Test AUPRC: {test_auprc:.4f} | Test AUROC: {test_auroc:.4f} | Test PPV@90R: {test_ppv_at_90_recall:.4f}"
        )
        log_metrics(
            epoch,
            optimizer,
            avg_train_loss,
            train_accuracy,
            avg_valid_loss,
            valid_accuracy,
            valid_auprc,
            valid_auroc,
            valid_ppv_at_90_recall,
            valid_f1,
            valid_specificity,
            valid_sensitivity,
            valid_total,
            test_auprc,
            test_auroc,
            test_ppv_at_90_recall,
            test_f1,
            test_specificity,
            test_sensitivity,
            test_accuracy,
            test_total,
        )

        if valid_ppv_at_90_recall > best_valid_ppv_at_90_recall:
            best_valid_ppv_at_90_recall = valid_ppv_at_90_recall
            save_path = os.path.join(args.save_dir, f"{args.experiment_id}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"   -> Saved new best model to {save_path} (PPV@90Recall: {valid_ppv_at_90_recall:.4f})")

    final_save_path = os.path.join(args.save_dir, f"{args.experiment_id}_final.pt")
    torch.save(model.state_dict(), final_save_path)

    print(f"Class mapping: {class_names}")
    print("Training finished! Check your WandB dashboard.")
    wandb.finish()

if __name__ == "__main__":
    main(get_args_parser().parse_args())
