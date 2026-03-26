import os
import inspect
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from argparse import ArgumentParser
from torch.optim import AdamW
from metrics import (
    compute_group_eval_metrics,
    collect_scores,
    log_metrics,
    project_operating_metrics_to_prevalence,
)
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
    parser.add_argument("--backbone-name", type=str, default="vit_base_patch16_dinov3", help="timm DinoV3 backbone name")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained DinoV3 weights")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained", help="Disable pretrained DinoV3 weights")
    parser.add_argument(
        "--testset-images-dir",
        type=str,
        default="./data/EVC_Barretts_FullSet/images",
        help="Path to external testset images used for per-epoch testset metrics",
    )

    parser.set_defaults(pretrained=True)
    return parser


def main(args):
    # Log into Weights & Biases so we can see the graphs later
    wandb.init(project="RARE25-Project", name=args.experiment_id, config=vars(args))
    
    # Setup directories and devices
    os.makedirs(args.save_dir, exist_ok=True) # Ensure save directory exists
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_loader, valid_loader, _, _, class_names = prepare_datasets(args, device)
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
        pretrained=args.pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    # Keep this at 0.01 if you want the same projected 1% validation/test metrics as test-model.
    projected_prevalence = 0.01

    # TRAINING LOOP --------------------------------------------------------------------------------------------------------
    best_valid_projected_ppv = float("-inf")
    best_valid_fpr = float("inf")

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
        # Threshold selection happens here: we choose it on the full validation split once per epoch,
        # then reuse that exact threshold for validation/test reporting and the projected 1% metrics.
        valid_metrics = compute_group_eval_metrics(valid_targets, valid_scores)
        valid_threshold = valid_metrics["Threshold"]
        valid_projected_metrics = project_operating_metrics_to_prevalence(
            valid_metrics,
            prevalence=projected_prevalence,
        )
        test_targets, test_scores = collect_scores(model, testset_loader, device)
        test_metrics = compute_group_eval_metrics(test_targets, test_scores, threshold=valid_threshold)
        test_projected_metrics = project_operating_metrics_to_prevalence(
            test_metrics,
            prevalence=projected_prevalence,
        )

    
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {avg_valid_loss:.4f} | Val Acc: {valid_accuracy:.4f} | "
            f"Val AUPRC: {valid_metrics['AUPRC']:.4f} | Val AUROC: {valid_metrics['AUROC']:.4f} | "
            f"Val PPV@90R: {valid_metrics['PPV@90RECALL']:.4f} | Val Thr: {valid_threshold:.4f} | "
            f"Val TPR: {valid_metrics['TPR']:.4f} | Val FPR: {valid_metrics['FPR']:.4f} | Val PPV: {valid_metrics['PPV']:.4f} | "
            f"1%Val PPV: {valid_projected_metrics['Projected PPV']:.4f} | "
            f"1%Val FP/1000: {valid_projected_metrics['Projected FP per 1000']:.2f} | "
            f"Test AUPRC: {test_metrics['AUPRC']:.4f} | Test AUROC: {test_metrics['AUROC']:.4f} | "
            f"Test Thr: {test_metrics['Threshold']:.4f} | Test TPR: {test_metrics['TPR']:.4f} | "
            f"Test FPR: {test_metrics['FPR']:.4f} | Test PPV: {test_metrics['PPV']:.4f} | "
            f"1%Test PPV: {test_projected_metrics['Projected PPV']:.4f} | "
            f"1%Test FP/1000: {test_projected_metrics['Projected FP per 1000']:.2f}"
        )
        # Keep the same metric dictionaries and namespaces here if you want W&B logging to stay aligned with test-model.
        log_metrics(
            epoch,
            optimizer,
            avg_train_loss,
            train_accuracy,
            avg_valid_loss,
            valid_accuracy,
            valid_metrics,
            test_metrics,
            valid_projected_metrics,
            test_projected_metrics,
        )

        current_valid_projected_ppv = (
            valid_projected_metrics["Projected PPV"]
            if np.isfinite(valid_projected_metrics["Projected PPV"])
            else float("-inf")
        )
        current_valid_fpr = (
            valid_projected_metrics["FPR"]
            if np.isfinite(valid_projected_metrics["FPR"])
            else float("inf")
        )
        same_projected_ppv = (
            (
                not np.isfinite(current_valid_projected_ppv)
                and not np.isfinite(best_valid_projected_ppv)
                and current_valid_projected_ppv == best_valid_projected_ppv
            )
            or (
                np.isfinite(current_valid_projected_ppv)
                and np.isfinite(best_valid_projected_ppv)
                and np.isclose(current_valid_projected_ppv, best_valid_projected_ppv)
            )
        )
        # Best-checkpoint selection lives here: maximize projected 1% validation PPV and break ties with lower validation FPR.
        is_better_checkpoint = (
            current_valid_projected_ppv > best_valid_projected_ppv
            or (same_projected_ppv and current_valid_fpr < best_valid_fpr)
        )

        if is_better_checkpoint:
            best_valid_projected_ppv = current_valid_projected_ppv
            best_valid_fpr = current_valid_fpr
            save_path = os.path.join(args.save_dir, f"{args.experiment_id}_best.pt")
            torch.save(model.state_dict(), save_path)
            print(
                f"   -> Saved new best model to {save_path} "
                f"(1% PPV: {valid_projected_metrics['Projected PPV']:.4f}, "
                f"FPR: {valid_metrics['FPR']:.4f}, Threshold: {valid_threshold:.4f})"
            )

    final_save_path = os.path.join(args.save_dir, f"{args.experiment_id}_final.pt")
    torch.save(model.state_dict(), final_save_path)

    print(f"Class mapping: {class_names}")
    print("Training finished! Check your WandB dashboard.")
    wandb.finish()

if __name__ == "__main__":
    main(get_args_parser().parse_args())
