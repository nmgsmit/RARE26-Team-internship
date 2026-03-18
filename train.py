import os
import inspect
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
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
