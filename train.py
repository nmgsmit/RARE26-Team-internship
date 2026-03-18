import os
import inspect
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from argparse import ArgumentParser
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Subset
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
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

    init_kwargs = {}

    if "in_channels" in params:
        init_kwargs["in_channels"] = 3
    elif "in_chans" in params:
        init_kwargs["in_chans"] = 3

    if "n_classes" in params:
        init_kwargs["n_classes"] = n_classes
    elif "num_classes" in params:
        init_kwargs["num_classes"] = n_classes

    if "backbone_name" in params:
        init_kwargs["backbone_name"] = args.backbone_name
    elif "backbone" in params:
        init_kwargs["backbone"] = args.backbone_name

    if "pretrained" in params:
        init_kwargs["pretrained"] = args.pretrained
    elif "use_pretrained" in params:
        init_kwargs["use_pretrained"] = args.pretrained

    try:
        return Model(**init_kwargs)
    except TypeError as exc:
        raise TypeError(
            "Failed to instantiate Model with compatible arguments. "
            f"Detected constructor signature: {sig}. "
            f"Tried kwargs: {sorted(init_kwargs.keys())}"
        ) from exc

def main(args):
    # Log into Weights & Biases so we can see the graphs later
    wandb.init(project="RARE25-Project", name=args.experiment_id, config=vars(args))
    
    # Setup directories and devices
    os.makedirs(args.save_dir, exist_ok=True) # Ensure save directory exists
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DATA ---------------------------------------------------------------------------------------------------------------
    # In the coming part different steps of data loading and preparation are performed.
    # -------------------------------------------------------------------------------------------------------- DATA LOADING 
    # Check if the data is in a folder, if yes then this step will be skipped. Otherwise we download via huggingface.
    # Do note that you need to fill in your personal huggingface token in the .env file for this to work.

    if not os.path.exists(args.data_dir):
        from huggingface_hub import snapshot_download 
        
        print("Data not found locally. Downloading folders from Hugging Face...")

        hf_token = os.getenv("HF_TOKEN")

        if not hf_token:
            env_path = os.path.join(os.path.dirname(__file__), ".env")
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as env_file:
                    for line in env_file:
                        stripped = line.strip()
                        if stripped.startswith("HF_TOKEN="):
                            hf_token = stripped.split("=", 1)[1].strip().strip('"').strip("'")
                            break
        
        if not hf_token:
            raise ValueError("Could not find HF_TOKEN in .env file! Make sure it is set.")

        # Download the repo contents directly into your ./data folder!
        snapshot_download(
            repo_id="TimJaspersTue/RARE25-train", 
            repo_type="dataset",
            local_dir=args.data_dir, 
            token=hf_token      
        )
        print("Data downloaded successfully.")
    # ----------------------------------------------------------------------------------------------------- DATA PREPARATION 
    # Standard: resize to 224x224 (quite standard), we can change it later!
    transform = Compose([
        ToImage(),
        Resize((224, 224)), 
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalization of the colour channels (standard)
    ])

    # <<!-- We can add more transformations here if you want to experiment with data augmentation. -->

    # --------------------------------------------------------------------------------------------- COMBINE DATASETS AND SPLIT
    # ImageFolder will automatically assign labels based on folder names (ndbe=0, neo=1).
    discovered_centers = [
        f for f in os.listdir(args.data_dir)
        if f.startswith('center') and os.path.isdir(os.path.join(args.data_dir, f))
    ]

    if args.centers is not None:
        unknown_centers = sorted(set(args.centers) - set(discovered_centers))
        if unknown_centers:
            raise ValueError(
                f"Requested centers not found: {unknown_centers}. Available centers: {sorted(discovered_centers)}"
            )
        centers = args.centers
    else:
        centers = discovered_centers

    if args.debug_center1_balanced:
        if "center_1" not in discovered_centers:
            raise ValueError(
                f"debug mode requires center_1 in {args.data_dir}. Available centers: {sorted(discovered_centers)}"
            )
        centers = ["center_1"]

    if len(centers) == 0:
        raise ValueError(
            f"No center folders found in {args.data_dir}. Expected folders like 'center_1', 'center_2'."
        )

    print(f"Using centers: {centers}")
    train_datasets = []
    valid_datasets = []
    
    for center in centers:
        center_path = os.path.join(args.data_dir, center)
        ds = ImageFolder(root=center_path, transform=transform)
            
        # SAFETY CHECK: Ensure labels are consistently 0=ndbe, 1=neo across all centers!
        assert ds.class_to_idx == {'ndbe': 0, 'neo': 1}, f"CRITICAL WARNING: Class mapping in {center} is backwards or broken: {ds.class_to_idx}"

        if args.debug_center1_balanced and center == "center_1":
            ndbe_label = ds.class_to_idx["ndbe"]
            neo_label = ds.class_to_idx["neo"]
            ndbe_indices = [idx for idx, label in enumerate(ds.targets) if label == ndbe_label][:args.debug_class_count]
            neo_indices = [idx for idx, label in enumerate(ds.targets) if label == neo_label][:args.debug_class_count]

            if len(ndbe_indices) < args.debug_class_count or len(neo_indices) < args.debug_class_count:
                raise ValueError(
                    f"center_1 needs at least {args.debug_class_count} samples per class, found ndbe={len(ndbe_indices)} neo={len(neo_indices)}"
                )

            selected_indices = ndbe_indices + neo_indices
            selected_targets = [ds.targets[idx] for idx in selected_indices]
            print(
                f"Debug subset active: center_1 with {len(ndbe_indices)} ndbe + {len(neo_indices)} neo "
                f"= {len(selected_indices)} total samples"
            )
        else:
            selected_indices = list(range(len(ds)))
            selected_targets = ds.targets

        # STRATIFIED SPLIT: Split this specific center while maintaining ndbe/neo ratios
        # We split over selected_indices so debug mode can cap class counts first.
        train_idx, val_idx = train_test_split(
            selected_indices,
            train_size=args.DatasetSplit / 100.0, 
            stratify=selected_targets,
            random_state=args.seed
            )
            
        # Append the subsets to our lists
        train_datasets.append(Subset(ds, train_idx))
        valid_datasets.append(Subset(ds, val_idx))
    
    # Merge all the center subsets together
    train_ds = ConcatDataset(train_datasets)
    valid_ds = ConcatDataset(valid_datasets)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Build external testset loader once; metrics are recomputed on it every epoch.
    testset_images_dir = Path(args.testset_images_dir)
    if not testset_images_dir.exists():
        raise FileNotFoundError(f"Testset images directory not found: {testset_images_dir}")

    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    testset_image_paths = sorted(
        p for p in testset_images_dir.iterdir() if p.is_file() and p.suffix.lower() in image_suffixes
    )
    if len(testset_image_paths) == 0:
        raise ValueError(f"No image files found in testset directory: {testset_images_dir}")

    testset_ds = ExternalTestsetDataset(testset_image_paths, transform)
    testset_loader = DataLoader(
        testset_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Using testset images from {testset_images_dir} ({len(testset_image_paths)} samples)")
    
    # MODEL SETUP ----------------------------------------------------------------------------------------------------------
    from model import Model

    class_names = train_datasets[0].dataset.classes
    n_classes = len(class_names)

    model = build_model_compat(Model, args, n_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError(
            "Model has no trainable parameters. "
            "Check model.py implementation and constructor arguments."
        )
    optimizer = AdamW(trainable_params, lr=args.lr)

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
            predictions = torch.argmax(outputs, dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

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
