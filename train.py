import os
import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser
from torch.optim import AdamW
from metrics import (
    compute_group_eval_metrics,
    collect_scores,
    log_metrics,
    project_operating_metrics_to_prevalence,
)
from data import prepare_datasets
from gradcam import evaluate_gradcam_barrett_dataset, evaluate_gradcam_segmentation_dataset
from testdata import load_barrett_gradcam_dataset, load_external_testset, load_segmentation_testset
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
    parser.add_argument("--backbone-name", type=str, default="vit_base_patch16_dinov3.lvd1689m", help="timm backbone name")
    parser.add_argument(
        "--backbone-weights-path",
        type=str,
        default=None,
        help="Optional local checkpoint used to initialize the backbone instead of timm pretrained weights.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Square resize used for both train/validation images and the external testset.",
    )
    parser.add_argument(
        "--testset-images-dir",
        type=str,
        default="./data/EVC_Barretts_FullSet/images",
        help="Path to external testset images used for per-epoch testset metrics",
    )
    parser.add_argument(
        "--segmentation-images-dir",
        type=str,
        default=None,
        help=(
            "Optional image directory used for Grad-CAM-vs-segmentation evaluation. "
            "Defaults to --testset-images-dir when --segmentation-masks-dir is set."
        ),
    )
    parser.add_argument(
        "--segmentation-masks-dir",
        type=str,
        default=None,
        help="Optional directory containing segmentation masks matched to the Grad-CAM evaluation images.",
    )
    parser.add_argument(
        "--gradcam-batch-size",
        type=int,
        default=8,
        help="Batch size used for Grad-CAM segmentation evaluation.",
    )
    parser.add_argument(
        "--gradcam-target-class",
        type=int,
        default=1,
        help="Class index used when generating Grad-CAM heatmaps.",
    )
    parser.add_argument(
        "--gradcam-threshold",
        type=float,
        default=0.5,
        help="Threshold applied to normalized Grad-CAM heatmaps to obtain binary masks for Dice scoring.",
    )
    parser.add_argument(
        "--gradcam-log-samples",
        type=int,
        default=8,
        help="Maximum number of qualitative Grad-CAM examples to log to W&B on evaluation epochs.",
    )
    parser.add_argument(
        "--gradcam-eval-every",
        type=int,
        default=1,
        help="Evaluate and log Grad-CAM segmentation metrics every N epochs.",
    )
    parser.add_argument(
        "--gradcam-skip-empty-masks",
        dest="gradcam_skip_empty_masks",
        action="store_true",
        help="Skip empty ground-truth masks when computing the mean Dice score.",
    )
    parser.add_argument(
        "--gradcam-include-empty-masks",
        dest="gradcam_skip_empty_masks",
        action="store_false",
        help="Include empty ground-truth masks in the mean Dice score.",
    )
    parser.add_argument(
        "--post-train-gradcam",
        action="store_true",
        help="Run the Barrett full-set Grad-CAM evaluator after training and log it to the same W&B run.",
    )
    parser.add_argument(
        "--post-train-gradcam-dataset-root",
        type=str,
        default="../data/EVC_Barretts_FullSet",
        help="Root directory for the post-training Barrett Grad-CAM evaluation dataset.",
    )
    parser.add_argument(
        "--post-train-gradcam-checkpoint",
        type=str,
        default="best",
        choices=("best", "final"),
        help="Which saved checkpoint to evaluate after training.",
    )
    parser.add_argument(
        "--post-train-gradcam-thresholds",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated Grad-CAM thresholds used for post-training Dice/IoU sweeps.",
    )
    parser.add_argument(
        "--post-train-gradcam-display-threshold",
        type=float,
        default=0.5,
        help="Display threshold used in post-training Grad-CAM qualitative panels.",
    )
    parser.add_argument(
        "--post-train-gradcam-log-best-k",
        type=int,
        default=8,
        help="Number of best positive Barrett Grad-CAM examples to log after training.",
    )
    parser.add_argument(
        "--post-train-gradcam-log-worst-k",
        type=int,
        default=8,
        help="Number of worst positive Barrett Grad-CAM examples to log after training.",
    )
    parser.add_argument(
        "--post-train-gradcam-log-hard-neg-k",
        type=int,
        default=8,
        help="Number of hard negative Barrett Grad-CAM examples to log after training.",
    )
    parser.set_defaults(gradcam_skip_empty_masks=True)
    return parser


def main(args):
    # Log into Weights & Biases so we can see the graphs later
    wandb.init(project="RARE25-Project", name=args.experiment_id, config=vars(args))
    
    # Setup directories and devices
    os.makedirs(args.save_dir, exist_ok=True) # Ensure save directory exists
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_loader, valid_loader, train_ds, _, class_names = prepare_datasets(args, device)
    testset_loader, _, testset_image_paths = load_external_testset(
        args.testset_images_dir, args.batch_size, args.num_workers, device, args.input_size
    )
    print(f"Using testset images from {args.testset_images_dir} ({len(testset_image_paths)} samples)")

    segmentation_loader = None
    if args.segmentation_masks_dir:
        segmentation_images_dir = args.segmentation_images_dir or args.testset_images_dir
        segmentation_loader, _, matched_segmentation_samples, unmatched_segmentation_images = load_segmentation_testset(
            segmentation_images_dir=segmentation_images_dir,
            segmentation_masks_dir=args.segmentation_masks_dir,
            batch_size=args.gradcam_batch_size,
            num_workers=args.num_workers,
            device=device,
            input_size=args.input_size,
        )
        print(
            "Using segmentation Grad-CAM evaluation from "
            f"{segmentation_images_dir} with masks in {args.segmentation_masks_dir} "
            f"({len(matched_segmentation_samples)} matched images, "
            f"{len(unmatched_segmentation_images)} unmatched)"
        )
    
    # MODEL SETUP ----------------------------------------------------------------------------------------------------------
    from model import Model, load_model_checkpoint

    n_classes = len(class_names)
    model = Model(
        in_channels=3,
        n_classes=n_classes,
        backbone_name=args.backbone_name,
        backbone_weights_path=args.backbone_weights_path,
        input_size=args.input_size,
    ).to(device)
    if args.backbone_weights_path:
        print(f"Loaded backbone weights from {args.backbone_weights_path}")

    train_labels = torch.tensor(train_ds.df["label"].tolist(), dtype=torch.long)
    class_counts = torch.bincount(train_labels, minlength=n_classes).float()
    if torch.any(class_counts == 0):
        raise ValueError(f"At least one class has zero training samples: {class_counts.tolist()}")
    class_weights = class_counts.sum() / (n_classes * class_counts)
    class_weights = class_weights.to(device)
    print(f"Using balanced cross entropy with class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    projected_prevalence = 0.01

    # TRAINING LOOP --------------------------------------------------------------------------------------------------------
    best_valid_projected_ppv = float("-inf")
    best_valid_fpr = float("inf")
    best_save_path = os.path.join(args.save_dir, f"{args.experiment_id}_best.pt")

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
        gradcam_payload = None
        if segmentation_loader is not None and (epoch + 1) % args.gradcam_eval_every == 0:
            gradcam_payload = evaluate_gradcam_segmentation_dataset(
                model=model,
                loader=segmentation_loader,
                device=device,
                target_class=args.gradcam_target_class,
                threshold=args.gradcam_threshold,
                max_log_samples=args.gradcam_log_samples,
                skip_empty_masks=args.gradcam_skip_empty_masks,
                split_name="segmentation",
            )

        gradcam_summary = ""
        if gradcam_payload is not None:
            gradcam_summary = (
                f" | Seg Mean Dice: {gradcam_payload['segmentation/mean_dice']:.4f} "
                f"| Seg Scored: {gradcam_payload['segmentation/dice_scored_samples']} "
                f"| Seg Skipped Empty: {gradcam_payload['segmentation/dice_skipped_empty_masks']}"
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
            f"{gradcam_summary}"
        )
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
            extra_payload=gradcam_payload,
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
        is_better_checkpoint = (
            current_valid_projected_ppv > best_valid_projected_ppv
            or (same_projected_ppv and current_valid_fpr < best_valid_fpr)
        )

        if is_better_checkpoint:
            best_valid_projected_ppv = current_valid_projected_ppv
            best_valid_fpr = current_valid_fpr
            torch.save(model.state_dict(), best_save_path)
            print(
                f"   -> Saved new best model to {best_save_path} "
                f"(1% PPV: {valid_projected_metrics['Projected PPV']:.4f}, "
                f"FPR: {valid_metrics['FPR']:.4f}, Threshold: {valid_threshold:.4f})"
            )

    final_save_path = os.path.join(args.save_dir, f"{args.experiment_id}_final.pt")
    torch.save(model.state_dict(), final_save_path)

    if args.post_train_gradcam:
        print(
            f"Running post-training Barrett Grad-CAM evaluation from "
            f"{args.post_train_gradcam_dataset_root} within the same W&B run..."
        )
        gradcam_loader, _, _, gradcam_dataset_qa = load_barrett_gradcam_dataset(
            dataset_root=args.post_train_gradcam_dataset_root,
            batch_size=args.gradcam_batch_size,
            num_workers=args.num_workers,
            device=device,
            input_size=args.input_size,
        )

        checkpoint_to_evaluate = final_save_path
        if args.post_train_gradcam_checkpoint == "best":
            if os.path.exists(best_save_path):
                checkpoint_to_evaluate = best_save_path
            else:
                print(
                    f"Requested best-checkpoint Grad-CAM evaluation, but {best_save_path} was not found. "
                    f"Falling back to {final_save_path}."
                )

        load_model_checkpoint(model, checkpoint_to_evaluate)
        print(f"Loaded checkpoint for post-training Grad-CAM evaluation: {checkpoint_to_evaluate}")

        gradcam_result = evaluate_gradcam_barrett_dataset(
            model=model,
            loader=gradcam_loader,
            device=device,
            thresholds=args.post_train_gradcam_thresholds,
            target_class=args.gradcam_target_class,
            display_threshold=args.post_train_gradcam_display_threshold,
            log_best_k=args.post_train_gradcam_log_best_k,
            log_worst_k=args.post_train_gradcam_log_worst_k,
            log_hard_neg_k=args.post_train_gradcam_log_hard_neg_k,
            prefix="gradcam",
            dataset_qa=gradcam_dataset_qa,
        )
        wandb.log(gradcam_result["payload"])
        print(
            "Post-training Grad-CAM summary | "
            f"mAP consensus: {gradcam_result['payload']['gradcam/positive/mAP_consensus']:.4f} | "
            f"Expert mAP mean: {gradcam_result['payload']['gradcam/positive/mAP_expert_mean']:.4f} | "
            f"Dice AUC: {gradcam_result['payload']['gradcam/positive/dice_auc']:.4f} | "
            f"IoU AUC: {gradcam_result['payload']['gradcam/positive/iou_auc']:.4f} | "
            f"Negative mean prob: {gradcam_result['payload']['gradcam/negative/mean_positive_class_probability']:.4f}"
        )

    print(f"Class mapping: {class_names}")
    print("Training finished! Check your WandB dashboard.")
    wandb.finish()

if __name__ == "__main__":
    main(get_args_parser().parse_args())
