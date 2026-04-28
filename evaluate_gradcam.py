from argparse import ArgumentParser
from pathlib import Path

import torch
import wandb

from gradcam import evaluate_gradcam_barrett_dataset
from model import Model, load_model_checkpoint
from testdata import load_barrett_gradcam_dataset


DEFAULT_THRESHOLDS = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"


def get_args_parser():
    parser = ArgumentParser("Standalone Grad-CAM evaluation for the Barrett full set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="../data/EVC_Barretts_FullSet",
        help="Root directory containing images and annotations_bmp.",
    )
    parser.add_argument("--input-size", type=int, default=224, help="Square input size used by the model.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size used for Grad-CAM evaluation.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument(
        "--gradcam-target-class",
        type=int,
        default=1,
        help="Class index used when generating Grad-CAM heatmaps.",
    )
    parser.add_argument(
        "--gradcam-thresholds",
        type=str,
        default=DEFAULT_THRESHOLDS,
        help="Comma-separated Grad-CAM thresholds used for Dice/IoU sweeps.",
    )
    parser.add_argument("--log-best-k", type=int, default=8, help="Number of best positive examples to log.")
    parser.add_argument("--log-worst-k", type=int, default=8, help="Number of worst positive examples to log.")
    parser.add_argument(
        "--log-hard-neg-k",
        type=int,
        default=8,
        help="Number of hard negative examples to log.",
    )
    parser.add_argument("--wandb-project", type=str, default="RARE25-Project", help="Weights & Biases project name.")
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional W&B run name. Defaults to '<checkpoint stem>-gradcam'.",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default="gradcam",
        help="W&B group used to collect standalone Grad-CAM runs.",
    )
    parser.add_argument(
        "--backbone-name",
        type=str,
        default="vit_base_patch16_dinov3.lvd1689m",
        help="Backbone architecture used to create the model before loading the checkpoint.",
    )
    return parser

def main(args):
    checkpoint_path = Path(args.checkpoint)
    run_name = args.wandb_run_name or f"{checkpoint_path.stem}-gradcam"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        group=args.wandb_group,
        config=vars(args),
    )

    loader, _, _, dataset_qa = load_barrett_gradcam_dataset(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        input_size=args.input_size,
    )
    print(
        f"Loaded Barrett Grad-CAM dataset from {args.dataset_root} "
        f"({dataset_qa['image_count']} images, "
        f"{dataset_qa['positive_image_count']} positives, "
        f"{dataset_qa['negative_image_count']} negatives)"
    )

    model = Model(
        in_channels=3,
        n_classes=2,
        backbone_name=args.backbone_name,
        input_size=args.input_size,
        pretrained=False,
    ).to(device)
    load_model_checkpoint(model, checkpoint_path)
    print(f"Loaded checkpoint from {checkpoint_path}")

    result = evaluate_gradcam_barrett_dataset(
        model=model,
        loader=loader,
        device=device,
        thresholds=args.gradcam_thresholds,
        target_class=args.gradcam_target_class,
        log_best_k=args.log_best_k,
        log_worst_k=args.log_worst_k,
        log_hard_neg_k=args.log_hard_neg_k,
        prefix="gradcam",
        dataset_qa=dataset_qa,
    )
    wandb.log(result["payload"])

    payload = result["payload"]
    print(
        "Grad-CAM summary | "
        f"mAP consensus: {payload['gradcam/positive/mAP_consensus']:.4f} | "
        f"Expert mAP mean: {payload['gradcam/positive/mAP_expert_mean']:.4f} | "
        f"Dice AUC: {payload['gradcam/positive/dice_auc']:.4f} | "
        f"IoU AUC: {payload['gradcam/positive/iou_auc']:.4f} | "
        f"Negative mean prob: {payload['gradcam/negative/mean_positive_class_probability']:.4f}"
    )
    wandb.finish()


if __name__ == "__main__":
    main(get_args_parser().parse_args())
