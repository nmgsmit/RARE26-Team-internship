import os
from argparse import SUPPRESS, ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from data import SimpleDataset, build_eval_transform, prepare_datasets
from gradcam import compute_vit_gradcam_batch, evaluate_gradcam_barrett_dataset, evaluate_gradcam_segmentation_dataset
from metrics import (
    collect_scores,
    compute_group_eval_metrics,
    log_metrics,
    project_operating_metrics_to_prevalence,
)
from model import (
    Model,
    create_model_checkpoint,
    load_encoder_checkpoint,
    load_model_checkpoint,
)
from roi_guidance import build_roi_record_from_cam, load_roi_records_from_json
from testdata import load_barrett_gradcam_dataset, load_external_testset, load_segmentation_testset


DEFAULT_GASTRONET_CKPT = "../Gastronet/dinov2.pth"
DEFAULT_SIMCLR_CKPT = "../Gastronet/RN50_GastroNet-5M_SIMCLRv2.pth"
DEFAULT_MOCOV2_CKPT = "../Gastronet/RN50_GastroNet-5M_MOCOv2.pth"
DEFAULT_RESNET50_CKPT = "../Gastronet/RN50_ImageNet_timm_resnet50.pth"
DEFAULT_DATA_DIR = "../data/Challenge_train_data"
DEFAULT_TESTSET_IMAGES_DIR = "../data/EVC_Barretts_FullSet/images"
DEFAULT_POST_TRAIN_GRADCAM_DATASET_ROOT = "../data/EVC_Barretts_FullSet"
DEFAULT_POST_TRAIN_GRADCAM_THRESHOLDS = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
PRETRAIN_LOSSES = {"supmin", "suppro"}
SUPERVISED_LOSSES = {"ce", "class-balanced"}
LOSS_ALIASES = {
    "balanced": "class-balanced",
    "balanced-ce": "class-balanced",
    "class-balanced-ce": "class-balanced",
    "class_balanced": "class-balanced",
}
BACKBONE_PRESETS = {
    "dinov3": {
        "backbone_name": "vit_base_patch16_dinov3.lvd1689m",
        "backbone_weights_path": None,
        "input_size": 224,
        "pretrained": True,
    },
    "gastronet": {
        "backbone_name": "vit_base_patch14_reg4_dinov2",
        "backbone_weights_path": DEFAULT_GASTRONET_CKPT,
        "input_size": 336,
        "pretrained": False,
    },
    "simclr": {
        "backbone_name": "resnet50",
        "backbone_weights_path": DEFAULT_SIMCLR_CKPT,
        "input_size": 224,
        "pretrained": False,
    },
    "mocov2": {
        "backbone_name": "resnet50",
        "backbone_weights_path": DEFAULT_MOCOV2_CKPT,
        "input_size": 224,
        "pretrained": False,
    },
    "resnet50": {
        "backbone_name": "resnet50",
        "backbone_weights_path": DEFAULT_RESNET50_CKPT,
        "input_size": 224,
        "pretrained": False,
    },
}
HEAD_TYPE_CHOICES = (
    "linear",
    "ln_linear",
    "mlp_fullwidth",
    "mlp_bottleneck",
    "residual_bottleneck",
    "cosine_linear",
)


def get_args_parser():
    parser = ArgumentParser("RARE25 configurable staged training")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["baseline", "pretrain", "finetune"],
        required=True,
        help="Training stage. Use baseline for supervised head training, pretrain for SupMin/SupPro, or finetune after a pretraining checkpoint.",
    )
    parser.add_argument(
        "--loss-name",
        type=str,
        default=None,
        help="Loss selection: ce, class-balanced, supmin, or suppro.",
    )
    parser.add_argument("--method", type=str, default=None, help=SUPPRESS)
    parser.add_argument("--encoder-ckpt", type=str, default=None)
    parser.add_argument(
        "--init-encoder-ckpt",
        type=str,
        default=None,
        help=(
            "Optional encoder checkpoint used to initialize the backbone and projection head before "
            "pretraining. This is intended for ROI curriculum re-pretraining."
        ),
    )
    parser.add_argument("--warmup-epochs", type=int, default=3)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--baseline-lr",
        type=float,
        default=None,
        help="Learning rate for baseline-stage classifier training. Defaults to --lr.",
    )
    parser.add_argument(
        "--pretrain-backbone-lr",
        type=float,
        default=None,
        help="Learning rate for backbone parameters during pretraining. Defaults to --lr.",
    )
    parser.add_argument(
        "--pretrain-proj-lr",
        type=float,
        default=None,
        help="Learning rate for projection-head parameters during pretraining. Defaults to 3e-4.",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=None,
        help="Learning rate for finetune-stage classifier training. Defaults to 3e-4.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--num-folds",
        type=int,
        default=1,
        help="Number of cross-validation folds. Use 1 for the default single split.",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=0,
        help="Zero-based validation fold index when --num-folds > 1.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", type=str, default="rare25-run")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="RARE25-Project",
        help="Weights & Biases project used for training runs.",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Optional Weights & Biases group for collecting related runs.",
    )
    parser.add_argument(
        "--head-type",
        type=str,
        default="mlp_fullwidth",
        choices=HEAD_TYPE_CHOICES,
        help="Classifier head used for baseline/finetune. Pretrain ignores this head and only learns the backbone + projection head.",
    )
    parser.add_argument(
        "--head-hidden-dim",
        type=int,
        default=None,
        help="Hidden width used by bottleneck-style heads.",
    )
    parser.add_argument(
        "--head-dropout",
        type=float,
        default=0.0,
        help="Dropout used by bottleneck-style heads.",
    )
    parser.add_argument(
        "--mlp-hidden-layers",
        type=int,
        default=1,
        help="Number of hidden layers for the full-width MLP head.",
    )
    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=None,
        help="Hidden width for the full-width MLP head. Defaults to the backbone feature dimension.",
    )
    parser.add_argument(
        "--mlp-dropout",
        type=float,
        default=0.0,
        help="Dropout applied after each hidden layer in the full-width MLP head.",
    )
    parser.add_argument(
        "--backbone-preset",
        type=str,
        choices=sorted(BACKBONE_PRESETS),
        default="gastronet",
        help=(
            "Convenience switch for the backbone setup. 'dinov3' uses timm pretrained "
            "DINOv3 at 224px. 'gastronet' uses the GastroNet DINOv2 checkpoint at 336px. "
            "'simclr' uses the GastroNet SIMCLRv2 RN50 checkpoint at 224px. "
            "'mocov2' uses the GastroNet MOCOv2 RN50 checkpoint at 224px. "
            "'resnet50' uses a locally stored timm/ImageNet pretrained ResNet-50 checkpoint at 224px."
        ),
    )
    parser.add_argument(
        "--backbone-name",
        type=str,
        default=None,
        help="Optional manual override for the timm backbone name.",
    )
    parser.add_argument(
        "--backbone-weights-path",
        type=str,
        default=None,
        help=(
            "Optional manual override for the backbone checkpoint path. "
            "The gastronet preset defaults to ../Gastronet/dinov2.pth and the simclr "
            "preset defaults to ../Gastronet/RN50_GastroNet-5M_SIMCLRv2.pth. "
            "The mocov2 preset defaults to ../Gastronet/RN50_GastroNet-5M_MOCOv2.pth. "
            "The resnet50 preset defaults to ../Gastronet/RN50_ImageNet_timm_resnet50.pth."
        ),
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Optional manual override for the square input size.",
    )
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument("--pretrained", action="store_true", dest="pretrained")
    pretrained_group.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    parser.set_defaults(pretrained=None)

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
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--base-temperature", type=float, default=0.07)
    parser.add_argument("--gradcam-batch-size", type=int, default=8)
    parser.add_argument("--gradcam-target-class", type=int, default=1)
    parser.add_argument("--gradcam-threshold", type=float, default=0.5)
    parser.add_argument("--gradcam-log-samples", type=int, default=10)
    parser.add_argument("--gradcam-eval-every", type=int, default=1)
    parser.add_argument(
        "--roi-guided-training",
        action="store_true",
        help=(
            "After a warmup period, replace the second training view of positive samples with "
            "an ROI-focused crop extracted from Grad-CAM."
        ),
    )
    parser.add_argument(
        "--roi-records-path",
        type=str,
        default=None,
        help=(
            "Optional path to saved ROI metadata JSON. When provided, the train split immediately "
            "uses those ROIs to guide positive-sample crops."
        ),
    )
    parser.add_argument(
        "--roi-start-epoch",
        type=int,
        default=20,
        help="Activate ROI-guided training after this many full epochs have completed.",
    )
    parser.add_argument(
        "--roi-focus-prob",
        type=float,
        default=1.0,
        help="Probability of replacing the second positive training view with an ROI-focused crop.",
    )
    parser.add_argument(
        "--roi-context-scale",
        type=float,
        default=2.0,
        help="Context multiplier applied around the ROI box before resizing to the training input size.",
    )
    parser.add_argument(
        "--roi-min-crop-scale",
        type=float,
        default=0.4,
        help="Minimum normalized crop size used for ROI-focused crops.",
    )
    parser.add_argument(
        "--roi-center-jitter",
        type=float,
        default=0.05,
        help="Random center jitter applied to ROI-focused crops as a fraction of crop width and height.",
    )
    parser.add_argument(
        "--roi-gradcam-threshold",
        type=float,
        default=0.6,
        help="Threshold applied to normalized Grad-CAM heatmaps when extracting pseudo-ROI boxes.",
    )
    parser.add_argument(
        "--roi-gradcam-min-prob",
        type=float,
        default=0.5,
        help="Minimum positive-class probability required before a Grad-CAM pseudo-ROI is accepted.",
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
        help="Run the Barrett full-set Grad-CAM evaluator after baseline or finetune training.",
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
        default=DEFAULT_POST_TRAIN_GRADCAM_THRESHOLDS,
        help="Comma-separated Grad-CAM thresholds used for post-training Dice/IoU sweeps.",
    )
    parser.add_argument(
        "--post-train-gradcam-display-threshold",
        type=float,
        default=0.5,
        help="Consensus-agreement threshold used for the white outline in post-training Grad-CAM panels.",
    )
    parser.add_argument("--post-train-gradcam-log-best-k", type=int, default=10)
    parser.add_argument("--post-train-gradcam-log-worst-k", type=int, default=10)
    parser.add_argument("--post-train-gradcam-log-hard-neg-k", type=int, default=10)

    parser.add_argument("--lambda-ce", type=float, default=1.0, help=SUPPRESS)
    parser.add_argument("--lambda-supmin", type=float, default=1.0, help=SUPPRESS)
    parser.add_argument("--lambda-suppro", type=float, default=1.0, help=SUPPRESS)

    parser.set_defaults(gradcam_skip_empty_masks=True)
    return parser


def canonicalize_loss_name(loss_name):
    if loss_name is None:
        return None

    normalized = loss_name.strip().lower()
    return LOSS_ALIASES.get(normalized, normalized)


def _flatten_roi_source_counts(source_counts):
    return {
        f"train/roi_source_count_{source}": int(count)
        for source, count in sorted(source_counts.items())
    }


def _format_roi_source_counts(source_counts):
    if not source_counts:
        return "none"
    return ", ".join(f"{source}={count}" for source, count in sorted(source_counts.items()))


def build_gradcam_roi_records(args, model, train_ds, device):
    positive_df = train_ds.df.loc[train_ds.df["label"] == args.gradcam_target_class].copy()
    positive_df["img"] = positive_df["img"].astype(str)

    if positive_df.empty:
        return {}, 0

    eval_ds = SimpleDataset(positive_df, build_eval_transform(args.input_size))
    eval_loader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=args.gradcam_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    roi_records = {}
    image_paths = positive_df["img"].tolist()
    path_offset = 0
    was_training = model.training
    model.eval()

    try:
        for images, _labels in eval_loader:
            batch_paths = image_paths[path_offset:path_offset + images.size(0)]
            path_offset += images.size(0)
            images = images.to(device)

            cams, probs = compute_vit_gradcam_batch(
                model=model,
                images=images,
                target_class=args.gradcam_target_class,
            )
            cams = cams.detach().cpu()
            probs = probs.detach().cpu()

            for image_path, cam, prob in zip(batch_paths, cams, probs):
                positive_prob = float(prob.item())
                if positive_prob < args.roi_gradcam_min_prob:
                    continue

                roi_record = build_roi_record_from_cam(
                    cam.numpy(),
                    threshold=args.roi_gradcam_threshold,
                    score=positive_prob,
                )
                if roi_record is None:
                    continue
                roi_records[str(image_path)] = roi_record
    finally:
        if was_training:
            model.train()

    return roi_records, len(image_paths)


def activate_saved_train_roi_guidance(args, train_ds):
    if not args.roi_records_path:
        return None

    roi_records, metadata = load_roi_records_from_json(args.roi_records_path)
    train_image_paths = set(train_ds.df["img"].astype(str).tolist())
    matched_records = {
        image_path: record for image_path, record in roi_records.items() if image_path in train_image_paths
    }
    unmatched_record_count = len(roi_records) - len(matched_records)
    train_ds.set_roi_records(matched_records, active=True)
    dataset_stats = train_ds.get_roi_guidance_stats()

    payload = {
        "train/roi_records_loaded_total": len(roi_records),
        "train/roi_records_loaded_matched": len(matched_records),
        "train/roi_records_loaded_unmatched": unmatched_record_count,
        "train/roi_positive_images": dataset_stats["roi_positive_images"],
        "train/roi_positive_candidates": dataset_stats["roi_positive_candidates"],
        "train/roi_mean_coverage": dataset_stats["roi_mean_coverage"],
    }
    payload.update(_flatten_roi_source_counts(dataset_stats["roi_source_counts"]))
    wandb.log(payload, step=0)

    metadata_checkpoint = metadata.get("checkpoint", "unknown")
    print(
        "Loaded saved ROI guidance | "
        f"path={args.roi_records_path} | "
        f"matched={len(matched_records)}/{len(roi_records)} | "
        f"unmatched={unmatched_record_count} | "
        f"roi positives={dataset_stats['roi_positive_images']}/{dataset_stats['roi_positive_candidates']} | "
        f"sources={_format_roi_source_counts(dataset_stats['roi_source_counts'])} | "
        f"checkpoint={metadata_checkpoint}"
    )
    return {
        "records_total": len(roi_records),
        "matched_records": len(matched_records),
        "unmatched_records": unmatched_record_count,
        "dataset_stats": dataset_stats,
        "metadata": metadata,
    }


def refresh_train_roi_guidance(args, model, train_ds, device, epoch_index):
    if args.stage == "pretrain":
        raise ValueError("ROI-guided training is only supported for baseline or finetune stages.")

    gradcam_records, gradcam_candidates = build_gradcam_roi_records(
        args,
        model,
        train_ds,
        device,
    )
    train_ds.set_roi_records(gradcam_records, active=True)
    dataset_stats = train_ds.get_roi_guidance_stats()

    payload = {
        "train/roi_epoch_activated": epoch_index + 1,
        "train/roi_records_total": len(gradcam_records),
        "train/roi_records_gradcam": len(gradcam_records),
        "train/roi_gradcam_candidates": gradcam_candidates,
        "train/roi_positive_images": dataset_stats["roi_positive_images"],
        "train/roi_positive_candidates": dataset_stats["roi_positive_candidates"],
        "train/roi_mean_coverage": dataset_stats["roi_mean_coverage"],
    }
    payload.update(_flatten_roi_source_counts(dataset_stats["roi_source_counts"]))
    wandb.log(payload, step=epoch_index + 1)

    print(
        "Activated ROI-guided training | "
        f"epoch={epoch_index + 1} | "
        f"roi positives={dataset_stats['roi_positive_images']}/{dataset_stats['roi_positive_candidates']} | "
        f"sources={_format_roi_source_counts(dataset_stats['roi_source_counts'])} | "
        f"gradcam accepted={len(gradcam_records)}/{gradcam_candidates}"
    )
    return {
        "records_total": len(gradcam_records),
        "gradcam_records": len(gradcam_records),
        "gradcam_candidates": gradcam_candidates,
        "dataset_stats": dataset_stats,
    }


def resolve_runtime_config(args):
    args.data_dir = DEFAULT_DATA_DIR
    args.testset_images_dir = DEFAULT_TESTSET_IMAGES_DIR
    args.post_train_gradcam_dataset_root = DEFAULT_POST_TRAIN_GRADCAM_DATASET_ROOT

    if args.loss_name is None:
        args.loss_name = args.method
    if args.loss_name is None:
        args.loss_name = {
            "baseline": "class-balanced",
            "pretrain": "supmin",
            "finetune": "class-balanced",
        }[args.stage]
    args.loss_name = canonicalize_loss_name(args.loss_name)

    valid_losses = PRETRAIN_LOSSES | SUPERVISED_LOSSES
    if args.loss_name not in valid_losses:
        raise ValueError(
            f"Unknown loss '{args.loss_name}'. Expected one of {sorted(valid_losses)}."
        )

    if args.stage == "pretrain" and args.loss_name not in PRETRAIN_LOSSES:
        raise ValueError(
            "Pretrain stage only supports SupMin or SupPro losses. "
            "Use --loss-name supmin or --loss-name suppro."
        )
    if args.stage != "pretrain" and args.loss_name not in SUPERVISED_LOSSES:
        raise ValueError(
            "Baseline and finetune stages only support supervised losses. "
            "Use --loss-name ce or --loss-name class-balanced."
        )
    if args.roi_guided_training and args.roi_records_path:
        raise ValueError(
            "--roi-guided-training and --roi-records-path are mutually exclusive. "
            "Use saved ROI records for offline ROI-guided pretraining, or online Grad-CAM refresh for finetuning."
        )
    if args.init_encoder_ckpt and args.stage != "pretrain":
        raise ValueError("--init-encoder-ckpt is only supported for pretrain stage.")
    if not 0.0 <= args.roi_focus_prob <= 1.0:
        raise ValueError(f"--roi-focus-prob must be in [0, 1], got {args.roi_focus_prob}.")
    if args.roi_context_scale <= 0.0:
        raise ValueError(f"--roi-context-scale must be > 0, got {args.roi_context_scale}.")
    if not 0.0 < args.roi_min_crop_scale <= 1.0:
        raise ValueError(
            f"--roi-min-crop-scale must be in (0, 1], got {args.roi_min_crop_scale}."
        )
    if args.roi_center_jitter < 0.0:
        raise ValueError(f"--roi-center-jitter must be >= 0, got {args.roi_center_jitter}.")
    if not 0.0 <= args.roi_gradcam_threshold <= 1.0:
        raise ValueError(
            f"--roi-gradcam-threshold must be in [0, 1], got {args.roi_gradcam_threshold}."
        )
    if not 0.0 <= args.roi_gradcam_min_prob <= 1.0:
        raise ValueError(
            f"--roi-gradcam-min-prob must be in [0, 1], got {args.roi_gradcam_min_prob}."
        )

    if args.head_hidden_dim is not None and args.head_hidden_dim <= 0:
        raise ValueError(f"--head-hidden-dim must be > 0, got {args.head_hidden_dim}.")
    if args.mlp_hidden_layers < 0:
        raise ValueError(f"--mlp-hidden-layers must be >= 0, got {args.mlp_hidden_layers}.")
    if args.mlp_hidden_dim is not None and args.mlp_hidden_dim <= 0:
        raise ValueError(f"--mlp-hidden-dim must be > 0, got {args.mlp_hidden_dim}.")
    if not 0.0 <= args.head_dropout < 1.0:
        raise ValueError(f"--head-dropout must be in [0, 1), got {args.head_dropout}.")
    if not 0.0 <= args.mlp_dropout < 1.0:
        raise ValueError(f"--mlp-dropout must be in [0, 1), got {args.mlp_dropout}.")
    if args.num_folds <= 0:
        raise ValueError(f"--num-folds must be >= 1, got {args.num_folds}.")
    if args.num_folds == 1:
        args.fold_index = 0
    elif args.fold_index < 0 or args.fold_index >= args.num_folds:
        raise ValueError(
            f"--fold-index must be in [0, {args.num_folds - 1}] when using cross-validation, "
            f"got {args.fold_index}."
        )

    if args.baseline_lr is None:
        args.baseline_lr = args.lr
    if args.pretrain_backbone_lr is None:
        args.pretrain_backbone_lr = args.lr
    if args.pretrain_proj_lr is None:
        args.pretrain_proj_lr = 3e-4
    if args.finetune_lr is None:
        args.finetune_lr = 3e-4

    preset = BACKBONE_PRESETS[args.backbone_preset]
    if args.backbone_name is None:
        args.backbone_name = preset["backbone_name"]
    if args.backbone_weights_path is None:
        args.backbone_weights_path = preset["backbone_weights_path"]
    if args.input_size is None:
        args.input_size = preset["input_size"]
    if args.pretrained is None:
        args.pretrained = preset["pretrained"]

    return args


def suppro_loss(features, labels, temperature, base_temperature, class_weights=None):
    features = F.normalize(features, dim=-1)
    device = features.device
    _, views, _ = features.shape

    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    logits = torch.matmul(contrast_feature, contrast_feature.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    mask = mask.repeat(views, views)

    logits_mask = torch.ones_like(mask)
    logits_mask.fill_diagonal_(0)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
    loss = -(temperature / base_temperature) * mean_log_prob_pos

    if class_weights is not None:
        repeated_labels = labels.view(-1).repeat(views)
        loss = loss * class_weights[repeated_labels]

    return loss.mean()


def supmin_loss(embeddings, labels, margin=0.1):
    sim = torch.matmul(embeddings, embeddings.T)
    dist = 1.0 - sim

    same = (labels[:, None] == labels[None, :]).float()
    diff = 1.0 - same

    diag = torch.eye(labels.size(0), device=labels.device)
    same = same * (1.0 - diag)
    diff = diff * (1.0 - diag)

    pos = (same * dist).sum() / (same.sum() + 1e-12)
    neg = (diff * F.relu(margin - dist)).sum() / (diff.sum() + 1e-12)
    return pos + neg


def build_pretrain_scheduler(optimizer, warmup_epochs, total_epochs):
    warmup = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / max(1, warmup_epochs))
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=0.0,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def build_finetune_scheduler(optimizer, total_epochs):
    return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=0.0)


def resolve_post_train_gradcam_dataset_root(args):
    return Path(args.post_train_gradcam_dataset_root)


def validate_post_train_gradcam_dataset(args):
    dataset_root = resolve_post_train_gradcam_dataset_root(args)
    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "annotations_bmp"
    if not images_dir.exists():
        raise FileNotFoundError(f"Grad-CAM images directory not found: {images_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Grad-CAM annotations directory not found: {annotations_dir}")
    return dataset_root


def run_post_training_gradcam(args, model, device, final_save_path, best_save_path):
    gradcam_dataset_root = validate_post_train_gradcam_dataset(args)
    print(
        f"Running post-training Barrett Grad-CAM evaluation from {gradcam_dataset_root} "
        "within the same W&B run..."
    )
    gradcam_loader, _, _, gradcam_dataset_qa = load_barrett_gradcam_dataset(
        dataset_root=gradcam_dataset_root,
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
    payload = {}
    if gradcam_result["scalar_payload"]:
        payload.update(gradcam_result["scalar_payload"])
    if gradcam_result["media_payload"]:
        payload.update(gradcam_result["media_payload"])
    if payload:
        wandb.log(payload)
    for key, value in gradcam_result["summary_payload"].items():
        wandb.summary[key] = value
    print(
        "Post-training Grad-CAM summary | "
        f"mAP consensus: {gradcam_result['summary_payload']['gradcam/positive/mAP_consensus']:.4f} | "
        f"Consensus mass: {gradcam_result['summary_payload']['gradcam/positive/consensus_mass']:.4f} | "
        f"Expert mAP mean: {gradcam_result['summary_payload']['gradcam/positive/mAP_expert_mean']:.4f} | "
        f"Dice AUC: {gradcam_result['summary_payload']['gradcam/positive/dice_auc']:.4f} | "
        f"IoU AUC: {gradcam_result['summary_payload']['gradcam/positive/iou_auc']:.4f} | "
        f"Negative mean prob: {gradcam_result['summary_payload']['gradcam/negative/mean_positive_class_probability']:.4f} | "
        f"Flat/near-zero CAM frac: {gradcam_result['summary_payload']['gradcam/overall/fraction_flat_or_near_zero_cams']:.4f}"
    )


def build_supervised_criterion(loss_name, train_ds, n_classes, device):
    if loss_name == "ce":
        print("Using standard cross entropy.")
        return nn.CrossEntropyLoss()

    train_labels = torch.tensor(train_ds.df["label"].tolist(), dtype=torch.long)
    class_counts = torch.bincount(train_labels, minlength=n_classes).float()
    if torch.any(class_counts == 0):
        raise ValueError(f"At least one class has zero training samples: {class_counts.tolist()}")

    class_weights = class_counts.sum() / (n_classes * class_counts)
    class_weights = class_weights.to(device)
    print(f"Using class-balanced cross entropy with weights: {class_weights.tolist()}")
    return nn.CrossEntropyLoss(weight=class_weights)


def configure_stage(model, args):
    if args.stage == "baseline":
        for parameter in model.proj_head.parameters():
            parameter.requires_grad = False
        optimizer = AdamW(model.cls_head.parameters(), lr=args.baseline_lr)
        scheduler = None
        return optimizer, scheduler

    if args.stage == "pretrain":
        if args.init_encoder_ckpt:
            load_encoder_checkpoint(model, args.init_encoder_ckpt)
            print(f"Loaded initialization encoder checkpoint from {args.init_encoder_ckpt}")
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True
        for parameter in model.proj_head.parameters():
            parameter.requires_grad = True
            
        optimizer = SGD(
            [
                {"params": model.backbone.parameters(), "lr": args.pretrain_backbone_lr},
                {"params": model.proj_head.parameters(), "lr": args.pretrain_proj_lr}
            ],
            momentum=0.9,
            weight_decay=1e-4,
        )
        scheduler = build_pretrain_scheduler(optimizer, args.warmup_epochs, args.epochs)
        return optimizer, scheduler

    if args.stage == "finetune":
        if not args.encoder_ckpt:
            raise ValueError("--encoder-ckpt is required for finetune stage")

        load_encoder_checkpoint(model, args.encoder_ckpt)
        print(f"Loaded encoder checkpoint from {args.encoder_ckpt}")

        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        for parameter in model.proj_head.parameters():
            parameter.requires_grad = False
        for parameter in model.cls_head.parameters():
            parameter.requires_grad = True

        optimizer = SGD(
            model.cls_head.parameters(),
            lr=args.finetune_lr,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        scheduler = None
        
        return optimizer, scheduler

    raise ValueError(f"Unknown stage: {args.stage}")


def main(args):
    args = resolve_runtime_config(args)
    if args.roi_guided_training and args.stage == "pretrain":
        raise ValueError(
            "--roi-guided-training is only supported for baseline or finetune stages. "
            "Use --roi-records-path for ROI-guided pretraining with saved Grad-CAM crops."
        )
    wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.experiment_id,
        config=vars(args),
    )

    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"Stage: {args.stage} | Loss: {args.loss_name} | Backbone preset: {args.backbone_preset}"
    )
    if args.num_folds > 1:
        print(f"Cross-validation: fold {args.fold_index + 1}/{args.num_folds}")
    print(
        f"Resolved backbone: {args.backbone_name} | input size: {args.input_size} | "
        f"pretrained: {args.pretrained} | backbone weights: {args.backbone_weights_path}"
    )
    print(
        f"Head: {args.head_type} | head hidden dim: {args.head_hidden_dim} | "
        f"MLP hidden layers: {args.mlp_hidden_layers} | MLP hidden dim: {args.mlp_hidden_dim}"
    )
    if args.roi_records_path:
        print(f"Saved ROI guidance configured from {args.roi_records_path}")
    if args.roi_guided_training:
        print(
            "ROI-guided training configured | "
            f"activation after {args.roi_start_epoch} epochs | "
            f"gradcam threshold={args.roi_gradcam_threshold} | "
            f"min positive prob={args.roi_gradcam_min_prob}"
        )
    if args.stage == "pretrain":
        print(
            "Pretrain mode only learns the backbone and projection head. "
            "It saves an encoder checkpoint that you use later with --stage finetune."
        )

    if args.stage != "pretrain" and args.post_train_gradcam:
        validate_post_train_gradcam_dataset(args)

    train_loader, valid_loader, train_ds, _, class_names = prepare_datasets(args, device)
    if args.roi_records_path:
        activate_saved_train_roi_guidance(args, train_ds)
    testset_loader, _, testset_image_paths = load_external_testset(
        args.testset_images_dir,
        args.batch_size,
        args.num_workers,
        device,
        args.input_size,
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

    model = Model(
        in_channels=3,
        n_classes=len(class_names),
        backbone_name=args.backbone_name,
        backbone_weights_path=args.backbone_weights_path,
        input_size=args.input_size,
        freeze_backbone=(args.stage == "baseline"),
        pretrained=args.pretrained,
        proj_dim=128,
        head_type=args.head_type,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        mlp_hidden_layers=args.mlp_hidden_layers,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_dropout=args.mlp_dropout,
    ).to(device)

    if args.backbone_weights_path:
        print(f"Backbone weights initialized from {args.backbone_weights_path}")
    print(f"Using classifier head: {model.classifier_description} (type={args.head_type})")

    checkpoint_model_config = {
        "in_channels": 3,
        "n_classes": len(class_names),
        "backbone_preset": args.backbone_preset,
        "backbone_name": args.backbone_name,
        "num_folds": args.num_folds,
        "fold_index": args.fold_index,
        "input_size": args.input_size,
        "pretrained": False,
        "proj_dim": 128,
        "head_type": args.head_type,
        "head_hidden_dim": args.head_hidden_dim,
        "head_dropout": args.head_dropout,
        "mlp_hidden_layers": args.mlp_hidden_layers,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "mlp_dropout": args.mlp_dropout,
    }

    criterion = None
    if args.stage != "pretrain":
        criterion = build_supervised_criterion(args.loss_name, train_ds, len(class_names), device)

    optimizer, scheduler = configure_stage(model, args)
    projected_prevalence = 0.01
    best_valid_projected_ppv = float("-inf")
    best_valid_fpr = float("inf")
    best_save_path = os.path.join(args.save_dir, f"{args.experiment_id}_best.pt")
    final_save_path = os.path.join(args.save_dir, f"{args.experiment_id}_final.pt")
    roi_refresh_state = {"completed": False}

    for epoch in range(args.epochs):
        if args.roi_guided_training and not roi_refresh_state["completed"] and epoch >= args.roi_start_epoch:
            refresh_train_roi_guidance(args, model, train_ds, device, epoch)
            roi_refresh_state["completed"] = True

        model.train()
        train_loss = 0.0
        train_ce = 0.0
        train_supmin = 0.0
        train_suppro = 0.0
        train_correct = 0
        train_total = 0

        for images1, images2, labels in train_loader:
            images1 = images1.to(device)
            images2 = images2.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()

            if args.stage == "pretrain":
                out1 = model(images1, return_embedding=True)
                out2 = model(images2, return_embedding=True)
                emb1 = out1["embedding"]
                emb2 = out2["embedding"]
                emb_pair = torch.stack([emb1, emb2], dim=1)

                loss_ce = torch.tensor(0.0, device=device)
                if args.loss_name == "suppro":
                    with torch.no_grad():
                        class_counts = torch.bincount(labels, minlength=len(class_names)).float().to(device)
                        class_weights = 1.0 / torch.clamp(class_counts, min=1.0)
                        class_weights = class_weights / class_weights.mean()
                    loss_suppro = suppro_loss(
                        emb_pair,
                        labels,
                        temperature=args.temperature,
                        base_temperature=args.base_temperature,
                        class_weights=class_weights,
                    )
                    loss_supmin = torch.tensor(0.0, device=device)
                    loss = loss_suppro
                else:
                    loss_supmin = 0.5 * (supmin_loss(emb1, labels) + supmin_loss(emb2, labels))
                    loss_suppro = torch.tensor(0.0, device=device)
                    loss = loss_supmin

                preds = torch.zeros_like(labels)
            else:
                logits1 = model(images1)
                logits2 = model(images2)
                loss_ce = 0.5 * (criterion(logits1, labels) + criterion(logits2, labels))
                loss_supmin = torch.tensor(0.0, device=device)
                loss_suppro = torch.tensor(0.0, device=device)
                loss = loss_ce
                preds = torch.argmax(logits1, dim=1)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss += loss.item() * batch_size
            train_ce += loss_ce.item() * batch_size
            train_supmin += loss_supmin.item() * batch_size
            train_suppro += loss_suppro.item() * batch_size
            if args.stage != "pretrain":
                train_correct += (preds == labels).sum().item()
            train_total += batch_size

        if scheduler is not None:
            scheduler.step()

        avg_train_loss = train_loss / max(1, train_total)
        avg_train_ce = train_ce / max(1, train_total)
        avg_train_supmin = train_supmin / max(1, train_total)
        avg_train_suppro = train_suppro / max(1, train_total)
        train_accuracy = (
            train_correct / max(1, train_total) if args.stage != "pretrain" else float("nan")
        )

        if args.stage == "pretrain":
            print(
                f"Epoch {epoch + 1:02d}/{args.epochs} | "
                f"Pretrain Loss: {avg_train_loss:.4f} | "
                f"SupPro: {avg_train_suppro:.4f} | SupMin: {avg_train_supmin:.4f}"
            )
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "stage": "pretrain",
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "pretrain/loss_total": avg_train_loss,
                    "pretrain/loss_ce": avg_train_ce,
                    "pretrain/loss_supmin": avg_train_supmin,
                    "pretrain/loss_suppro": avg_train_suppro,
                },
                step=epoch + 1,
            )
            continue

        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        valid_targets = []
        valid_scores = []

        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                valid_loss += loss.item() * images.size(0)
                predictions = torch.argmax(outputs, dim=1)
                valid_correct += (predictions == labels).sum().item()
                valid_total += labels.size(0)

                probs = torch.softmax(outputs, dim=1)[:, 1]
                valid_scores.extend(probs.detach().cpu().tolist())
                valid_targets.extend(labels.detach().cpu().tolist())

        avg_valid_loss = valid_loss / max(1, valid_total)
        valid_accuracy = valid_correct / max(1, valid_total)
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

        extra_payload = {
            "stage": args.stage,
            "loss_name": args.loss_name,
            "train/accuracy": train_accuracy,
            "valid/accuracy": valid_accuracy,
            "train/loss_ce": avg_train_ce,
            "train/loss_supmin": avg_train_supmin,
            "train/loss_suppro": avg_train_suppro,
        }
        if gradcam_payload is not None:
            extra_payload.update(gradcam_payload)

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
            extra_payload=extra_payload,
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
            torch.save(
                create_model_checkpoint(
                    model,
                    checkpoint_model_config,
                    extra_metadata={
                        "experiment_id": args.experiment_id,
                        "epoch": epoch + 1,
                        "num_folds": args.num_folds,
                        "fold_index": args.fold_index,
                        "selected_threshold": valid_threshold,
                        "stage": args.stage,
                        "loss_name": args.loss_name,
                    },
                ),
                best_save_path,
            )
            print(
                f"   -> Saved new best model to {best_save_path} "
                f"(1% PPV: {valid_projected_metrics['Projected PPV']:.4f}, "
                f"FPR: {valid_metrics['FPR']:.4f}, Threshold: {valid_threshold:.4f})"
            )

    if args.stage == "pretrain":
        encoder_path = os.path.join(args.save_dir, f"{args.experiment_id}_encoder.pt")
        torch.save(
            {
                "backbone": model.backbone.state_dict(),
                "proj_head": model.proj_head.state_dict(),
                "backbone_name": args.backbone_name,
                "backbone_preset": args.backbone_preset,
                "num_folds": args.num_folds,
                "fold_index": args.fold_index,
                "input_size": args.input_size,
                "loss_name": args.loss_name,
                "model_config": checkpoint_model_config,
            },
            encoder_path,
        )
        print(f"Saved encoder checkpoint: {encoder_path}")
    else:
        torch.save(
            create_model_checkpoint(
                model,
                checkpoint_model_config,
                extra_metadata={
                    "experiment_id": args.experiment_id,
                    "epoch": args.epochs,
                    "num_folds": args.num_folds,
                    "fold_index": args.fold_index,
                    "stage": args.stage,
                    "loss_name": args.loss_name,
                },
            ),
            final_save_path,
        )
        print(f"Saved final model: {final_save_path}")

        if args.post_train_gradcam:
            run_post_training_gradcam(
                args=args,
                model=model,
                device=device,
                final_save_path=final_save_path,
                best_save_path=best_save_path,
            )

    print(f"Class mapping: {class_names}")
    print("Training finished! Check your WandB dashboard.")
    wandb.finish()


if __name__ == "__main__":
    main(get_args_parser().parse_args())
