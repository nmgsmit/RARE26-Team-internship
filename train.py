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
from torch.utils.data import DataLoader, TensorDataset

from data import SimpleDataset, build_eval_transform, prepare_datasets
from gradcam import (
    compute_vit_gradcam_batch,
    evaluate_gradcam_barrett_dataset,
    evaluate_gradcam_segmentation_dataset,
)
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
}
CLASSIFIER_INPUT_CHOICES = ("pooled", "projection")
FINETUNE_TRAIN_MODE_CHOICES = ("last_block", "probe")
SMOTE_FEATURE_SPACE_CHOICES = ("pooled", "projection")
HEAD_TYPE_CHOICES = (
    "linear",
    "ln_linear",
    "mlp_fullwidth",
    "mlp_bottleneck",
    "residual_bottleneck",
    "cosine_linear",
)


def sampling_strategy_arg(value):
    try:
        return float(value)
    except ValueError:
        return value


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
            "pretraining. This is intended for curriculum-style re-pretraining."
        ),
    )
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument(
        "--classifier-input",
        type=str,
        default=None,
        choices=CLASSIFIER_INPUT_CHOICES,
        help=(
            "Feature space consumed by the classifier head. Defaults to pooled features for "
            "standard finetuning and projection embeddings for SMOTE finetuning."
        ),
    )
    parser.add_argument(
        "--finetune-train-mode",
        type=str,
        default=None,
        choices=FINETUNE_TRAIN_MODE_CHOICES,
        help=(
            "Finetune optimization setup. 'last_block' matches the standard image finetune, "
            "while 'probe' freezes the encoder and only learns the classifier head."
        ),
    )
    parser.add_argument(
        "--finetune-with-smote",
        action="store_true",
        help=(
            "Train a projection-space classifier on frozen encoder embeddings augmented with "
            "SMOTE, optional plausibility filtering, and constrained refinement."
        ),
    )
    parser.add_argument(
        "--smote-feature-space",
        type=str,
        default="projection",
        choices=SMOTE_FEATURE_SPACE_CHOICES,
        help="Feature space used for SMOTE generation, filtering, refinement, and probe training.",
    )
    parser.add_argument("--smote-neighbors", type=int, default=5)
    parser.add_argument(
        "--smote-sampling-strategy",
        type=sampling_strategy_arg,
        default="minority",
        help="SMOTE sampling strategy passed to imbalanced-learn, e.g. minority, auto, or a float ratio.",
    )
    parser.add_argument(
        "--smote-synthetic-ratio",
        type=float,
        default=None,
        help="Optional target synthetic-to-real sample ratio for minority oversampling.",
    )
    parser.add_argument(
        "--smote-energy-filter",
        action="store_true",
        help="Enable energy-based filtering of draft SMOTE embeddings before head training.",
    )
    parser.add_argument(
        "--smote-knn-filter",
        action="store_true",
        help="Enable kNN plausibility filtering of draft SMOTE embeddings before head training.",
    )
    parser.add_argument(
        "--smote-knn-neighbors",
        type=int,
        default=None,
        help="Neighborhood size used by the kNN plausibility filter. Defaults to --smote-neighbors.",
    )
    parser.add_argument(
        "--smote-knn-support-quantile",
        type=float,
        default=0.5,
        help="Quantile of real-minority support used as the kNN acceptance threshold.",
    )
    parser.add_argument(
        "--smote-knn-minority-purity",
        type=float,
        default=1.0,
        help="Required minority-label purity among a synthetic sample's nearest real neighbors.",
    )
    parser.add_argument(
        "--smote-knn-margin",
        type=float,
        default=0.0,
        help="Minimum nearest-neighbor similarity margin between minority and majority support.",
    )
    parser.add_argument(
        "--smote-knn-center-aware",
        action="store_true",
        help="Evaluate kNN plausibility within training centers before accepting a synthetic embedding.",
    )
    parser.add_argument(
        "--smote-energy-refine-steps",
        type=int,
        default=0,
        help="Number of gradient-based refinement steps applied to accepted SMOTE embeddings.",
    )
    parser.add_argument(
        "--smote-energy-refine-step-size",
        type=float,
        default=0.05,
        help="Step size for each SMOTE energy refinement update.",
    )
    parser.add_argument("--smote-energy-epochs", type=int, default=25)
    parser.add_argument("--smote-energy-lr", type=float, default=1e-3)
    parser.add_argument("--smote-energy-weight-decay", type=float, default=1e-4)
    parser.add_argument("--smote-energy-batch-size", type=int, default=256)
    parser.add_argument("--smote-energy-hidden-dim", type=int, default=256)
    parser.add_argument("--smote-energy-layers", type=int, default=2)
    parser.add_argument("--smote-energy-dropout", type=float, default=0.1)
    parser.add_argument("--smote-energy-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--smote-energy-noise-std", type=float, default=0.15)
    parser.add_argument("--smote-energy-noise-copies", type=int, default=2)
    parser.add_argument("--smote-energy-majority-ratio", type=float, default=1.0)
    parser.add_argument("--smote-energy-refine-anchor-weight", type=float, default=5.0)
    parser.add_argument("--smote-energy-refine-margin-weight", type=float, default=2.0)
    parser.add_argument("--smote-energy-refine-target-margin", type=float, default=0.05)
    parser.add_argument(
        "--smote-warmstart-epochs",
        type=int,
        default=3,
        help=(
            "Number of embedding-space warm-start epochs run on real + synthetic samples "
            "before standard image finetuning when --finetune-train-mode is not probe."
        ),
    )

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
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
        help="Convenience switch for the backbone setup. 'dinov3' uses timm pretrained DINOv3 at 224px. 'gastronet' uses the GastroNet DINOv2 checkpoint at 336px.",
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
        help="Optional manual override for the backbone checkpoint path. The gastronet preset defaults to ../Gastronet/dinov2.pth.",
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
        "--testset-images-dir",
        type=str,
        default="./data/EVC_Barretts_FullSet/images",
        help="Path to external testset images used for per-epoch testset metrics.",
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
            "uses those ROIs to guide positive-sample crops. This is the path you use for ROI-guided pretraining."
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
        "--post-train-gradcam-dataset-root",
        type=str,
        default=None,
        help=(
            "Root directory for the post-training Barrett Grad-CAM evaluation dataset. "
            "Defaults to the parent directory of --testset-images-dir."
        ),
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


def resolve_runtime_config(args):
    requested_classifier_input = args.classifier_input
    requested_finetune_train_mode = args.finetune_train_mode
    if args.loss_name is None:
        args.loss_name = args.method
    if args.loss_name is None:
        args.loss_name = {
            "baseline": "class-balanced",
            "pretrain": "supmin",
            "finetune": "ce",
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
    if args.finetune_train_mode is None:
        args.finetune_train_mode = "probe" if args.finetune_with_smote else "last_block"
    if args.classifier_input is None:
        if args.finetune_with_smote and args.finetune_train_mode == "probe":
            args.classifier_input = "projection"
        else:
            args.classifier_input = "pooled"
    if args.finetune_with_smote and args.stage != "finetune":
        raise ValueError("--finetune-with-smote is only supported with --stage finetune.")
    if (
        args.finetune_with_smote
        and args.finetune_train_mode == "probe"
        and args.classifier_input != args.smote_feature_space
    ):
        raise ValueError(
            "SMOTE finetuning requires --classifier-input to match --smote-feature-space so "
            "the classifier operates in the same embedding space used for synthesis."
        )
    if (
        args.smote_energy_filter
        or args.smote_knn_filter
        or args.smote_energy_refine_steps > 0
    ) and not args.finetune_with_smote:
        raise ValueError(
            "SMOTE filtering/refinement requires --finetune-with-smote."
        )
    if args.smote_energy_refine_steps < 0:
        raise ValueError(
            f"--smote-energy-refine-steps must be >= 0, got {args.smote_energy_refine_steps}."
        )
    if args.smote_synthetic_ratio is not None and args.smote_synthetic_ratio <= 0.0:
        raise ValueError(
            f"--smote-synthetic-ratio must be positive when provided, got {args.smote_synthetic_ratio}."
        )
    if args.smote_knn_neighbors is not None and args.smote_knn_neighbors <= 0:
        raise ValueError(
            f"--smote-knn-neighbors must be positive when provided, got {args.smote_knn_neighbors}."
        )
    if not 0.0 <= args.smote_knn_support_quantile <= 1.0:
        raise ValueError(
            "--smote-knn-support-quantile must be in [0, 1], "
            f"got {args.smote_knn_support_quantile}."
        )
    if not 0.0 < args.smote_knn_minority_purity <= 1.0:
        raise ValueError(
            f"--smote-knn-minority-purity must be in (0, 1], got {args.smote_knn_minority_purity}."
        )
    if args.smote_energy_refine_anchor_weight < 0.0:
        raise ValueError(
            "--smote-energy-refine-anchor-weight must be >= 0, "
            f"got {args.smote_energy_refine_anchor_weight}."
        )
    if args.smote_energy_refine_margin_weight < 0.0:
        raise ValueError(
            "--smote-energy-refine-margin-weight must be >= 0, "
            f"got {args.smote_energy_refine_margin_weight}."
        )
    if args.smote_energy_refine_target_margin < 0.0:
        raise ValueError(
            "--smote-energy-refine-target-margin must be >= 0, "
            f"got {args.smote_energy_refine_target_margin}."
        )
    if args.smote_warmstart_epochs < 0:
        raise ValueError(
            f"--smote-warmstart-epochs must be >= 0, got {args.smote_warmstart_epochs}."
        )
    if (
        args.finetune_with_smote
        and args.finetune_train_mode != "probe"
        and requested_classifier_input is None
    ):
        print(
            "SMOTE + real finetune detected without an explicit classifier space. "
            "Defaulting to pooled classifier features so the SMOTE stage warm-starts the "
            "same head used by the stronger image finetune baseline."
        )
    if (
        args.finetune_with_smote
        and args.finetune_train_mode == "probe"
        and requested_finetune_train_mode is None
    ):
        print(
            "SMOTE run is using the default finetune_train_mode=probe. "
            "Set --finetune-train-mode last_block to use SMOTE as a warm-start for actual finetuning."
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


def suppro_loss(features, labels, temperature=0.07, base_temperature=0.07, class_weights=None):
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
    return Path(args.post_train_gradcam_dataset_root or Path(args.testset_images_dir).parent)


def validate_post_train_gradcam_dataset(args):
    dataset_root = resolve_post_train_gradcam_dataset_root(args)
    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "annotations_bmp"
    if not images_dir.exists():
        raise FileNotFoundError(f"Grad-CAM images directory not found: {images_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Grad-CAM annotations directory not found: {annotations_dir}")
    return dataset_root


def run_post_training_gradcam(args, model, device, final_save_path, best_save_path, segmentation_loader=None):
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

    if segmentation_loader is not None:
        segmentation_payload = evaluate_gradcam_segmentation_dataset(
            model=model,
            loader=segmentation_loader,
            device=device,
            target_class=args.gradcam_target_class,
            threshold=args.gradcam_threshold,
            max_log_samples=args.gradcam_log_samples,
            skip_empty_masks=args.gradcam_skip_empty_masks,
            split_name="segmentation",
        )
        if segmentation_payload:
            wandb.log(segmentation_payload)
            for key, value in segmentation_payload.items():
                wandb.summary[key] = value
        print(
            "Post-training segmentation Grad-CAM summary | "
            f"Mean Dice: {segmentation_payload['segmentation/mean_dice']:.4f} | "
            f"Scored: {segmentation_payload['segmentation/dice_scored_samples']} | "
            f"Skipped empty: {segmentation_payload['segmentation/dice_skipped_empty_masks']}"
        )


def _flatten_roi_source_counts(source_counts):
    return {f"train/roi_source_{source}_count": count for source, count in sorted(source_counts.items())}


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
    eval_loader = DataLoader(
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


class EnergyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, hidden_layers=2, dropout=0.1):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"EnergyMLP hidden_dim must be positive, got {hidden_dim}.")
        if hidden_layers <= 0:
            raise ValueError(f"EnergyMLP hidden_layers must be positive, got {hidden_layers}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"EnergyMLP dropout must be in [0, 1), got {dropout}.")

        layers = []
        in_features = int(input_dim)
        for _ in range(int(hidden_layers)):
            layers.append(nn.Linear(in_features, int(hidden_dim)))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            in_features = int(hidden_dim)
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def sample_rows_with_replacement(rng, X, sample_count):
    if sample_count <= 0:
        return np.empty((0, X.shape[1]), dtype=np.float32)
    replace = len(X) < sample_count
    indices = rng.choice(len(X), size=sample_count, replace=replace)
    return X[indices].astype(np.float32)


def make_tensor_only_loader(features, batch_size, shuffle=False):
    features_tensor = torch.tensor(features, dtype=torch.float32)
    return DataLoader(TensorDataset(features_tensor), batch_size=batch_size, shuffle=shuffle)


def make_feature_loader(features, labels, args, shuffle):
    return DataLoader(
        TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        ),
        batch_size=args.batch_size,
        shuffle=shuffle,
    )


def l2_normalize_rows(features, eps=1e-12):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature array, got shape {features.shape}.")
    if len(features) == 0:
        return features.astype(np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, eps, None)
    return (features / norms).astype(np.float32)


def normalize_features_for_space(features, feature_space):
    features = np.asarray(features, dtype=np.float32)
    if feature_space == "projection":
        return l2_normalize_rows(features)
    return features.astype(np.float32)


def compute_label_fraction(labels, label_value):
    labels = np.asarray(labels, dtype=int)
    if len(labels) == 0:
        return 0.0
    return float(np.mean(labels == int(label_value)))


def build_smote_filter_mode(args):
    filters = []
    if args.smote_energy_filter:
        filters.append("energy")
    if args.smote_knn_filter:
        filters.append("knn")
    return "+".join(filters) if filters else "none"


def score_energy_model(energy_model, features, args, device):
    if len(features) == 0:
        return np.empty(0, dtype=np.float32)

    energy_model.eval()
    energies = []
    loader = make_tensor_only_loader(features, batch_size=args.smote_energy_batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_features,) in loader:
            batch_energy = energy_model(batch_features.to(device))
            energies.append(batch_energy.detach().cpu().numpy())
    return np.concatenate(energies, axis=0).astype(np.float32)


def build_energy_negatives(features, labels, args):
    class_counts = np.bincount(labels, minlength=2)
    minority_class = int(np.argmin(class_counts))
    minority = normalize_features_for_space(
        features[labels == minority_class], args.smote_feature_space
    )
    majority = normalize_features_for_space(
        features[labels != minority_class], args.smote_feature_space
    )

    if len(minority) == 0 or len(majority) == 0:
        raise ValueError("Energy-based SMOTE filtering requires both classes in the train split.")

    rng = np.random.default_rng(args.seed)
    majority_target = max(1, int(round(float(args.smote_energy_majority_ratio) * len(minority))))
    negative_parts = [sample_rows_with_replacement(rng, majority, majority_target)]

    noise_copies = max(0, int(args.smote_energy_noise_copies))
    if noise_copies > 0:
        base = np.repeat(minority, noise_copies, axis=0)
        noise = rng.normal(0.0, float(args.smote_energy_noise_std), size=base.shape).astype(np.float32)
        negative_parts.append(
            normalize_features_for_space(base + noise, args.smote_feature_space)
        )

    negatives = np.concatenate(negative_parts, axis=0).astype(np.float32)
    return minority, negatives, minority_class


def train_smote_energy_model(features, labels, args, device):
    positives, negatives, minority_class = build_energy_negatives(features, labels, args)
    energy_features = np.concatenate([positives, negatives], axis=0).astype(np.float32)
    energy_labels = np.concatenate(
        [
            np.ones(len(positives), dtype=np.float32),
            np.zeros(len(negatives), dtype=np.float32),
        ],
        axis=0,
    )

    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(energy_features))
    energy_features = energy_features[permutation]
    energy_labels = energy_labels[permutation]

    energy_model = EnergyMLP(
        input_dim=features.shape[1],
        hidden_dim=args.smote_energy_hidden_dim,
        hidden_layers=args.smote_energy_layers,
        dropout=args.smote_energy_dropout,
    ).to(device)
    optimizer = AdamW(
        energy_model.parameters(),
        lr=args.smote_energy_lr,
        weight_decay=args.smote_energy_weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(energy_features, dtype=torch.float32),
            torch.tensor(energy_labels, dtype=torch.float32),
        ),
        batch_size=args.smote_energy_batch_size,
        shuffle=True,
    )

    final_loss = 0.0
    for _ in range(max(1, int(args.smote_energy_epochs))):
        energy_model.train()
        epoch_loss = 0.0
        epoch_total = 0
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            logits = -energy_model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_features.size(0)
            epoch_total += batch_features.size(0)
        final_loss = epoch_loss / max(1, epoch_total)

    real_energies = score_energy_model(energy_model, positives, args, device)
    majority_energies = score_energy_model(energy_model, features[labels != minority_class], args, device)
    threshold = float(np.quantile(real_energies, float(args.smote_energy_threshold_quantile)))
    stats = {
        "minority_class": minority_class,
        "energy_train_loss": float(final_loss),
        "energy_threshold": threshold,
        "real_energy_mean": float(real_energies.mean()),
        "real_energy_std": float(real_energies.std()),
        "real_energy_p95": float(np.quantile(real_energies, 0.95)),
        "majority_energy_mean": float(majority_energies.mean()),
        "majority_energy_std": float(majority_energies.std()),
    }
    return energy_model, stats


def _get_rowwise_kth_largest(similarities, k):
    if similarities.ndim != 2:
        raise ValueError(f"Expected a 2D similarity matrix, got shape {similarities.shape}.")
    if similarities.shape[1] == 0:
        return np.full(similarities.shape[0], -np.inf, dtype=np.float32)
    k = max(1, min(int(k), similarities.shape[1]))
    kth_index = similarities.shape[1] - k
    return np.partition(similarities, kth_index, axis=1)[:, kth_index].astype(np.float32)


def _get_neighbor_purity(similarities, labels, target_class, k):
    if similarities.ndim != 2:
        raise ValueError(f"Expected a 2D similarity matrix, got shape {similarities.shape}.")
    if similarities.shape[1] == 0:
        return np.zeros(similarities.shape[0], dtype=np.float32)
    k = max(1, min(int(k), similarities.shape[1]))
    topk_indices = np.argpartition(similarities, similarities.shape[1] - k, axis=1)[:, -k:]
    topk_labels = labels[topk_indices]
    return (topk_labels == int(target_class)).mean(axis=1).astype(np.float32)


def _compute_real_minority_support_distribution(minority_features, minority_centers, k, center_aware):
    supports = []
    for index in range(len(minority_features)):
        if center_aware:
            candidate_mask = minority_centers == minority_centers[index]
            candidate_mask[index] = False
            if not np.any(candidate_mask):
                candidate_mask = np.ones(len(minority_features), dtype=bool)
                candidate_mask[index] = False
        else:
            candidate_mask = np.ones(len(minority_features), dtype=bool)
            candidate_mask[index] = False

        candidate_features = minority_features[candidate_mask]
        if len(candidate_features) == 0:
            supports.append(1.0)
            continue
        similarities = minority_features[index : index + 1] @ candidate_features.T
        supports.append(float(_get_rowwise_kth_largest(similarities, k)[0]))
    return np.asarray(supports, dtype=np.float32)


def filter_smote_knn_embeddings(features, labels, centers, synthetic_features, args):
    synthetic_features = normalize_features_for_space(synthetic_features, args.smote_feature_space)
    if len(synthetic_features) == 0:
        return synthetic_features, {
            "knn_filter_neighbors": 0,
            "knn_filter_threshold": 0.0,
            "knn_filter_support_quantile": float(args.smote_knn_support_quantile),
            "knn_filter_minority_purity": float(args.smote_knn_minority_purity),
            "knn_filter_margin": float(args.smote_knn_margin),
            "knn_filter_center_aware": int(bool(args.smote_knn_center_aware)),
            "knn_filter_draft_support_mean": 0.0,
            "knn_filter_draft_support_std": 0.0,
            "knn_filter_accepted_support_mean": 0.0,
            "knn_filter_accepted_support_std": 0.0,
            "knn_filter_draft_purity_mean": 0.0,
            "knn_filter_accepted_purity_mean": 0.0,
            "knn_filter_draft_margin_mean": 0.0,
            "knn_filter_accepted_margin_mean": 0.0,
            "knn_filter_accepted_total": 0,
            "knn_filter_rejected_total": 0,
        }

    normalized_features = l2_normalize_rows(features)
    labels = np.asarray(labels, dtype=int)
    centers = np.asarray(centers)
    class_counts = np.bincount(labels, minlength=2)
    minority_class = int(np.argmin(class_counts))
    minority_features = normalized_features[labels == minority_class]
    minority_centers = centers[labels == minority_class]
    if len(minority_features) == 0:
        raise ValueError("kNN SMOTE filtering requires at least one minority feature.")

    k = int(args.smote_knn_neighbors or args.smote_neighbors)
    real_supports = _compute_real_minority_support_distribution(
        minority_features,
        minority_centers,
        k=k,
        center_aware=bool(args.smote_knn_center_aware),
    )
    support_threshold = float(
        np.quantile(real_supports, float(args.smote_knn_support_quantile))
    )

    if args.smote_knn_center_aware:
        support_candidates = []
        purity_candidates = []
        margin_candidates = []
        for center in np.unique(centers):
            center_mask = centers == center
            center_real = normalized_features[center_mask]
            center_labels = labels[center_mask]
            center_minority = center_real[center_labels == minority_class]
            if len(center_minority) == 0:
                continue

            minority_similarities = synthetic_features @ center_minority.T
            support_candidates.append(_get_rowwise_kth_largest(minority_similarities, k))
            nearest_minority = np.max(minority_similarities, axis=1)

            center_majority = center_real[center_labels != minority_class]
            if len(center_majority) > 0:
                nearest_majority = np.max(synthetic_features @ center_majority.T, axis=1)
            else:
                nearest_majority = np.full(len(synthetic_features), -np.inf, dtype=np.float32)

            center_similarities = synthetic_features @ center_real.T
            purity_candidates.append(
                _get_neighbor_purity(
                    center_similarities,
                    center_labels.astype(int),
                    minority_class,
                    k,
                )
            )
            margin_candidates.append((nearest_minority - nearest_majority).astype(np.float32))

        if not support_candidates:
            support_candidates = [
                _get_rowwise_kth_largest(synthetic_features @ minority_features.T, k)
            ]
            purity_candidates = [
                _get_neighbor_purity(
                    synthetic_features @ normalized_features.T,
                    labels,
                    minority_class,
                    k,
                )
            ]
            majority_features = normalized_features[labels != minority_class]
            nearest_minority = np.max(synthetic_features @ minority_features.T, axis=1)
            if len(majority_features) > 0:
                nearest_majority = np.max(synthetic_features @ majority_features.T, axis=1)
            else:
                nearest_majority = np.full(len(synthetic_features), -np.inf, dtype=np.float32)
            margin_candidates = [(nearest_minority - nearest_majority).astype(np.float32)]

        support_matrix = np.stack(support_candidates, axis=1)
        purity_matrix = np.stack(purity_candidates, axis=1)
        margin_matrix = np.stack(margin_candidates, axis=1)
        best_center_indices = np.argmax(support_matrix, axis=1)
        row_indices = np.arange(len(synthetic_features))
        draft_support = support_matrix[row_indices, best_center_indices]
        draft_purity = purity_matrix[row_indices, best_center_indices]
        draft_margin = margin_matrix[row_indices, best_center_indices]
    else:
        minority_similarities = synthetic_features @ minority_features.T
        draft_support = _get_rowwise_kth_largest(minority_similarities, k)
        draft_purity = _get_neighbor_purity(
            synthetic_features @ normalized_features.T,
            labels,
            minority_class,
            k,
        )
        majority_features = normalized_features[labels != minority_class]
        nearest_minority = np.max(minority_similarities, axis=1)
        if len(majority_features) > 0:
            nearest_majority = np.max(synthetic_features @ majority_features.T, axis=1)
        else:
            nearest_majority = np.full(len(synthetic_features), -np.inf, dtype=np.float32)
        draft_margin = (nearest_minority - nearest_majority).astype(np.float32)

    keep_mask = (
        (draft_support >= support_threshold)
        & (draft_purity >= float(args.smote_knn_minority_purity))
        & (draft_margin >= float(args.smote_knn_margin))
    )
    accepted_features = synthetic_features[keep_mask].astype(np.float32)
    accepted_support = draft_support[keep_mask]
    accepted_purity = draft_purity[keep_mask]
    accepted_margin = draft_margin[keep_mask]

    stats = {
        "knn_filter_neighbors": int(k),
        "knn_filter_threshold": support_threshold,
        "knn_filter_support_quantile": float(args.smote_knn_support_quantile),
        "knn_filter_minority_purity": float(args.smote_knn_minority_purity),
        "knn_filter_margin": float(args.smote_knn_margin),
        "knn_filter_center_aware": int(bool(args.smote_knn_center_aware)),
        "knn_filter_draft_support_mean": float(draft_support.mean()),
        "knn_filter_draft_support_std": float(draft_support.std()),
        "knn_filter_accepted_support_mean": float(accepted_support.mean())
        if accepted_support.size > 0
        else 0.0,
        "knn_filter_accepted_support_std": float(accepted_support.std())
        if accepted_support.size > 0
        else 0.0,
        "knn_filter_draft_purity_mean": float(draft_purity.mean()),
        "knn_filter_accepted_purity_mean": float(accepted_purity.mean())
        if accepted_purity.size > 0
        else 0.0,
        "knn_filter_draft_margin_mean": float(draft_margin.mean()),
        "knn_filter_accepted_margin_mean": float(accepted_margin.mean())
        if accepted_margin.size > 0
        else 0.0,
        "knn_filter_accepted_total": int(keep_mask.sum()),
        "knn_filter_rejected_total": int((~keep_mask).sum()),
    }
    return accepted_features, stats


def refine_synthetic_embeddings(energy_model, features, reference_features, labels, args, device):
    if len(features) == 0 or int(args.smote_energy_refine_steps) <= 0:
        return features.astype(np.float32)

    refined = []
    energy_model.eval()
    loader = make_tensor_only_loader(features, batch_size=args.smote_energy_batch_size, shuffle=False)
    step_size = float(args.smote_energy_refine_step_size)
    anchor_weight = float(args.smote_energy_refine_anchor_weight)
    margin_weight = float(args.smote_energy_refine_margin_weight)
    target_margin = float(args.smote_energy_refine_target_margin)
    labels = np.asarray(labels, dtype=int)
    minority_class = int(np.argmin(np.bincount(labels, minlength=2)))
    minority_tensor = torch.tensor(
        reference_features[labels == minority_class],
        dtype=torch.float32,
        device=device,
    )
    majority_tensor = torch.tensor(
        reference_features[labels != minority_class],
        dtype=torch.float32,
        device=device,
    )

    for (batch_features,) in loader:
        anchor = batch_features.to(device)
        z = anchor
        for _ in range(int(args.smote_energy_refine_steps)):
            z = z.detach().requires_grad_(True)
            objective = energy_model(z).mean()
            if anchor_weight > 0.0:
                objective = objective + anchor_weight * ((z - anchor) ** 2).sum(dim=1).mean()
            if margin_weight > 0.0 and len(minority_tensor) > 0 and len(majority_tensor) > 0:
                nearest_minority = torch.max(z @ minority_tensor.T, dim=1).values
                nearest_majority = torch.max(z @ majority_tensor.T, dim=1).values
                margin_shortfall = F.relu(target_margin + nearest_majority - nearest_minority)
                objective = objective + margin_weight * margin_shortfall.mean()
            gradient = torch.autograd.grad(objective, z)[0]
            with torch.no_grad():
                z = z - step_size * gradient
                if args.smote_feature_space == "projection":
                    z = F.normalize(z, dim=-1)
        refined.append(z.detach().cpu().numpy())

    return normalize_features_for_space(np.concatenate(refined, axis=0), args.smote_feature_space)


def build_smote_variant_name(args):
    filter_mode = build_smote_filter_mode(args).replace("+", "_")
    refine_tag = "constrained_refine" if int(args.smote_energy_refine_steps) > 0 else "no_refine"
    return f"{args.smote_feature_space}_smote_{filter_mode}_{refine_tag}"


def fit_smote_resampler(features, labels, centers, args, device):
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError(
            "SMOTE finetuning requires imbalanced-learn. Install the branch requirements with "
            "`pip install -r requirements.txt`."
        ) from exc

    features = normalize_features_for_space(features, args.smote_feature_space)
    labels = np.asarray(labels, dtype=int)
    centers = np.asarray(centers)
    class_counts = np.bincount(labels, minlength=2)
    if np.any(class_counts == 0):
        raise ValueError(f"SMOTE requires both classes in the train split, got {class_counts.tolist()}.")

    minority_count = int(np.min(class_counts))
    if minority_count < 2:
        raise ValueError(f"SMOTE needs at least two minority samples, got {class_counts.tolist()}.")

    requested_neighbors = int(args.smote_neighbors)
    n_neighbors = min(requested_neighbors, minority_count - 1)
    if n_neighbors != requested_neighbors:
        print(
            f"Reduced SMOTE k_neighbors from {requested_neighbors} to {n_neighbors} "
            f"because the minority class has {minority_count} samples."
        )

    sampling_strategy = args.smote_sampling_strategy
    if args.smote_synthetic_ratio is not None:
        minority_class = int(np.argmin(class_counts))
        synthetic_count = int(round(float(args.smote_synthetic_ratio) * len(labels)))
        target_count = int(class_counts[minority_class] + synthetic_count)
        sampling_strategy = {minority_class: target_count}
        print(
            "Using SMOTE synthetic-to-real ratio target: "
            f"{args.smote_synthetic_ratio} -> target class {minority_class} count {target_count}."
        )

    sampler = SMOTE(
        k_neighbors=n_neighbors,
        random_state=args.seed,
        sampling_strategy=sampling_strategy,
    )
    resampled_features, resampled_labels = sampler.fit_resample(features, labels)
    resampled_features = normalize_features_for_space(resampled_features, args.smote_feature_space)
    resampled_labels = resampled_labels.astype(int)

    synthetic_total = int(len(resampled_features) - len(features))
    synthetic_features = normalize_features_for_space(
        resampled_features[len(features):], args.smote_feature_space
    )
    accepted_synthetic = synthetic_features
    energy_model = None
    minority_class = int(np.argmin(class_counts))
    diagnostics = {
        "variant": build_smote_variant_name(args),
        "feature_space": args.smote_feature_space,
        "filter_mode": build_smote_filter_mode(args),
        "draft_total": synthetic_total,
        "accepted_total": 0,
        "rejected_total": 0,
        "accepted_ratio": 0.0,
        "energy_filter_enabled": int(bool(args.smote_energy_filter)),
        "knn_filter_enabled": int(bool(args.smote_knn_filter)),
        "refine_enabled": int(int(args.smote_energy_refine_steps) > 0),
        "refine_steps": int(args.smote_energy_refine_steps),
        "energy_model_trained": 0,
        "energy_filter_accepted_total": 0,
        "energy_filter_rejected_total": 0,
        "knn_filter_accepted_total": 0,
        "knn_filter_rejected_total": 0,
        "energy_train_loss": 0.0,
        "energy_threshold": 0.0,
        "real_energy_mean": 0.0,
        "real_energy_std": 0.0,
        "real_energy_p95": 0.0,
        "majority_energy_mean": 0.0,
        "majority_energy_std": 0.0,
        "draft_energy_mean": 0.0,
        "draft_energy_std": 0.0,
        "accepted_energy_mean": 0.0,
        "accepted_energy_std": 0.0,
        "refined_energy_mean": 0.0,
        "refined_energy_std": 0.0,
        "train_positive_fraction": compute_label_fraction(labels, 1),
        "synthetic_positive_fraction": 0.0,
        "final_positive_fraction": 0.0,
    }

    if synthetic_total == 0:
        return resampled_features, resampled_labels, diagnostics

    if args.smote_energy_filter or int(args.smote_energy_refine_steps) > 0:
        energy_model, energy_stats = train_smote_energy_model(features, labels, args, device)
        diagnostics.update(energy_stats)
        diagnostics["energy_model_trained"] = 1

        draft_energies = score_energy_model(energy_model, synthetic_features, args, device)
        diagnostics["draft_energy_mean"] = float(draft_energies.mean())
        diagnostics["draft_energy_std"] = float(draft_energies.std())

        if args.smote_energy_filter:
            keep_mask = draft_energies <= diagnostics["energy_threshold"]
            accepted_synthetic = synthetic_features[keep_mask].astype(np.float32)
            diagnostics["energy_filter_accepted_total"] = int(keep_mask.sum())
            diagnostics["energy_filter_rejected_total"] = int((~keep_mask).sum())
        else:
            diagnostics["energy_filter_accepted_total"] = int(len(accepted_synthetic))

    if args.smote_knn_filter:
        accepted_synthetic, knn_stats = filter_smote_knn_embeddings(
            features,
            labels,
            centers,
            accepted_synthetic,
            args,
        )
        diagnostics.update(knn_stats)

    if energy_model is not None and len(accepted_synthetic) > 0:
        accepted_energies = score_energy_model(energy_model, accepted_synthetic, args, device)
        diagnostics["accepted_energy_mean"] = float(accepted_energies.mean())
        diagnostics["accepted_energy_std"] = float(accepted_energies.std())
    elif energy_model is not None:
        diagnostics["accepted_energy_mean"] = 0.0
        diagnostics["accepted_energy_std"] = 0.0
    else:
        diagnostics["accepted_energy_mean"] = diagnostics["draft_energy_mean"]
        diagnostics["accepted_energy_std"] = diagnostics["draft_energy_std"]

    if int(args.smote_energy_refine_steps) > 0 and len(accepted_synthetic) > 0:
        accepted_synthetic = refine_synthetic_embeddings(
            energy_model,
            accepted_synthetic,
            features,
            labels,
            args,
            device,
        )
        refined_energies = score_energy_model(energy_model, accepted_synthetic, args, device)
        diagnostics["refined_energy_mean"] = float(refined_energies.mean())
        diagnostics["refined_energy_std"] = float(refined_energies.std())

    final_features = np.concatenate([features, accepted_synthetic], axis=0).astype(np.float32)
    final_labels = np.concatenate(
        [
            labels.astype(int),
            np.full(len(accepted_synthetic), minority_class, dtype=int),
        ],
        axis=0,
    )
    diagnostics["accepted_total"] = int(len(accepted_synthetic))
    diagnostics["rejected_total"] = int(synthetic_total - len(accepted_synthetic))
    diagnostics["accepted_ratio"] = diagnostics["accepted_total"] / max(1, synthetic_total)
    diagnostics["synthetic_positive_fraction"] = compute_label_fraction(
        np.full(len(accepted_synthetic), minority_class, dtype=int), 1
    )
    diagnostics["final_positive_fraction"] = compute_label_fraction(final_labels, 1)
    return final_features, final_labels, diagnostics


def build_supervised_criterion_from_labels(loss_name, labels, n_classes, device):
    if loss_name == "ce":
        print("Using standard cross entropy.")
        return nn.CrossEntropyLoss()

    label_tensor = torch.tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(label_tensor, minlength=n_classes).float()
    if torch.any(class_counts == 0):
        raise ValueError(f"At least one class has zero samples: {class_counts.tolist()}")

    class_weights = class_counts.sum() / (n_classes * class_counts)
    class_weights = class_weights.to(device)
    print(f"Using class-balanced cross entropy with weights: {class_weights.tolist()}")
    return nn.CrossEntropyLoss(weight=class_weights)


def build_supervised_criterion(loss_name, train_ds, n_classes, device):
    return build_supervised_criterion_from_labels(
        loss_name,
        train_ds.df["label"].tolist(),
        n_classes,
        device,
    )


def set_finetune_trainable_parameters(model, args):
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    for parameter in model.proj_head.parameters():
        parameter.requires_grad = False
    for parameter in model.cls_head.parameters():
        parameter.requires_grad = True

    if args.finetune_train_mode == "probe":
        return

    if args.finetune_train_mode != "last_block":
        raise ValueError(f"Unsupported finetune_train_mode {args.finetune_train_mode!r}.")

    if hasattr(model.backbone, "blocks") and len(model.backbone.blocks) > 0:
        for parameter in model.backbone.blocks[-1].parameters():
            parameter.requires_grad = True
    else:
        print("[WARN] backbone has no .blocks attribute; keeping the backbone frozen.")

    if model.classifier_input == "projection":
        for parameter in model.proj_head.parameters():
            parameter.requires_grad = True


def configure_stage(model, args):
    if args.stage == "baseline":
        trainable_parameters = list(model.cls_head.parameters())
        if model.classifier_input == "projection":
            for parameter in model.proj_head.parameters():
                parameter.requires_grad = True
            trainable_parameters = list(model.proj_head.parameters()) + trainable_parameters
        else:
            for parameter in model.proj_head.parameters():
                parameter.requires_grad = False
        optimizer = AdamW(trainable_parameters, lr=args.lr)
        scheduler = None
        return optimizer, scheduler

    if args.stage == "pretrain":
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True
        for parameter in model.proj_head.parameters():
            parameter.requires_grad = True
        for parameter in model.cls_head.parameters():
            parameter.requires_grad = False
        optimizer = SGD(
            list(model.backbone.parameters()) + list(model.proj_head.parameters()),
            lr=args.lr,
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
        set_finetune_trainable_parameters(model, args)

        optimizer = AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.lr,
        )
        scheduler = build_finetune_scheduler(optimizer, args.epochs)
        return optimizer, scheduler

    raise ValueError(f"Unknown stage: {args.stage}")


def extract_dataset_embeddings(model, dataset, args, device, feature_space):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            pooled_features = model.encode(images.to(device))
            if feature_space == "projection":
                batch_features = model.project(pooled_features)
            elif feature_space == "pooled":
                batch_features = pooled_features
            else:
                raise ValueError(f"Unsupported feature_space {feature_space!r}.")
            features.append(batch_features.detach().cpu().numpy())
            labels.append(batch_labels.numpy())
    features = np.concatenate(features, axis=0).astype(np.float32)
    return normalize_features_for_space(features, feature_space), np.concatenate(labels, axis=0).astype(int)


def run_smote_finetune(
    args,
    model,
    train_loader,
    train_ds,
    valid_loader,
    testset_loader,
    class_names,
    device,
    checkpoint_model_config,
    best_save_path,
    final_save_path,
):
    if not args.encoder_ckpt:
        raise ValueError("--encoder-ckpt is required for finetune stage")

    load_encoder_checkpoint(model, args.encoder_ckpt)
    print(f"Loaded encoder checkpoint from {args.encoder_ckpt}")
    set_finetune_trainable_parameters(model, args)

    eval_train_ds = SimpleDataset(train_ds.df, build_eval_transform(args.input_size))
    train_features, train_labels = extract_dataset_embeddings(
        model,
        eval_train_ds,
        args,
        device,
        feature_space=args.smote_feature_space,
    )
    train_centers = train_ds.df["center"].astype(str).to_numpy()
    print(
        "Extracted frozen train embeddings for SMOTE finetune | "
        f"feature_space={args.smote_feature_space} | classifier_input={model.classifier_input} | "
        f"samples={len(train_labels)} | class0={(train_labels == 0).sum()} | class1={(train_labels == 1).sum()}"
    )

    resampled_features, resampled_labels, smote_diagnostics = fit_smote_resampler(
        train_features,
        train_labels,
        train_centers,
        args,
        device,
    )
    print(
        "SMOTE diagnostics | "
        f"variant={smote_diagnostics['variant']} | "
        f"draft={smote_diagnostics['draft_total']} | "
        f"accepted={smote_diagnostics['accepted_total']} | "
        f"rejected={smote_diagnostics['rejected_total']}"
    )

    def run_validation_and_logging(epoch_index, optimizer, avg_train_loss, train_accuracy, criterion):
        nonlocal best_valid_projected_ppv, best_valid_fpr

        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        valid_scores = []
        valid_targets = []

        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                batch_size = labels.size(0)
                valid_loss += loss.item() * batch_size
                valid_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                valid_total += batch_size
                valid_scores.extend(torch.softmax(outputs, dim=1)[:, 1].detach().cpu().tolist())
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

        print(
            f"Epoch {epoch_index + 1:02d}/{args.epochs} | "
            f"SMOTE Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {avg_valid_loss:.4f} | Val Acc: {valid_accuracy:.4f} | "
            f"Val AUPRC: {valid_metrics['AUPRC']:.4f} | Val AUROC: {valid_metrics['AUROC']:.4f} | "
            f"Val PPV@90R: {valid_metrics['PPV@90RECALL']:.4f} | Val Thr: {valid_threshold:.4f} | "
            f"Val TPR: {valid_metrics['TPR']:.4f} | Val FPR: {valid_metrics['FPR']:.4f} | "
            f"1%Val PPV: {valid_projected_metrics['Projected PPV']:.4f} | "
            f"Test AUPRC: {test_metrics['AUPRC']:.4f} | Test AUROC: {test_metrics['AUROC']:.4f}"
        )

        log_metrics(
            epoch_index,
            optimizer,
            avg_train_loss,
            train_accuracy,
            avg_valid_loss,
            valid_accuracy,
            valid_metrics,
            test_metrics,
            valid_projected_metrics,
            test_projected_metrics,
            extra_payload={
                "stage": args.stage,
                "loss_name": args.loss_name,
                "train/accuracy": train_accuracy,
                "valid/accuracy": valid_accuracy,
                **smote_payload,
            },
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
                        "epoch": epoch_index + 1,
                        "selected_threshold": valid_threshold,
                        "stage": args.stage,
                        "loss_name": args.loss_name,
                        "finetune_with_smote": True,
                        "smote_diagnostics": smote_diagnostics,
                    },
                ),
                best_save_path,
            )
            print(
                f"   -> Saved new best model to {best_save_path} "
                f"(1% PPV: {valid_projected_metrics['Projected PPV']:.4f}, "
                f"FPR: {valid_metrics['FPR']:.4f}, Threshold: {valid_threshold:.4f})"
            )

    projected_prevalence = 0.01
    best_valid_projected_ppv = float("-inf")
    best_valid_fpr = float("inf")
    smote_payload = {f"smote/{key}": value for key, value in smote_diagnostics.items()}
    roi_refresh_state = {"completed": False}

    warmstart_criterion = build_supervised_criterion_from_labels(
        args.loss_name,
        resampled_labels.tolist(),
        len(class_names),
        device,
    )
    feature_loader = make_feature_loader(resampled_features, resampled_labels, args, shuffle=True)
    warmstart_parameters = list(model.cls_head.parameters())
    if model.classifier_input == "projection":
        warmstart_parameters = list(model.proj_head.parameters()) + warmstart_parameters
    warmstart_optimizer = AdamW(warmstart_parameters, lr=args.lr)
    warmstart_epochs = (
        args.epochs if args.finetune_train_mode == "probe" else int(args.smote_warmstart_epochs)
    )
    if warmstart_epochs > 0:
        warmstart_scheduler = build_finetune_scheduler(warmstart_optimizer, warmstart_epochs)
        for warm_epoch in range(warmstart_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_features, batch_labels in feature_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                warmstart_optimizer.zero_grad()
                logits = model.classify(batch_features)
                loss = warmstart_criterion(logits, batch_labels)
                loss.backward()
                warmstart_optimizer.step()

                batch_size = batch_labels.size(0)
                train_loss += loss.item() * batch_size
                train_correct += (torch.argmax(logits, dim=1) == batch_labels).sum().item()
                train_total += batch_size

            warmstart_scheduler.step()

            avg_train_loss = train_loss / max(1, train_total)
            train_accuracy = train_correct / max(1, train_total)
            print(
                f"Warm-start Epoch {warm_epoch + 1:02d}/{max(1, warmstart_epochs)} | "
                f"Loss: {avg_train_loss:.4f} | Acc: {train_accuracy:.4f}"
            )

            if args.finetune_train_mode == "probe":
                run_validation_and_logging(
                    warm_epoch,
                    warmstart_optimizer,
                    avg_train_loss,
                    train_accuracy,
                    warmstart_criterion,
                )

    if args.finetune_train_mode == "probe":
        torch.save(
            create_model_checkpoint(
                model,
                checkpoint_model_config,
                extra_metadata={
                    "experiment_id": args.experiment_id,
                    "epoch": args.epochs,
                    "stage": args.stage,
                    "loss_name": args.loss_name,
                    "finetune_with_smote": True,
                    "smote_diagnostics": smote_diagnostics,
                },
            ),
            final_save_path,
        )
        print(f"Saved final model: {final_save_path}")
        return

    print(
        "Starting image-level finetuning after SMOTE warm-start | "
        f"train_mode={args.finetune_train_mode} | classifier_input={args.classifier_input}"
    )
    image_criterion = build_supervised_criterion(args.loss_name, train_ds, len(class_names), device)
    image_optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    image_scheduler = build_finetune_scheduler(image_optimizer, args.epochs)

    for epoch in range(args.epochs):
        if args.roi_guided_training and not roi_refresh_state["completed"] and epoch >= args.roi_start_epoch:
            refresh_train_roi_guidance(args, model, train_ds, device, epoch)
            roi_refresh_state["completed"] = True

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images1, images2, labels in train_loader:
            images1 = images1.to(device)
            images2 = images2.to(device)
            labels = labels.to(device).long()

            image_optimizer.zero_grad()
            logits1 = model(images1)
            logits2 = model(images2)
            loss = 0.5 * (image_criterion(logits1, labels) + image_criterion(logits2, labels))
            loss.backward()
            image_optimizer.step()

            batch_size = labels.size(0)
            train_loss += loss.item() * batch_size
            train_correct += (torch.argmax(logits1, dim=1) == labels).sum().item()
            train_total += batch_size

        image_scheduler.step()

        avg_train_loss = train_loss / max(1, train_total)
        train_accuracy = train_correct / max(1, train_total)
        run_validation_and_logging(epoch, image_optimizer, avg_train_loss, train_accuracy, image_criterion)

    torch.save(
        create_model_checkpoint(
            model,
            checkpoint_model_config,
            extra_metadata={
                "experiment_id": args.experiment_id,
                "epoch": args.epochs,
                "stage": args.stage,
                "loss_name": args.loss_name,
                "finetune_with_smote": True,
                "smote_diagnostics": smote_diagnostics,
            },
        ),
        final_save_path,
    )
    print(f"Saved final model: {final_save_path}")


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
    print(
        f"Resolved backbone: {args.backbone_name} | input size: {args.input_size} | "
        f"pretrained: {args.pretrained} | backbone weights: {args.backbone_weights_path}"
    )
    print(
        f"Head: {args.head_type} | head hidden dim: {args.head_hidden_dim} | "
        f"MLP hidden layers: {args.mlp_hidden_layers} | MLP hidden dim: {args.mlp_hidden_dim}"
    )
    print(
        f"Classifier input: {args.classifier_input} | finetune train mode: {args.finetune_train_mode}"
    )
    if args.roi_records_path:
        print(f"Saved ROI guidance configured from {args.roi_records_path}")
    if args.init_encoder_ckpt:
        print(f"Pretrain encoder initialization configured from {args.init_encoder_ckpt}")
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
        classifier_input=args.classifier_input,
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
    if args.stage == "pretrain" and args.init_encoder_ckpt:
        load_encoder_checkpoint(model, args.init_encoder_ckpt)
        print(f"Loaded initialization encoder checkpoint from {args.init_encoder_ckpt}")

    checkpoint_model_config = {
        "in_channels": 3,
        "n_classes": len(class_names),
        "backbone_name": args.backbone_name,
        "input_size": args.input_size,
        "pretrained": False,
        "proj_dim": 128,
        "classifier_input": args.classifier_input,
        "head_type": args.head_type,
        "head_hidden_dim": args.head_hidden_dim,
        "head_dropout": args.head_dropout,
        "mlp_hidden_layers": args.mlp_hidden_layers,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "mlp_dropout": args.mlp_dropout,
    }

    if args.stage == "finetune" and args.finetune_with_smote:
        run_smote_finetune(
            args=args,
            model=model,
            train_loader=train_loader,
            train_ds=train_ds,
            valid_loader=valid_loader,
            testset_loader=testset_loader,
            class_names=class_names,
            device=device,
            checkpoint_model_config=checkpoint_model_config,
            best_save_path=os.path.join(args.save_dir, f"{args.experiment_id}_best.pt"),
            final_save_path=os.path.join(args.save_dir, f"{args.experiment_id}_final.pt"),
        )
        if args.post_train_gradcam:
            run_post_training_gradcam(
                args=args,
                model=model,
                device=device,
                final_save_path=os.path.join(args.save_dir, f"{args.experiment_id}_final.pt"),
                best_save_path=os.path.join(args.save_dir, f"{args.experiment_id}_best.pt"),
                segmentation_loader=segmentation_loader,
            )
        print(f"Class mapping: {class_names}")
        print("Training finished! Check your WandB dashboard.")
        wandb.finish()
        return

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

        extra_payload = {
            "stage": args.stage,
            "loss_name": args.loss_name,
            "train/accuracy": train_accuracy,
            "valid/accuracy": valid_accuracy,
            "train/loss_ce": avg_train_ce,
            "train/loss_supmin": avg_train_supmin,
            "train/loss_suppro": avg_train_suppro,
        }

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
        # Best-checkpoint selection lives here: maximize projected 1% validation PPV and break ties with lower validation FPR.
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
                segmentation_loader=segmentation_loader,
            )

    print(f"Class mapping: {class_names}")
    print("Training finished! Check your WandB dashboard.")
    wandb.finish()


if __name__ == "__main__":
    main(get_args_parser().parse_args())
