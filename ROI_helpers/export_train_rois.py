from argparse import ArgumentParser
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# Add parent directory to path so we can import main modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import build_train_val_dataframes
from model import Model, load_model_checkpoint, resolve_model_kwargs_from_checkpoint
from ROI_helpers.roi_guidance import save_roi_records_to_json
from train import build_gradcam_roi_records


def filter_model_kwargs_for_init(model_kwargs):
    valid_keys = {
        key for key in inspect.signature(Model.__init__).parameters
        if key not in {"self", "kwargs"}
    }
    return {key: value for key, value in dict(model_kwargs).items() if key in valid_keys}


def get_args_parser():
    parser = ArgumentParser("Export train-split Grad-CAM ROIs for ROI-guided pretraining")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained baseline/finetune checkpoint used to generate Grad-CAM ROIs.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Training data directory containing the center subfolders.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Destination JSON path for the saved ROI metadata.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Grad-CAM export batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument(
        "--gradcam-target-class",
        type=int,
        default=1,
        help="Class index used when generating Grad-CAM ROIs.",
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
        "--backbone-name",
        type=str,
        default="vit_base_patch16_dinov3.lvd1689m",
        help="Fallback backbone name used only if the checkpoint lacks model_config metadata.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Fallback square input size used only if the checkpoint lacks model_config metadata.",
    )
    return parser


def main(args):
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    resolved_model_kwargs = resolve_model_kwargs_from_checkpoint(
        checkpoint,
        fallback_kwargs={
            "in_channels": 3,
            "n_classes": 2,
            "backbone_name": args.backbone_name,
            "input_size": args.input_size,
            "pretrained": False,
        },
    )

    train_df, _, class_names = build_train_val_dataframes(args.data_dir)
    resolved_model_kwargs = filter_model_kwargs_for_init(resolved_model_kwargs)
    resolved_model_kwargs["n_classes"] = len(class_names)
    effective_input_size = int(resolved_model_kwargs.get("input_size", args.input_size))

    model = Model(**resolved_model_kwargs).to(device)
    load_model_checkpoint(model, checkpoint_path, map_location=device)
    print(
        f"Loaded ROI source checkpoint from {checkpoint_path} | "
        f"backbone={resolved_model_kwargs['backbone_name']} | input_size={effective_input_size}"
    )

    export_args = SimpleNamespace(
        gradcam_target_class=args.gradcam_target_class,
        gradcam_batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=effective_input_size,
        roi_gradcam_threshold=args.roi_gradcam_threshold,
        roi_gradcam_min_prob=args.roi_gradcam_min_prob,
    )
    train_ds_proxy = SimpleNamespace(df=train_df)
    roi_records, positive_candidates = build_gradcam_roi_records(
        export_args,
        model,
        train_ds_proxy,
        device,
    )

    metadata = {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(Path(args.data_dir).resolve()),
        "split": "train",
        "train_image_count": int(len(train_df)),
        "positive_candidates": int(positive_candidates),
        "roi_records_total": int(len(roi_records)),
        "gradcam_target_class": int(args.gradcam_target_class),
        "roi_gradcam_threshold": float(args.roi_gradcam_threshold),
        "roi_gradcam_min_prob": float(args.roi_gradcam_min_prob),
        "input_size": int(effective_input_size),
    }
    if isinstance(checkpoint, dict):
        if "experiment_id" in checkpoint:
            metadata["source_experiment_id"] = checkpoint["experiment_id"]
        if "epoch" in checkpoint:
            metadata["source_epoch"] = checkpoint["epoch"]

    save_roi_records_to_json(output_path, roi_records, metadata=metadata)
    print(
        f"Saved {len(roi_records)} train ROI records to {output_path} "
        f"from {positive_candidates} positive candidates."
    )


if __name__ == "__main__":
    main(get_args_parser().parse_args())
