from argparse import ArgumentParser
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import (
    SimpleDataset,
    build_dataset_dataframe,
    build_eval_transform,
    build_train_val_dataframes,
)
from gradcam import compute_vit_gradcam_batch
from model import Model, load_model_checkpoint, resolve_model_kwargs_from_checkpoint


def filter_model_kwargs_for_init(model_kwargs):
    valid_keys = {
        key for key in inspect.signature(Model.__init__).parameters
        if key not in {"self", "kwargs"}
    }
    return {key: value for key, value in dict(model_kwargs).items() if key in valid_keys}


def load_all_train_val_dataframe(data_dir, split_random_state=42):
    full_df, class_names = build_dataset_dataframe(data_dir)
    train_df, val_df, split_class_names = build_train_val_dataframes(
        data_dir,
        random_state=split_random_state,
    )

    if class_names != split_class_names:
        raise ValueError("Class names from the full dataframe and split dataframe do not match.")

    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"

    split_df = pd.concat([train_df, val_df], ignore_index=True)
    split_df["img"] = split_df["img"].astype(str)

    full_df = full_df.copy()
    full_df["img"] = full_df["img"].astype(str)

    if len(full_df) != len(split_df):
        raise ValueError("Train + val does not cover the same number of images as the full dataframe.")
    if set(full_df["img"]) != set(split_df["img"]):
        raise ValueError("Train + val does not cover exactly the same image paths as the full dataframe.")

    split_map = split_df.set_index("img")["split"].to_dict()
    full_df["split"] = full_df["img"].map(split_map).fillna("unknown")
    return full_df.reset_index(drop=True), class_names


def load_roi_source_model(checkpoint_path, class_names, device, fallback_backbone_name, fallback_input_size):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    resolved_model_kwargs = resolve_model_kwargs_from_checkpoint(
        checkpoint,
        fallback_kwargs={
            "in_channels": 3,
            "n_classes": len(class_names),
            "backbone_name": fallback_backbone_name,
            "input_size": fallback_input_size,
            "pretrained": False,
        },
    )
    resolved_model_kwargs = filter_model_kwargs_for_init(resolved_model_kwargs)
    resolved_model_kwargs["n_classes"] = len(class_names)

    model = Model(**resolved_model_kwargs).to(device)
    load_model_checkpoint(model, checkpoint_path, map_location=device)
    model.eval()
    return model, resolved_model_kwargs, checkpoint


def compute_gradcam_catalog(df, model, input_size, target_class, batch_size, num_workers, device):
    eval_df = df.copy().reset_index(drop=True)
    eval_ds = SimpleDataset(eval_df[["img", "label"]].copy(), build_eval_transform(input_size))
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    records = []
    raw_cam_store = {}
    path_offset = 0
    was_training = model.training
    model.eval()

    try:
        for images, labels in tqdm(eval_loader, desc="Computing Grad-CAM cache"):
            batch_rows = eval_df.iloc[path_offset:path_offset + len(labels)]
            path_offset += len(labels)

            images = images.to(device)
            cams, probs, raw_cams = compute_vit_gradcam_batch(
                model=model,
                images=images,
                target_class=target_class,
                return_raw=True,
            )
            cams = cams.detach().cpu()
            raw_cams = raw_cams.detach().cpu()
            probs = probs.detach().cpu().numpy()

            for row, cam_tensor, raw_cam_tensor, prob in zip(
                batch_rows.itertuples(index=False), cams, raw_cams, probs
            ):
                image_path = str(row.img)
                raw_cam_store[image_path] = raw_cam_tensor.numpy().astype(np.float32)
                records.append(
                    {
                        "img": image_path,
                        "label": int(row.label),
                        "center": str(row.center),
                        "split": str(row.split),
                        "positive_prob": float(prob),
                        "cam_peak": float(cam_tensor.max().item()),
                        "cam_mean": float(cam_tensor.mean().item()),
                    }
                )
    finally:
        if was_training:
            model.train()

    results_df = pd.DataFrame(records).reset_index(drop=True)
    return results_df, raw_cam_store


def build_target_class_tag(target_classes):
    normalized = sorted({int(target_class) for target_class in target_classes})
    if normalized == [0, 1]:
        return "both-classes"
    if len(normalized) == 1:
        return f"class{normalized[0]}"
    joined = "-".join(str(target_class) for target_class in normalized)
    return f"classes-{joined}"


def resolve_default_cache_path(checkpoint_path, target_classes):
    checkpoint_path = Path(checkpoint_path)
    output_dir = checkpoint_path.parent / "roi_records"
    class_tag = build_target_class_tag(target_classes)
    return output_dir / f"{checkpoint_path.stem}.{class_tag}.gradcam_cache.npz"


def save_raw_gradcam_cache(cache_path, image_paths, raw_cam_store_by_class, float_dtype=np.float16):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    ordered_paths = [str(path) for path in image_paths]
    payload = {
        "image_paths": np.asarray(ordered_paths, dtype=object),
        "target_classes": np.asarray(sorted(raw_cam_store_by_class.keys()), dtype=np.int64),
    }
    for target_class, raw_cam_store in sorted(raw_cam_store_by_class.items()):
        raw_cam_tensor = np.stack(
            [np.asarray(raw_cam_store[path], dtype=np.float32) for path in ordered_paths],
            axis=0,
        ).astype(float_dtype)
        payload[f"raw_cams_class{int(target_class)}"] = raw_cam_tensor

    if len(raw_cam_store_by_class) == 1:
        sole_target_class = next(iter(sorted(raw_cam_store_by_class.keys())))
        payload["raw_cams"] = payload[f"raw_cams_class{int(sole_target_class)}"]

    np.savez_compressed(cache_path, **payload)
    return cache_path.resolve()


def save_cache_manifest(
    json_path,
    cache_path,
    results_by_class,
    checkpoint_path,
    data_dir,
    target_classes,
    input_size,
):
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_target_classes = sorted({int(target_class) for target_class in target_classes})
    first_results_df = results_by_class[normalized_target_classes[0]]
    if len(normalized_target_classes) == 1:
        gradcam_results = first_results_df.to_dict(orient="records")
        gradcam_results_by_class = None
    else:
        gradcam_results = []
        gradcam_results_by_class = {
            str(target_class): results_by_class[target_class].to_dict(orient="records")
            for target_class in normalized_target_classes
        }

    payload = {
        "metadata": {
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "data_dir": str(Path(data_dir).resolve()),
            "split": "train+val",
            "image_count_total": int(len(first_results_df)),
            "target_class": int(normalized_target_classes[0]) if len(normalized_target_classes) == 1 else None,
            "target_classes": normalized_target_classes,
            "input_size": int(input_size),
            "gradcam_cache_path": str(Path(cache_path).resolve()),
            "raw_gradcam_saved": True,
            "cache_manifest_only": True,
        },
        "roi_records": {},
        "gradcam_results": gradcam_results,
    }
    if gradcam_results_by_class is not None:
        payload["gradcam_results_by_class"] = gradcam_results_by_class

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return json_path.resolve()


def parse_args():
    parser = ArgumentParser("Build a train+val Grad-CAM raw cache for ROI calibration on a cluster.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint used to compute Grad-CAMs.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data/Challenge_train_data",
        help="Training data directory containing the center subfolders.",
    )
    parser.add_argument(
        "--output-cache-path",
        type=str,
        default=None,
        help=(
            "Optional destination .npz path. If omitted, the script writes "
            "./checkpoints/roi_records/<checkpoint-stem>.<class-tag>.gradcam_cache.npz "
            "next to the checkpoint."
        ),
    )
    parser.add_argument(
        "--output-json-path",
        type=str,
        default=None,
        help=(
            "Optional JSON manifest path compatible with roi_gradcam_calibration.ipynb cache loading. "
            "If provided, the notebook can reuse the cache immediately."
        ),
    )
    parser.add_argument(
        "--target-class",
        type=int,
        nargs="+",
        default=[1],
        help="One or more class indices used for Grad-CAM, for example --target-class 0 1.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Grad-CAM cache batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument(
        "--fallback-backbone-name",
        type=str,
        default="vit_base_patch14_reg4_dinov2",
        help="Fallback backbone name if the checkpoint lacks model_config metadata.",
    )
    parser.add_argument(
        "--fallback-input-size",
        type=int,
        default=336,
        help="Fallback square input size if the checkpoint lacks model_config metadata.",
    )
    parser.add_argument(
        "--cache-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Storage dtype for raw Grad-CAM maps in the .npz cache.",
    )
    parser.add_argument(
        "--split-random-state",
        type=int,
        default=42,
        help="Random state used to reconstruct the train/val split labels in the cached metadata table.",
    )
    return parser.parse_args()


def main(args):
    checkpoint_path = Path(args.checkpoint).resolve()
    normalized_target_classes = sorted({int(target_class) for target_class in args.target_class})
    cache_path = (
        Path(args.output_cache_path).resolve()
        if args.output_cache_path
        else resolve_default_cache_path(checkpoint_path, normalized_target_classes).resolve()
    )
    json_path = Path(args.output_json_path).resolve() if args.output_json_path else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_df, class_names = load_all_train_val_dataframe(
        args.data_dir,
        split_random_state=args.split_random_state,
    )
    model, resolved_model_kwargs, _checkpoint = load_roi_source_model(
        checkpoint_path=checkpoint_path,
        class_names=class_names,
        device=device,
        fallback_backbone_name=args.fallback_backbone_name,
        fallback_input_size=args.fallback_input_size,
    )
    effective_input_size = int(resolved_model_kwargs.get("input_size", args.fallback_input_size))
    print(
        f"Loaded checkpoint from {checkpoint_path} | "
        f"backbone={resolved_model_kwargs.get('backbone_name')} | "
        f"input_size={effective_input_size} | device={device}"
    )
    print(f"Computing raw Grad-CAM cache for {len(all_df)} train+val images from {Path(args.data_dir).resolve()}")

    results_by_class = {}
    raw_cam_store_by_class = {}
    for target_class in normalized_target_classes:
        print(f"Computing Grad-CAMs for target class {target_class}")
        results_df, raw_cam_store = compute_gradcam_catalog(
            df=all_df,
            model=model,
            input_size=effective_input_size,
            target_class=target_class,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        results_by_class[target_class] = results_df
        raw_cam_store_by_class[target_class] = raw_cam_store

    float_dtype = np.float16 if args.cache_dtype == "float16" else np.float32
    saved_cache_path = save_raw_gradcam_cache(
        cache_path=cache_path,
        image_paths=all_df["img"].astype(str).tolist(),
        raw_cam_store_by_class=raw_cam_store_by_class,
        float_dtype=float_dtype,
    )
    print(f"Saved raw Grad-CAM cache to {saved_cache_path}")

    if json_path is not None:
        saved_json_path = save_cache_manifest(
            json_path=json_path,
            cache_path=saved_cache_path,
            results_by_class=results_by_class,
            checkpoint_path=checkpoint_path,
            data_dir=args.data_dir,
            target_classes=normalized_target_classes,
            input_size=effective_input_size,
        )
        print(f"Saved notebook-compatible cache manifest to {saved_json_path}")

    for target_class in normalized_target_classes:
        print(f"Summary for target class {target_class}")
        label_summary = (
            results_by_class[target_class]
            .groupby(["split", "label"])
            .size()
            .rename("count")
            .reset_index()
            .sort_values(["split", "label"])
        )
        print(label_summary.to_string(index=False))


if __name__ == "__main__":
    main(parse_args())
