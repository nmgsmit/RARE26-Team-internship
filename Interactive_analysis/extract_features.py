"""
Stage 2: extract features from every full-model checkpoint listed in runs.csv,
against ONE OR MORE datasets in a single invocation.
For each (row in runs.csv) × (dataset configured in DATASETS):
  - if model finetune is empty → skip with status "pending" (e.g. P2 rows)
  - if the file is missing on disk → skip with status "missing"
  - if features_out/<experiment_id>__<dataset_tag>/meta.json exists → skip (idempotent)
  - otherwise: load model, run the loader, save outputs
Output (one folder per (run, dataset)):
  features_out/<experiment_id>__<dataset_tag>/
      features_pooled.npy   (N, feat_dim)
      features_proj.npy     (N, proj_dim)
      labels.npy            (N,)
      paths.npy             (N,)
      deployed_logits.npy   (N, n_classes)
      deployed_probs.npy    (N, n_classes)
      head_linear.npz       (optional: only if head is linear)
      head_linear_kind.json
      meta.json
DATASETS configuration:
    Each entry produces a separate set of folders. The `tag` becomes the
    suffix in the folder name, so you can have e.g.:
        P1_BB_GastronetDinoV2_t1__evc_test/   (the held-out EVC set)
        P1_BB_GastronetDinoV2_t1__train_all/  (the training-style dataset)
    side by side, and compare_runs.py will let you filter by tag.
    Each entry has:
        kind:       "flat" or "centers"
        tag:        short name used in the folder suffix (no spaces)
        - if kind == "flat":
            data_dir: flat folder of images with labels in filenames
        - if kind == "centers":
            data_dir: training-style folder with centerN/{ndbe,neo}/ subdirs
            split:    "val", "train", or "all"
Usage: edit the CONFIGURATION block at the bottom and run
    python extract_features.py
"""
import json
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from data import eval_only_loader, flat_eval_loader
from model import (
    LayerNormLinearHead,
    Model,
    load_model_checkpoint,
    resolve_model_kwargs_from_checkpoint,
)


def _maybe_capture_linear_head(head: nn.Module):
    """If the deployed head is a single linear (or LN+Linear), capture its
    weights so feature_analysis.py can draw the real deployed boundary."""
    if isinstance(head, nn.Linear):
        return {
            "kind": "linear",
            "weight": head.weight.detach().cpu().numpy(),
            "bias": head.bias.detach().cpu().numpy() if head.bias is not None else None,
        }
    if isinstance(head, LayerNormLinearHead):
        return {
            "kind": "ln_linear",
            "weight": head.classifier.weight.detach().cpu().numpy(),
            "bias": head.classifier.bias.detach().cpu().numpy()
                    if head.classifier.bias is not None else None,
            "ln_weight": head.norm.weight.detach().cpu().numpy(),
            "ln_bias": head.norm.bias.detach().cpu().numpy(),
            "ln_eps": float(head.norm.eps),
        }
    return None


def _build_loader_for_dataset(dataset_spec, *, input_size, seed,
                              batch_size, num_workers):
    """Build a loader for a single dataset spec entry."""
    if dataset_spec["kind"] == "flat":
        loader, _, class_names = flat_eval_loader(
            data_dir=str(dataset_spec["data_dir"]),
            batch_size=batch_size,
            num_workers=num_workers,
            input_size=input_size,
        )
    elif dataset_spec["kind"] == "centers":
        loader, _, class_names = eval_only_loader(
            data_dir=str(dataset_spec["data_dir"]),
            seed=seed,
            batch_size=batch_size,
            num_workers=num_workers,
            input_size=input_size,
            split=dataset_spec.get("split", "all"),
        )
    else:
        raise ValueError(f"Unknown dataset kind: {dataset_spec['kind']!r}")
    return loader, class_names


def _load_full_model(checkpoint_path: Path, device):
    """Load a full-model checkpoint. Fails clearly if given an
    encoder-only file by mistake."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (isinstance(checkpoint, dict)
        and isinstance(checkpoint.get("backbone"), dict)
        and isinstance(checkpoint.get("proj_head"), dict)
        and "model_state_dict" not in checkpoint
        and "state_dict" not in checkpoint
        and "model" not in checkpoint
        and "teacher" not in checkpoint):
        raise ValueError(
            f"{checkpoint_path.name} looks like an encoder-only checkpoint "
            f"(backbone + proj_head only, no classifier head). "
            f"Use a *_finetune_best.pt file instead."
        )
    resolved = resolve_model_kwargs_from_checkpoint(checkpoint)
    resolved["pretrained"] = False
    resolved["backbone_weights_path"] = None

    # ========================================================
    # PATCH FOR MODEL 1 COMPATIBILITY
    # Model 1 lacks the internal kwarg-stripping that Model 2 has.
    # We manually strip training-only metadata here so timm doesn't crash.
    # ========================================================
    for stale_key in ("backbone_preset", "num_folds", "fold_index", "classifier_input_dim"):
        resolved.pop(stale_key, None)

    model = Model(**resolved)
    load_model_checkpoint(model, checkpoint_path, strict=True)
    model = model.to(device).eval()
    input_size = int(resolved.get("input_size", 224))
    return model, input_size

# def _load_full_model(checkpoint_path: Path, device):
#     """Load a full-model checkpoint. Fails clearly if given an
#     encoder-only file by mistake."""
#     checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
#     if (isinstance(checkpoint, dict)
#         and isinstance(checkpoint.get("backbone"), dict)
#         and isinstance(checkpoint.get("proj_head"), dict)
#         and "model_state_dict" not in checkpoint
#         and "state_dict" not in checkpoint
#         and "model" not in checkpoint
#         and "teacher" not in checkpoint):
#         raise ValueError(
#             f"{checkpoint_path.name} looks like an encoder-only checkpoint "
#             f"(backbone + proj_head only, no classifier head). "
#             f"Use a *_finetune_best.pt file instead."
#         )
#     resolved = resolve_model_kwargs_from_checkpoint(checkpoint)
#     resolved["pretrained"] = False
#     resolved["backbone_weights_path"] = None
#     model = Model(**resolved)
#     load_model_checkpoint(model, checkpoint_path, strict=True)
#     model = model.to(device).eval()
#     input_size = int(resolved.get("input_size", 224))
#     return model, input_size


@torch.no_grad()
def _extract_features_for_model(model, loader, device):
    feats_pooled, feats_proj, labels_all, paths_all, logits_all = [], [], [], [], []
    for batch_idx, (images, labels, paths) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        tokens = model.forward_tokens(images)
        pooled = model.pooled_features_from_tokens(tokens)
        projected = model.project(pooled)
        classifier_features = model.classifier_features_from_pooled(pooled)
        logits = model.classify(classifier_features)
        feats_pooled.append(pooled.cpu().numpy())
        feats_proj.append(projected.cpu().numpy())
        logits_all.append(logits.cpu().numpy())
        labels_all.append(np.asarray(labels))
        paths_all.extend(paths)
        if batch_idx % 20 == 0:
            print(f"      batch {batch_idx + 1}/{len(loader)}")
    logits = np.concatenate(logits_all, axis=0)
    return {
        "features_pooled": np.concatenate(feats_pooled, axis=0),
        "features_proj": np.concatenate(feats_proj, axis=0),
        "labels": np.concatenate(labels_all, axis=0).astype(np.int64),
        "paths": np.asarray(paths_all, dtype=object),
        "deployed_logits": logits,
        "deployed_probs": torch.softmax(torch.from_numpy(logits), dim=-1).numpy(),
    }


def _save_outputs(output_dir: Path, arrays: dict, meta: dict, head_capture):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "features_pooled.npy", arrays["features_pooled"])
    np.save(output_dir / "features_proj.npy", arrays["features_proj"])
    np.save(output_dir / "labels.npy", arrays["labels"])
    np.save(output_dir / "paths.npy", arrays["paths"], allow_pickle=True)
    np.save(output_dir / "deployed_logits.npy", arrays["deployed_logits"])
    np.save(output_dir / "deployed_probs.npy", arrays["deployed_probs"])
    if head_capture is not None:
        np.savez(output_dir / "head_linear.npz", **{
            k: v for k, v in head_capture.items() if v is not None and k != "kind"
        })
        (output_dir / "head_linear_kind.json").write_text(
            json.dumps({"kind": head_capture["kind"]})
        )
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def _resolve_checkpoint_path(checkpoints_root: Path, ckpt_file_str: str):
    """Look up the .pt file: try the literal path first, then rglob the
    checkpoints root for a matching filename."""
    direct_path = checkpoints_root / ckpt_file_str
    if direct_path.exists():
        return direct_path
    matches = list(checkpoints_root.rglob(ckpt_file_str))
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        print(f"  [warn] {ckpt_file_str} appears in multiple subfolders, "
              f"using {matches[0]}")
    return matches[0]


def process_run_dataset(row, dataset_spec, *, checkpoints_root, output_root,
                        loader_cache, seed, batch_size, num_workers,
                        device, force):
    """Process one (run, dataset) pair. loader_cache is keyed by
    (dataset_tag, input_size) so we don't rebuild loaders unnecessarily."""
    exp_id = row["experiment_id"]
    dataset_tag = dataset_spec["tag"]
    label = f"{exp_id} × {dataset_tag}"

    ckpt_file = row.get("model finetune")
    if not ckpt_file or pd.isna(ckpt_file) or str(ckpt_file).strip() == "":
        return "pending", {
            "experiment_id": exp_id, "dataset_tag": dataset_tag,
            "reason": "no model finetune in runs.csv",
        }

    ckpt_path = _resolve_checkpoint_path(
        Path(checkpoints_root), str(ckpt_file).strip(),
    )
    if ckpt_path is None:
        return "missing", {
            "experiment_id": exp_id, "dataset_tag": dataset_tag,
            "looked_for": str(ckpt_file).strip(),
        }

    output_dir = Path(output_root) / f"{exp_id}__{dataset_tag}"
    if not force and (output_dir / "meta.json").exists():
        return "skipped_done", {
            "experiment_id": exp_id, "dataset_tag": dataset_tag,
            "output_dir": str(output_dir),
        }

    print(f"\n[{label}]")
    print(f"    loading {ckpt_path}")
    model, input_size = _load_full_model(ckpt_path, device)

    cache_key = (dataset_tag, input_size)
    if cache_key not in loader_cache:
        print(f"    [loader] building '{dataset_tag}' loader at "
              f"input_size={input_size}")
        loader_cache[cache_key] = _build_loader_for_dataset(
            dataset_spec, input_size=input_size, seed=seed,
            batch_size=batch_size, num_workers=num_workers,
        )
    loader, class_names = loader_cache[cache_key]

    print(f"    feat_dim={model.feat_dim}, proj_dim={model.proj_dim}, "
          f"input_size={input_size}")
    print(f"    head: {model.classifier_description}")
    arrays = _extract_features_for_model(model, loader, device)
    head_capture = _maybe_capture_linear_head(model.head)
    meta = {
        "experiment_id": exp_id,
        "checkpoint_path": str(ckpt_path),
        "model_kind": "full_model",
        "feat_dim": int(model.feat_dim),
        "proj_dim": int(model.proj_dim),
        "input_size": int(input_size),
        "n_samples": int(arrays["features_pooled"].shape[0]),
        "class_names": list(class_names),
        "dataset_tag": dataset_tag,
        "dataset_kind": dataset_spec["kind"],
        "dataset_data_dir": str(dataset_spec["data_dir"]),
        "head_linear_captured": head_capture is not None,
        "classifier_input": model.classifier_input,
        "head_type": model.head_type,
        "classifier_description": model.classifier_description,
        "n_classes": int(arrays["deployed_logits"].shape[1]),
    }
    if dataset_spec["kind"] == "centers":
        meta["dataset_split"] = dataset_spec.get("split", "all")

    _save_outputs(output_dir, arrays, meta, head_capture)
    print(f"    saved -> {output_dir}")
    return "ok", {
        "experiment_id": exp_id, "dataset_tag": dataset_tag,
        "output_dir": str(output_dir),
    }


def main(*, runs_csv, checkpoints_root, datasets, output_dir, seed,
         batch_size, num_workers, device, force):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    runs = pd.read_csv(runs_csv)
    print(f"Loaded {len(runs)} rows from {runs_csv}")
    print(f"Will process {len(datasets)} dataset(s): "
          f"{[d['tag'] for d in datasets]}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    loader_cache: dict[tuple, tuple] = {}
    statuses = []

    for _, row in runs.iterrows():
        exp_id_val = row.get("experiment_id")
        if exp_id_val is None or pd.isna(exp_id_val) or str(exp_id_val).strip() == "":
            continue
        for dataset_spec in datasets:
            try:
                status, info = process_run_dataset(
                    row.to_dict(), dataset_spec,
                    checkpoints_root=checkpoints_root,
                    output_root=output_dir,
                    loader_cache=loader_cache,
                    seed=seed, batch_size=batch_size,
                    num_workers=num_workers,
                    device=device, force=force,
                )
                statuses.append({"status": status, **info})
            except Exception as e:
                print(f"  [ERROR] {row['experiment_id']} × "
                      f"{dataset_spec['tag']}: {e}")
                traceback.print_exc()
                statuses.append({
                    "status": "error",
                    "experiment_id": row["experiment_id"],
                    "dataset_tag": dataset_spec["tag"],
                    "error": str(e),
                })

    (Path(output_dir) / "extraction_summary.json").write_text(
        json.dumps(statuses, indent=2)
    )

    by_status: dict[str, list[str]] = {}
    for s in statuses:
        key = s["status"]
        label = f"{s.get('experiment_id', '?')} × {s.get('dataset_tag', '?')}"
        by_status.setdefault(key, []).append(label)

    print(f"\n{'='*60}")
    print("Extraction summary:")
    for status, labels in sorted(by_status.items()):
        print(f"  {status}: {len(labels)}")
        for lbl in labels:
            print(f"    - {lbl}")


if __name__ == "__main__":
    # ============================================================
    # ---> CONFIGURATION <---
    # ============================================================
    RUNS_CSV = "runs.csv"
    CHECKPOINTS_ROOT = "Checkpoints"
    OUTPUT_DIR = "features_out"

    # Datasets to extract features for. Each entry produces a separate set
    # of feature folders (e.g. <experiment_id>__evc_test/ and
    # <experiment_id>__train_all/). compare_runs.py shows them side by side.
    #
    # Comment out an entry to skip it.
    DATASETS = [
        {
            "kind": "flat",
            "tag":  "evc_test",
            "data_dir": "../../EVC_Barretts_FullSet 2/images",
        },
        {
            "kind": "centers",
            "tag":  "train_all",
            "data_dir": "../../data",
            "split": "all",       # "val", "train", or "all"
        },
    ]

    SEED = 42
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    DEVICE = "mps"    # "cuda", "mps", "cpu", or None to auto-detect
    FORCE = False     # True to re-extract folders that already have meta.json

    main(
        runs_csv=RUNS_CSV,
        checkpoints_root=CHECKPOINTS_ROOT,
        datasets=DATASETS,
        output_dir=OUTPUT_DIR,
        seed=SEED,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        force=FORCE,
    )