"""Two-stage training entry point.

Stage 1 (pretrain):
    SupPro supervised-contrastive pretraining of the encoder + projection MLP.
    Saves <experiment_id>_encoder.pt containing {backbone, proj_head, model_config}.

Stage 2 (finetune):
    Loads a pretrain encoder, freezes the backbone + projection MLP, extracts
    deterministic features on the LOCO train split, and fits one or more
    sklearn heads (KNN with chosen k values, SVM with chosen C values).
    Saves one .pt per (head, hyperparameter) plus, when running under --loco
    with multiple folds, ensemble bundles that mirror train.py:_run_ensemble.

LOCO + ensemble:
    --loco runs all folds (one per center) and, after the last fold, builds
    one ensemble_<head>_<value>.pt per head choice that averages each fold's
    positive-class probabilities at inference time.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader

from data import (
    SimpleDataset,
    build_dataset_dataframe,
    build_eval_transform,
    prepare_datasets,
    seed_worker,
    split_dataframe,
)
from metrics import (
    collect_scores,
    compute_group_eval_metrics,
    compute_separation_metrics,
    log_val_metrics,
)
from model import (
    Model,
    create_model_checkpoint,
    load_encoder_checkpoint,
)


BACKBONE_PRESETS = {
    "gastronet": "vit_base_patch14_reg4_dinov2",
    "dinov3":    "vit_base_patch16_dinov3.lvd1689m",
    "simclr":    "resnet50",
    "mocov2":    "resnet50",
    "resnet50":  "resnet50",
}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Loss + schedulers
# ---------------------------------------------------------------------------
def suppro_loss(features, labels, temperature, base_temperature):
    """SupPro / SupCon loss over a (B, V, D) stack of projection-head views."""
    features = F.normalize(features, dim=-1)
    device = features.device
    _, views, _ = features.shape

    contrast = torch.cat(torch.unbind(features, dim=1), dim=0)
    logits = torch.matmul(contrast, contrast.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device).repeat(views, views)
    logits_mask = torch.ones_like(mask).fill_diagonal_(0)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
    loss = -(temperature / base_temperature) * mean_log_prob_pos
    return loss.mean()


def build_pretrain_scheduler(optimizer, warmup_epochs, total_epochs):
    warmup = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / max(1, warmup_epochs))
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=0.0,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
def _csv_floats(value):
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _csv_ints(value):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def get_args_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Stage selection
    p.add_argument("--stage", choices=["pretrain", "finetune"], required=True)

    # Backbone + checkpoint
    p.add_argument("--backbone-preset", default="gastronet", choices=list(BACKBONE_PRESETS))
    p.add_argument("--backbone-weights-path", default=None,
                   help="Path to backbone init weights. Required for finetuned-from-weights backbones "
                        "(Gastronet, SimCLR, MoCo-v2). Optional for timm-pretrained backbones.")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip timm's default pretrained=True. Used for resnet50 ImageNet baseline.")
    p.add_argument("--input-size", type=int, default=336)

    # Data + split
    p.add_argument("--data-dir", default="../data/Challenge_train_data")
    p.add_argument("--loco", action="store_true",
                   help="Leave-one-center-out CV. With --num-folds=#centers, runs all folds + builds ensemble.")
    p.add_argument("--num-folds", type=int, default=1,
                   help="Total folds. --loco runs fold_index=0..num_folds-1. "
                        "Without --loco, k=1 means 80/20 stratified split.")
    p.add_argument("--fold-index", type=int, default=0,
                   help="Which fold to start from (or which single fold to run). When --loco is set, "
                        "the orchestrator iterates fold_index..num_folds-1.")

    # Pretrain hyperparameters
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--pretrain-backbone-lr", type=float, default=1e-5)
    p.add_argument("--pretrain-proj-lr", type=float, default=3e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--base-temperature", type=float, default=0.07)

    # Sampler + augmentation
    p.add_argument("--balanced-sampler", action="store_true",
                   help="Use BalancedBatchSampler with --pos-ratio.")
    p.add_argument("--pos-ratio", type=float, default=0.2,
                   help="Positive fraction per batch when --balanced-sampler is set.")
    p.add_argument("--augmentation-intensity", type=int, default=3, choices=[1, 2, 3, 4],
                   help="Augmentation preset: 1=low, 2=medium, 3=strong, 4=extreme.")
    p.add_argument("--crop-min-scale", type=float, default=0.95,
                   help="Lower bound of the random-crop scale range [min, 1.0] used to build the "
                        "two SupPro views. 0.95 reproduces the best run; 1.0 disables cropping.")
    p.add_argument("--inner-val-frac", type=float, default=0.0,
                   help="DEPRECATED / DO NOT USE (>0): a random within-center inner-val split LEAKS at "
                        "the patient level (multiple frames per patient land in both splits), so isel "
                        "measures memorisation. Keep 0. 0 = select best epoch on the cross-center probe "
                        "(optimistic; only valid as a monitoring signal).")

    # Finetune (sklearn-head) hyperparameter sweeps
    p.add_argument("--head-types", default="knn",
                   help="Comma-separated heads to fit during finetune: subset of {knn, svm}.")
    p.add_argument("--knn-neighbors", type=_csv_ints, default=[5],
                   help="Comma-separated list of k values when head_types includes 'knn'.")
    p.add_argument("--svm-C", type=_csv_floats, default=[2.0],
                   help="Comma-separated list of C values when head_types includes 'svm'.")
    p.add_argument("--encoder-ckpt", default=None,
                   help="(Finetune only) Encoder .pt from a prior pretrain run. When --loco is set this "
                        "is a comma-separated list with one entry per fold.")

    # Output + logging
    p.add_argument("--save-dir", default="./checkpoints/clean_baseline")
    p.add_argument("--experiment-id", default="clean_baseline")
    p.add_argument("--wandb-project", default="RARE25-Project")
    p.add_argument("--wandb-group", default="clean-baseline")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])

    # Runtime
    p.add_argument("--num-workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    return p


# ---------------------------------------------------------------------------
# Pretrain stage
# ---------------------------------------------------------------------------
def _build_model_for_pretrain(args, n_classes, device) -> Model:
    return Model(
        in_channels=3,
        n_classes=n_classes,
        backbone_name=BACKBONE_PRESETS[args.backbone_preset],
        backbone_weights_path=args.backbone_weights_path,
        input_size=args.input_size,
        freeze_backbone=False,
        pretrained=(not args.no_pretrained),
        proj_dim=128,
        classifier_input="pooled",
        # Heads only exist on this branch as KNN/SVM. Pretrain doesn't touch the
        # head (the contrastive loss is on the projection head), so we pick any
        # cheap-to-construct head; KNN with the default k is fine.
        head_type="knn",
        knn_neighbors=5,
    ).to(device)


def _pretrain_one_fold(args, fold_index, device) -> str:
    """Returns the path to the saved encoder.pt."""
    fold_args = copy.deepcopy(args)
    fold_args.fold_index = fold_index
    fold_args.balanced_sampler = True  # SupPro pretrain always wants balanced batches

    # Reproducibility: re-seed per fold so a fold is identical whether run alone
    # (--fold-index i) or as part of a --loco sweep. Without this, fold i inherits
    # whatever RNG state the previous folds left behind -> different init/data order.
    seed_everything(int(args.seed) + fold_index)

    (train_loader, valid_loader, train_ds, _val_ds, class_names, train_df, val_df,
     inner_train_loader, inner_val_loader) = prepare_datasets(fold_args, device)

    model = _build_model_for_pretrain(fold_args, len(class_names), device)
    optimizer = AdamW([
        {"params": model.backbone.parameters(), "lr": fold_args.pretrain_backbone_lr},
        {"params": model.proj_head.parameters(), "lr": fold_args.pretrain_proj_lr},
    ])
    scheduler = build_pretrain_scheduler(optimizer, fold_args.warmup_epochs, fold_args.epochs)

    fold_suffix = f"_fold{fold_index}" if fold_args.loco else (
        f"_fold{fold_index}" if fold_args.num_folds > 1 else ""
    )
    experiment_id = f"{fold_args.experiment_id}_pretrain{fold_suffix}"
    save_dir = Path(fold_args.save_dir) / experiment_id
    save_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=fold_args.wandb_project,
        group=fold_args.wandb_group,
        name=experiment_id,
        config=vars(fold_args),
        mode=fold_args.wandb_mode,
        reinit=True,
    )

    print(f"\n=== Pretrain fold {fold_index} | {experiment_id} ===")
    print(f"Backbone: {fold_args.backbone_preset} ({BACKBONE_PRESETS[fold_args.backbone_preset]})")
    print(f"Loss: SupPro | T={fold_args.temperature} base_T={fold_args.base_temperature}")
    print(f"Save dir: {save_dir}")

    best_val_loss = float("inf")
    best_probe_score = float("-inf")   # encoder selected by PPV@90R@1%prev (isel, or probe if no inner split)
    best_probe_epoch = 0
    best_state = None
    encoder_path = save_dir / f"{experiment_id}_encoder.pt"

    # Deterministic (eval-transform) loader over the TRAIN center, for the
    # per-epoch cross-center kNN probe below.
    train_eval_loader = DataLoader(
        SimpleDataset(train_df, build_eval_transform(fold_args.input_size)),
        batch_size=fold_args.batch_size, shuffle=False,
        num_workers=fold_args.num_workers, pin_memory=True, worker_init_fn=seed_worker,
    )

    for epoch in range(fold_args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images1, images2, labels = batch
            images1 = images1.to(device, non_blocking=True)
            images2 = images2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            tokens1 = model.forward_tokens(images1)
            tokens2 = model.forward_tokens(images2)
            feat1 = model.pooled_features_from_tokens(tokens1)
            feat2 = model.pooled_features_from_tokens(tokens2)
            proj1 = model.project(feat1)
            proj2 = model.project(feat2)
            features = torch.stack([proj1, proj2], dim=1)  # (B, 2, D)

            loss = suppro_loss(
                features, labels,
                temperature=fold_args.temperature,
                base_temperature=fold_args.base_temperature,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= max(1, len(train_loader))

        # Cheap val signal: SupPro loss on the val set with the same view recipe.
        # Sklearn-head metrics (AUROC/AUPRC) only become meaningful after the
        # finetune fit, so we don't run them here.
        model.eval()
        val_loss = 0.0
        val_feat, val_proj, val_labels = [], [], []
        with torch.no_grad():
            # Two views on val too. SimpleDataset returns (image, label); we
            # duplicate the same view since augmentation is off at eval.
            for images, labels in valid_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                feat = model.encode(images)
                proj = model.project(feat)
                features = torch.stack([proj, proj], dim=1)
                val_loss += suppro_loss(
                    features, labels,
                    temperature=fold_args.temperature,
                    base_temperature=fold_args.base_temperature,
                ).item()
                val_feat.append(feat.cpu())
                val_proj.append(proj.cpu())
                val_labels.append(labels.cpu())
        val_loss /= max(1, len(valid_loader))

        y = torch.cat(val_labels).numpy()
        X_val = torch.cat(val_feat).numpy()
        # sep/* : pooled feature space the KNN/SVM head actually uses (the
        #   inference-time representation; this is what should track downstream).
        # sep_proj/* : projection space the SupPro loss directly optimises. It
        #   rises ~tautologically; the gap to sep/* shows how much "separation"
        #   is just the discarded projection head, not transferable features.
        sep = compute_separation_metrics(X_val, y)
        sep_proj = {k.replace("sep/", "sep_proj/"): v
                    for k, v in compute_separation_metrics(torch.cat(val_proj).numpy(), y).items()}

        # probe/* : cross-center LOGISTIC probe (continuous scores). Fit on the
        # TRAIN center's StandardScaler'd pooled features, score the held-out
        # center -> an honest per-epoch preview of the LOCO downstream metric.
        # A kNN probe is useless here: its score-0 ties pin PPV@90R at the
        # prevalence floor and hide the encoder actually improving.
        probe = {}
        X_tr, y_tr = _extract_features(model, train_eval_loader, device)
        if len(np.unique(y_tr)) >= 2:
            scaler = StandardScaler().fit(X_tr)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(scaler.transform(X_tr), y_tr)
            pos_col = list(clf.classes_).index(1)
            val_scores = clf.predict_proba(scaler.transform(X_val))[:, pos_col]
            pm = compute_group_eval_metrics(y, val_scores)
            # FPR/TPR are at the 90%-recall threshold -> FPR@90recall is the
            # deployment-bottleneck number (PPV at low prevalence is FPR-limited).
            probe = {f"probe/{k}": pm[k] for k in
                     ("PPV@90RECALL", "PPV@90RECALL@0.01PREV", "AUROC", "AUPRC", "FPR", "TPR")}

        # isel/* : HONEST epoch-selection signal. Logistic fit on inner-train,
        # scored on inner-val (both from the TRAIN center) -> never touches the
        # eval center, so the held-out-center report stays unbiased. probe/* (on
        # the eval center) is logged for monitoring only, NOT for selection.
        isel = {}
        if inner_val_loader is not None:
            Xi_tr, yi_tr = _extract_features(model, inner_train_loader, device)
            Xi_va, yi_va = _extract_features(model, inner_val_loader, device)
            if len(np.unique(yi_tr)) >= 2 and len(np.unique(yi_va)) >= 2:
                isc = StandardScaler().fit(Xi_tr)
                iclf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(isc.transform(Xi_tr), yi_tr)
                ipc = list(iclf.classes_).index(1)
                isv = iclf.predict_proba(isc.transform(Xi_va))[:, ipc]
                im = compute_group_eval_metrics(yi_va, isv)
                isel = {f"isel/{k}": im[k] for k in
                        ("PPV@90RECALL", "PPV@90RECALL@0.01PREV", "AUROC", "AUPRC", "FPR")}

        wandb.log({
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "valid_loss": val_loss,
            **sep,
            **sep_proj,
            **probe,
            **isel,
        })
        print(
            f"  epoch {epoch + 1}/{fold_args.epochs} | train {train_loss:.4f} | "
            f"val {val_loss:.4f} | probe PPV@90R={probe.get('probe/PPV@90RECALL', float('nan')):.4f} "
            f"AUROC={probe.get('probe/AUROC', float('nan')):.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Select the encoder by PPV@90RECALL rescaled to the ~1% deployment
        # prevalence (the challenge operating point), using the HONEST inner-val
        # split (isel) when available; fall back to the cross-center probe (on
        # the held-out center) otherwise -- that's the default (inner_val_frac=0),
        # so by default selection happens on the held-out center.
        sel_score = isel.get("isel/PPV@90RECALL@0.01PREV",
                              probe.get("probe/PPV@90RECALL@0.01PREV", float("-inf")))
        if np.isnan(sel_score):
            sel_score = float("-inf")
        if best_state is None or sel_score > best_probe_score:
            best_probe_score = sel_score
            best_probe_epoch = epoch + 1
            best_state = {
                "backbone": copy.deepcopy(model.backbone.state_dict()),
                "proj_head": copy.deepcopy(model.proj_head.state_dict()),
            }

    # Save the BEST-probe encoder (highest cross-center logistic-probe AUPRC),
    # falling back to the final weights when no probe ran (e.g. epochs=0).
    wandb.summary["best_probe_epoch"] = best_probe_epoch
    wandb.summary["best_probe_ppv90r_1pct"] = best_probe_score if best_state else None
    print(f"Best-probe epoch: {best_probe_epoch} (PPV@90R@1%prev={best_probe_score:.4f})")
    torch.save({
        "backbone": best_state["backbone"] if best_state else model.backbone.state_dict(),
        "proj_head": best_state["proj_head"] if best_state else model.proj_head.state_dict(),
        "backbone_name": BACKBONE_PRESETS[fold_args.backbone_preset],
        "backbone_preset": fold_args.backbone_preset,
        "input_size": fold_args.input_size,
        "num_folds": fold_args.num_folds,
        "fold_index": fold_index,
        "best_probe_epoch": best_probe_epoch,
        "model_config": {
            "in_channels": 3,
            "n_classes": len(class_names),
            "backbone_preset": fold_args.backbone_preset,
            "backbone_name": BACKBONE_PRESETS[fold_args.backbone_preset],
            "input_size": fold_args.input_size,
            "pretrained": False,
            "proj_dim": 128,
            "head_type": "knn",   # placeholder; overridden at finetune time
            "knn_neighbors": 5,
            "svm_C": 2.0,
            "classifier_input": "pooled",
        },
    }, encoder_path)
    print(f"Saved encoder -> {encoder_path}")

    run.finish()
    return str(encoder_path)


# ---------------------------------------------------------------------------
# Finetune stage
# ---------------------------------------------------------------------------
def _build_model_for_finetune(encoder_ckpt_path, head_type, hp_value, n_classes,
                              args, device) -> tuple[Model, dict]:
    encoder_ckpt = torch.load(encoder_ckpt_path, map_location="cpu", weights_only=False)
    raw_cfg = encoder_ckpt.get("model_config", {}) if isinstance(encoder_ckpt, dict) else {}
    cfg = dict(raw_cfg) if isinstance(raw_cfg, dict) else {}

    cfg.setdefault("backbone_name", encoder_ckpt.get("backbone_name") if isinstance(encoder_ckpt, dict) else None)
    cfg.setdefault("input_size", encoder_ckpt.get("input_size", args.input_size))
    cfg["in_channels"] = cfg.get("in_channels", 3)
    cfg["n_classes"] = n_classes
    cfg["pretrained"] = False
    cfg["head_type"] = head_type
    if head_type == "knn":
        cfg["knn_neighbors"] = int(hp_value)
    elif head_type == "svm":
        cfg["svm_C"] = float(hp_value)
    cfg.pop("backbone_weights_path", None)
    for meta in ("backbone_preset", "num_folds", "fold_index", "loss_name"):
        cfg.pop(meta, None)

    model = Model(**cfg).to(device)
    load_encoder_checkpoint(model, encoder_ckpt_path, strict=False)
    # The encoder is fixed during finetune.
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.proj_head.parameters():
        p.requires_grad = False
    return model, cfg


@torch.no_grad()
def _extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    for images, ys in loader:
        images = images.to(device, non_blocking=True)
        pooled = model.encode(images)
        feats.append(model.classifier_features_from_pooled(pooled).cpu().numpy())
        labels.append(ys.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def _finetune_one_fold(args, fold_index, encoder_path, device):
    """Returns {(head_type, hp_value): {payload_dict for ensemble building}}."""
    fold_args = copy.deepcopy(args)
    fold_args.fold_index = fold_index

    # Build train + val splits with the eval transform — features must be
    # deterministic for sklearn fitting.
    df, class_names = build_dataset_dataframe(fold_args.data_dir)
    train_df, val_df, _holdout = split_dataframe(df, fold_args)

    eval_transform = build_eval_transform(fold_args.input_size)
    train_ds = SimpleDataset(train_df, eval_transform)
    val_ds = SimpleDataset(val_df, eval_transform)
    train_loader = DataLoader(train_ds, batch_size=fold_args.batch_size, shuffle=False,
                              num_workers=fold_args.num_workers, pin_memory=True,
                              worker_init_fn=seed_worker)
    val_loader = DataLoader(val_ds, batch_size=fold_args.batch_size, shuffle=False,
                            num_workers=fold_args.num_workers, pin_memory=True,
                            worker_init_fn=seed_worker)

    # Extract features once per fold (shared across all (head_type, hp) variants).
    print(f"\n=== Finetune fold {fold_index} | encoder={encoder_path} ===")
    # We need a model just to extract features. Any head choice works; rebuild
    # per (head_type, hp_value) so the saved checkpoint matches the head.
    sentinel_model, _ = _build_model_for_finetune(
        encoder_path, "knn", 5, len(class_names), fold_args, device,
    )
    print("Extracting training features ...")
    X_train, y_train = _extract_features(sentinel_model, train_loader, device)
    print(f"  train features: {X_train.shape} (pos={int((y_train == 1).sum())}, neg={int((y_train == 0).sum())})")

    head_types = [h.strip() for h in fold_args.head_types.split(",") if h.strip()]
    fold_payloads: dict[tuple[str, float | int], dict] = {}

    fold_suffix = f"_fold{fold_index}" if fold_args.loco else (
        f"_fold{fold_index}" if fold_args.num_folds > 1 else ""
    )

    for head_type in head_types:
        if head_type == "knn":
            hp_values = fold_args.knn_neighbors
        elif head_type == "svm":
            hp_values = fold_args.svm_C
        else:
            raise ValueError(f"Unsupported head_type '{head_type}'. Use 'knn' or 'svm'.")

        for hp_value in hp_values:
            head_tag = f"{head_type}{hp_value if head_type == 'knn' else f'C{hp_value}'}"
            experiment_id = f"{fold_args.experiment_id}_finetune_{head_tag}{fold_suffix}"
            save_dir = Path(fold_args.save_dir) / f"{fold_args.experiment_id}_finetune" / head_tag
            save_dir.mkdir(parents=True, exist_ok=True)

            run = wandb.init(
                project=fold_args.wandb_project,
                group=fold_args.wandb_group,
                name=experiment_id,
                config={**vars(fold_args), "head_type": head_type, "head_hp": hp_value, "fold_index": fold_index},
                mode=fold_args.wandb_mode,
                reinit=True,
            )

            model, cfg = _build_model_for_finetune(
                encoder_path, head_type, hp_value, len(class_names), fold_args, device,
            )
            print(f"  Fitting {head_tag} on {len(y_train)} samples ...")
            model.head.fit(X_train, y_train)

            # Validation metrics
            y_true, y_score = collect_scores(model, val_loader, device, tta=True)
            val_metrics = compute_group_eval_metrics(y_true, y_score)
            print(
                f"  {head_tag} val | AUROC={val_metrics['AUROC']:.4f} "
                f"AUPRC={val_metrics['AUPRC']:.4f} "
                f"PPV@90R={val_metrics['PPV@90RECALL']:.4f} "
                f"Threshold={val_metrics['Threshold']:.4f}"
            )

            # Single W&B point summarizing the fit.
            wandb.log({
                "head_type": head_type,
                "head_hp": hp_value,
                "val/AUROC": val_metrics["AUROC"],
                "val/AUPRC": val_metrics["AUPRC"],
                "val/PPV@90RECALL": val_metrics["PPV@90RECALL"],
                "val/Threshold": val_metrics["Threshold"],
                "val/PPV": val_metrics["PPV"],
                "val/TPR": val_metrics["TPR"],
                "val/FPR": val_metrics["FPR"],
            })
            for key, value in val_metrics.items():
                wandb.summary[f"val/{key}"] = value
            run.finish()

            # Save full checkpoint for downstream submission.
            payload = create_model_checkpoint(
                model,
                model_config=cfg,
                extra_metadata={
                    "source_encoder": encoder_path,
                    "fold_index": fold_index,
                    "head_type": head_type,
                    "head_hp": hp_value,
                    "val_metrics": dict(val_metrics),
                    "val_targets": y_true,
                    "val_scores": y_score,
                    "input_size": fold_args.input_size,
                },
            )
            out_path = save_dir / f"{experiment_id}.pt"
            torch.save(payload, out_path)
            print(f"  Saved -> {out_path}")
            fold_payloads[(head_type, hp_value)] = {
                "fold_index": fold_index,
                "source_encoder": encoder_path,
                "model_config": dict(cfg),
                "model_state_dict": payload["model_state_dict"],
                "sklearn_head_state": payload.get("sklearn_head_state"),
                "val_metrics": dict(val_metrics),
                "val_targets": y_true,
                "val_scores": y_score,
                "per_fold_ckpt": str(out_path),
                "input_size": fold_args.input_size,
                "head_type": head_type,
                "head_hp": hp_value,
            }
    return fold_payloads, len(class_names)


def _save_ensemble_bundles(per_fold_results: list[dict], args):
    """For each (head_type, hp_value) seen in >=2 folds, write one ensemble bundle."""
    if not per_fold_results:
        return
    out_dir = Path(args.save_dir) / f"{args.experiment_id}_ensembles"
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, float | int], list[dict]] = {}
    for fold_dict in per_fold_results:
        for key, payload in fold_dict.items():
            grouped.setdefault(key, []).append(payload)

    for (head_type, hp_value), payloads in grouped.items():
        if len(payloads) < 2:
            print(f"[warn] {head_type}({hp_value}): only {len(payloads)} fold(s); skipping ensemble.")
            continue
        payloads.sort(key=lambda p: p["fold_index"])
        # Pooled threshold across the LOCO val sets (each fold contributes
        # predictions only on its held-out center, no overlap).
        all_targets, all_scores = [], []
        for fp in payloads:
            all_targets.extend(fp["val_targets"])
            all_scores.extend(fp["val_scores"])
        pooled_metrics = compute_group_eval_metrics(all_targets, all_scores)
        print(
            f"  Pooled LOCO val | {head_type}({hp_value}) | "
            f"AUROC={pooled_metrics['AUROC']:.4f} "
            f"AUPRC={pooled_metrics['AUPRC']:.4f} "
            f"PPV@90R={pooled_metrics['PPV@90RECALL']:.4f}"
        )

        bundle = {
            "is_ensemble": True,
            "ensemble_type": "loco_mean_proba",
            "ensemble_n_folds": len(payloads),
            "head_type": head_type,
            "head_hp": hp_value,
            "folds": payloads,
            "fold_indices": [fp["fold_index"] for fp in payloads],
            "source_encoders": [fp["source_encoder"] for fp in payloads],
            "model_config": dict(payloads[0]["model_config"]),
            "input_size": payloads[0]["input_size"],
            "pooled_val_metrics": dict(pooled_metrics),
        }
        head_tag = f"{head_type}{hp_value if head_type == 'knn' else f'C{hp_value}'}"
        out_path = out_dir / f"ensemble_{head_tag}.pt"
        torch.save(bundle, out_path)
        print(f"  Saved ensemble bundle -> {out_path}")

        # One comparable W&B row per ablation: the pooled cross-center (LOCO)
        # metrics. Logged into wandb.summary so each ensemble is a single sortable
        # row, distinct from the per-fold finetune runs (filter run_type=ensemble).
        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=f"{args.experiment_id}_ensemble_{head_tag}",
            job_type="ensemble",
            config={
                "run_type": "ensemble",
                "head_type": head_type,
                "head_hp": hp_value,
                "ensemble_n_folds": len(payloads),
                "fold_indices": [fp["fold_index"] for fp in payloads],
            },
            mode=args.wandb_mode,
            reinit=True,
        )
        wandb.log({f"pooled/{k}": v for k, v in pooled_metrics.items()})
        for key, value in pooled_metrics.items():
            wandb.summary[f"pooled/{key}"] = value
        run.finish()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _resolve_fold_range(args) -> list[int]:
    if args.loco:
        if args.num_folds <= 1:
            raise ValueError("--loco requires --num-folds >= 2.")
        return list(range(args.fold_index, args.num_folds))
    return [args.fold_index]


def _resolve_encoder_paths(args, n_folds_to_run: int) -> list[str]:
    if args.encoder_ckpt is None:
        raise ValueError("--encoder-ckpt is required for --stage finetune.")
    paths = [p.strip() for p in args.encoder_ckpt.split(",") if p.strip()]
    if len(paths) != n_folds_to_run:
        raise ValueError(
            f"--encoder-ckpt has {len(paths)} entries but the run covers "
            f"{n_folds_to_run} fold(s). Pass one comma-separated path per fold."
        )
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Encoder checkpoint not found: {path}")
    return paths


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    folds_to_run = _resolve_fold_range(args)
    print(f"Folds to run: {folds_to_run}")

    if args.stage == "pretrain":
        for fold_index in folds_to_run:
            _pretrain_one_fold(args, fold_index, device)
        return

    # finetune
    encoder_paths = _resolve_encoder_paths(args, len(folds_to_run))
    per_fold_results = []
    for fold_index, encoder_path in zip(folds_to_run, encoder_paths):
        fold_payloads, _ = _finetune_one_fold(args, fold_index, encoder_path, device)
        per_fold_results.append(fold_payloads)
    if args.loco and len(per_fold_results) >= 2:
        print("\n=== Building ensemble bundles ===")
        _save_ensemble_bundles(per_fold_results, args)


if __name__ == "__main__":
    parser = get_args_parser()
    parsed = parser.parse_args()
    main(parsed)
