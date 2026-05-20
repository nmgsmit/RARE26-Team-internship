"""Inspect a LOCO submission checkpoint.

Usage:
    python inspect_submission.py ./checkpoints/<RUN_TAG>/finetune/<RUN_TAG>_finetune_submission.pt

Prints the calibrated inference threshold and the headline metrics that were
recorded when the submission .pt was saved. Use this to confirm what you're
about to upload.
"""
import argparse
import json
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description="Inspect a LOCO submission .pt")
    parser.add_argument("checkpoint", help="Path to <RUN_TAG>_submission.pt")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location="cpu")

    keys_of_interest = [
        "experiment_id",
        "stage",
        "loss_name",
        "selected_threshold",
        "loco_folds",
        "loco_centers",
        "loco_pooled_val_AUPRC",
        "loco_pooled_val_AUROC",
        "loco_pooled_val_PPV@90R",
        "submission_test_AUPRC",
        "submission_test_AUROC",
        "submission_test_PPV@90R",
        "submission_kind",
    ]

    summary = {k: ckpt.get(k, None) for k in keys_of_interest}
    model_config = ckpt.get("model_config", {})
    summary["input_size"] = model_config.get("input_size")
    summary["backbone_name"] = model_config.get("backbone_name")
    summary["head_type"] = model_config.get("head_type")
    summary["n_classes"] = model_config.get("n_classes")

    print(json.dumps(summary, indent=2))

    threshold = summary.get("selected_threshold")
    if threshold is not None:
        print(
            f"\nInference rule: classify as neoplasia iff "
            f"softmax_prob_class1 >= {threshold:.6f}"
        )


if __name__ == "__main__":
    main()
