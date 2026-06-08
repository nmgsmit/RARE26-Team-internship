#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figures_results.py
=====================
Generate the projection figures for the RARE25 Results section, styled for a
two-column IEEE paper. Outputs vector PDFs (preferred for LaTeX) and PNGs.

It reproduces the data-loading and PCA conventions of the marimo static
notebook (notebook_static.py), but renders with matplotlib at print quality
instead of plotly:

  * Projection-head features  ->  2-D PCA (n_components=2, random_state=42)
  * Two rows per experiment:
        Row A "by class"   : NDBE vs neoplasia
        Row B "by center"  : same coordinates, coloured by acquisition center
  * Train = train_all  (translucent circles, drawn first)
    Test  = evc_test    (opaque diamonds, dark outline, drawn on top)
  * Okabe-Ito colourblind-safe palette.

Figures produced (filenames match the .tex Results section):
  1. P0P1_projection.pdf
        GastroNet DINOv2 + CE  |  GastroNet DINOv2 + SupPro  |  DINOv3 + SupPro
  2. P2_temperature_projection.pdf
        tau = 0.07 | 0.1 | 0.3 | 0.5
  3. P3_bb_lr_projection.pdf
        backbone lr 1e-7 | 1e-6 | 1e-5 | 1e-4 | 1e-3
  4. P3_proj_lr_projection.pdf
        projection-head lr 3e-5 | 3e-4 | 3e-3
  5. P3_cls_lr_projection.pdf
        classifier lr 3e-6 | 3e-5 | 3e-4 | 3e-3 | 3e-2
  6. P4_batch_size_projection.pdf
        bs = 4 | 8 | 16 | 32
  7. P6_balanced_projection.pdf
        balanced 5% | 25% | 50%
  8. P8_cropscale_projection.pdf   (optional bonus, KNN head shown)
        crop scale 0.4 | 0.6 | 0.8 | 0.95

USAGE
-----
    # Generate all figures:
    python make_paper_figures.py --root features_out --out figures_paper

    # Generate only specific figures:
    python make_paper_figures.py --root features_out --out figures_paper --figures p0p1 p2

    # Skip the optional P8 figure:
    python make_paper_figures.py --root features_out --out figures_paper --skip-p8

Available figure keys for --figures: p0p1, p2, p3_bb, p3_proj, p3_cls, p4, p6, p8
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — must come before pyplot import
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA


# ── FIX 1: rcParams set at module level, before any figure is created ─────
# Previously this was inside build_figure(), after plt.subplots(), so the
# font settings were applied too late to affect already-created text objects.
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "pdf.fonttype": 42,   # embed fonts as Type 42 (TrueType) in PDF
    "ps.fonttype": 42,    # same for EPS/PS — required for IEEE submission
})


# ──────────────────────────────────────────────────────────────────────────
# Palette (Okabe-Ito, colourblind-safe) — mirrors the notebook.
# ──────────────────────────────────────────────────────────────────────────
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9",
    "bluish_green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermilion": "#D55E00", "reddish_purple": "#CC79A7",
}
CLASS_COLORS = [OKABE_ITO["bluish_green"], OKABE_ITO["vermilion"]]  # ndbe, neo
CENTER_COLORS = [OKABE_ITO["blue"], OKABE_ITO["orange"],
                 OKABE_ITO["reddish_purple"], OKABE_ITO["sky_blue"],
                 OKABE_ITO["yellow"], OKABE_ITO["black"]]

TRAIN_ALPHA = 0.5
TEST_ALPHA  = 0.95
TEST_CENTER_LABEL = "EVC (test center)"

# Marker sizes (points^2 for matplotlib scatter)
TRAIN_SIZE = 9
TEST_SIZE  = 12


# ──────────────────────────────────────────────────────────────────────────
# Stems to load for each figure.
# ──────────────────────────────────────────────────────────────────────────
STEMS_P0P1 = [
    ("P0_Base_GastronetDinoV2_1e-3_t1", "GastroNet DINOv2\nCross-Entropy"),
    ("P1_BB_GastronetDinoV2_t1",        "GastroNet DINOv2\nSupPro"),
    ("P1_BB_DinoV3_t1",                 "DINOv3\nSupPro"),
]

STEMS_P2 = [
    ("P2_Temp_0.07_t1", r"$\tau$ = 0.07"),
    ("P2_Temp_0.1_t1",  r"$\tau$ = 0.1"),
    ("P2_Temp_0.3_t1",  r"$\tau$ = 0.3"),
    ("P2_Temp_0.5_t1",  r"$\tau$ = 0.5"),
]

STEMS_P3_BB = [
    ("P3_BBLR_1e-7_t1", "lr = 1e-7"),
    ("P3_BBLR_1e-6_t1", "lr = 1e-6"),
    ("P3_BBLR_1e-5_t1", "lr = 1e-5"),
    ("P3_BBLR_1e-4_t1", "lr = 1e-4"),
    ("P3_BBLR_1e-3_t1", "lr = 1e-3"),
]

STEMS_P3_PROJ = [
    ("P3_ProjLR_3e-5_t1", "lr = 3e-5"),
    ("P3_ProjLR_3e-4_t1", "lr = 3e-4"),
    ("P3_ProjLR_3e-3_t1", "lr = 3e-3"),
]

STEMS_P3_CLS = [
    ("P3_ClassLR_3e-6_t1", "lr = 3e-6"),
    ("P3_ClassLR_3e-5_t1", "lr = 3e-5"),
    ("P3_ClassLR_3e-4_t1", "lr = 3e-4"),
    ("P3_ClassLR_3e-3_t1", "lr = 3e-3"),
    ("P3_ClassLR_3e-2_t1", "lr = 3e-2"),
]

STEMS_P4 = [
    ("P4_Batch_4_t1",  "bs = 4"),
    ("P4_Batch_8_t1",  "bs = 8"),
    ("P4_Batch_16_t1", "bs = 16"),
    ("P4_Batch_32_t1", "bs = 32"),
]

STEMS_P6 = [
    ("P6_BalSam_05_Linear", "Balanced 5%"),
    ("P6_BalSam_25_Linear", "Balanced 25%"),
    ("P6_BalSam_50_Linear", "Balanced 50%"),
]

STEMS_P8 = [
    ("P8_scale04_finetune_knn",  "Crop scale 0.4"),
    ("P8_scale06_finetune_knn",  "Crop scale 0.6"),
    ("P8_scale08_finetune_knn",  "Crop scale 0.8"),
    ("P8_scale095_finetune_knn", "Crop scale 0.95"),
]

# Registry used by --figures flag: key -> (stems_list, out_stem)
FIGURE_REGISTRY = {
    "p0p1":    (STEMS_P0P1,    "P0P1_projection"),
    "p2":      (STEMS_P2,      "P2_temperature_projection"),
    "p3_bb":   (STEMS_P3_BB,   "P3_bb_lr_projection"),
    "p3_proj": (STEMS_P3_PROJ, "P3_proj_lr_projection"),
    "p3_cls":  (STEMS_P3_CLS,  "P3_cls_lr_projection"),
    "p4":      (STEMS_P4,      "P4_batch_size_projection"),
    "p6":      (STEMS_P6,      "P6_balanced_projection"),
    "p8":      (STEMS_P8,      "P8_cropscale_projection"),
}


# ──────────────────────────────────────────────────────────────────────────
# Data loading — faithful to notebook_static.py
# ──────────────────────────────────────────────────────────────────────────
def resolve_centers(folder, paths, n, split):
    folder = Path(folder)
    for name in ("centers.npy", "center.npy", "domains.npy", "domain.npy",
                 "sites.npy", "site.npy", "hospital.npy", "hospitals.npy"):
        f = folder / name
        if f.exists():
            try:
                arr = np.load(f, allow_pickle=True)
                if len(arr) == n:
                    return np.array([str(x) for x in arr])
            except Exception:
                pass
    meta = folder / "meta.json"
    if meta.exists():
        try:
            m = json.loads(meta.read_text())
        except Exception:
            m = {}
        for k in ("centers", "center", "sites", "site", "domains", "domain",
                  "hospital", "hospitals"):
            if k in m:
                v = m[k]
                if isinstance(v, (list, tuple)) and len(v) == n:
                    return np.array([str(x) for x in v])
                if isinstance(v, (str, int)):
                    return np.array([str(v)] * n)
    pat_center = re.compile(
        r"(?:center|centre|site|hospital|clinic|domain)[\s_\-]?([A-Za-z0-9]+)",
        re.IGNORECASE)
    pat_evc = re.compile(r"EVC[\s_\-]?Barret", re.IGNORECASE)
    if paths is not None and len(paths) == n:
        labels, any_hit = [], False
        for p in paths:
            s = str(p)
            mm = pat_center.search(s)
            if mm:
                labels.append(f"center {mm.group(1)}"); any_hit = True
            elif pat_evc.search(s):
                labels.append("EVC (test)"); any_hit = True
            else:
                labels.append("unknown")
        if any_hit:
            return np.array(labels)
    return np.array(["unknown"] * n)


def load_experiment(root, stem):
    root = Path(root).expanduser()
    feats_all, labels_all, tags_all, centers_all = [], [], [], []
    class_names = None
    for split in ("train_all", "evc_test"):
        folder = root / f"{stem}__{split}"
        if not folder.exists():
            cands = list(root.glob(f"{stem}*{split}*"))
            if not cands:
                continue
            folder = cands[0]
        meta_path = folder / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if class_names is None:
            class_names = meta["class_names"]
        feats  = np.load(folder / "features_proj.npy")
        lbls   = np.load(folder / "labels.npy")
        paths_f = folder / "paths.npy"
        paths = None
        if paths_f.exists():
            try:
                paths = np.load(paths_f, allow_pickle=True)
            except Exception:
                paths = None
        centers = resolve_centers(folder, paths, len(lbls), split)
        feats_all.append(feats)
        labels_all.append(lbls)
        tags_all.extend([split] * len(lbls))
        centers_all.append(centers)
    if not feats_all:
        return None
    return {
        "features":    np.concatenate(feats_all, 0),
        "labels":      np.concatenate(labels_all, 0),
        "tags":        np.array(tags_all),
        "centers":     np.concatenate(centers_all, 0),
        "class_names": class_names,
    }


def pca_coords(X):
    return PCA(n_components=2, random_state=42).fit_transform(X)


# ──────────────────────────────────────────────────────────────────────────
# Panel drawing
# ──────────────────────────────────────────────────────────────────────────
def style_axis(ax):
    """Clean IEEE-style panel: no ticks, thin frame, stretched aspect ratio."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")
    
    # "auto" allows Matplotlib to stretch the Y-axis to fill the square subplot,
    # recreating the exact visual separation seen in the interactive tool.
    ax.set_aspect("auto")


def draw_class_panel(ax, coords, bundle):
    cn = bundle["class_names"]
    y  = bundle["labels"]
    t  = bundle["tags"]
    # Draw train first (translucent, under) then test on top (opaque, outlined)
    for split in ("train_all", "evc_test"):
        is_tr = split == "train_all"
        for ci in range(len(cn)):
            m = (y == ci) & (t == split)
            if not m.any():
                continue
            ax.scatter(
                coords[m, 0], coords[m, 1],
                s=TRAIN_SIZE if is_tr else TEST_SIZE,
                marker="o" if is_tr else "D",
                facecolor=CLASS_COLORS[ci % len(CLASS_COLORS)],
                alpha=TRAIN_ALPHA if is_tr else TEST_ALPHA,
                linewidths=0.2 if is_tr else 0.6,
                edgecolors="white" if is_tr else "#222222",
                zorder=1 if is_tr else 3,
                rasterized=True,   # keeps PDF file size manageable
            )
    style_axis(ax)


def draw_center_panel(ax, coords, bundle):
    t       = bundle["tags"]
    centers = bundle["centers"].astype(object).copy()
    centers[t == "evc_test"] = TEST_CENTER_LABEL
    uniq = sorted(set(centers.tolist()))
    if TEST_CENTER_LABEL in uniq:
        uniq.remove(TEST_CENTER_LABEL)
        uniq = uniq + [TEST_CENTER_LABEL]   # test center always last / consistent color
    color_of = {lab: CENTER_COLORS[i % len(CENTER_COLORS)]
                for i, lab in enumerate(uniq)}
    for split in ("train_all", "evc_test"):
        is_tr = split == "train_all"
        for lab in uniq:
            m = (centers == lab) & (t == split)
            if not m.any():
                continue
            ax.scatter(
                coords[m, 0], coords[m, 1],
                s=TRAIN_SIZE if is_tr else TEST_SIZE,
                marker="o" if is_tr else "D",
                facecolor=color_of[lab],
                alpha=TRAIN_ALPHA if is_tr else TEST_ALPHA,
                linewidths=0.2 if is_tr else 0.6,
                edgecolors="white" if is_tr else "#222222",
                zorder=1 if is_tr else 3,
                rasterized=True,
            )
    style_axis(ax)
    return uniq, color_of


# ──────────────────────────────────────────────────────────────────────────
# Figure assembly
# ──────────────────────────────────────────────────────────────────────────
def build_figure(root, stems, out_stem, out_dir, full_width=True, panel_in=1.55):
    """Load data, build the 2-row subplot figure, and write PDF + PNG.

    Parameters
    ----------
    root      : path to features_out directory
    stems     : list of (stem, panel_title) tuples
    out_stem  : filename stem (no extension)
    out_dir   : output directory
    full_width: if True cap at IEEE two-column width (7.16 in); else half-width
    panel_in  : desired panel size in inches before clamping
    """
    loaded, missing = [], []
    for stem, title in stems:
        b = load_experiment(root, stem)
        if b is None or b["features"].shape[0] < 2:
            missing.append(stem)
        else:
            loaded.append((stem, title, b))

    if missing:
        print(f"  [warn] {out_stem}: missing/empty -> {', '.join(missing)}")
    if not loaded:
        print(f"  [skip] {out_stem}: no data found, figure not written.")
        return None

    n           = len(loaded)
    coords_list = [pca_coords(b["features"]) for _, _, b in loaded]

    # ── FIX 2: derive fig_h from the *clamped* width so panels stay square ─
    # Previously fig_h was computed from the unclamped panel_in, so for wide
    # figures (e.g. 5-panel P3 sweeps) the height was taller than the panels
    # were wide after clamping, making each panel taller than it is wide.
    max_w       = 7.16 if full_width else 3.5
    legend_w    = 1.4   # inches reserved for the right-hand legend column
    raw_w       = panel_in * n + legend_w
    fig_w       = min(raw_w, max_w)
    panel_actual = (fig_w - legend_w) / n   # actual panel width after clamping
    fig_h       = panel_actual * 2 + 0.5    # 2 rows + small top/bottom margin

    fig, axes = plt.subplots(2, n, figsize=(fig_w, fig_h), squeeze=False)

    centers_legend = {}
    for j, ((_, title, b), coords) in enumerate(zip(loaded, coords_list)):
        draw_class_panel(axes[0][j], coords, b)
        uniq, color_of = draw_center_panel(axes[1][j], coords, b)
        for lab in uniq:
            centers_legend.setdefault(lab, color_of[lab])
        axes[0][j].set_title(title, fontsize=12, pad=6)

    axes[0][0].set_ylabel("by class",  fontsize=12)
    axes[1][0].set_ylabel("by center", fontsize=12)

    # ── Legends (right margin) ────────────────────────────────────────────
    class_names    = loaded[0][2]["class_names"]
    class_handles  = [
        Line2D([], [], marker="o", linestyle="none",
               markerfacecolor=CLASS_COLORS[ci % len(CLASS_COLORS)],
               markeredgecolor="#222", markersize=6, label=str(cname))
        for ci, cname in enumerate(class_names)
    ]
    split_handles  = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="#888",
               markeredgecolor="white", markersize=6, alpha=0.6, label="train"),
        Line2D([], [], marker="D", linestyle="none", markerfacecolor="#888",
               markeredgecolor="#222", markersize=6, label="test (EVC)"),
    ]
    center_handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=c,
               markeredgecolor="#222", markersize=6, label=str(lab))
        for lab, c in centers_legend.items()
    ]

# ── Legends (bottom margin) ───────────────────────────────────────────
    fig.legend(
        handles=class_handles + split_handles,
        loc="upper center", 
        bbox_to_anchor=(0.5, 0.18),  # <--- Shifted to sit just below the new bottom
        ncol=4, frameon=False, 
        fontsize=11,                 # <--- Increased from 9
        handletextpad=0.3,
        title="Class / Split", 
        title_fontsize=12,           # <--- Increased from 10
    )
    
    fig.legend(
        handles=center_handles,
        loc="upper center", 
        bbox_to_anchor=(0.5, 0.08),  # <--- Tucked tighter under the first legend
        ncol=3, 
        frameon=False, 
        fontsize=11,                 # <--- Increased from 9
        handletextpad=0.3, 
        title="Center", 
        title_fontsize=12,           # <--- Increased from 10
    )

    # <--- Reduced 'bottom' from 0.32 to 0.20 to pull the figures down closer to the legends
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.20, 
                        wspace=0.06, hspace=0.08)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{out_stem}.pdf"
    png = out_dir / f"{out_stem}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [ok]   wrote {pdf}  and  {png}")
    return pdf


# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", default="features_out",
                    help="features root directory (default: features_out)")
    ap.add_argument("--out", default="figures",
                    help="output directory for figures (default: figures_paper)")
    ap.add_argument("--skip-p8", action="store_true",
                    help="do not generate the optional P8 crop-scale figure")
    # ── FIX 4: --figures flag so you can regenerate a single figure ───────
    ap.add_argument(
        "--figures", nargs="+", metavar="KEY",
        help=(
            "only generate these figures (space-separated keys). "
            f"choices: {', '.join(sorted(FIGURE_REGISTRY))}. "
            "default: all"
        ),
    )
    args = ap.parse_args()

    print(f"Loading features from: {args.root}")
    print(f"Writing figures to:    {args.out}\n")

    # Determine which figures to build
    if args.figures:
        unknown = set(args.figures) - set(FIGURE_REGISTRY)
        if unknown:
            ap.error(f"Unknown figure key(s): {', '.join(sorted(unknown))}. "
                     f"Valid keys: {', '.join(sorted(FIGURE_REGISTRY))}")
        keys_to_run = [k for k in FIGURE_REGISTRY if k in args.figures]
    else:
        keys_to_run = list(FIGURE_REGISTRY)

    if args.skip_p8 and "p8" in keys_to_run:
        keys_to_run.remove("p8")

    for key in keys_to_run:
        stems, out_stem = FIGURE_REGISTRY[key]
        print(f"{out_stem}:")
        build_figure(args.root, stems, out_stem, args.out, full_width=True)


if __name__ == "__main__":
    main()