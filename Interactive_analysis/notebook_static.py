"""
Static comparison notebook (marimo).
Run with:
    marimo edit notebook_static.py

Non-interactive: every figure is hardcoded. For each experiment we load
*both* the `__evc_test` and `__train_all` folders, project the **projection
head** features with PCA, and render TWO rows of panels per group:

  Row A ("class"):  ndbe vs neo, train + test on the same axes. The deployed
                    model's p(class 1) heatmap background is OPT-IN (off by
                    default) because for non-probabilistic heads the stored
                    array may be a raw decision score, not a calibrated
                    probability; see SHOW_PROB_HEATMAP below.
  Row B ("center"): the SAME PCA coordinates, but colored by acquisition
                    *center* (hospital/site) to visualize domain shift
                    between training centers and the test center.

Rendering choices (per request):
  • Train points are drawn FIRST and made translucent; test points are drawn
    ON TOP, opaque, with a dark outline, so the test set stays visible.
  • Colors use the Okabe–Ito colorblind-safe palette; the probability heatmap
    (when shown) uses a colorblind-safe diverging scale (no red/green confusion).
  • Layout is tuned for IEEE-style figure extraction: clean serif-free fonts,
    thin panel frames, no gridlines, compact margins, panel letters in titles.

Groups follow the experimental design:
  P0 — Base architectures, cross-entropy loss
  P1 — Same backbones, SupCon loss
  P2 — Temperature sweep
  P3 — Learning-rate sweeps (backbone, projection head, classifier)
  P4 — Batch size sweep
  P5 — Combined SupPro + SupMin loss weighting
  P6 — Balanced sampling (oversample minority class)
  P7 — Crop scale + crop type (ROI vs random) on input images
  P8 — Classifier head sweep (KNN / Linear / SVM / MLP) across crop scales
"""
import marimo
__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _imports():
    import json
    import re
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.decomposition import PCA
    return PCA, Path, go, json, make_subplots, mo, np, re


@app.cell
def _intro(mo):
    mo.md("""
    # Static checkpoint comparison

    Projection-head (128-D) features projected to 2-D with **PCA**. Each
    experiment gets **two rows** of panels sharing the same PCA coordinates:

    **Row A — class.** ndbe vs neo, for both splits. Markers:
      ◆ = test (`evc_test`), drawn on top &nbsp;•&nbsp; ● = train (`train_all`),
      drawn first and made translucent so the test set stays visible.
      The **background heatmap** of the deployed model's p(class 1) is **opt-in**
      (off by default) — see `SHOW_PROB_HEATMAP` in the style cell.

    **Row B — center (domain shift).** The same points, recolored by
    acquisition **center**. Training centers vs the test center; systematic
    separation here indicates domain shift. Test still uses the ◆ marker.
    """)
    return


@app.cell
def _config(mo):
    features_root_ui = mo.ui.text(
        value="features_out",
        label="Features root directory",
        full_width=True,
    )
    features_root_ui
    return (features_root_ui,)


@app.cell
def _style_constants():
    # ── Okabe–Ito colorblind-safe palette ────────────────────────────────
    # https://jfly.uni-koeln.de/color/  — distinguishable under the common
    # forms of color-vision deficiency and in greyscale.
    OKABE_ITO = {
        "black":        "#000000",
        "orange":       "#E69F00",
        "sky_blue":     "#56B4E9",
        "bluish_green": "#009E73",
        "yellow":       "#F0E442",
        "blue":         "#0072B2",
        "vermilion":    "#D55E00",
        "reddish_purple": "#CC79A7",
    }
    # Class colors (2-class: ndbe / neo). Chosen to avoid the red/green pair.
    CLASS_COLORS = [OKABE_ITO["bluish_green"], OKABE_ITO["vermilion"]]
    # Center colors, used in row B. Ordered list; test center gets its own.
    CENTER_COLORS = [
        OKABE_ITO["blue"],
        OKABE_ITO["orange"],
        OKABE_ITO["reddish_purple"],
        OKABE_ITO["sky_blue"],
        OKABE_ITO["yellow"],
        OKABE_ITO["black"],
    ]
    # Diverging, colorblind-safe heatmap for p(class 1): blue (low) -> grey ->
    # orange (high). Avoids the RdBu red/green ambiguity for deuteranopes.
    PROB_COLORSCALE = [
        [0.0, "#0072B2"],   # p≈0  -> blue   (class 0)
        [0.5, "#F2F2F2"],   # p≈.5 -> light grey (ambiguous)
        [1.0, "#E69F00"],   # p≈1  -> orange (class 1)
    ]
    # Marker opacities: train translucent, test opaque & outlined.
    TRAIN_ALPHA = 0.35
    TEST_ALPHA  = 0.95
    HEATMAP_ALPHA = 0.22
    # If True, the test split is shown as its own distinct center in row B
    # (most common for domain-shift plots: train centers vs a held-out center).
    TEST_IS_OWN_CENTER = True
    TEST_CENTER_LABEL = "EVC (test center)"
    # ── Probability heatmap: OPT-IN ───────────────────────────────────────
    # The Row-A background heatmap claims to show the deployed model's
    # p(class 1). That claim is only honest when `deployed_probs.npy` holds a
    # genuine, calibrated probability in [0, 1]. For non-probabilistic heads
    # (KNN vote-free, SVM decision_function) the stored values may be raw
    # scores, and we do NOT have the decision thresholds, so the background
    # would be an uninterpretable, authoritative-looking surface. Default OFF.
    # Set True only when every loaded array is a real calibrated probability.
    SHOW_PROB_HEATMAP = False
    return (
        CLASS_COLORS,
        CENTER_COLORS,
        PROB_COLORSCALE,
        TRAIN_ALPHA,
        TEST_ALPHA,
        HEATMAP_ALPHA,
        TEST_IS_OWN_CENTER,
        TEST_CENTER_LABEL,
        SHOW_PROB_HEATMAP,
    )


@app.cell
def _experiment_groups():
    # Each entry is (group_id, title, subtitle, [(stem, panel_label), ...])
    experiment_groups = [
        (
            "P0",
            "P0 — Base architectures (Cross-Entropy loss)",
            "Five backbones, all trained with standard cross-entropy. "
            "Compare to P1 below: **only the loss function differs**.",
            [
                ("P0_Base_ResNet50_1e-3_t1",            "ResNet50"),
                ("P0_Base_GastroSimCLR_1e-3_t1",        "GastroSimCLR"),
                ("P0_Base_GastronetDinoV2_1e-3_t1",     "GastronetDinoV2"),
                ("P0_Base_GastroMoCO_1e-3_t1",          "GastroMoCO"),
                ("P0_Base_DinoV3_1e-3_t1",              "DinoV3"),
            ],
        ),
        (
            "P1",
            "P1 — Same backbones, SupCon (SupPro) loss",
            "Same five backbones as P0 but trained with the supervised contrastive "
            "loss. **The ONLY change versus P0 is the loss function** — anything "
            "different in the embedding geometry is attributable to that.",
            [
                ("P1_BB_ResNet50_t1",            "ResNet50"),
                ("P1_BB_GastroSimCLR_t1",        "GastroSimCLR"),
                ("P1_BB_GastronetDinoV2_t1",     "GastronetDinoV2"),
                ("P1_BB_GastroMoCO_t1",          "GastroMoCO"),
                ("P1_BB_DinoV3_t1",              "DinoV3"),
            ],
        ),
        (
            "P2",
            "P2 — Temperature sweep (SupCon loss)",
            "We vary the **temperature** of the contrastive loss while keeping "
            "everything else fixed.",
            [
                ("P2_Temp_0.07_t1", "τ = 0.07"),
                ("P2_Temp_0.1_t1",  "τ = 0.1"),
                ("P2_Temp_0.3_t1",  "τ = 0.3"),
                ("P2_Temp_0.5_t1",  "τ = 0.5"),
            ],
        ),
        (
            "P3_BB",
            "P3 — Backbone learning rate sweep",
            "We vary **only the backbone learning rate**.",
            [
                ("P3_BBLR_1e-7_t1", "lr = 1e-7"),
                ("P3_BBLR_1e-6_t1", "lr = 1e-6"),
                ("P3_BBLR_1e-5_t1", "lr = 1e-5"),
                ("P3_BBLR_1e-4_t1", "lr = 1e-4"),
                ("P3_BBLR_1e-3_t1", "lr = 1e-3"),
            ],
        ),
        (
            "P3_PROJ",
            "P3 — Projection-head learning rate sweep",
            "We vary **only the projection-head learning rate**.",
            [
                ("P3_ProjLR_3e-5_t1", "lr = 3e-5"),
                ("P3_ProjLR_3e-4_t1", "lr = 3e-4"),
                ("P3_ProjLR_3e-3_t1", "lr = 3e-3"),
            ],
        ),
        (
            "P3_CLS",
            "P3 — Classifier learning rate sweep",
            "We vary **only the classifier-head learning rate** (values "
            "indicated in the panel labels).",
            [
                ("P3_ClassLR_3e-6_t1", "lr = 3e-6"),
                ("P3_ClassLR_3e-5_t1", "lr = 3e-5"),
                ("P3_ClassLR_3e-4_t1", "lr = 3e-4"),
                ("P3_ClassLR_3e-3_t1", "lr = 3e-3"),
                ("P3_ClassLR_3e-2_t1", "lr = 3e-2"),
            ],
        ),
        (
            "P4",
            "P4 — Batch size sweep",
            "We vary the **batch size** used during contrastive training.",
            [
                ("P4_Batch_4_t1",  "bs = 4"),
                ("P4_Batch_8_t1",  "bs = 8"),
                ("P4_Batch_16_t1", "bs = 16"),
                ("P4_Batch_32_t1", "bs = 32"),
            ],
        ),
        (
            "P5",
            "P5 — Combined SupPro + SupMin loss",
            "Linear combination of SupPro and SupMin losses. Weights below.",
            [
                ("P5_pro1_min0_t1",      "SupPro 1.0 / SupMin 0.0"),
                ("P5_pro0_min1_t1",      "SupPro 0.0 / SupMin 1.0"),
                ("P5_pro05_min05_t1",    "SupPro 0.5 / SupMin 0.5"),
                ("P5_pro025_min075_t1",  "SupPro 0.25 / SupMin 0.75"),
                ("P5_pro075_min025_t1",  "SupPro 0.75 / SupMin 0.25"),
                ("P5_pro09_min01_t1",    "SupPro 0.9 / SupMin 0.1"),
                ("P5_pro01_min09_t1",    "SupPro 0.1 / SupMin 0.9"),
            ],
        ),
        (
            "P6",
            "P6 — Balanced sampling",
            "Same DINOv2 backbone + SupPro loss as the P2–P5 base, but the "
            "training sampler **oversamples the minority class**. We vary "
            "**only the balanced-sampling fraction** (panel labels). Everything "
            "else (backbone, τ=0.1, lrs, batch size, Linear head, random crop) "
            "is held fixed.",
            [
                ("P6_BalSam_05_Linear", "balanced 5%"),
                ("P6_BalSam_25_Linear", "balanced 25%"),
                ("P6_BalSam_50_Linear", "balanced 50%"),
            ],
        ),
        (
            "P7",
            "P7 — Crop scale & crop type",
            "We vary the **crop applied to the input images** — both the crop "
            "*scale* (0.4 / 0.8 / 1.0) and the crop *type* (ROI-guided vs "
            "random). Backbone, loss, lrs, batch size and 50% balanced sampling "
            "are held fixed; **only the cropping changes**. (At scale 1.0 the "
            "crop type is irrelevant — the whole image is kept.)",
            [
                ("P7_ROI_crop04",      "ROI · scale 0.4"),
                ("P7_Random_crop04",   "Random · scale 0.4"),
                ("P7_ROI_crop08_REAL", "ROI · scale 0.8"),
                ("P7_Random_crop08",   "Random · scale 0.8"),
                ("P7_Random_crop1",    "Random · scale 1.0"),
            ],
        ),
        (
            "P8_scale04",
            "P8 — Classifier head sweep · crop scale [0.4, 1.0]",
            "Same DINOv2 backbone + SupPro features, but we swap the **classifier "
            "head** (KNN / Linear / SVM / MLP) on top of frozen features. Crop "
            "scale range is [0.4, 1.0] with 50/50 ROI+Random crops and 20/80 "
            "balanced sampling. **Only the head type changes across panels.**",
            [
                ("P8_scale04_finetune_knn",           "KNN"),
                ("P8_scale04_finetune_linear",        "Linear"),
                ("P8_scale04_finetune_svm",           "SVM"),
                ("P8_scale04_finetune_mlp_fullwidth", "MLP"),
            ],
        ),
        (
            "P8_scale06",
            "P8 — Classifier head sweep · crop scale [0.6, 1.0]",
            "As above, with crop scale range [0.6, 1.0]. **Only the head type "
            "changes across panels.**",
            [
                ("P8_scale06_finetune_knn",           "KNN"),
                ("P8_scale06_finetune_linear",        "Linear"),
                ("P8_scale06_finetune_svm",           "SVM"),
                ("P8_scale06_finetune_mlp_fullwidth", "MLP"),
            ],
        ),
        (
            "P8_scale08",
            "P8 — Classifier head sweep · crop scale [0.8, 1.0]",
            "As above, with crop scale range [0.8, 1.0]. **Only the head type "
            "changes across panels.**",
            [
                ("P8_scale08_finetune_knn",           "KNN"),
                ("P8_scale08_finetune_linear",        "Linear"),
                ("P8_scale08_finetune_svm",           "SVM"),
                ("P8_scale08_finetune_mlp_fullwidth", "MLP"),
            ],
        ),
        (
            "P8_scale095",
            "P8 — Classifier head sweep · crop scale [0.95, 1.0]",
            "As above, with crop scale range [0.95, 1.0] (near-full image). "
            "**Only the head type changes across panels.**",
            [
                ("P8_scale095_finetune_knn",           "KNN"),
                ("P8_scale095_finetune_linear",        "Linear"),
                ("P8_scale095_finetune_svm",           "SVM"),
                ("P8_scale095_finetune_mlp_fullwidth", "MLP"),
            ],
        ),
    ]
    return (experiment_groups,)


@app.function
def resolve_centers(folder, paths, n, split):
    """Best-effort acquisition-center label for each of the `n` rows of ONE
    split folder. Tries, in order:
      1. a centers/domains/sites/hospital .npy array in the folder
      2. a 'centers'/'center'/'site'/'domain' field in meta.json
      3. a regex on the image paths (looks for center/site/hospital tokens)
    Returns a length-`n` numpy array of string labels, or an array of
    "unknown" if nothing resolves. Public name so cells can use it.
    """
    import json
    import re
    from pathlib import Path
    import numpy as np

    folder = Path(folder)

    # 1) explicit array file
    for _name in ("centers.npy", "center.npy", "domains.npy", "domain.npy",
                  "sites.npy", "site.npy", "hospital.npy", "hospitals.npy"):
        _f = folder / _name
        if _f.exists():
            try:
                arr = np.load(_f, allow_pickle=True)
                if len(arr) == n:
                    return np.array([str(x) for x in arr])
            except Exception:
                pass

    # 2) meta.json field (either a per-row list, or a single scalar for the split)
    _meta = folder / "meta.json"
    if _meta.exists():
        try:
            meta = json.loads(_meta.read_text())
        except Exception:
            meta = {}
        for _k in ("centers", "center", "sites", "site", "domains", "domain",
                   "hospital", "hospitals"):
            if _k in meta:
                _v = meta[_k]
                if isinstance(_v, (list, tuple)) and len(_v) == n:
                    return np.array([str(x) for x in _v])
                if isinstance(_v, (str, int)):
                    return np.array([str(_v)] * n)

    # 3) parse from image paths
    #    Matches this project's layout:
    #      train: ../../data/center_1/...  ../../data/center_2/...  (any center_N)
    #      test:  ../../EVC_Barretts_FullSet 2/images/...  -> the EVC test center
    #    plus generic fallbacks (center1, site-3, hospitalA, c01, ...).
    _pat_center = re.compile(
        r"(?:center|centre|site|hospital|clinic|domain)[\s_\-]?([A-Za-z0-9]+)",
        re.IGNORECASE,
    )
    _pat_evc = re.compile(r"EVC[\s_\-]?Barret", re.IGNORECASE)
    if paths is not None and len(paths) == n:
        _labels = []
        _any = False
        for _p in paths:
            _s = str(_p)
            _m = _pat_center.search(_s)
            if _m:
                _labels.append(f"center {_m.group(1)}")
                _any = True
            elif _pat_evc.search(_s):
                # The EVC Barrett's full set is the held-out test center.
                _labels.append("EVC (test)")
                _any = True
            else:
                _labels.append("unknown")
        if _any:
            return np.array(_labels)

    # nothing worked
    return np.array(["unknown"] * n)


@app.function
def load_experiment(features_root, stem):
    """Load projection-head features + labels + deployed probs + center labels
    for ONE stem. Returns dict with merged train+test arrays and a `tags`
    array marking the split. Returns None if neither split is on disk.
    Public name (no leading underscore) so it's visible from cells.
    """
    import json
    from pathlib import Path
    import numpy as np
    root = Path(features_root).expanduser()
    feats_all, labels_all, probs_all, tags_all, centers_all = [], [], [], [], []
    class_names = None
    for split in ("train_all", "evc_test"):
        folder = root / f"{stem}__{split}"
        if not folder.exists():
            # Fall back to any folder whose name starts with stem and ends with split
            candidates = list(root.glob(f"{stem}*{split}*"))
            if not candidates:
                continue
            folder = candidates[0]
        meta_path = folder / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if class_names is None:
            class_names = meta["class_names"]
        feats = np.load(folder / "features_proj.npy")  # projection head
        lbls = np.load(folder / "labels.npy")
        probs_raw = np.load(folder / "deployed_probs.npy")
        if probs_raw.ndim == 2:
            probs = probs_raw[:, 1]
        else:
            probs = probs_raw
        # paths are optional but needed for the path-based center fallback
        _paths_f = folder / "paths.npy"
        if _paths_f.exists():
            try:
                _paths = np.load(_paths_f, allow_pickle=True)
            except Exception:
                _paths = None
        else:
            _paths = None
        _centers = resolve_centers(folder, _paths, len(lbls), split)
        feats_all.append(feats)
        labels_all.append(lbls)
        probs_all.append(probs)
        tags_all.extend([split] * len(lbls))
        centers_all.append(_centers)
    if not feats_all:
        return None
    return {
        "features":       np.concatenate(feats_all, axis=0),
        "labels":         np.concatenate(labels_all, axis=0),
        "deployed_probs": np.concatenate(probs_all, axis=0),
        "tags":           np.array(tags_all),
        "centers":        np.concatenate(centers_all, axis=0),
        "class_names":    class_names,
    }


@app.function
def pca_coords(X):
    """2-D PCA of a feature matrix (shared by both rows so coordinates match)."""
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=42).fit_transform(X)


@app.function
def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


@app.function
def prob_heatmap_trace(coords, p, colorscale, alpha):
    """Build the semi-transparent p(class 1) heatmap background trace.

    NOTE: this is a 2-D PCA-space interpolation of the stored per-point probs,
    not the classifier's true decision surface. It is only meaningful when the
    stored probs are genuine calibrated probabilities; see SHOW_PROB_HEATMAP.
    """
    import numpy as np
    import plotly.graph_objects as go
    _x0, _x1 = coords[:, 0].min(), coords[:, 0].max()
    _y0, _y1 = coords[:, 1].min(), coords[:, 1].max()
    _pad_x = (_x1 - _x0) * 0.08 + 1e-6
    _pad_y = (_y1 - _y0) * 0.08 + 1e-6
    _xg = np.linspace(_x0 - _pad_x, _x1 + _pad_x, 60)
    _yg = np.linspace(_y0 - _pad_y, _y1 + _pad_y, 60)
    _Xg, _Yg = np.meshgrid(_xg, _yg)
    _grid_pts = np.column_stack([_Xg.ravel(), _Yg.ravel()])
    try:
        from scipy.interpolate import RBFInterpolator
        _rbf = RBFInterpolator(coords, p, kernel="linear", smoothing=0.1)
        _Pg = np.clip(_rbf(_grid_pts).reshape(_Xg.shape), 0.0, 1.0)
    except Exception:
        _diffs = _grid_pts[:, None, :] - coords[None, :, :]
        _dists = np.linalg.norm(_diffs, axis=-1)
        _nn = np.argmin(_dists, axis=1)
        _Pg = p[_nn].reshape(_Xg.shape)
    return go.Heatmap(
        x=_xg, y=_yg, z=_Pg, zmin=0.0, zmax=1.0,
        colorscale=colorscale, opacity=alpha,
        showscale=False, hoverinfo="skip",
    )


@app.function
def make_class_panel(fig, row, col, coords, bundle, style):
    """Row A: color by CLASS. Train drawn first (translucent), test on top
    (opaque + outline). Heatmap of p(class 1) behind everything — only when
    SHOW_PROB_HEATMAP is True."""
    import numpy as np
    import plotly.graph_objects as go
    (CLASS_COLORS, CENTER_COLORS, PROB_COLORSCALE, TRAIN_ALPHA, TEST_ALPHA,
     HEATMAP_ALPHA, TEST_IS_OWN_CENTER, TEST_CENTER_LABEL,
     SHOW_PROB_HEATMAP) = style
    class_names = bundle["class_names"]
    y = bundle["labels"]
    p = bundle["deployed_probs"]
    t = bundle["tags"]

    if SHOW_PROB_HEATMAP:
        fig.add_trace(prob_heatmap_trace(coords, p, PROB_COLORSCALE, HEATMAP_ALPHA),
                      row=row, col=col)

    # Draw order: train first (so it sits UNDER), then test on top.
    for _split in ("train_all", "evc_test"):
        _is_train = _split == "train_all"
        _alpha = TRAIN_ALPHA if _is_train else TEST_ALPHA
        _symbol = "circle" if _is_train else "diamond"
        for _cls_idx, _cls_name in enumerate(class_names):
            _mask = (y == _cls_idx) & (t == _split)
            if not _mask.any():
                continue
            _color = CLASS_COLORS[_cls_idx % len(CLASS_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=coords[_mask, 0], y=coords[_mask, 1],
                    mode="markers",
                    name=f"{_cls_name} ({_split})",
                    marker=dict(
                        size=5 if _is_train else 8,
                        symbol=_symbol,
                        color=hex_to_rgba(_color, _alpha),
                        line=dict(
                            width=0.3 if _is_train else 1.1,
                            color="rgba(255,255,255,0.6)" if _is_train else "#222222",
                        ),
                    ),
                    showlegend=False,
                    hovertemplate=(
                        f"{_cls_name} / {_split}<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )


@app.function
def make_center_panel(fig, row, col, coords, bundle, style):
    """Row B: same coords, color by CENTER (domain shift). Train translucent &
    drawn first, test opaque diamonds on top. Test optionally its own center."""
    import numpy as np
    import plotly.graph_objects as go
    (CLASS_COLORS, CENTER_COLORS, PROB_COLORSCALE, TRAIN_ALPHA, TEST_ALPHA,
     HEATMAP_ALPHA, TEST_IS_OWN_CENTER, TEST_CENTER_LABEL,
     SHOW_PROB_HEATMAP) = style
    t = bundle["tags"]
    centers = bundle["centers"].astype(object).copy()

    # Optionally relabel the test split as a single distinct center.
    if TEST_IS_OWN_CENTER:
        centers[t == "evc_test"] = TEST_CENTER_LABEL

    # Stable color assignment: sort labels, but keep the test center last so
    # it gets a consistent distinctive color.
    _uniq = sorted(set(centers.tolist()))
    if TEST_IS_OWN_CENTER and TEST_CENTER_LABEL in _uniq:
        _uniq.remove(TEST_CENTER_LABEL)
        _uniq = _uniq + [TEST_CENTER_LABEL]
    _color_of = {lab: CENTER_COLORS[i % len(CENTER_COLORS)]
                 for i, lab in enumerate(_uniq)}

    for _split in ("train_all", "evc_test"):
        _is_train = _split == "train_all"
        _alpha = TRAIN_ALPHA if _is_train else TEST_ALPHA
        _symbol = "circle" if _is_train else "diamond"
        for _lab in _uniq:
            _mask = (centers == _lab) & (t == _split)
            if not _mask.any():
                continue
            _color = _color_of[_lab]
            fig.add_trace(
                go.Scatter(
                    x=coords[_mask, 0], y=coords[_mask, 1],
                    mode="markers",
                    name=f"{_lab} ({_split})",
                    marker=dict(
                        size=5 if _is_train else 8,
                        symbol=_symbol,
                        color=hex_to_rgba(_color, _alpha),
                        line=dict(
                            width=0.3 if _is_train else 1.1,
                            color="rgba(255,255,255,0.6)" if _is_train else "#222222",
                        ),
                    ),
                    showlegend=False,
                    hovertemplate=f"{_lab} / {_split}<extra></extra>",
                ),
                row=row, col=col,
            )
    return _uniq, _color_of


@app.function
def render_group(features_root, group, style):
    """Render one experiment group: TWO subplot rows (class, then center)
    per group, as a marimo-displayable object. Public name."""
    import numpy as np
    import marimo as mo
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    (CLASS_COLORS, CENTER_COLORS, PROB_COLORSCALE, TRAIN_ALPHA, TEST_ALPHA,
     HEATMAP_ALPHA, TEST_IS_OWN_CENTER, TEST_CENTER_LABEL,
     SHOW_PROB_HEATMAP) = style

    group_id, title, subtitle, items = group
    loaded, missing = [], []
    for stem, label in items:
        bundle = load_experiment(features_root, stem)
        if bundle is None:
            missing.append(stem)
        elif bundle["features"].shape[0] >= 2:
            loaded.append((stem, label, bundle))
        else:
            missing.append(stem)
    if not loaded:
        return mo.vstack([
            mo.md(f"## {title}"),
            mo.md(subtitle),
            mo.md(f"⚠️ *None of the checkpoints in this group were found under "
                  f"`{features_root}`. Missing:* `{', '.join(missing)}`"),
        ])

    n = len(loaded)
    # Precompute shared PCA coords once per checkpoint (used by both rows).
    coords_list = [pca_coords(b["features"]) for _, _, b in loaded]

    # ── Two subplot rows: row 1 = class, row 2 = center ───────────────────
    _col_titles = [lbl for _, lbl, _ in loaded]
    fig = make_subplots(
        rows=2, cols=n,
        subplot_titles=_col_titles + [""] * n,  # titles only on top row
        horizontal_spacing=0.025,
        vertical_spacing=0.10,
        row_titles=["by class", "by center (domain shift)"],
    )

    _centers_seen = {}  # label -> color, accumulated for the legend
    for _i, ((_stem, _label, _bundle), _coords) in enumerate(zip(loaded, coords_list)):
        _col = _i + 1
        make_class_panel(fig, 1, _col, _coords, _bundle, style)
        _uniq, _color_of = make_center_panel(fig, 2, _col, _coords, _bundle, style)
        for _lab in _uniq:
            _centers_seen.setdefault(_lab, _color_of[_lab])

    # ── Manual legends (placed to the right) ──────────────────────────────
    class_names = loaded[0][2]["class_names"]
    # Class legend entries (row-1 semantics): class color × split marker
    for _split in ("train_all", "evc_test"):
        _is_train = _split == "train_all"
        _symbol = "circle" if _is_train else "diamond"
        _alpha = TRAIN_ALPHA if _is_train else TEST_ALPHA
        for _cls_idx, _cls_name in enumerate(class_names):
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="markers",
                    name=f"{_cls_name} ({_split})",
                    marker=dict(
                        size=12, symbol=_symbol,
                        color=hex_to_rgba(CLASS_COLORS[_cls_idx % len(CLASS_COLORS)], _alpha),
                        line=dict(width=1.0 if not _is_train else 0.3,
                                  color="#222222" if not _is_train else "rgba(255,255,255,0.6)"),
                    ),
                    legendgroup="class", legendgrouptitle_text="Row A — class",
                    showlegend=True,
                ),
                row=1, col=1,
            )
    # Center legend entries (row-2 semantics)
    for _lab, _color in _centers_seen.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                name=str(_lab),
                marker=dict(size=12, symbol="circle",
                            color=hex_to_rgba(_color, 0.9),
                            line=dict(width=0.4, color="#222222")),
                legendgroup="center", legendgrouptitle_text="Row B — center",
                showlegend=True,
            ),
            row=2, col=1,
        )
    # Colorbar for the probability heatmap (shown once) — only when the
    # heatmap itself is shown, otherwise it advertises a scale with no surface.
    if SHOW_PROB_HEATMAP:
        fig.add_trace(
            go.Heatmap(
                x=[None], y=[None], z=[[0, 1]], zmin=0.0, zmax=1.0,
                colorscale=PROB_COLORSCALE, showscale=True,
                colorbar=dict(
                    title=dict(text="p(class 1)", side="right"),
                    thickness=12, len=0.4, x=1.005, y=0.80,
                    tickvals=[0, 0.5, 1.0], ticktext=["0", "0.5", "1"],
                ),
                visible=False,
            ),
            row=1, col=1,
        )

    # ── IEEE-style layout: clean, framed, gridless, compact ───────────────

    _panel_w = 250

    fig.update_layout(
        height=550,            # Keep a fixed height so it doesn't get vertically squished
        autosize=True,         # <--- THIS IS THE MAGIC COMMAND. It forces it to fit your window!

        # Generous margins so nothing ever gets cut off
        margin=dict(l=80, r=220, t=46, b=80),

        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Helvetica, Arial, sans-serif", size=12, color="#111111"),

        legend=dict(
            x=1.02, y=1.0,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#cccccc",
            borderwidth=1,
            font=dict(size=13),
            grouptitlefont=dict(size=15),
            tracegroupgap=16,
            itemsizing="constant"
        ),
    )


    # Thin black frame around each panel, no gridlines, no ticks (IEEE figure).
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False,
                     showline=True, linewidth=1, linecolor="#333333",
                     mirror=True, ticks="", title_text="")
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False,
                     showline=True, linewidth=1, linecolor="#333333",
                     mirror=True, ticks="", title_text="")
    # Subplot + row titles slightly smaller for print.
    for _ann in fig.layout.annotations:
        _ann.font.size = 12

    pieces = [mo.md(f"## {title}"), mo.md(subtitle)]

    # Global X-Axis Title (Bottom Center)
    fig.add_annotation(
        x=0.5, y=-0.05,
        xref="paper", yref="paper",
        text="Principal Component 1",
        showarrow=False,
        font=dict(size=16, color="#111111")  # INCREASED font size
    )

    # Global Y-Axis Title (Middle Left, Rotated)
    fig.add_annotation(
        x=-0.03, y=0.5,         # Adjusted leftward positioning
        xref="paper", yref="paper",
        text="Principal Component 2",
        textangle=-90,
        showarrow=False,
        font=dict(size=16, color="#111111")  # INCREASED font size
    )

    if missing:
        pieces.append(mo.md(
            f"<sub>⚠️ skipped (not found / empty): `{', '.join(missing)}`</sub>"))
    # If centers never resolved, say so plainly under the figure.
    if set(_centers_seen.keys()) <= {"unknown", TEST_CENTER_LABEL}:
        pieces.append(mo.md(
            "<sub>ℹ️ *Center labels could not be resolved from the feature "
            "folders, so the bottom row is not informative. Save a "
            "`centers.npy` per split folder, add a `centers` field to "
            "`meta.json`, or encode the center in the image paths.*</sub>"))
    pieces.append(fig)
    return mo.vstack(pieces)


@app.cell
def _style_bundle(
    CLASS_COLORS, CENTER_COLORS, PROB_COLORSCALE, TRAIN_ALPHA, TEST_ALPHA,
    HEATMAP_ALPHA, TEST_IS_OWN_CENTER, TEST_CENTER_LABEL, SHOW_PROB_HEATMAP,
):
    # Bundle style constants into a single tuple passed through to renderers.
    style = (
        CLASS_COLORS, CENTER_COLORS, PROB_COLORSCALE, TRAIN_ALPHA, TEST_ALPHA,
        HEATMAP_ALPHA, TEST_IS_OWN_CENTER, TEST_CENTER_LABEL, SHOW_PROB_HEATMAP,
    )
    return (style,)


@app.cell
def _p0(experiment_groups, features_root_ui, style):
    _view = render_group(features_root_ui.value, experiment_groups[0], style)
    _view
    return


@app.cell
def _p1(experiment_groups, features_root_ui, style):
    _view = render_group(features_root_ui.value, experiment_groups[1], style)
    _view
    return


@app.cell
def _p1_v_p0_callout(mo):
    mo.md("""
    > **P0 vs P1 — the only change between these two rows is the loss function.**
    > P0 uses standard cross-entropy; P1 uses the SupPro (supervised contrastive)
    > loss. Any difference in cluster geometry between the two is attributable
    > to the loss change alone.
    """)
    return


@app.cell
def _p2(experiment_groups, features_root_ui, mo, style):
    _view = mo.vstack([
        mo.md("---"),
        render_group(features_root_ui.value, experiment_groups[2], style),
    ])
    _view
    return


@app.cell
def _p3_bb(experiment_groups, features_root_ui, mo, style):
    _view = mo.vstack([
        mo.md("---"),
        render_group(features_root_ui.value, experiment_groups[3], style),
    ])
    _view
    return


@app.cell
def _p3_proj(experiment_groups, features_root_ui, style):
    _view = render_group(features_root_ui.value, experiment_groups[4], style)
    _view
    return


@app.cell
def _p3_cls(experiment_groups, features_root_ui, style):
    _view = render_group(features_root_ui.value, experiment_groups[5], style)
    _view
    return


@app.cell
def _p4(experiment_groups, features_root_ui, mo, style):
    _view = mo.vstack([
        mo.md("---"),
        render_group(features_root_ui.value, experiment_groups[6], style),
    ])
    _view
    return


@app.cell
def _p5(experiment_groups, features_root_ui, mo, style):
    _view = mo.vstack([
        mo.md("---"),
        render_group(features_root_ui.value, experiment_groups[7], style),
    ])
    _view
    return


@app.cell
def _p6(experiment_groups, features_root_ui, mo, style):
    _view = mo.vstack([
        mo.md("---"),
        render_group(features_root_ui.value, experiment_groups[8], style),
    ])
    _view
    return


@app.cell
def _p7(experiment_groups, features_root_ui, mo, style):
    _view = mo.vstack([
        mo.md("---"),
        render_group(features_root_ui.value, experiment_groups[9], style),
    ])
    _view
    return


@app.cell
def _p8_scale04(experiment_groups, features_root_ui, mo, style):
    _view = mo.vstack([
        mo.md("---"),
        render_group(features_root_ui.value, experiment_groups[10], style),
    ])
    _view
    return


@app.cell
def _p8_scale06(experiment_groups, features_root_ui, style):
    _view = render_group(features_root_ui.value, experiment_groups[11], style)
    _view
    return


@app.cell
def _p8_scale08(experiment_groups, features_root_ui, style):
    _view = render_group(features_root_ui.value, experiment_groups[12], style)
    _view
    return


@app.cell
def _p8_scale095(experiment_groups, features_root_ui, style):
    _view = render_group(features_root_ui.value, experiment_groups[13], style)
    _view
    return


if __name__ == "__main__":
    app.run()