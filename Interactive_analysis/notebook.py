"""
Stage 4: interactive UMAP/PCA notebook (marimo).

Run with:
    marimo run notebook.py

Reads from features_out/<checkpoint_name>/ produced by extract_features.py.
Select MULTIPLE feature folders to merge them on the same plot.

Box-select points in the scatter plot to preview their source images and
recompute statistics on the selected subset.
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

# ── Numba / UMAP threading safety ─────────────────────────────────────────
# marimo runs cells reactively and can fit UMAP in more than one cell at once
# (e.g. the main projection and the two side-by-side comparison panels). UMAP
# uses Numba; Numba's default "workqueue" parallel layer is NOT threadsafe and
# aborts the whole process on concurrent access ("Concurrent access has been
# detected"). We avoid this two ways:
#   1. Pin a threadsafe layer (TBB if available, else OpenMP), verified by
#      actually running a parallel function so an unavailable layer can't slip
#      through. This is the primary fix.
#   2. Force single-threaded Numba. Free here because UMAP already runs
#      single-threaded whenever random_state is set (hence the "n_jobs
#      overridden to 1" warning), so we lose no speed and remove the hazard.
# A lock inside fit_umap_safe (a FUNCTION ATTRIBUTE, not a module global —
# marimo mangles module-level underscore names so @app.function helpers can't
# see them) additionally serializes fits as a belt-and-suspenders guard.
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numba


def _numba_layer_works(layer):
    """Set THREADING_LAYER=layer and verify it actually LOADS at runtime by
    executing a tiny parallel-region function. Just assigning the config string
    is not enough — an unavailable layer (e.g. TBB not installed) only errors
    when a parallel function first runs, which is exactly the crash we're
    trying to pre-empt. Returns True iff the layer ran successfully."""
    try:
        numba.config.THREADING_LAYER = layer
        import numpy as _np

        @numba.njit(parallel=True, cache=False)
        def _probe(a):
            s = 0.0
            for _i in numba.prange(a.shape[0]):
                s += a[_i]
            return s

        _probe(_np.ones(8, dtype=_np.float64))
        return True
    except Exception:
        return False


# Prefer a threadsafe layer (tbb, then omp). Fall back to workqueue only if
# neither threadsafe layer loads.
_chosen_layer = None
for _layer in ("tbb", "omp"):
    if _numba_layer_works(_layer):
        _chosen_layer = _layer
        break
if _chosen_layer is None:
    try:
        numba.config.THREADING_LAYER = "workqueue"
    except Exception:
        pass
    _chosen_layer = "workqueue"


@app.function
def fit_umap_safe(X, n_neighbors, min_dist, metric, seed, n_components=2):
    """Fit UMAP, serialized through a lock so concurrent cell execution can
    never trigger Numba's non-threadsafe parallel region. Public name so any
    cell can call it. The lock lives as an ATTRIBUTE OF THIS FUNCTION rather
    than a module global, because marimo mangles module-level underscore names
    (`_UMAP_LOCK` → `_cell_<hash>_UMAP_LOCK`) which an @app.function cannot
    resolve. Returns the 2-D (or n_components-D) embedding."""
    import threading
    import warnings
    import umap

    # Lazily create the lock once, attached to the function object itself.
    _lock = getattr(fit_umap_safe, "_lock", None)
    if _lock is None:
        _lock = threading.Lock()
        fit_umap_safe._lock = _lock

    _n = max(2, int(min(n_neighbors, max(2, X.shape[0] - 1))))
    with _lock:
        with warnings.catch_warnings():
            # benign: UMAP forces single-threaded when random_state is set.
            warnings.filterwarnings(
                "ignore", message=".*n_jobs value.*overridden.*"
            )
            reducer = umap.UMAP(
                n_neighbors=_n,
                min_dist=float(min_dist),
                metric=metric,
                n_components=int(n_components),
                random_state=int(seed),
            )
            return reducer.fit_transform(X)


@app.cell
def _imports():
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import umap
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    return (
        LogisticRegression,
        PCA,
        Path,
        StandardScaler,
        StratifiedKFold,
        accuracy_score,
        balanced_accuracy_score,
        cross_validate,
        go,
        json,
        mo,
        np,
        roc_auc_score,
        umap,
    )


@app.function
def cmp_load_stem(features_root, dirs, feature_kind):
    """Load one experiment stem's merged train+test arrays for the comparison
    view. `dirs` is the list of folders for this stem (from stem_to_dirs).
    `feature_kind` is 'pooled' or 'projection'. Returns a dict bundle or None.
    Public name so the comparison cell can call it.
    """
    import json
    import numpy as np

    feature_file = (
        "features_pooled.npy" if feature_kind == "pooled" else "features_proj.npy"
    )
    feats_all, labels_all, probs_all, tags_all, paths_all = [], [], [], [], []
    class_names = None
    for d in dirs:
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if class_names is None:
            class_names = meta["class_names"]
        feats = np.load(d / feature_file)
        lbls = np.load(d / "labels.npy")
        pts = np.load(d / "paths.npy", allow_pickle=True)
        probs_raw = np.load(d / "deployed_probs.npy")
        probs = probs_raw[:, 1] if probs_raw.ndim == 2 else probs_raw
        _tag = meta.get("dataset_tag", d.name.split("__")[-1])
        feats_all.append(feats)
        labels_all.append(lbls)
        probs_all.append(probs)
        paths_all.extend(pts)
        tags_all.extend([_tag] * len(lbls))
    if not feats_all:
        return None
    return {
        "features":       np.concatenate(feats_all, axis=0),
        "labels":         np.concatenate(labels_all, axis=0),
        "deployed_probs": np.concatenate(probs_all, axis=0),
        "tags":           np.array(tags_all),
        "paths":          np.asarray(paths_all, dtype=object),
        "class_names":    class_names,
    }


@app.function
def cmp_build_scatter(bundle, title, proj_method, seed,
                      n_neighbors, min_dist, metric):
    """Build a single Plotly scatter for ONE checkpoint bundle, with an
    INDEPENDENT 2D projection fit on that checkpoint's own features. Returns
    (figure, metrics_dict). Public name so the comparison cell can call it.
    """
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score

    X = bundle["features"]
    y = bundle["labels"]
    p = bundle["deployed_probs"]
    t = bundle["tags"]
    class_names = bundle["class_names"]

    # ── Independent projection (each checkpoint fit on its own data) ──────
    if proj_method == "PCA" or X.shape[0] < 4:
        coords = PCA(n_components=2, random_state=int(seed)).fit_transform(X)
        proj_name = "PCA"
    else:
        try:
            # Locked helper: serializes UMAP fits so the two comparison panels
            # (and the main projection) can't run a Numba parallel region at
            # the same time.
            coords = fit_umap_safe(X, n_neighbors, min_dist, metric, seed)
            proj_name = "UMAP"
        except Exception:
            # umap not installed or failed → fall back to PCA so the panel
            # still renders.
            coords = PCA(n_components=2, random_state=int(seed)).fit_transform(X)
            proj_name = "PCA (UMAP unavailable)"

    _CLASS_COLORS = ["#009E73", "#D55E00", "#0072B2", "#E69F00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
    
    def hex_to_rgba(hex_color, alpha=1.0):
        h = hex_color.lstrip("#")
        if len(h) == 6:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"
        return hex_color

    fig = go.Figure()
    _splits = sorted(set(t.tolist()), key=lambda x: ("train" not in x.lower(), x))
    for _split in _splits:
        _is_train = "train" in _split.lower()
        _sym = "circle" if _is_train else "diamond"
        _alpha = 0.35 if _is_train else 0.95
        _lw = 0.4 if _is_train else 1.1
        _lc = "rgba(255,255,255,0.6)" if _is_train else "#222222"
        _sz = 6 if _is_train else 8

        for _ci, _cn in enumerate(class_names):
            _m = (y == _ci) & (t == _split)
            if not _m.any():
                continue
            
            _color_rgba = hex_to_rgba(_CLASS_COLORS[_ci % len(_CLASS_COLORS)], _alpha)
            fig.add_trace(go.Scatter(
                x=coords[_m, 0], y=coords[_m, 1],
                mode="markers",
                name=f"{_cn} ({_split})",
                marker=dict(
                    size=_sz,
                    symbol=_sym,
                    color=_color_rgba,
                    line=dict(width=_lw, color=_lc),
                ),
                hovertemplate=(
                    f"{_cn} / {_split}<br>"
                    "p(class 1)=%{customdata:.3f}<extra></extra>"
                ),
                customdata=p[_m],
            ))

    fig.update_layout(
        title=f"{title}  ·  {proj_name}",
        width=460, height=460,
        margin=dict(l=30, r=10, t=50, b=30),
        plot_bgcolor="white",
        legend=dict(font=dict(size=9), bgcolor="rgba(255,255,255,0.6)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False,
                     showticklabels=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False,
                     showticklabels=False)

    # ── Threshold-free metric: deployed ROC-AUC over the merged data ──────
    if len(np.unique(y)) >= 2:
        try:
            _auc = float(roc_auc_score(y, p))
        except ValueError:
            _auc = float("nan")
    else:
        _auc = float("nan")
    metrics = {
        "checkpoint": title,
        "n_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "deployed_auc": _auc,
    }
    return fig, metrics


@app.function
def representation_metrics(X, y, cac_top_r=0.10, gpu_t=2.0, eps=1e-12):
    """Representation-quality metrics following Mildenberger et al. (2025).

    In addition to standard within-class cluster-distance metrics, we report a
    cross-class consistency metric that catches representation collapse the
    within-class metrics miss. All distances are computed on **L2-normalized**
    features, so they live on the unit sphere (matching the paper's setup).

    Inputs
    ------
    X : (n, d) float array of features (raw; normalized internally).
    y : (n,)  int label array.
    cac_top_r : fraction r for the CAC top-r% nearest-neighbor neighborhood.
    gpu_t : scale t in the Gaussian-potential kernel exp(-t * d^2).

    Returns a dict with:
      n, n_per_class (dict),
      SAD : Sample Alignment Distance  (lower better) — mean L2 distance from
            each sample to its nearest SAME-class neighbor. Within-class
            tightness.
      CAD : Class Alignment Distance   (lower better) — mean pairwise within-
            class L2 distance, averaged across classes.
      CAC : Class Alignment Consistency (higher better, max 1.0) — for each
            sample, the fraction of its top-r% nearest neighbors (any class)
            that share its class; averaged over all samples. The single most
            useful number — captures whether local neighborhoods are class-pure.
      GPU : Gaussian Potential Uniformity (lower better) — log of the mean
            Gaussian potential exp(-t*d^2) over all pairs. A too-low value
            combined with a bad CAC means features are uniform but class-mixed.

    Any metric that is undefined for the given data (e.g. CAD needs ≥2 samples
    in a class; CAC/SAD need ≥2 samples total) is returned as NaN.
    """
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n = X.shape[0]
    nan = float("nan")
    if n < 2:
        return {
            "n": int(n), "n_per_class": {},
            "SAD": nan, "CAD": nan, "CAC": nan, "GPU": nan,
        }

    # L2-normalize onto the unit sphere (paper's setup).
    _norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(_norms, eps)

    # Full pairwise Euclidean distance matrix on the sphere.
    # (n is a few thousand here, so the dense matrix is fine.)
    _gram = Xn @ Xn.T
    _sq = np.maximum(0.0, 2.0 - 2.0 * _gram)
    D = np.sqrt(_sq, dtype=np.float64)
    np.fill_diagonal(D, np.inf)  # exclude self for nearest-neighbor queries

    classes = np.unique(y)
    n_per_class = {int(c): int((y == c).sum()) for c in classes}

    # ── SAD: nearest SAME-class neighbor distance, averaged over samples ──
    _sad_vals = []
    for _i in range(n):
        _same = (y == y[_i])
        _same[_i] = False
        if _same.any():
            _sad_vals.append(D[_i, _same].min())
    SAD = float(np.mean(_sad_vals)) if _sad_vals else nan

    # ── CAD: mean within-class pairwise distance, averaged across classes ─
    _cad_per_class = []
    for _c in classes:
        _idx = np.where(y == _c)[0]
        if _idx.size >= 2:
            _sub = D[np.ix_(_idx, _idx)]
            _finite = _sub[np.isfinite(_sub)]
            if _finite.size:
                _cad_per_class.append(float(_finite.mean()))
    CAD = float(np.mean(_cad_per_class)) if _cad_per_class else nan

    # ── CAC: fraction of top-r% nearest neighbors sharing the class ───────
    _k = max(1, int(round(cac_top_r * (n - 1))))
    # argpartition for the k smallest distances per row (excludes self via inf)
    _nn = np.argpartition(D, _k - 1, axis=1)[:, :_k]
    _same_frac = (y[_nn] == y[:, None]).mean(axis=1)
    CAC = float(_same_frac.mean())

    # ── GPU: log mean Gaussian potential over all unordered pairs ─────────
    _iu = np.triu_indices(n, k=1)
    _pair_d2 = _sq[_iu]
    _pot = np.exp(-gpu_t * _pair_d2)
    GPU = float(np.log(_pot.mean() + eps))

    return {
        "n": int(n), "n_per_class": n_per_class,
        "SAD": SAD, "CAD": CAD, "CAC": CAC, "GPU": GPU,
    }


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
    #      train: .../data/center_1/...  .../data/center_2/...  (any center_N)
    #      test:  .../EVC_Barretts_FullSet .../images/...  -> the EVC test center
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


@app.cell
def _intro(mo):
    mo.md("""
    # SupCon-style feature evaluation

    Visualizes the learned representation space for one or more checkpoints.
    Select an experiment below to plot its train and test splits in the same 
    representation space and see how well the test data aligns with the training clusters.

    **Box-select** points in the scatter plot (toolbar → box-select tool, active by
    default) to preview their source images and recompute statistics on the subset.
    """)
    return


@app.cell
def _features_root(mo):
    features_root = mo.ui.text(
        value="features_out",
        label="Features root directory",
        full_width=True,
    )
    features_root
    return (features_root,)


@app.cell
def _discover_checkpoints(Path, features_root, mo):
    root = Path(features_root.value).expanduser()
    if not root.exists():
        mo.stop(True, mo.md(f"⚠️ `{root}` does not exist."))

    if (root / "meta.json").exists():
        checkpoint_dirs = [root]
    else:
        checkpoint_dirs = sorted(
            p for p in root.iterdir()
            if p.is_dir() and (p / "meta.json").exists()
        )
    mo.stop(
        not checkpoint_dirs,
        mo.md(f"⚠️ No checkpoint folders with `meta.json` under `{root}`."),
    )

    # ── Group folders by experiment stem ─────────────────────────────────
    # Folder names look like:  <STEM>__<split>  where split ∈ {evc_test, train_all}
    # We expose stems to the user; selecting a stem implicitly loads BOTH splits.
    # Folders that don't follow the convention fall back to using their full
    # name as the stem (so legacy folders still work).
    _stem_to_dirs = {}
    _known_splits = ("evc_test", "train_all")
    for _d in checkpoint_dirs:
        _stem = _d.name
        for _split in _known_splits:
            _suffix = f"__{_split}"
            if _d.name.endswith(_suffix):
                _stem = _d.name[: -len(_suffix)]
                break
        _stem_to_dirs.setdefault(_stem, []).append(_d)

    experiment_stems = sorted(_stem_to_dirs.keys())
    stem_to_dirs = {k: _stem_to_dirs[k] for k in experiment_stems}
    return experiment_stems, stem_to_dirs


@app.cell
def _config_controls(experiment_stems, mo):
    # Changed from multiselect to a single-select dropdown
    experiment_picker = mo.ui.dropdown(
        options=experiment_stems,
        value=experiment_stems[0],
        label="Select experiment (both train_all + evc_test are loaded)",
    )
    feature_picker = mo.ui.dropdown(
        options=["pooled (768-D, paper's r)", "projection (128-D, paper's z)"],
        value="pooled (768-D, paper's r)",
        label="Features",
    )
    projection_picker = mo.ui.dropdown(
        options=["UMAP", "PCA"],
        value="UMAP",
        label="Projection Method",
    )
    n_neighbors = mo.ui.slider(
        start=5, stop=100, step=1, value=15, label="UMAP n_neighbors",
    )
    min_dist = mo.ui.slider(
        start=0.0, stop=0.99, step=0.01, value=0.1, label="UMAP min_dist",
    )
    metric = mo.ui.dropdown(
        options=["euclidean", "cosine"],
        value="cosine",
        label="UMAP metric",
    )
    seed = mo.ui.number(value=42, label="Random seed", start=0, stop=10_000)

    controls = mo.vstack([
        mo.hstack([experiment_picker]),
        mo.hstack([projection_picker, feature_picker]),
        mo.hstack([metric, n_neighbors, min_dist]),
        mo.hstack([seed]),
    ])
    controls
    return (
        experiment_picker,
        feature_picker,
        metric,
        min_dist,
        n_neighbors,
        projection_picker,
        seed,
    )


@app.cell
def _threshold_control(mo):
    threshold = mo.ui.slider(
        start=0.0, stop=1.0, step=0.01, value=0.5,
        label="Confusion-matrix threshold (chosen, not deployed)",
        show_value=True,
    )
    return (threshold,)


@app.cell
def _load_data(experiment_picker, feature_picker, json, mo, np, stem_to_dirs):
    mo.stop(not experiment_picker.value, mo.md("⚠️ Select an experiment."))

    # Get the underlying folders for the chosen experiment stem
    _selected_dirs = stem_to_dirs.get(experiment_picker.value, [])

    mo.stop(
        not _selected_dirs,
        mo.md("⚠️ No folders matched the picked experiment."),
    )

    feature_kind = "pooled" if feature_picker.value.startswith("pooled") else "projection"
    feature_file = "features_pooled.npy" if feature_kind == "pooled" else "features_proj.npy"

    all_features = []
    all_labels = []
    all_paths = []
    all_probs = []
    all_tags = []
    all_centers = []

    class_names = None

    for s_dir in _selected_dirs:
        meta = json.loads((s_dir / "meta.json").read_text())
        if class_names is None:
            class_names = meta["class_names"]

        feats = np.load(s_dir / feature_file)
        lbls = np.load(s_dir / "labels.npy")
        pts = np.load(s_dir / "paths.npy", allow_pickle=True)
        probs_raw = np.load(s_dir / "deployed_probs.npy")

        # Prefer dataset_tag from meta; fall back to suffix after "__"
        _tag = meta.get("dataset_tag", s_dir.name.split("__")[-1])

        # Acquisition-center label per row (for the domain-shift view).
        # The split component is the suffix after "__" (evc_test / train_all).
        _split_name = s_dir.name.split("__")[-1]
        _centers = resolve_centers(s_dir, pts, len(lbls), _split_name)

        all_features.append(feats)
        all_labels.append(lbls)
        all_paths.extend(pts)
        all_centers.append(_centers)

        if probs_raw.ndim == 2:
            all_probs.append(probs_raw[:, 1])
        else:
            all_probs.append(probs_raw)

        all_tags.extend([_tag] * len(lbls))

    # --- Gracefully catch dimensionality mismatches just in case ---
    if all_features:
        _expected_dim = all_features[0].shape[1]
        for _i, _f in enumerate(all_features):
            if _f.shape[1] != _expected_dim:
                mo.stop(
                    True,
                    mo.md(
                        f"⚠️ **Dimensionality Mismatch!**\n\n"
                        f"The data from `{_selected_dirs[_i].name}` has **{_f.shape[1]}-D** features, "
                        f"but earlier splits have **{_expected_dim}-D** features."
                    )
                )

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    paths = np.asarray(all_paths, dtype=object)
    deployed_probs = np.concatenate(all_probs, axis=0)
    tags = np.array(all_tags)
    centers = np.concatenate(all_centers, axis=0)

    summary_md = mo.md(
        f"""
        **Loaded experiment:** {experiment_picker.value} <br>
        ({len(_selected_dirs)} folder(s)) <br>
        **Features:** {feature_kind} ({features.shape[1]}-D) &nbsp;&nbsp;
        **Total Samples:** {len(labels)}
        """
    )
    summary_md
    return centers, class_names, deployed_probs, features, labels, paths, tags


@app.cell
def _project_2d(
    PCA,
    features,
    metric,
    min_dist,
    mo,
    n_neighbors,
    projection_picker,
    seed,
):
    import hashlib

    def _array_key(arr):
        return hashlib.md5(arr.tobytes()).hexdigest()

    @mo.cache
    def _fit_umap(features_key, features, n_neighbors, min_dist, metric, seed):
        # Route through the locked helper so this fit can't overlap with the
        # comparison-panel fits (avoids the Numba workqueue concurrency abort).
        return fit_umap_safe(features, n_neighbors, min_dist, metric, seed)

    @mo.cache
    def _fit_pca(features_key, features, seed):
        reducer = PCA(n_components=2, random_state=int(seed))
        return reducer.fit_transform(features)

    _key = _array_key(features)

    if projection_picker.value == "PCA":
        coords_2d = _fit_pca(_key, features, seed.value)
        proj_name = "PCA"
    else:
        coords_2d = _fit_umap(
            _key, features, n_neighbors.value, min_dist.value, metric.value, seed.value,
        )
        proj_name = "UMAP"
    return coords_2d, proj_name


@app.cell
def _scatter_plot(
    class_names,
    coords_2d,
    deployed_probs,
    go,
    labels,
    mo,
    np,
    paths,
    proj_name,
    tags,
):
    def hex_to_rgba(hex_color, alpha=1.0):
        h = hex_color.lstrip("#")
        if len(h) == 6:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"
        return hex_color

    _CLASS_COLORS = ["#009E73", "#D55E00", "#0072B2", "#E69F00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

    _fig = go.Figure()

    _unique_tags = np.unique(tags)
    _splits = sorted(_unique_tags, key=lambda x: ("train" not in x.lower(), x))
    
    for _t in _splits:
        _is_train = "train" in _t.lower()
        _marker_symbol = "circle" if _is_train else "diamond"
        _alpha = 0.35 if _is_train else 0.95
        _lw = 0.3 if _is_train else 1.1
        _lc = "rgba(255,255,255,0.6)" if _is_train else "#222222"
        _sz = 6 if _is_train else 8

        for _cls_idx, _cls_name in enumerate(class_names):
            _mask = (labels == _cls_idx) & (tags == _t)
            if not np.any(_mask):
                continue

            _cls_idxs = np.where(_mask)[0]
            _hover_text = [
                f"Dataset: {_t}<br>Path: {str(paths[i])}<br>"
                f"label={_cls_name}<br>p(class 1) deployed={float(deployed_probs[i]):.3f}"
                for i in _cls_idxs
            ]

            _color_rgba = hex_to_rgba(_CLASS_COLORS[_cls_idx % len(_CLASS_COLORS)], _alpha)

            _fig.add_trace(go.Scatter(
                x=coords_2d[_mask, 0],
                y=coords_2d[_mask, 1],
                mode="markers",
                name=f"{_cls_name} ({_t})",
                marker=dict(
                    size=_sz,
                    symbol=_marker_symbol,
                    color=_color_rgba,
                    line=dict(width=_lw, color=_lc),
                ),
                text=_hover_text,
                hovertemplate="%{text}<extra></extra>",
            ))

    _fig.update_layout(
        width=500, height=500,
        xaxis_title=f"{proj_name} 1",
        yaxis_title=f"{proj_name} 2",
        legend=dict(x=1.02, y=1, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=40, r=120, t=20, b=40),
        plot_bgcolor="white",
        dragmode="select",  # box-select tool active by default
    )
    _fig.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    _fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False)

    scatter_plot = mo.ui.plotly(_fig)
    return (scatter_plot,)


@app.cell
def _center_scatter(
    centers,
    coords_2d,
    go,
    mo,
    np,
    proj_name,
    tags,
):
    # ── Domain-shift view ────────────────────────────────────────────────
    # SAME projection coordinates as the class-colored scatter above, but
    # recolored by acquisition CENTER (hospital / site). Systematic separation
    # between training centers and the test center indicates domain shift.
    def _hex_to_rgba(hex_color, alpha=1.0):
        h = hex_color.lstrip("#")
        if len(h) == 6:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"
        return hex_color

    _CENTER_COLORS = [
        "#0072B2", "#E69F00", "#CC79A7", "#56B4E9", "#F0E442",
        "#009E73", "#D55E00", "#000000",
    ]
    _TEST_CENTER_LABEL = "EVC (test center)"

    # Relabel the test split as its own distinct center (most common setup:
    # train centers vs a held-out test center).
    _centers = centers.astype(object).copy()
    _is_test = np.array(["train" not in str(t).lower() for t in tags])
    _centers[_is_test] = _TEST_CENTER_LABEL

    # Stable color assignment; keep the test center last for a consistent color.
    _uniq = sorted(set(_centers.tolist()))
    if _TEST_CENTER_LABEL in _uniq:
        _uniq.remove(_TEST_CENTER_LABEL)
        _uniq = _uniq + [_TEST_CENTER_LABEL]
    _color_of = {lab: _CENTER_COLORS[i % len(_CENTER_COLORS)]
                 for i, lab in enumerate(_uniq)}

    _fig_c = go.Figure()
    # train first (under), then test on top
    for _is_train in (True, False):
        _alpha = 0.35 if _is_train else 0.95
        _sym = "circle" if _is_train else "diamond"
        _lw = 0.3 if _is_train else 1.1
        _lc = "rgba(255,255,255,0.6)" if _is_train else "#222222"
        _sz = 6 if _is_train else 8
        _split_mask = (~_is_test) if _is_train else _is_test
        for _lab in _uniq:
            _mask = (_centers == _lab) & _split_mask
            if not np.any(_mask):
                continue
            _split_word = "train" if _is_train else "test"
            _fig_c.add_trace(go.Scatter(
                x=coords_2d[_mask, 0], y=coords_2d[_mask, 1],
                mode="markers",
                name=f"{_lab} ({_split_word})",
                marker=dict(
                    size=_sz, symbol=_sym,
                    color=_hex_to_rgba(_color_of[_lab], _alpha),
                    line=dict(width=_lw, color=_lc),
                ),
                hovertemplate=f"{_lab} / {_split_word}<extra></extra>",
            ))

    _fig_c.update_layout(
        width=500, height=500,
        xaxis_title=f"{proj_name} 1",
        yaxis_title=f"{proj_name} 2",
        legend=dict(x=1.02, y=1, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=40, r=120, t=20, b=40),
        plot_bgcolor="white",
    )
    _fig_c.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    _fig_c.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False)

    # If centers never resolved, say so plainly instead of a useless one-color plot.
    _resolved = set(_uniq) - {"unknown", _TEST_CENTER_LABEL}
    if _resolved:
        center_plot = mo.as_html(_fig_c)
    else:
        center_plot = mo.md(
            "ℹ️ *Center labels could not be resolved for this experiment. Save a "
            "`centers.npy` per split folder, add a `centers` field to "
            "`meta.json`, or encode the center in the image paths "
            "(e.g. `.../center_1/...`) to enable this view.*"
        )
    return (center_plot,)


@app.cell
def _plots_layout(center_plot, mo, scatter_plot):
    plots_layout = mo.hstack(
        [
            mo.vstack([
                mo.md("### Main Projection\n*Box-select to preview source images.*"),
                scatter_plot
            ]),
            mo.vstack([
                mo.md("### By acquisition center (domain shift)\n*◆ = test (drawn on top), ● = train (translucent).*"),
                center_plot
            ])
        ],
        wrap=True,
        gap=2.0
    )
    plots_layout
    return (plots_layout,)


@app.cell
def _selection_mask(coords_2d, np, scatter_plot):
    # scatter_plot.ranges is {"x": [xmin, xmax], "y": [ymin, ymax]} when a box
    # selection exists, else an empty dict. We compute the mask in Python from
    # the 2D coords because .indices is unreliable across multi-trace figures.
    _ranges = scatter_plot.ranges or {}
    if "x" in _ranges and "y" in _ranges:
        _xmin, _xmax = _ranges["x"]
        _ymin, _ymax = _ranges["y"]
        selection_mask = (
            (coords_2d[:, 0] >= _xmin) & (coords_2d[:, 0] <= _xmax) &
            (coords_2d[:, 1] >= _ymin) & (coords_2d[:, 1] <= _ymax)
        )
    else:
        selection_mask = np.zeros(len(coords_2d), dtype=bool)
    return (selection_mask,)


@app.cell
def _repr_metrics_intro(mo):
    mo.md(
        """
        ## Representation quality metrics

        Following **Mildenberger et al. (2025)**, in addition to standard
        cluster-distance metrics, we report a metric that compares embeddings
        *across* classes — it catches representation collapse that within-class
        metrics miss.

        - **SAD** (Sample Alignment Distance, lower = better): mean L2 distance
          between an image and its nearest same-class neighbor. Within-class
          tightness.
        - **CAD** (Class Alignment Distance, lower = better): mean within-class
          L2 distance averaged across classes.
        - **CAC** (Class Alignment Consistency, higher = better, max 1.0): for
          each sample, what fraction of its top-r% nearest neighbors share its
          class. The most useful single number — captures whether local
          neighborhoods are class-pure.
        - **GPU** (Gaussian Potential Uniformity, lower = better, but remember a
          too-low value with bad CAC means features are uniform but class-mixed).

        Computed on **L2-normalized** features so distances are on the unit
        sphere (matching the paper's setup).
        """
    )
    return


@app.cell
def _repr_metrics_table(
    class_names, features, labels, mo, np, tags,
):
    import pandas as _pd

    # One row per dataset split, plus an ALL row. n0/n1 are per-class counts.
    _rows = []

    def _metrics_row(_group_name, _Xg, _yg):
        _m = representation_metrics(_Xg, _yg)
        _npc = _m["n_per_class"]
        return {
            "group": _group_name,
            "n":  _m["n"],
            "n0": int(_npc.get(0, 0)),
            "n1": int(_npc.get(1, 0)),
            "SAD (within, lower better)": _m["SAD"],
            "CAD (within, lower better)": _m["CAD"],
            "CAC (cross, higher better)": _m["CAC"],
            "GPU (uniformity, lower better)": _m["GPU"],
        }

    # ALL first
    _rows.append(_metrics_row("ALL", features, labels))
    # then each dataset tag
    for _t in sorted(set(tags.tolist())):
        _mask = tags == _t
        _rows.append(_metrics_row(str(_t), features[_mask], labels[_mask]))

    repr_metrics_df = _pd.DataFrame(_rows)

    repr_metrics_table = mo.vstack([
        mo.md("### Per-dataset representation metrics"),
        mo.ui.table(repr_metrics_df, selection=None),
        mo.md(
            "*Read these together. Low SAD/CAD with low CAC = collapsed "
            "(everything close, classes mixed). High CAC + reasonable GPU = "
            "clean separation with good spread. A CAC drop on the test set vs. "
            "the train set quantifies how much structure the model loses "
            "out-of-distribution.*"
        ),
    ])
    repr_metrics_table
    return


@app.cell
def _selection_stats(
    LogisticRegression,
    StandardScaler,
    StratifiedKFold,
    accuracy_score,
    balanced_accuracy_score,
    class_names,
    coords_2d,
    cross_validate,
    deployed_probs,
    features,
    labels,
    mo,
    np,
    roc_auc_score,
    selection_mask,
    tags,
    threshold,
):
    mo.stop(
        not selection_mask.any(),
        mo.md(
            "### Selection statistics\n"
            "*Box-select points in the scatter to see stats on the selected subset.*"
        ),
    )

    _sel = selection_mask
    _n = int(_sel.sum())
    _sel_labels = labels[_sel]
    _sel_probs = deployed_probs[_sel]
    _sel_tags = tags[_sel]

    # Class & dataset composition
    _class_counts = {
        class_names[int(c)]: int((_sel_labels == c).sum())
        for c in np.unique(_sel_labels)
    }
    _tag_counts = {str(t): int((_sel_tags == t).sum()) for t in np.unique(_sel_tags)}

    # Deployed-model performance on the subset, at the CHOSEN threshold
    # (slider) — the deployed model's real threshold is unknown. ROC-AUC below
    # is threshold-free.
    _thr = float(threshold.value)
    _preds = (_sel_probs >= _thr).astype(int)
    _unique_lbls = np.unique(_sel_labels)
    _acc = accuracy_score(_sel_labels, _preds) if len(_unique_lbls) >= 1 else float("nan")
    if len(_unique_lbls) >= 2:
        _bacc = balanced_accuracy_score(_sel_labels, _preds)
        try:
            _auc = roc_auc_score(_sel_labels, _sel_probs)
        except ValueError:
            _auc = float("nan")
    else:
        _bacc = float("nan")
        _auc = float("nan")

    # 5-fold linear probe on the FULL feature vectors of the subset.
    # Only meaningful when both classes are present and we have enough samples
    # per class to stratify into 5 folds.
    _min_class = (
        min((_sel_labels == c).sum() for c in _unique_lbls)
        if len(_unique_lbls) >= 2 else 0
    )
    if len(_unique_lbls) >= 2 and _min_class >= 5 and _n >= 20:
        _scaler = StandardScaler().fit(features[_sel])
        _X = _scaler.transform(features[_sel])
        _cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        _cv_results = cross_validate(
            LogisticRegression(class_weight="balanced", max_iter=2000),
            _X, _sel_labels, cv=_cv,
            scoring=["accuracy", "balanced_accuracy"],
        )
        _probe_acc = _cv_results["test_accuracy"].mean()
        _probe_acc_std = _cv_results["test_accuracy"].std()
        _probe_bacc = _cv_results["test_balanced_accuracy"].mean()
        _probe_bacc_std = _cv_results["test_balanced_accuracy"].std()
        _probe_line = (
            f"- **Linear probe on subset features** (5-fold CV): "
            f"acc = {_probe_acc:.3f} ± {_probe_acc_std:.3f}, "
            f"balanced acc = {_probe_bacc:.3f} ± {_probe_bacc_std:.3f}"
        )
    else:
        _probe_line = (
            f"- *Linear probe skipped:* need both classes with ≥5 samples each "
            f"and ≥20 total (have {_n} samples, "
            f"{len(_unique_lbls)} class(es), min-class={int(_min_class)}).*"
        )

    # Spatial extent of the selection box
    _xs = coords_2d[_sel, 0]
    _ys = coords_2d[_sel, 1]
    _extent = (
        f"x ∈ [{_xs.min():.2f}, {_xs.max():.2f}], "
        f"y ∈ [{_ys.min():.2f}, {_ys.max():.2f}]"
    )

    _class_str = ", ".join(f"{k}={v}" for k, v in _class_counts.items())
    _tag_str = ", ".join(f"{k}={v}" for k, v in _tag_counts.items())

    stats_md = mo.md(
        f"""
    ### Selection statistics

    - **Selected points:** {_n} &nbsp;&nbsp; **Extent:** {_extent}
    - **By class:** {_class_str}
    - **By dataset:** {_tag_str}
    - **Deployed model on subset @ thr={_thr:.2f}:** acc = {_acc:.3f}, balanced acc = {_bacc:.3f}, ROC-AUC = {_auc:.3f} *(AUC is threshold-free)*
    {_probe_line}
    """
    )
    stats_md
    return


@app.cell
def _selection_preview(
    class_names,
    deployed_probs,
    labels,
    mo,
    np,
    paths,
    selection_mask,
    tags,
):
    import base64
    from io import BytesIO

    try:
        from PIL import Image
        _pil_ok = True
        _pil_err = None
    except Exception as _e:
        _pil_ok = False
        _pil_err = str(_e)

    import pandas as _pd

    mo.stop(
        not selection_mask.any(),
        mo.md(
            "### Selection preview\n"
            "*Box-select points in the scatter plot above to preview their images.*"
        ),
    )

    _sel_idx = np.where(selection_mask)[0]
    _n_total = len(_sel_idx)
    _max_preview = 12
    _preview_idx = _sel_idx[:_max_preview]

    _thumbs_html = []
    if not _pil_ok:
        _thumbs_html.append(
            f'<div style="color:red;">Pillow not installed: {_pil_err}. '
            f'Install with `pip install Pillow` to see thumbnails.</div>'
        )
    else:
        for _i in _preview_idx:
            _p = str(paths[_i])
            try:
                with Image.open(_p) as _im:
                    _im = _im.convert("RGB")
                    _im.thumbnail((96, 96))
                    _buf = BytesIO()
                    _im.save(_buf, format="PNG")
                    _b64 = base64.b64encode(_buf.getvalue()).decode("ascii")
                _img_tag = (
                    f'<img src="data:image/png;base64,{_b64}" '
                    f'style="display:block;margin:auto;">'
                )
            except Exception as _e:
                _img_tag = (
                    f'<div style="color:red;font-size:10px;">'
                    f'err: {type(_e).__name__}</div>'
                )
            _caption = (
                f'<div style="font-size:10px;text-align:center;margin-top:4px;">'
                f'<b>{class_names[int(labels[_i])]}</b><br>'
                f'{tags[_i]}<br>'
                f'p={float(deployed_probs[_i]):.2f}'
                f'</div>'
            )
            _thumbs_html.append(
                f'<div style="border:1px solid #ddd;padding:4px;border-radius:4px;'
                f'background:white;">{_img_tag}{_caption}</div>'
            )

    _grid = (
        '<div style="display:grid;grid-template-columns:repeat(6,1fr);'
        'gap:8px;margin-top:8px;">'
        + "".join(_thumbs_html)
        + "</div>"
    )

    _df = _pd.DataFrame({
        "path": [str(paths[i]) for i in _sel_idx],
        "label": [class_names[int(labels[i])] for i in _sel_idx],
        "dataset": [tags[i] for i in _sel_idx],
        "p(class 1)": [float(deployed_probs[i]) for i in _sel_idx],
    })

    preview_view = mo.vstack([
        mo.md(
            f"### Selection preview\n"
            f"**{_n_total} points selected.** "
            f"Showing first {min(_n_total, _max_preview)} thumbnails."
        ),
        mo.Html(_grid),
        mo.md("**Full selection table:**"),
        mo.ui.table(_df),
    ])
    preview_view
    return


@app.cell
def _confusion_and_hist(
    class_names, deployed_probs, go, labels, mo, np, roc_auc_score, threshold
):
    _CLASS_COLORS = ["#009E73", "#D55E00", "#0072B2", "#E69F00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
    
    # ── Threshold note ────────────────────────────────────────────────────
    # The deployed model's true operating threshold is UNKNOWN. The confusion
    # matrix below is computed at a *chosen* threshold (the slider), purely for
    # exploration. The ROC curve and ROC-AUC are threshold-free and are the
    # trustworthy summary of the deployed probabilities.
    _thr = float(threshold.value)
    _preds = (deployed_probs >= _thr).astype(int)

    from sklearn.metrics import confusion_matrix, roc_curve
    _cm = confusion_matrix(labels, _preds, labels=list(range(len(class_names))))

    _fig_cm = go.Figure(data=go.Heatmap(
        z=_cm,
        x=[f"Pred {c}" for c in class_names],
        y=[f"True {c}" for c in class_names],
        text=_cm,
        texttemplate="%{text}",
        textfont=dict(size=14),
        colorscale="Blues",
        showscale=False,
    ))
    _fig_cm.update_layout(
        title=dict(text=f"Confusion Matrix @ thr={_thr:.2f}<br><sub>chosen, NOT deployed</sub>",
                   font=dict(size=14), x=0.5, xanchor="center"),
        autosize=True, height=380,
        margin=dict(l=60, r=20, t=60, b=50),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    _fig_cm.update_yaxes(autorange="reversed")

    # ── Probability histogram (threshold-free), with the chosen cut-off drawn
    _fig_hist = go.Figure()
    for _i, _cls in enumerate(class_names):
        _fig_hist.add_trace(go.Histogram(
            x=deployed_probs[labels == _i],
            name=f"True {_cls}",
            opacity=0.7,
            nbinsx=20,
            marker_color=_CLASS_COLORS[_i % len(_CLASS_COLORS)]
        ))
    _fig_hist.add_vline(
        x=_thr, line=dict(color="black", dash="dash", width=2),
        annotation_text=f"thr={_thr:.2f}", annotation_position="top",
    )
    _fig_hist.update_layout(
        title=dict(text="Model confidence<br><sub>p(class 1)</sub>",
                   font=dict(size=14), x=0.5, xanchor="center"),
        barmode="overlay",
        xaxis_title="Predicted probability",
        yaxis_title="Number of images",
        autosize=True, height=380,
        margin=dict(l=60, r=20, t=60, b=50),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top",
                    bgcolor="rgba(255,255,255,0.6)", font=dict(size=10)),
    )

    # ── ROC curve (threshold-free) ────────────────────────────────────────
    # Only well-defined when both classes are present.
    _has_both = len(np.unique(labels)) >= 2
    if _has_both:
        _fpr, _tpr, _ = roc_curve(labels, deployed_probs)
        _auc = roc_auc_score(labels, deployed_probs)
        _fig_roc = go.Figure()
        _fig_roc.add_trace(go.Scatter(
            x=_fpr, y=_tpr, mode="lines",
            name=f"ROC (AUC={_auc:.3f})",
            line=dict(color="#0072B2", width=2),
        ))
        # Mark the operating point at the chosen threshold
        _tp = int(((deployed_probs >= _thr) & (labels == 1)).sum())
        _fn = int(((deployed_probs < _thr) & (labels == 1)).sum())
        _fp = int(((deployed_probs >= _thr) & (labels == 0)).sum())
        _tn = int(((deployed_probs < _thr) & (labels == 0)).sum())
        _tpr_pt = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
        _fpr_pt = _fp / (_fp + _tn) if (_fp + _tn) else 0.0
        _fig_roc.add_trace(go.Scatter(
            x=[_fpr_pt], y=[_tpr_pt], mode="markers",
            name=f"@ thr={_thr:.2f}",
            marker=dict(color="black", size=10, symbol="x"),
        ))
        _fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="chance", line=dict(color="#aaa", dash="dot"),
            showlegend=False,
        ))
        _fig_roc.update_layout(
            title=dict(text="ROC curve<br><sub>threshold-free</sub>",
                       font=dict(size=14), x=0.5, xanchor="center"),
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
            autosize=True, height=380,
            margin=dict(l=60, r=20, t=60, b=50),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(x=0.97, y=0.05, xanchor="right", yanchor="bottom",
                        bgcolor="rgba(255,255,255,0.6)", font=dict(size=10)),
        )
        _fig_roc.update_xaxes(range=[-0.02, 1.02], constrain="domain")
        _fig_roc.update_yaxes(range=[-0.02, 1.02], scaleanchor="x", scaleratio=1)
    else:
        _fig_roc = go.Figure()
        _fig_roc.add_annotation(
            text="ROC needs both classes present",
            showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
        )
        _fig_roc.update_layout(title=dict(text="ROC curve<br><sub>threshold-free</sub>",
                                          font=dict(size=14), x=0.5, xanchor="center"),
                               autosize=True, height=380,
                               margin=dict(l=60, r=20, t=60, b=50))

    confusion_view = mo.vstack([
        mo.md(
            "### Deployed-model diagnostics\n"
            "The confusion matrix uses the **chosen** slider threshold — the "
            "deployed model's real threshold is unknown. The **ROC curve and "
            "AUC are threshold-free** and are the reliable summary; the ✕ marks "
            "where the chosen threshold sits on the curve."
        ),
        threshold,
        mo.hstack(
            [mo.as_html(_fig_cm), mo.as_html(_fig_hist), mo.as_html(_fig_roc)],
            widths="equal", gap=1.0, align="stretch",
        ),
    ])
    confusion_view
    return (confusion_view,)


@app.cell
def _comparison_table(experiment_picker, json, mo, stem_to_dirs):
    _selected_dirs = stem_to_dirs.get(experiment_picker.value, [])

    rows = []
    for d in _selected_dirs:
        probe_file = d / "linear_probe.json"
        if not probe_file.exists():
            continue
        result = json.loads(probe_file.read_text())
        rows.append({
            "Dataset": result.get("dataset_tag", d.name),
            "deployed_acc": result["deployed"]["accuracy"],
            "deployed_bal_acc": result["deployed"]["balanced_accuracy"],
            "deployed_auc": result["deployed"].get("roc_auc"),
            "pooled_probe_bal_acc": result["post_hoc_pooled"]["balanced_accuracy_mean"],
            "pooled_probe_auc": result["post_hoc_pooled"].get("roc_auc_mean"),
            "proj_probe_bal_acc": result["post_hoc_projection"]["balanced_accuracy_mean"],
            "proj_probe_auc": result["post_hoc_projection"].get("roc_auc_mean"),
        })
    if not rows:
        _view = mo.md(
            "*Run `python linear_probe.py` on these folders to populate the "
            "linear separability comparison table here.*"
        )
    else:
        import pandas as _pd
        df = _pd.DataFrame(rows)
        _view = mo.vstack([mo.md("## Linear separability comparison"), mo.ui.table(df)])
    _view
    return


@app.cell
def _cmp_controls(experiment_stems, mo):
    mo.md("---")
    cmp_intro = mo.md(
        "## Side-by-side checkpoint comparison\n"
        "Pick **two experiments** to view their representation spaces next to "
        "each other. Each panel is projected **independently** (its own PCA/UMAP "
        "fit on its own features), so compare *cluster geometry* — separation, "
        "train/test overlap, tightness — rather than absolute coordinates. The "
        "deployed **ROC-AUC** under each panel is the threshold-free, comparable "
        "number."
    )

    _default_b = experiment_stems[1] if len(experiment_stems) > 1 else experiment_stems[0]
    cmp_a = mo.ui.dropdown(
        options=experiment_stems, value=experiment_stems[0], label="Checkpoint A",
    )
    cmp_b = mo.ui.dropdown(
        options=experiment_stems, value=_default_b, label="Checkpoint B",
    )
    cmp_feature_picker = mo.ui.dropdown(
        options=["pooled (768-D, paper's r)", "projection (128-D, paper's z)"],
        value="pooled (768-D, paper's r)",
        label="Features",
    )
    cmp_proj_picker = mo.ui.dropdown(
        options=["PCA", "UMAP"], value="PCA", label="Projection (per panel)",
    )

    cmp_controls = mo.vstack([
        cmp_intro,
        mo.hstack([cmp_a, cmp_b]),
        mo.hstack([cmp_proj_picker, cmp_feature_picker]),
    ])
    cmp_controls
    return cmp_a, cmp_b, cmp_feature_picker, cmp_proj_picker


@app.cell
def _cmp_view(
    cmp_a,
    cmp_b,
    cmp_feature_picker,
    cmp_proj_picker,
    metric,
    min_dist,
    mo,
    n_neighbors,
    seed,
    stem_to_dirs,
):
    import pandas as _pd

    _feature_kind = (
        "pooled" if cmp_feature_picker.value.startswith("pooled") else "projection"
    )

    def _panel(_stem):
        _dirs = stem_to_dirs.get(_stem, [])
        _bundle = cmp_load_stem(None, _dirs, _feature_kind)
        if _bundle is None:
            return (
                mo.md(f"⚠️ *No loadable folders for* `{_stem}`."),
                {"checkpoint": _stem, "n_samples": 0,
                 "feature_dim": 0, "deployed_auc": float("nan")},
            )
        _fig, _metrics = cmp_build_scatter(
            _bundle, _stem, cmp_proj_picker.value, seed.value,
            n_neighbors.value, min_dist.value, metric.value,
        )
        return _fig, _metrics

    _fig_a, _met_a = _panel(cmp_a.value)
    _fig_b, _met_b = _panel(cmp_b.value)

    _table = _pd.DataFrame([_met_a, _met_b])

    cmp_view = mo.vstack([
        mo.hstack([_fig_a, _fig_b]),
        mo.md("**Threshold-free comparison (deployed ROC-AUC):**"),
        mo.ui.table(_table),
    ])
    cmp_view
    return


if __name__ == "__main__":
    app.run()

