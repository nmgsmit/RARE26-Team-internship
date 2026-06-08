"""
Stage 4: interactive UMAP/PCA notebook (marimo), DEMO EDITION.

Run with:
    marimo edit notebook.py
"""

# ── Pin Numba to single-threaded BEFORE any umap/numba import ─────────────────
import os as _os
_os.environ.setdefault("NUMBA_NUM_THREADS", "1")
_os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")  # Wide layout for side-by-side plots

# ── Store UMAP lock on the threading module, survives marimo's name mangling ──
import threading as _threading_init
if not hasattr(_threading_init, "umap_lock"):
    _threading_init.umap_lock = _threading_init.Lock()


# ── Public helper: UMAP fit (serialized through the module-level lock) ────────
@app.function
def fit_umap_safe(X, n_neighbors, min_dist, metric, seed, n_components=2):
    import threading
    import warnings
    import umap

    _lock = getattr(threading, "umap_lock", None)
    if _lock is None:
        threading.umap_lock = threading.Lock()
        _lock = threading.umap_lock

    _n = max(2, int(min(n_neighbors, max(2, X.shape[0] - 1))))
    with _lock:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*n_jobs value.*overridden.*")
            reducer = umap.UMAP(
                n_neighbors=_n,
                min_dist=float(min_dist),
                metric=metric,
                n_components=int(n_components),
                random_state=int(seed),
            )
            return reducer.fit_transform(X)
# ── Public helper: load experiment for overview ───────────────────────────────
@app.function
def ov_load_experiment(features_root, stem, feature_kind="projection"):
    import json
    from pathlib import Path
    import numpy as np

    _feature_file = "features_pooled.npy" if feature_kind == "pooled" else "features_proj.npy"

    features_root = Path(features_root).expanduser()
    feats_all, labels_all, tags_all, centers_all, paths_all, probs_all = [], [], [], [], [], []
    class_names = None
    for _split in ("train_all", "evc_test"):
        _folder = features_root / f"{stem}____{_split}" if not (features_root / f"{stem}__{_split}").exists() else features_root / f"{stem}__{_split}"
        if not _folder.exists():
            _cands = list(features_root.glob(f"{stem}*{_split}*"))
            if not _cands:
                continue
            _folder = _cands[0]
        _meta_path = _folder / "meta.json"
        if not _meta_path.exists():
            continue
        _meta = json.loads(_meta_path.read_text())
        if class_names is None:
            class_names = _meta["class_names"]
        # Fall back to projection features if pooled file is absent.
        _ff = _folder / _feature_file
        if not _ff.exists():
            _ff = _folder / "features_proj.npy"
        _feats = np.load(_ff)
        _lbls = np.load(_folder / "labels.npy")
        _paths_f = _folder / "paths.npy"
        _paths = None
        if _paths_f.exists():
            try:
                _paths = np.load(_paths_f, allow_pickle=True)
            except Exception:
                _paths = None

        _probs_f = _folder / "deployed_probs.npy"
        if _probs_f.exists():
            _probs_raw = np.load(_probs_f)
            _probs = _probs_raw[:, 1] if _probs_raw.ndim == 2 else _probs_raw
        else:
            _probs = np.zeros(len(_lbls))

        _centers = resolve_centers(_folder, _paths, len(_lbls), _split)
        feats_all.append(_feats)
        labels_all.append(_lbls)
        tags_all.extend([_split] * len(_lbls))
        centers_all.append(_centers)
        probs_all.append(_probs)
        paths_all.extend(_paths if _paths is not None else [""] * len(_lbls))
    if not feats_all:
        return None
    return {
        "features":       np.concatenate(feats_all, 0),
        "labels":         np.concatenate(labels_all, 0),
        "tags":           np.array(tags_all),
        "centers":        np.concatenate(centers_all, 0),
        "deployed_probs": np.concatenate(probs_all, 0),
        "paths":          np.asarray(paths_all, dtype=object),
        "class_names":    class_names,
    }


# ── Public helper: PCA 2D ─────────────────────────────────────────────────────
@app.function
def ov_pca_coords(X):
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=42).fit_transform(X)
# ── Public helper: Interactive Phase Scatter ──────────────────────────────────
@app.function
def build_phase_view(bundle, proj_method="PCA", seed=42,
                     n_neighbors=15, min_dist=0.1, metric="cosine",
                     max_points=600):
    """Project + subsample ONCE so coordinates are stable.

    Returns a dict of parallel per-point arrays. Crucially, the projection and
    the subsample are computed a single time here, independent of how the points
    are later coloured (by class or by center). That guarantees a point keeps the
    exact same (x, y) when you toggle the colour mode, only its colour changes.
    """
    import numpy as np

    X = bundle["features"]
    y = bundle["labels"]
    t = bundle["tags"]
    class_names = bundle["class_names"]
    paths = bundle.get("paths", np.array([""] * len(y), dtype=object))
    probs = bundle.get("deployed_probs", np.zeros(len(y)))
    centers = bundle.get("centers", np.array(["unknown"] * len(y), dtype=object))

    if proj_method == "UMAP":
        try:
            coords = fit_umap_safe(X, n_neighbors, min_dist, metric, seed)
            proj_label = "UMAP"
        except Exception:
            coords = ov_pca_coords(X)
            proj_label = "PCA (UMAP failed)"
    else:
        coords = ov_pca_coords(X)
        proj_label = "PCA"

    is_test = np.array(["train" not in str(_tg).lower() for _tg in t])

    # Single deterministic subsample over ALL points (not per colour group).
    n = coords.shape[0]
    idx = np.arange(n)
    if n > max_points:
        idx = np.random.default_rng(seed).choice(n, max_points, replace=False)
        idx.sort()

    # Center labels: held-out test points are relabelled to one "test" group.
    centers = np.asarray(centers, dtype=object).copy()
    centers[is_test] = "EVC (test)"

    return {
        "xs":          coords[idx, 0].astype(float),
        "ys":          coords[idx, 1].astype(float),
        "labels_idx":  y[idx].astype(int),
        "class_names": list(class_names),
        "centers":     centers[idx],
        "is_test":     is_test[idx],
        "paths":       np.asarray([str(paths[i]) for i in idx], dtype=object),
        "probs":       np.asarray([float(probs[i]) for i in idx], dtype=float),
        "proj_label":  proj_label,
    }


@app.function
def render_phase_fig(view, title, color_by="class", width=250, height=250):
    """Render a scatter from a precomputed stable view, colouring by class or
    center. Point positions come straight from `view` and never change between
    colour modes."""
    import numpy as np
    import plotly.graph_objects as go

    def _rgba(hex_color, alpha=1.0):
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    xs = view["xs"]; ys = view["ys"]
    y = view["labels_idx"]
    class_names = view["class_names"]
    is_test = view["is_test"]
    paths = view["paths"]; probs = view["probs"]
    centers = view["centers"]
    proj_label = view["proj_label"]

    fig = go.Figure()

    if color_by == "class":
        _CLASS_COLORS = ["#009E73", "#D55E00"]
        groups = [(ci, cn) for ci, cn in enumerate(class_names)]
        def _key(i):     return y[i]
        def _color(g):   return _CLASS_COLORS[g % 2]
        def _name(g, cn): return cn
        group_ids = [g[0] for g in groups]
        group_lbls = {g[0]: g[1] for g in groups}
    else:  # center
        _CENTER_COLORS = ["#0072B2", "#E69F00", "#CC79A7", "#56B4E9", "#F0E442", "#009E73", "#D55E00", "#000000"]
        uniq = sorted(set(centers.tolist()))
        if "EVC (test)" in uniq:
            uniq.remove("EVC (test)"); uniq.append("EVC (test)")
        color_of = {lab: _CENTER_COLORS[i % len(_CENTER_COLORS)] for i, lab in enumerate(uniq)}
        group_ids = uniq
        group_lbls = {u: u for u in uniq}

    # Draw train first (translucent circles), then test (opaque diamonds on top).
    for _is_train_pass in (True, False):
        _alpha = 0.45 if _is_train_pass else 0.95
        _sym = "circle" if _is_train_pass else "diamond"
        _lw = 0.3 if _is_train_pass else 0.9
        _lc = "rgba(255,255,255,0.6)" if _is_train_pass else "#222222"
        _sz = 5 if _is_train_pass else 7
        split_mask = (~is_test) if _is_train_pass else is_test
        split_word = "train" if _is_train_pass else "test"

        for g in group_ids:
            if color_by == "class":
                m = (y == g) & split_mask
                color = _rgba(["#009E73", "#D55E00"][g % 2], _alpha)
                gname = group_lbls[g]
                clabels = [class_names[int(y[i])] for i in np.where(m)[0]]
            else:
                m = (centers == g) & split_mask
                color = _rgba(color_of[g], _alpha)
                gname = group_lbls[g]
                clabels = [class_names[int(y[i])] for i in np.where(m)[0]]
            if not m.any():
                continue
            idxs = np.where(m)[0]
            custom = [[paths[i], class_names[int(y[i])], float(probs[i])] for i in idxs]
            fig.add_trace(go.Scatter(
                x=xs[idxs].tolist(), y=ys[idxs].tolist(),
                mode="markers",
                name=f"{gname} ({split_word})",
                marker=dict(size=_sz, symbol=_sym, color=color,
                            line=dict(width=_lw, color=_lc)),
                customdata=custom,
                hovertemplate=(
                    f"<b>{gname}</b> / {split_word}<br>"
                    "class: %{customdata[1]}<br>"
                    "p: %{customdata[2]:.3f}<br>"
                    "%{customdata[0]}<extra></extra>"
                ),
                showlegend=False,
            ))

    fig.update_layout(
        width=width, height=height,
        margin=dict(l=6, r=6, t=6, b=6),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        dragmode="select",
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False,
                     showline=True, linewidth=1, linecolor="#333", mirror=True, ticks="")
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False,
                     showline=True, linewidth=1, linecolor="#333", mirror=True, ticks="")

    plotted = {
        "xs": xs, "ys": ys,
        "paths": paths,
        "labels": np.asarray([class_names[int(c)] for c in y], dtype=object),
        "probs": probs,
    }
    return fig, plotted
# ── Public helper: range-based preview for phase panels ──────────────────────
@app.function
def render_range_preview(panels, mo, prefer_index=None):
    """Preview images for points inside box-selections across phase panels.

    `panels` is a list of (plotted, ranges) tuples, where `plotted` is the dict
    returned by render_phase_fig and `ranges` is widget.ranges (the box
    extents). This does NOT depend on customdata surviving the frontend
    round-trip; it masks the originally-plotted coordinates by the selected box,
    exactly like the deep-dive scatter does.

    If `prefer_index` is given (the last-touched panel), that panel's selection
    is shown and the others are ignored. Falls back to the first panel that has
    an active selection.
    """
    import base64
    from io import BytesIO
    import numpy as np

    _empty_msg = mo.md(
        "<div style='text-align:center; color:#888; margin-top:2rem; font-size:13px;'>"
        "<i>Box-select points in any scatter plot<br>to preview images here.</i>"
        "</div>"
    )

    def _select_from(_plotted, _ranges):
        if not _ranges or "x" not in _ranges or "y" not in _ranges:
            return None
        if _plotted is None or len(_plotted.get("xs", [])) == 0:
            return None
        _xr = _ranges["x"]; _yr = _ranges["y"]
        _xmin, _xmax = min(_xr), max(_xr)
        _ymin, _ymax = min(_yr), max(_yr)
        _xs = _plotted["xs"]; _ys = _plotted["ys"]
        _mask = (_xs >= _xmin) & (_xs <= _xmax) & (_ys >= _ymin) & (_ys <= _ymax)
        if not _mask.any():
            return None
        return (
            _plotted["paths"][_mask].tolist(),
            _plotted["labels"][_mask].tolist(),
            _plotted["probs"][_mask].tolist(),
        )

    # Prefer the last-touched panel; show only its selection ("others ignored").
    # Only fall back to scanning other panels when nothing has been touched yet.
    _sel = None
    if prefer_index is not None and 0 <= prefer_index < len(panels):
        _sel = _select_from(*panels[prefer_index])
    else:
        for _plotted, _ranges in panels:
            _sel = _select_from(_plotted, _ranges)
            if _sel is not None:
                break

    if _sel is None:
        return _empty_msg

    _sel_paths, _sel_labels, _sel_probs = _sel

    try:
        from PIL import Image as _PILImage
        _pil_ok = True
    except Exception as _e:
        _pil_ok = False
        _pil_err = str(_e)

    _n_total = len(_sel_paths)
    _max_preview = 8
    _thumbs_html = []
    if not _pil_ok:
        _thumbs_html.append(f'<div style="color:red;">Pillow not installed: {_pil_err}</div>')
    else:
        for _k in range(min(_n_total, _max_preview)):
            _path = str(_sel_paths[_k])
            _label = str(_sel_labels[_k])
            _prob = float(_sel_probs[_k])
            try:
                if _path:
                    with _PILImage.open(_path) as _im:
                        _im = _im.convert("RGB")
                        _im.thumbnail((150, 150))
                        _buf = BytesIO()
                        _im.save(_buf, format="PNG")
                        _b64 = base64.b64encode(_buf.getvalue()).decode("ascii")
                    _img_tag = f'<img src="data:image/png;base64,{_b64}" style="display:block;margin:auto;max-width:100%;height:auto;border-radius:4px;">'
                else:
                    _img_tag = '<div style="width:100%;aspect-ratio:1;background:#f1f5f9;border-radius:4px;margin:auto;display:flex;align-items:center;justify-content:center;color:#94a3b8;font-size:11px;">no path</div>'
            except Exception as _e:
                _img_tag = f'<div style="color:red;font-size:11px;">err: {type(_e).__name__}</div>'

            _caption = (
                f'<div style="font-size:12px;text-align:center;margin-top:4px;line-height:1.2;">'
                f'<strong style="color:#333;">{_label}</strong><br>'
                f'<span style="color:#666;">p={_prob:.2f}</span></div>'
            )
            _thumbs_html.append(
                f'<div style="border:1px solid #e2e8f0;padding:5px;border-radius:6px;background:white;box-shadow:0 1px 2px rgba(0,0,0,0.05);">'
                f'{_img_tag}{_caption}</div>'
            )

    _grid = (
        '<div style="display:grid;grid-template-columns:repeat(auto-fill, minmax(120px, 1fr));gap:10px;margin-top:12px;">'
        + "".join(_thumbs_html) + "</div>"
    )
    return mo.vstack([
        mo.md(f"**{_n_total} points selected**<br><span style='font-size:12px;color:#666;'>(showing up to {_max_preview}, biggest preview)</span>"),
        mo.Html(_grid)
    ])


# ── Public helper: persistent PPV@90R progress chart ─────────────────────────
@app.function
def md_inline(text):
    """Tiny inline-markdown → HTML converter for **bold** and *italic* so we can
    embed explanation text inside a styled HTML block."""
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return text


@app.function
def progress_chart(reveal_up_to):
    """The constant figure that grows each slide. Shows the running best
    PPV@90Recall milestone for each phase, revealing points up to `reveal_up_to`
    (a milestone index). The RARE25 1st-place baseline is always shown."""
    import plotly.graph_objects as go

    # (label, sublabel, PPV@90R). Index 0 is the baseline; 1..5 are the phases.
    milestones = [
        ("RARE25<br>1st place", "baseline 2025", 0.0355),
        ("GastroNet DINOv2<br>CE", "Phase 1", 0.0136),
        ("GastroNet DINOv2<br>SupPro", "Phase 1", 0.0146),
        ("τ = 0.10", "Phase 2", 0.015),
        ("balanced 25%", "Phase 3", 0.015),
        ("random crop 0.8", "Phase 4", 0.017),
        ("k-NN, crop 0.95", "Phase 5", 0.039),
    ]
    k = max(1, min(len(milestones), reveal_up_to))
    shown = milestones[:k]

    xs = list(range(len(shown)))
    ys = [m[2] for m in shown]
    labels = [m[0] for m in shown]
    subs = [m[1] for m in shown]

    fig = go.Figure()

    # Baseline reference line across the whole axis.
    fig.add_hline(y=0.0355, line=dict(color="#888", dash="dot", width=1),
                  annotation_text="RARE25 best (0.0355)", annotation_position="top left",
                  annotation_font=dict(size=9, color="#888"))

    # The progress line for the phases (index >= 1).
    if len(shown) >= 2:
        fig.add_trace(go.Scatter(
            x=xs[1:], y=ys[1:], mode="lines+markers",
            line=dict(color="#0072B2", width=2.5),
            marker=dict(size=11, color="#0072B2", line=dict(color="white", width=1.5)),
            hovertemplate="%{text}<br>PPV@90R=%{y:.4f}<extra></extra>",
            text=[f"{labels[i]} ({subs[i]})" for i in range(1, len(shown))],
            showlegend=False,
        ))

    # The baseline point itself (distinct marker).
    fig.add_trace(go.Scatter(
        x=[0], y=[ys[0]], mode="markers",
        marker=dict(size=12, color="#888", symbol="diamond", line=dict(color="white", width=1.5)),
        hovertemplate="RARE25 1st place<br>PPV@90R=%{y:.4f}<extra></extra>",
        showlegend=False,
    ))

    # Highlight the most recently revealed point.
    fig.add_trace(go.Scatter(
        x=[xs[-1]], y=[ys[-1]], mode="markers",
        marker=dict(size=16, color="#D55E00", symbol="star", line=dict(color="white", width=1.5)),
        hovertemplate="latest: %{y:.4f}<extra></extra>",
        showlegend=False,
    ))

    # Value labels above each point.
    for i in range(len(shown)):
        fig.add_annotation(x=xs[i], y=ys[i], text=f"<b>{ys[i]:.4f}</b>",
                           showarrow=False, yshift=16, font=dict(size=10, color="#333"))

    fig.update_layout(
        title=dict(text="Progress: PPV@90Recall vs RARE25 best", font=dict(size=13), x=0.5, xanchor="center"),
        height=340, margin=dict(l=50, r=20, t=50, b=80),
        plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
        yaxis_title="PPV@90Recall",
    )
    fig.update_xaxes(
        tickmode="array", tickvals=xs,
        ticktext=[f"{labels[i]}" for i in range(len(shown))],
        tickfont=dict(size=8), showgrid=False,
        showline=True, linecolor="#333", range=[-0.4, max(6.4, len(shown)-0.6)],
    )
    fig.update_yaxes(
        range=[0, 0.045], showgrid=True, gridcolor="#eee",
        showline=True, linecolor="#333",
    )
    return fig


# ── Public helper: compact per-phase legend ──────────────────────────────────
@app.function
def phase_legend_html(mode="both"):
    """Small legend shown next to each phase's scatter. `mode` is 'class',
    'center', or 'both'."""
    def _dot(color, shape="circle", outline="#222"):
        _radius = "50%" if shape == "circle" else "2px"
        _rot = "" if shape == "circle" else "transform:rotate(45deg);"
        return (
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'background:{color};border:1px solid {outline};'
            f'border-radius:{_radius};{_rot}margin-right:4px;vertical-align:middle;"></span>'
        )
    rows = []
    if mode in ("class", "both"):
        rows.append(
            f'<div><b>Class:</b>&nbsp; {_dot("#009E73")}NDBE'
            f'&nbsp;&nbsp;{_dot("#D55E00")}neoplasia</div>'
        )
    if mode in ("center", "both"):
        rows.append(
            f'<div><b>Center:</b>&nbsp; {_dot("#0072B2")}center 1'
            f'&nbsp;&nbsp;{_dot("#E69F00")}center 2'
            f'&nbsp;&nbsp;{_dot("#CC79A7")}EVC (test)</div>'
        )
    rows.append(
        f'<div><b>Split:</b>&nbsp; {_dot("#888","circle","#fff")}train (●)'
        f'&nbsp;&nbsp;{_dot("#888","diamond")}test (◆, on top)</div>'
    )
    return (
        '<div style="font-size:11px;line-height:1.8;padding:6px 8px;'
        'background:rgba(0,0,0,0.02);border-radius:6px;">'
        + "".join(rows) + "</div>"
    )


# ── Public helper: static legend HTML ────────────────────────────────────────
# ── Public helper: example image strips for slide 0 ──────────────────────────
@app.function
def example_images_html(features_root_value, mo, thumb_px=200, n_each=4):
    import base64
    import numpy as np
    from io import BytesIO
    from pathlib import Path
    try:
        from PIL import Image as _PILImage
    except Exception:
        return mo.md("*(Pillow not available; install to preview example images.)*")

    root = Path(features_root_value).expanduser()
    cand = None
    if root.exists():
        for p in sorted(root.glob("*")):
            if (p / "paths.npy").exists() and (p / "labels.npy").exists():
                cand = p
                break
    if cand is None:
        return mo.md(
            "*(No `paths.npy` found under the features root; example images will appear "
            "here once real data is present.)*"
        )

    paths = np.load(cand / "paths.npy", allow_pickle=True)
    labels = np.load(cand / "labels.npy")

    def _strip(which_label, title, color):
        idxs = np.where(labels == which_label)[0][:n_each]
        thumbs = []
        for i in idxs:
            try:
                with _PILImage.open(str(paths[i])) as im:
                    im = im.convert("RGB"); im.thumbnail((thumb_px, thumb_px))
                    buf = BytesIO(); im.save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                tag = (f'<img src="data:image/png;base64,{b64}" '
                       f'style="width:{thumb_px}px;height:{thumb_px}px;object-fit:cover;'
                       f'border-radius:6px;display:block;">')
            except Exception:
                tag = f'<div style="width:{thumb_px}px;height:{thumb_px}px;background:#eee;border-radius:6px;"></div>'
            thumbs.append(f"<div>{tag}</div>")
        return (
            f'<div style="flex:1;min-width:0;">'
            f'<div style="font-size:14px;font-weight:600;color:{color};margin-bottom:6px;">{title}</div>'
            f'<div style="display:flex;gap:8px;flex-wrap:nowrap;">{"".join(thumbs)}</div></div>'
        )

    html = (
        '<div style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap;">'
        + _strip(0, "NDBE (non-dysplastic)", "#009E73")
        + _strip(1, "Neoplasia", "#D55E00")
        + '</div>'
    )
    return mo.Html(html)

# ── Public helper: resolve acquisition centers ────────────────────────────────
@app.function
def resolve_centers(folder, paths, n, split):
    import json
    import re
    from pathlib import Path
    import numpy as np

    folder = Path(folder)
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
                _labels.append("EVC (test)")
                _any = True
            else:
                _labels.append("unknown")
        if _any:
            return np.array(_labels)

    return np.array(["unknown"] * n)


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1: imports
# ═══════════════════════════════════════════════════════════════════════════════
@app.cell
def _imports():
    import os as _os_imports
    _os_imports.environ.setdefault("NUMBA_NUM_THREADS", "1")
    _os_imports.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import umap
    from sklearn.decomposition import PCA

    return (
        PCA,
        Path,
        go,
        json,
        mo,
        np,
    )




# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD CONTROLS: slide navigation + per-phase display switches
# ═══════════════════════════════════════════════════════════════════════════════
@app.cell
def _controls(mo):
    SLIDE_TITLES = [
        "0 · The problem & dataset",
        "1 · Phase 1: backbone × loss",
        "2 · Phase 2: temperature τ",
        "3 · Phase 3: balanced sampling",
        "4 · Phase 4: ROI vs random crops",
        "5 · Phase 5: head × crop",
        "6 · Final: best config & conclusions",
    ]
    slide = mo.ui.slider(
        start=0, stop=6, step=1, value=0, show_value=False,
        label="Slide", full_width=True,
    )

    # A single set of display switches drives whichever slide is visible.
    # off = class / PCA / projection ; on = center / UMAP / pooled
    # (Plain top-level switches; marimo tracks these reliably; a dict of
    # switches does NOT reliably trigger re-render.)
    col_sw  = mo.ui.switch(value=False, label="colour by center")
    proj_sw = mo.ui.switch(value=False, label="UMAP")
    feat_sw = mo.ui.switch(value=False, label="pooled features")

    # Toggle the bottom-right panel between the results table and the PPV@90R
    # progress chart, to save vertical space.
    view_sw = mo.ui.radio(
        options=["PPV@90R progress", "Results table"],
        value="PPV@90R progress", inline=True,
    )

    # Tracks which scatter panel was box-selected most recently, so the preview
    # shows ONLY the last-touched panel's selection.
    get_last_sel, set_last_sel = mo.state(None)

    return (SLIDE_TITLES, slide, col_sw, proj_sw, feat_sw, view_sw,
            get_last_sel, set_last_sel)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG: features root + per-phase experiment definitions & result tables
# ═══════════════════════════════════════════════════════════════════════════════
@app.cell
def _config(mo):
    features_root = mo.ui.text(value="features_out", label="Features root directory", full_width=True)

    # Each phase: list of (stem, short title) panels to show on that slide.
    PHASE_STEMS = {
        1: [
            ("P0_Base_GastronetDinoV2_1e-3_t1", "GastroNet DINOv2 · CE"),
            ("P1_BB_GastronetDinoV2_t1",        "GastroNet DINOv2 · SupPro ✓"),
            ("P1_BB_DinoV3_t1",                 "DINOv3 · SupPro"),
        ],
        2: [
            ("P2_Temp_0.07_t1", "τ = 0.07"),
            ("P2_Temp_0.1_t1",  "τ = 0.10 ✓"),
            ("P2_Temp_0.3_t1",  "τ = 0.30"),
            ("P2_Temp_0.5_t1",  "τ = 0.50"),
        ],
        3: [
            ("P6_BalSam_05_Linear", "Balanced 5%"),
            ("P6_BalSam_25_Linear", "Balanced 25% ✓"),
            ("P6_BalSam_50_Linear", "Balanced 50%"),
        ],
        4: [
            ("P7_Random_crop04",   "Random 0.4"),
            ("P7_Random_crop08",   "Random 0.8 ✓"),
            ("P7_Random_crop1",    "Full image"),
            ("P7_ROI_crop04",      "ROI 0.4"),
            ("P7_ROI_crop08_REAL", "ROI 0.8"),
        ],
        5: [
            ("P8_scale04_finetune_knn",  "crop 0.4"),
            ("P8_scale06_finetune_knn",  "crop 0.6"),
            ("P8_scale08_finetune_knn",  "crop 0.8"),
            ("P8_scale095_finetune_knn", "crop 0.95"),
        ],
        6: [
            ("P8_scale095_finetune_knn", "Best · k-NN crop 0.95"),
        ],
    }

    import pandas as _pd
    PHASE_TABLES = {
        1: _pd.DataFrame([
            {"Backbone": "Gastro DINOv2",   "Loss": "CE", "PPV@90R": 0.0136, "AUROC": 0.825, "AUPRC": 0.379, "Cons.Mass": 0.461},
            {"Backbone": "Gastro DINOv2",   "Loss": "SP", "PPV@90R": 0.0146, "AUROC": 0.848, "AUPRC": 0.442, "Cons.Mass": 0.568, "Note": "✓ best"},
            {"Backbone": "Gastro SimCLR",   "Loss": "CE", "PPV@90R": 0.0105, "AUROC": 0.711, "AUPRC": 0.132, "Cons.Mass": 0.541},
            {"Backbone": "Gastro SimCLR",   "Loss": "SP", "PPV@90R": 0.0121, "AUROC": 0.824, "AUPRC": 0.197, "Cons.Mass": 0.412},
            {"Backbone": "Gastro MoCo",     "Loss": "CE", "PPV@90R": 0.0113, "AUROC": 0.747, "AUPRC": 0.175, "Cons.Mass": 0.609},
            {"Backbone": "Gastro MoCo",     "Loss": "SP", "PPV@90R": 0.0114, "AUROC": 0.703, "AUPRC": 0.131, "Cons.Mass": 0.528},
            {"Backbone": "Gastro ResNet50", "Loss": "CE", "PPV@90R": 0.0106, "AUROC": 0.645, "AUPRC": 0.049, "Cons.Mass": 0.401},
            {"Backbone": "Gastro ResNet50", "Loss": "SP", "PPV@90R": 0.0100, "AUROC": 0.585, "AUPRC": 0.095, "Cons.Mass": 0.272},
            {"Backbone": "DINOv3",          "Loss": "CE", "PPV@90R": 0.0103, "AUROC": 0.700, "AUPRC": 0.113, "Cons.Mass": 0.467},
            {"Backbone": "DINOv3",          "Loss": "SP", "PPV@90R": 0.0097, "AUROC": 0.687, "AUPRC": 0.100, "Cons.Mass": 0.408},
        ]),
        2: _pd.DataFrame([
            {"Temperature": "0.07", "PPV@90R": 0.012, "AUROC": 0.83, "AUPRC": 0.43, "Cons.Mass": 0.26},
            {"Temperature": "0.10", "PPV@90R": 0.015, "AUROC": 0.89, "AUPRC": 0.56, "Cons.Mass": 0.70, "Note": "✓ best"},
            {"Temperature": "0.30", "PPV@90R": 0.012, "AUROC": 0.85, "AUPRC": 0.52, "Cons.Mass": 0.39},
            {"Temperature": "0.50", "PPV@90R": 0.011, "AUROC": 0.84, "AUPRC": 0.52, "Cons.Mass": 0.65},
        ]),
        3: _pd.DataFrame([
            {"Positive share": "5%",  "PPV@90R": 0.010, "AUROC": 0.83, "AUPRC": 0.52, "Cons.Mass": 0.32},
            {"Positive share": "25%", "PPV@90R": 0.015, "AUROC": 0.87, "AUPRC": 0.50, "Cons.Mass": 0.33, "Note": "✓ best"},
            {"Positive share": "50%", "PPV@90R": 0.013, "AUROC": 0.86, "AUPRC": 0.53, "Cons.Mass": 0.24},
        ]),
        4: _pd.DataFrame([
            {"Cropping Strategy": "Random Crop (0.4)", "PPV@90R": 0.014, "AUROC": 0.86, "AUPRC": 0.47, "Cons.Mass": 0.75},
            {"Cropping Strategy": "Random Crop (0.8)", "PPV@90R": 0.017, "AUROC": 0.89, "AUPRC": 0.49, "Cons.Mass": 0.59, "Note": "✓ best PPV"},
            {"Cropping Strategy": "Full Image",        "PPV@90R": 0.016, "AUROC": 0.89, "AUPRC": 0.51, "Cons.Mass": 0.65},
            {"Cropping Strategy": "ROI Crop (0.4)",    "PPV@90R": 0.013, "AUROC": 0.86, "AUPRC": 0.50, "Cons.Mass": 0.67},
            {"Cropping Strategy": "ROI Crop (0.8)",    "PPV@90R": 0.014, "AUROC": 0.87, "AUPRC": 0.53, "Cons.Mass": 0.71},
        ]),
        5: _pd.DataFrame([
            {"Best Configuration": "Linear (0.60)", "PPV@90R": 0.015, "AUROC": 0.88, "AUPRC": 0.57, "Cons.Mass": 0.66},
            {"Best Configuration": "k-NN (0.95)",   "PPV@90R": 0.039, "AUROC": 0.79, "AUPRC": 0.50, "Cons.Mass": None, "Note": "✓ best PPV"},
            {"Best Configuration": "MLP (0.60)",    "PPV@90R": 0.013, "AUROC": 0.86, "AUPRC": 0.52, "Cons.Mass": 0.64},
            {"Best Configuration": "SVM (0.95)",    "PPV@90R": 0.016, "AUROC": 0.90, "AUPRC": 0.56, "Cons.Mass": None, "Note": "best AUROC"},
        ]),
    }
    return (features_root, PHASE_STEMS, PHASE_TABLES)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA: load all phase bundles + build stable views (positions fixed here)
# ═══════════════════════════════════════════════════════════════════════════════
@app.cell
def _load_all(Path, features_root, mo, col_sw, proj_sw, feat_sw, PHASE_STEMS,
              ov_load_experiment, build_phase_view):
    _root = Path(features_root.value).expanduser()

    # Projection + feature kind apply globally (driven by the single switches).
    _pm = "UMAP" if proj_sw.value else "PCA"
    _fk = "pooled" if feat_sw.value else "projection"

    phase_views = {}      # phase -> list of (title, view_or_None)
    for _ph, _panels in PHASE_STEMS.items():
        _views = []
        for _stem, _title in _panels:
            _b = ov_load_experiment(_root, _stem, _fk)
            if _b is None or _b["features"].shape[0] < 4:
                _views.append((_title, None))
            else:
                _views.append((_title, build_phase_view(_b, proj_method=_pm)))
        phase_views[_ph] = _views

    return (phase_views,)




# ═══════════════════════════════════════════════════════════════════════════════
# CURRENT-SLIDE WIDGETS: built as TRACKED cell globals so box-select .ranges works
# ═══════════════════════════════════════════════════════════════════════════════
@app.cell
def _slide_widgets(mo, slide, col_sw, phase_views, render_phase_fig, set_last_sel):
    _s = int(slide.value)
    _cb = "center" if col_sw.value else "class"

    # Reset "last selected" whenever the slide or colouring changes (the old
    # panels no longer exist).
    set_last_sel(None)

    # marimo tracks reactivity per top-level variable. A *list* of widgets does
    # NOT propagate a child's box-select to dependent cells, so each plotly widget
    # must be its own named global. We use 6 fixed slots (pw0..pw5): phase slides
    # use up to 5; the final slide uses pw0 (by class) and pw1 (by center).
    pw0 = pw1 = pw2 = pw3 = pw4 = pw5 = None
    pp0 = pp1 = pp2 = pp3 = pp4 = pp5 = None
    cur_titles = []
    cur_kind = "info"

    def _mk(view, title, cb, slot, w=200, h=200):
        _fig, _pl = render_phase_fig(view, title, color_by=cb, width=w, height=h)
        # When this panel is box-selected, record its slot as the last touched.
        _w = mo.ui.plotly(_fig, on_change=lambda _v, _i=slot: set_last_sel(_i))
        return _w, _pl

    if 1 <= _s <= 5:
        cur_kind = "phase"
        _panels = phase_views[_s]
        _slots = []
        for _slot, (_title, _view) in enumerate(_panels[:6]):
            cur_titles.append(_title)
            if _view is None:
                _slots.append((None, None))
            else:
                _slots.append(_mk(_view, _title, _cb, _slot))
        # pad to 6
        while len(_slots) < 6:
            _slots.append((None, None))
        (pw0, pp0), (pw1, pp1), (pw2, pp2), (pw3, pp3), (pw4, pp4), (pw5, pp5) = _slots

    elif _s == 6:
        cur_kind = "final"
        _title6, _view6 = phase_views[6][0]
        if _view6 is not None:
            pw0, pp0 = _mk(_view6, "Best · by class",  "class",  0, 300, 300)
            pw1, pp1 = _mk(_view6, "Best · by center", "center", 1, 300, 300)
            cur_titles = ["Best · by class", "Best · by center"]

    return (pw0, pw1, pw2, pw3, pw4, pw5,
            pp0, pp1, pp2, pp3, pp4, pp5,
            cur_titles, cur_kind)


# ═══════════════════════════════════════════════════════════════════════════════
# PREVIEW: references every named widget so a box-select on ANY re-runs this cell
# ═══════════════════════════════════════════════════════════════════════════════
@app.cell
def _slide_preview(mo, pw0, pw1, pw2, pw3, pw4, pw5,
                   pp0, pp1, pp2, pp3, pp4, pp5, get_last_sel, render_range_preview):
    _widgets = [pw0, pw1, pw2, pw3, pw4, pw5]
    _plotted = [pp0, pp1, pp2, pp3, pp4, pp5]
    # Build the full 6-slot panel list (so prefer_index lines up with slot index).
    _panels = [
        (_plotted[_i], getattr(_widgets[_i], "ranges", None) if _widgets[_i] is not None else None)
        for _i in range(6)
    ]
    cur_preview = render_range_preview(_panels, mo, prefer_index=get_last_sel())
    return (cur_preview,)


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD LAYOUT (matches the sketch): header → toggles+legend → images
#   → [preview | table | PPV@90R] → slider at the bottom.
# ═══════════════════════════════════════════════════════════════════════════════
@app.cell
def _dashboard(mo, slide, SLIDE_TITLES, col_sw, proj_sw, feat_sw, view_sw,
               phase_views, PHASE_TABLES, features_root,
               progress_chart, phase_legend_html, example_images_html, md_inline,
               pw0, pw1, pw2, pw3, pw4, pw5, cur_titles, cur_kind, cur_preview):
    _s = int(slide.value)
    _slot_widgets = [pw0, pw1, pw2, pw3, pw4, pw5]
    _active = [w for w in _slot_widgets if w is not None]

    # Pair each active widget with its caption (titles never clip this way, unlike
    # an in-plot title on a narrow panel).
    def _captioned(widgets, titles, font=14):
        _cols = []
        for _w, _t in zip(widgets, titles):
            _cols.append(mo.vstack([
                mo.Html(f"<div style='text-align:center;font-weight:600;"
                        f"font-size:{font}px;margin-bottom:2px;'>{_t}</div>"),
                _w,
            ], gap=0.1))
        return _cols

    _toggles = mo.hstack([
        mo.md("**class ↔ center:**"), col_sw,
        mo.md("&nbsp;&nbsp;**PCA ↔ UMAP:**"), proj_sw,
        mo.md("&nbsp;&nbsp;**proj ↔ pooled:**"), feat_sw,
    ], justify="start", gap=0.4, wrap=True)

    _slider_bar = mo.vstack([
        slide,
        mo.md("<span style='font-size:11px;color:#888;'>Drag the slider to move through the 7 slides.</span>"),
    ])

    _EXPLAIN = {
        1: ("Phase 1 · backbone × loss",
            "**What:** frozen backbones × loss (CE vs SupPro) as feature extractors.",
            "**Conclusion:** GastroNet DINOv2 + SupPro separates by *class* and mixes *centers* "
            "(good); DINOv3 sorts by center. Domain-specific pretraining wins.",
            "⚠️ Phase 1 predates the data-leakage fix, read as *directionally informative*; "
            "Phase 5 has the clean LOCO numbers.", "warn"),
        2: ("Phase 2 · temperature τ",
            "**What:** sweeping the SupPro temperature τ ∈ {0.07, 0.10, 0.30, 0.50}.",
            "**Conclusion:** τ = 0.10 gives the cleanest geometry and best metrics "
            "(consensus-mass 0.26 → 0.70).",
            "⚠️ Still pre-LOCO; treat absolute values as indicative, trends as reliable.", "warn"),
        3: ("Phase 3 · balanced sampling",
            "**What:** forcing a fixed share of positives per batch so SupPro always has positive pairs.",
            "**Conclusion:** 25% generalised best; 5% has too few positives, 50% over-samples and collapses.",
            "⚠️ The 50/50 run hit a perfect 1.0 on validation, a symptom of the patient-level "
            "leakage later fixed with LOCO.", "warn"),
        4: ("Phase 4 · ROI vs random crops",
            "**What:** ROI-guided crops vs random crops at two scales, plus full image.",
            "**Conclusion:** inconclusive: every strategy lands in the same narrow PPV band, "
            "pointing to the head and leakage as the real bottleneck.",
            "⚠️ Pre-LOCO; the flat differences here partly reflect the unresolved leakage.", "warn"),
        5: ("Phase 5 · head × crop",
            "**What:** with **LOCO** + deterministic training, sweeping the **crop scale** "
            "(0.4 → 0.95). The scatter panels below show the feature space at each crop scale. "
            "these are *identical across all classifier heads* (k-NN, SVM, Linear, MLP), since the "
            "head is trained on top of the same frozen features. The head is chosen separately.",
            "**Conclusion:** **k-NN won.** Non-parametric heads read SupPro's clusters directly; "
            "k-NN at crop 0.95 peaks at **PPV@90R = 0.039**, the single biggest win, beating SVM, "
            "Linear, and MLP. Use the **Results table** toggle below to compare heads.",
            "✅ Phase 5 uses leave-one-center-out splits, so validation tracks the hidden server.", "success"),
    }

    # ---- SLIDE 0 : problem & dataset ---------------------------------------
    if _s == 0:
        _hero_html = """
        <div style="background:linear-gradient(135deg,#0b3d5c 0%,#0072B2 100%);
                    border-radius:14px;padding:26px 30px;color:white;margin-bottom:18px;">
          <div style="font-size:32px;font-weight:700;line-height:1.2;margin-bottom:10px;">
            Detecting early neoplasia in Barrett's Esophagus
          </div>
          <div style="font-size:18px;line-height:1.55;opacity:0.95;max-width:1150px;">
            Barrett's Esophagus can progress to esophageal cancer, so patients are screened for
            early <b>neoplasia</b>. The catch: lesions are <b>extremely rare</b> and look a lot like
            normal folds, vessels, and light reflections, so a detector has to be precise without
            drowning clinicians in false alarms.
          </div>
        </div>
        """
        _rare_html = """
        <div style="background:#eef4fb;border-left:5px solid #0072B2;border-radius:10px;
                    padding:16px 20px;margin-bottom:18px;font-size:16px;line-height:1.55;color:#234;">
          <b>The RARE26 challenge.</b> Part of the EndoVis framework, RARE26 benchmarks early-neoplasia
          detection in Barrett's Esophagus under <i>realistic low-prevalence</i> conditions. Last year's
          top methods leaned on heavy model ensembles and still produced too many false positives;
          RARE26 pushes for models that stay accurate <i>and</i> deployable across hospitals. Primary
          metric: <b>PPV@90Recall</b> (precision when catching 90% of lesions).
        </div>
        """
        # Single card holding the "biggest struggle" copy on the left and the
        # imbalance bar chart on the right, so there is no awkward empty gap.
        _struggle_card = """
        <div style="border:1px solid #e3e3e3;border-radius:12px;padding:20px 24px;
                    background:#fafafa;margin-bottom:18px;
                    display:flex;gap:32px;align-items:center;flex-wrap:wrap;">
          <div style="flex:1;min-width:320px;font-size:17px;line-height:1.7;color:#333;">
            <div style="font-size:20px;font-weight:700;color:#b3450f;margin-bottom:8px;">
              The single biggest struggle: class imbalance
            </div>
            <ul style="margin:0;padding-left:20px;">
              <li><b>&asymp; 5%</b> neoplasia in training, <b>&lt; 1%</b> in real practice</li>
              <li><b>19 : 1</b> ratio, so most batches contain almost no positives</li>
              <li>standard models over-predict "healthy", giving <b>too many false alarms</b></li>
            </ul>
          </div>
          <div style="flex:0 0 auto;text-align:center;">
            <div style="display:flex;align-items:flex-end;gap:30px;height:190px;justify-content:center;">
              <div style="text-align:center;">
                <div style="width:96px;height:180px;background:#009E73;border-radius:8px 8px 0 0;"></div>
                <div style="font-size:15px;margin-top:6px;"><b>NDBE</b><br>2937</div>
              </div>
              <div style="text-align:center;">
                <div style="width:96px;height:10px;background:#D55E00;border-radius:8px 8px 0 0;"></div>
                <div style="font-size:15px;margin-top:6px;"><b>neoplasia</b><br>158</div>
              </div>
            </div>
            <div style="font-size:13px;color:#888;margin-top:8px;">&asymp; 5% prevalence &middot; 19:1 ratio</div>
          </div>
        </div>
        """
        _body = mo.vstack([
            mo.Html(_hero_html),
            mo.Html(_rare_html),
            mo.Html(_struggle_card),
            mo.Html("<div style='font-size:19px;font-weight:700;margin:4px 0 6px 0;'>"
                    "What the data looks like: NDBE vs neoplasia</div>"),
            example_images_html(features_root.value, mo, thumb_px=110, n_each=4),
            mo.Html("<div style='font-size:13px;color:#888;margin-top:4px;'>"
                    "Data: RARE26 challenge (CC-BY-NC-SA). Two Dutch centers &middot; "
                    "internal test = EndoVisSub-Barrett (100 images, 5-expert masks).</div>"),
        ], gap=0.3)
        _out = mo.vstack([_body, mo.md("---"), _slider_bar])

    # ---- SLIDES 1..5 : phase results (sketch layout) -----------------------
    elif 1 <= _s <= 5:
        _phase_title, _what, _concl, _disc, _disc_kind = _EXPLAIN[_s]
        _legend = mo.Html(phase_legend_html("center" if col_sw.value else "class"))

        # Header: phase title + What + Conclusion together in ONE block, larger
        # text for presentation.
        _header = mo.Html(
            f"<div style='font-size:17px;line-height:1.5;'>"
            f"<div style='font-size:24px;font-weight:600;margin-bottom:6px;'>{_phase_title}</div>"
            f"<div style='margin-bottom:4px;'>{md_inline(_what)}</div>"
            f"<div>{md_inline(_concl)}</div>"
            f"</div>"
        )

        # Toggles (left)  |  Legend (right)
        _toggle_legend = mo.hstack([
            _toggles, _legend,
        ], justify="space-between", gap=1.0, align="start", widths=[3, 2], wrap=True)

        # Big images row: horizontal, side by side (no wrap), each captioned.
        _images = mo.hstack(_captioned(_active, cur_titles, font=15),
                            justify="start", gap=0.5, wrap=False)

        # Bottom-right panel toggles between the table and the PPV@90R chart.
        if view_sw.value == "Results table":
            _right_panel = mo.vstack([mo.md("**Results**"),
                                      mo.ui.table(PHASE_TABLES[_s], selection=None, pagination=False)])
        else:
            _right_panel = mo.vstack([mo.md("**PPV@90Recall progress**"),
                                      mo.as_html(progress_chart(reveal_up_to=(_s + 2)))])

        # Bottom row: preview | view-toggle + (table OR chart)
        _bottom = mo.hstack([
            mo.vstack([mo.md("**Box-select preview**"), cur_preview]),
            mo.vstack([view_sw, _right_panel]),
        ], justify="start", gap=1.5, align="start", widths=[1, 2], wrap=False)

        _out = mo.vstack([
            _header,
            _toggle_legend,
            _images,
            _bottom,
            _slider_bar,
            # Disclaimer sits BELOW the slider line (may run slightly off-screen).
            mo.callout(mo.md(_disc), kind=_disc_kind),
        ])

    # ---- SLIDE 6 : final --------------------------------------------------
    else:
        _legend = mo.Html(phase_legend_html("both"))
        _concl = mo.Html(
            "<div style='font-size:17px;line-height:1.55;'>"
            "<div style='font-size:26px;font-weight:700;margin-bottom:8px;'>Final configuration &amp; conclusions</div>"
            "<b>Best pipeline:</b> GastroNet DINOv2 · SupPro (τ = 0.10) · 20% balanced sampling · "
            "crop 0.95 · <b>k-NN (k=5)</b> head →"
            "Internal validation peaked at PPV@90R = 0.039. On the hidden test server, our model scored 0.0405 (RARE25) and 0.0223 (RARE26)."
            "<ol style='margin:8px 0 0 0;'>"
            "<li><b>Domain-specific pretraining wins</b>: GastroNet DINOv2 separates by <i>class</i>, not <i>center</i>.</li>"
            "<li><b>The head matters as much as the backbone</b>: non-parametric k-NN/SVM can clssify SupPro's "
            "clusters better than parametric alternatives.</li>"
            "</ol>"
            "<div style='margin-top:8px;color:#a33;'>⚠️ <b>Not yet clinical:</b> 0.0405 ≈ 1 true positive per 25 false alarms.</div>"
            "</div>"
        )

        # RARE25 / RARE26 comparison table (from the report's Table IX).
        _rare_table_html = """
        <div style="font-size:16px;margin-top:6px;">
          <div style="font-weight:600;margin-bottom:6px;">Performance comparison on RARE25 and RARE26</div>
          <table style="border-collapse:collapse;font-size:15px;">
            <tr style="border-bottom:2px solid #333;">
              <th style="text-align:left;padding:6px 16px;">Challenge</th>
              <th style="text-align:left;padding:6px 16px;">Submission</th>
              <th style="padding:6px 16px;">PPV@90R</th>
              <th style="padding:6px 16px;">AUROC</th>
              <th style="padding:6px 16px;">AUPRC</th></tr>
            <tr><td style="padding:6px 16px;">RARE25</td><td style="padding:6px 16px;">Ours</td>
              <td style="text-align:center;"><b>0.0405</b></td><td style="text-align:center;">0.7915</td><td style="text-align:center;"><b>0.5048</b></td></tr>
            <tr style="border-bottom:1px solid #ccc;"><td style="padding:6px 16px;">RARE25</td><td style="padding:6px 16px;">1st place 2025</td>
              <td style="text-align:center;">0.0355</td><td style="text-align:center;"><b>0.9215</b></td><td style="text-align:center;">0.4857</td></tr>
            <tr><td style="padding:6px 16px;">RARE26</td><td style="padding:6px 16px;">Ours</td>
              <td style="text-align:center;"><b>0.0223</b></td><td style="text-align:center;">0.6906</td><td style="text-align:center;">0.2104</td></tr>
            <tr><td style="padding:6px 16px;">RARE26</td><td style="padding:6px 16px;">1st place 2025</td>
              <td style="text-align:center;">0.0152</td><td style="text-align:center;"><b>0.8030</b></td><td style="text-align:center;"><b>0.2937</b></td></tr>
          </table>
          <div style="font-size:13px;color:#666;margin-top:6px;">A single model, beats the 2025 winner on PPV@90R for both challenges which used an ensemble.</div>
        </div>
        """
        _images = mo.hstack(_captioned(_active, cur_titles, font=16),
                            justify="start", gap=1.0, wrap=False)
        _bottom = mo.hstack([
            mo.vstack([mo.md("**Box-select preview**"), cur_preview]),
            mo.vstack([mo.Html(_rare_table_html)]),
        ], justify="start", gap=1.5, align="start", widths=[1, 2], wrap=False)

        _out = mo.vstack([
            _concl,
            mo.Html("<div style='font-size:17px;font-weight:600;margin-top:6px;'>"
                    "Best feature space: same points, by class (left) and center (right):</div>"),
            _images,
            _legend,
            _bottom,
            _slider_bar,
        ])

    _out
    return

if __name__ == "__main__":
    app.run()