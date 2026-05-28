"""
Stage 0: raw image / dataset analysis (marimo).

Run with:
    marimo run notebook_data.py

Looks at the data BEFORE any model touches it.
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _imports():
    import re
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from PIL import Image

    return Image, Path, go, mo, np, pd, px, re


@app.cell
def _intro(mo):
    mo.md(r"""
    # Raw dataset analysis

    Before any model sees these images, we want to understand what's in
    the data:

    - **Where do the images come from?** Per-center counts, test-set
      patient counts, class balance.
    - **What do they look like?** Resolution and aspect-ratio
      distributions — useful to spot scanner / cropping differences.
    - **Color / intensity domain shift.** Mean RGB statistics per class
      and per center — useful to see whether the test-set distribution
      drifts away from training.
    - **Visual sanity check.** A small random grid of images per
      (center, class).
    - **Raw pixel structure.** PCA and UMAP on flattened, downsampled
      raw pixels. The model's job is partly to *undo* whatever obvious
      structure exists at the raw-pixel level (e.g. brightness, scope
      edge) and replace it with class-relevant structure.
    """)
    return


@app.cell
def _path_inputs(mo):
    center_1_dir = mo.ui.text(
        value="../../data/center_1",
        label="Center 1 data directory (center_1/{ndbe,neo}/)",
        full_width=True,
    )
    center_2_dir = mo.ui.text(
        value="../../data/center_2",
        label="Center 2 data directory (center_2/{ndbe,neo}/)",
        full_width=True,
    )
    test_dir = mo.ui.text(
        value="../../EVC_Barretts_FullSet 2/images",
        label="Barretts images directory (flat, filenames like pat01_im1_NDBT.png)",
        full_width=True,
    )
    mo.vstack([center_1_dir, center_2_dir, test_dir])
    return center_1_dir, center_2_dir, test_dir


@app.cell
def _scan_inputs(Image, Path, mo, pd, re, center_1_dir, center_2_dir, test_dir):
    """
    Walk both directory trees, parse out (path, label, center, patient,
    width, height) for every image. Resolution probing uses PIL's lazy
    open() so we don't decode pixels here.
    """

    def _scan_train(root: Path):
        _rows = []
        if not root.exists():
            return _rows
        if (root / "ndbe").is_dir() and (root / "neo").is_dir():
            center_dirs = [root]
        else:
            center_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name.startswith("center")]
        for center_dir in center_dirs:
            for class_name in ("ndbe", "neo"):
                cls_dir = center_dir / class_name
                if not cls_dir.is_dir():
                    continue
                for img_path in sorted(cls_dir.iterdir()):
                    if not img_path.is_file():
                        continue
                    if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                        continue
                    try:
                        with Image.open(img_path) as im:
                            w, h = im.size
                            mode = im.mode
                    except Exception:
                        continue
                    _rows.append({
                        "path": str(img_path),
                        "split": "train",
                        "center": center_dir.name,
                        "class_name": class_name,
                        "label": 0 if class_name == "ndbe" else 1,
                        "patient": None,
                        "width": w,
                        "height": h,
                        "mode": mode,
                    })
        return _rows

    _suffix_to_class = {"NDBT": "ndbe", "ACHD": "neo"}
    _patient_re = re.compile(r"^(pat\d+)_", re.IGNORECASE)

    def _scan_test(root: Path):
        _rows = []
        if not root.exists():
            return _rows
        for img_path in sorted(root.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                continue
            stem_parts = img_path.stem.rsplit("_", 1)
            if len(stem_parts) != 2:
                continue
            suffix = stem_parts[1].upper()
            cls = _suffix_to_class.get(suffix)
            if cls is None:
                continue
            m = _patient_re.match(img_path.name)
            patient = m.group(1).lower() if m else None
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                    mode = im.mode
            except Exception:
                continue
            _rows.append({
                "path": str(img_path),
                "split": "test",
                "center": "test",
                "class_name": cls,
                "label": 0 if cls == "ndbe" else 1,
                "patient": patient,
                "width": w,
                "height": h,
                "mode": mode,
            })
        return _rows

    _train_rows = []
    _train_rows.extend(_scan_train(Path(center_1_dir.value).expanduser()))
    _train_rows.extend(_scan_train(Path(center_2_dir.value).expanduser()))
    _test_rows = _scan_test(Path(test_dir.value).expanduser())
    df = pd.DataFrame(_train_rows + _test_rows)

    mo.stop(
        len(df) == 0,
        mo.md("⚠️ No images found. Check the paths above.")
    )

    df["aspect"] = df["width"] / df["height"]
    df["megapixels"] = (df["width"] * df["height"]) / 1e6

    mo.md(
        f"Scanned **{len(df)}** images: "
        f"{(df['split'] == 'train').sum()} train, "
        f"{(df['split'] == 'test').sum()} test. "
        f"Centers in train: {sorted(df.loc[df.split == 'train', 'center'].unique())}."
    )
    return (df,)


@app.cell
def _counts_table(df, mo, pd):
    """
    Class counts per (center, class) for training, plus the test set.
    """
    _train = df[df["split"] == "train"]
    _test = df[df["split"] == "test"]

    _train_counts = (
        _train.groupby(["center", "class_name"]).size().unstack(fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    _train_counts["total"] = _train_counts.get("ndbe", 0) + _train_counts.get("neo", 0)
    _train_counts["neo_pct"] = (
        _train_counts.get("neo", 0) / _train_counts["total"] * 100
    ).round(1)

    _test_counts = (
        _test.groupby("class_name").size().rename("count").reset_index()
    )
    _test_n_patients = _test["patient"].nunique()
    _test_n_patients_per_class = (
        _test.groupby("class_name")["patient"].nunique().rename("unique_patients").reset_index()
    )
    _test_table = _test_counts.merge(_test_n_patients_per_class, on="class_name")

    if not _train_counts.empty:
        _grand = pd.DataFrame([{
            "center": "TOTAL",
            "ndbe": _train_counts.get("ndbe", pd.Series(dtype=int)).sum(),
            "neo": _train_counts.get("neo", pd.Series(dtype=int)).sum(),
            "total": _train_counts["total"].sum(),
            "neo_pct": round(
                _train_counts.get("neo", pd.Series(dtype=int)).sum()
                / max(_train_counts["total"].sum(), 1) * 100,
                1,
            ),
        }])
        _train_counts = pd.concat([_train_counts, _grand], ignore_index=True)

    _out = mo.vstack([
        mo.md("## Class counts per center (training)"),
        mo.ui.table(_train_counts, selection=None),
        mo.md(
            f"## Test set\n"
            f"**{len(_test)} images** from **{_test_n_patients} unique patients**."
        ),
        mo.ui.table(_test_table, selection=None),
    ])
    _out
    return


@app.cell
def _patients_per_class_chart(df, mo, px):
    _test = df[df["split"] == "test"]
    mo.stop(len(_test) == 0, mo.md("*(no test data)*"))

    _pat_per_class = (
        _test.groupby("class_name")["patient"].nunique().reset_index(name="patients")
    )
    _fig_pat = px.bar(
        _pat_per_class,
        x="class_name", y="patients",
        title="Test set: unique patients per class",
        text="patients",
        color="class_name",
        color_discrete_map={"ndbe": "#1f77b4", "neo": "#d62728"},
    )
    _fig_pat.update_layout(width=450, height=350, showlegend=False)

    _imgs_per_pat = (
        _test.groupby(["patient", "class_name"]).size().reset_index(name="n_images")
    )
    _fig_imgs = px.histogram(
        _imgs_per_pat, x="n_images", color="class_name",
        nbins=int(_imgs_per_pat["n_images"].max()) if len(_imgs_per_pat) else 1,
        title="Test set: images per patient",
        color_discrete_map={"ndbe": "#1f77b4", "neo": "#d62728"},
    )
    _fig_imgs.update_layout(width=500, height=350, barmode="overlay")
    _fig_imgs.update_traces(opacity=0.7)

    mo.hstack([mo.ui.plotly(_fig_pat), mo.ui.plotly(_fig_imgs)])
    return


@app.cell
def _aspect_size(df, mo, px):
    _df_plot = df.assign(group=df["split"] + " / " + df["center"])

    _fig_aspect = px.histogram(
        _df_plot, x="aspect", color="group", nbins=40,
        title="Aspect ratio (width / height)",
        opacity=0.6, barmode="overlay",
    )
    _fig_aspect.update_layout(width=550, height=400)

    _fig_size = px.scatter(
        _df_plot, x="width", y="height", color="group",
        title="Image resolution (each dot = one image)",
        opacity=0.5,
        hover_data=["path"],
    )
    _fig_size.update_layout(width=550, height=400)

    mo.vstack([
        mo.md(
            "## Image geometry\n"
            "If aspect ratios cluster differently between centers / test, "
            "the resize-to-square transform is silently changing field of "
            "view between groups."
        ),
        mo.hstack([mo.ui.plotly(_fig_aspect), mo.ui.plotly(_fig_size)]),
    ])
    return


@app.cell
def _color_stats_controls(df, mo):
    _n_total = len(df)
    color_sample = mo.ui.slider(
        start=1, stop=max(2, min(2000, _n_total)), step=1,
        value=min(500, max(1, _n_total)),
        label=f"Color stats: random sample size (out of {_n_total})",
        full_width=True,
    )
    color_sample
    return (color_sample,)


@app.cell
def _color_stats(Image, color_sample, df, mo, np, pd, px):
    _rng = np.random.default_rng(0)
    _n = min(color_sample.value, len(df))
    _idxs = _rng.choice(len(df), size=_n, replace=False)

    _rows = []
    for _i in _idxs:
        _row = df.iloc[int(_i)]
        try:
            with Image.open(_row["path"]) as _im:
                _im = _im.convert("RGB")
                _im.thumbnail((96, 96))
                _arr = np.asarray(_im, dtype=np.float32) / 255.0
        except Exception:
            continue
        _r, _g, _b = _arr.mean(axis=(0, 1)).tolist()
        _intensity = float(_arr.mean())
        _rows.append({
            "split": _row["split"],
            "center": _row["center"],
            "class_name": _row["class_name"],
            "mean_R": _r, "mean_G": _g, "mean_B": _b,
            "mean_intensity": _intensity,
            "path": _row["path"],
        })

    color_df = pd.DataFrame(_rows)
    color_df["group"] = color_df["split"] + " / " + color_df["center"]

    _fig_int = px.box(
        color_df, x="group", y="mean_intensity", color="class_name",
        title="Mean image intensity by group and class",
        color_discrete_map={"ndbe": "#1f77b4", "neo": "#d62728"},
    )
    _fig_int.update_layout(width=700, height=400)

    _fig_rgb = px.scatter(
        color_df, x="mean_R", y="mean_G", color="group",
        symbol="class_name",
        title="Mean color: R vs G channel",
        hover_data=["path", "mean_B"],
        opacity=0.7,
    )
    _fig_rgb.update_layout(width=600, height=500)

    mo.vstack([
        mo.md(
            "## Color & intensity\n"
            "If the test set's mean RGB sits clearly outside the training "
            "centers, that's a low-level domain shift. The augmentation "
            "pipeline (`ColorJitter`) helps but won't fully bridge a large "
            "gap."
        ),
        mo.hstack([mo.ui.plotly(_fig_int), mo.ui.plotly(_fig_rgb)]),
    ])
    return


@app.cell
def _sample_grid_controls(df, mo):
    _groups = sorted((df["split"] + " / " + df["center"] + " / " + df["class_name"]).unique())
    group_pick = mo.ui.dropdown(options=_groups, value=_groups[0] if _groups else None, label="Group to view")
    n_imgs = mo.ui.slider(start=4, stop=24, step=4, value=8, label="How many images")
    img_w = mo.ui.slider(start=120, stop=300, step=20, value=180, label="Image width (px)")
    seed_pick = mo.ui.number(value=0, start=0, stop=10_000, label="Random seed")
    mo.hstack([group_pick, n_imgs, img_w, seed_pick])
    return group_pick, img_w, n_imgs, seed_pick


@app.cell
def _sample_grid(df, group_pick, img_w, mo, n_imgs, np, seed_pick):
    mo.stop(not group_pick.value, mo.md("*Select a group above*"))

    _parts = group_pick.value.split(" / ")
    mo.stop(len(_parts) != 3, mo.md("*pick a group above*"))

    _sp, _ct, _cl = _parts
    _sub = df[(df["split"] == _sp) & (df["center"] == _ct) & (df["class_name"] == _cl)]
    mo.stop(len(_sub) == 0, mo.md(f"*no images for {group_pick.value}*"))

    _rng = np.random.default_rng(int(seed_pick.value))
    _pick = _rng.choice(len(_sub), size=min(int(n_imgs.value), len(_sub)), replace=False)
    _items = []
    for _i in _pick:
        _row = _sub.iloc[int(_i)]
        _cap = f"`{_row['class_name']}` · `{_row['center']}`"
        if _row["patient"]:
            _cap += f" · `{_row['patient']}`"
        _items.append(mo.vstack([
            mo.image(str(_row["path"]), width=int(img_w.value)),
            mo.md(_cap),
        ]))
    mo.vstack([
        mo.md(f"### Random samples — {group_pick.value} ({len(_sub)} total)"),
        mo.hstack(_items, wrap=True),
    ])
    return


@app.cell
def _pixel_pca_umap_controls(df, mo):
    _n_total = len(df)
    pca_sample = mo.ui.slider(
        start=1, stop=max(2, min(3000, _n_total)), step=1,
        value=min(1000, max(1, _n_total)),
        label=f"PCA / UMAP sample size (out of {_n_total})",
        full_width=True,
    )
    thumb_size = mo.ui.dropdown(
        options=["32", "48", "64"], value="48",
        label="Thumbnail edge (px) for raw-pixel features",
    )
    method = mo.ui.dropdown(
        options=["PCA", "UMAP"], value="PCA",
        label="Method",
    )
    color_by = mo.ui.dropdown(
        options=["class_name", "center", "split", "split+class"],
        value="split+class",
        label="Color points by",
    )
    seed_p = mo.ui.number(value=42, start=0, stop=10_000, label="Seed")
    mo.vstack([
        mo.md(
            "## PCA / UMAP on raw pixels\n"
            "We resize each image to a small square, flatten to a vector, "
            "and project to 2D. This shows what structure exists *before* "
            "any learned representation. Compare with the feature-space "
            "notebook to see how much the model has reorganized things."
        ),
        mo.hstack([method, color_by]),
        mo.hstack([thumb_size, seed_p, pca_sample]),
    ])
    return color_by, method, pca_sample, seed_p, thumb_size


@app.cell
def _pixel_features(Image, df, mo, np, pca_sample, seed_p, thumb_size):
    def _compute(n: int, edge: int, seed: int, paths_tuple, splits_tuple,
                 centers_tuple, classes_tuple, patients_tuple):
        _rng = np.random.default_rng(seed)
        n = min(n, len(paths_tuple))
        _idxs = _rng.choice(len(paths_tuple), size=n, replace=False)

        _feats = np.zeros((n, edge * edge * 3), dtype=np.float32)
        _meta_rows = []
        _keep = []

        # Safe fallback for PIL 10+
        _resample_method = getattr(Image, "Resampling", Image).BILINEAR

        for _k, _i in enumerate(_idxs):
            try:
                with Image.open(paths_tuple[_i]) as _im:
                    _im = _im.convert("RGB").resize((edge, edge), _resample_method)
                    _arr = np.asarray(_im, dtype=np.float32) / 255.0
            except Exception:
                continue
            _feats[len(_keep)] = _arr.reshape(-1)
            _meta_rows.append({
                "path": paths_tuple[_i],
                "split": splits_tuple[_i],
                "center": centers_tuple[_i],
                "class_name": classes_tuple[_i],
                "patient": patients_tuple[_i],
            })
            _keep.append(len(_keep))
        _feats = _feats[:len(_keep)]
        return _feats, _meta_rows

    _paths_t = tuple(df["path"].tolist())
    _splits_t = tuple(df["split"].tolist())
    _centers_t = tuple(df["center"].tolist())
    _classes_t = tuple(df["class_name"].tolist())
    _patients_t = tuple(df["patient"].fillna("").tolist())

    feats, meta_rows = _compute(
        int(pca_sample.value),
        int(thumb_size.value),
        int(seed_p.value),
        _paths_t, _splits_t, _centers_t, _classes_t, _patients_t,
    )
    mo.md(f"Built raw-pixel feature matrix: shape `{feats.shape}`.")
    return feats, meta_rows


@app.cell
def _pixel_projection(feats, method, mo, seed_p):
    def _pca(arr, seed):
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=int(seed)).fit_transform(arr)

    def _umap(arr, seed):
        import umap
        return umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1,
            metric="euclidean", random_state=int(seed),
        ).fit_transform(arr)

    mo.stop(len(feats) == 0, mo.md("⚠️ No features computed."))

    coords = _pca(feats, seed_p.value) if method.value == "PCA" else _umap(feats, seed_p.value)
    return (coords,)


@app.cell
def _pixel_scatter(color_by, coords, go, meta_rows, method, mo, pd):
    meta_df = pd.DataFrame(meta_rows)
    if color_by.value == "split+class":
        meta_df["color_key"] = meta_df["split"] + " / " + meta_df["class_name"]
    else:
        meta_df["color_key"] = meta_df[color_by.value]

    _palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    _fig = go.Figure()
    for _k, _key in enumerate(sorted(meta_df["color_key"].unique())):
        _m = meta_df["color_key"] == _key
        _is_test = meta_df.loc[_m, "split"].iloc[0] == "test" if _m.any() else False
        _fig.add_trace(go.Scatter(
            x=coords[_m.values, 0], y=coords[_m.values, 1],
            mode="markers",
            name=_key,
            marker=dict(
                size=8 if _is_test else 6,
                symbol="diamond" if _is_test else "circle",
                color=_palette[_k % len(_palette)],
                opacity=0.75,
                line=dict(width=0.5, color="white"),
            ),
            text=[
                f"{_r['class_name']} · {_r['center']} · {_r['split']}"
                + (f" · {_r['patient']}" if _r["patient"] else "")
                + f"<br>{_r['path']}"
                for _, _r in meta_df[_m].iterrows()
            ],
            customdata=meta_df.loc[_m, "path"].values,
            hovertemplate="%{text}<extra></extra>",
        ))

    _fig.update_layout(
        width=900, height=650,
        title=f"{method.value} on raw pixels (colored by {color_by.value})",
        xaxis_title=f"{method.value} 1", yaxis_title=f"{method.value} 2",
        plot_bgcolor="white",
        legend=dict(x=1.02, y=1, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=40, r=180, t=50, b=40),
    )
    _fig.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    _fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False)

    plot = mo.ui.plotly(_fig)
    plot
    return meta_df, plot


@app.cell
def _selection_viewer(meta_df, mo, plot):
    _sel = plot.value
    if not _sel or "points" not in _sel or len(_sel["points"]) == 0:
        _view = mo.md(
            "*Box-select or lasso-select points on the scatter to see "
            "their images here.*"
        )
    else:
        _chosen_paths = []
        for _p in _sel["points"]:
            _cd = _p.get("customdata")
            if isinstance(_cd, list) and _cd:
                _chosen_paths.append(_cd[0])
            elif isinstance(_cd, str):
                _chosen_paths.append(_cd)
        _chosen_paths = _chosen_paths[:16]
        _items = []
        for _path in _chosen_paths:
            _row = meta_df[meta_df["path"] == _path]
            if len(_row) == 0:
                continue
            _row = _row.iloc[0]
            _cap = f"`{_row['class_name']}` · `{_row['center']}` · `{_row['split']}`"
            _items.append(mo.vstack([
                mo.image(_path, width=160),
                mo.md(_cap),
            ]))
        _view = mo.vstack([
            mo.md(f"### Selected samples ({len(_chosen_paths)})"),
            mo.hstack(_items, wrap=True),
        ])
    _view
    return


@app.cell
def _what_to_look_for(mo):
    mo.md(r"""
    ## What to look for

    - **Test set away from training centers in raw-pixel space?**
      Likely a domain shift in capture conditions (scope, lighting,
      processor). The model has to bridge this; it's fine if features
      do but raw pixels don't.
    - **One class neatly separated already in raw pixels?** Be a bit
      suspicious — there may be a non-anatomical confound (e.g.
      NDBT/ACHD images coming from different scopes). Check by coloring
      by center.
    - **Patient counts in the test set.** With only ~10–30 patients,
      per-image AUC is optimistic; consider also reporting per-patient
      accuracy in the feature notebook.
    - **Aspect ratios.** If they differ a lot per group, the
      square-resize transform is changing the field of view in a
      group-dependent way.
    """)
    return


if __name__ == "__main__":
    app.run()
