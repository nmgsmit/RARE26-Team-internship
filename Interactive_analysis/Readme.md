# SupCon-style feature evaluation

This repo evaluates learned representations from SupPro/SupMin/Baseline
checkpoints. The intended flow is:

1. Inspect the raw dataset with `notebook_dataexploration.py`.
2. Use `runs.csv` plus the checkpoint folders to extract features.
3. Share or download the resulting `features_out/` directory.
4. Run the interactive and static notebooks for UMAP-style exploration.

**Note: Python 3.10 or higher (e.g., Python 3.12) is required** to run the latest versions of the `marimo` notebook package.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch on Apple Silicon will use MPS automatically when the scripts are run
with `--device mps`. CPU is the default fallback.

## 1. Explore the raw data first

Before touching the model checkpoints, open the raw dataset notebook:

```bash
marimo run notebook_dataexploration.py
```

This notebook is meant to inspect the raw images and answer questions such as:

- how many images are in each split and class
- how the training centers are distributed
- whether the image sizes, aspect ratios, or color statistics differ across centers
- what the raw images look like before any model sees them

If a OneDrive link is available for the shared raw data or shared features,
add it here and point people to the download location, for example:

- OneDrive download link: [paste link here](PASTE_LINK_HERE)

If not, the repo expects the raw inputs to live in the local paths configured
in `notebook_dataexploration.py` and `extract_features.py`.

## 2. Understand `runs.csv`

`runs.csv` is the manifest that connects an experiment to the checkpoint file
that should be evaluated. Each row is one run, and the important columns are:

- `experiment_id`: the folder name used for outputs under `features_out/`
- `model finetune`: the checkpoint filename to load from `Checkpoints/`
- `phase`: the training stage used to create that checkpoint
- `backbone`, `temp`, `backbone_lr`, `proj_lr`, `class_lr`, `batch_size`,
  `loss`, `Head type`, `Crop type`, `notes`: metadata for tracking runs

When you add new checkpoints, update `runs.csv` so the extraction script knows
what to load. The script uses `experiment_id` plus the dataset tag to name the
output folder, for example `features_out/P0_Base_ResNet50_1e-3_t1__evc_test/`.

## 3. Prepare the local folder layout

If you are regenerating features instead of downloading the shared `features_out`, ensure your files are structured correctly. Since this analysis folder sits inside the repository, it expects the external data folders to be located two levels up (../../):

```text
Workspace Root/
├── data/
│   ├── center_1/
│   │   ├── ndbe/
│   │   └── neo/
│   └── center_2/
│       ├── ndbe/
│       └── neo/
├── EVC_Barretts_FullSet 2/
│   └── images/
│       <flat test images with labels in the filenames>
│
└── RARE26-Team-internship/
    └── Interactive_analysis/      <-- (You run the scripts from here)
        ├── Checkpoints/
        │   <your checkpoint .pt files>
        ├── runs.csv
        └── extract_features.py
```

The current extraction script expects the checkpoint root to be `Checkpoints/`
and the output root to be `features_out/`. If your folders live somewhere else,
update the configuration block at the bottom of `extract_features.py`.

## 4. Extract features

```bash
python extract_features.py
```

The script reads `runs.csv`, resolves each checkpoint under `Checkpoints/`, and
extracts features for every dataset listed in the `DATASETS` block. It writes
one folder per experiment and dataset, for example:

- `features_out/<experiment_id>__evc_test/`
- `features_out/<experiment_id>__train_all/`

Each output folder contains the pooled features, projection features, labels,
image paths, deployed logits, deployed probabilities, and a `meta.json` file.

If you already received `features_out/` from OneDrive, you can skip this step
and use the shared folder directly.


## 5. Open the notebooks

Interactive notebook:

```bash
marimo run notebook.py
```

Static notebook:

```bash
marimo run notebook_static.py
```

The interactive notebook is the one to use for day-to-day exploration. It lets
you pick checkpoints and feature types, tune UMAP parameters, and click points
to preview the source images. The static notebook is useful if you want a more
shareable, less reactive version of the same analysis.

## Nice-to-know points

- `notebook_dataexploration.py` is the best place to sanity-check the raw
  dataset before model evaluation.
- `extract_features.py` can process multiple checkpoints from `runs.csv` in one
  run.
- The repo keeps both pooled and projection features, so you can compare the
  encoder representation and the projected representation.
- The train split expects `centerN/{ndbe,neo}/` folders, while the test split
  expects a flat image directory with class labels encoded in the filenames.
- If you change the checkpoint naming scheme, make sure `runs.csv` matches the
  new filenames exactly.

## File map

```text
notebook_dataexploration.py  Raw dataset exploration
extract_features.py          Feature extraction from checkpoints
notebook.py                  Interactive UMAP notebook
notebook_static.py           Static UMAP notebook

model.py                     Model definition and checkpoint loading
data.py                      Dataset / dataloader helpers
runs.csv                     Run manifest for feature extraction
```