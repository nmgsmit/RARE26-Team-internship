# SupCon-style feature evaluation

This folder evaluates learned representations from SupPro/SupMin/Baseline
checkpoints. The current workflow is:

1. Use `runs.csv` plus the checkpoint folders to extract features.
2. Share or download the resulting `features_out/` directory.
3. Open `notebook.py` for interactive UMAP/PCA exploration.
4. Use `figures_results.py` when you want publication-style comparison
  figures.

**Note: Python 3.10 or higher (e.g., Python 3.12) is required** to run the latest versions of the `marimo` notebook package.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch on Apple Silicon will use MPS automatically when the scripts are run
with `--device mps`. CPU is the default fallback.

## 1. Understand `runs.csv`

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

## 2. Prepare the local folder layout

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

## 3. Extract features

You dont have to extract the features manually, this can also be done by downloading it from Onedrive in our shared Rare26 team intership folder. Saves a lot of time!


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


## 4. Open the analysis tools

Interactive notebook:

```bash
marimo run notebook.py
```

Paper figure generator:

```bash
python figures_results.py
```

`notebook.py` is the main analysis surface now. It lets you pick checkpoints
and feature types, tune UMAP parameters, and click points to preview the source
images. `figures_results.py` is the non-interactive path for generating the
projection figures used in reports or papers.

`notebook_data.py` and `notebook_static.py` are kept only as legacy references
and are not part of the current workflow.

## Nice-to-know points

- `extract_features.py` can process multiple checkpoints from `runs.csv` in one
  run.
- The repo keeps both pooled and projection features, so you can compare the
  encoder representation and the projected representation.
- The train split expects `centerN/{ndbe,neo}/` folders, while the test split
  expects a flat image directory with class labels encoded in the filenames.
- If you change the checkpoint naming scheme, make sure `runs.csv` matches the
  new filenames exactly.
- `figures_results.py` follows the same `features_out/` layout as the notebook,
  so it can be run directly after feature extraction.

## File map

```text
extract_features.py          Feature extraction from checkpoints
notebook.py                  Interactive UMAP notebook
figures_results.py           Publication-style comparison figures

notebook_data.py             Legacy raw-dataset notebook
notebook_static.py           Legacy static comparison notebook

model.py                     Model definition and checkpoint loading
data.py                      Dataset / dataloader helpers
runs.csv                     Run manifest for feature extraction
```