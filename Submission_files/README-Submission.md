# Challenge Submission Guide

This guide explains how to package your trained model into a self-contained Docker image, export it to a `.tar` file, and submit it to the challenge server.

It is based on:
- [Submission_files/predict.py](Submission_files/predict.py)
- [model.py](model.py)
- [Submission_files/Dockerfile](Submission_files/Dockerfile)
- [train.py](train.py)

---

## 1. What the server expects

Your container runs `predict.py` as entrypoint (see [Submission_files/Dockerfile](Submission_files/Dockerfile)).

Inside the container, paths are fixed:

- Input images: `IMAGE_DIR = "/data"`
- Output predictions: `OUTPUT_DIR = "/output"`
- Model weights: `MODEL_PATH = "/app/model.pt"`

`predict.py` loads `Model` with strict weights loading.
So your `/app/model.pt` must match the architecture in [model.py](model.py).

---

## 2. Put your best trained checkpoint in place

Training saves checkpoints from [train.py](train.py), typically under:
`checkpoints/<experiment-id>`

Pick your best checkpoint and copy/rename it to:

- `Submission_files/model.pt`

Example from repo root:

```bash
cp "checkpoints/<experiment-id>.pt" "Submission_files/model.pt"
```

On Windows PowerShell:

```powershell
Copy-Item "checkpoints\<experiment-id>.pt" "Submission_files\model.pt"
```

---

## 3. Build the Docker image

From repo root:

```bash
docker build -t nncv-submission:latest -f Submission_files/Dockerfile Submission_files
```

This creates a self-contained image with:
- `predict.py`
- `model.py` (copied by Dockerfile from the build context setup you use)
- `model.pt`

---

## 4. Test locally before exporting

Create local folders:
- `./local_data` with `.png` images
- `./local_output` for predictions

Run (Linux/macOS shell):

```bash
docker run --rm \
  -v "$(pwd)/local_data:/data" \
  -v "$(pwd)/local_output:/output" \
  nncv-submission:latest
```

Run (Windows PowerShell):

```powershell
docker run --rm `
  -v "${PWD}\local_data:/data" `
  -v "${PWD}\local_output:/output" `
  nncv-submission:latest
```

Expected behavior:
- It reads all `*.png` from `/data`
- It writes predictions to `/output`

---

## 5. Export image to `.tar` for submission

```bash
docker save -o nncv_submission.tar nncv-submission:latest
```

You then submit `nncv_submission.tar`.

---

## 6. Challenge server endpoints

To be added.

---

## 7. Recommended workflow per benchmark

- Keep one stable baseline image/tag.
- Create separate model variants per benchmark.
- Export each as a separate tar, for example:
  - `submission_baseline_peak.tar`
  - `submission_robustness.tar`
  - `submission_efficiency.tar`
  - `submission_ood.tar`

Example:

```bash
docker build -t nncv-submission:efficiency -f Submission_files/Dockerfile Submission_files
docker save -o submission_efficiency.tar nncv-submission:efficiency
```

---

## 8. Common failure checks

- `model.pt` missing from `Submission_files/` before build.
- `model.pt` incompatible with `Model` in [model.py](model.py).
- Input/output paths changed from `/data` and `/output`.
- Built from wrong Docker context.

---

Good practice: test the container locally end-to-end before every submission.
