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

- Input images: `IMAGE_DIR = "/data/test"`
- Output predictions: `OUTPUT_DIR = "/output"`
- Model weights: `MODEL_PATH = "/app/model.pt"`

`predict.py` loads `Model` with strict weights loading.
So your `/app/model.pt` must match the architecture in [model.py](model.py).
For this branch, that means the Gastronet setup from [main_gastronet.sh](main_gastronet.sh):
`vit_base_patch14_reg4_dinov2` with `input_size=336`.

---

## 2. Put your best trained checkpoint in place

Training saves checkpoints from [train.py](train.py), typically under:
`checkpoints/<experiment-id>`

Pick your best checkpoint and copy/rename it to:

- `model.pt` in the repo root

Example from repo root:

```bash
cp "checkpoints/<experiment-id>.pt" "model.pt"
```

On Windows PowerShell:

```powershell
Copy-Item "checkpoints\<experiment-id>.pt" "model.pt"
```

---

## 3. Build the Docker image

From repo root:

```bash
docker build -t team-internship:latest -f Submission_files/Dockerfile .
```

This creates a self-contained image with:
- `predict.py`
- `model.py`
- `model.pt`

---

## 4. IGNORE - Test locally before exporting - NOT WORKING CURRENTLY

Create local folders:
- `./local_data` with `.png` images
- `./local_output` for predictions

Run (Linux/macOS shell):

```bash
docker run --rm -v "$(pwd)/data/EVC_Barretts_FullSet/images:/data/test" -v "$(pwd)/local_output:/output" team-internship:latest
```

Run (Windows PowerShell):

```powershell
docker run --rm -v "${PWD}\data\EVC_Barretts_FullSet\images:/data/test" -v "${PWD}\local_output:/output" team-internship:latest
```

Expected behavior:
- It reads test images from `/data/test`
- It writes predictions to `/output`

---

## 5. Export image to `.tar` for submission

```bash
docker save -o RARE26-submission.tar team-internship:latest
```

You then submit `RARE26-submission.tar`.

---

## 6. Challenge server endpoints

To be added.

---


## 7. Common failure checks

- `model.pt` missing from the repo root before build.
- `model.pt` incompatible with `Model` in [model.py](model.py).
- Input/output paths changed from `/data` and `/output`.
- Built from wrong Docker context.

---

Good practice: test the container locally end-to-end before every submission.
