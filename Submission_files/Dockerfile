FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# scikit-learn is pinned so fitted KNN/SVM estimators unpickle without
# cross-version drift. predict.py passes weights_only=False to torch.load
# to handle the PyTorch 2.6+ default tightening for sklearn pickles.
RUN python -m pip install --no-cache-dir \
        torch torchvision timm pillow numpy \
        "scikit-learn>=1.4,<1.7"

COPY Submission_files/predict.py /app/predict.py
COPY model.py /app/model.py
COPY model.pt /app/model.pt

ENTRYPOINT ["python", "/app/predict.py"]
