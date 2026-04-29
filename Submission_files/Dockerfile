FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN python -m pip install --no-cache-dir torch torchvision timm pillow

COPY Submission_files/predict.py /app/predict.py
COPY model.py /app/model.py
COPY model.pt /app/model.pt

# Run prediction script on container start
ENTRYPOINT ["python", "/app/predict.py"]
