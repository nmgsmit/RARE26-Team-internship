FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# No extra dependencies needed
COPY predict.py /app/predict.py

# Run prediction script on container start
ENTRYPOINT ["python", "/app/predict.py"]
