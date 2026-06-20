FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libglib2.0-0 \
    libgl1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /model /csv /data/images

COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -r requirements.txt

COPY main.py main.py
COPY service service
COPY scripts scripts
COPY config.example.yaml config.example.yaml
# Default config baked in; mount your own ./config.yaml over it to customise.
RUN cp config.example.yaml config.yaml

EXPOSE 8000

CMD ["python3", "main.py"]
