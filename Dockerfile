FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip git ca-certificates \
    && python3 -m pip install --no-cache-dir --upgrade pip==23.3.2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/BIAM
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python3", "biam_main.py", "--config", "configs/biam_default.yaml"]
