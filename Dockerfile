ARG BASE_IMAGE=nvidia/cuda:12.1.1-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    ESRGAN_MODEL_DIR=/app/.cache/realesrgan

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential \
    pkg-config \
    libcairo2-dev \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi8 \
    wget \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=
RUN pip install --upgrade pip \
    && if [ -n "$PIP_EXTRA_INDEX_URL" ]; then \
         pip install -r /app/requirements.txt --index-url "$PIP_INDEX_URL" --extra-index-url "$PIP_EXTRA_INDEX_URL"; \
       else \
         pip install -r /app/requirements.txt --index-url "$PIP_INDEX_URL"; \
       fi \
    && pip install spandrel

# Download Real-ESRGAN model weights
RUN mkdir -p /app/.cache/realesrgan \
    && wget -q -O /app/.cache/realesrgan/RealESRGAN_x4plus.pth \
       https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

COPY . /app

RUN mkdir -p /app/outputs /app/uploads /app/.cache/huggingface

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
