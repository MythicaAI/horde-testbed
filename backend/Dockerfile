FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libjpeg-dev \
    zlib1g-dev \
    libglx0 \
    libegl1 \
    libgl1-mesa-dev \
    libosmesa6-dev \
    libegl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install fastapi uvicorn numpy torch torchvision pillow scikit-image moderngl OpenEXR piq pandas torch_dct tensorboard matplotlib

EXPOSE 8000 6006

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]
