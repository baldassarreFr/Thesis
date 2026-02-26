# Use NVIDIA's PyTorch image for Jetson AGX Orin
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libhdf5-dev \
    hdf5-tools \
    build-essential \
    zlib1g-dev \
    libjpeg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies via pip
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt

# Set the working directory
WORKDIR /dino