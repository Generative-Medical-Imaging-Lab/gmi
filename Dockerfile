# FROM python:3.12-slim
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg git build-essential libglib2.0-0 libsm6 libxrender1 libxext6 && rm -rf /var/lib/apt/lists/*

RUN ln -sf python3 /usr/bin/python && ln -sf pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Create gmi_base directory
RUN mkdir -p /gmi_base

# Set working directory
WORKDIR /gmi_base

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /gmi_base/
RUN pip install -r requirements.txt

# Copy GMI source code
COPY gmi/ /gmi_base/gmi/
COPY setup.py /gmi_base/

# Install GMI package in editable mode
RUN pip install -e .

# Set default command
CMD ["tail", "-f", "/dev/null"] 
