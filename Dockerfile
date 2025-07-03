# FROM python:3.12-slim
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Install system dependencies using apt package manager
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg git build-essential libglib2.0-0 libsm6 libxrender1 libxext6 && rm -rf /var/lib/apt/lists/*

# Make python3 and pip3 symlinks
RUN ln -sf python3 /usr/bin/python && ln -sf pip3 /usr/bin/pip

# Upgrade pip python package manager
RUN pip install --upgrade pip

# Install PyTorch 12.8 with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Create a non-root user with the same UID/GID as the host user
# If not specified, use 1000 as default user id and group id
# 
# For Linux hosts: Pass your UID/GID to ensure files created in the container are owned by your user
#    Example: docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t gmi-image .
#    Or use docker-compose: docker compose build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)
#
# For macOS/Windows: Docker Desktop handles file permissions automatically, so you can use defaults
#    Example: docker build -t gmi-image .
#    Or use docker-compose: docker compose build

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g $GROUP_ID gmi_user && \
    useradd -u $USER_ID -g $GROUP_ID -m -s /bin/bash gmi_user

# Create gmi_base directory and set ownership
RUN mkdir -p /gmi_base && chown gmi_user:gmi_user /gmi_base

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

# Change ownership of all copied files
RUN chown -R gmi_user:gmi_user /gmi_base

# Switch to non-root user
USER gmi_user

# Set default command
CMD ["tail", "-f", "/dev/null"] 
