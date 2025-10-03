# FROM python:3.12-slim
# FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ARG CUDA_MAJOR=12
ARG CUDA_MINOR=9
FROM nvidia/cuda:${CUDA_MAJOR}.${CUDA_MINOR}.0-devel-ubuntu22.04 AS base_image


# Install system dependencies using apt package manager
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3 
RUN apt-get install -y python3-pip 
RUN apt-get install -y python3-venv
RUN apt-get install -y ffmpeg 
RUN apt-get install -y git 
RUN apt-get install -y build-essential libglib2.0-0 libsm6 libxrender1 libxext6
RUN apt-get install -y sudo
RUN rm -rf /var/lib/apt/lists/*

# Cache buster for user creation - rebuild from here if UID/GID changes
ARG CACHEBUST=1

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
RUN groupadd -o -g $GROUP_ID -f gmi_user && \
    useradd -o -u $USER_ID -g $GROUP_ID -m -s /bin/bash gmi_user && \
    usermod -aG sudo gmi_user && \
    echo "gmi_user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Create workspace directory and set ownership
RUN mkdir -p /workspace/ && chown gmi_user:gmi_user /workspace/
RUN mkdir -p /data/ && chown gmi_user:gmi_user /data/

# Create a python virtal environment with proper ownership
RUN python3 -m venv /opt/venv && chown -R gmi_user:gmi_user /opt/venv

# Prepend the virtual environment path to PATH to make it the default
ENV PATH="/opt/venv/bin:$PATH"

# Switch to non-root user early so all pip installs happen as the user
USER gmi_user

# Upgrade pip python package manager in the virtual environment
RUN pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# Set working directory
WORKDIR /workspace/

# Set default command
CMD ["tail", "-f", "/dev/null"] 
