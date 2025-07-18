# Generative Medical Imaging Laboratory (GMI)

The **Generative Medical Imaging Laboratory (GMI)** is a comprehensive Python package designed for advanced computational imaging research, with a focus on medical imaging applications. GMI provides a complete toolkit for linear algebra operations, stochastic differential equations, diffusion models, and image reconstruction tasks, all built on PyTorch.

## 🚀 Quick Start

### Using Docker (Recommended)

The easiest way to get started with GMI is using Docker, which provides a pre-configured environment with all dependencies.

```bash
# Clone the repository
git clone https://github.com/Generative-Medical-Imaging-Lab/gmi.git
cd gmi

# Build and start the Docker container
docker compose up -d

# Execute commands in the container
docker exec -it gmi-container bash
```

### Direct Installation

```bash
# Install from GitHub
pip install git+https://github.com/Generative-Medical-Imaging-Lab/gmi.git

# Or install in development mode
git clone https://github.com/Generative-Medical-Imaging-Lab/gmi.git
cd gmi
pip install -e .
```

## 📁 Directory Structure

```
gmi_base/                          # Root directory (git clone location)
├── Dockerfile                      # Docker configuration
├── docker-compose.yml             # Docker Compose configuration
├── setup.py                       # Package installation script
├── requirements.txt               # Python dependencies
├── main.py                        # Command-line interface entry point
├── README.md                      # This file
├── LICENSE                        # MIT License
├── gmi/                           # Main package source code
│   ├── __init__.py
│   ├── commands/                  # CLI commands
│   │   └── visualize_dataset.py
│   ├── datasets/                  # Dataset implementations
│   │   ├── core.py               # Base dataset classes
│   │   ├── mnist.py              # MNIST dataset
│   │   ├── medmnist.py           # MedMNIST datasets
│   │   └── synthrad2023.py       # SynthRAD2023 dataset
│   ├── diffusion/                 # Diffusion model implementations
│   │   ├── core.py
│   │   ├── unconditional.py
│   │   └── diffusion_posterior.py
│   ├── distribution/              # Probability distributions
│   │   ├── core.py
│   │   ├── gaussian.py
│   │   ├── lognormal.py
│   │   └── uniform.py
│   ├── linalg/                    # Linear algebra utilities
│   │   ├── core.py
│   │   ├── fourier.py
│   │   ├── interp.py
│   │   ├── misc.py
│   │   ├── permute.py
│   │   ├── polar.py
│   │   └── sparse.py
│   ├── loss_function/             # Loss functions
│   │   └── inv_t_weighted_mse.py
│   ├── lr_scheduler/              # Learning rate schedulers
│   │   └── linear_warmup.py
│   ├── network/                   # Neural network architectures
│   │   ├── densenet.py
│   │   ├── diffusers_unet_2D.py
│   │   ├── simplecnn.py
│   │   ├── lambda_layer.py
│   │   └── diffusion/
│   │       └── MedMNISTDiffusion.py
│   ├── samplers/                  # Sampling utilities
│   │   ├── core.py
│   │   ├── dataloader_sampler.py
│   │   ├── dataset_sampler.py
│   │   └── module_sampler.py
│   ├── sde/                       # Stochastic differential equations
│   │   ├── core.py
│   │   ├── diagonal.py
│   │   ├── fourier.py
│   │   └── scalar.py
│   ├── tasks/                     # Task implementations
│   │   └── reconstruction.py      # Image reconstruction tasks
│   └── train/                     # Training utilities
│       └── __init__.py            # Training functions
├── examples/                      # Example scripts and configurations
│   ├── visualize_all_datasets/    # Dataset visualization examples
│   ├── medmnist_generation/       # MedMNIST generation examples
│   ├── medmnist_restoration/      # MedMNIST restoration examples
│   ├── medmnist_restoration_from_config/
│   ├── mnist_denoising/           # MNIST denoising examples
│   └── test_reconstruction_yaml.py
└── gmi_data/                      # Data storage directory
    ├── datasets/                  # Downloaded datasets
    │   ├── MNIST/                 # MNIST dataset files
    │   └── MedMNIST/              # MedMNIST dataset files
    ├── models/                    # Trained model checkpoints
    └── outputs/                   # Output files and visualizations
        └── visualizations/        # Generated visualizations
```

## 🐳 Docker Development Environment

This project uses a Docker container for a controlled development environment. The container is already running and contains all necessary dependencies. We develop the GMI package by editing files on the host system and testing them inside the container.

### Key Concepts

- **Host System**: Your local machine where you edit files
- **Docker Container**: Running environment with all dependencies
- **No Rebuilding**: The container is already set up and running
- **Live Development**: Edit files on host, test immediately in container

### Container Setup and Architecture

The GMI container is built from `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04` and includes:
- Python 3.12 with PyTorch 12.8 (CUDA support)
- All GMI dependencies from `requirements.txt`
- GMI package installed in editable mode (`pip install -e .`)
- Non-root user (`gmi_user`) with proper file permissions
- Working directory: `/gmi_base` (maps to your project root)

### Volume Mounting
The `docker-compose.yml` mounts your local project directory to `/gmi_base` in the container:
- Host: `./` (your project root)
- Container: `/gmi_base`
- Changes on host are immediately available in container

### GPU Support
The container is configured with NVIDIA GPU support:
- Uses NVIDIA Docker runtime
- All GPUs are available to the container
- CUDA 12.8.1 with cuDNN support

### Container Management

#### Container Status
The container is already running and ready to use. You don't need to rebuild or restart it.

#### Starting the Container (if needed)
```bash
# Build and start the container
docker compose up -d

# Check container status
docker compose ps
```

### Running Commands

#### Option 1: From Outside Container (Docker Exec)

Use this when you're working from your host system and want to execute commands inside the container.

```bash
# Execute commands inside the running container
docker exec -it gmi-container <command>

# Examples:
docker exec -it gmi-container python main.py --help
docker exec -it gmi-container python examples/modular_configs/run_default_study.py
docker exec -it gmi-container bash examples/modular_configs/run_all_studies_cli.sh
```

#### Option 2: From Inside Container

Use this when you're already inside the container (e.g., through an interactive session).

```bash
# Start interactive bash session
docker exec -it gmi-container bash

# Or start interactive Python session
docker exec -it gmi-container python
```

#### Common Commands
```bash
# Test import
docker exec -it gmi-container python -c "import gmi; print('Package loaded successfully')"

# Test CLI help
docker exec -it gmi-container python main.py --help

# Test specific command
docker exec -it gmi-container python main.py train-image-reconstructor examples/modular_configs/training_config.yaml --experiment-name test

# Run example scripts
docker exec -it gmi-container python examples/modular_configs/run_default_study.py
docker exec -it gmi-container python examples/modular_configs/run_all_modular_studies.py
docker exec -it gmi-container bash examples/modular_configs/run_all_studies_cli.sh
```

### Development Workflow

1. **Edit Files**: Edit GMI package files on your host system
2. **Test Changes**: Run tests inside the container to verify changes
3. **Debug Issues**: Use stack traces to identify and fix problems
4. **Run Examples**: Once components work, run full examples

### Debugging Tips

```bash
# Check container status
docker ps

# Check container logs
docker logs gmi-container

# Interactive debugging
docker exec -it gmi-container python
docker exec -it gmi-container bash

# Check GPU availability
docker exec -it gmi-container nvidia-smi
docker exec -it gmi-container python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Troubleshooting

- **Container Not Responding**: `docker restart gmi-container`
- **Permission Issues**: Check file permissions with `docker exec -it gmi-container ls -la /gmi_base/`
- **Memory Issues**: Monitor with `docker stats gmi-container`

This development environment provides a controlled, reproducible setup for developing and testing the GMI package. The key is to understand that you're editing on the host but testing in the container, and the `docker exec` command is your bridge between the two.

## 🖥️ GUI Display Configuration

When running GUI applications from the Docker container on a remote Linux server and displaying on a local Windows machine, you need to configure X11 forwarding. This section explains how to set up the display environment for minimal command overhead.

### Prerequisites

1. **Windows X Server**: Install and run an X server on your Windows machine:
   - **VcXsrv**: Download from [sourceforge](https://sourceforge.net/projects/vcxsrv/)
   - **MobaXterm**: Includes built-in X server
   - **WSL2**: If using WSL2, X server is typically available

2. **Network Access**: Ensure your Windows machine can accept X11 connections from your Linux server

### Setup Steps

#### 1. Configure Windows X Server

**For VcXsrv:**
```bash
# Start VcXsrv with these settings:
vcxsrv.exe :0 -multiwindow -ac -nowgl
```
- `-ac` disables access control (for testing - use `xhost` for production)
- `-nowgl` disables OpenGL acceleration if not needed

**For MobaXterm:**
- X server is typically started automatically
- Access control is usually disabled by default

#### 2. Allow Remote Connections (from Windows)

From your Windows machine, allow your Linux server to connect:
```bash
# Allow specific IP (replace with your Linux server IP)
xhost + 10.251.165.3
# Or allow all connections (less secure, for testing only)
xhost +
```

#### 3. Set Display Variable on Linux Server

Before starting the Docker container, set the DISPLAY environment variable:
```bash
# Set your Windows machine's IP address
export DISPLAY=10.251.165.7:0
# Verify the variable is set
echo $DISPLAY
```

#### 4. Start Container with Display Configuration

The `docker-compose.yml` is already configured to pass the DISPLAY variable to the container:

```yaml
environment:
  - PYTHONPATH=/gmi_base
  - DISPLAY=${DISPLAY:-:0}
```

Start the container:
```bash
docker compose up -d
```

#### 5. Run GUI Applications

Now you can run GUI applications without specifying DISPLAY in every command:
```bash
# Run GUI application (no need for -e DISPLAY=...)
docker exec -it gmi-container python gui.py

# Or run from inside the container
docker exec -it gmi-container bash
python gui.py
```

### Minimal Setup Commands

Here's the complete minimal setup sequence:

```bash
# 1. Set display variable (replace with your Windows IP)
export DISPLAY=10.251.165.7:0
# 2. Start container
docker compose up -d
# 3. Run GUI application
docker exec -it gmi-container python gui.py
```

### Troubleshooting

**Common Issues:**

1. **"Could not connect to display" error:**
   - Verify DISPLAY variable is set correctly
   - Check network connectivity: `nc -zv <windows_ip> 6000`
   - Ensure X server is running on Windows
   - Verify `xhost +` was run on Windows
2. **"Qt platform plugin could not be initialized" error:**
   - The container already includes necessary Qt dependencies
   - This usually indicates a display connection issue, not missing libraries
3. **"XDG_RUNTIME_DIR not set" warning:**
   - This is a harmless warning and doesn't affect functionality
   - Can be ignored or set with: `export XDG_RUNTIME_DIR=/tmp/runtime-gmi_user`

**Testing Display Connection:**
```bash
# Test basic X11 connection
docker exec -it gmi-container xclock

# Test with a simple Qt application
docker exec -it gmi-container python -c "import sys; from PyQt5.QtWidgets import QApplication, QLabel; app = QApplication(sys.argv); label = QLabel('Hello from Docker!'); label.show(); app.exec_()"
```

### Security Considerations

For production use, consider these security improvements:
- Use `xhost + <specific_ip>` instead of `xhost +`
- Configure firewall rules to restrict X11 traffic
- Use SSH X11 forwarding as an alternative to direct X11 connections

## 🎯 Core Features

### 1. Image Reconstruction Tasks

GMI provides a comprehensive framework for image reconstruction tasks:

```python
from gmi.tasks.reconstruction import ImageReconstructionTask

# Create a reconstruction task from YAML configuration
task = ImageReconstructionTask.from_config('config.yaml', device='cuda')

# Sample images, measurements, and reconstructions
images, measurements, reconstructions = task(
    image_batch_size=4,
    measurement_batch_size=1,
    reconstruction_batch_size=1
)
```

### 2. Training Framework

Built-in training utilities with support for:
- Early stopping
- Exponential Moving Average (EMA)
- WandB logging
- Model checkpointing
- Learning rate scheduling

```python
from gmi.train import train
from gmi.tasks.reconstruction import ImageReconstructionTask

# Load task and create loss closure
task = ImageReconstructionTask.from_config('config.yaml')
loss_closure = task.loss_closure(loss_fn)

# Train the model
train_losses, val_losses, eval_metrics = train(
    train_loader=train_loader,
    loss_closure=loss_closure,
    num_epochs=100,
    validation_loader=val_loader,
    use_ema=True,
    wandb_project="gmi-reconstruction",
    save_best_model_path="best_model.pth"
)
```

### 3. Dataset Support

Comprehensive dataset support including:
- **MNIST**: Standard handwritten digits dataset
- **MedMNIST**: 12 medical imaging datasets with multiple sizes (28x28, 64x64, 128x128, 224x224)
  - PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST
  - PneumoniaMNIST, RetinaMNIST, BreastMNIST, BloodMNIST
  - TissueMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST
- **SynthRAD2023**: Synthetic CT dataset

### 4. Neural Network Architectures

Pre-implemented architectures:
- SimpleCNN: Lightweight convolutional network
- DenseNet: Dense connection architecture
- Diffusers UNet 2D: Diffusion model backbone
- Custom lambda layers for specialized operations

### 5. Diffusion Models

Complete diffusion model implementation:
- Unconditional diffusion models
- Conditional diffusion models
- Various sampling strategies
- Training utilities

## 📊 Getting Started Examples

### 1. Visualize All Datasets

This is a great place to start as it downloads all available datasets:

```bash
# Run the comprehensive visualization script
docker exec gmi-container bash examples/visualize_all_datasets/visualize_all_datasets.sh
```

This script will:
- Download all MedMNIST variants in multiple sizes
- Generate visualizations for each dataset
- Save results to `gmi_data/outputs/visualizations/`

### 2. Test Reconstruction Task

```bash
# Test the reconstruction task with YAML configuration
docker exec gmi-container python examples/test_reconstruction_yaml.py
```

### 3. MNIST Denoising

```bash
# Run MNIST denoising example
docker exec gmi-container python examples/mnist_denoising/main.py
```

### 4. MedMNIST Restoration

```bash
# Run MedMNIST restoration with configuration
docker exec gmi-container python examples/medmnist_restoration_from_config/main.py
```

## 🔧 Configuration

GMI uses YAML configuration files for defining reconstruction tasks:

```yaml
# Example config.yaml
dataset:
  class: gmi.datasets.MNIST
  params:
    train: true
    download: true
    images_only: false

measurement_simulator:
  class: gmi.distribution.gaussian.AdditiveWhiteGaussianNoise
  params:
    noise_standard_deviation: 0.1

image_reconstructor:
  class: gmi.network.SimpleCNN
  params:
    input_channels: 1
    output_channels: 1
    hidden_channels_list: [16, 32]
    activation: relu
    dim: 2
```

## 🧪 Training and Evaluation

### Training a Model

```python
import torch
from gmi.tasks.reconstruction import ImageReconstructionTask
from gmi.train import train

# Load task from configuration
task = ImageReconstructionTask.from_config('config.yaml', device='cuda')

# Create loss function
loss_fn = torch.nn.MSELoss()

# Create loss closure
loss_closure = task.loss_closure(loss_fn)

# Train the model
train_losses, val_losses, eval_metrics = train(
    train_loader=train_loader,
    loss_closure=loss_closure,
    num_epochs=100,
    num_iterations=100,
    validation_loader=val_loader,
    use_ema=True,
    early_stopping=True,
    wandb_project="gmi-experiment",
    save_best_model_path="best_model.pth"
)
```

### Evaluating a Model

```python
# Load trained model
model = task.image_reconstructor
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate on test set
model.eval()
with torch.no_grad():
    test_images, test_measurements, test_reconstructions = task(
        image_batch_size=32,
        measurement_batch_size=1,
        reconstruction_batch_size=1
    )
    
    # Calculate metrics
    mse = torch.mean((test_reconstructions - test_images) ** 2)
    print(f"Test MSE: {mse.item():.4f}")
```

## 📈 Monitoring and Logging

### WandB Integration

```python
# Enable WandB logging during training
train_losses, val_losses, eval_metrics = train(
    train_loader=train_loader,
    loss_closure=loss_closure,
    wandb_project="gmi-reconstruction",
    wandb_config={
        "model": "SimpleCNN",
        "dataset": "MNIST",
        "noise_level": 0.1
    }
)
```

### Custom Evaluation Functions

```python
def custom_eval_fn(model, wandb_project, wandb_config, epoch):
    """Custom evaluation function called during training."""
    model.eval()
    with torch.no_grad():
        # Run evaluation
        test_loss = evaluate_model(model, test_loader)
        
        # Log to WandB
        if wandb_project:
            wandb.log({
                "test_loss": test_loss,
                "epoch": epoch
            })
        
        return {"test_loss": test_loss}

# Use in training
train_losses, val_losses, eval_metrics = train(
    train_loader=train_loader,
    loss_closure=loss_closure,
    eval_fn=custom_eval_fn,
    epochs_per_evaluation=10
)
```

## 🔍 Command Line Interface

GMI provides a CLI for common tasks:

```bash
# Visualize a specific dataset
gmi visualize-dataset --dataset mnist
gmi visualize-dataset --dataset PathMNIST_128_train

# List available commands
gmi --help
```

## 📦 Requirements

### System Requirements
- Python 3.10 or higher
- CUDA-compatible GPU (recommended)
- Docker (for containerized usage)

### Key Dependencies
- **PyTorch**: Deep learning framework
- **TorchVision**: Computer vision utilities
- **Diffusers**: Diffusion model implementations
- **MedMNIST**: Medical imaging datasets
- **WandB**: Experiment tracking
- **PyYAML**: Configuration file parsing
- **Click**: Command-line interface

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🆘 Support

For questions and support:
- Open an issue on GitHub
- Check the examples directory for usage patterns
- Review the test files for implementation details



