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

## 🐳 Docker Usage

### Starting the Container

```bash
# Build and start the container
docker compose up -d

# Check container status
docker compose ps
```

### Executing Commands

```bash
# Run a single command
docker exec gmi-container python -c "import gmi; print('GMI loaded successfully!')"

# Start an interactive shell
docker exec -it gmi-container bash

# Run the visualization script (downloads all datasets)
docker exec gmi-container bash examples/visualize_all_datasets/visualize_all_datasets.sh

# Run a specific example
docker exec gmi-container python examples/test_reconstruction_yaml.py
```

### Data Persistence

The `gmi_data/` directory is mounted as a volume, so:
- Downloaded datasets persist between container restarts
- Model checkpoints are saved to `gmi_data/models/`
- Visualizations are saved to `gmi_data/outputs/visualizations/`

### GPU Support

The Docker configuration includes NVIDIA GPU support. Ensure you have:
- NVIDIA Docker runtime installed
- Compatible NVIDIA drivers
- Docker configured for GPU access

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

---

**For AI Agents**: This package provides a complete framework for medical image reconstruction and generation tasks. The `gmi_data/` directory structure is automatically created and managed by the package. Use Docker for consistent environments, and start with the visualization script to download all datasets. The training framework supports various architectures and can be extended with custom components through YAML configuration files.

