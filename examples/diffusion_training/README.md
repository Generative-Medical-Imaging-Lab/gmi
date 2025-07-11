# Diffusion Model Training Examples

This directory contains examples for training unconditional diffusion models using the GMI package.

## Overview

The examples demonstrate how to train diffusion models that can generate images from noise using the reverse diffusion process. The training includes:

- **Unconditional generation**: Models learn to generate images without any conditioning
- **Configurable reverse process**: Control sampling parameters during evaluation
- **WandB integration**: Log generated samples and training metrics
- **Modular configuration**: Easy swapping of datasets and model architectures

## Files

- `diffusion_training_config.yaml` - Example configuration for training a diffusion model on BloodMNIST
- `train_diffusion_example.py` - Example script demonstrating how to use the training command
- `README.md` - This documentation file

## Quick Start

### 1. Basic Training

Train a diffusion model using the example configuration:

```bash
# From the project root
python main.py train-diffusion-model examples/diffusion_training/diffusion_training_config.yaml
```

This will train a model to generate blood cell images from the BloodMNIST dataset.

### 2. Using the Example Script

Run the provided example script:

```bash
# From the project root
python examples/diffusion_training/train_diffusion_example.py
```

This will automatically download the BloodMNIST dataset and start training.

### 3. Custom Configuration

Create your own configuration file:

```yaml
experiment_name: "my_diffusion_experiment"

train_dataset:
  class: gmi.datasets.MedMNIST
  params:
    dataset_name: "BloodMNIST"
    size: 28
    split: "train"
    images_only: true

diffusion_backbone:
  class: gmi.network.DiffusersUnet2D_Size28
  params:
    in_channels: 3
    out_channels: 3
    layers_per_block: 2
    block_out_channels: [8, 16, 32]

training:
  num_epochs: 100
  learning_rate: 0.001
  batch_size: 32
  
  # Reverse process sampling parameters
  reverse_t_start: 1.0
  reverse_t_end: 0.0
  reverse_spacing: "linear"
  reverse_sampler: "euler"
  reverse_timesteps: 50
```

## Configuration Parameters

### Training Parameters

- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimization
- `batch_size`: Batch size for training
- `use_ema`: Whether to use exponential moving average
- `early_stopping`: Whether to use early stopping

### Reverse Process Parameters

- `reverse_t_start`: Starting time for reverse process (default: 1.0)
- `reverse_t_end`: Ending time for reverse process (default: 0.0)
- `reverse_spacing`: Timestep spacing ('linear' or 'logarithmic')
- `reverse_sampler`: Sampling method ('euler' or 'heun')
- `reverse_timesteps`: Number of timesteps for sampling

## Output

Training produces:

- `best_model.pth`: Best model checkpoint
- `final_config.yaml`: Final configuration used
- `test_samples_epoch_*.png`: Generated samples during training
- WandB logs: Training metrics and generated samples (if enabled)

## Supported Datasets

- **BloodMNIST**: Blood cell microscopy images (3 channels, 28x28)
- **MNIST**: Handwritten digits (1 channel, 28x28)
- **MedMNIST**: Medical imaging datasets (1-3 channels, various sizes)
- **Custom datasets**: Any dataset compatible with PyTorch DataLoader

## Supported Model Architectures

- **DiffusersUnet2D_Size28**: UNet architecture optimized for 28x28 images
- **Custom backbones**: Any PyTorch module that takes (x_t, t) as input

## Advanced Usage

### Modular Configuration

Use the modular configuration system for easy experimentation:

```bash
python main.py train-diffusion-model config.yaml \
  --train-dataset datasets/mnist_train.yaml \
  --diffusion-backbone networks/complex_unet_1ch.yaml
```

### Command Line Overrides

Override any configuration parameter:

```bash
python main.py train-diffusion-model config.yaml \
  --experiment-name my_experiment \
  --device cuda
```

## Related Examples

- `examples/modular_configs/` - Modular configuration system for systematic experiments
- `examples/medmnist_generation/` - MedMNIST-specific generation examples
- `examples/medmnist_restoration/` - Conditional diffusion model examples 