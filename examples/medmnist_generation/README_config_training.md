# Diffusion Model Training from Configuration Files

This directory contains examples demonstrating how to train diffusion models using configuration files with the GMI package.

## Overview

The new config-based training functionality allows you to:

1. **Train from YAML config files** - Load all training parameters from a single YAML file
2. **Train from config dictionaries** - Pass configuration as Python dictionaries
3. **Modular configuration** - Easily swap datasets, model architectures, and training parameters
4. **Reproducible experiments** - Save and share complete experiment configurations

## Files

- `main.py` - Original example with hardcoded parameters
- `train_from_config.py` - New example demonstrating config-based training
- `medmnist_diffusion_config.yaml` - Generated configuration file (created by the example)
- `README_config_training.md` - This documentation file

## Quick Start

### 1. Run the Config-Based Training Example

```bash
# From the project root
python examples/medmnist_generation/train_from_config.py
```

This will:
- Create a configuration file (`medmnist_diffusion_config.yaml`)
- Offer you three training options:
  1. Train from config file (using class method)
  2. Train from config dictionary (using instance method)
  3. Both methods for comparison

### 2. Use the CLI Command

You can also use the existing CLI command with the generated config:

```bash
# From the project root
python main.py train-diffusion-model examples/medmnist_generation/medmnist_diffusion_config.yaml
```

## Configuration File Structure

The configuration file follows this structure:

```yaml
experiment_name: "my_diffusion_experiment"

# Dataset configurations
train_dataset:
  class: gmi.datasets.MedMNIST
  params:
    dataset_name: "BloodMNIST"
    size: 28
    split: "train"
    images_only: true

validation_dataset:
  class: gmi.datasets.MedMNIST
  params:
    dataset_name: "BloodMNIST"
    size: 28
    split: "val"
    images_only: true

test_dataset:
  class: gmi.datasets.MedMNIST
  params:
    dataset_name: "BloodMNIST"
    size: 28
    split: "test"
    images_only: true

# Diffusion backbone configuration
diffusion_backbone:
  class: gmi.network.DiffusersUnet2D_Size28
  params:
    in_channels: 3
    out_channels: 3
    layers_per_block: 2
    block_out_channels: [32, 64, 128]
    down_block_types: ["DownBlock2D", "DownBlock2D", "DownBlock2D"]
    up_block_types: ["UpBlock2D", "UpBlock2D", "UpBlock2D"]

# Training configuration
training:
  num_epochs: 100
  num_iterations_train: 100
  learning_rate: 0.001
  batch_size: 32
  num_workers: 2
  # ... more training parameters
```

## API Usage

### Method 1: Class Method (Recommended)

```python
from gmi.diffusion.core import DiffusionModel

# Train from YAML config file
train_losses, val_losses, eval_metrics = DiffusionModel.train_from_config_file(
    config_path="path/to/config.yaml",
    device=None,  # Auto-detect
    experiment_name=None,  # Use from config
    output_dir=None  # Use default
)
```

### Method 2: Instance Method

```python
from gmi.diffusion.core import DiffusionModel
from gmi.config import load_object_from_dict

# Load config dictionary
config_dict = {
    'experiment_name': 'my_experiment',
    'train_dataset': {...},
    'diffusion_backbone': {...},
    'training': {...}
}

# Load diffusion backbone
diffusion_backbone = load_object_from_dict(config_dict['diffusion_backbone'])

# Create diffusion model
diffusion_model = DiffusionModel(diffusion_backbone=diffusion_backbone)

# Train from config
train_losses, val_losses, eval_metrics = diffusion_model.train_diffusion_model_from_config(
    config_dict=config_dict,
    device='cuda',
    experiment_name=None,
    output_dir=None
)
```

## Configuration Parameters

### Required Components

- `train_dataset`: Training dataset configuration
- `diffusion_backbone`: Neural network backbone configuration
- `experiment_name`: Name for the experiment

### Optional Components

- `validation_dataset`: Validation dataset configuration
- `test_dataset`: Test dataset configuration
- `training`: Training parameters (see below)

### Training Parameters

All training parameters are optional with sensible defaults:

```yaml
training:
  # Basic training
  num_epochs: 100
  num_iterations_train: 100
  learning_rate: 0.001
  batch_size: 32
  
  # Data loading
  num_workers: 4
  shuffle_train: true
  shuffle_val: true
  shuffle_test: false
  
  # Advanced features
  use_ema: true
  ema_decay: 0.999
  early_stopping: true
  patience: 20
  
  # Validation and testing
  num_iterations_val: 10
  num_iterations_test: 5
  final_test_iterations: 20
  
  # Reverse process sampling
  reverse_t_start: 1.0
  reverse_t_end: 0.0
  reverse_spacing: "linear"
  reverse_sampler: "euler"
  reverse_timesteps: 50
  
  # Logging and monitoring
  verbose: true
  wandb_project: "my-project"
  save_checkpoints: true
  test_save_plots: true
```

## Benefits of Config-Based Training

1. **Reproducibility**: Complete experiment configuration is saved
2. **Modularity**: Easy to swap components without code changes
3. **Scalability**: Run multiple experiments with different configs
4. **Version Control**: Track configuration changes in git
5. **Sharing**: Share experiments by sharing config files
6. **Hyperparameter Tuning**: Easy to modify parameters for optimization

## Output Structure

Training creates the following directory structure:

```
gmi_data/outputs/{experiment_name}/
├── final_config.yaml          # Complete configuration used
├── best_model.pth             # Best model checkpoint
├── test_samples_epoch_*.png   # Generated samples during training
└── wandb/                     # WandB logs (if enabled)
```

## Examples

### Custom Dataset

```yaml
train_dataset:
  class: gmi.datasets.MNIST
  params:
    split: "train"
    images_only: true
```

### Custom Backbone

```yaml
diffusion_backbone:
  class: gmi.network.SimpleCNN
  params:
    in_channels: 1
    out_channels: 1
    hidden_channels: [32, 64, 128]
```

### Different Training Parameters

```yaml
training:
  num_epochs: 500
  learning_rate: 1e-5
  batch_size: 64
  wandb_project: "my-custom-experiment"
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure the class paths in config are correct
2. **Memory Issues**: Reduce batch_size or model size
3. **CUDA Errors**: Set device to 'cpu' if GPU memory is insufficient
4. **Config Validation**: Check that all required parameters are present

### Debug Mode

Enable verbose logging:

```yaml
training:
  verbose: true
  very_verbose: true
```

This will print detailed information about the training process and help identify issues. 