# MedMNIST Restoration from Configuration

This example demonstrates how to train image reconstruction models using YAML configuration files and the GMI CLI. This approach provides a clean, reproducible way to experiment with different datasets, architectures, and training parameters.

## 📁 File Structure

```
medmnist_restoration_from_config/
├── config.yaml              # Main configuration file
├── run_training.sh          # Simple shell script to run training
└── README.md                # This file
```

## 🚀 Quick Start

### Run Training

Simply execute the shell script to start training:

```bash
# From the project root directory
docker exec -it gmi-container ./examples/medmnist_restoration_from_config/run_training.sh

# Or from within the container
cd examples/medmnist_restoration_from_config
./run_training.sh
```

### Manual Command

You can also run the command manually:

```bash
# From the project root directory
docker exec -it gmi-container python main.py train-image-reconstructor examples/medmnist_restoration_from_config/config.yaml --device cpu

# Or from within the container
python main.py train-image-reconstructor config.yaml --device cpu
```

## 🆕 Enhanced Training Features

The training now includes:

### Test Phase with Metrics and Visualization
- **Automatic Test Evaluation**: Runs test evaluation during training
- **Multiple Metrics**: RMSE, PSNR, SSIM, LPIPS
- **Visualization**: Automatic generation of reconstruction plots
- **WandB Integration**: Logs metrics and plots to WandB

### Separate Closures
- **Train Closure**: Handles training loss computation
- **Validation Closure**: Handles validation loss computation  
- **Test Closure**: Handles test evaluation and visualization

### Configuration Parameters
- `test_plot_vmin`/`test_plot_vmax`: Colorbar range for plots
- `test_save_plots`: Whether to save reconstruction plots
- `num_iterations_test`: Number of test iterations per epoch

## 📋 Configuration File

The `config.yaml` file contains all the parameters for the experiment:

### Experiment Configuration
- `experiment_name`: Name for the experiment (used in model storage)

### Dataset Configuration
- `dataset.class`: Dataset class to use (e.g., `gmi.datasets.MedMNIST`)
- `dataset.params`: Dataset-specific parameters
  - For MedMNIST: `dataset_name` (e.g., "ChestMNIST", "OrganAMNIST", etc.)

### Measurement Simulator
- `measurement_simulator.class`: Forward process simulator
- `measurement_simulator.params`: Simulator parameters (e.g., noise level)

### Image Reconstructor
- `image_reconstructor.class`: Neural network architecture
- `image_reconstructor.params`: Network parameters (channels, activation, etc.)

### Training Configuration
- `training.num_epochs`: Number of training epochs
- `training.num_iterations_train`: Training iterations per epoch
- `training.num_iterations_val`: Validation iterations per epoch
- `training.num_iterations_test`: Test iterations per epoch
- `training.learning_rate`: Learning rate
- `training.batch_size`: Batch size for all phases
- `training.num_workers`: Number of workers for data loading
- `training.shuffle_train`: Whether to shuffle training data
- `training.shuffle_val`: Whether to shuffle validation data
- `training.shuffle_test`: Whether to shuffle test data
- `training.use_ema`: Whether to use Exponential Moving Average
- `training.ema_decay`: EMA decay factor
- `training.early_stopping`: Whether to use early stopping
- `training.patience`: Number of epochs to wait before early stopping
- `training.val_loss_smoothing`: Validation loss smoothing factor
- `training.min_delta`: Minimum change for early stopping
- `training.verbose`: Whether to print training progress
- `training.very_verbose`: Whether to print very detailed progress
- `training.wandb_project`: WandB project name (set to null to disable)
- `training.wandb_config`: Additional WandB configuration
- `training.save_checkpoints`: Whether to save model checkpoints
- `training.test_plot_vmin`: Minimum value for test plot colorbar
- `training.test_plot_vmax`: Maximum value for test plot colorbar
- `training.test_save_plots`: Whether to save reconstruction plots

## 📊 Output Structure

Trained models and outputs are stored in `gmi_data/outputs/` with the following structure:

```
gmi_data/outputs/
└── experiment_name/
    ├── best_model.pth           # Best model based on validation loss
    └── test_plots/              # Test reconstruction plots
        ├── reconstruction_plot_*.png
        └── ...
```

## 🔧 Customization

### Using Different Datasets
Change the dataset in `config.yaml`:

```yaml
dataset:
  class: gmi.datasets.MedMNIST
  params:
    dataset_name: "OrganAMNIST"  # Try different MedMNIST datasets
    size: 28
    split: "train"
    images_only: true
```

### Adjusting Noise Level
Modify the measurement simulator:

```yaml
measurement_simulator:
  class: gmi.random_variable.gaussian.AdditiveWhiteGaussianNoise
  params:
    noise_standard_deviation: 0.2  # Increase for more noise
```

### Changing Network Architecture
Modify the image reconstructor:

```yaml
image_reconstructor:
  class: gmi.network.SimpleCNN
  params:
    input_channels: 1
    output_channels: 1
    hidden_channels_list: [32, 64, 128, 64, 32]  # Deeper network
    activation: "silu"
    dim: 2
``` 