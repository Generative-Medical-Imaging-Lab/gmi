# ChestMNIST Poisson Denoising (Reorganized)

This is a reorganized version of the ChestMNIST Poisson denoising example with modular structure.

## Structure

- `config.yml` - Configuration file with all hyperparameters
- `parse_config.py` - Configuration parser  
- `train_denoiser.py` - Model definitions and training utilities
- `animate_training_process.py` - Training animation functionality
- `main.py` - Main script that orchestrates training

## Usage

1. **Configure parameters**: Edit `config.yml` to set hyperparameters, physics parameters, etc.

2. **Run training**:
```bash
cd /workspace/gmi/examples/ChestMNIST_poisson_denoising_reorganized
python main.py
```

3. **Output**: 
   - Trained model saved to `outputs/chest_denoiser.pth`
   - Training animation saved to `outputs/chest_xray_denoising_animation.mp4`

## Configuration Options

Key parameters in `config.yml`:

- `physics.mu`: Attenuation coefficient (default: 4.0)
- `physics.I0`: Incident photon intensity (default: 100.0)
- `model.noise_std`: Noise standard deviation for denoiser
- `training.num_epochs`: Number of training epochs
- `training.learning_rate`: Learning rate

## Model Architecture

- **Physics Model**: Poisson noise simulation `y ~ Poisson(I₀ * exp(-μx))`
- **Log Correction**: `y_corr = -log(clip(y, 1e-10) / I₀)`
- **Denoiser**: U-Net based conditional Gaussian denoiser
- **Input**: Log-corrected noisy measurements
- **Output**: Clean chest X-ray images

## Physics Parameters

- `μ = 4.0`: Global attenuation coefficient
- `I₀ = 100`: Incident photon intensity (dose level)
- The forward model simulates realistic X-ray physics with Poisson noise

## Features

- Animated training visualization showing denoising progress
- Physics-based noise simulation
- Log correction preprocessing
- Configurable physics and model parameters
- PSNR/MSE evaluation metrics
- Model checkpointing