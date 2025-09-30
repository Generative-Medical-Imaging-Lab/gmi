# Chest X-ray Generative Denoiser with Animation

This example demonstrates supervised learning for training a conditional denoiser for chest X-ray images using physically-motivated noise modeling.

## Physics Model

The script implements the following physics-based measurement model:

```
q(x): Data distribution of chest X-ray images (ChestMNIST dataset)
q(y|x) = Poisson(λ = I₀ * exp(-μX))
```

Where:
- `X` represents line integrals (clean chest X-ray images, values in [0,1])
- `μ = 4.0`: Global attenuation coefficient
- `I₀ = 1e2`: Incident photon intensity
- `y`: Noisy, nonlinear measurements

## Denoising Approach

The conditional denoiser `p_θ(x|y)` uses a two-stage approach:

1. **Fixed Log Correction**: `y_corr = -log(clip(y, ε) / I₀)`
2. **Neural Network Denoiser**: SimpleCNN architecture maps `y_corr → x`

## Features

- **Supervised Learning**: Trains on pairs of clean images and simulated noisy measurements
- **Physics-Based Simulation**: Uses Poisson noise model for realistic X-ray physics
- **Animated Training**: Creates MP4 animation showing training progress
- **Real-Time Visualization**: Shows original images, noisy measurements, log corrections, and reconstructions

## Usage

```bash
cd /workspace/gmi/examples/chest_xray_denoising_animation
python main.py
```

## Outputs

The script creates an `outputs/` directory containing:

- `chest_xray_denoising_animation.mp4`: Animated training visualization
- `final_denoising_results.png`: Final results comparison
- `chest_xray_denoiser.pth`: Saved trained model

## Architecture Details

### Physics Simulator
- Implements `XRayPhysicsSimulator` class
- Computes Poisson rate: `λ = I₀ * exp(-μX)`
- Robust Poisson sampling with clamping
- Minimum counts clipped to 1 to avoid zero count cases

### Denoiser Network
- `LogCorrectionDenoiser` with two stages
- Uses GMI's `SimpleCNN` with SiLU activations
- Architecture: `[32, 64, 128, 256, 128, 64, 32]` channels
- Sigmoid output activation for [0,1] range

### Training Setup
- Adam optimizer with 1e-3 learning rate
- MSE reconstruction loss
- Batch size: 32
- ChestMNIST dataset (64x64 grayscale images)

## Animation Features

The training animation shows:
- **Loss Curves**: Train/validation MSE loss over epochs
- **Sample Visualization**: Side-by-side comparison of:
  - Original clean images
  - Noisy measurements (y)
  - Log-corrected measurements (y_corr)
  - Neural network reconstructions

## Requirements

- PyTorch
- GMI library
- matplotlib (for animation)
- ffmpeg (for MP4 export)
- tqdm (for progress bars)

## Customization

Key parameters can be modified:

```python
# Physics parameters
MU = 4.0      # Attenuation coefficient
I0 = 1e2      # Incident photon intensity

# Training parameters
num_epochs = 15
batch_size = 32
learning_rate = 1e-3

# Network architecture
hidden_channels = [32, 64, 128, 256, 128, 64, 32]
```

## Theory Background

This approach demonstrates:
- **Physics-informed deep learning** for medical imaging
- **Supervised denoising** with realistic noise models
- **Two-stage processing**: Log-domain correction + neural denoising
- **X-ray physics**: Beer-Lambert law and Poisson photon noise

The log correction step `y_corr = -log(y/I₀)` approximately inverts the exponential attenuation, transforming the problem closer to additive noise in log-domain.