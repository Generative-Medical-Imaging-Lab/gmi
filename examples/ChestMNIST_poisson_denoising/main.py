"""
Chest X-ray Generative Denoiser with Animation

This script implements a supervised learning approach to train a conditional denoiser
p_θ(x|y) where:
- x are clean chest X-ray images from ChestMNIST dataset (values in [0,1])
- y are noisy, nonlinear measurements following: q(y|x) = Poisson(λ = I₀ * exp(-μX))
- μ = 4.0 (global attenuation coefficient)
- I₀ = 1e3 (incident photon intensity)

The denoising approach uses:
1. Fixed nonlinear log correction: y_corr = -log(clip(y, 1e-10) / I₀)
2. Conditional Gaussian denoiser as a mean estimator with identity covariance

Creates animated visualizations of the denoising process during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gmi
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from functools import partial

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Physics parameters
MU = 4.0  # Attenuation coefficient
I0 = 1e2  # Incident photon intensity

# Create output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

# Dataset setup
medmnist_dataset_root = '/workspace/gmi/gmi_data/datasets/medmnist_dataset_root/'
os.makedirs(medmnist_dataset_root, exist_ok=True)

print("=== Loading ChestMNIST Dataset ===")
dataset_train = gmi.datasets.MedMNIST('ChestMNIST', 
                                      split='train',
                                      root=medmnist_dataset_root,
                                      size=64, 
                                      download=True,
                                      images_only=True)  # We only need images for denoising

dataset_val = gmi.datasets.MedMNIST('ChestMNIST',
                                    split='val',
                                    root=medmnist_dataset_root,
                                    size=64, 
                                    download=True,
                                    images_only=True)

print(f"Dataset sizes - Train: {len(dataset_train)}, Val: {len(dataset_val)}")

# Data loaders
batch_size = 32
dataloader_train = torch.utils.data.DataLoader(dataset_train, 
                                                batch_size=batch_size, 
                                                shuffle=True,
                                                num_workers=4,
                                                pin_memory=True)

dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True)


class XRayPhysicsSimulator(nn.Module):
    """
    Simulates X-ray physics measurement process:
    y ~ Poisson(λ = I₀ * exp(-μX))
    
    Where X represents line integrals (clean image values in [0,1])
    """
    def __init__(self, mu=MU, I0=I0):
        super().__init__()
        self.mu = mu
        self.I0 = I0
    
    def forward(self, x):
        """
        Args:
            x: Clean images [batch_size, channels, height, width], values in [0,1]
        
        Returns:
            y: Noisy measurements with same shape as x
        """
        # Compute Poisson rate parameter: λ = I₀ * exp(-μX)
        lambda_param = self.I0 * torch.exp(-self.mu * x)
        
        # Sample from Poisson distribution
        # Note: Poisson sampling can be unstable for small lambda values
        # We'll use a robust implementation
        lambda_param = torch.clamp(lambda_param, min=1e-10, max=1e6)
        y = torch.poisson(lambda_param)
        
        # Clip minimum counts to 1 to avoid zero count cases
        y = torch.clamp(y, min=1.0)
        
        return y


class LogCorrectionPreprocessor(nn.Module):
    """
    Fixed log correction preprocessing:
    y_corr = -log(clip(y, ε) / I₀)
    """
    def __init__(self, I0=I0, epsilon=1e-10):
        super().__init__()
        self.I0 = I0
        self.epsilon = epsilon
    
    def forward(self, y):
        """
        Apply fixed log correction step:
        y_corr = -log(clip(y, ε) / I₀)
        """
        # Clip to avoid log(0)
        y_clipped = torch.clamp(y, min=self.epsilon)
        
        # Apply log correction
        y_corr = -torch.log(y_clipped / self.I0)
        
        return y_corr


class XRayConditionalDenoiser(nn.Module):
    """
    X-ray conditional denoiser that combines log correction preprocessing 
    with a conditional Gaussian denoiser from GMI.
    
    This implements the full pipeline:
    1. Log correction: y_corr = -log(clip(y, ε) / I₀)
    2. Conditional Gaussian denoising: p(x|y_corr) = N(μ_θ(y_corr), σ²I)
    """
    def __init__(self, noise_std=0.1, mu=MU, I0=I0):
        super().__init__()
        
        # Log correction preprocessor
        self.log_corrector = LogCorrectionPreprocessor(I0=I0)
        
        # Create mean estimation network
        mean_estimator = nn.Sequential(
            gmi.network.SimpleCNN(
                input_channels=1,
                output_channels=1,
                hidden_channels_list=[32, 64, 128, 256, 128, 64, 32],
                activation=nn.SiLU(),
                dim=2
            ),
            nn.Sigmoid()  # Ensure output is in [0,1] range
        )
        
        # Store components separately for custom implementation
        self.mean_estimator = mean_estimator
        self.noise_std = noise_std
    
    def log_prob(self, y, x):
        """Compute log probability of clean image x given measurement y"""
        # Apply log correction to measurements
        y_corr = self.log_corrector(y)
        
        # Get mean estimate from the network
        mu = self.mean_estimator(y_corr)
        
        # Compute Gaussian log probability: log p(x|y) = log N(x; μ_θ(y), σ²I)
        # log N(x; μ, σ²I) = -0.5 * (||x - μ||² / σ² + d*log(2πσ²))
        residual = (x - mu).flatten(start_dim=1)  # Flatten spatial dimensions
        squared_error = torch.sum(residual ** 2, dim=1)  # Sum over spatial dimensions
        
        d = residual.shape[1]  # Number of dimensions
        log_prob = -0.5 * (squared_error / (self.noise_std ** 2) + d * math.log(2 * math.pi * (self.noise_std ** 2)))
        
        return log_prob
    
    def get_mean_estimate(self, y):
        """Get the mean estimate μ_θ(y) with log correction preprocessing"""
        y_corr = self.log_corrector(y)
        return self.mean_estimator(y_corr)
    
    def get_log_corrected(self, y):
        """Get log-corrected measurements for visualization"""
        return self.log_corrector(y)
    
    def sample(self, y):
        """Sample from the conditional distribution p(x|y) with log correction preprocessing"""
        y_corr = self.log_corrector(y)
        mu = self.mean_estimator(y_corr)
        # Add Gaussian noise with standard deviation noise_std
        noise = torch.randn_like(mu) * self.noise_std
        return mu + noise


# Initialize physics simulator and conditional denoiser
physics_simulator = XRayPhysicsSimulator().to(device)
conditional_denoiser = XRayConditionalDenoiser(noise_std=0.1).to(device)

# Training setup
optimizer = torch.optim.Adam(conditional_denoiser.parameters(), lr=1e-3)

# For animation
training_history = {
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'sample_images': [],
    'sample_measurements': [],
    'sample_corrections': [],
    'sample_reconstructions': []
}
    
# Training parameters  
num_epochs = 50
num_iterations = 100
num_iterations_val = 20

# Storage for losses and predictions
all_train_losses = []
all_val_losses = []

print(f"\n=== Starting X-ray Denoising Training ===")
print(f"Device: {device}")
print(f"Model parameters: {sum(p.numel() for p in conditional_denoiser.parameters()):,}")
print(f"Training samples: {len(dataset_train)}")
print(f"Validation samples: {len(dataset_val)}")

# Set up visualization - reduced size by half
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
plt.subplots_adjust(hspace=0.4, wspace=0.4)

def animate(frame):
    """Animation function called by FuncAnimation"""
    global all_train_losses, all_val_losses, training_history
    
    print(f"Training epoch {frame + 1}/{num_epochs}...")
    
    # Prepare loss closures for this epoch
    def train_loss_closure(x_batch):
        conditional_denoiser.train()
        x_batch = x_batch.to(device)
        
        # Simulate noisy measurements
        with torch.no_grad():
            y_batch = physics_simulator(x_batch)
        
        # Compute negative log-likelihood for denoising
        return -conditional_denoiser.log_prob(y_batch, x_batch).mean()
    
    def val_loss_closure(x_batch):
        conditional_denoiser.eval()
        with torch.no_grad():
            x_batch = x_batch.to(device)
            
            # Simulate noisy measurements
            y_batch = physics_simulator(x_batch)
            
            return -conditional_denoiser.log_prob(y_batch, x_batch).mean()
    
    # Train for one epoch using GMI's train function
    train_losses, val_losses = gmi.train(
        train_data=dataloader_train, 
        val_data=dataloader_val,
        train_loss_closure=train_loss_closure,
        val_loss_closure=val_loss_closure,
        num_epochs=1, 
        num_iterations=num_iterations,
        num_iterations_val=num_iterations_val,
        optimizer=optimizer,
        device=device, 
        verbose=True
    )
    
    # Store losses
    all_train_losses.extend(train_losses)
    all_val_losses.extend(val_losses)
    training_history['epoch'].append(frame)
    training_history['train_loss'].extend(train_losses)
    training_history['val_loss'].extend(val_losses)
    
    # Generate samples for visualization - use different sample each epoch
    with torch.no_grad():
        # Get a random sample from validation dataset (not dataloader)
        # This ensures we get different samples each epoch
        sample_idx = np.random.randint(0, len(dataset_val))
        x_sample = dataset_val[sample_idx].unsqueeze(0).to(device)  # Add batch dimension
        
        # Simulate measurements
        y_sample = physics_simulator(x_sample)
        
        # Get log corrected version
        y_corr_sample = conditional_denoiser.get_log_corrected(y_sample)
        
        # Get denoised predictions (mean of conditional Gaussian)
        x_pred_sample = conditional_denoiser.get_mean_estimate(y_sample)
        
        # Store for animation
        training_history['sample_images'].append(x_sample.cpu().numpy())
        training_history['sample_measurements'].append(y_sample.cpu().numpy())
        training_history['sample_corrections'].append(y_corr_sample.cpu().numpy())
        training_history['sample_reconstructions'].append(x_pred_sample.cpu().numpy())
    
    # Clear previous plots
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    
    # Plot 1: Training and validation loss curves
    epochs_so_far = list(range(1, len(all_train_losses) + 1))
    if all_train_losses:
        ax1.plot(epochs_so_far, all_train_losses, 'b-', label='Train Loss', linewidth=2)
    if all_val_losses:
        ax1.plot(epochs_so_far, all_val_losses, 'r-', label='Val Loss', linewidth=2)
    
    ax1.set_xlim(0, num_epochs)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Negative Log-Likelihood')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plots 2-4: Show denoising results for current epoch
    if len(training_history['sample_images']) > 0:
        current_idx = len(training_history['sample_images']) - 1
        
        # Get current samples (single sample per epoch now)
        images = training_history['sample_images'][current_idx]
        measurements = training_history['sample_measurements'][current_idx]
        corrections = training_history['sample_corrections'][current_idx]
        reconstructions = training_history['sample_reconstructions'][current_idx]
        
        # Show the single sample (batch size = 1)
        # Original image
        im_orig = images[0, 0]  # [batch_idx=0, channel_idx=0]
        ax2.imshow(im_orig, cmap='gray', vmin=0, vmax=1)
        ax2.set_title('Original Image', fontsize=12)
        ax2.axis('off')
        
        # Log corrected measurements
        im_corr = corrections[0, 0]  # [batch_idx=0, channel_idx=0]
        ax3.imshow(im_corr, cmap='gray')
        ax3.set_title('Log Corrected (-log(y/I₀))', fontsize=12)
        ax3.axis('off')
        
        # Denoised reconstruction
        im_recon = reconstructions[0, 0]  # [batch_idx=0, channel_idx=0]
        ax4.imshow(im_recon, cmap='gray', vmin=0, vmax=1)
        ax4.set_title('Denoised Reconstruction', fontsize=12)
        ax4.axis('off')
    
    # Update figure title with loss information
    train_loss = train_losses[0] if train_losses else 0
    val_loss = val_losses[0] if val_losses else 0
    fig.suptitle(f'X-ray Denoising - Epoch {frame + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}', 
                fontsize=16)

# Create animation-safe dataloaders (num_workers=0 for animation)
print("Setting up animation-safe dataloaders...")
dataloader_train_anim = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
dataloader_val_anim = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

# Store original dataloaders
dataloader_train_orig = dataloader_train
dataloader_val_orig = dataloader_val

# Use animation-safe dataloaders
dataloader_train = dataloader_train_anim
dataloader_val = dataloader_val_anim

# Create the animation
print("Creating training animation...")
ani = animation.FuncAnimation(
    fig, animate, frames=num_epochs, interval=1000, blit=False, repeat=False
)

# Save as MP4
script_dir = os.path.dirname(os.path.abspath(__file__))
animation_path = os.path.join(script_dir, 'outputs', 'chest_xray_denoising_animation.mp4')
print(f"Saving MP4 animation to: {animation_path}")

writer = animation.FFMpegWriter(fps=2, bitrate=1200)
ani.save(animation_path, writer=writer)
print(f"MP4 animation saved: {animation_path}")

# Restore original dataloaders
dataloader_train = dataloader_train_orig
dataloader_val = dataloader_val_orig

# Save final model
model_path = os.path.join(output_dir, 'chest_xray_denoiser.pth')
torch.save({
    'denoiser_state_dict': conditional_denoiser.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'training_history': training_history
}, model_path)
print(f"Model saved to: {model_path}")

print(f"Final train loss: {all_train_losses[-1]:.4f}" if all_train_losses else "No training completed")
print(f"Final val loss: {all_val_losses[-1]:.4f}" if all_val_losses else "No validation completed")

plt.show()