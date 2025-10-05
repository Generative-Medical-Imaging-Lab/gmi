import torch
import torch.nn as nn
import torch.nn.functional as F
import gmi
import numpy as np
from typing import Tuple

class XRayPhysicsSimulator:
    """Simulates X-ray physics with Poisson noise"""
    
    def __init__(self, mu=4.0, I0=1e2):
        self.mu = mu  # Attenuation coefficient
        self.I0 = I0  # Incident photon intensity
    
    def forward_model(self, x):
        """
        Forward model: simulate noisy X-ray measurements
        x: clean images in [0,1]
        returns: noisy measurements following Poisson(I0 * exp(-mu * x))
        """
        # Ensure x is in [0,1] range
        x = torch.clamp(x, 0, 1)
        
        # Simulate X-ray physics: y ~ Poisson(I0 * exp(-mu * x))
        lambda_param = self.I0 * torch.exp(-self.mu * x)
        
        # Sample from Poisson distribution
        y = torch.poisson(lambda_param)
        
        # Clip minimum counts to 1 to avoid zero count cases
        y = torch.clamp(y, min=1.0)
        
        return y
    
    def log_correction(self, y):
        """
        Apply log correction to noisy measurements
        y_corr = -log(clip(y, 1.0) / I0)
        Note: y should already be clipped to minimum 1 in forward_model
        """
        # Clip to minimum photon count of 1 (should already be done in forward_model)
        y_clipped = torch.clamp(y, min=1.0)
        
        # Apply log correction
        y_corr = -torch.log(y_clipped / self.I0)
        
        return y_corr

class XRayConditionalDenoiser(nn.Module):
    """
    X-ray conditional denoiser that combines log correction preprocessing 
    with a conditional Gaussian denoiser from GMI.
    """
    
    def __init__(self, mu=4.0, I0=1e2, noise_std=0.1):
        super().__init__()
        
        # Physics parameters
        self.mu = mu
        self.I0 = I0
        
        # Physics simulator
        self.physics = XRayPhysicsSimulator(mu=mu, I0=I0)
        
        # Conditional Gaussian denoiser (identity covariance, estimated mean)
        # Uses SimpleCNN to estimate the mean of p(x|y_corr)
        mean_estimator = nn.Sequential(
            gmi.network.SimpleCNN(
                input_channels=1,  # Log-corrected measurements
                output_channels=1,  # Clean images
                hidden_channels_list=[32, 64, 128, 256, 128, 64, 32],
                activation=nn.SiLU(),
                dim=2
            ),
            nn.Sigmoid()  # Ensure output is in [0,1] range
        )
        self.conditional_gaussian = gmi.random_variable.ConditionalGaussianDenoiser(
            mean_estimator=mean_estimator,
            noise_std=0.1  # Fixed noise standard deviation
        )
    
    def get_log_corrected(self, y):
        """Apply log correction to measurements"""
        return self.physics.log_correction(y)
    
    def get_mean_estimate(self, y):
        """Get mean estimate of clean image given noisy measurements"""
        y_corr = self.get_log_corrected(y)
        return self.conditional_gaussian.get_mean_estimate(y_corr)
    
    def log_prob(self, y, x):
        """
        Compute log probability log p(x|y) for training.
        This implements log N(x; μ_θ(y_corr), σ²I)
        """
        # Apply log correction to measurements
        y_corr = self.get_log_corrected(y)
        
        # Get mean estimate from the network
        mu = self.conditional_gaussian.get_mean_estimate(y_corr)
        
        # Compute Gaussian log probability: log p(x|y) = log N(x; μ_θ(y), σ²I)
        # log N(x; μ, σ²I) = -0.5 * (||x - μ||² / σ² + d*log(2πσ²))
        residual = (x - mu).flatten(start_dim=1)  # Flatten spatial dimensions
        squared_error = torch.sum(residual ** 2, dim=1)  # Sum over spatial dimensions
        
        d = residual.shape[1]  # Number of dimensions
        noise_std = self.conditional_gaussian.noise_std
        log_prob = -0.5 * (squared_error / (noise_std ** 2) + d * torch.log(torch.tensor(2 * torch.pi * (noise_std ** 2), device=x.device)))
        
        return log_prob
    
    def sample(self, y, num_samples=1):
        """Sample from posterior p(x|y)"""
        y_corr = self.get_log_corrected(y)
        return self.conditional_gaussian.sample(y_corr, num_samples)
    
    def forward(self, y):
        """Forward pass returns mean estimate"""
        return self.get_mean_estimate(y)

def create_datasets_and_loaders(config):
    """Create datasets and data loaders"""
    data_config = config.data
    
    # Create datasets
    dataset_train = gmi.datasets.MedMNIST(
        data_config['dataset_name'], 
        split='train',
        root=data_config['dataset_root'],
        size=data_config['image_size'], 
        download=True,
        images_only=True  # Only return images, not labels
    )
    
    dataset_val = gmi.datasets.MedMNIST(
        data_config['dataset_name'],
        split='val',
        root=data_config['dataset_root'],
        size=data_config['image_size'], 
        download=True,
        images_only=True  # Only return images, not labels
    )
    
    dataset_test = gmi.datasets.MedMNIST(
        data_config['dataset_name'],
        split='test',
        root=data_config['dataset_root'],
        size=data_config['image_size'], 
        download=True,
        images_only=True  # Only return images, not labels
    )
    
    # Create data loaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=data_config['batch_size'], 
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        num_workers=data_config['num_workers']
    )
    
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=data_config['batch_size'],
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        num_workers=data_config['num_workers']
    )
    
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=data_config['batch_size'],
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        num_workers=data_config['num_workers']
    )
    
    return (dataset_train, dataset_val, dataset_test, 
            dataloader_train, dataloader_val, dataloader_test)

def create_model_and_optimizer(config):
    """Create model and optimizer"""
    physics_config = config.physics
    model_config = config.model
    training_config = config.training
    
    # Create physics simulator
    physics_simulator = XRayPhysicsSimulator(
        mu=physics_config['mu'],
        I0=physics_config['I0']
    )
    
    # Create conditional denoiser
    conditional_denoiser = XRayConditionalDenoiser(
        mu=physics_config['mu'],
        I0=physics_config['I0'],
        noise_std=model_config['noise_std']
    ).to(config.device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        conditional_denoiser.parameters(), 
        lr=float(training_config['learning_rate']),
        weight_decay=float(training_config['weight_decay'])
    )
    
    return physics_simulator, conditional_denoiser, optimizer

def compute_metrics(conditional_denoiser, physics_simulator, dataloader_test, device):
    """Compute denoising metrics on test set"""
    conditional_denoiser.eval()
    
    total_mse = 0.0
    total_psnr = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, x_batch in enumerate(dataloader_test):
            if batch_idx >= 50:  # Limit for efficiency
                break
            
            x_batch = x_batch.to(device)
            
            # Simulate noisy measurements
            y_batch = physics_simulator.forward_model(x_batch)
            
            # Get denoised reconstruction
            x_pred = conditional_denoiser.get_mean_estimate(y_batch)
            
            # Compute MSE
            mse = F.mse_loss(x_pred, x_batch)
            total_mse += mse.item() * len(x_batch)
            
            # Compute PSNR
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            total_psnr += psnr.item() * len(x_batch)
            
            total_samples += len(x_batch)
    
    avg_mse = total_mse / total_samples
    avg_psnr = total_psnr / total_samples
    
    return {
        'mse': avg_mse,
        'psnr': avg_psnr
    }