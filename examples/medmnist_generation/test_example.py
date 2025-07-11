#!/usr/bin/env python3
"""
Test script to verify the updated MedMNIST diffusion example works correctly.
This script tests the basic functionality without running the full training.
"""

import torch
import gmi
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_diffusion_model_creation():
    """Test that we can create a diffusion model with the new API."""
    print("Testing diffusion model creation...")
    
    # Test device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create diffusion model using the smaller DiffusersUnet2D_Size28 network
    diffusion_backbone = gmi.network.DiffusersUnet2D_Size28(
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(8, 16, 32),  # Small network for 28x28 images
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D")
    ).to(device)
    
    forward_SDE = gmi.sde.SongVarianceExplodingSDE(
        noise_variance=lambda t: t, 
        noise_variance_prime=lambda t: t*0.0 + 1.0
    )
    
    diffusion_model = gmi.diffusion.DiffusionModel(
        diffusion_backbone=diffusion_backbone, 
        forward_SDE=forward_SDE
    )
    
    print("✓ Diffusion model created successfully")
    return diffusion_model

def test_forward_pass():
    """Test that the forward pass works correctly."""
    print("Testing forward pass...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion_model = test_diffusion_model_creation()
    
    # Create dummy data (28x28 images)
    batch_size = 2
    x_0 = torch.randn(batch_size, 1, 28, 28).to(device)
    
    # Test forward pass
    loss = diffusion_model(x_0)
    
    print(f"✓ Forward pass successful, loss: {loss.item():.4f}")
    return diffusion_model

def test_train_diffusion_model_method():
    """Test that the train_diffusion_model method exists and has correct signature."""
    print("Testing train_diffusion_model method...")
    
    diffusion_model = test_forward_pass()
    
    # Check that the method exists
    assert hasattr(diffusion_model, 'train_diffusion_model'), "train_diffusion_model method not found"
    
    # Check the method signature
    import inspect
    sig = inspect.signature(diffusion_model.train_diffusion_model)
    params = list(sig.parameters.keys())
    
    # Check for required parameters
    required_params = ['dataset', 'self']
    for param in required_params:
        assert param in params, f"Required parameter '{param}' not found in train_diffusion_model"
    
    print("✓ train_diffusion_model method has correct signature")
    print(f"  Parameters: {params}")

def test_time_sampler():
    """Test that we can create and use a time sampler."""
    print("Testing time sampler...")
    
    class TimeSampler(gmi.samplers.Sampler):
        def __init__(self):
            super().__init__()
        
        def sample(self, batch_size):
            log10_t_mu = -1.0
            log10_t_sigma = 1.0 
            log10_t = torch.randn(batch_size) * log10_t_sigma + log10_t_mu  # Shape: [batch_size]
            t = 10**log10_t
            return t
    
    time_sampler = TimeSampler()
    samples = time_sampler.sample(5)
    
    print(f"✓ Time sampler created successfully, sample shape: {samples.shape}")
    assert samples.shape == (5,), f"Expected shape (5,), got {samples.shape}"

def test_reverse_process():
    """Test that the reverse process sampling works."""
    print("Testing reverse process sampling...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion_model = test_forward_pass()
    
    # Create initial noise (28x28 images)
    batch_size = 2
    x_t = torch.randn(batch_size, 1, 28, 28).to(device)
    
    # Test reverse process
    timesteps = torch.linspace(1.0, 0.0, 10).to(device)
    
    diffusion_model.eval()
    with torch.no_grad():
        samples = diffusion_model.sample_reverse_process(
            x_t, 
            timesteps, 
            sampler='euler', 
            return_all=False, 
            verbose=False
        )
    
    # Handle both tensor and list outputs
    if isinstance(samples, torch.Tensor):
        print(f"✓ Reverse process sampling successful, output shape: {samples.shape}")
    else:
        print(f"✓ Reverse process sampling successful, output type: {type(samples)}")

def test_weighted_mse_loss():
    """Test that the weighted MSE loss function works correctly."""
    print("Testing weighted MSE loss...")
    
    class WeightedMSELoss(torch.nn.Module):
        def __init__(self):
            super(WeightedMSELoss, self).__init__()
        
        def forward(self, x_0, x_t, t):
            loss_weights = 1/t
            loss_weights = loss_weights.reshape(-1, 1, 1, 1)
            return torch.sum(loss_weights*(x_0 - x_t)**2)
    
    weighted_mse_loss_fn = WeightedMSELoss()
    
    # Test with dummy data
    batch_size = 2
    x_0 = torch.randn(batch_size, 1, 28, 28)
    x_t = torch.randn(batch_size, 1, 28, 28)
    t = torch.rand(batch_size, 1) * 0.5 + 0.1  # Avoid division by zero
    
    loss = weighted_mse_loss_fn(x_0, x_t, t)
    print(f"✓ Weighted MSE loss successful, loss: {loss.item():.4f}")

def main():
    """Run all tests."""
    print("Running MedMNIST diffusion example tests...")
    print("=" * 50)
    
    try:
        test_time_sampler()
        test_weighted_mse_loss()
        test_diffusion_model_creation()
        test_forward_pass()
        test_train_diffusion_model_method()
        test_reverse_process()
        
        print("=" * 50)
        print("✓ All tests passed! The updated example should work correctly.")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 