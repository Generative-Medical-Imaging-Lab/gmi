#!/usr/bin/env python3
"""
ChestMNIST Poisson Denoising
Main script that orchestrates the training process
"""

import os
import torch
import numpy as np
from parse_config import parse_config
from train_denoiser import (
    create_datasets_and_loaders,
    create_model_and_optimizer,
    compute_metrics
)
from animate_training_process import animate_training_process

def save_model(physics_simulator, conditional_denoiser, optimizer, config, training_history, metrics=None):
    """Save trained model and metadata"""
    if config.training['save_model']:
        model_path = os.path.join(config.output['output_dir'], config.training['model_save_path'])
        
        save_dict = {
            'denoiser_state_dict': conditional_denoiser.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_history': training_history,
            'config': config.config,
            'physics_params': {
                'mu': config.physics['mu'],
                'I0': config.physics['I0']
            }
        }
        
        if metrics is not None:
            save_dict['metrics'] = metrics
        
        torch.save(save_dict, model_path)
        print(f"Model saved to: {model_path}")

def main():
    """Main training function"""
    # Parse configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    config = parse_config(config_path)
    
    print("=== ChestMNIST Poisson Denoising ===")
    print(f"Device: {config.device}")
    print(f"Physics - μ: {config.physics['mu']}, I₀: {config.physics['I0']}")
    
    # Create datasets and data loaders
    (dataset_train, dataset_val, dataset_test, 
     dataloader_train, dataloader_val, dataloader_test) = create_datasets_and_loaders(config)
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(dataset_train)}")
    print(f"  Val: {len(dataset_val)}")  
    print(f"  Test: {len(dataset_test)}")
    
    # Create model and optimizer
    physics_simulator, conditional_denoiser, optimizer = create_model_and_optimizer(config)
    
    print(f"Model parameters: {sum(p.numel() for p in conditional_denoiser.parameters()):,}")
    
    # Run animated training process
    train_losses, val_losses, training_history = animate_training_process(
        config, physics_simulator, conditional_denoiser, optimizer,
        dataloader_train, dataloader_val, dataset_test
    )
    
    # Final evaluation
    print("\n=== Final Evaluation on Test Set ===")
    metrics = compute_metrics(conditional_denoiser, physics_simulator, dataloader_test, config.device)
    
    print(f"Test MSE: {metrics['mse']:.6f}")
    print(f"Test PSNR: {metrics['psnr']:.2f} dB")
    
    # Save model
    save_model(physics_simulator, conditional_denoiser, optimizer, config, training_history, metrics)
    
    print("\n=== Training Complete ===")
    return physics_simulator, conditional_denoiser, metrics

if __name__ == "__main__":
    physics_simulator, conditional_denoiser, metrics = main()