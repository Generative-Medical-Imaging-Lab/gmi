#!/usr/bin/env python3
"""
Example script for training a diffusion model from a configuration file.
This demonstrates the new config-based training functionality.
"""

import os
import sys
import yaml
from pathlib import Path
import torch

# Add the parent directory to the path so we can import gmi
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def create_config_file():
    """Create a configuration file for the MedMNIST diffusion training."""
    
    config = {
        'experiment_name': 'medmnist_diffusion_from_config',
        
        # Dataset configurations
        'train_dataset': {
            'class': 'gmi.datasets.MedMNIST',
            'params': {
                'dataset_name': 'BloodMNIST',
                'size': 28,
                'split': 'train',
                'images_only': True
            }
        },
        
        'validation_dataset': {
            'class': 'gmi.datasets.MedMNIST',
            'params': {
                'dataset_name': 'BloodMNIST',
                'size': 28,
                'split': 'val',
                'images_only': True
            }
        },
        
        'test_dataset': {
            'class': 'gmi.datasets.MedMNIST',
            'params': {
                'dataset_name': 'BloodMNIST',
                'size': 28,
                'split': 'test',
                'images_only': True
            }
        },
        
        # Diffusion backbone configuration
        'diffusion_backbone': {
            'class': 'gmi.network.DiffusersUnet2D_Size28',
            'params': {
                'in_channels': 3,
                'out_channels': 3,
                'layers_per_block': 2,
                'block_out_channels': [32, 64, 128],
                'down_block_types': ['DownBlock2D', 'DownBlock2D', 'DownBlock2D'],
                'up_block_types': ['UpBlock2D', 'UpBlock2D', 'UpBlock2D']
            }
        },
        
        # Training configuration
        'training': {
            'num_epochs': 100,  # Reduced for faster training
            'num_iterations_train': 100,
            'num_iterations_val': 10,
            'num_iterations_test': 5,
            'learning_rate': 1e-4,
            'batch_size': 32,
            'num_workers': 2,
            'shuffle_train': True,
            'shuffle_val': True,
            'shuffle_test': False,
            'use_ema': True,
            'ema_decay': 0.999,
            'early_stopping': True,
            'patience': 20,
            'val_loss_smoothing': 0.9,
            'min_delta': 1e-6,
            'verbose': True,
            'very_verbose': False,
            'wandb_project': 'gmi-medmnist-diffusion-config',
            'wandb_config': None,
            'save_checkpoints': True,
            'test_plot_vmin': 0,
            'test_plot_vmax': 1,
            'test_save_plots': True,
            'final_test_iterations': 20,
            'reverse_t_start': 1.0,
            'reverse_t_end': 0.0,
            'reverse_spacing': 'linear',
            'reverse_sampler': 'euler',
            'reverse_timesteps': 50
        }
    }
    
    # Save the config file
    config_path = Path(__file__).parent / 'medmnist_diffusion_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Created configuration file: {config_path}")
    return config_path

def train_from_config_file(config_path):
    """Train a diffusion model from a configuration file."""
    
    print(f"🚀 Training diffusion model from config: {config_path}")
    print("=" * 60)
    
    # Import the diffusion model class
    from gmi.diffusion.core import DiffusionModel
    
    # Train using the class method
    train_losses, val_losses, eval_metrics = DiffusionModel.train_from_config_file(
        config_path=config_path,
        device=None,  # Auto-detect
        experiment_name=None,  # Use from config
        output_dir=None  # Use default
    )
    
    print("\n✅ Training completed successfully!")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"Final validation loss: {val_losses[-1]:.4f}")
    
    return train_losses, val_losses, eval_metrics

def train_from_config_dict():
    """Train a diffusion model from a configuration dictionary."""
    
    print("🚀 Training diffusion model from config dictionary")
    print("=" * 60)
    
    # Import required modules
    from gmi.diffusion.core import DiffusionModel
    from gmi.config import load_object_from_dict
    
    # Create configuration dictionary
    config_dict = {
        'experiment_name': 'medmnist_diffusion_dict',
        
        'train_dataset': {
            'class': 'gmi.datasets.MedMNIST',
            'params': {
                'dataset_name': 'BloodMNIST',
                'size': 28,
                'split': 'train',
                'images_only': True
            }
        },
        
        'validation_dataset': {
            'class': 'gmi.datasets.MedMNIST',
            'params': {
                'dataset_name': 'BloodMNIST',
                'size': 28,
                'split': 'val',
                'images_only': True
            }
        },
        
        'test_dataset': {
            'class': 'gmi.datasets.MedMNIST',
            'params': {
                'dataset_name': 'BloodMNIST',
                'size': 28,
                'split': 'test',
                'images_only': True
            }
        },
        
        'diffusion_backbone': {
            'class': 'gmi.network.DiffusersUnet2D_Size28',
            'params': {
                'in_channels': 3,
                'out_channels': 3,
                'layers_per_block': 2,
                'block_out_channels': [16, 32, 64],  # Smaller for faster training
                'down_block_types': ['DownBlock2D', 'DownBlock2D', 'DownBlock2D'],
                'up_block_types': ['UpBlock2D', 'UpBlock2D', 'UpBlock2D']
            }
        },
        
        'training': {
            'num_epochs': 50,  # Even smaller for demo
            'num_iterations_train': 50,
            'num_iterations_val': 5,
            'num_iterations_test': 3,
            'learning_rate': 1e-4,
            'batch_size': 16,
            'num_workers': 2,
            'shuffle_train': True,
            'shuffle_val': True,
            'shuffle_test': False,
            'use_ema': True,
            'ema_decay': 0.999,
            'early_stopping': True,
            'patience': 10,
            'val_loss_smoothing': 0.9,
            'min_delta': 1e-6,
            'verbose': True,
            'very_verbose': False,
            'wandb_project': None,  # Disable WandB for demo
            'wandb_config': None,
            'save_checkpoints': True,
            'test_plot_vmin': 0,
            'test_plot_vmax': 1,
            'test_save_plots': True,
            'final_test_iterations': 10,
            'reverse_t_start': 1.0,
            'reverse_t_end': 0.0,
            'reverse_spacing': 'linear',
            'reverse_sampler': 'euler',
            'reverse_timesteps': 30
        }
    }
    
    # Load diffusion backbone
    diffusion_backbone = load_object_from_dict(config_dict['diffusion_backbone'])
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diffusion_backbone = diffusion_backbone.to(device)
    
    # Create diffusion model
    diffusion_model = DiffusionModel(diffusion_backbone=diffusion_backbone)
    
    # Train using the instance method
    train_losses, val_losses, eval_metrics = diffusion_model.train_diffusion_model_from_config(
        config_dict=config_dict,
        device=device,
        experiment_name=None,
        output_dir=None
    )
    
    print("\n✅ Training completed successfully!")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"Final validation loss: {val_losses[-1]:.4f}")
    
    return train_losses, val_losses, eval_metrics

def main():
    """Main function to demonstrate config-based training."""
    
    print("🎯 GMI Diffusion Model Training from Config Example")
    print("=" * 60)
    
    # Check if we're in the right directory
    example_dir = Path(__file__).parent
    print(f"Working directory: {example_dir}")
    
    # Create config file
    config_path = create_config_file()
    
    # Choose training method
    print("\nChoose training method:")
    print("1. Train from config file (using class method)")
    print("2. Train from config dictionary (using instance method)")
    print("3. Both (for comparison)")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == '1':
        train_from_config_file(config_path)
    elif choice == '2':
        train_from_config_dict()
    elif choice == '3':
        print("\n" + "="*60)
        print("TRAINING FROM CONFIG FILE")
        print("="*60)
        train_from_config_file(config_path)
        
        print("\n" + "="*60)
        print("TRAINING FROM CONFIG DICTIONARY")
        print("="*60)
        train_from_config_dict()
    else:
        print("Invalid choice. Running both methods...")
        train_from_config_file(config_path)
        train_from_config_dict()
    
    print("\n🎉 All training examples completed!")
    print(f"Check the output directories for results and generated samples.")

if __name__ == "__main__":
    main() 