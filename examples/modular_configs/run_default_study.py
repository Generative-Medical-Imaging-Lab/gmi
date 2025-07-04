#!/usr/bin/env python3
"""
Run a default image reconstruction study using the training config defaults.

This script runs a single training experiment using the default components
defined in training_config.yaml without any overrides.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gmi.commands.train_image_reconstructor import train_image_reconstructor_from_configs

def main():
    """Run the default study using training config defaults."""
    
    # Path to the training config (contains all defaults)
    config_path = "examples/modular_configs/training_config.yaml"
    
    print("🚀 Running default image reconstruction study...")
    print(f"📁 Using config: {config_path}")
    print("📊 Using default components from training config")
    
    try:
        # Run training with default config (no overrides)
        train_losses, val_losses, eval_metrics = train_image_reconstructor_from_configs(
            config_paths=[config_path],
            device="cuda"  # Explicitly use GPU
        )
        
        print("✅ Default study completed successfully!")
        print(f"📈 Final train loss: {train_losses[-1]:.4f}")
        if val_losses:
            print(f"📊 Final validation loss: {val_losses[-1]:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error running default study: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 