#!/usr/bin/env python3
"""
Example script for training a diffusion model using the GMI package.
This script demonstrates how to use the train_diffusion_model command.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import gmi
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """Main function to demonstrate diffusion model training."""
    
    print("🚀 GMI Diffusion Model Training Example")
    print("=" * 50)
    
    # Check if we're in the right directory
    config_path = Path(__file__).parent / "diffusion_training_config.yaml"
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("Please make sure you're running this from the correct directory.")
        return
    
    print(f"✅ Found configuration file: {config_path}")
    
    # Import the command function
    try:
        from gmi.commands.train_diffusion_model import train_diffusion_model
        print("✅ Successfully imported train_diffusion_model command")
    except ImportError as e:
        print(f"❌ Failed to import train_diffusion_model: {e}")
        return
    
    # Train the diffusion model
    print("\n🎯 Starting diffusion model training...")
    print("This will train an unconditional diffusion model on BloodMNIST.")
    print("The model will learn to generate blood cell images from noise.")
    print("During training, the model will generate samples using reverse process sampling.")
    
    try:
        train_losses, val_losses, eval_metrics = train_diffusion_model(
            config_path=str(config_path),
            device=None,  # Auto-detect
            experiment_name="example_diffusion_training"
        )
        
        print("\n✅ Training completed successfully!")
        print(f"Final training loss: {train_losses[-1]:.4f}")
        if val_losses:
            print(f"Final validation loss: {val_losses[-1]:.4f}")
        
        print("\n📁 Results saved to: gmi_data/outputs/example_diffusion_training/")
        print("   - best_model.pth: Best model checkpoint")
        print("   - final_config.yaml: Final configuration used")
        print("   - test_samples_epoch_*.png: Generated blood cell samples during training")
        print("   - WandB logs: Generated samples logged to WandB (if enabled)")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 