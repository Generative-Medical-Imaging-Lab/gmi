#!/usr/bin/env python3
"""
Simple script to run diffusion model training from a configuration file.
This script can be used for automated training runs without user interaction.
"""

import sys
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import gmi
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    """Main function to run training from config file."""
    
    parser = argparse.ArgumentParser(description="Train diffusion model from config file")
    parser.add_argument("config_path", type=str, help="Path to YAML configuration file")
    parser.add_argument("--device", type=str, default=None, help="Device to use (default: auto-detect)")
    parser.add_argument("--experiment-name", type=str, default=None, help="Override experiment name")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--method", type=str, choices=["class", "instance"], default="class", 
                       help="Training method: 'class' (recommended) or 'instance'")
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config_path)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    print(f"🚀 Starting diffusion model training from config: {config_path}")
    print(f"Method: {args.method}")
    print(f"Device: {args.device or 'auto-detect'}")
    print("=" * 60)
    
    try:
        from gmi.diffusion.core import DiffusionModel
        
        if args.method == "class":
            # Use class method (recommended)
            train_losses, val_losses, eval_metrics = DiffusionModel.train_from_config_file(
                config_path=args.config_path,
                device=args.device,
                experiment_name=args.experiment_name,
                output_dir=args.output_dir
            )
        else:
            # Use instance method
            import yaml
            from gmi.config import load_object_from_dict
            
            # Load config
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Load diffusion backbone
            diffusion_backbone = load_object_from_dict(config_dict['diffusion_backbone'])
            
            # Set device
            device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
            diffusion_backbone = diffusion_backbone.to(device)
            
            # Create diffusion model
            diffusion_model = DiffusionModel(diffusion_backbone=diffusion_backbone)
            
            # Train from config
            train_losses, val_losses, eval_metrics = diffusion_model.train_diffusion_model_from_config(
                config_dict=config_dict,
                device=device,
                experiment_name=args.experiment_name,
                output_dir=args.output_dir
            )
        
        print("\n✅ Training completed successfully!")
        print(f"Final train loss: {train_losses[-1]:.4f}")
        if val_losses:
            print(f"Final validation loss: {val_losses[-1]:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 