"""
Command-line interface for GMI.
Provides CLI commands for training and evaluating models.
"""

import argparse
import sys
from pathlib import Path
import torch

from ..tasks.reconstruction import ImageReconstructionTask
from ..config import load_components_from_yaml

def train_image_reconstructor():
    """CLI command to train an image reconstruction model from YAML config."""
    parser = argparse.ArgumentParser(description="Train an image reconstruction model")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--device", type=str, default=None, help="Device to use (default: auto-detect)")
    parser.add_argument("--experiment-name", type=str, default=None, help="Override experiment name")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        # Load components from config
        components = load_components_from_yaml(config_path)
        
        # Extract components
        dataset = components.get('dataset')
        measurement_simulator = components.get('measurement_simulator')
        image_reconstructor = components.get('image_reconstructor')
        
        if not all([dataset, measurement_simulator, image_reconstructor]):
            print("Error: Configuration must contain 'dataset', 'measurement_simulator', and 'image_reconstructor'")
            sys.exit(1)
        
        # Set device
        device = args.device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {device}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Create task
        task = ImageReconstructionTask(
            image_dataset=dataset,
            measurement_simulator=measurement_simulator,
            image_reconstructor=image_reconstructor,
            device=device
        )
        
        # Get experiment name
        experiment_name = args.experiment_name
        if experiment_name is None:
            # Try to get from config
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            experiment_name = config.get('experiment_name', 'unnamed_experiment')
        
        print(f"Starting training for experiment: {experiment_name}")
        
        # Train the model
        train_losses, val_losses, eval_metrics = task.train_image_reconstructor(
            experiment_name=experiment_name
        )
        
        print(f"Training completed!")
        print(f"Final train loss: {train_losses[-1]:.4f}")
        if val_losses:
            print(f"Final validation loss: {val_losses[-1]:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def evaluate_image_reconstructor():
    """CLI command to evaluate an image reconstruction model from YAML config."""
    parser = argparse.ArgumentParser(description="Evaluate an image reconstruction model")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device to use (default: auto-detect)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        # Load components from config
        components = load_components_from_yaml(config_path)
        
        # Extract components
        dataset = components.get('dataset')
        measurement_simulator = components.get('measurement_simulator')
        image_reconstructor = components.get('image_reconstructor')
        
        if not all([dataset, measurement_simulator, image_reconstructor]):
            print("Error: Configuration must contain 'dataset', 'measurement_simulator', and 'image_reconstructor'")
            sys.exit(1)
        
        # Set device
        device = args.device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {device}")
        
        # Create task
        task = ImageReconstructionTask(
            image_dataset=dataset,
            measurement_simulator=measurement_simulator,
            image_reconstructor=image_reconstructor,
            device=device
        )
        
        # Load model checkpoint
        print(f"Loading model from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        task.image_reconstructor.load_state_dict(checkpoint)
        
        # TODO: Implement evaluation logic
        print("Evaluation functionality not yet implemented")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="GMI Command Line Interface")
    parser.add_argument("command", choices=["train", "evaluate"], help="Command to run")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--device", type=str, default=None, help="Device to use (default: auto-detect)")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_image_reconstructor()
    elif args.command == "evaluate":
        evaluate_image_reconstructor()
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1) 