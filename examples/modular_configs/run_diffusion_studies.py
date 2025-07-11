#!/usr/bin/env python3
"""
Run modular diffusion model training studies.
This script demonstrates how to train diffusion models with different configurations.
"""

import os
import sys
from pathlib import Path
import yaml

# Add the parent directory to the path so we can import gmi
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def run_diffusion_study(dataset_name, backbone_name, experiment_name=None):
    """Run a single diffusion training study."""
    
    if experiment_name is None:
        experiment_name = f"{dataset_name}_{backbone_name}"
    
    print(f"\n🎯 Running diffusion study: {experiment_name}")
    print(f"   Dataset: {dataset_name}")
    print(f"   Backbone: {backbone_name}")
    
    # Load the base training config
    base_config_path = Path(__file__).parent / "diffusion_training_config.yaml"
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with specific components
    dataset_config_path = Path(__file__).parent / "datasets" / f"{dataset_name}_train.yaml"
    backbone_config_path = Path(__file__).parent / "diffusion_backbones" / f"{backbone_name}.yaml"
    
    # Load dataset config
    if dataset_config_path.exists():
        with open(dataset_config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
            # Extract the dataset config (first key)
            dataset_key = list(dataset_config.keys())[0]
            config['train_dataset'] = dataset_config[dataset_key]
            config['validation_dataset'] = dataset_config[dataset_key]
            config['test_dataset'] = dataset_config[dataset_key]
    else:
        print(f"⚠️  Dataset config not found: {dataset_config_path}")
        return False
    
    # Load backbone config
    if backbone_config_path.exists():
        with open(backbone_config_path, 'r') as f:
            backbone_config = yaml.safe_load(f)
            # Extract the backbone config (first key)
            backbone_key = list(backbone_config.keys())[0]
            config['diffusion_backbone'] = backbone_config[backbone_key]
    else:
        print(f"⚠️  Backbone config not found: {backbone_config_path}")
        return False
    
    # Update experiment name
    config['experiment_name'] = experiment_name
    
    # Save the combined config
    output_dir = Path("gmi_data/outputs") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    combined_config_path = output_dir / "combined_config.yaml"
    with open(combined_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    # Train the model
    try:
        from gmi.commands.train_diffusion_model import train_diffusion_model_from_configs
        
        train_losses, val_losses, eval_metrics = train_diffusion_model_from_configs(
            config_paths=[str(combined_config_path)],
            experiment_name=experiment_name
        )
        
        print(f"✅ Completed diffusion study: {experiment_name}")
        print(f"   Final train loss: {train_losses[-1]:.4f}")
        if val_losses:
            print(f"   Final validation loss: {val_losses[-1]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed diffusion study {experiment_name}: {e}")
        return False

def main():
    """Main function to run diffusion studies."""
    
    print("🚀 GMI Modular Diffusion Studies")
    print("=" * 50)
    
    # Define the studies to run
    studies = [
        # (dataset_name, backbone_name, experiment_name)
        ("mnist", "simple_unet_1ch", "mnist_simple_unet_1ch"),
        ("mnist", "complex_unet_1ch", "mnist_complex_unet_1ch"),
        ("bloodmnist", "simple_unet_3ch", "bloodmnist_simple_unet_3ch"),
        ("bloodmnist", "complex_unet_3ch", "bloodmnist_complex_unet_3ch"),
        ("chestmnist", "simple_unet_1ch", "chestmnist_simple_unet_1ch"),
        ("chestmnist", "complex_unet_1ch", "chestmnist_complex_unet_1ch"),
    ]
    
    print(f"📋 Running {len(studies)} diffusion studies:")
    for dataset, backbone, experiment in studies:
        print(f"   - {experiment}")
    
    # Run the studies
    successful_studies = 0
    total_studies = len(studies)
    
    for dataset, backbone, experiment in studies:
        if run_diffusion_study(dataset, backbone, experiment):
            successful_studies += 1
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Successful studies: {successful_studies}/{total_studies}")
    print(f"   Failed studies: {total_studies - successful_studies}/{total_studies}")
    
    if successful_studies == total_studies:
        print("🎉 All diffusion studies completed successfully!")
    else:
        print("⚠️  Some studies failed. Check the output above for details.")

if __name__ == "__main__":
    main() 