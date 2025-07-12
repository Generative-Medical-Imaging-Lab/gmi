#!/usr/bin/env python3
"""
Test script to verify the harmonized diffusion config works correctly.
This script tests the config loading and model creation without running full training.
"""

import os
import sys
import yaml
import torch
from pathlib import Path

# Add the parent directory to the path so we can import gmi
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_config_loading():
    """Test that the harmonized config file can be loaded correctly."""
    print("Testing harmonized config loading...")
    
    config_path = Path(__file__).parent / 'medmnist_diffusion_harmonized.yaml'
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    print("✅ Config file loaded successfully")
    print(f"   Experiment name: {config_dict.get('experiment_name')}")
    print(f"   Has train_dataset: {'train_dataset' in config_dict}")
    print(f"   Has diffusion_backbone: {'diffusion_backbone' in config_dict}")
    print(f"   Has forward_SDE: {'forward_SDE' in config_dict}")
    print(f"   Has training section: {'training' in config_dict}")
    
    return config_dict

def test_component_loading(config_dict):
    """Test that all components can be loaded from the config."""
    print("\nTesting component loading...")
    
    from gmi.config import load_object_from_dict
    
    # Create required directory for MedMNIST
    root_dir = "gmi_data/datasets/MedMNIST/BloodMNIST"
    os.makedirs(root_dir, exist_ok=True)
    print(f"Created directory: {root_dir}")
    
    # Test dataset loading
    try:
        train_dataset = load_object_from_dict(config_dict['train_dataset'])
        print("✅ Train dataset loaded successfully")
        print(f"   Dataset type: {type(train_dataset)}")
        print(f"   Dataset length: {len(train_dataset)}")
    except Exception as e:
        print(f"❌ Failed to load train dataset: {e}")
        return False
    
    # Test diffusion backbone loading
    try:
        diffusion_backbone = load_object_from_dict(config_dict['diffusion_backbone'])
        print("✅ Diffusion backbone loaded successfully")
        print(f"   Backbone type: {type(diffusion_backbone)}")
    except Exception as e:
        print(f"❌ Failed to load diffusion backbone: {e}")
        return False
    
    # Test forward_SDE loading (optional)
    if 'forward_SDE' in config_dict:
        try:
            forward_SDE = load_object_from_dict(config_dict['forward_SDE'])
            print("✅ Forward SDE loaded successfully")
            print(f"   SDE type: {type(forward_SDE)}")
        except Exception as e:
            print(f"❌ Failed to load forward_SDE: {e}")
            return False
    
    # Test training_loss_fn loading (optional)
    if 'training_loss_fn' in config_dict:
        try:
            training_loss_fn = load_object_from_dict(config_dict['training_loss_fn'])
            print("✅ Training loss function loaded successfully")
            print(f"   Loss function type: {type(training_loss_fn)}")
        except Exception as e:
            print(f"❌ Failed to load training_loss_fn: {e}")
            return False
    
    # Test training_time_sampler loading (optional)
    if 'training_time_sampler' in config_dict:
        try:
            training_time_sampler = load_object_from_dict(config_dict['training_time_sampler'])
            print("✅ Training time sampler loaded successfully")
            print(f"   Sampler type: {type(training_time_sampler)}")
        except Exception as e:
            print(f"❌ Failed to load training_time_sampler: {e}")
            return False
    
    return True

def test_diffusion_model_creation(config_dict):
    """Test that a diffusion model can be created from the config."""
    print("\nTesting diffusion model creation...")
    
    from gmi.diffusion.core import DiffusionModel
    from gmi.config import load_object_from_dict
    
    try:
        # Load components
        diffusion_backbone = load_object_from_dict(config_dict['diffusion_backbone'])
        
        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        diffusion_backbone = diffusion_backbone.to(device)
        
        # Load optional components
        forward_SDE = None
        training_loss_fn = None
        training_time_sampler = None
        training_time_uncertainty_sampler = None
        
        if 'forward_SDE' in config_dict:
            forward_SDE = load_object_from_dict(config_dict['forward_SDE'])
        if 'training_loss_fn' in config_dict:
            training_loss_fn = load_object_from_dict(config_dict['training_loss_fn'])
        if 'training_time_sampler' in config_dict:
            training_time_sampler = load_object_from_dict(config_dict['training_time_sampler'])
        if 'training_time_uncertainty_sampler' in config_dict:
            training_time_uncertainty_sampler = load_object_from_dict(config_dict['training_time_uncertainty_sampler'])
        
        # Create diffusion model
        diffusion_model = DiffusionModel(
            diffusion_backbone=diffusion_backbone,
            forward_SDE=forward_SDE,
            training_loss_fn=training_loss_fn,
            training_time_sampler=training_time_sampler,
            training_time_uncertainty_sampler=training_time_uncertainty_sampler
        )
        
        print("✅ Diffusion model created successfully")
        print(f"   Model type: {type(diffusion_model)}")
        print(f"   Has train_diffusion_model_from_config method: {hasattr(diffusion_model, 'train_diffusion_model_from_config')}")
        
        return diffusion_model
        
    except Exception as e:
        print(f"❌ Failed to create diffusion model: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_config_method():
    """Test the train_diffusion_model_from_config method."""
    print("\nTesting train_diffusion_model_from_config method...")
    
    from gmi.diffusion.core import DiffusionModel
    from gmi.config import load_object_from_dict
    
    try:
        # Load config
        config_path = Path(__file__).parent / 'medmnist_diffusion_harmonized.yaml'
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Test the method exists
        diffusion_backbone = load_object_from_dict(config_dict['diffusion_backbone'])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        diffusion_backbone = diffusion_backbone.to(device)
        
        diffusion_model = DiffusionModel(diffusion_backbone=diffusion_backbone)
        
        # Check if the method exists
        if hasattr(diffusion_model, 'train_diffusion_model_from_config'):
            print("✅ train_diffusion_model_from_config method exists")
            
            # Test that it can be called (but don't actually train)
            import inspect
            sig = inspect.signature(diffusion_model.train_diffusion_model_from_config)
            params = list(sig.parameters.keys())
            print(f"   Method parameters: {params}")
            
            return True
        else:
            print("❌ train_diffusion_model_from_config method not found")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test config method: {e}")
        return False

def test_class_method():
    """Test the train_from_config_file class method."""
    print("\nTesting train_from_config_file class method...")
    
    from gmi.diffusion.core import DiffusionModel
    
    try:
        # Check if the class method exists
        if hasattr(DiffusionModel, 'train_from_config_file'):
            print("✅ train_from_config_file class method exists")
            
            # Test that it can be called (but don't actually train)
            import inspect
            sig = inspect.signature(DiffusionModel.train_from_config_file)
            params = list(sig.parameters.keys())
            print(f"   Method parameters: {params}")
            
            return True
        else:
            print("❌ train_from_config_file class method not found")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test class method: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Testing Harmonized Diffusion Config")
    print("=" * 50)
    
    # Test 1: Config loading
    config_dict = test_config_loading()
    if not config_dict:
        return False
    
    # Test 2: Component loading
    if not test_component_loading(config_dict):
        return False
    
    # Test 3: Diffusion model creation
    diffusion_model = test_diffusion_model_creation(config_dict)
    if diffusion_model is None:
        return False
    
    # Test 4: Config method
    if not test_config_method():
        return False
    
    # Test 5: Class method
    if not test_class_method():
        return False
    
    print("\n🎉 All tests passed! The harmonized config is working correctly.")
    print("\nYou can now use the config file with:")
    print("1. diffusion_model.train_diffusion_model_from_config(config_dict)")
    print("2. DiffusionModel.train_from_config_file('medmnist_diffusion_harmonized.yaml')")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 