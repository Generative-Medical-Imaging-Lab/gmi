"""
Command for training image reconstruction models from YAML configuration.
"""

import yaml
import torch
import argparse
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

from ..tasks.reconstruction import ImageReconstructionTask
from ..config import load_components_from_dict, load_and_merge_configs, load_object_from_dict


def train_image_reconstructor_from_configs(config_paths: List[Union[str, Path]], 
                                         device: Optional[str] = None, 
                                         experiment_name: Optional[str] = None,
                                         train_dataset_path: Optional[str] = None,
                                         validation_dataset_path: Optional[str] = None,
                                         test_dataset_path: Optional[str] = None,
                                         measurement_simulator_path: Optional[str] = None,
                                         image_reconstructor_path: Optional[str] = None):
    """
    Train an image reconstruction model from multiple YAML configuration files.
    
    Args:
        config_paths: List of paths to YAML configuration files
        device: Device to use (default: auto-detect)
        experiment_name: Override experiment name
        train_dataset_path: Optional path to override train dataset config
        validation_dataset_path: Optional path to override validation dataset config
        test_dataset_path: Optional path to override test dataset config
        measurement_simulator_path: Optional path to override measurement simulator config
        image_reconstructor_path: Optional path to override image reconstructor config
        
    Returns:
        Tuple of (train_losses, val_losses, eval_metrics)
    """
    # Load and merge all config files
    config_dict = load_and_merge_configs(config_paths)
    
    # Override components with specific config files if provided
    if train_dataset_path:
        train_config = load_and_merge_configs([train_dataset_path])
        # Map the component name to the standard key
        for key, value in train_config.items():
            if 'train_dataset' in key or 'dataset' in key:
                config_dict['train_dataset'] = value
                break
    
    if validation_dataset_path:
        val_config = load_and_merge_configs([validation_dataset_path])
        for key, value in val_config.items():
            if 'validation_dataset' in key:
                config_dict['validation_dataset'] = value
                break
    
    if test_dataset_path:
        test_config = load_and_merge_configs([test_dataset_path])
        for key, value in test_config.items():
            if 'test_dataset' in key:
                config_dict['test_dataset'] = value
                break
    
    if measurement_simulator_path:
        sim_config = load_and_merge_configs([measurement_simulator_path])
        for key, value in sim_config.items():
            if 'measurement_simulator' in key or 'noise' in key:
                config_dict['measurement_simulator'] = value
                break
    
    if image_reconstructor_path:
        recon_config = load_and_merge_configs([image_reconstructor_path])
        for key, value in recon_config.items():
            if 'image_reconstructor' in key or 'linear_conv' in key or 'unet' in key or 'cnn' in key:
                config_dict['image_reconstructor'] = value
                break
    
    return _train_from_config_dict(config_dict, device, experiment_name)


def train_image_reconstructor(config_path: str, device: Optional[str] = None, experiment_name: Optional[str] = None):
    """
    Train an image reconstruction model from a single YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        device: Device to use (default: auto-detect)
        experiment_name: Override experiment name
        
    Returns:
        Tuple of (train_losses, val_losses, eval_metrics)
    """
    # Load full YAML config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return _train_from_config_dict(config_dict, device, experiment_name)


def _train_from_config_dict(config_dict: Dict[str, Any], device: Optional[str] = None, experiment_name: Optional[str] = None):
    """
    Internal function to train from a configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        device: Device to use (default: auto-detect)
        experiment_name: Override experiment name
        
    Returns:
        Tuple of (train_losses, val_losses, eval_metrics)
    """
    # Extract components using the new naming convention
    train_dataset = config_dict.get('train_dataset')
    validation_dataset = config_dict.get('validation_dataset')
    test_dataset = config_dict.get('test_dataset')
    measurement_simulator = config_dict.get('measurement_simulator')
    image_reconstructor = config_dict.get('image_reconstructor')
    
    # Load components if they are config dictionaries
    if isinstance(train_dataset, dict):
        train_dataset = load_object_from_dict(train_dataset)
    if isinstance(validation_dataset, dict):
        validation_dataset = load_object_from_dict(validation_dataset)
    if isinstance(test_dataset, dict):
        test_dataset = load_object_from_dict(test_dataset)
    if isinstance(measurement_simulator, dict):
        measurement_simulator = load_object_from_dict(measurement_simulator)
    if isinstance(image_reconstructor, dict):
        image_reconstructor = load_object_from_dict(image_reconstructor)
    
    # Use train_dataset as the main dataset
    dataset = train_dataset
    
    # Validation and test datasets are already loaded above
    
    if not all([dataset, measurement_simulator, image_reconstructor]):
        raise ValueError("Configuration must contain 'dataset', 'measurement_simulator', and 'image_reconstructor'")
    
    # Set device
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
    if experiment_name is None:
        experiment_name = config_dict.get('experiment_name', 'unnamed_experiment')
    
    print(f"Starting training for experiment: {experiment_name}")
    
    # Save the final combined config to the experiment directory
    experiment_dir = Path(f"gmi_data/outputs/{experiment_name}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    config_save_path = experiment_dir / "final_config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    print(f"Saved final configuration to: {config_save_path}")
    
    # Extract training configuration
    training_config = config_dict.get('training', {})
    
    # Train the model with training config using updated parameter names
    train_losses, val_losses, eval_metrics = task.train_image_reconstructor(
        val_data=validation_dataset,
        test_data=test_dataset,
        experiment_name=experiment_name,
        num_epochs=training_config.get('num_epochs', 100),
        num_iterations_train=training_config.get('num_iterations_train', 100),
        learning_rate=training_config.get('learning_rate', 0.001),
        train_batch_size=training_config.get('batch_size', 4),
        val_batch_size=training_config.get('batch_size', 4),
        test_batch_size=training_config.get('batch_size', 4),
        train_num_workers=training_config.get('num_workers', 4),
        val_num_workers=training_config.get('num_workers', 4),
        test_num_workers=training_config.get('num_workers', 4),
        shuffle_train=training_config.get('shuffle_train', True),
        shuffle_val=training_config.get('shuffle_val', True),
        shuffle_test=training_config.get('shuffle_test', False),
        use_ema=training_config.get('use_ema', True),
        ema_decay=training_config.get('ema_decay', 0.999),
        early_stopping=training_config.get('early_stopping', True),
        patience=training_config.get('patience', 10),
        val_loss_smoothing=training_config.get('val_loss_smoothing', 0.9),
        min_delta=float(training_config.get('min_delta', 1e-6)),
        num_iterations_val=training_config.get('num_iterations_val', 10),
        num_iterations_test=training_config.get('num_iterations_test', 10),
        verbose=training_config.get('verbose', True),
        very_verbose=training_config.get('very_verbose', False),
        wandb_project=training_config.get('wandb_project', None),
        wandb_config=training_config.get('wandb_config', None),
        save_checkpoints=training_config.get('save_checkpoints', True),
        test_plot_vmin=training_config.get('test_plot_vmin', 0),
        test_plot_vmax=training_config.get('test_plot_vmax', 1),
        test_save_plots=training_config.get('test_save_plots', True)
    )
    
    print(f"Training completed!")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"Final validation loss: {val_losses[-1]:.4f}")
    
    return train_losses, val_losses, eval_metrics 