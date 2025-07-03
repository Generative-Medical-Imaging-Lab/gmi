"""
Command for training image reconstruction models from YAML configuration.
"""

import click
import yaml
import torch
from pathlib import Path
from typing import Dict, Any

from ..tasks.reconstruction import ImageReconstructionTask
from ..config import load_components_from_dict


def train_image_reconstructor(config_path: str, device: str = None, experiment_name: str = None):
    """
    Train an image reconstruction model from YAML configuration.
    
    Args:
        config_path: Path to YAML configuration file
        device: Device to use (default: auto-detect)
        experiment_name: Override experiment name
    """
    # Load full YAML config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Only keep component keys
    component_keys = ['dataset', 'measurement_simulator', 'image_reconstructor']
    components_config = {k: v for k, v in config_dict.items() if k in component_keys}
    
    # Load components
    components = load_components_from_dict(components_config)
    
    # Extract components
    dataset = components.get('dataset')
    measurement_simulator = components.get('measurement_simulator')
    image_reconstructor = components.get('image_reconstructor')
    
    # Load validation dataset if specified
    validation_dataset = None
    if 'validation_dataset' in config_dict:
        validation_config = {k: v for k, v in config_dict.items() if k == 'validation_dataset'}
        validation_components = load_components_from_dict(validation_config)
        validation_dataset = validation_components.get('validation_dataset')
    
    # Load test dataset if specified
    test_dataset = None
    if 'test_dataset' in config_dict:
        test_config = {k: v for k, v in config_dict.items() if k == 'test_dataset'}
        test_components = load_components_from_dict(test_config)
        test_dataset = test_components.get('test_dataset')
    
    if not all([dataset, measurement_simulator, image_reconstructor]):
        raise click.UsageError("Configuration must contain 'dataset', 'measurement_simulator', and 'image_reconstructor'")
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    click.echo(f"Using device: {device}")
    if device == 'cuda':
        click.echo(f"GPU: {torch.cuda.get_device_name(0)}")
    
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
    
    click.echo(f"Starting training for experiment: {experiment_name}")
    
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
    
    click.echo(f"Training completed!")
    click.echo(f"Final train loss: {train_losses[-1]:.4f}")
    if val_losses:
        click.echo(f"Final validation loss: {val_losses[-1]:.4f}")
    
    return train_losses, val_losses, eval_metrics 