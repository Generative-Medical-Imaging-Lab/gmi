#!/usr/bin/env python3
"""
GMI - Generative Medical Imaging Laboratory
Command-line interface for GMI package
"""

import click
from gmi.commands import visualize_dataset

@click.group()
@click.version_option(version="0.0", prog_name="gmi")
def cli():
    """
    🏥 Generative Medical Imaging Laboratory (GMI) - Command Line Interface
    
    A comprehensive toolkit for generative medical imaging research.
    
    \b
    Examples:
        gmi visualize-dataset --dataset mnist
        gmi visualize-dataset --dataset medmnist
    """
    pass

# Add the visualize_dataset command
@cli.command()
@click.option('--dataset', '-d', required=True, type=str,
              help='Name of the dataset to visualize (e.g., mnist, medmnist)')
def visualize_dataset_cmd(dataset):
    """
    📊 Visualize a specified dataset from the GMI package.
    
    This command loads and displays visualizations of the specified dataset,
    including sample images, statistics, and metadata.
    
    \b
    Examples:
        gmi visualize-dataset --dataset mnist
        gmi visualize-dataset --dataset medmnist
        gmi visualize-dataset -d organamnist
    """
    # Validate dataset parameter
    if not dataset or not dataset.strip():
        raise click.UsageError("Dataset name cannot be empty. Please provide a valid dataset name.")
    
    # Call the actual command function
    visualize_dataset.visualize_dataset(dataset)

# Add the train_image_reconstructor command
@cli.command()
@click.argument('config', type=click.Path(exists=True))
@click.option('--device', type=str, default=None, help='Device to use (default: auto-detect)')
@click.option('--experiment-name', type=str, default=None, help='Override experiment name')
@click.option('--train-dataset', type=click.Path(exists=True), help='Override train dataset config file')
@click.option('--validation-dataset', type=click.Path(exists=True), help='Override validation dataset config file')
@click.option('--test-dataset', type=click.Path(exists=True), help='Override test dataset config file')
@click.option('--measurement-simulator', type=click.Path(exists=True), help='Override measurement simulator config file')
@click.option('--image-reconstructor', type=click.Path(exists=True), help='Override image reconstructor config file')
def train_image_reconstructor(config, device, experiment_name, train_dataset, validation_dataset, test_dataset, measurement_simulator, image_reconstructor):
    """
    🚀 Train an image reconstruction model from YAML configuration.
    
    This command loads a YAML configuration file and trains an image reconstruction model
    using the specified dataset, measurement simulator, and image reconstructor.
    
    The config file can contain defaults for all components, which can be overridden
    using the optional component-specific config files.
    
    \b
    Examples:
        gmi train-image-reconstructor config.yaml
        gmi train-image-reconstructor config.yaml --device cuda
        gmi train-image-reconstructor config.yaml --experiment-name my_experiment
        gmi train-image-reconstructor config.yaml --train-dataset datasets/mnist_train.yaml --image-reconstructor networks/linear_conv_1ch.yaml
    """
    from gmi.commands.train_image_reconstructor import train_image_reconstructor_from_configs
    
    # Define input variables directly (no argument parsing)
    config_paths = [config]
    device = device
    experiment_name = experiment_name
    train_dataset_path = train_dataset
    validation_dataset_path = validation_dataset
    test_dataset_path = test_dataset
    measurement_simulator_path = measurement_simulator
    image_reconstructor_path = image_reconstructor
    
    # Call the command function with all arguments
    train_image_reconstructor_from_configs(
        config_paths=config_paths,
        device=device,
        experiment_name=experiment_name,
        train_dataset_path=train_dataset_path,
        validation_dataset_path=validation_dataset_path,
        test_dataset_path=test_dataset_path,
        measurement_simulator_path=measurement_simulator_path,
        image_reconstructor_path=image_reconstructor_path
    )

# Add the train_diffusion_model command
@cli.command()
@click.argument('config', type=click.Path(exists=True))
@click.option('--device', type=str, default=None, help='Device to use (default: auto-detect)')
@click.option('--experiment-name', type=str, default=None, help='Override experiment name')
@click.option('--train-dataset', type=click.Path(exists=True), help='Override train dataset config file')
@click.option('--validation-dataset', type=click.Path(exists=True), help='Override validation dataset config file')
@click.option('--test-dataset', type=click.Path(exists=True), help='Override test dataset config file')
@click.option('--diffusion-backbone', type=click.Path(exists=True), help='Override diffusion backbone config file')
def train_diffusion_model(config, device, experiment_name, train_dataset, validation_dataset, test_dataset, diffusion_backbone):
    """
    🚀 Train a diffusion model from YAML configuration.
    
    This command loads a YAML configuration file and trains a diffusion model
    using the specified dataset and diffusion backbone.
    
    The config file can contain defaults for all components, which can be overridden
    using the optional component-specific config files.
    
    \b
    Examples:
        gmi train-diffusion-model config.yaml
        gmi train-diffusion-model config.yaml --device cuda
        gmi train-diffusion-model config.yaml --experiment-name my_experiment
        gmi train-diffusion-model config.yaml --train-dataset datasets/mnist_train.yaml --diffusion-backbone networks/unet.yaml
    """
    from gmi.commands.train_diffusion_model import train_diffusion_model_from_configs
    
    # Define input variables directly (no argument parsing)
    config_paths = [config]
    device = device
    experiment_name = experiment_name
    train_dataset_path = train_dataset
    validation_dataset_path = validation_dataset
    test_dataset_path = test_dataset
    diffusion_backbone_path = diffusion_backbone
    
    # Call the command function with all arguments
    train_diffusion_model_from_configs(
        config_paths=config_paths,
        device=device,
        experiment_name=experiment_name,
        train_dataset_path=train_dataset_path,
        validation_dataset_path=validation_dataset_path,
        test_dataset_path=test_dataset_path,
        diffusion_backbone_path=diffusion_backbone_path
    )

# Add the evaluate_image_reconstructor command
@cli.command()
@click.argument('config', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--device', type=str, default=None, help='Device to use (default: auto-detect)')
@click.option('--output-dir', type=click.Path(), default=None, help='Output directory for results')
def evaluate_image_reconstructor(config, model_path, device, output_dir):
    """
    📊 Evaluate an image reconstruction model from YAML configuration.
    
    This command loads a trained model checkpoint and evaluates it using the specified
    configuration and dataset.
    
    \b
    Examples:
        gmi evaluate-image-reconstructor config.yaml model.pth
        gmi evaluate-image-reconstructor config.yaml model.pth --device cuda
        gmi evaluate-image-reconstructor config.yaml model.pth --output-dir results/
    """
    from gmi.tasks.reconstruction import ImageReconstructionTask
    from gmi.config import load_components_from_yaml
    import torch
    
    # Load configuration
    components = load_components_from_yaml(config)
    
    # Extract components
    dataset = components.get('dataset')
    measurement_simulator = components.get('measurement_simulator')
    image_reconstructor = components.get('image_reconstructor')
    
    if not all([dataset, measurement_simulator, image_reconstructor]):
        raise click.UsageError("Configuration must contain 'dataset', 'measurement_simulator', and 'image_reconstructor'")
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    click.echo(f"Using device: {device}")
    
    # Create task
    task = ImageReconstructionTask(
        image_dataset=dataset,
        measurement_simulator=measurement_simulator,
        image_reconstructor=image_reconstructor,
        device=device
    )
    
    # Load model checkpoint
    click.echo(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    task.image_reconstructor.load_state_dict(checkpoint)
    
    # TODO: Implement evaluation logic
    click.echo("Evaluation functionality not yet implemented")




if __name__ == "__main__":
    cli() 