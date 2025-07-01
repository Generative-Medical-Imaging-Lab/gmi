#!/usr/bin/env python3
"""
GMI - Generative Medical Imaging Laboratory
Command-line interface for GMI package
"""

import click
from gmi.commands import visualize_dataset

# Known commands list
KNOWN_COMMANDS = {
    "visualize_dataset": visualize_dataset.main
}

@click.group()
@click.version_option(version="0.0", prog_name="gmi")
def cli():
    """
    🏥 Generative Medical Imaging Laboratory (GMI) - Command Line Interface
    
    A comprehensive toolkit for generative medical imaging research.
    
    \b
    Examples:
        gmi visualize_dataset --dataset mnist
        gmi visualize_dataset --dataset medmnist
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
    # Assert that dataset argument is provided (explicit check)
    assert dataset is not None, "Dataset name must be provided with --dataset argument"
    assert len(dataset.strip()) > 0, "Dataset name cannot be empty"
    
    # Call the actual command function
    visualize_dataset.visualize_dataset(dataset)

if __name__ == "__main__":
    cli() 