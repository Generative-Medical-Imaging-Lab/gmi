#!/usr/bin/env python3
"""
Run all modular studies using the GMI library directly (not the CLI).
This script demonstrates how to use the GMI Python API for modular experiments.
"""
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gmi.commands.train_image_reconstructor import train_image_reconstructor_from_configs

CONFIG_DIR = Path(__file__).parent

def discover_components(subdir):
    """Return a list of component names (file stems) in the given subdir."""
    return [f.stem for f in (CONFIG_DIR / subdir).glob("*.yaml")]

def main():
    # Discover train datasets, measurement simulators, and image reconstructors
    train_datasets = [f for f in discover_components("datasets") if f.endswith("_train")]
    simulators = discover_components("measurement_simulators")
    all_reconstructors = discover_components("image_reconstructors")
    training_path = CONFIG_DIR / "training_config.yaml"

    # Define channel mapping for datasets
    dataset_channels = {
        'mnist': 1,
        'chestmnist': 1,
        'bloodmnist': 3
    }

    # Filter reconstructors based on dataset channels
    def get_compatible_reconstructors(dataset_name):
        """Get reconstructorcls compatible with the given dataset."""
        channels = dataset_channels.get(dataset_name)
        if channels is None:
            print(f"Warning: Unknown dataset '{dataset_name}', using all reconstructors")
            return all_reconstructors
        
        # Filter reconstructors by channel suffix
        compatible_reconstructors = [r for r in all_reconstructors if r.endswith(f"_{channels}ch")]
        if not compatible_reconstructors:
            print(f"Warning: No {channels}-channel reconstructors found for {dataset_name}")
            return []
        
        return compatible_reconstructors

    # Calculate total studies with channel-based filtering
    total_studies = 0
    for dataset in train_datasets:
        dataset_name = dataset.replace('_train', '')
        compatible_reconstructors = get_compatible_reconstructors(dataset_name)
        total_studies += len(simulators) * len(compatible_reconstructors)

    print(f"Running {total_studies} studies with channel-based filtering:")
    print(f"Train datasets: {train_datasets}")
    print(f"Measurement simulators: {simulators}")
    print(f"All image reconstructors: {all_reconstructors}")
    print(f"Channel mapping: {dataset_channels}")

    study_count = 0
    results = []

    for dataset in train_datasets:
        dataset_name = dataset.replace('_train', '')
        compatible_reconstructors = get_compatible_reconstructors(dataset_name)
        
        print(f"\nProcessing dataset: {dataset_name} (compatible reconstructors: {compatible_reconstructors})")
        
        for simulator in simulators:
            for reconstructor in compatible_reconstructors:
                study_count += 1
                print(f"\n{'='*60}")
                # Create experiment name
                experiment_name = f"{dataset_name}_{simulator}_{reconstructor}"
                
                print(f"Study {study_count}/{total_studies}: {dataset_name} + {simulator} + {reconstructor}")
                print(f"Experiment name: {experiment_name}")
                print(f"{'='*60}")
                
                # Construct paths for this study
                train_dataset_path = CONFIG_DIR / "datasets" / f"{dataset}.yaml"
                val_dataset_path = CONFIG_DIR / "datasets" / f"{dataset.replace('_train', '_val')}.yaml"
                test_dataset_path = CONFIG_DIR / "datasets" / f"{dataset.replace('_train', '_test')}.yaml"
                simulator_path = CONFIG_DIR / "measurement_simulators" / f"{simulator}.yaml"
                reconstructor_path = CONFIG_DIR / "image_reconstructors" / f"{reconstructor}.yaml"
                
                try:
                    # Run training with component overrides
                    train_losses, val_losses, eval_metrics = train_image_reconstructor_from_configs(
                        config_paths=[training_path],
                        device="cuda",  # Explicitly use GPU
                        experiment_name=experiment_name,
                        train_dataset_path=str(train_dataset_path),
                        validation_dataset_path=str(val_dataset_path),
                        test_dataset_path=str(test_dataset_path),
                        measurement_simulator_path=str(simulator_path),
                        image_reconstructor_path=str(reconstructor_path)
                    )
                    
                    results.append({
                        'dataset': dataset_name,
                        'measurement_simulator': simulator,
                        'image_reconstructor': reconstructor,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'eval_metrics': eval_metrics
                    })
                    print(f"✓ Study {study_count} completed successfully")
                    
                    # Small delay to ensure WandB run is properly finished
                    import time
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"✗ Study {study_count} failed: {e}")
                    results.append({
                        'dataset': dataset_name,
                        'measurement_simulator': simulator,
                        'image_reconstructor': reconstructor,
                        'error': str(e)
                    })
    
    print(f"\n{'='*60}")
    print(f"All studies completed! {len([r for r in results if 'error' not in r])}/{len(results)} successful")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 