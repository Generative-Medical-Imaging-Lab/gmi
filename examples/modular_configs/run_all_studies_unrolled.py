#!/usr/bin/env python3
"""
Run all modular studies using explicit function calls (unrolled version).
This script runs all 27 studies (3 datasets x 3 noise levels x 3 reconstructors) with explicit calls.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gmi.commands.train_image_reconstructor import train_image_reconstructor_from_configs

CONFIG_DIR = Path(__file__).parent
OUTPUT_DIR = CONFIG_DIR / "outputs"

def run_single_study(experiment_name, dataset_name, simulator_name, reconstructor_name, study_num, total_studies):
    """Run a single study with error handling."""
    try:
        print(f"\n{'='*60}")
        print(f"Study {study_num}/{total_studies}: {dataset_name} + {simulator_name} + {reconstructor_name}")
        print(f"Experiment name: {experiment_name}")
        print(f"{'='*60}")
        
        # Construct paths for this study
        train_dataset_path = CONFIG_DIR / "datasets" / f"{dataset_name}_train.yaml"
        val_dataset_path = CONFIG_DIR / "datasets" / f"{dataset_name}_val.yaml"
        test_dataset_path = CONFIG_DIR / "datasets" / f"{dataset_name}_test.yaml"
        simulator_path = CONFIG_DIR / "measurement_simulators" / f"{simulator_name}.yaml"
        reconstructor_path = CONFIG_DIR / "image_reconstructors" / f"{reconstructor_name}.yaml"
        training_path = CONFIG_DIR / "training_config.yaml"
        
        # Run training with component overrides
        train_losses, val_losses, eval_metrics = train_image_reconstructor_from_configs(
            config_paths=[training_path],
            device="cuda",
            experiment_name=experiment_name,
            output_dir=str(OUTPUT_DIR),
            train_dataset_path=str(train_dataset_path),
            validation_dataset_path=str(val_dataset_path),
            test_dataset_path=str(test_dataset_path),
            measurement_simulator_path=str(simulator_path),
            image_reconstructor_path=str(reconstructor_path)
        )
        
        print(f"✓ Study {study_num} completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Study {study_num} failed: {e}")
        return False

def main():
    """Run all 27 studies with explicit function calls."""
    print("🚀 Running all 27 modular studies (unrolled version)")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📊 Total studies: 27 (3 datasets x 3 noise levels x 3 reconstructors)")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # MNIST studies (1 channel)
    run_single_study("mnist_low_noise_simple_cnn_1ch", "mnist", "low_noise", "simple_cnn_1ch", 1, 27)  # MNIST + low_noise + simple_cnn_1ch
    run_single_study("mnist_low_noise_linear_conv_1ch", "mnist", "low_noise", "linear_conv_1ch", 2, 27)  # MNIST + low_noise + linear_conv_1ch
    run_single_study("mnist_low_noise_diffusers_unet_28_1ch", "mnist", "low_noise", "diffusers_unet_28_1ch", 3, 27)  # MNIST + low_noise + diffusers_unet_28_1ch
    run_single_study("mnist_medium_noise_simple_cnn_1ch", "mnist", "medium_noise", "simple_cnn_1ch", 4, 27)  # MNIST + medium_noise + simple_cnn_1ch
    run_single_study("mnist_medium_noise_linear_conv_1ch", "mnist", "medium_noise", "linear_conv_1ch", 5, 27)  # MNIST + medium_noise + linear_conv_1ch
    run_single_study("mnist_medium_noise_diffusers_unet_28_1ch", "mnist", "medium_noise", "diffusers_unet_28_1ch", 6, 27)  # MNIST + medium_noise + diffusers_unet_28_1ch
    run_single_study("mnist_high_noise_simple_cnn_1ch", "mnist", "high_noise", "simple_cnn_1ch", 7, 27)  # MNIST + high_noise + simple_cnn_1ch
    run_single_study("mnist_high_noise_linear_conv_1ch", "mnist", "high_noise", "linear_conv_1ch", 8, 27)  # MNIST + high_noise + linear_conv_1ch
    run_single_study("mnist_high_noise_diffusers_unet_28_1ch", "mnist", "high_noise", "diffusers_unet_28_1ch", 9, 27)  # MNIST + high_noise + diffusers_unet_28_1ch
    
    # BloodMNIST studies (3 channels)
    run_single_study("bloodmnist_low_noise_simple_cnn_3ch", "bloodmnist", "low_noise", "simple_cnn_3ch", 10, 27)  # BloodMNIST + low_noise + simple_cnn_3ch
    run_single_study("bloodmnist_low_noise_linear_conv_3ch", "bloodmnist", "low_noise", "linear_conv_3ch", 11, 27)  # BloodMNIST + low_noise + linear_conv_3ch
    run_single_study("bloodmnist_low_noise_diffusers_unet_28_3ch", "bloodmnist", "low_noise", "diffusers_unet_28_3ch", 12, 27)  # BloodMNIST + low_noise + diffusers_unet_28_3ch
    run_single_study("bloodmnist_medium_noise_simple_cnn_3ch", "bloodmnist", "medium_noise", "simple_cnn_3ch", 13, 27)  # BloodMNIST + medium_noise + simple_cnn_3ch
    run_single_study("bloodmnist_medium_noise_linear_conv_3ch", "bloodmnist", "medium_noise", "linear_conv_3ch", 14, 27)  # BloodMNIST + medium_noise + linear_conv_3ch
    run_single_study("bloodmnist_medium_noise_diffusers_unet_28_3ch", "bloodmnist", "medium_noise", "diffusers_unet_28_3ch", 15, 27)  # BloodMNIST + medium_noise + diffusers_unet_28_3ch
    run_single_study("bloodmnist_high_noise_simple_cnn_3ch", "bloodmnist", "high_noise", "simple_cnn_3ch", 16, 27)  # BloodMNIST + high_noise + simple_cnn_3ch
    run_single_study("bloodmnist_high_noise_linear_conv_3ch", "bloodmnist", "high_noise", "linear_conv_3ch", 17, 27)  # BloodMNIST + high_noise + linear_conv_3ch
    run_single_study("bloodmnist_high_noise_diffusers_unet_28_3ch", "bloodmnist", "high_noise", "diffusers_unet_28_3ch", 18, 27)  # BloodMNIST + high_noise + diffusers_unet_28_3ch
    
    # ChestMNIST studies (1 channel)
    run_single_study("chestmnist_low_noise_simple_cnn_1ch", "chestmnist", "low_noise", "simple_cnn_1ch", 19, 27)  # ChestMNIST + low_noise + simple_cnn_1ch
    run_single_study("chestmnist_low_noise_linear_conv_1ch", "chestmnist", "low_noise", "linear_conv_1ch", 20, 27)  # ChestMNIST + low_noise + linear_conv_1ch
    run_single_study("chestmnist_low_noise_diffusers_unet_28_1ch", "chestmnist", "low_noise", "diffusers_unet_28_1ch", 21, 27)  # ChestMNIST + low_noise + diffusers_unet_28_1ch
    run_single_study("chestmnist_medium_noise_simple_cnn_1ch", "chestmnist", "medium_noise", "simple_cnn_1ch", 22, 27)  # ChestMNIST + medium_noise + simple_cnn_1ch
    run_single_study("chestmnist_medium_noise_linear_conv_1ch", "chestmnist", "medium_noise", "linear_conv_1ch", 23, 27)  # ChestMNIST + medium_noise + linear_conv_1ch
    run_single_study("chestmnist_medium_noise_diffusers_unet_28_1ch", "chestmnist", "medium_noise", "diffusers_unet_28_1ch", 24, 27)  # ChestMNIST + medium_noise + diffusers_unet_28_1ch
    run_single_study("chestmnist_high_noise_simple_cnn_1ch", "chestmnist", "high_noise", "simple_cnn_1ch", 25, 27)  # ChestMNIST + high_noise + simple_cnn_1ch
    run_single_study("chestmnist_high_noise_linear_conv_1ch", "chestmnist", "high_noise", "linear_conv_1ch", 26, 27)  # ChestMNIST + high_noise + linear_conv_1ch
    run_single_study("chestmnist_high_noise_diffusers_unet_28_1ch", "chestmnist", "high_noise", "diffusers_unet_28_1ch", 27, 27)  # ChestMNIST + high_noise + diffusers_unet_28_1ch
    
    print(f"\n{'='*60}")
    print(f"All 27 studies completed!")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 