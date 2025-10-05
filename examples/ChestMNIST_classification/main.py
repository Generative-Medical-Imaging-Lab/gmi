#!/usr/bin/env python3
"""
ChestMNIST Multi-Label Classification
Main script that orchestrates the training process
"""

import os
import torch
import numpy as np
from parse_config import parse_config
from train_classifier import (
    create_datasets_and_loaders, 
    compute_base_rates,
    create_model_and_optimizer,
    labels_to_multihot,
    CHEST_CODEBOOK
)
from animate_training_process import animate_training_process



def save_model(model, conditional_multilabel, optimizer, config, metrics=None):
    """Save trained model and metadata"""
    if config.training['save_model']:
        # Construct full path using output directory
        model_path = os.path.join(config.output['output_dir'], config.training['model_save_path'])
        
        save_dict = {
            'classifier_state_dict': model.state_dict(),
            'conditional_multilabel_state_dict': conditional_multilabel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.config,
        }
        
        if metrics is not None:
            save_dict['metrics'] = metrics
        
        torch.save(save_dict, model_path)
        print(f"Model saved to: {model_path}")

def main():
    """Main training function"""
    # Parse configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    config = parse_config(config_path)
    
    print("=== ChestMNIST Multi-Label Classification ===")
    print(f"Device: {config.device}")
    
    model, conditional_multilabel, optimizer = None, None, None
    
    if config.execution['run_training']:
        print("\n--- Running Training ---")
        
        # Create datasets and data loaders
        (dataset_train, dataset_val, dataset_test, 
         dataloader_train, dataloader_val, dataloader_test) = create_datasets_and_loaders(config)
        
        print(f"Dataset sizes:")
        print(f"  Train: {len(dataset_train)}")
        print(f"  Val: {len(dataset_val)}")  
        print(f"  Test: {len(dataset_test)}")
        
        # Compute base rates for prior
        base_rates, prior_logits = compute_base_rates(dataloader_train)
        
        # Create model and optimizer
        model, conditional_multilabel, optimizer = create_model_and_optimizer(
            config, prior_logits
        )
        
        print(f"Model parameters: {sum(p.numel() for p in conditional_multilabel.parameters()):,}")
        
        # Run animated training process
        train_losses, val_losses = animate_training_process(
            config, model, conditional_multilabel, optimizer,
            dataloader_train, dataloader_val, dataset_test
        )
        
        # Save model
        save_model(model, conditional_multilabel, optimizer, config)
        
        print("\n=== Training Complete ===")
    else:
        print("Training skipped (run_training=False)")
    
    if config.execution['run_evaluation']:
        print("\n--- Running Evaluation ---")
        from evaluate_model import run_evaluation
        results_fixed, results_optimal, label_aucs, label_aps = run_evaluation(config)
        print("\n=== Evaluation Complete ===")
    else:
        print("Evaluation skipped (run_evaluation=False)")
    
    return model, conditional_multilabel

if __name__ == "__main__":
    model, conditional_multilabel = main()