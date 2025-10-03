#!/usr/bin/env python3
"""
ChestMNIST End-to-End Study
Combines pre-trained classifier and denoiser for comprehensive evaluation
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gmi
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# Add parent directories to path for importing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ChestMNIST_classification_reorganized'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ChestMNIST_poisson_denoising_reorganized'))

from parse_config import parse_config
from train_classifier import ResNet50ChestClassifier, labels_to_multihot, CHEST_CODEBOOK
from train_denoiser import XRayPhysicsSimulator, XRayConditionalDenoiser

def load_pretrained_classifier(config):
    """Load pre-trained classifier model"""
    classifier_path = config.pretrained_models['classifier_path']
    
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Pre-trained classifier not found at: {classifier_path}")
    
    print(f"Loading pre-trained classifier from: {classifier_path}")
    checkpoint = torch.load(classifier_path, map_location=config.device)
    
    # Create classifier with same architecture
    classifier = ResNet50ChestClassifier(num_labels=14)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    classifier = classifier.to(config.device)
    classifier.eval()
    
    return classifier

def load_pretrained_denoiser(config):
    """Load pre-trained denoiser model"""
    denoiser_path = config.pretrained_models['denoiser_path']
    
    if not os.path.exists(denoiser_path):
        raise FileNotFoundError(f"Pre-trained denoiser not found at: {denoiser_path}")
    
    print(f"Loading pre-trained denoiser from: {denoiser_path}")
    checkpoint = torch.load(denoiser_path, map_location=config.device)
    
    # Get physics parameters from saved model
    physics_params = checkpoint.get('physics_params', {'mu': 4.0, 'I0': 100.0})
    
    # Create denoiser with same architecture
    denoiser = XRayConditionalDenoiser(
        mu=physics_params['mu'],
        I0=physics_params['I0'],
        noise_std=0.1  # Default value, should match training
    )
    denoiser.load_state_dict(checkpoint['denoiser_state_dict'])
    denoiser = denoiser.to(config.device)
    denoiser.eval()
    
    return denoiser, physics_params

def create_physics_simulator(config, dose_level):
    """Create physics simulator for given dose level"""
    return XRayPhysicsSimulator(
        mu=config.simulation['mu'],
        I0=dose_level  # dose_level is the I0 parameter
    )

def evaluate_classification_performance(classifier, images, true_labels, config):
    """Evaluate classification performance on a batch of images"""
    with torch.no_grad():
        # Get predicted probabilities
        logits = classifier.forward_likelihood_only(images)
        predicted_probs = torch.sigmoid(logits).cpu().numpy()
        
        # Convert true labels to numpy
        if isinstance(true_labels, torch.Tensor):
            true_labels_np = true_labels.cpu().numpy()
        else:
            true_labels_np = labels_to_multihot(true_labels).cpu().numpy()
        
        # Compute metrics
        metrics = {}
        
        # Per-sample exact match accuracy
        predicted_binary = (predicted_probs > 0.5).astype(int)
        exact_matches = np.all(true_labels_np == predicted_binary, axis=1)
        metrics['exact_match_accuracy'] = np.mean(exact_matches)
        
        # Hamming accuracy (average per-label accuracy)
        hamming_accuracy = np.mean(true_labels_np == predicted_binary)
        metrics['hamming_accuracy'] = hamming_accuracy
        
        # Multi-label AUC (macro-averaged)
        try:
            label_aucs = []
            for i in range(14):
                if len(np.unique(true_labels_np[:, i])) > 1:  # Only if both classes present
                    auc = roc_auc_score(true_labels_np[:, i], predicted_probs[:, i])
                    label_aucs.append(auc)
            metrics['macro_auc'] = np.mean(label_aucs) if label_aucs else 0.0
        except:
            metrics['macro_auc'] = 0.0
        
        # Average precision (macro-averaged)
        try:
            label_maps = []
            for i in range(14):
                if len(np.unique(true_labels_np[:, i])) > 1:
                    ap = average_precision_score(true_labels_np[:, i], predicted_probs[:, i])
                    label_maps.append(ap)
            metrics['macro_map'] = np.mean(label_maps) if label_maps else 0.0
        except:
            metrics['macro_map'] = 0.0
        
        return metrics, predicted_probs

def run_simulation_study(config):
    """Run the main simulation study"""
    print("=== ChestMNIST End-to-End Simulation Study ===")
    
    # Load pre-trained models
    classifier = load_pretrained_classifier(config)
    denoiser, denoiser_physics_params = load_pretrained_denoiser(config)
    
    print(f"Loaded classifier with {sum(p.numel() for p in classifier.parameters()):,} parameters")
    print(f"Loaded denoiser with {sum(p.numel() for p in denoiser.parameters()):,} parameters")
    print(f"Denoiser physics params: μ={denoiser_physics_params['mu']}, I₀={denoiser_physics_params['I0']}")
    
    # Create test dataset
    dataset_test = gmi.datasets.MedMNIST(
        config.data['dataset_name'],
        split='test',
        root=config.data['dataset_root'],
        size=config.data['image_size'],
        download=True
    )
    
    # Limit to specified number of test samples
    num_samples = min(config.simulation['num_test_samples'], len(dataset_test))
    indices = np.random.choice(len(dataset_test), num_samples, replace=False)
    
    print(f"Evaluating on {num_samples} test samples")
    
    # Storage for results
    results = []
    sample_images = {}  # For visualization
    
    # Test different dose levels
    dose_levels = config.simulation['dose_levels']
    
    for dose_idx, dose_level in enumerate(dose_levels):
        print(f"\n--- Dose Level {dose_idx+1}/{len(dose_levels)}: I₀ = {dose_level} ---")
        
        # Create physics simulator for this dose level
        physics_sim = create_physics_simulator(config, dose_level)
        
        # Storage for this dose level
        dose_results = {
            'dose_level': dose_level,
            'clean_metrics': {},
            'noisy_metrics': {},
            'log_corrected_metrics': {},
            'denoised_metrics': {}
        }
        
        # Collect data for all samples at this dose level
        clean_images = []
        noisy_images = []
        log_corrected_images = []
        denoised_images = []
        all_labels = []
        
        print("Processing samples...")
        for i in tqdm(indices):
            clean_img, label = dataset_test[i]
            clean_img = clean_img.unsqueeze(0).to(config.device)  # Add batch dim
            
            # Simulate noisy measurement
            noisy_img = physics_sim.forward_model(clean_img)
            
            # Apply log correction (what goes into denoiser)
            log_corrected_img = physics_sim.log_correction(noisy_img)
            
            # Apply denoiser
            with torch.no_grad():
                denoised_img = denoiser.get_mean_estimate(noisy_img)
            
            # Store for batch evaluation
            clean_images.append(clean_img)
            noisy_images.append(noisy_img)
            log_corrected_images.append(log_corrected_img)
            denoised_images.append(denoised_img)
            all_labels.append(label)
        
        # Convert to tensors
        clean_batch = torch.cat(clean_images, dim=0)
        noisy_batch = torch.cat(noisy_images, dim=0)
        log_corrected_batch = torch.cat(log_corrected_images, dim=0)  
        denoised_batch = torch.cat(denoised_images, dim=0)
        
        print("Evaluating classification performance...")
        
        # Evaluate classifier on clean images (ground truth performance)
        clean_metrics, clean_probs = evaluate_classification_performance(
            classifier, clean_batch, all_labels, config
        )
        dose_results['clean_metrics'] = clean_metrics
        
        # Evaluate classifier on noisy images (worst case)
        noisy_metrics, noisy_probs = evaluate_classification_performance(
            classifier, noisy_batch, all_labels, config
        )
        dose_results['noisy_metrics'] = noisy_metrics
        
        # Evaluate classifier on log-corrected images (denoiser input)
        log_corrected_metrics, log_corrected_probs = evaluate_classification_performance(
            classifier, log_corrected_batch, all_labels, config
        )
        dose_results['log_corrected_metrics'] = log_corrected_metrics
        
        # Evaluate classifier on denoised images (our method)
        denoised_metrics, denoised_probs = evaluate_classification_performance(
            classifier, denoised_batch, all_labels, config
        )
        dose_results['denoised_metrics'] = denoised_metrics
        
        # Store sample images for visualization
        if dose_idx == 0:  # Only store for first dose level
            sample_images = {
                'clean': clean_batch[:config.output['num_visualization_samples']].cpu(),
                'noisy': noisy_batch[:config.output['num_visualization_samples']].cpu(),
                'log_corrected': log_corrected_batch[:config.output['num_visualization_samples']].cpu(),
                'denoised': denoised_batch[:config.output['num_visualization_samples']].cpu(),
                'labels': all_labels[:config.output['num_visualization_samples']]
            }
        
        results.append(dose_results)
        
        # Print summary for this dose level
        print(f"Clean Image Classification:")
        print(f"  Exact Match: {clean_metrics['exact_match_accuracy']:.4f}")
        print(f"  Hamming: {clean_metrics['hamming_accuracy']:.4f}")
        print(f"  Macro AUC: {clean_metrics['macro_auc']:.4f}")
        
        print(f"Noisy Image Classification:")
        print(f"  Exact Match: {noisy_metrics['exact_match_accuracy']:.4f}")
        print(f"  Hamming: {noisy_metrics['hamming_accuracy']:.4f}")
        print(f"  Macro AUC: {noisy_metrics['macro_auc']:.4f}")
        
        print(f"Log-Corrected Image Classification:")
        print(f"  Exact Match: {log_corrected_metrics['exact_match_accuracy']:.4f}")
        print(f"  Hamming: {log_corrected_metrics['hamming_accuracy']:.4f}")
        print(f"  Macro AUC: {log_corrected_metrics['macro_auc']:.4f}")
        
        print(f"Denoised Image Classification:")
        print(f"  Exact Match: {denoised_metrics['exact_match_accuracy']:.4f}")
        print(f"  Hamming: {denoised_metrics['hamming_accuracy']:.4f}")
        print(f"  Macro AUC: {denoised_metrics['macro_auc']:.4f}")
    
    return results, sample_images

def save_results_to_csv(results, config):
    """Save results to CSV file"""
    csv_path = os.path.join(config.output['output_dir'], config.output['results_csv'])
    
    # Flatten results for CSV
    rows = []
    for result in results:
        dose_level = result['dose_level']
        
        for method, metrics in result.items():
            if method == 'dose_level':
                continue
            
            row = {
                'dose_level': dose_level,
                'method': method.replace('_metrics', ''),
                'exact_match_accuracy': metrics['exact_match_accuracy'],
                'hamming_accuracy': metrics['hamming_accuracy'],
                'macro_auc': metrics['macro_auc'],
                'macro_map': metrics['macro_map']
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    return df

def create_visualizations(results, sample_images, config):
    """Create visualization plots"""
    viz_dir = os.path.join(config.output['output_dir'], config.output['visualization_dir'])
    
    # Plot 1: Performance vs Dose Level
    dose_levels = [r['dose_level'] for r in results]
    
    methods = ['clean', 'noisy', 'log_corrected', 'denoised']
    method_names = ['Clean (Ground Truth)', 'Noisy', 'Log-Corrected', 'Denoised (Ours)']
    colors = ['green', 'red', 'orange', 'blue']
    
    metrics = ['exact_match_accuracy', 'hamming_accuracy', 'macro_auc']
    metric_names = ['Exact Match Accuracy', 'Hamming Accuracy', 'Macro AUC']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for metric_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[metric_idx]
        
        for method_idx, (method, method_name, color) in enumerate(zip(methods, method_names, colors)):
            values = [r[f'{method}_metrics'][metric] for r in results]
            ax.plot(dose_levels, values, 'o-', label=method_name, color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel('Dose Level (I₀)')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs Dose Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'performance_vs_dose.png'), dpi=300, bbox_inches='tight')
    print(f"Performance plot saved to: {os.path.join(viz_dir, 'performance_vs_dose.png')}")
    
    # Plot 2: Sample Images
    if config.output['save_sample_images'] and sample_images:
        num_samples = min(5, len(sample_images['labels']))
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        image_types = ['clean', 'noisy', 'log_corrected', 'denoised']
        image_titles = ['Clean', 'Noisy', 'Log-Corrected', 'Denoised']
        
        for sample_idx in range(num_samples):
            label = sample_images['labels'][sample_idx]
            
            # Find active labels
            if isinstance(label, torch.Tensor):
                label_np = label.numpy()
            else:
                label_np = np.array(label)
            
            active_labels = [CHEST_CODEBOOK[i] for i, val in enumerate(label_np) if val == 1]
            label_text = ', '.join(active_labels[:2]) if active_labels else 'No Findings'
            if len(active_labels) > 2:
                label_text += f' (+{len(active_labels)-2} more)'
            
            for img_idx, (img_type, img_title) in enumerate(zip(image_types, image_titles)):
                ax = axes[sample_idx, img_idx]
                
                img = sample_images[img_type][sample_idx, 0].numpy()  # [H, W]
                ax.imshow(img, cmap='gray')
                ax.set_title(f'{img_title}')
                ax.axis('off')
                
                # Add label text on first image
                if img_idx == 0:
                    ax.text(0.02, 0.98, label_text, transform=ax.transAxes, 
                           verticalalignment='top', fontsize=8, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'sample_images.png'), dpi=300, bbox_inches='tight')
        print(f"Sample images saved to: {os.path.join(viz_dir, 'sample_images.png')}")

def main():
    """Main function"""
    # Parse configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    config = parse_config(config_path)
    
    print(f"Device: {config.device}")
    
    # Run simulation study
    results, sample_images = run_simulation_study(config)
    
    # Save results to CSV
    df = save_results_to_csv(results, config)
    
    # Create visualizations
    create_visualizations(results, sample_images, config)
    
    # Print summary
    print("\n=== Study Complete ===")
    print(f"Results saved to: {config.output['output_dir']}")
    
    # Print improvement summary
    print("\n=== Denoising Improvement Summary ===")
    for result in results:
        dose = result['dose_level']
        
        # Compare denoised vs log-corrected (input to denoiser)
        log_corr_auc = result['log_corrected_metrics']['macro_auc']
        denoised_auc = result['denoised_metrics']['macro_auc']
        improvement = denoised_auc - log_corr_auc
        
        print(f"Dose I₀={dose:6.1f}: Log-Corrected AUC={log_corr_auc:.4f}, Denoised AUC={denoised_auc:.4f}, Improvement={improvement:+.4f}")

if __name__ == "__main__":
    main()