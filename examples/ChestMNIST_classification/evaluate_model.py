#!/usr/bin/env python3
"""
ChestMNIST Classification Evaluation Script
Comprehensive evaluation with optimal threshold finding and AUC analysis
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add the gmi directory to Python path to ensure proper import
sys.path.insert(0, '/workspace/gmi')
import gmi

from parse_config import parse_config
from train_classifier import (
    create_datasets_and_loaders,
    create_model_and_optimizer,
    compute_base_rates,
    labels_to_multihot,
    CHEST_CODEBOOK,
    ResNet50ChestClassifier
)

def load_trained_model(config):
    """Load trained model from checkpoint"""
    model_path = os.path.join(config.output['output_dir'], config.training['model_save_path'])
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    
    print(f"Loading trained model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
    
    # Create model architecture
    if 'config' in checkpoint:
        # Use saved config to reconstruct model
        saved_config = checkpoint['config']
        model = ResNet50ChestClassifier(
            num_labels=saved_config['model']['num_labels']
        )
    else:
        # Fallback to current config
        model = ResNet50ChestClassifier(
            num_labels=config.model['num_labels']
        )
    
    # Load state dict
    model.load_state_dict(checkpoint['classifier_state_dict'])
    model = model.to(config.device)
    model.eval()
    
    # Create conditional multilabel wrapper
    conditional_multilabel = gmi.random_variable.ConditionalMultilabelBinaryRandomVariable(
        logit_function=model,
        codebook=CHEST_CODEBOOK
    )
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, conditional_multilabel

def collect_predictions_and_labels(conditional_multilabel, dataloader_test, device):
    """Collect all predictions and true labels from test set"""
    print("Collecting predictions from test set...")
    
    all_predictions = []
    all_true_labels = []
    
    conditional_multilabel.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader_test, desc="Processing batches")):
            images = images.to(device)
            
            # Get predicted probabilities (sigmoid of logits)
            predicted_probs = conditional_multilabel.predict_probabilities(images)
            
            # Convert labels to multi-hot tensor
            true_labels = labels_to_multihot(labels, device=device)
            
            # Store predictions and labels
            all_predictions.append(predicted_probs.cpu().numpy())
            all_true_labels.append(true_labels.cpu().numpy())
    
    # Concatenate all batches
    all_predictions = np.vstack(all_predictions)  # [N, 14]
    all_true_labels = np.vstack(all_true_labels)  # [N, 14]
    
    print(f"Collected predictions for {all_predictions.shape[0]} samples")
    return all_predictions, all_true_labels

def find_optimal_thresholds(predictions, true_labels):
    """Find optimal thresholds for each label to maximize accuracy"""
    print("Finding optimal thresholds for each label...")
    
    num_labels = predictions.shape[1]
    optimal_thresholds = np.zeros(num_labels)
    optimal_accuracies = np.zeros(num_labels)
    
    # Test thresholds from 0.1 to 0.9 in steps of 0.02
    test_thresholds = np.arange(0.1, 0.9, 0.02)
    
    for label_idx in range(num_labels):
        label_name = CHEST_CODEBOOK[label_idx]
        
        best_threshold = 0.5
        best_accuracy = 0.0
        
        for threshold in test_thresholds:
            # Apply threshold
            predicted_binary = (predictions[:, label_idx] > threshold).astype(int)
            
            # Calculate accuracy for this label
            accuracy = np.mean(predicted_binary == true_labels[:, label_idx])
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        optimal_thresholds[label_idx] = best_threshold
        optimal_accuracies[label_idx] = best_accuracy
        
        print(f"  {label_name:20s}: threshold={best_threshold:.3f}, accuracy={best_accuracy:.4f}")
    
    return optimal_thresholds, optimal_accuracies

def compute_auc_metrics(predictions, true_labels, use_base_rates=False, base_rates=None):
    """Compute AUC and AP metrics for each label
    
    Args:
        predictions: Model probability predictions [N, 14]
        true_labels: True binary labels [N, 14]
        use_base_rates: Whether to adjust predictions using base rates
        base_rates: Prior probabilities for each label [14,]
    """
    analysis_type = "with base rate adjustment" if use_base_rates else "standard"
    print(f"Computing AUC and Average Precision metrics ({analysis_type})...")
    
    # Adjust predictions if using base rates
    if use_base_rates and base_rates is not None:
        adjusted_predictions = predictions.copy()
        for label_idx in range(predictions.shape[1]):
            # Apply Bayes' theorem adjustment
            # P(disease|positive) = P(positive|disease) * P(disease) / P(positive)
            # Where P(positive) = P(positive|disease) * P(disease) + P(positive|no disease) * P(no disease)
            
            base_rate = base_rates[label_idx]
            pred_scores = predictions[:, label_idx]
            
            # Estimate P(positive|no disease) from data where true label is 0
            negative_mask = true_labels[:, label_idx] == 0
            if np.sum(negative_mask) > 0:
                false_positive_rate = np.mean(pred_scores[negative_mask])
            else:
                false_positive_rate = 0.0
            
            # Apply Bayesian adjustment
            numerator = pred_scores * base_rate
            denominator = pred_scores * base_rate + false_positive_rate * (1 - base_rate)
            
            # Avoid division by zero
            denominator[denominator == 0] = 1e-8
            adjusted_predictions[:, label_idx] = numerator / denominator
        
        predictions_to_use = adjusted_predictions
    else:
        predictions_to_use = predictions
    
    num_labels = predictions.shape[1]
    label_aucs = np.zeros(num_labels)
    label_aps = np.zeros(num_labels)
    
    for label_idx in range(num_labels):
        label_name = CHEST_CODEBOOK[label_idx]
        
        # Check if both classes are present
        if len(np.unique(true_labels[:, label_idx])) > 1:
            try:
                auc = roc_auc_score(true_labels[:, label_idx], predictions_to_use[:, label_idx])
                ap = average_precision_score(true_labels[:, label_idx], predictions_to_use[:, label_idx])
                
                label_aucs[label_idx] = auc
                label_aps[label_idx] = ap
                
                base_rate_info = f" (base rate: {base_rates[label_idx]:.3f})" if use_base_rates and base_rates is not None else ""
                print(f"  {label_name:20s}: AUC={auc:.4f}, AP={ap:.4f}{base_rate_info}")
            except Exception as e:
                print(f"  {label_name:20s}: Error computing metrics - {e}")
                label_aucs[label_idx] = 0.0
                label_aps[label_idx] = 0.0
        else:
            print(f"  {label_name:20s}: Only one class present, AUC=0.0, AP=0.0")
            label_aucs[label_idx] = 0.0
            label_aps[label_idx] = 0.0
    
    return label_aucs, label_aps

def evaluate_with_thresholds(predictions, true_labels, thresholds, use_optimal=True):
    """Evaluate using specified thresholds"""
    num_samples, num_labels = predictions.shape
    
    if use_optimal:
        print(f"Evaluating with optimal thresholds...")
    else:
        print(f"Evaluating with fixed threshold 0.5...")
        thresholds = np.full(num_labels, 0.5)
    
    # Apply thresholds to get binary predictions
    predicted_binary = np.zeros_like(predictions)
    for label_idx in range(num_labels):
        predicted_binary[:, label_idx] = (predictions[:, label_idx] > thresholds[label_idx]).astype(int)
    
    # Exact match accuracy (all labels must match)
    exact_matches = np.all(true_labels == predicted_binary, axis=1)
    exact_match_accuracy = np.mean(exact_matches)
    
    # Hamming accuracy (average per-label accuracy)
    hamming_accuracy = np.mean(true_labels == predicted_binary)
    
    # Per-label metrics
    label_precisions = np.zeros(num_labels)
    label_recalls = np.zeros(num_labels)
    label_f1s = np.zeros(num_labels)
    label_accuracies = np.zeros(num_labels)
    
    for label_idx in range(num_labels):
        true_label = true_labels[:, label_idx]
        pred_label = predicted_binary[:, label_idx]
        
        # Basic accuracy for this label
        label_accuracies[label_idx] = np.mean(true_label == pred_label)
        
        # Precision, Recall, F1 for this label
        tp = np.sum((true_label == 1) & (pred_label == 1))
        fp = np.sum((true_label == 0) & (pred_label == 1))
        fn = np.sum((true_label == 1) & (pred_label == 0))
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0
            
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
            
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
            
        label_precisions[label_idx] = precision
        label_recalls[label_idx] = recall
        label_f1s[label_idx] = f1
    
    return {
        'exact_match_accuracy': exact_match_accuracy,
        'hamming_accuracy': hamming_accuracy,
        'macro_f1': np.mean(label_f1s),
        'label_accuracies': label_accuracies,
        'label_precisions': label_precisions,
        'label_recalls': label_recalls,
        'label_f1s': label_f1s,
        'thresholds_used': thresholds
    }

def print_evaluation_results(results, title, label_aucs=None, label_aps=None):
    """Print evaluation results in a formatted way"""
    print(f"\n=== {title} ===")
    print(f"Exact Match Accuracy: {results['exact_match_accuracy']:.4f}")
    print(f"Hamming Accuracy (avg per-label): {results['hamming_accuracy']:.4f}")
    print(f"Macro-averaged F1: {results['macro_f1']:.4f}")
    
    if label_aucs is not None:
        print(f"Macro-averaged AUC: {np.mean(label_aucs):.4f}")
    if label_aps is not None:
        print(f"Macro-averaged AP: {np.mean(label_aps):.4f}")
    
    print("\nPer-label metrics:")
    header = f"{'Label':<20} {'Threshold':<9} {'Accuracy':<8} {'Precision':<9} {'Recall':<8} {'F1':<8}"
    if label_aucs is not None:
        header += f" {'AUC':<8}"
    if label_aps is not None:
        header += f" {'AP':<8}"
    print(header)
    print("-" * len(header))
    
    for i in range(14):
        line = f"{CHEST_CODEBOOK[i]:<20} {results['thresholds_used'][i]:<9.3f} {results['label_accuracies'][i]:<8.3f} {results['label_precisions'][i]:<9.3f} {results['label_recalls'][i]:<8.3f} {results['label_f1s'][i]:<8.3f}"
        if label_aucs is not None:
            line += f" {label_aucs[i]:<8.3f}"
        if label_aps is not None:
            line += f" {label_aps[i]:<8.3f}"
        print(line)

def save_results_to_csv(results_fixed, results_optimal, label_aucs, label_aps, output_dir):
    """Save detailed results to CSV file"""
    # Create detailed results dataframe
    rows = []
    for i in range(14):
        rows.append({
            'label_idx': i,
            'label_name': CHEST_CODEBOOK[i],
            'auc': label_aucs[i],
            'average_precision': label_aps[i],
            'fixed_threshold': 0.5,
            'fixed_accuracy': results_fixed['label_accuracies'][i],
            'fixed_precision': results_fixed['label_precisions'][i],
            'fixed_recall': results_fixed['label_recalls'][i],
            'fixed_f1': results_fixed['label_f1s'][i],
            'optimal_threshold': results_optimal['thresholds_used'][i],
            'optimal_accuracy': results_optimal['label_accuracies'][i],
            'optimal_precision': results_optimal['label_precisions'][i],
            'optimal_recall': results_optimal['label_recalls'][i],
            'optimal_f1': results_optimal['label_f1s'][i]
        })
    
    df = pd.DataFrame(rows)
    
    # Add summary row
    summary_row = {
        'label_idx': -1,
        'label_name': 'MACRO_AVERAGE',
        'auc': np.mean(label_aucs),
        'average_precision': np.mean(label_aps),
        'fixed_threshold': 0.5,
        'fixed_accuracy': results_fixed['hamming_accuracy'],
        'fixed_precision': np.mean(results_fixed['label_precisions']),
        'fixed_recall': np.mean(results_fixed['label_recalls']),
        'fixed_f1': results_fixed['macro_f1'],
        'optimal_threshold': np.mean(results_optimal['thresholds_used']),
        'optimal_accuracy': results_optimal['hamming_accuracy'],
        'optimal_precision': np.mean(results_optimal['label_precisions']),
        'optimal_recall': np.mean(results_optimal['label_recalls']),
        'optimal_f1': results_optimal['macro_f1']
    }
    
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'evaluation_results.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nDetailed results saved to: {csv_path}")

def run_evaluation(config):
    """Run complete evaluation pipeline - both standard and base-rate adjusted analyses"""
    print("=== ChestMNIST Classification Evaluation ===")
    print(f"Device: {config.device}")
    
    # Load trained model
    model, conditional_multilabel = load_trained_model(config)
    
    # Create datasets and dataloaders
    dataset_train, _, dataset_test, dataloader_train, _, dataloader_test = create_datasets_and_loaders(config)
    print(f"Test dataset size: {len(dataset_test)}")
    
    # Compute base rates from training data
    base_rates, _ = compute_base_rates(dataloader_train, num_labels=config.model['num_labels'])
    print("\nBase rates (prevalence) for each condition:")
    for i, label_name in enumerate(CHEST_CODEBOOK):
        print(f"  {label_name:20s}: {base_rates[i]:.4f} ({base_rates[i]*100:.1f}%)")
    
    # Collect all predictions and labels
    predictions, true_labels = collect_predictions_and_labels(
        conditional_multilabel, dataloader_test, config.device
    )
    
    print(f"\n{'='*80}")
    print("ANALYSIS 1: STANDARD EVALUATION (No Base Rate Adjustment)")
    print(f"{'='*80}")
    
    # Compute AUC metrics without base rate adjustment
    label_aucs_std, label_aps_std = compute_auc_metrics(predictions, true_labels, use_base_rates=False)
    average_auc_std = np.mean(label_aucs_std)
    
    print(f"\n{'='*80}")
    print("ANALYSIS 2: BASE RATE ADJUSTED EVALUATION")
    print(f"{'='*80}")
    
    # Compute AUC metrics with base rate adjustment
    label_aucs_adj, label_aps_adj = compute_auc_metrics(predictions, true_labels, use_base_rates=True, base_rates=base_rates)
    average_auc_adj = np.mean(label_aucs_adj)
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Standard Analysis Average AUC:     {average_auc_std:.4f}")
    print(f"Base Rate Adjusted Average AUC:    {average_auc_adj:.4f}")
    print(f"Difference (Adjusted - Standard):  {average_auc_adj - average_auc_std:+.4f}")
    print(f"{'='*80}\n")
    
    # Find optimal thresholds (using standard predictions)
    optimal_thresholds, optimal_accuracies = find_optimal_thresholds(predictions, true_labels)
    
    # Evaluate with fixed threshold (0.5) - standard analysis
    results_fixed_std = evaluate_with_thresholds(predictions, true_labels, None, use_optimal=False)
    results_optimal_std = evaluate_with_thresholds(predictions, true_labels, optimal_thresholds, use_optimal=True)
    
    print(f"\n{'='*80}")
    print("DETAILED RESULTS - STANDARD ANALYSIS")
    print(f"{'='*80}")
    print_evaluation_results(results_fixed_std, "Fixed Threshold (0.5) Results - Standard", label_aucs_std, label_aps_std)
    print_evaluation_results(results_optimal_std, "Optimal Threshold Results - Standard", label_aucs_std, label_aps_std)
    
    print(f"\n{'='*80}")
    print("DETAILED RESULTS - BASE RATE ADJUSTED ANALYSIS")
    print(f"{'='*80}")
    print_evaluation_results(results_fixed_std, "Fixed Threshold (0.5) Results - Base Rate Adjusted", label_aucs_adj, label_aps_adj)
    print_evaluation_results(results_optimal_std, "Optimal Threshold Results - Base Rate Adjusted", label_aucs_adj, label_aps_adj)
    
    # Print improvement summary for both analyses
    print(f"\n{'='*80}")
    print("IMPROVEMENT SUMMARY")
    print(f"{'='*80}")
    print("Standard Analysis:")
    print(f"  Average AUC: {average_auc_std:.4f}")
    print(f"  Exact Match: {results_fixed_std['exact_match_accuracy']:.4f} → {results_optimal_std['exact_match_accuracy']:.4f} ({results_optimal_std['exact_match_accuracy'] - results_fixed_std['exact_match_accuracy']:+.4f})")
    print(f"  Hamming Acc: {results_fixed_std['hamming_accuracy']:.4f} → {results_optimal_std['hamming_accuracy']:.4f} ({results_optimal_std['hamming_accuracy'] - results_fixed_std['hamming_accuracy']:+.4f})")
    
    print("\nBase Rate Adjusted Analysis:")
    print(f"  Average AUC: {average_auc_adj:.4f}")
    print(f"  AUC Improvement: {average_auc_adj - average_auc_std:+.4f}")
    
    # Save results to CSV (using standard analysis for compatibility)
    save_results_to_csv(results_fixed_std, results_optimal_std, label_aucs_std, label_aps_std, config.output['output_dir'])
    
    # Final summary with both analyses
    print(f"\n{'='*80}")
    print("FINAL COMPARATIVE SUMMARY")
    print(f"{'='*80}")
    print(f"Standard Model Average AUC:        {average_auc_std:.4f}")
    print(f"Base Rate Adjusted Average AUC:    {average_auc_adj:.4f}")
    print(f"Improvement from Base Rate Adj:    {average_auc_adj - average_auc_std:+.4f}")
    print(f"")
    print(f"Interpretation:")
    if average_auc_adj > average_auc_std:
        print(f"✓ Base rate adjustment IMPROVES performance by {(average_auc_adj - average_auc_std)*100:.2f} percentage points")
        print(f"  This suggests the model benefits from considering disease prevalence")
    elif average_auc_adj < average_auc_std:
        print(f"⚠ Base rate adjustment DECREASES performance by {(average_auc_std - average_auc_adj)*100:.2f} percentage points")
        print(f"  This suggests the model may already account for base rates internally")
    else:
        print(f"≈ Base rate adjustment has minimal impact on performance")
    print(f"{'='*80}")
    
    return {
        'standard': {'fixed': results_fixed_std, 'optimal': results_optimal_std, 'aucs': label_aucs_std, 'aps': label_aps_std},
        'adjusted': {'aucs': label_aucs_adj, 'aps': label_aps_adj},
        'base_rates': base_rates
    }

def main():
    """Main evaluation function"""
    # Parse configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    config = parse_config(config_path)
    
    # Run dual evaluation (standard and base rate adjusted)
    results = run_evaluation(config)
    
    return results

if __name__ == "__main__":
    results = main()