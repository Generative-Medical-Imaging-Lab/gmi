#!/usr/bin/env python3
"""
Test script for the categorical random variable implementation.
"""

import torch
import sys
import os

# Add the GMI library to the path
sys.path.insert(0, '/workspace/gmi')
import gmi

def test_categorical_random_variable():
    """Test the basic categorical random variable."""
    print("Testing CategoricalRandomVariable...")
    
    # Create a simple categorical with 3 classes
    logits = torch.tensor([1.0, 2.0, 0.5])
    codebook = {0: "Class A", 1: "Class B", 2: "Class C"}
    
    categorical = gmi.random_variable.CategoricalRandomVariable(logits, codebook)
    
    # Test sampling
    samples = categorical.sample(10)
    print(f"Sample shape: {samples.shape}")
    print(f"Sample sum per row (should be 1): {samples.sum(dim=1)}")
    
    # Test log_prob
    test_onehot = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    log_probs = categorical.log_prob(test_onehot)
    print(f"Log probabilities: {log_probs}")
    
    # Test codebook
    for i in range(3):
        print(f"Class {i}: {categorical.get_class_name(i)}")
    
    print("CategoricalRandomVariable test passed!\n")

def test_conditional_categorical_random_variable():
    """Test the conditional categorical random variable."""
    print("Testing ConditionalCategoricalRandomVariable...")
    
    # Create a simple logit function (linear layer)
    logit_function = torch.nn.Linear(10, 3)  # 10 input features, 3 classes
    codebook = {0: "Dog", 1: "Cat", 2: "Bird"}
    
    conditional_categorical = gmi.random_variable.ConditionalCategoricalRandomVariable(
        logit_function, codebook
    )
    
    # Test forward pass
    x = torch.randn(5, 10)  # 5 samples, 10 features each
    logits = conditional_categorical(x)
    print(f"Logits shape: {logits.shape}")
    
    # Test predictions
    predictions = conditional_categorical.predict_classes(x)
    probs = conditional_categorical.predict_proba(x)
    print(f"Predictions: {predictions}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Probabilities sum per row: {probs.sum(dim=1)}")
    
    # Test log_prob
    y_onehot = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    log_probs = conditional_categorical.log_prob(x, y_onehot)
    print(f"Log probabilities: {log_probs}")
    
    # Test loss closures
    batch = (x, y_onehot)
    train_loss = conditional_categorical.train_loss_closure(batch)
    eval_loss = conditional_categorical.eval_loss_closure(batch)
    print(f"Train loss: {train_loss.item():.4f}")
    print(f"Eval loss: {eval_loss.item():.4f}")
    
    print("ConditionalCategoricalRandomVariable test passed!\n")

if __name__ == "__main__":
    test_categorical_random_variable()
    test_conditional_categorical_random_variable()
    print("All tests passed! 🎉")