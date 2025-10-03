# ChestMNIST End-to-End Study

This study combines pre-trained ChestMNIST classifier and denoiser models to evaluate the complete pipeline: noisy X-ray → denoising → classification.

## Prerequisites

Before running this study, you must first train both component models:

1. **Train the classifier**:
```bash
cd /workspace/gmi/examples/ChestMNIST_classification_reorganized
python main.py
```

2. **Train the denoiser**:
```bash
cd /workspace/gmi/examples/ChestMNIST_poisson_denoising_reorganized  
python main.py
```

## Study Design

The end-to-end study evaluates classification performance under different conditions:

1. **Clean Images**: Ground truth performance on clean chest X-rays
2. **Noisy Images**: Worst case - classifier applied directly to Poisson-corrupted images
3. **Log-Corrected Images**: Classifier applied to log-corrected noisy images (denoiser input)
4. **Denoised Images**: Our method - classifier applied to denoised images

## Usage

1. **Configure study parameters**: Edit `config.yml`

2. **Run the study**:
```bash
cd /workspace/gmi/examples/ChestMNIST_end_to_end_study
python main.py
```

## Configuration Options

Key parameters in `config.yml`:

- `simulation.dose_levels`: List of I₀ values to test (photon dose levels)
- `simulation.num_test_samples`: Number of test samples to evaluate
- `pretrained_models.classifier_path`: Path to trained classifier
- `pretrained_models.denoiser_path`: Path to trained denoiser

## Output

The study generates:

1. **CSV Results**: `outputs/end_to_end_results.csv` with detailed metrics
2. **Performance Plots**: `outputs/visualizations/performance_vs_dose.png`
3. **Sample Images**: `outputs/visualizations/sample_images.png`

## Evaluation Metrics

- **Exact Match Accuracy**: Fraction of samples where all 14 labels are predicted correctly
- **Hamming Accuracy**: Average per-label classification accuracy
- **Macro AUC**: Area under ROC curve, averaged across labels
- **Macro mAP**: Mean average precision, averaged across labels

## Expected Results

- **High Dose** (high I₀): Denoising provides minimal improvement (clean signal)
- **Low Dose** (low I₀): Denoising should significantly improve classification performance
- **Comparison**: Denoised images should outperform both noisy and log-corrected images

## Study Questions

1. How does classification performance degrade with decreasing dose?
2. At what dose levels does denoising provide significant benefit?
3. How much improvement does denoising provide over log-correction alone?
4. Which pathology labels benefit most from denoising?

This study provides a comprehensive evaluation of the entire medical imaging pipeline from physics simulation to final diagnosis.