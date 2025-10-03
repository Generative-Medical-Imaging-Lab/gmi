# ChestMNIST Reorganized Projects Overview

This directory contains three reorganized ChestMNIST projects that work together to provide a complete end-to-end medical imaging analysis pipeline.

## Project Structure

```
gmi/examples/
├── ChestMNIST_classification_reorganized/     # Multi-label chest X-ray classifier
├── ChestMNIST_poisson_denoising_reorganized/  # Physics-based X-ray denoiser  
└── ChestMNIST_end_to_end_study/              # Combined evaluation study
```

## Workflow

### 1. Train Classifier (`ChestMNIST_classification_reorganized/`)

Trains a ResNet50-based multi-label classifier for 14 chest pathologies:

```bash
cd ChestMNIST_classification_reorganized/
python main.py
```

**Output**: `outputs/chest_classifier.pth`

### 2. Train Denoiser (`ChestMNIST_poisson_denoising_reorganized/`)

Trains a U-Net based denoiser that handles Poisson noise from X-ray physics:

```bash 
cd ChestMNIST_poisson_denoising_reorganized/
python main.py
```

**Output**: `outputs/chest_denoiser.pth`

### 3. Run End-to-End Study (`ChestMNIST_end_to_end_study/`)

Evaluates the complete pipeline across different dose levels:

```bash
cd ChestMNIST_end_to_end_study/
python main.py
```

**Output**: Performance analysis and visualizations

## Key Features

### Modular Design
- Each project has its own `config.yml` for hyperparameters
- Separate `train_*.py` modules for model definitions
- `animate_training_process.py` for training visualization
- `main.py` orchestrates the complete workflow

### Configuration-Driven
- All hyperparameters controlled via YAML configs
- Easy to modify training parameters, model architecture, etc.
- Consistent interface across all projects

### Physics-Based Simulation
- Realistic X-ray physics: `y ~ Poisson(I₀ * exp(-μx))`
- Log correction preprocessing: `y_corr = -log(y/I₀)`
- Multiple dose levels for comprehensive evaluation

### Comprehensive Evaluation
- Multi-label classification metrics (exact match, Hamming accuracy, AUC)
- Denoising quality metrics (MSE, PSNR)
- End-to-end pipeline evaluation across dose levels
- Visualization of results and sample images

## Scientific Questions Addressed

1. **How well can we classify chest pathologies from X-ray images?**
2. **Can we denoise low-dose X-ray images effectively?**
3. **Does denoising improve downstream classification performance?**
4. **At what dose levels does denoising provide the most benefit?**
5. **How does the complete pipeline perform compared to individual components?**

## Applications

This reorganized structure enables:

- **Research**: Easy modification of models, physics, and evaluation metrics
- **Clinical Studies**: Systematic evaluation across dose levels
- **Algorithm Development**: Modular components for improving individual stages
- **Educational**: Clear separation of concerns for learning medical imaging pipelines

## Dependencies

Each project includes a `requirements.txt` file. Main dependencies:
- PyTorch
- GMI (Generative Medical Imaging library)
- MedMNIST dataset
- HuggingFace Transformers (for ResNet50)
- Standard scientific Python stack

## Next Steps

After running all three projects, you'll have:

1. **Trained Models**: Both classifier and denoiser ready for deployment
2. **Performance Analysis**: Comprehensive evaluation across dose levels  
3. **Visualizations**: Training animations and result plots
4. **Benchmarks**: Baseline performance for future improvements

This provides a complete foundation for medical imaging research combining deep learning, physics simulation, and clinical evaluation.