# ChestMNIST Classification (Reorganized)

This is a reorganized version of the ChestMNIST classification example with modular structure.

## Structure

- `config.yml` - Configuration file with all hyperparameters
- `parse_config.py` - Configuration parser
- `train_classifier.py` - Model definitions and training utilities
- `animate_training_process.py` - Training animation functionality
- `main.py` - Main script that orchestrates training

## Usage

1. **Configure parameters**: Edit `config.yml` to set hyperparameters, paths, etc.

2. **Run training**:
```bash
cd /workspace/gmi/examples/ChestMNIST_classification_reorganized
python main.py
```

3. **Output**: 
   - Trained model saved to `outputs/chest_classifier.pth`
   - Training animation saved to `outputs/chestmnist_classification_animation.mp4`

## Configuration Options

Key parameters in `config.yml`:

- `data.batch_size`: Batch size for training
- `training.num_epochs`: Number of training epochs
- `training.learning_rate`: Learning rate
- `model.backbone`: Pre-trained backbone model
- `output.output_dir`: Output directory for results

## Model Architecture

- ResNet50 backbone (from HuggingFace)  
- Multi-label classification head (14 binary outputs)
- Prior logits based on training data class frequencies
- Input: 64x64 grayscale chest X-ray images
- Output: 14 binary disease labels

## Features

- Animated training visualization
- Multi-label evaluation metrics
- Data augmentation
- Configurable hyperparameters
- Model checkpointing