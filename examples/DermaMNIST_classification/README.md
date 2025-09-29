# DermaMNIST Classification Example

This example demonstrates a CNN-based classification approach for the DermaMNIST dataset using the GMI framework.

## Key Components

### 1. Dataset
- **DermaMNIST**: 7 classes of skin lesions from the HAM10000 dataset
- **Image size**: 3×64×64 (RGB images)
- **Classes**:
  - actinic keratoses and intraepithelial carcinoma
  - basal cell carcinoma
  - benign keratosis-like lesions
  - dermatofibroma
  - melanoma
  - melanocytic nevi
  - vascular lesions

### 2. Categorical Random Variable
A new implementation in `gmi.random_variable.categorical` that includes:

- **CategoricalRandomVariable**: Basic categorical distribution defined by logits
- **ConditionalCategoricalRandomVariable**: Conditional categorical with neural network mapping
- Proper log probability computation using log-softmax
- Support for codebooks to map class indices to human-readable names

### 3. CNN Classifier
- Based on `gmi.network.SimpleCNN`
- Input: 3×64×64 RGB images
- Output: 7-class logits
- Architecture: [16, 32, 64, 128, 256, 128, 64, 32, 16] hidden channels

### 4. Training & Visualization
- **Training**: 15 epochs with cross-entropy loss (negative log-likelihood)
- **Animation**: Creates MP4 showing training progress with:
  - Single image per frame (cycling through different images)
  - True label displayed as subplot title
  - Bar chart comparing predicted probabilities vs true one-hot vector
  - Y-axis from 0 to 1 showing probability mass for each class

### 5. Results
- Final test accuracy: ~68%
- Model shows strong performance on "melanocytic nevi" class
- Other classes may need more training data or class balancing

## Files Created

1. `main.py` - Main classification example with training and animation
2. `gmi/random_variable/categorical.py` - Categorical random variable implementation
3. `dermamnist_classification_animation.mp4` - Training progress visualization
4. `test_categorical.py` - Unit tests for categorical implementation

## Usage

```bash
cd /workspace/gmi/examples/DermaMNIST_classification
python main.py
```

This will:
1. Download the DermaMNIST dataset (if not already present)
2. Train the CNN classifier for 15 epochs
3. Generate an MP4 animation showing training progress
4. Report final test accuracy and per-class performance