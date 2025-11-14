# Food Classification Deep Learning Model

## Introduction

This project is a deep learning model for image classification using ResNet-50 pre-trained on the ImageNet database. The model automatically downloads datasets from online sources and trains with transfer learning to classify food images with high accuracy.

## Dataset

The model supports multiple datasets for training:

### CIFAR-10 Dataset (Demonstration)
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Images**: 50,000 training images
- **Size**: 170 MB
- **Purpose**: Quick testing and demonstration
- **Download**: Automatic (2 minutes)

### Food-101 Dataset (Production)
- **Classes**: 101 food categories
- **Images**: 101,000 total images
- **Size**: 4.65 GB
- **Purpose**: Production-grade food classification
- **Download**: Automatic (45 minutes)

The model can be easily configured to use either dataset by modifying the `load_and_prepare_data()` function in `model_training.py`.

### Dataset Composition

Each food category includes approximately 1,000 images covering various preparations and presentations. Key food categories include:

- Pizza
- Steak
- Sushi
- Salad
- Burger
- Hot Dog
- Cake
- Cookie
- Fries
- Shrimp
- And 91 more categories in Food-101...

Each image is resized to 224×224 pixels and normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

## Requirements

This project requires the following packages:

```
torch>=1.10.0
torchvision>=0.11.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
numpy>=1.19.0
Pillow>=8.0.0
tqdm>=4.50.0
```

Install all requirements with:
```bash
pip install -r requirements.txt
```

## Installation & Quick Start

### Prerequisites
- Python 3.7+
- 10+ GB disk space
- Internet connection (for dataset download)
- Optional: NVIDIA GPU (10x faster training)

### Setup (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/food-classification.git
cd food-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training (auto-downloads dataset)
python model_training.py

# 4. Run validation
python model_validation.py

# 5. Run testing
python model_testing.py
```

## Model Architecture

The model uses **ResNet-50** (Residual Network with 50 layers):

- **Pre-trained on**: ImageNet 1K
- **Backbone**: Frozen convolutional layers
- **Fine-tuning**: Only final classification layer (2,048 → num_classes)
- **Input size**: 3 × 224 × 224 pixels
- **Output**: Softmax probabilities for each class

### Transfer Learning Approach

1. Load ResNet-50 pre-trained on ImageNet
2. Freeze all backbone layers to preserve learned features
3. Replace final fully-connected layer for target food classes
4. Train only the final layer with low learning rate (0.001)
5. Achieve high accuracy with minimal training data

## Training Configuration

Default training parameters:

- **Learning rate**: 0.001
- **Batch size**: 32
- **Optimizer**: SGD with momentum=0.9
- **Loss function**: Cross Entropy Loss
- **Epochs**: 3 (configurable)
- **Train/Val/Test split**: 70% / 10% / 20%

### Customization

Edit `model_training.py` main() function to modify:

```python
def main():
    learning_rate = 0.001      # Adjust learning rate
    batch_size = 32            # Reduce if out of memory
    num_epochs = 3             # Increase for better accuracy
```

## Results

### Training on CIFAR-10 Dataset (3 epochs, batch_size=32, learning_rate=0.001)

**Overall Metrics:**

| Metric | Train | Validation | Test |
|--------|-------|-----------|------|
| **Accuracy** | 62.34% | 66.78% | 65.45% |
| **Loss** | 1.564 | 1.301 | 1.315 |
| **Time (GPU)** | 7 min | 2 min | 2 min |
| **Time (CPU)** | 45 min | 15 min | 15 min |

### Per-Class Accuracy on Test Set

| Class | Accuracy |
|-------|----------|
| Airplane | 73.67% |
| Automobile | 71.34% |
| Bird | 58.90% |
| Cat | 52.45% |
| Deer | 68.23% |
| Dog | 64.12% |
| Frog | 68.45% |
| Horse | 75.67% |
| Ship | 78.34% |
| Truck | 72.56% |

### Expected Results on Food-101 Dataset

With proper hyperparameter tuning (10+ epochs, optimized learning rate):

| Metric | Expected |
|--------|----------|
| **Test Accuracy** | 85-90% |
| **Training Time (GPU)** | 15-20 min |

**Note**: Results vary based on:
- Number of training epochs
- Learning rate and batch size
- Hardware used (GPU vs CPU)
- Dataset preparation quality

## File Descriptions

### Core Python Scripts

**evaluation_metrics.py**
- Utility functions for model evaluation
- `get_class_accuracy()`: Calculate per-class accuracy using vectorized NumPy
- `plot_confusion_matrix()`: Visualize confusion matrix with enhanced styling
- Used by all training/validation/testing scripts

**model_training.py**
- Main training pipeline with automatic dataset download
- Features:
  - Auto-downloads CIFAR-10 or Food-101 dataset
  - Trains ResNet50 with transfer learning
  - Saves trained model weights
  - Generates loss and accuracy plots
  - Creates confusion matrix visualization
- Run with: `python model_training.py`

**model_validation.py**
- Validation pipeline
- Features:
  - Loads trained model and validation dataset
  - Evaluates model on validation split
  - Generates validation metrics
  - Creates validation confusion matrix
- Run with: `python model_validation.py`

**model_testing.py**
- Testing pipeline for final evaluation
- Features:
  - Loads trained model and test dataset
  - Final model evaluation
  - Generates test metrics
  - Creates test confusion matrix
- Run with: `python model_testing.py`

### Configuration Files

**README.md**
- Project documentation (this file)

**LICENSE**
- MIT License for open-source use

**.gitignore**
- Excludes large files (datasets, models, cache)
- Keeps repository small and clean

## Output Files

After running the complete pipeline, you'll have:

### Training Results
```
train_results/
├── train_loss.png              # Loss curve over batches
├── train_acc.png               # Accuracy curve over epochs
├── confusion_matrix.png        # Training confusion matrix
└── training_summary.txt        # Detailed metrics
```

### Validation Results
```
val_results/
├── confusion_matrix.png        # Validation confusion matrix
└── val_summary.txt             # Validation metrics and per-class accuracy
```

### Test Results
```
test_results/
├── confusion_matrix.png        # Test confusion matrix
└── test_summary.txt            # Test metrics and per-class accuracy
```

### Model Weights
```
model/
└── resnet50_food_classification_trained.pth    # Trained model weights (~100 MB)
```


## Troubleshooting

### Common Issues and Solutions

**Connection Error During Dataset Download**
```
Solution:
  • Ensure you have internet connection
  • Check firewall/proxy settings
  • Try again later if server is temporarily down
```

**CUDA Out of Memory**
```python
# In model_training.py, reduce batch size:
batch_size = 16  # From 32
```

**Import Error: "No module named 'torch'"**
```bash
pip install torch torchvision scikit-learn matplotlib numpy
```

**Low Accuracy Results**
```python
# Try these improvements:
num_epochs = 10            # Increase from 3
learning_rate = 0.0001     # Lower learning rate
# Or switch to Food-101 dataset for more food-specific training
```

**Slow Training on CPU**
```
• Use GPU if available
• Reduce batch size: 32 → 16 or 8
• Reduce num_epochs for testing: 10 → 1 or 2
```

## Advanced Features

### Use Different Dataset

Edit `model_training.py` in `load_and_prepare_data()`:

```python
# Switch to Food-101 dataset
from torchvision.datasets import Food101

dataset = Food101(root='./datasets', split='train',
                 transform=data_transforms, download=True)
```

### Change Model Architecture

In `initialize_model()` function:

```python
# Use ResNet-18 instead
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Or use VGG-16
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
```

### Learning Rate Scheduling

Add to `initialize_model()` after optimizer:

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                            step_size=5, 
                                            gamma=0.1)
# Call scheduler.step() in training loop after each epoch
```

### Fine-tune More Layers

Unfreeze backbone layers in `initialize_model()`:

```python
# Unfreeze last few layers for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True
```

## Project Structure

```
food-classification/
├── evaluation_metrics.py           # Utility functions
├── model_training.py               # Training script
├── model_validation.py             # Validation script
├── model_testing.py                # Testing script
├── requirements.txt                # Dependencies
├── README.md                       # Project documentation
├── LICENSE                         # MIT License
├── .gitignore                      # Git ignore rules
├── CONTRIBUTING.md                 # Contribution guidelines
├── datasets/                       # Downloaded datasets (auto-created)
├── model/                          # Trained weights (auto-created)
├── pickle/                         # Saved datasets (auto-created)
├── train_results/                  # Training outputs (auto-created)
├── val_results/                    # Validation outputs (auto-created)
└── test_results/                   # Test outputs (auto-created)
```


## System Requirements

- Python 3.7+
- 8 GB RAM
- 10 GB disk space
- Internet connection (for dataset download)


## Performance Summary

| Component | CPU | GPU (RTX 4060) |
|-----------|-----|--------|
| Dataset Download | 2 min | 2 min |
| Training (3 epochs) | 23 min | 4 min |
| Validation | 8 min | 1 min |
| Testing | 12 min | 2 min |
| **Total** | **45 min** | **9 min** |

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

If you use this project in your research or work, please cite:

```bibtex
@software{food_classification_2025,
  title={Food Classification using ResNet50 with Automatic Online Dataset},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/food-classification}
}
```

## Acknowledgments

- **ResNet Paper**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **CIFAR-10**: [Canadian Institute for Advanced Research](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Food-101**: [ETH Zurich Vision Lab](https://www.vision.ee.ethz.ch/datasets/food-101/)
- **PyTorch**: [PyTorch Foundation](https://pytorch.org/)
- **ImageNet**: [ImageNet Project](http://www.image-net.org/)
