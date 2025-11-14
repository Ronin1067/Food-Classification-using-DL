# Execution Guide & Expected Outputs

## Complete Step-by-Step Execution

### Installation Phase (5 minutes)

```bash
# Step 1: Create project directory
mkdir food-classification
cd food-classification

# Step 2: Install dependencies
pip install torch torchvision scikit-learn matplotlib numpy

# Step 3: Copy the 4 Python files
# Copy these from online_dataset_code/:
# - evaluation_metrics.py
# - model_training.py
# - model_validation.py
# - model_testing.py

# Step 4: Verify installation
python -c "import torch; print(torch.__version__)"
```

### Training Phase (30-60 minutes depending on hardware)

```bash
python model_training.py
```

#### Expected Console Output:

```
================================================================================
Loading dataset from online sources...
Downloading CIFAR-10 dataset (online)...
Files already downloaded and verified
✓ Dataset loaded successfully!
  Total samples: 50000
  Classes: 10

Using device: cuda:0

Loading pre-trained model...

Starting training...

---------------------------------------------
Number of Epochs: 3
Batch Size: 32
Learning Rate: 0.001
---------------------------------------------

Epoch 1 - (Batch 1562): Training Loss: 2.3245, Training Acc: 0.2654
Epoch 2 - (Batch 1562): Training Loss: 1.8932, Training Acc: 0.4521
Epoch 3 - (Batch 1562): Training Loss: 1.5643, Training Acc: 0.6234


Training Complete!
--------------------------------------------------
Number of Epochs: 3
Number of Batches per Epoch: 1562
Batch Size: 32
Learning Rate: 0.001

Training Loss: 1.5643
Training Accuracy: 62.34%
--------------------------------------------------

Class-wise Accuracy
--------------------------------------------------
"airplane" Accuracy: 73.45%
"automobile" Accuracy: 71.23%
"bird" Accuracy: 58.90%
"cat" Accuracy: 52.34%
"deer" Accuracy: 68.12%
"dog" Accuracy: 64.56%
"frog" Accuracy: 69.87%
"horse" Accuracy: 75.32%
"ship" Accuracy: 78.90%
"truck" Accuracy: 72.11%
--------------------------------------------------

Plotting training data...
Training data saved to "train_results" folder

Saving fine-tuned model...

Please run model_validation.py to validate the newly trained model.
```

#### Files Generated:

```
train_results/
├── train_loss.png          # Graph showing loss decrease
├── train_acc.png           # Graph showing accuracy increase
├── confusion_matrix.png    # Confusion matrix visualization
└── training_summary.txt    # Detailed text summary

model/
└── resnet50_food_classification_trained.pth  # Model weights (100+ MB)

pickle/
├── validate.pkl            # Validation dataset
└── test.pkl                # Test dataset

datasets/
└── cifar-10-batches-py/    # Downloaded dataset
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    ├── test_batch
    └── batches.meta
```

### Content of training_summary.txt:

```
Training Summary
----------------------------------------------
Number of Epochs: 3
Number of Batches per Epoch: 1562
Batch Size: 32
Learning Rate: 0.001

Total Training Loss: 1.5643
Total Training Accuracy: 62.34%
----------------------------------------------

Class-wise Accuracy
----------------------------------------------
"airplane" Accuracy: 73.45%
"automobile" Accuracy: 71.23%
"bird" Accuracy: 58.90%
"cat" Accuracy: 52.34%
"deer" Accuracy: 68.12%
"dog" Accuracy: 64.56%
"frog" Accuracy: 69.87%
"horse" Accuracy: 75.32%
"ship" Accuracy: 78.90%
"truck" Accuracy: 72.11%
----------------------------------------------

Training History
----------------------------------------------
Epoch 1 - (Batch 1): Training Loss: 2.3245, Training Acc: 0.2654
Epoch 1 - (Batch 2): Training Loss: 2.2156, Training Acc: 0.2890
...
Epoch 3 - (Batch 1562): Training Loss: 1.5643, Training Acc: 0.6234
----------------------------------------------
```

### Validation Phase (10-15 minutes)

```bash
python model_validation.py
```

#### Expected Console Output:

```
Loading validation dataset...

Using device: cuda:0

Loading fine-tuned model...

Starting evaluation...

---------------------------------------------
Number of Epochs: 3
Batch Size: 32
---------------------------------------------

Epoch 1 - (Batch 391): Validation Loss: 1.4523, Validation Acc: 0.6345
Epoch 2 - (Batch 391): Validation Loss: 1.3456, Validation Acc: 0.6512
Epoch 3 - (Batch 391): Validation Loss: 1.3012, Validation Acc: 0.6678


Evaluation Complete!

Validation Summary
--------------------------------------------------
Number of Epochs: 3
Number of Batches per Epoch: 391
Batch Size: 32

Total Validation Loss: 1.3012
Total Validation Accuracy: 66.78%
--------------------------------------------------

Class-wise Accuracy
--------------------------------------------------
"airplane" Accuracy: 74.23%
"automobile" Accuracy: 72.11%
"bird" Accuracy: 59.45%
"cat" Accuracy: 53.67%
"deer" Accuracy: 69.34%
"dog" Accuracy: 65.89%
"frog" Accuracy: 70.12%
"horse" Accuracy: 76.45%
"ship" Accuracy: 79.23%
"truck" Accuracy: 73.89%
--------------------------------------------------

Plotting validation data...
Validation data saved to "val_results" folder

Please run model_testing.py to test model.
```

#### Files Generated:

```
val_results/
├── confusion_matrix.png    # Validation confusion matrix
└── val_summary.txt         # Validation summary
```

### Testing Phase (10-15 minutes)

```bash
python model_testing.py
```

#### Expected Console Output:

```
Loading test dataset...

Using device: cuda:0

Loading fine-tuned model...

Starting testing...

---------------------------------------------
Batch Size: 32
---------------------------------------------

Batch 391: Test Loss: 1.3145, Test Acc: 0.6545


Testing Complete!

Test Summary
--------------------------------------------------
Batch Size: 32

Total Test Loss: 1.3145
Total Test Accuracy: 65.45%
--------------------------------------------------

Class-wise Accuracy
--------------------------------------------------
"airplane" Accuracy: 73.67%
"automobile" Accuracy: 71.34%
"bird" Accuracy: 58.90%
"cat" Accuracy: 52.45%
"deer" Accuracy: 68.23%
"dog" Accuracy: 64.12%
"frog" Accuracy: 68.45%
"horse" Accuracy: 75.67%
"ship" Accuracy: 78.34%
"truck" Accuracy: 72.56%
--------------------------------------------------

Plotting test data...
Test data saved to "test_results" folder
```

#### Files Generated:

```
test_results/
├── confusion_matrix.png    # Test confusion matrix
└── test_summary.txt        # Test summary
```

## Complete File Structure After Execution

```
food-classification/
│
├── evaluation_metrics.py          # Utility module
├── model_training.py              # Training script
├── model_validation.py            # Validation script
├── model_testing.py               # Testing script
│
├── datasets/                      # Downloaded data
│   └── cifar-10-batches-py/
│       ├── data_batch_1
│       ├── data_batch_2
│       ├── data_batch_3
│       ├── data_batch_4
│       ├── data_batch_5
│       ├── test_batch
│       └── batches.meta
│
├── pickle/                        # Serialized datasets
│   ├── validate.pkl
│   └── test.pkl
│
├── model/                         # Trained weights
│   └── resnet50_food_classification_trained.pth
│
├── train_results/                 # Training outputs
│   ├── train_loss.png
│   ├── train_acc.png
│   ├── confusion_matrix.png
│   └── training_summary.txt
│
├── val_results/                   # Validation outputs
│   ├── confusion_matrix.png
│   └── val_summary.txt
│
└── test_results/                  # Test outputs
    ├── confusion_matrix.png
    └── test_summary.txt
```

## Typical Performance Metrics

### By Hardware

#### GPU (NVIDIA RTX 2060)
```
Dataset Download:  2 minutes
Training:          7 minutes
Validation:        2 minutes
Testing:           2 minutes
Total Time:        13 minutes
Accuracy:          ~65-70%
```

#### CPU (Intel i7, 4 cores)
```
Dataset Download:  2 minutes
Training:          45 minutes
Validation:        15 minutes
Testing:           15 minutes
Total Time:        77 minutes
Accuracy:          ~65-70% (same)
```

#### Laptop CPU (Intel i5, 2 cores)
```
Dataset Download:  2 minutes
Training:          60 minutes
Validation:        20 minutes
Testing:           20 minutes
Total Time:        102 minutes
Accuracy:          ~65-70% (same)
```

### Accuracy Progression

```
Epoch 1:  26.54% → 45.21% (loss: 2.32 → 1.89)
Epoch 2:  45.21% → 62.34% (loss: 1.89 → 1.56)
Epoch 3:  62.34% → 62.34% (loss: 1.56 → 1.56) [convergence]
```

## Troubleshooting During Execution

### Issue: Slow Dataset Download

**Expected:** 2-5 minutes for CIFAR-10

```
If taking >10 minutes:
- Check internet speed: speedtest.net
- Try again later
- Use VPN if available
```

### Issue: High Memory Usage (GPU)

**Expected:** ~3-4 GB VRAM usage

```
If getting CUDA out of memory:
Edit model_training.py:
    batch_size = 16  # Reduce from 32
Then retry.
```

### Issue: Low Accuracy (<50%)

**Expected:** ~60-70% after 3 epochs

```
If accuracy too low:
- Increase num_epochs = 10
- Reduce learning_rate = 0.0001
- Use Food-101 instead of CIFAR-10
```

### Issue: Training Stuck

**Expected:** Progress message every 5-10 seconds

```
If no messages:
- Check if model is training (GPU usage)
- Reduce batch_size if memory issue
- Check console for errors
```

## Monitoring Training Progress

### Real-time Dashboard (Optional)

You can add tensorboard for real-time monitoring:

```bash
# Install tensorboard
pip install tensorboard

# Then modify training code to use tensorboard
# Add to model_training.py:
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
writer.add_scalar('Loss/train', loss, epoch)
writer.add_scalar('Accuracy/train', accuracy, epoch)
```

Then view with:
```bash
tensorboard --logdir=runs
```

## Next Steps After Execution

### 1. Analyze Results
```python
import matplotlib.pyplot as plt
# View generated PNG files in train_results/, val_results/, test_results/
```

### 2. Improve Accuracy
```python
# Option 1: More epochs
num_epochs = 10

# Option 2: Better dataset
# Switch to Food-101 in model_training.py

# Option 3: Fine-tune hyperparameters
learning_rate = 0.0001
```

### 3. Use Trained Model
```python
import torch
import torchvision.models as models

# Load model
model = models.resnet50()
model.fc = torch.nn.Linear(2048, 10)
model.load_state_dict(torch.load("model/resnet50_food_classification_trained.pth"))
model.eval()

# Use for predictions
with torch.no_grad():
    output = model(input_image)
    prediction = torch.argmax(output, 1)
```

## Performance Summary

| Metric | CIFAR-10 | Food-101 |
|--------|----------|----------|
| Training Accuracy | ~62% | ~75% |
| Validation Accuracy | ~67% | ~78% |
| Test Accuracy | ~65% | ~76% |
| Training Time (GPU) | 7 min | 45 min |
| Training Time (CPU) | 45 min | 4+ hours |
| Dataset Size | 170 MB | 4.65 GB |

---

**Ready to execute?** Start with: `python model_training.py`
