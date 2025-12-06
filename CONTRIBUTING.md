# Contributing to Food Classification Project


---

## 🎯 How to Contribute

### 1. Fork the Repository

Click the "Fork" button on the top right of the repository page.

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/Food-Classification-Using-ResNet-50.git
cd Food-Classification-Using-ResNet-50
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions

### 4. Make Your Changes

- Write clean, well-documented code
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Include type hints where applicable

### 5. Test Your Changes

```bash
# Run the notebook to verify
jupyter notebook notebooks/Food_Classification_Notebook.ipynb

# Or run Python scripts
python src/model_training.py
```

### 6. Commit Your Changes

```bash
git commit -m "[TYPE] Brief description of changes"
```

**Commit message format:**
```
[FEATURE] Add model ensemble support
[FIX] Fix CUDA memory error in training
[DOCS] Update README with installation instructions
[REFACTOR] Optimize data loading pipeline
[TEST] Add unit tests for evaluation metrics
```

### 7. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 8. Submit a Pull Request

1. Go to the original repository
2. Click "New Pull Request"
3. Select your branch
4. Write a descriptive PR description
5. Submit the PR

---

## 📋 Code Style Guidelines

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use meaningful variable names

**Example:**

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to use (cuda or cpu)
    
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc
```

### Documentation

- Add docstrings to all functions and classes
- Use Google-style docstrings
- Document parameters, return types, and exceptions

### Type Hints

```python
from typing import Tuple, List

def process_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process a batch of images and labels."""
    return images.to(device), labels.to(device)
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_model.py

# Run with verbose output
python -m pytest -v
```

### Writing Tests

```python
import pytest
import torch

def test_model_output_shape():
    """Test that model output has correct shape."""
    from src.model import ResNet50Model
    
    model = ResNet50Model(num_classes=10)
    batch_size = 32
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    output = model(input_tensor)
    
    assert output.shape == (batch_size, 10)
```

---

## 🐛 Reporting Issues

### Bug Reports

When reporting a bug, please include:

1. **Title** - Concise description of the issue
2. **Environment** - Python version, PyTorch version, OS
3. **Reproduction** - Steps to reproduce the bug
4. **Expected behavior** - What should happen
5. **Actual behavior** - What actually happens
6. **Error message** - Full traceback if applicable
7. **Screenshots** - If relevant

**Example:**

```
Title: CUDA out of memory error during training

Environment:
- Python 3.9.0
- PyTorch 2.0.0
- GPU: NVIDIA RTX 3090
- OS: Ubuntu 20.04

Steps to reproduce:
1. Load Food_Classification_Notebook.ipynb
2. Set batch_size to 128
3. Click "Run All"

Expected: Training completes without errors
Actual: CUDA out of memory error after epoch 3

Error message:
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB...
```

### Feature Requests

When requesting a feature:

1. **Title** - Clear feature name
2. **Description** - What you want to add
3. **Use case** - Why this feature is needed
4. **Implementation ideas** - Optional suggestions

---

## 📚 Documentation

### Updating Documentation

- Edit `.md` files in `docs/` folder
- Update inline comments in code
- Keep README.md in sync with changes
- Update CITATION.cff if needed

### Documentation Standards

- Use clear, concise language
- Include examples where applicable
- Link to related sections
- Keep formatting consistent

---

## 🔄 Review Process

### What We Look For

- ✅ Code follows style guidelines
- ✅ Tests pass successfully
- ✅ Documentation is updated
- ✅ Commit messages are descriptive
- ✅ No unnecessary dependencies added
- ✅ Changes address the issue or feature request

### Timeline

- Simple changes: Review within 1-2 days
- Complex changes: Review within 3-5 days
- Additional discussion may be needed

---

## 🎓 Resources

- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

---

## ✅ Checklist Before Submitting PR

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added to complex sections
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests pass locally
- [ ] Commit messages are descriptive
- [ ] No unnecessary dependencies added

---

