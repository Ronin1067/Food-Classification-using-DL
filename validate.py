"""
Validation script for ResNet50 food classification model.

This module loads the trained model and evaluates it on the validation dataset,
saving detailed results including confusion matrix and per-class accuracies.
"""

from stat_utils import get_class_accuracy, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import torchvision.models as models
import matplotlib.pyplot as plt
import pickle as pk
import torch
import sys
import os


def load_validation_data():
    """
    Load validation dataset from pickle file.

    Returns
    -------
    tuple
        (val_dataset, val_loader, batch_size, num_epochs, classes)
    """
    print("\nLoading validation dataset...")

    try:
        with open('pickle/validate.pkl', 'rb') as handle:
            val_dataset = pk.load(handle)
            val_loader = pk.load(handle)
            batch_size = pk.load(handle)
            num_epochs = pk.load(handle)
            classes = pk.load(handle)
        return val_dataset, val_loader, batch_size, num_epochs, classes
    except FileNotFoundError:
        print("Error: Validation dataset not found.")
        print("Please run train.py before running validation.")
        sys.exit(1)


def load_trained_model(num_classes, device):
    """
    Load the trained ResNet50 model.

    Parameters
    ----------
    num_classes : int
        Number of output classes
    device : torch.device
        Device to load model on

    Returns
    -------
    nn.Module
        Loaded model
    """
    print("Loading fine-tuned model...")

    try:
        model_dict = torch.load(
            "model/resnet50_food_classification_trained.pth",
            map_location=device
        )
    except FileNotFoundError:
        print("Error: Fine-tuned model not found.")
        print("Please run train.py before running validation.")
        sys.exit(1)

    # Initialize model architecture
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_fc_inputs = model.fc.in_features
    model.fc = torch.nn.Linear(num_fc_inputs, num_classes)

    # Load trained weights
    model.load_state_dict(model_dict)
    model = model.to(device)

    return model


def validate_epoch(model, val_loader, criterion, device):
    """
    Validate the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        The model to validate
    val_loader : DataLoader
        Validation data loader
    criterion : nn.Module
        Loss function
    device : torch.device
        Device to validate on

    Returns
    -------
    tuple
        (val_loss, val_accuracy, y_pred, y_true, batch_count)
    """
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    batch_count = 0
    y_pred = []
    y_true = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            batch_count += 1

            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Calculate statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)

            # Store predictions and labels
            y_pred.extend([int(tensor) for tensor in preds])
            y_true.extend([int(tensor) for tensor in labels.data])

    # Calculate metrics
    val_loss = running_loss / len(val_loader.dataset)
    val_accuracy = correct_predictions.double() / len(val_loader.dataset)

    return val_loss, val_accuracy, y_pred, y_true, batch_count


def save_validation_results(val_loss_history, val_acc_history, cm, 
                           num_classes, classes, class_acc, num_epochs, 
                           batch_count, batch_size, val_loss, val_acc):
    """
    Save all validation results to files and plots.

    Parameters
    ----------
    val_loss_history : list
        Loss history per epoch
    val_acc_history : list
        Accuracy history per epoch
    cm : numpy.ndarray
        Confusion matrix
    num_classes : int
        Number of classes
    classes : list
        Class names
    class_acc : numpy.ndarray
        Per-class accuracies
    num_epochs : int
        Number of epochs
    batch_count : int
        Number of batches per epoch
    batch_size : int
        Batch size used
    val_loss : float
        Final validation loss
    val_acc : float
        Final validation accuracy
    """
    print("\nPlotting validation data...")

    # Create results directory
    if not os.path.exists("val_results"):
        os.makedirs("val_results")

    # Plot confusion matrix
    plot_confusion_matrix(cm, num_classes, classes)
    plt.savefig('val_results/confusion_matrix.png')
    plt.close()

    # Save validation summary to text file
    with open('val_results/val_summary.txt', "w") as f:
        f.write("Validation Summary\n")
        f.write("-" * 46 + "\n")
        f.write(f"Number of Epochs: {num_epochs}\n")
        f.write(f"Number of Batches per Epoch: {batch_count}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"\nTotal Validation Loss: {val_loss:.4f}\n")
        f.write(f"Total Validation Accuracy: {val_acc * 100:.2f}%\n")
        f.write("-" * 46 + "\n")

        f.write("\nClass-wise Accuracy\n")
        f.write("-" * 46 + "\n")
        for i in range(num_classes):
            f.write(f'"{classes[i]}" Accuracy: {class_acc[i]:.2f}%\n')
        f.write("-" * 46 + "\n")

        f.write("\nValidation History\n")
        f.write("-" * 46 + "\n")
        for epoch in range(num_epochs):
            for batch in range(batch_count):
                idx = (epoch * batch_count) + batch
                f.write(
                    f'Epoch {epoch + 1} - (Batch {batch + 1}): '
                    f'Validation Loss: {val_loss_history[idx]:.4f}, '
                    f'Validation Acc: {val_acc_history[idx]:.4f}\n'
                )

    print("Validation data saved to \"val_results\" folder")


def main():
    """Main validation function."""
    # Load validation data
    val_dataset, val_loader, batch_size, num_epochs, classes = load_validation_data()

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load trained model
    num_classes = len(classes)
    model = load_trained_model(num_classes, device)

    # Setup loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Validation setup
    val_loss_history = []
    val_acc_history = []

    print("Starting evaluation...")
    print("\n" + "-" * 45)
    print(f"Number of Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print("-" * 45 + "\n")

    # Validation loop
    for epoch in range(num_epochs):
        val_loss, val_acc, y_pred, y_true, batch_count = validate_epoch(
            model, val_loader, criterion, device
        )

        # Store metrics
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(
            f'Epoch {epoch + 1} - (Batch {batch_count}): '
            f'Validation Loss: {val_loss:.4f}, '
            f'Validation Acc: {val_acc:.4f}'
        )

    # Calculate final metrics
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    class_acc = get_class_accuracy(
        y_true=y_true, y_pred=y_pred, num_classes=num_classes
    )

    # Print summary
    print("\n\nEvaluation Complete!")
    print("\nValidation Summary")
    print("-" * 46)
    print(f"Number of Epochs: {num_epochs}")
    print(f"Number of Batches per Epoch: {batch_count}")
    print(f"Batch Size: {batch_size}")
    print(f"\nTotal Validation Loss: {val_loss:.4f}")
    print(f"Total Validation Accuracy: {val_acc * 100:.2f}%")
    print("-" * 46)

    print("\nClass-wise Accuracy")
    print("-" * 46)
    for i in range(num_classes):
        print(f'"{classes[i]}" Accuracy: {class_acc[i]:.2f}%')
    print("-" * 46)

    # Save results
    save_validation_results(
        val_loss_history, val_acc_history, cm, num_classes, 
        classes, class_acc, num_epochs, batch_count, batch_size, 
        val_loss, val_acc
    )

    print("\nPlease run test.py to test model.")


if __name__ == "__main__":
    main()
