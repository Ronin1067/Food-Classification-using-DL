"""
Testing script for ResNet50 model using online dataset.

This module loads the trained model and evaluates it on the test dataset,
saving detailed results including confusion matrix and per-class accuracies.
"""

from evaluation_metrics import get_class_accuracy, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import torchvision.models as models
import matplotlib.pyplot as plt
import pickle as pk
import torch
import sys
import os


def load_test_data():
    """Load test dataset from pickle file."""
    print("\nLoading test dataset...")

    try:
        with open('pickle/test.pkl', 'rb') as handle:
            test_dataset = pk.load(handle)
            test_loader = pk.load(handle)
            batch_size = pk.load(handle)
            num_epochs = pk.load(handle)
            classes = pk.load(handle)
        return test_dataset, test_loader, batch_size, num_epochs, classes
    except FileNotFoundError:
        print("Error: Test dataset not found.")
        print("Please run model_training.py before running test.")
        sys.exit(1)


def load_trained_model(num_classes, device):
    """Load the trained ResNet50 model."""
    print("Loading fine-tuned model...")

    try:
        model_dict = torch.load(
            "model/resnet50_food_classification_trained.pth",
            map_location=device
        )
    except FileNotFoundError:
        print("Error: Fine-tuned model not found.")
        print("Please run model_training.py before running test.")
        sys.exit(1)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_fc_inputs = model.fc.in_features
    model.fc = torch.nn.Linear(num_fc_inputs, num_classes)

    model.load_state_dict(model_dict)
    model = model.to(device)

    return model


def test_model(model, test_loader, criterion, device):
    """Test the model on the test dataset."""
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    batch_count = 0
    y_pred = []
    y_true = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            batch_count += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)

            y_pred.extend([int(tensor) for tensor in preds])
            y_true.extend([int(tensor) for tensor in labels.data])

    test_loss = running_loss / len(test_loader.dataset)
    test_accuracy = correct_predictions.double() / len(test_loader.dataset)

    return test_loss, test_accuracy, y_pred, y_true, batch_count


def save_test_results(test_loss_history, test_acc_history, cm, 
                     num_classes, classes, class_acc, batch_count, 
                     batch_size, test_loss, test_acc):
    """Save all test results to files and plots."""
    print("\nPlotting test data...")

    if not os.path.exists("test_results"):
        os.makedirs("test_results")

    plot_confusion_matrix(cm, num_classes, classes)
    plt.savefig('test_results/confusion_matrix.png')
    plt.close()

    with open('test_results/test_summary.txt', "w") as f:
        f.write("Test Summary\n")
        f.write("-" * 46 + "\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"\nTotal Test Loss: {test_loss:.4f}\n")
        f.write(f"Total Test Accuracy: {test_acc * 100:.2f}%\n")
        f.write("-" * 46 + "\n")

        f.write("\nClass-wise Accuracy\n")
        f.write("-" * 46 + "\n")
        for i in range(num_classes):
            f.write(f'"{classes[i]}" Accuracy: {class_acc[i]:.2f}%\n')

    print("Test data saved to \"test_results\" folder")


def main():
    """Main testing function."""
    test_dataset, test_loader, batch_size, num_epochs, classes = load_test_data()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = len(classes)
    model = load_trained_model(num_classes, device)

    criterion = torch.nn.CrossEntropyLoss()

    test_loss_history = []
    test_acc_history = []

    print("Starting testing...")
    print("\n" + "-" * 45)
    print(f"Batch Size: {batch_size}")
    print("-" * 45 + "\n")

    test_loss, test_acc, y_pred, y_true, batch_count = test_model(
        model, test_loader, criterion, device
    )

    test_loss_history.append(test_loss)
    test_acc_history.append(test_acc)

    print(
        f'Batch {batch_count}: '
        f'Test Loss: {test_loss:.4f}, '
        f'Test Acc: {test_acc:.4f}'
    )

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    class_acc = get_class_accuracy(
        y_true=y_true, y_pred=y_pred, num_classes=num_classes
    )

    print("\n\nTesting Complete!")
    print("\nTest Summary")
    print("-" * 46)
    print(f"Batch Size: {batch_size}")
    print(f"\nTotal Test Loss: {test_loss:.4f}")
    print(f"Total Test Accuracy: {test_acc * 100:.2f}%")
    print("-" * 46)

    print("\nClass-wise Accuracy")
    print("-" * 46)
    for i in range(num_classes):
        print(f'"{classes[i]}" Accuracy: {class_acc[i]:.2f}%')
    print("-" * 46)

    save_test_results(
        test_loss_history, test_acc_history, cm, num_classes, 
        classes, class_acc, batch_count, batch_size, test_loss, test_acc
    )


if __name__ == "__main__":
    main()
