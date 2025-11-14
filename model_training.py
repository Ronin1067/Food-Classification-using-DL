"""
Training script for ResNet50 food classification model.

This module handles dataset loading, model initialization, training loop,
and saving results and trained model weights.
"""

from evaluation_metrics import get_class_accuracy, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import pickle as pk
import torch
import sys
import os


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = ['pickle', 'train_results', 'model']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def load_and_prepare_data(batch_size):
    """
    Load and prepare the dataset with train/validation/test splits.

    Parameters
    ----------
    batch_size : int
        Batch size for data loaders

    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader, train_dataset, 
         val_dataset, test_dataset, classes)
    """
    print("\nLoading dataset...")

    # Define data transformations
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    try:
        dataset = ImageFolder(root='dataset', transform=data_transforms)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

    classes = dataset.classes

    # Split dataset (70% train, 10% validation, 20% test)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return (train_loader, val_loader, test_loader, train_dataset, 
            val_dataset, test_dataset, classes)


def save_split_data(val_dataset, val_loader, test_dataset, test_loader, 
                   batch_size, num_epochs, classes):
    """
    Save validation and test datasets for later use.

    Parameters
    ----------
    val_dataset : Dataset
        Validation dataset
    val_loader : DataLoader
        Validation data loader
    test_dataset : Dataset
        Test dataset
    test_loader : DataLoader
        Test data loader
    batch_size : int
        Batch size
    num_epochs : int
        Number of epochs
    classes : list
        List of class names
    """
    # Save validation data
    with open('pickle/validate.pkl', 'wb') as handle:
        pk.dump(val_dataset, handle)
        pk.dump(val_loader, handle)
        pk.dump(batch_size, handle)
        pk.dump(num_epochs, handle)
        pk.dump(classes, handle)

    # Save test data
    with open('pickle/test.pkl', 'wb') as handle:
        pk.dump(test_dataset, handle)
        pk.dump(test_loader, handle)
        pk.dump(batch_size, handle)
        pk.dump(num_epochs, handle)
        pk.dump(classes, handle)


def initialize_model(num_classes, learning_rate, device):
    """
    Initialize ResNet50 model with transfer learning setup.

    Parameters
    ----------
    num_classes : int
        Number of output classes
    learning_rate : float
        Learning rate for optimizer
    device : torch.device
        Device to train on (CPU or GPU)

    Returns
    -------
    tuple
        (model, optimizer, criterion)
    """
    print("Loading pre-trained model...")

    # Load pre-trained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze all pre-trained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    num_fc_inputs = model.fc.in_features
    model.fc = torch.nn.Linear(num_fc_inputs, num_classes)

    # Move model to device
    model = model.to(device)

    # Setup optimizer and loss function
    optimizer = torch.optim.SGD(
        model.fc.parameters(), lr=learning_rate, momentum=0.9
    )
    criterion = torch.nn.CrossEntropyLoss()

    return model, optimizer, criterion


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        The model to train
    train_loader : DataLoader
        Training data loader
    criterion : nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    device : torch.device
        Device to train on

    Returns
    -------
    tuple
        (epoch_loss, epoch_accuracy, y_pred, y_true, batch_count)
    """
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    batch_count = 0
    y_pred = []
    y_true = []

    for inputs, labels in train_loader:
        batch_count += 1

        # Move data to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate statistics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)

        # Store predictions and labels
        y_pred.extend([int(tensor) for tensor in preds])
        y_true.extend([int(tensor) for tensor in labels.data])

    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct_predictions.double() / len(train_loader.dataset)

    return epoch_loss, epoch_accuracy, y_pred, y_true, batch_count


def save_training_results(train_loss_history, train_acc_epoch_history, cm, 
                         num_classes, classes, class_acc, num_epochs, 
                         batch_count, batch_size, learning_rate, train_loss, 
                         train_acc, train_acc_batch_history):
    """
    Save all training results to files and plots.
    """
    print("\nPlotting training data...")

    # Plot loss history
    plt.figure()
    plt.plot(train_loss_history)
    plt.title('Training Loss History')
    plt.xlabel('Batches')
    plt.ylabel('Training Loss')
    plt.savefig("train_results/train_loss.png")
    plt.close()

    # Plot accuracy history
    plt.figure()
    plt.plot(train_acc_epoch_history)
    plt.title('Training Accuracy History')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.savefig("train_results/train_acc.png")
    plt.close()

    # Plot confusion matrix
    plot_confusion_matrix(cm, num_classes, classes)
    plt.savefig('train_results/confusion_matrix.png')
    plt.close()

    # Save training summary
    with open('train_results/training_summary.txt', "w") as f:
        f.write("Training Summary\n")
        f.write("-" * 46 + "\n")
        f.write(f"Number of Epochs: {num_epochs}\n")
        f.write(f"Number of Batches per Epoch: {batch_count}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate:.3f}\n")
        f.write(f"\nTotal Training Loss: {train_loss:.4f}\n")
        f.write(f"Total Training Accuracy: {train_acc * 100:.2f}%\n")
        f.write("-" * 46 + "\n")

        f.write("\nClass-wise Accuracy\n")
        f.write("-" * 46 + "\n")
        for i in range(num_classes):
            f.write(f'"{classes[i]}" Accuracy: {class_acc[i]:.2f}%\n')
        f.write("-" * 46 + "\n")

        f.write("\nTraining History\n")
        f.write("-" * 46 + "\n")
        for epoch in range(num_epochs):
            for batch in range(batch_count):
                idx = (epoch * batch_count) + batch
                f.write(
                    f'Epoch {epoch + 1} - (Batch {batch + 1}): '
                    f'Training Loss: {train_loss_history[idx]:.4f}, '
                    f'Training Acc: {train_acc_batch_history[idx]:.4f}\n'
                )

    print("Training data saved to \"train_results\" folder")


def main():
    """Main training function."""
    # Hyperparameters
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 10

    # Setup directories
    setup_directories()

    # Load and prepare data
    (train_loader, val_loader, test_loader, train_dataset, 
     val_dataset, test_dataset, classes) = load_and_prepare_data(batch_size)

    # Save validation and test data
    save_split_data(val_dataset, val_loader, test_dataset, test_loader, 
                   batch_size, num_epochs, classes)

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    num_classes = len(classes)
    model, optimizer, criterion = initialize_model(
        num_classes, learning_rate, device
    )

    # Training setup
    train_loss_history = []
    train_acc_epoch_history = []
    train_acc_batch_history = []

    print("Starting training...")
    print("\n" + "-" * 45)
    print(f"Number of Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate:.3f}")
    print("-" * 45 + "\n")

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss, epoch_acc, y_pred, y_true, batch_count = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Store metrics
        train_loss_history.append(epoch_loss)
        train_acc_batch_history.append(epoch_acc)
        train_acc_epoch_history.append(epoch_acc)

        print(
            f'Epoch {epoch + 1} - (Batch {batch_count}): '
            f'Training Loss: {epoch_loss:.4f}, '
            f'Training Acc: {epoch_acc:.4f}'
        )

    # Calculate final metrics
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    class_acc = get_class_accuracy(
        y_true=y_true, y_pred=y_pred, num_classes=num_classes
    )

    # Print summary
    print("\n\nTraining Complete!")
    print("-" * 46)
    print(f"Number of Epochs: {num_epochs}")
    print(f"Number of Batches per Epoch: {batch_count}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate:.3f}")
    print(f"\nTraining Loss: {epoch_loss:.4f}")
    print(f"Training Accuracy: {epoch_acc * 100:.2f}%")
    print("-" * 46)

    print("\nClass-wise Accuracy")
    print("-" * 46)
    for i in range(num_classes):
        print(f'"{classes[i]}" Accuracy: {class_acc[i]:.2f}%')
    print("-" * 46)

    # Save results
    save_training_results(
        train_loss_history, train_acc_epoch_history, cm, num_classes, 
        classes, class_acc, num_epochs, batch_count, batch_size, 
        learning_rate, epoch_loss, epoch_acc, train_acc_batch_history
    )

    # Save model
    print("\nSaving fine-tuned model...")
    torch.save(model.state_dict(), 
              "model/resnet50_food_classification_trained.pth")

    print("\nPlease run model_validation.py to validate the newly trained model.")


if __name__ == "__main__":
    main()
