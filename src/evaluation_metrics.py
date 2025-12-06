"""
Statistical utility functions for model evaluation and visualization.

This module provides functions for computing per-class accuracy and 
visualizing confusion matrices with enhanced styling.
"""

import matplotlib.pyplot as plt
import numpy as np


def get_class_accuracy(y_true, y_pred, num_classes):
    """Calculate per-class accuracy using vectorized NumPy operations."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    class_acc = np.zeros(num_classes)

    for i in range(num_classes):
        class_mask = (y_true == i)
        if class_mask.sum() > 0:
            class_correct = ((y_true == i) & (y_pred == i)).sum()
            class_acc[i] = (class_correct / class_mask.sum()) * 100
        else:
            class_acc[i] = 0.0

    return class_acc


def plot_confusion_matrix(cm, num_classes, classes, normalize=True, 
                         cmap='Blues', figsize=None):
    """Plot an enhanced confusion matrix with better styling."""
    cm = np.array(cm)

    if figsize is None:
        figsize = (max(8, num_classes * 1.5), max(6, num_classes * 1.5))

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    ax.figure.colorbar(im, ax=ax)

    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)

    thresh = cm.max() / 2.0

    for i in range(num_classes):
        row_sum = cm[i].sum()
        if row_sum == 0:
            row_sum = 1

        for j in range(num_classes):
            value = cm[i, j]
            percentage = (value / row_sum) * 100

            text_color = "white" if value > thresh else "black"

            if normalize:
                if percentage > 0:
                    text = f"{int(value)}\n({percentage:.1f}%)"
                else:
                    text = "-"
            else:
                text = f"{int(value)}"

            ax.text(j, i, text, ha='center', va='center', 
                   color=text_color, fontsize=10)

    fig.tight_layout()

    return fig, ax
