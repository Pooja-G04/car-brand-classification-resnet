import matplotlib.pyplot as plt
import torch

def plot_losses_accuracies(train_losses, train_accs, val_losses, val_accs):
    """
    Plots the training and validation losses and accuracies.

    Args:
        train_losses (list): List of training losses per epoch.
        train_accs (list): List of training accuracies per epoch.
        val_losses (list): List of validation losses per epoch.
        val_accs (list): List of validation accuracies per epoch.
    """
    epochs = range(len(train_accs))  # Create epochs list for plotting

    # Move tensors to CPU if necessary
    train_losses = [l.cpu() if isinstance(l, torch.Tensor) else l for l in train_losses]
    train_accs = [a.cpu() if isinstance(a, torch.Tensor) else a for a in train_accs]
    val_losses = [l.cpu() if isinstance(l, torch.Tensor) else l for l in val_losses]
    val_accs = [a.cpu() if isinstance(a, torch.Tensor) else a for a in val_accs]

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs, 'r', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, 'r', label='Training Loss')
    plt.plot(epochs, val_losses, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
