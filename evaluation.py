import torch
import torch.nn.functional as F
from torcheval.metrics import MulticlassF1Score, MulticlassPrecision, MulticlassAccuracy

def evaluate_model(model, dataloader, criterion, device, num_classes=10, average="macro"):
    """
    Evaluates the model on the given dataloader.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (DataLoader): Dataloader for the evaluation dataset.
        criterion (Loss): Loss function.
        device (torch.device): Device (CPU or GPU).
        num_classes (int): Number of classes in the dataset.
        average (str): Metric averaging strategy.

    Returns:
        tuple: F1 score, precision, accuracy, and average loss.
    """
    model.to(device)  # Move model to device
    model.eval()  # Set model to evaluation mode

    total_loss = 0  # Initialize total loss
    # Initialize metrics for evaluation
    f1_metric = MulticlassF1Score(num_classes=num_classes, average=average)
    precision_metric = MulticlassPrecision(num_classes=num_classes, average=average)
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes, average=average)

    with torch.no_grad():  # Disable gradient calculations
        for inputs, labels in dataloader:
            inputs = inputs.to(device)  # Move inputs to device
            labels = labels.to(device)  # Move labels to device

            # Get model outputs and compute loss
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels).item()
            total_loss += batch_loss

            # Compute softmax probabilities for metrics
            softmax_outputs = F.softmax(outputs, dim=-1)
            # Update metrics with predictions and ground truth
            f1_metric.update(softmax_outputs, labels) # considers both precision and recall, giving a better measure of the model’s accuracy, especially when the dataset is imbalanced.
            precision_metric.update(softmax_outputs, labels) # ratio of correctly predicted positive observations to the total predicted positives.
            accuracy_metric.update(softmax_outputs, labels) # ratio of correctly predicted observations to the total observations

    # Compute final metrics and return results
    return f1_metric.compute(), precision_metric.compute(), accuracy_metric.compute(), total_loss / len(dataloader)
