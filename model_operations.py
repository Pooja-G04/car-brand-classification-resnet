import torch
import matplotlib.pyplot as plt
from visualization import imshow

def visualize_model(model, dataloaders, class_names, device, num_images=10):
    """
    Visualizes predictions of the model on validation data.

    Args:
        model (torch.nn.Module): The trained model.
        dataloaders (dict): Dataloaders for datasets.
        class_names (list): List of class names.
        device (torch.device): Device (CPU or GPU).
        num_images (int): Number of images to display.
    """
    model.eval()  # Set model to evaluation mode
    images_so_far = 0  # Counter for displayed images
    fig = plt.figure(figsize=(15, 10))  # Create a larger figure for better visualization
    displayed_classes = set() 

    with torch.no_grad():  # Disable gradient calculations
        for inputs, labels in dataloaders['val']:
            # Move inputs and labels to the appropriate device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Get model predictions
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get predicted class indices
            
            # Loop through batch and display images with predictions
            for j in range(inputs.size(0)):
                # Skip if the class has already been displayed
                if class_names[labels[j]] in displayed_classes:
                    continue
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)  # Create subplot
                ax.axis('off')  # Remove axes
                ax.set_title(f'True: {class_names[labels[j]]}\nPred: {class_names[preds[j]]}')  # Set title
                imshow(inputs.cpu().data[j])  # Display the image
                # Stop if we've displayed the required number of images

                # Add the class to the set of displayed classes
                displayed_classes.add(class_names[labels[j]])
                if images_so_far == num_images:
                    plt.show()
                    return
    plt.show()
