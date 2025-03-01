import matplotlib.pyplot as plt
import numpy as np
import torchvision

def imshow(inp, title=None):
    """
    Displays an image with optional title.

    Args:
        inp (Tensor): Image tensor.
        title (str): Optional title for the image.
    """
    # Convert tensor to numpy array and transpose dimensions
    inp = inp.numpy().transpose((1, 2, 0))
    
    # Denormalize the image using mean and std deviation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)  # Clip values to be in range [0, 1]
    
    # Display the image
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.show()

def display_batch(dataloader, class_names):
    """
    Displays a batch of images from the dataloader.

    Args:
        dataloader (DataLoader): Dataloader for a dataset.
        class_names (list): List of class names.
    """
    # Get a batch of images and labels
    inputs, classes = next(iter(dataloader))
    
    # Create a grid of images
    out = torchvision.utils.make_grid(inputs)
    
    # Display the images with class labels as titles
    imshow(out, title=[class_names[x] for x in classes])
