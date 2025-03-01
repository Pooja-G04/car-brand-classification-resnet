import os
import torch
from torchvision import datasets, transforms

def prepare_data(data_dir='./cars_data_10_split'):
    """
    Prepares dataloaders for training, validation, and testing.

    Args:
        data_dir (str): Directory containing train, val, and test subfolders.

    Returns:
        tuple: Dataloaders, dataset sizes, class names, and the device to use.
    """

    # Define data augmentation and normalization for training
    # and normalization for validation and testing
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),  # Randomly crop image to 224x224
            transforms.RandomHorizontalFlip(),  # Apply random horizontal flip
            transforms.ToTensor(),  # Convert image to PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet stats
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),  # Resize image to 256 on the shortest side
            transforms.CenterCrop(224),  # Center crop image to 224x224
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load datasets for train, val, and test using ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }

    # Create dataloaders for each dataset
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                       shuffle=True if x == 'train' else False, num_workers=4)
        for x in ['train', 'val', 'test']
    }

    # Get dataset sizes and class names
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    # Determine the device (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return dataloaders, dataset_sizes, class_names, device
