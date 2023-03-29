from torchvision.datasets import ImageFolder
from config import train_dataset_directory, test_dataset_directory


def load_alzheimer_mri_dataset(train_transform, test_transform) -> (ImageFolder, ImageFolder):
    train = load_alzheimer_mri_dataset_train(transform=train_transform)
    test = load_alzheimer_mri_dataset_test(transform=test_transform)
    return train, test


def load_alzheimer_mri_dataset_train(transform) -> ImageFolder:
    train = ImageFolder(train_dataset_directory, transform=transform)
    return train


def load_alzheimer_mri_dataset_test(transform) -> ImageFolder:
    test = ImageFolder(test_dataset_directory, transform=transform)
    return test
