from torchvision.datasets import ImageFolder
from alzheimerdetection.config import train_dataset_directory, test_dataset_directory


def load_alzheimer_mri_dataset() -> (ImageFolder, ImageFolder):
    train = ImageFolder(train_dataset_directory)
    test = ImageFolder(test_dataset_directory)
    return train, test


def load_alzheimer_mri_dataset_train() -> ImageFolder:
    train = ImageFolder(train_dataset_directory)
    return train


def load_alzheimer_mri_dataset_test() -> ImageFolder:
    test = ImageFolder(test_dataset_directory)
    return test
