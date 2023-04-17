from torch import flatten
from torch.nn import Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU, Sequential
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from alzheimerdetection.models.alzheimermodeltrainer import AlzheimerModelTrainer


class ConvNetTrainer(AlzheimerModelTrainer):
    def __init__(self, run_id):
        super().__init__(ConvNet(), 'convnet', run_id)

    def get_preprocessing(self):
        return Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class ConvNet(Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.convs = Sequential(
            Conv2d(3, 6, 3),
            MaxPool2d(3, 3),
            Conv2d(6, 12, 5),
            MaxPool2d(5, 5),
            Conv2d(12, 24, 7),
            MaxPool2d(7, 7),
        )
        self.classifier = Sequential(
            Linear(24, 256),
            ReLU(),
            Dropout(),
            Linear(256, 4),
        )

    def forward(self, x):
        x = self.convs(x)
        x = flatten(x, 1)
        x = self.classifier(x)
        return x
