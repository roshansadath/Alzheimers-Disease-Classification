from torch import save, softmax
from torch.hub import load
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from alzheimerdetection.models.abstractmodel import AbstractModel


class AlexNet(AbstractModel):
    def __init__(self):
        self.model = load('pytorch/vision:v0.10.0', 'alexnet')
        input_size = self.model.classifier[-1].in_features
        output_size = 4
        self.model.classifier[-1] = Linear(input_size, output_size)


    def get_preprocessing(self):
        return Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def train(self, training_data):
        criterion = CrossEntropyLoss()
        optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        trainset = DataLoader(training_data, batch_size=32)

        for epoch in tqdm(range(2), position=0, leave=False, desc='epoch'):
            running_loss = 0.0
            for i, (inputs, labels) in tqdm(enumerate(iter(trainset), 0), position=1, leave=False, desc='batch'):
                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

    def save(self, path):
        save(self.model, path)
