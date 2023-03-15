from torch.hub import load
from torch.nn import CrossEntropyLoss, Linear
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from alzheimerdetection.models.alzheimermodel import AlzheimerModel


class AlexNet(AlzheimerModel):
    def __init__(self):
        model = load('pytorch/vision:v0.10.0', 'alexnet')
        input_size = model.classifier[-1].in_features
        output_size = 4
        model.classifier[-1] = Linear(input_size, output_size)
        super().__init__(model)

    def _get_preprocessing(self):
        return Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _train(self, training_data, device):
        batch_size = 64
        epochs = 2
        criterion = CrossEntropyLoss()
        optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        trainset = DataLoader(training_data, batch_size=batch_size)

        for _ in tqdm(range(epochs), position=0, leave=False, desc='epoch'):
            for i, data in tqdm(enumerate(iter(trainset), 0), position=1, leave=False, desc='batch', total=len(trainset)):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
