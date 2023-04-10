from torch.hub import load
from torch.nn import CrossEntropyLoss, Linear, Module
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from alzheimerdetection.models.alzheimermodeltrainer import AlzheimerModelTrainer


class AlexNetLSTMTrainer(AlzheimerModelTrainer):
    def __init__(self, run_id):
        super().__init__(AlexNetLSTM(), 'alexnetlstm', run_id)

    def get_preprocessing(self):
        return Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class AlexNetLSTM(Module):
    def __init__(self):
        super(AlexNetLSTM, self).__init__()
        alexnet = load('pytorch/vision:v0.10.0', 'alexnet', weights = None)
        self.features = alexnet.features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.conv_to_lstm = nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.lstm = nn.LSTM(64 * 3 * 3, 128, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.conv_to_lstm(x)
        x = x.view(x.shape[0], 1, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x
