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
        # This transformation has to be applied according to the documentation of the model
        # see: https://pytorch.org/hub/pytorch_vision_alexnet/
        return Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class AlexNetLSTM(Module):
    def __init__(self):
        super(AlexNetLSTM, self).__init__()
        alexnet = load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
        self.features = alexnet.features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.conv_to_lstm = nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.lstm = nn.LSTM(64 * 6 * 6, 128, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.conv_to_lstm(x)
        x = x.view(x.size(0), -1, 64 * 6 * 6) # flatten to (batch_size, seq_len, input_size)
        x, _ = self.lstm(x) # output shape: (batch_size, seq_len, hidden_size)
        x = x[:, -1, :] # take the last hidden state
        x = self.classifier(x)
        return x
