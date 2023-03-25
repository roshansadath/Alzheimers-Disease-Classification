from torch.hub import load
from torch.nn import CrossEntropyLoss, Linear, Module
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from alzheimerdetection.models.alzheimermodeltrainer import AlzheimerModelTrainer


class AlexNetTrainer(AlzheimerModelTrainer):
    def __init__(self, run_id):
        super().__init__(AlexNet(), 'alexnet', run_id)

    def get_preprocessing(self):
        # This transformation has to be applied according to the documentation of the model
        # see: https://pytorch.org/hub/pytorch_vision_alexnet/
        return Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class AlexNet(Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        alexnet = load('pytorch/vision:v0.10.0', 'alexnet', pretrained = False)
        input_size = alexnet.classifier[-1].in_features
        output_size = 4
        alexnet.classifier[-1] = Linear(input_size, output_size)
        self.layer = alexnet

    def forward(self, x):
        return self.layer(x)
