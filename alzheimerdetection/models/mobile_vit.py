import torch
from torch.nn import Module
import torch.nn as nn
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from alzheimerdetection.models.alzheimermodeltrainer import AlzheimerModelTrainer


class MobileVITTrainer(AlzheimerModelTrainer):
    def __init__(self, run_id):
        super().__init__(MobileVIT(), 'transformer', run_id)

    def get_preprocessing(self):
        return Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class MobileVIT(Module):
    def __init__(self):
        super(MobileVIT, self).__init__()
        num_layers, patch_size, image_size, depth = 12, 16, 224, 512
        num_heads, d_ff, num_classes = 8, 2048, 4

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(3, depth, patch_size, patch_size)

        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, depth))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, depth))
        self.dropout = nn.Dropout(0.1)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(depth, num_heads, d_ff, 0.1), num_layers
        )

        self.fc = nn.Linear(depth, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x += self.position_embedding
        x = self.dropout(x)

        x = self.transformer_encoder(x)

        x = x[:, 0, :]

        x = self.fc(x)

        return x
