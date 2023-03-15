from abc import ABC, abstractmethod

import torch
from torch import save
from torch.cuda import is_available

from alzheimerdetection.data import load_alzheimer_mri_dataset_train


class AlzheimerModel(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def _get_preprocessing(self):
        pass

    @abstractmethod
    def _train(self, training_data, device):
        pass

    def fit(self):
        training_data = load_alzheimer_mri_dataset_train(self._get_preprocessing())
        device = torch.device('cuda:0' if is_available() else 'cpu')
        self._train(training_data, device)

    def save(self, path):
        save(self.model, path)
