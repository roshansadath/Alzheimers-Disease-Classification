from abc import ABC, abstractmethod

from torch import save

from alzheimerdetection.data import load_alzheimer_mri_dataset_train


class AlzheimerModel(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def _get_preprocessing(self):
        pass

    @abstractmethod
    def _train(self, training_data):
        pass

    def fit(self):
        training_data = load_alzheimer_mri_dataset_train(self._get_preprocessing())
        self._train(training_data)

    def save(self, path):
        save(self.model, path)
