from abc import ABC, abstractmethod


class AbstractModel(ABC):

    @abstractmethod
    def get_preprocessing(self):
        pass

    @abstractmethod
    def train(self, training_data):
        pass
