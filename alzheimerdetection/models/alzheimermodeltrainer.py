from abc import ABC

import torch
from torch import save
from torch.cuda import is_available
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from alzheimerdetection.data import load_alzheimer_mri_dataset_train
from alzheimerdetection.hyperparameters import hyperparameters


class AlzheimerModelTrainer(ABC):
    def __init__(self, model, hyperparameter_key):
        self.model = model
        self.hyperparameters = hyperparameters[hyperparameter_key]

    def get_preprocessing(self):
        return None

    def train(self):
        criterion = CrossEntropyLoss()
        optimizer = SGD(self.model.parameters(), lr=self.hyperparameters['learning_rate'],
                        momentum=self.hyperparameters['momentum'])

        trainset = self.__get_trainset()

        device = torch.device('cuda:0' if is_available() else 'cpu')

        for _ in tqdm(range(self.hyperparameters['epochs']), position=0, leave=False, desc='epoch'):
            for data in tqdm(iter(trainset), position=1, leave=False, desc='batch', total=len(trainset)):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def __get_trainset(self):
        preprocessing_pipeline = self.get_preprocessing()
        training_data = load_alzheimer_mri_dataset_train(preprocessing_pipeline)
        trainset = DataLoader(training_data, batch_size=self.hyperparameters['batch_size'])
        return trainset

    def save(self, path):
        save(self.model, path)
