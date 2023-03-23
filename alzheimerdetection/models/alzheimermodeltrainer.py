from abc import ABC

import torch
from torch import save
from torch.cuda import is_available
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from alzheimerdetection.config import tensorboard_directory
from alzheimerdetection.data import load_alzheimer_mri_dataset_train
from alzheimerdetection.hyperparameters import hyperparameters


class AlzheimerModelTrainer(ABC):
    def __init__(self, model, hyperparameter_key, run_id):
        self.model = model
        self.hyperparameters = hyperparameters[hyperparameter_key]
        self.run_id = run_id

    def get_preprocessing(self):
        return None

    def train(self):
        tensorboard_writer = self._setup_tensorboard()
        criterion = CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(),
                         lr=self.hyperparameters['learning_rate'],
                         betas=self.hyperparameters['betas'])

        trainset = self.__get_trainset()

        device = torch.device('cuda:0' if is_available() else 'cpu')

        running_loss = 0.0
        for epoch in tqdm(range(self.hyperparameters['epochs']), position=0, leave=False, desc='epoch'):
            for i, data in tqdm(enumerate(iter(trainset), 0), position=1, leave=False, desc='batch', total=len(trainset)):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9:
                    tensorboard_writer.add_scalar('training loss', running_loss / 10, epoch * len(trainset) + i)
                    running_loss = 0.0


    def __get_trainset(self):
        preprocessing_pipeline = self.get_preprocessing()
        training_data = load_alzheimer_mri_dataset_train(preprocessing_pipeline)
        trainset = DataLoader(training_data, batch_size=self.hyperparameters['batch_size'])
        return trainset

    def _setup_tensorboard(self):
        tensorboard_writer = SummaryWriter(tensorboard_directory / self.run_id)
        # tensorboard_writer.add_graph(self.model)
        return tensorboard_writer

    def save(self, path):
        save(self.model, path)
