from abc import ABC

import torch
from torch import load, no_grad, save
from torch.cuda import is_available
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import AUROC, Accuracy, F1Score
from tqdm import tqdm

from config import tensorboard_directory
from data import load_alzheimer_mri_dataset_test, load_alzheimer_mri_dataset_train
from hyperparameters import hyperparameters
from metrics.crossentropy import CrossEntropy


class AlzheimerModelTrainer(ABC):
    def __init__(self, model, hyperparameter_key, run_id):
        self.model = model
        self.hyperparameters = hyperparameters[hyperparameter_key]
        self.run_id = run_id

    def get_preprocessing(self):
        return None

    def train(self):
        tensorboard_writer = self._setup_tensorboard()
        device = torch.device('cuda:0' if is_available() else 'cpu')
        trainset = self.__get_trainset()

        criterion = CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(),
                         lr=self.hyperparameters['learning_rate'],
                         betas=self.hyperparameters['betas'])
        self.model.train()
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

    def test(self):
        testset = self.__get_testset()
        device = torch.device('cuda:0' if is_available() else 'cpu')
        metrics = [
            CrossEntropy(),
            Accuracy(task='multiclass', num_classes=4),
            F1Score(task='multiclass', num_classes=4),
            AUROC(task='multiclass', num_classes=4)
        ]

        loss = CrossEntropyLoss()
        self.model.eval()
        test_loss, correct = 0, 0

        with no_grad():
            for x, y in testset:
                x, y = x.to(device), y.to(device)
                pred = self.model(x)
                test_loss += loss(pred, y).item()
                for metric in metrics:
                    metric(pred, y)

        return [metric.compute() for metric in metrics]

    def save(self, path):
        save(self.model, path)

    def load(self, path):
        self.model = load(path)

    def __get_trainset(self):
        preprocessing_pipeline = self.get_preprocessing()
        training_data = load_alzheimer_mri_dataset_train(preprocessing_pipeline)
        trainset = DataLoader(training_data, batch_size=self.hyperparameters['batch_size'])
        return trainset

    def __get_testset(self):
        test_data = load_alzheimer_mri_dataset_test(self.get_preprocessing())
        testset = DataLoader(test_data, batch_size=self.hyperparameters['batch_size'])
        return testset

    def _setup_tensorboard(self):
        tensorboard_writer = SummaryWriter(tensorboard_directory / self.run_id)
        tensorboard_writer.add_graph(self.model)
        
        return tensorboard_writer

