from abc import ABC

import torch
from torch import load, no_grad, save
from torch.cuda import is_available
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import AUROC, Accuracy, ConfusionMatrix, F1Score
from tqdm import tqdm

from alzheimerdetection.config import tensorboard_directory
from alzheimerdetection.data import load_alzheimer_mri_dataset_test, load_alzheimer_mri_dataset_train
from alzheimerdetection.hyperparameters import hyperparameters
from alzheimerdetection.metrics.crossentropy import CrossEntropy


class AlzheimerModelTrainer(ABC):
    def __init__(self, model, hyperparameter_key, run_id):
        self.model = model
        self.hyperparameters = hyperparameters[hyperparameter_key]
        self.run_id = run_id
        self.metrics = [
            CrossEntropy(),
            Accuracy(task='multiclass', num_classes=4),
            F1Score(task='multiclass', num_classes=4),
            AUROC(task='multiclass', num_classes=4),
            ConfusionMatrix(task="multiclass", num_classes=4)
        ]
        self.device = torch.device('cuda:0' if is_available() else 'cpu')

    def get_preprocessing(self):
        return None

    def train(self):
        tensorboard_writer = self._setup_tensorboard()
        trainset = self.__get_trainset()
        testset = self.__get_testset()

        criterion = CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(),
                         lr=self.hyperparameters['learning_rate'],
                         betas=self.hyperparameters['betas'])

        self.model.to(self.device)
        [metric.to(self.device) for metric in self.metrics]
        training_loss = CrossEntropy()
        training_loss.to(self.device)
        for epoch in tqdm(range(self.hyperparameters['epochs']), position=0, leave=False, desc='epoch'):
            self.model.train()
            for i, data in tqdm(enumerate(iter(trainset), 0), position=1, leave=False, desc='batch', total=len(trainset)):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                training_loss(outputs, labels)

            self.model.eval()
            with no_grad():
                step = epoch
                tensorboard_writer.add_scalar('loss/train', training_loss.compute(), step)
                training_loss.reset()

                scores = self._compute_metrics(testset)
                tensorboard_writer.add_scalar('loss/test', scores[0], step)
                for score, name in zip(scores[1:], ['accuracy', 'f1score', 'AUROC']):
                    tensorboard_writer.add_scalar(f'metric/{name}', score, step)

    def test(self):
        testset = self.__get_testset()
        [metric.to(self.device) for metric in self.metrics]
        self.model.to(self.device)
        self.model.eval()
        return self._compute_metrics(testset)

    def save(self, path):
        save(self.model, path)

    def load(self, path):
        self.model = load(path, self.device)

    def __get_trainset(self):
        preprocessing_pipeline = self.get_preprocessing()
        training_data = load_alzheimer_mri_dataset_train(preprocessing_pipeline)
        trainset = DataLoader(training_data, batch_size=self.hyperparameters['batch_size'], shuffle=True, num_workers=12)
        return trainset

    def __get_testset(self):
        test_data = load_alzheimer_mri_dataset_test(self.get_preprocessing())
        testset = DataLoader(test_data, batch_size=self.hyperparameters['batch_size'])
        return testset

    def _setup_tensorboard(self):
        tensorboard_writer = SummaryWriter(tensorboard_directory / self.run_id)
        # tensorboard_writer.add_graph(self.model)
        return tensorboard_writer

    def _compute_metrics(self, testset):
        with no_grad():
            for x, y in testset:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                for metric in self.metrics:
                    metric(pred, y)

        scores = [metric.compute() for metric in self.metrics]
        [metric.reset() for metric in self.metrics]

        return scores
