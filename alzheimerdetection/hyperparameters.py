from yaml import Loader, load

from alzheimerdetection.config import hyperparameter_path

with open(hyperparameter_path, 'r') as file:
    hyperparameters = load(file, Loader = Loader)