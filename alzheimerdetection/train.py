from argparse import ArgumentParser
from datetime import datetime
from typing import Dict

from randomname import get_name

from config import output_directory
from models.alzheimermodeltrainer import AlzheimerModelTrainer
from models.alexnet import AlexNetTrainer

models: Dict[str, AlzheimerModelTrainer] = {
    "alexnet": AlexNetTrainer,
    "cnn": lambda: ...,
    "transformer": lambda: ...,
}


def main():
    parser = ArgumentParser(description="Trains deep learning models for alzheimer's disease detection")
    parser.add_argument("models", choices=models.keys(), nargs="+")
    args = parser.parse_args()

    run_id = get_run_id()
    print(f"Starting run {run_id} for models: {args.models}")
    for model_name in args.models:
        print(f"Training {model_name}")
        model = models[model_name](run_id)
        model.train()
        model.save(output_directory / f'{model_name}-{run_id}.pth')


def get_run_id():
    run_name = get_name()
    run_date = datetime.utcnow().strftime('%Y-%m-%d-%H_%M_%S')
    return f'{run_date}-{run_name}'


if __name__ == "__main__":
    main()
