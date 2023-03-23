from argparse import ArgumentParser
from typing import Dict

from alzheimerdetection.config import output_directory
from alzheimerdetection.models.alexnet import AlexNetTrainer
from alzheimerdetection.models.alzheimermodeltrainer import AlzheimerModelTrainer

models: Dict[str, AlzheimerModelTrainer] = {
    "alexnet": AlexNetTrainer,
    "cnn": lambda: ...,
    "transformer": lambda: ...,
}


def main():
    parser = ArgumentParser(description="Trains deep learning models for alzheimer's disease detection")
    parser.add_argument("--models", "-m", choices=models.keys(), nargs="+")
    parser.add_argument("--path", "-p", nargs="+")
    args = parser.parse_args()

    for model_name, path in zip(args.models, args.path):
        print(f"Training {model_name}")
        model = models[model_name]("test")
        model.load(output_directory/path)
        metrics = model.test()
        for metric in metrics:
            print(metric)


if __name__ == "__main__":
    main()