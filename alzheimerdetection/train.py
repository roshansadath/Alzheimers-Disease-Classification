from argparse import ArgumentParser
from datetime import datetime
from typing import Dict

from randomname import get_name

from alzheimerdetection.config import output_directory
from alzheimerdetection.models.alzheimermodeltrainer import AlzheimerModelTrainer
from alzheimerdetection.models.alexnet import AlexNetTrainer
from alzheimerdetection.models.alexnet_lstm import AlexNetLSTMTrainer
from alzheimerdetection.models.mobile_vit import MobileVITTrainer
from alzheimerdetection.models.simple import ConvNetTrainer

models: Dict[str, AlzheimerModelTrainer] = {
    "alexnet": AlexNetTrainer,
    "alexnetlstm": AlexNetLSTMTrainer,
    "transformer": MobileVITTrainer,
    "conv": ConvNetTrainer,
}


def main():
    parser = ArgumentParser(description = "Trains deep learning models for alzheimer's disease detection")
    parser.add_argument("models", choices=models.keys(), nargs = "+")
    args = parser.parse_args()

    run_id = get_run_id()
    print(f"Starting run {run_id} for models: {args.models}")
    for model_name in args.models:
        print(f"Training {model_name}")
        run_name = f'{model_name}-{run_id}'
        model = models[model_name](run_name)
        model.train()
        model.save(output_directory / f'{run_name}.pth')


def get_run_id():
    run_name = get_name()
    run_date = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')
    return f'{run_date}-{run_name}'


if __name__ == "__main__":
    main()
