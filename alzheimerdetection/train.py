from argparse import ArgumentParser
from config import config

model_names = [
    "alexnet",
    "cnn",
    "transformer"
]
def main():
    parser = ArgumentParser(description="Trains deep learning models for alzheimer's disease detection")
    parser.add_argument("models", choices=model_names, nargs="+")
    args = parser.parse_args()
    print(config)


if __name__ == "__main__":
    main()