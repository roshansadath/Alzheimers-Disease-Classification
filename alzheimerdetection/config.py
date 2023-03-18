from pathlib import Path

train_dataset_directory = Path(__file__).parent.parent / "data" / "train"
test_dataset_directory = Path(__file__).parent.parent / "data" / "test"
output_directory = Path(__file__).parent.parent / "out"
hyperparameter_path = Path(__file__).parent / "hyperparameters.yaml"