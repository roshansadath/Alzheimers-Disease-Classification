from argparse import ArgumentParser
from typing import Dict

from matplotlib.pyplot import show, title

from alzheimerdetection.config import output_directory
from alzheimerdetection.models.alexnet import AlexNetTrainer
from alzheimerdetection.models.alzheimermodeltrainer import AlzheimerModelTrainer
from alzheimerdetection.models.alexnet_lstm import AlexNetLSTMTrainer
from alzheimerdetection.models.mobile_vit import MobileVITTrainer

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from seaborn import scatterplot
from pandas import DataFrame

models: Dict[str, AlzheimerModelTrainer] = {
    "alexnet": AlexNetTrainer,
    "alexnetlstm": AlexNetLSTMTrainer,
    "transformer": MobileVITTrainer,
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
        print(metrics)
        for metric in metrics[:-2]:
            print(metric)

        plot_tsne(metrics[-2])
        plot_confusion_matrix(metrics[-1])


def plot_confusion_matrix(matrix):
    m = ConfusionMatrixDisplay(matrix.detach().numpy())
    m.plot()
    show()


def plot_tsne(accumulator):
    features, labels = accumulator
    tsne = TSNE(n_components=2, perplexity=25, learning_rate=600, n_iter=900)
    tsne_features = tsne.fit_transform(features)

    tsne_df = DataFrame(data=tsne_features, columns=['t-SNE 1', 't-SNE 2'])
    tsne_df['label'] = labels

    label_to_class = {
        0: "MildDemented",
        1: "ModerateDemented",
        2: "NonDemented",
        3: "VeryMilDemented"
    }
    tsne_df['label'] = tsne_df['label'].map(lambda x: label_to_class[x])

    scatterplot(data=tsne_df, x='t-SNE 1', y='t-SNE 2', hue='label', palette='tab10')
    show()


if __name__ == "__main__":
    main()