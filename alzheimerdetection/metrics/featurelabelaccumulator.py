from typing import Any

from numpy import concatenate, vstack
from torchmetrics import Metric


class FeatureLabelAccumulator(Metric):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.accumulated_features = []
        self.accumulated_labels = []

    def update(self, features: Any, labels: Any) -> None:
        self.accumulated_features.append(features.detach().numpy())
        self.accumulated_labels.append(labels.detach().numpy())

    def compute(self) -> Any:
        accumulated_features = vstack(self.accumulated_features)
        accumulated_labels = concatenate(self.accumulated_labels)
        return accumulated_features, accumulated_labels
