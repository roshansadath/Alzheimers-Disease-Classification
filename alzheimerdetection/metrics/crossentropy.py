from typing import Any

from torch.nn import CrossEntropyLoss
from torchmetrics import Metric


class CrossEntropy(Metric):
    def __init__(self):
        super().__init__()
        self.cross_entropy_sum = 0
        self.count = 0
        self.cross_entropy = CrossEntropyLoss()

    def update(self, preds: Any, targets: Any) -> None:
        self.cross_entropy_sum = self.cross_entropy(preds, targets)
        self.count += 1

    def compute(self) -> Any:
        return self.cross_entropy_sum/self.count
