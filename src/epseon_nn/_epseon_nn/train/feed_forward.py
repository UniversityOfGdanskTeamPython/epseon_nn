"""Training code for feed forward neural networks."""
from __future__ import annotations

from epseon_nn._epseon_nn.data_loaders.data_loader import DataLoader
from epseon_nn._epseon_nn.data_loaders.json_data_loader import JsonDataLoader
from epseon_nn._epseon_nn.models.feed_forward import FeedForwardModel


class TrainFeedForwardModel:
    """Code for training FeedForwardModel network."""

    def __init__(self, data_loader: DataLoader) -> None:
        self.data_loader = data_loader

    def load_train_data(self) -> None:
        self.train_data = self.data_loader.get_training_data()

    def create_model(self) -> None:
        """Construct neural network model."""
        self.network = FeedForwardModel()

    def train_network(self) -> None:
        """Train neural network."""
        self.load_train_data()
        self.create_model()
        ...

    def save_model(self) -> None:
        """Save trained model to file."""
        ...


def train_network() -> None:
    """Train FeedForwardModel neural network."""
    trainer = TrainFeedForwardModel(JsonDataLoader())
    trainer.train_network()
    # maybe some validation code?
    trainer.save_model()
