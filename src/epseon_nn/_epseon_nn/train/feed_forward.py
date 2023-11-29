"""Training code for feed forward neural networks."""
from __future__ import annotations

from epseon_nn._epseon_nn.data_loaders.data_loader import DataLoader
from epseon_nn._epseon_nn.data_loaders.json_data_loader import JsonDataLoader
from epseon_nn._epseon_nn.models.feed_forward import FeedForwardModel

from tensorflow.keras.losses import MeanSquaredError
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
        self.network.compile(loss=MeanSquaredError, optimizer='adam', metrics=['accuracy'])
        self.network.fit(self.train_data[0], self.train_data[1], epochs=10, batch_size=32, validation_split=0.3)

    def save_model(self) -> None:
        """Save trained model to file."""
        weights = model.get_weights()
        with open('weights.txt', 'w') as f:
            for w in weights:
                f.write(str(w) + '\n')

    def validate_network(self) -> None:
        """Validate neural network."""
        self.load_train_data()
        self.create_model()
        self.network.compile(loss=MeanSquaredError(), optimizer='adam', metrics=['accuracy'])
        self.network.evaluate(self.train_data[0], self.train_data[1])


def train_network() -> None:
    """Train FeedForwardModel neural network."""
    trainer = TrainFeedForwardModel(JsonDataLoader())
    trainer.train_network()
    trainer.validate_network()
    trainer.save_model()
