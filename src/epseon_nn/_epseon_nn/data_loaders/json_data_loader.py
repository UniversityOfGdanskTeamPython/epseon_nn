"""Implementation of DataLoader interface for loading data from json files."""
from __future__ import annotations

from typing import TYPE_CHECKING

from epseon_nn._epseon_nn.data_loaders.data_loader import DataLoader

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    import torch
class JsonDataLoader(DataLoader):

    def __init__(self, directory: str) -> None:
        self.this_dir = Path(__file__).parent / "networks" / "data"

    def get_training_data(self) -> torch.Tensor:
        data = []
        for files in self.this_dir.iterdir():
            if files.is_file() and file.suffix == ".json":
                with file.open() as file:
                    data = json.load(file)
        return data



    def process_data(self) -> tuple[TensorDataset, TensorDataset]:
        """Return data loaded from some source."""
        x = torch.Tensor(get_training_data(self.this_dir)).reshape(1, 41)
        y = torch.Tensor(JsonDataLoader(self.this_dir.parent).get_training_data())
        scaler = StandardScaler().fit(y.reshape(41, 1))
        y = torch.Tensor(scaler.transform(y.reshape(41, 1)))

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            train_size=0.8,
            random_state=42,
        )

        train_data = TensorDataset(x_train, y_train)
        test_data = TensorDataset(x_test, y_test)

        return train_data, test_data
