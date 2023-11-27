"""Implementation of DataLoader interface for loading data from json files."""
from __future__ import annotations

from typing import TYPE_CHECKING

from epseon_nn._epseon_nn.data_loaders.data_loader import DataLoader

if TYPE_CHECKING:
    import torch


class JsonDataLoader(DataLoader):
    """Loader for JSON data."""

    def get_training_data(self) -> torch.Tensor:
        """Return data loaded from some source."""
        ...
