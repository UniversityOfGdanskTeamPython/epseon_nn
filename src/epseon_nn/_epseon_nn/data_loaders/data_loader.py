"""DataLoader interface class definition."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class DataLoader(ABC):
    """Abstract class with interface for loading data for neural network training."""

    @abstractmethod
    def get_training_data(self) -> torch.Tensor:
        """Return data loaded from some source."""
