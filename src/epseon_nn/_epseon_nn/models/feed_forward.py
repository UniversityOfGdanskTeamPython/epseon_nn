"""Module `epseon_nn.networks.feed_forward` contains all feed forward network model."""
from __future__ import annotations

import logging

import torch
from torch import nn


class FeedForwardModel(nn.Module):
    """Neural network with three linear layers and sigmoid activation function."""

    def __init__(self) -> None:
        """Initialize neural network."""
        super().__init__()
        self.layer_sizes = (120, 70, 30)
        self.activations = (nn.Sigmoid(), nn.Sigmoid(), nn.Sigmoid(), nn.Sigmoid())

        self.layers = nn.ModuleList(
            [
                nn.Linear(60, out_features=self.layer_sizes[0]),
                *(
                    nn.Linear(in_, out_)
                    for in_, out_ in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
                ),
            ],
        )
        self.output = nn.Linear(self.layer_sizes[-1], 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass data through layers and activation functions."""
        for activation, layer in zip(self.activations, self.layers):
            x = activation(layer(x))
        return self.output(x)


if __name__ == "__main__":
    model = FeedForwardModel()

    for name, param in model.named_parameters():
        logging.warning("%s -> %s", name, param.size())
