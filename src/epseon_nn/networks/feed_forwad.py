"""Module `epseon_nn.networks.feed_forward` contains all feed forward network model."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = Path(__file__).parent


def load_serialized_data_from_file(path: Path) -> List[float]:
    """Load serialized data from file in noted `path`.

    Parameters
    ----------
    path : Path
        Path to file to load.

    Returns
    -------
    List[float]
        Deserialized second column of values.
    """
    input_y: List[float] = []
    with path.open() as file_in:
        for line in file_in.readlines():
            _x, y = line.strip().split()
            input_y.append(float(y))
    return input_y


directories_list = list((THIS_DIR / "data").glob(pattern="wave.dat*"))

data_0 = [load_serialized_data_from_file(x) for x in directories_list]

X = torch.Tensor(data_0).reshape(100, 41, 165)
y = torch.Tensor(load_serialized_data_from_file(THIS_DIR / "data" / "sr9.spl"))
scaler = StandardScaler().fit(y.reshape(16500, 1))
y = torch.Tensor(scaler.transform(y.reshape(16500, 1)))
y = y.reshape(165, 100).T

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.8,
    random_state=42,
)

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, shuffle=True, batch_size=12)
valid_loader = DataLoader(test_data, batch_size=len(test_data.tensors[0]))


class EpseonNN(nn.Module):
    """Neural network with three linear layers and sigmoid activation function."""

    def __init__(self) -> None:
        """Initialize neural network."""
        super().__init__()
        self.layer_1 = nn.Linear(41 * 165, 20 * 165)
        self.layer_2 = nn.Linear(20 * 165, 500)
        self.activation_1 = nn.Sigmoid()
        self.output = nn.Linear(500, 165)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass data through layers and activation functions."""
        x = x.view(-1, 41 * 165)
        x = self.activation_1(self.layer_1(x))
        x = self.activation_1(self.layer_2(x))
        return self.output(x)


model = EpseonNN()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(num_epochs):
    loss = None
    model.train()
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        valid_loss = 0
        for inputs, targets in valid_loader:
            outputs = model(inputs)
            valid_loss += criterion(outputs, targets).item()
        valid_loss /= len(valid_loader)

    logging.debug(
        "Epoch %s: Train Loss: %s, Valid Loss: %s",
        epoch,
        loss.item() if loss else 0,
        valid_loss,
    )

torch.save(model, THIS_DIR / "weights" / "weights")

plt.plot(np.linspace(0, 10, 165), model(X[0])[0].detach().numpy())
plt.plot(np.linspace(0, 10, 165), y[0])
plt.show()
