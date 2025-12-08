from abc import ABC
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.nn.modules.loss import _Loss  # noqa
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm


@dataclass
class NeuralNetworkParams:
    batch_size: int
    n_epochs: int
    train_size: float
    shuffle: bool = True
    lambda1: float = 0
    lambda2: float = 0
    direction: int = 1  # 1: minimize,  -1: maximize


class BasicNeuralNetwork(ABC):

    def __init__(
        self,
        nn_model: nn.Module,
        criterion: _Loss,
        optimizer: Optimizer,
        dataset: Dataset,
        params: NeuralNetworkParams,
        scheduler: LRScheduler = None,
    ):

        self.nn_model = nn_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataset = dataset
        self.params = params
        self.train_loader, self.val_loader = self.create_dataloaders()

    def create_dataloaders(self) -> Tuple[DataLoader, ...]:
        dataset_size = len(self.dataset)  # noqa
        train_size = int(self.params.train_size * dataset_size)
        val_size = dataset_size - train_size

        if self.params.shuffle:
            train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])
        else:
            train_indices = list(range(0, train_size))  # First 80% of data
            val_indices = list(range(train_size, dataset_size))  # Last 20% of data
            # Create Subsets
            train_dataset = Subset(self.dataset, train_indices)
            val_dataset = Subset(self.dataset, val_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params.batch_size,
            shuffle=self.params.shuffle,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=val_size,
            shuffle=self.params.shuffle,
        )

        return train_loader, val_loader

    def fit(self) -> None:

        pbar = tqdm(range(self.params.n_epochs), desc="Epoch")
        for epoch in pbar:
            self.nn_model.train()  # Set model to training mode
            train_loss = torch.zeros(1)
            for batch_idx, (x_train, y_train) in enumerate(self.train_loader):

                self.optimizer.zero_grad()  # Reset gradients
                loss = self.calc_batch_loss(x_train, y_train)  # noqa
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update weights
                train_loss += loss.item()

            train_loss /= len(self.train_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            self.nn_model.eval()  # Set model to evaluation mode (no gradients)

            val_loss = torch.zeros(1)
            with torch.no_grad():
                for x_val, y_val in self.val_loader:
                    y_val_hat = self.nn_model(x_val)  # Forward pass
                    loss = self.criterion(y_val_hat.squeeze(), y_val)  # Compute loss
                    val_loss += loss.item()

            print(
                f"\nEpoch [{epoch + 1}/{self.params.n_epochs}], Avg Train Loss: {train_loss.item():.4f}, Avg Validation Loss: {val_loss.item():.7f}"
            )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.nn_model.eval()
        with torch.no_grad():
            return self.predict_inner(x)

    def calc_batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_hat = self.nn_model(x)
        loss = self.params.direction * self.criterion(y_hat.squeeze(), y)

        if self.params.lambda1:
            l1_reg = sum(param.abs().sum() for param in self.nn_model.parameters())  # L1 norm
            loss += self.params.lambda1 * l1_reg

        if self.params.lambda2:
            l2_reg = sum(param.abs().sum() for param in self.nn_model.parameters())  # L1 norm
            loss += self.params.lambda2 * l2_reg

        return loss

    def predict_inner(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.nn_model(x)
        return y_hat
