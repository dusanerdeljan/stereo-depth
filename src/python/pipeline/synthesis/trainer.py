from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.data
from torch import optim
from tqdm import tqdm

from pipeline.synthesis.kitti_dataset import KittiStereoDataset


@dataclass
class TrainerConfig:
    n_epochs: int = 100
    batch_size: int = 2
    learning_rate: float = 0.0002

    momentum: float = 0.9
    weight_decay: float = 1.0e-4
    step_size: int = 30
    gamma: float = 0.1
    save_path: Optional[str] = None


class Trainer:

    def __init__(self, model: nn.Module, config: TrainerConfig, train_data: KittiStereoDataset):
        self._model = model
        self._config = config
        self._train_data = train_data

        self._device = "cpu"
        if torch.cuda.is_available():
            self._device = torch.cuda.current_device()
            self._model = torch.nn.DataParallel(self._model).to(self._device)

    def save_checkpoint(self, epoch: int, model_dict: dict, optimizer_dict: dict) -> None:
        if self._config.save_path:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model_dict,
                "optimizer_state_dict": optimizer_dict,
            }, self._config.save_path)

    def train(self) -> None:
        train_loader = torch.utils.data.DataLoader(self._train_data, batch_size=self._config.batch_size, shuffle=True)

        criterion = nn.L1Loss()
        optimizer = optim.Adam(params=self._model.parameters(), lr=self._config.learning_rate,
                               weight_decay=self._config.weight_decay, betas=(self._config.momentum, 0.999))

        for epoch in range(self._config.n_epochs):
            train_loss = 0
            n_train_losses = 0
            self._model.train()
            for batch_idx, (left_full, left_downscaled, right_gt) in tqdm(enumerate(train_loader)):
                optimizer.zero_grad()

                left_full = left_full.to(self._device).to(self._device).float()
                left_downscaled = left_downscaled.to(self._device).float()
                right_gt = right_gt.to(self._device).float()

                right_output = self._model(left_full, left_downscaled)

                loss = criterion(right_output, right_gt)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_train_losses += 1

            print(f"Epoch: {epoch + 1}, loss: {train_loss / n_train_losses}")
            self.save_checkpoint(epoch, self._model.state_dict(), optimizer.state_dict())
