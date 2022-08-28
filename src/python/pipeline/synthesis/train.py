import os.path

import torch

from helpers.paths import DEEP3D_MODEL_WEIGHTS_PATH
from helpers.torch_helpers import fix_data_parallel_state_dict
from pipeline.synthesis.deep3d import Deep3D
from pipeline.synthesis.kitti_dataset import KittiStereoDataset
from pipeline.synthesis.trainer import TrainerConfig, Trainer


def train_deep3d_on_kitti_dataset(resume: bool = False) -> None:
    device = torch.device("cuda")
    model = Deep3D(device=device)
    if resume:
        assert os.path.exists(DEEP3D_MODEL_WEIGHTS_PATH), "Cannot resume training since no previous session was found."
        model.load_state_dict(fix_data_parallel_state_dict(torch.load(DEEP3D_MODEL_WEIGHTS_PATH)["model_state_dict"]))
        print("Loaded model from previous checkpoint, continuing training...")

    full_resolution = (384, 1280)

    downscale_factor = 4
    downscaled_resolution = (full_resolution[0] // downscale_factor, full_resolution[1] // downscale_factor)
    train_dataset = KittiStereoDataset(
        data_path="../../data/train",
        date="2011_09_26",
        drives=["0011", "0019", "0022", "0052", "0059", "0084", "0091", "0093", "0095", "0096"],
        full_resolution=full_resolution,
        downscaled_resolution=downscaled_resolution
    )

    trainer_config = TrainerConfig(
        batch_size=2,
        save_path=DEEP3D_MODEL_WEIGHTS_PATH
    )

    trainer = Trainer(
        model=model,
        config=trainer_config,
        train_data=train_dataset
    )
    trainer.train()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    train_deep3d_on_kitti_dataset(resume=True)
