import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm

from helpers.paths import python_project_relative_path, DEEP3D_MODEL_WEIGHTS_PATH
from pipeline.synthesis.deep3d import Deep3D
from pipeline.synthesis.kitti_dataset import KittiStereoDataset


def evaluate_deep3d_on_kitti_dataset() -> None:
    device = torch.device("cuda")
    model = Deep3D(device=device).to(device)
    model.load_state_dict(torch.load(DEEP3D_MODEL_WEIGHTS_PATH))

    full_resolution = (384, 1280)
    downscale_factor = 4
    downscaled_resolution = (full_resolution[0] // downscale_factor, full_resolution[1] // downscale_factor)
    eval_dataset = KittiStereoDataset(
        data_path=python_project_relative_path("data/train"),
        date="2011_09_26",
        drives=["0019", "0084"],
        full_resolution=full_resolution,
        downscaled_resolution=downscaled_resolution
    )
    model.eval()

    with torch.no_grad():
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1)
        metrics = nn.L1Loss()
        eval_loss = 0
        n_eval_losses = 0
        for batch_idx, (left_full, left_downscaled, right_gt) in tqdm(enumerate(eval_loader)):
            left_full = left_full.to(device).float()
            left_downscaled = left_downscaled.to(device).float()
            right_gt = right_gt.to(device).float()

            right_output = model(left_full, left_downscaled)
            
            loss = metrics(right_output, right_gt)

            eval_loss += loss.item()
            n_eval_losses += 1
            torch.cuda.empty_cache()

    print(f"Evaluation loss: {eval_loss / n_eval_losses}")


if __name__ == "__main__":
    evaluate_deep3d_on_kitti_dataset()
