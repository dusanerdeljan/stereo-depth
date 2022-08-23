import torch

from helpers.paths import DEEP3D_MODEL_WEIGHTS_PATH, DEEP3D_MODEL_TRACE_PATH
from helpers.torch_helpers import fix_data_parallel_state_dict
from pipeline.synthesis import Deep3D

if __name__ == "__main__":
    device = torch.device("cuda")
    model = Deep3D(device=device).to(device)
    state_dict = torch.load(DEEP3D_MODEL_WEIGHTS_PATH)
    model_state_dict = fix_data_parallel_state_dict(state_dict["model_state_dict"])
    model.load_state_dict(model_state_dict)
    model.eval()

    left_full = torch.randn(1, 3, 384, 1280, dtype=torch.float32, device=device)
    left_downscaled = torch.randn(1, 3, 96, 320, dtype=torch.float32, device=device)
    traced_script_module = torch.jit.trace(model, (left_full, left_downscaled))
    traced_script_module.save(DEEP3D_MODEL_TRACE_PATH)
