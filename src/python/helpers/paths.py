import os.path
from datetime import datetime

PYTHON_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
DEEP3D_MODEL_WEIGHTS_PATH = os.path.join(PYTHON_ROOT_PATH, "data", "model", "trained_0011_drive.pth")
DEEP3D_MODEL_TRACE_PATH = os.path.join(PYTHON_ROOT_PATH, "data", "model", "traced_deep3d_model_cuda.pt")


def python_project_relative_path(*relative_path: str) -> str:
    return os.path.join(PYTHON_ROOT_PATH, *relative_path)


def timestamp_folder_name() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
