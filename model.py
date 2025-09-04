import torch
import numpy as np
from typing import Union
from PIL import Image
import pathlib
import platform

if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# Global model instance
_model = None

def load_model(
    weights_path: str = r"D:\REZLER\PROJECTS\ppe\yolov5\runs\yolov5s-retrain\weights\best.pt",
    device: str = "cpu"
):
    """
    Load YOLOv5 model with custom weights.
    """
    global _model
    if _model is None:
        try:
            # Load YOLOv5 from local repo
            _model = torch.hub.load(
                r"D:\REZLER\PROJECTS\ppe\yolov5", 
                "custom", 
                path=weights_path, 
                source="local"
            )
            _model.to(device).eval()
            print(f"✅ Model loaded successfully on {device}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    return _model


def predict_ppe(
    image: Union[str, np.ndarray, Image.Image], 
    device: str = "cpu"
):
    """
    Run PPE detection on an input image (path, numpy array, or PIL Image).
    """
    model = load_model(device=device)

    # Convert input to PIL Image if needed
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Run inference
    results = model(image)
    return results  # YOLOv5's results object

