
import numpy
import torch
import torch.serialization
import numpy._core.multiarray

Float64DType = type(numpy.dtype('float64'))
torch.serialization.add_safe_globals([
    numpy.ndarray,
    numpy._core.multiarray._reconstruct,
    numpy.dtype,
    Float64DType
])

def load_model(weights_path: str = r"D:\ppe\yolov9\runs\train\yolov9-e-finetuning2\weights\best.pt", device: str = "cpu"):
    # Use torch.hub to load YOLOv9 model with custom weights
    model = torch.hub.load(r'D:\ppe\yolov9', 'custom', path=weights_path, source='local')
    model.to(device).eval()
    return model

model = load_model()

def predict_ppe(pil_img, device='cpu'):
    if model is None:
        raise RuntimeError('Model not loaded.')
    # Pass PIL image directly; YOLOv9 wrapper handles preprocessing
    results = model(pil_img)
    return results
