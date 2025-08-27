# PPE Detection API

This FastAPI project provides an endpoint to detect PPE (Personal Protective Equipment) in images using a custom-trained YOLOv9 model.

## Setup

1. Place your custom YOLOv9 `.pt` model file in the `yolov9` directory (or update the path in `main.py`).
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run the API:
   ```powershell
   uvicorn main:app --reload
   ```

## Usage

- POST `/detect` with an image file to get detection results.
- GET `/` for a health check.

## Environment Variable

- `PPE_MODEL_PATH`: Set this to override the default model path.

## Example Request

```bash
curl -X POST "http://localhost:8000/detect" -F "file=@your_image.jpg"
```
