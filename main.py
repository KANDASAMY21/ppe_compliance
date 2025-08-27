import os
import uuid
import tempfile
import subprocess
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import base64
import sys
import cv2

# --- YOLOv9 imports ---
sys.path.append(os.path.join(os.getcwd(), "yolov9"))
from model import load_model  # Your wrapper function to load YOLO model

# --- FastAPI setup ---
app = FastAPI(
    title="PPE Detection API",
    description="Detect PPE in images/videos using YOLOv9.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Serve static files ---
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Directories ---
PROCESSED_DIR = "processed_videos"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --- Job tracking dictionary ---
VIDEO_JOBS = {}

# --- Load YOLOv9 model ---
MODEL_PATH = r"D:\ppe\yolov9\runs\train\yolov9-e-finetuning2\weights\best.pt"
model = load_model(MODEL_PATH, device="cpu")


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
def home():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h2>🚀 PPE Detection API is running</h2>")


@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        temp_file = tempfile.SpooledTemporaryFile()
        temp_file.write(image_bytes)
        temp_file.seek(0)
        img = Image.open(temp_file).convert("RGB")
        results = model(img)

        rendered_img = results.render()[0]
        img_pil = Image.fromarray(rendered_img)
        buffered = tempfile.SpooledTemporaryFile()
        img_pil.save(buffered, format="JPEG")
        buffered.seek(0)
        img_str = base64.b64encode(buffered.read()).decode("utf-8")
        return JSONResponse({"image_base64": img_str})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# --- Video processing using subprocess ---
def process_video_subprocess(input_path, output_path, weights = r"D:\ppe\yolov9\runs\train\yolov9-e-finetuning2\weights\best.pt"):
    command = [
        "python", "yolov9/detect.py",
        "--weights", weights,
        "--source", input_path,
        "--imgsz", "640",
        "--conf-thres", "0.1",   # lower confidence so you see detections
        "--project", "runs/detect",
        "--name", "exp",
        "--exist-ok",
        "--save-txt",
        "--save-conf",
        "--save-crop"   # optional: saves cropped detections
    ]
    subprocess.run(command, check=True)

    # YOLO saves results to runs/detect/exp/{file}
    result_path = os.path.join("runs/detect", "exp", os.path.basename(input_path))

    # ✅ Check if detections exist
    label_dir = os.path.join("runs/detect/exp", "labels")
    if not os.path.exists(label_dir) or len(os.listdir(label_dir)) == 0:
        print("⚠️ No detections found, not writing processed video.")
        return None  # nothing to return

    # ✅ If detections exist, return processed video
    os.rename(result_path, output_path)
    return output_path

@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    temp_input_path = None
    try:
        suffix = os.path.splitext(file.filename)[1] or ".mp4"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_input_path = temp_file.name

        # Save uploaded video
        with open(temp_input_path, "wb") as f:
            while content := await file.read(1024 * 1024):
                f.write(content)

        job_id = str(uuid.uuid4())
        VIDEO_JOBS[job_id] = {"status": "processing"}

        background_tasks.add_task(process_video_subprocess, temp_input_path, job_id)

        return JSONResponse({
            "job_id": job_id,
            "status_url": f"/video-status/{job_id}",
            "download_url": f"/download-video/{job_id}"
        })

    except Exception as e:
        if temp_input_path and os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/video-status/{job_id}")
def video_status(job_id: str):
    job = VIDEO_JOBS.get(job_id)
    if not job:
        return {"status": "not_found"}
    return job


@app.get("/download-video/{job_id}")
def download_video(job_id: str):
    job = VIDEO_JOBS.get(job_id)
    if job and job.get("status") == "completed":
        return FileResponse(job["output"], media_type="video/mp4", filename=os.path.basename(job["output"]))
    return JSONResponse({"error": "Video not ready"}, status_code=404)
