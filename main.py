import os
import uuid
import tempfile
import subprocess
import shutil
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import base64
import sys
import cv2

# --- YOLOv5 imports ---
sys.path.append(os.path.join(os.getcwd(), "yolov5"))
from model import load_model  # Your wrapper function to load YOLO model

# --- FastAPI setup ---
app = FastAPI(
    title="PPE Detection API",
    description="Detect PPE in images/videos using YOLOv5.",
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
app.mount("/processed_videos", StaticFiles(directory=PROCESSED_DIR), name="processed_videos")

# --- Job tracking dictionary ---
VIDEO_JOBS = {}


# --- Load YOLOv5 model ---
MODEL_PATH = r"D:\REZLER\PROJECTS\ppe\yolov5\runs\yolov5s-retrain\weights\best.pt"
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
def process_video_subprocess(input_path, job_id, original_filename):
    try:
        print(f"Starting video processing for job {job_id}")
        output_filename = f"{job_id}_{original_filename}"
        output_path = os.path.join(PROCESSED_DIR, output_filename) # Save to processed_videos folder

        # Use a temporary directory for detect.py output
        temp_output_dir = tempfile.mkdtemp()

        command = [
            "python", "yolov5/detect.py",
            "--weights", MODEL_PATH,
            "--source", input_path,
            "--project", temp_output_dir, # Save to temp dir
            "--name", "detection",
            "--exist-ok",
            "--conf-thres", "0.5",
        ]
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)
        print("Subprocess finished.")

        # Find the processed video file
        processed_video_dir = os.path.join(temp_output_dir, "detection")
        print(f"Looking for processed video in: {processed_video_dir}")
        processed_files = os.listdir(processed_video_dir)
        print(f"Files in processed_video_dir: {processed_files}")
        video_files = [f for f in processed_files if f.endswith(('.mp4', '.avi', '.mov'))]

        if not video_files:
            raise Exception("Processed video file not found.")

        print(f"Found video file: {video_files[0]}")
        raw_output_path = os.path.join(processed_video_dir, video_files[0])

        # Convert with ffmpeg and save to static folder
        print(f"Converting video with ffmpeg. Output: {output_path}")
        subprocess.run(['ffmpeg', '-y', '-i', raw_output_path, '-vcodec', 'libx264', output_path])
        print("ffmpeg conversion finished.")

        # Clean up the temporary directory
        shutil.rmtree(temp_output_dir)

        VIDEO_JOBS[job_id] = {
            "status": "completed",
            "output_path": output_path,
            "filename": output_filename
        }
        print(f"Job {job_id} completed successfully.")

    except Exception as e:
        print(f"Error processing video for job {job_id}: {e}")
        VIDEO_JOBS[job_id] = {"status": "failed", "error": str(e)}
    finally:
        # Clean up the temporary input file
        if os.path.exists(input_path):
            os.unlink(input_path)

@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    temp_input_path = None
    try:
        suffix = os.path.splitext(file.filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_input_path = temp_file.name
            # Save uploaded video
            while content := await file.read(1024 * 1024):
                temp_file.write(content)

        job_id = str(uuid.uuid4())
        VIDEO_JOBS[job_id] = {"status": "processing"}

        background_tasks.add_task(process_video_subprocess, temp_input_path, job_id, file.filename)

        return JSONResponse({
            "job_id": job_id,
            "status_url": f"/video-status/{job_id}",
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
        return FileResponse(job["output_path"], media_type="video/mp4", filename=job["filename"])
    return JSONResponse({"error": "Video not ready or failed"}, status_code=404)
