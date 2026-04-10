from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from app.utils.audio_processing import run_audio_inference
from app.utils.video_processing import run_video_inference
from app.utils.fusion import late_fusion

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "backend/temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = os.path.join(TEMP_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    audio_score = run_audio_inference(file_path)
    video_score = run_video_inference(file_path)

    final_score = late_fusion(audio_score, video_score)

    prediction = "FAKE" if final_score > 0.5 else "REAL"

    return {
        "audio_score": audio_score,
        "video_score": video_score,
        "final_score": final_score,
        "prediction": prediction
    }
