import os
import torch
import librosa
import numpy as np
import torch.nn.functional as F

from app.models.audio_model import audio_model, DEVICE

def run_audio_inference(video_path):

    audio_path = "temp/temp_audio.wav"

    # 🔥 FIX: prevent blocking
    if not os.path.exists(audio_path):
        print("⚠️ Audio file missing — skipping audio")
        return 0.5

    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        if len(y) < 16000:
            return 0.5

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=16000,
            n_mels=128
        )

        mel = librosa.power_to_db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        mel = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            output = audio_model(mel)
            probs = F.softmax(output, dim=1)
            score = probs[:, 1].item()

        return score

    except Exception as e:
        print("Audio processing error:", e)
        return 0.5