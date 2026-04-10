import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import models
import random

# =============================
# CONFIG
# =============================

MODEL_PATH = "video_model_balanced.pth"

REAL_FOLDER = r"D:\truthseeker\data\raw\lavdf\real"
FAKE_FOLDER = r"D:\truthseeker\data\raw\lavdf\fake"

FRAMES_PER_VIDEO = 10
THRESHOLD = 0.45

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# =============================
# LOAD MODEL
# =============================

model = models.resnet18()

model.fc = torch.nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model = model.to(DEVICE)

model.eval()

# =============================
# VIDEO PROCESSING
# =============================

def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return None

    indices = np.linspace(0, total_frames - 1, FRAMES_PER_VIDEO).astype(int)

    frames = []

    for idx in indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.resize(frame, (224, 224))

        frame = frame.astype(np.float32) / 255.0
        frame = (frame - 0.5) / 0.5

        frame = np.transpose(frame, (2, 0, 1))

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    frames = np.stack(frames)

    tensor = torch.from_numpy(frames).float().to(DEVICE)

    with torch.no_grad():

        outputs = model(tensor)

        probs = F.softmax(outputs, dim=1)

        scores = probs[:, 1].cpu().numpy()

    score = 0.6 * np.mean(scores) + 0.4 * np.max(scores)

    return score

# =============================
# MAIN EVALUATION
# =============================

def main():

    tasks = []

    for v in os.listdir(REAL_FOLDER):
        tasks.append((os.path.join(REAL_FOLDER, v), 0))

    for v in os.listdir(FAKE_FOLDER):
        tasks.append((os.path.join(FAKE_FOLDER, v), 1))

    print("Total videos:", len(tasks))

    random.seed(42)
    tasks = random.sample(tasks, 500)

    TP = TN = FP = FN = 0

    for i, (video_path, label) in enumerate(tasks):

        score = process_video(video_path)

        if score is None:
            continue

        prediction = 1 if score >= THRESHOLD else 0

        if label == 1 and prediction == 1:
            TP += 1
        elif label == 0 and prediction == 0:
            TN += 1
        elif label == 0 and prediction == 1:
            FP += 1
        elif label == 1 and prediction == 0:
            FN += 1

        if i % 50 == 0:
            print("Processed:", i)

    total = TP + TN + FP + FN

    accuracy = (TP + TN) / total * 100

    print("\n==========================")
    print("TP (Fake detected):", TP)
    print("TN (Real detected):", TN)
    print("FP (Real → Fake):", FP)
    print("FN (Fake → Real):", FN)
    print("--------------------------")
    print("Accuracy: {:.2f}%".format(accuracy))
    print("==========================")

# =============================
# RUN
# =============================

if __name__ == "__main__":
    main()