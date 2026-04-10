import cv2
import torch
import numpy as np
import torch.nn.functional as F
from app.models.video_model import video_model, DEVICE

torch.backends.cudnn.benchmark = True

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def run_video_inference(video_path):

    cap = cv2.VideoCapture(video_path)

    scores = []
    frame_count = 0
    face_box = None

    max_frames = 300  # 🔥 prevent long processing

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_count > max_frames:
            print("⚠️ Max frame limit reached")
            break

        # DEBUG
        if frame_count % 50 == 0:
            print(f"Processing frame: {frame_count}")

        # process fewer frames (speed)
        if frame_count % 25 != 0:
            frame_count += 1
            continue

        # face detection occasionally
        if frame_count % 90 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5
            )

            if len(faces) > 0:
                face_box = faces[0]

        # crop face if exists
        if face_box is not None:
            x, y, w, h = face_box
            frame = frame[y:y+h, x:x+w]

        # resize
        frame = cv2.resize(frame, (224, 224))

        # normalize
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - 0.5) / 0.5

        # HWC → CHW
        frame = np.transpose(frame, (2, 0, 1))

        tensor = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = video_model(tensor)
            probs = F.softmax(output, dim=1)
            score = probs[:, 1].item()

        scores.append(score)
        frame_count += 1

    cap.release()

    if len(scores) == 0:
        return 0.0

    # 🔥 better aggregation
    return float(np.percentile(scores, 75))