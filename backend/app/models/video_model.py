import torch
import torch.nn as nn
import torchvision.models as models
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# 🔥 DIRECT MODEL (NO WRAPPER)
video_model = models.resnet18(weights=None)

video_model.fc = nn.Sequential(
    nn.Linear(video_model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 2)
)

video_model = video_model.to(DEVICE)

# 🔥 LOAD TRAINED MODEL
WEIGHTS_PATH = "D:/truthseeker/weights/best_video_model.pth"

if os.path.exists(WEIGHTS_PATH):
    video_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    print("🔥 Trained model loaded successfully!")
else:
    print("⚠️ No trained model found")

video_model.eval()