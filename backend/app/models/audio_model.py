import torch
import torch.nn as nn
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

class AudioModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(32,2)

    def forward(self,x):
        x = self.net(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)


audio_model = AudioModel().to(DEVICE)

# 🔥 LOAD TRAINED AUDIO MODEL (IF EXISTS)
WEIGHTS_PATH = "D:/truthseeker/weights/best_audio_model.pth"

if os.path.exists(WEIGHTS_PATH):
    audio_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    print("🔊 Trained audio model loaded!")
else:
    print("⚠️ No trained audio model found — using default weights")

audio_model.eval()