import os
import cv2
import json
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.amp import autocast, GradScaler

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

torch.backends.cudnn.benchmark = True

# =========================
# CONFIG
# =========================
NUM_FRAMES = 3
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4

DATA_ROOT = os.path.join(BASE_DIR, "data/raw/lav_df")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "dev")
METADATA_PATH = os.path.join(DATA_ROOT, "metadata.json")

WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
BEST_MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_video_model.pth")

os.makedirs(WEIGHTS_DIR, exist_ok=True)

# =========================
# TRANSFORMS
# =========================
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =========================
# DATASET
# =========================
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform

    def sample_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            return []

        indices = np.linspace(0, total_frames - 1, NUM_FRAMES).astype(int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()
        return frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        frames = self.sample_frames(self.video_paths[idx])

        if len(frames) == 0:
            frames = [np.zeros((224,224,3), dtype=np.uint8)] * NUM_FRAMES

        processed = [self.transform(f) for f in frames]
        return torch.stack(processed), torch.tensor(self.labels[idx], dtype=torch.long)

# =========================
# LOAD DATA
# =========================
def load_metadata():
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    print("Sample:", metadata[0])
    return metadata

def get_paths(folder, metadata, split_name):
    paths, labels = [], []

    for item in metadata:
        if item["split"] != split_name:
            continue

        video_name = os.path.basename(item["file"])
        path = os.path.join(folder, video_name)

        if not os.path.exists(path):
            continue

        label = 0 if not item["modify_video"] and not item["modify_audio"] else 1

        paths.append(path)
        labels.append(label)

    print(f"{split_name}: {len(paths)} samples")
    return paths, labels

def load_data():
    metadata = load_metadata()

    train_paths, train_labels = get_paths(TRAIN_DIR, metadata, "train")
    val_paths, val_labels = get_paths(VAL_DIR, metadata, "dev")

    train_loader = DataLoader(
        VideoDataset(train_paths, train_labels, train_transforms),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        VideoDataset(val_paths, val_labels, val_transforms),
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader

# =========================
# MODEL
# =========================
def build_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.layer1.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2)
    )

    return model.to(DEVICE)

# =========================
# TRAIN
# =========================
scaler = GradScaler("cuda")

def train_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    total_batches = len(loader)

    for i, (frames, labels) in enumerate(loader):
        frames = frames.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast("cuda"):
            loss = 0
            for j in range(frames.shape[1]):
                loss += criterion(model(frames[:, j]), labels)
            loss /= frames.shape[1]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if i % 50 == 0:
            print(f"Epoch {epoch} | Batch {i}/{total_batches} | Loss {loss:.4f}")

    return total_loss / total_batches

# =========================
# VALIDATION
# =========================
def validate(model, loader):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = torch.stack([model(frames[:, i]) for i in range(frames.shape[1])]).mean(0)
            preds = torch.argmax(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# =========================
# MAIN (RESUME FROM BEST MODEL)
# =========================
def main():
    train_loader, val_loader = load_data()
    model = build_model()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 🔥 RESUME FROM BEST MODEL
    start_epoch = 1
    best_acc = 0

    if os.path.exists(BEST_MODEL_PATH):
        print("🔁 Loading best model and starting from Epoch 2...")
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        start_epoch = 2

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\n🔥 Epoch {epoch}/{EPOCHS}")

        loss = train_epoch(model, train_loader, optimizer, criterion, epoch)
        acc = validate(model, val_loader)

        scheduler.step()

        print(f"\n✅ Epoch {epoch} Done | Loss={loss:.4f} | Val Acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("💾 Best model saved!")

    print("\n🚀 Training Complete | Best Acc:", best_acc)

if __name__ == "__main__":
    main()