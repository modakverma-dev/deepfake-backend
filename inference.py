import torch
import cv2
import os
import tempfile
from torch import nn
from torchvision import models, transforms
# import urllib.request
import requests

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(
        self,
        num_classes,
        latent_dim=2048,
        lstm_layers=1,
        hidden_dim=2048,
        bidirectional=False
    ):
        super(Model, self).__init__()

        backbone = models.resnext50_32x4d(pretrained=False)
        self.model = nn.Sequential(*list(backbone.children())[:-2])

        self.lstm = nn.LSTM(
            latent_dim,
            hidden_dim,
            lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
            bias=False 
        )

        self.relu = nn.LeakyReLU()
        self.dp1 = nn.Dropout(0.5)
        self.dp2 = nn.Dropout(0.5)

        self.linear1 = nn.Linear(hidden_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(b, t, 2048)

        x = self.dp1(x)
        x_lstm, _ = self.lstm(x)
        x_lstm = self.dp2(x_lstm)

        logits = self.linear1(torch.mean(x_lstm, dim=1))
        return fmap, logits


MODEL_PATH = "model/best_model.pt"
MODEL_URL = "https://huggingface.co/maddy08/deepfake-video-detection/resolve/main/best_model.pt"

def download_model():
    os.makedirs("model", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model...")
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("✅ Model downloaded")

def load_model():
    if not os.path.exists(MODEL_PATH):
        download_model()

    model = Model(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

# model = Model(num_classes=2)
# state_dict = torch.load(MODEL_PATH, map_location=device)
# model.load_state_dict(state_dict)
# model.to(device)
# model.eval()


IM_SIZE = 250
MAX_FRAMES = 30

frame_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_frames(video_path, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames


def predict_video(video_bytes: bytes, model):
    # Save temp video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        video_path = tmp.name

    frames = extract_frames(video_path)
    os.remove(video_path)

    if len(frames) == 0:
        raise ValueError("No frames extracted from video")

    processed = [frame_transform(f) for f in frames]
    video_tensor = torch.stack(processed).unsqueeze(0).to(device)

    with torch.no_grad():
        _, logits = model(video_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        print("pred_class",pred_class)
        confidence = probs[0, pred_class].item()
        
    CLASS_NAMES = {
        0: "REAL",
        1: "FAKE"
    }

    return {
        "prediction": CLASS_NAMES[pred_class],
        "confidence": float(confidence),
        "raw_class": int(pred_class)
    }

