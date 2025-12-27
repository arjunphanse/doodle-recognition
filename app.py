import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
CLASSES = [
    "The Eiffel Tower", "The Great Wall of China", "The Mona Lisa",
    "aircraft carrier", "airplane", "alarm clock", "ambulance", "angel",
    "animal migration", "ant", "anvil", "apple", "arm", "asparagus", "axe",
    "backpack", "banana", "bandage", "barn", "baseball bat", "baseball",
    "basket", "basketball", "bat", "bathtub", "beach", "bear", "beard", "bed",
    "bee", "belt", "bench", "bicycle", "binoculars", "bird", "birthday cake",
    "blackberry", "blueberry", "book", "boomerang", "bottlecap", "bowtie",
    "bracelet", "brain", "bread", "bridge", "broccoli", "broom", "bucket",
    "bulldozer",
]


class ConvNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


def build_resnet(num_classes: int):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_convnet():
    model_path = WEIGHTS_DIR / "quickdraw_cnn.pt"
    model = ConvNet(num_classes=len(CLASSES))
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def load_resnet():
    model_path = WEIGHTS_DIR / "quickdraw_resnet18.pt"
    model = build_resnet(num_classes=len(CLASSES))
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def load_dinov2():
    model_path = WEIGHTS_DIR / "quickdraw_dinov2_head.pt"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    in_dim = getattr(model, "embed_dim", None)
    if in_dim is None and hasattr(model, "head") and hasattr(model.head, "weight"):
        in_dim = model.head.weight.shape[1]
    if in_dim is None:
        raise RuntimeError("Could not determine DINOv2 feature dimension")
    model.head = nn.Linear(in_dim, len(CLASSES))
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


convnet_model = load_convnet()
resnet_model = load_resnet()
dinov2_model = load_dinov2()

dinov2_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", classes=CLASSES)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "pixels" not in data:
        return jsonify({"error": "No pixel data provided"}), 400
    pixels = data["pixels"]
    if not isinstance(pixels, list) or len(pixels) != 28 * 28:
        return jsonify({"error": "Expected 784-length list"}), 400

    arr_raw = np.array(pixels, dtype=np.float32).reshape(1, 1, 28, 28)
    # Invert to match training polarity and then normalize
    arr = 255.0 - arr_raw
    arr = arr / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr)

    # Prepare DINOv2 input
    dino_img = (255.0 - arr_raw).astype(np.uint8).reshape(28, 28)
    dino_tensor = dinov2_transform(dino_img).unsqueeze(0)

    with torch.no_grad():
        conv_logits = convnet_model(tensor)
        res_logits = resnet_model(tensor)
        dino_logits = dinov2_model(dino_tensor)
        conv_probs = torch.softmax(conv_logits, dim=1).squeeze(0).numpy().tolist()
        res_probs = torch.softmax(res_logits, dim=1).squeeze(0).numpy().tolist()
        dino_probs = torch.softmax(dino_logits, dim=1).squeeze(0).numpy().tolist()

    conv_results = sorted(
        [{"class": cls, "prob": float(p)} for cls, p in zip(CLASSES, conv_probs)],
        key=lambda x: x["prob"],
        reverse=True,
    )
    res_results = sorted(
        [{"class": cls, "prob": float(p)} for cls, p in zip(CLASSES, res_probs)],
        key=lambda x: x["prob"],
        reverse=True,
    )
    dino_results = sorted(
        [{"class": cls, "prob": float(p)} for cls, p in zip(CLASSES, dino_probs)],
        key=lambda x: x["prob"],
        reverse=True,
    )
    return jsonify({"convnet": conv_results, "resnet": res_results, "dinov2": dino_results})


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=False)
