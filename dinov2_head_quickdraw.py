# Train a DINOv2 classification head for 50-class QuickDraw doodles
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.transforms import InterpolationMode


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
FILENAMES = [name + ".npy" for name in CLASSES]


class QuickDrawDataset(torch.utils.data.Dataset):
    def __init__(self, root="data", samples_per_class=None, split="train", seed=1):
        assert split in {"train", "val", "test"}, "split must be 'train', 'val', or 'test'"

        images, labels = [], []
        root_path = Path(root)

        for label, fname in enumerate(FILENAMES):
            path = root_path / fname
            arr = np.load(path)  # shape (N, 784)
            if samples_per_class is not None:
                arr = arr[:samples_per_class]
            if arr.shape[0] == 0:
                continue

            rng = np.random.default_rng(seed + label)
            perm = rng.permutation(len(arr))
            n = len(arr)
            train_end = max(1, int(n * 0.8))
            val_end = min(n, train_end + max(1, int(n * 0.05)))
            if split == "train":
                idx = perm[:train_end]
            elif split == "val":
                idx = perm[train_end:val_end]
            else:
                idx = perm[val_end:]
            if idx.size == 0:
                continue

            arr = arr[idx].reshape(-1, 28, 28).astype("uint8")
            images.append(arr)
            labels.append(np.full(arr.shape[0], label, dtype=np.int64))

        if not images:
            raise RuntimeError("No images loaded. Check data directory and filenames.")

        self.images = np.concatenate(images, axis=0)
        self.labels = np.concatenate(labels, axis=0)

        # DINOv2 expects 3x224x224 with ImageNet normalization
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img = self.transform(img)
        return img, label


def build_model(num_classes, arch="dinov2_vits14"):
    # Loads a pretrained DINOv2 backbone and train a custom head
    model = torch.hub.load("facebookresearch/dinov2", arch)
    for p in model.parameters():
        p.requires_grad = False
    in_dim = getattr(model.head, "in_features", None) or getattr(model, "embed_dim", None)
    if in_dim is None:
        raise RuntimeError("Could not determine feature dimension from DINOv2 model.")
    model.head = nn.Linear(in_dim, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(enabled=device.type == "cuda"):
            logits = model(data)
            loss = criterion(logits, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * data.size(0)
        total += data.size(0)

    return running_loss / max(1, total)


def evaluate(model, loader, device, quiet=False, split_name="Val"):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0
    loss_sum = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = criterion(logits, target)
            loss_sum += loss.item() * data.size(0)
            total += data.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
    avg_loss = loss_sum / max(1, total)
    acc = 100.0 * correct / max(1, total)
    if not quiet:
        print(f"{split_name}: loss={avg_loss:.4f} acc={acc:.2f}%")
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser(description="QuickDraw 50-class head on DINOv2 ViT-S/14")
    parser.add_argument("--data-dir", type=str, default="data", help="directory with QuickDraw .npy files")
    parser.add_argument("--samples-per-class", type=int, default=None, help="limit samples per class")
    parser.add_argument("--batch-size", type=int, default=64, help="train batch size")
    parser.add_argument("--test-batch-size", type=int, default=256, help="eval batch size")
    parser.add_argument("--epochs", type=int, default=15, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for head")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--save-model", action="store_true", help="save best checkpoint")
    parser.add_argument("--arch", type=str, default="dinov2_vits14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
                        help="which DINOv2 backbone to use")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = QuickDrawDataset(root=args.data_dir, samples_per_class=args.samples_per_class,
                                split="train", seed=args.seed)
    val_ds = QuickDrawDataset(root=args.data_dir, samples_per_class=args.samples_per_class,
                              split="val", seed=args.seed)
    test_ds = QuickDrawDataset(root=args.data_dir, samples_per_class=args.samples_per_class,
                               split="test", seed=args.seed)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                               shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.test_batch_size,
                                             shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.test_batch_size,
                                              shuffle=False, num_workers=2, pin_memory=device.type == "cuda")

    model = build_model(num_classes=len(CLASSES), arch=args.arch).to(device)
    optimizer = optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        _, val_acc = evaluate(model, val_loader, device, quiet=True, split_name="Val")
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_acc:.2f}%")
        scheduler.step()
        if val_acc > best_val:
            best_val = val_acc
            best_state = model.state_dict()

    evaluate(model, test_loader, device, split_name="Test")

    if args.save_model and best_state is not None:
        torch.save(best_state, "quickdraw_dinov2_head.pt")


if __name__ == "__main__":
    main()
