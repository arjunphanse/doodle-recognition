# QuickDraw training script using ResNet18
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import models


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


def build_model(num_classes: int):
    model = models.resnet18(weights=None)
    # Adapt to single-channel input
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


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

            arr = arr[idx].reshape(-1, 28, 28).astype("float32") / 255.0
            arr = (arr - 0.5) / 0.5  # normalize to mean 0, std ~1
            images.append(arr)
            labels.append(np.full(arr.shape[0], label, dtype=np.int64))

        if not images:
            raise RuntimeError("No images loaded. Check data directory and filenames.")

        imgs = np.concatenate(images, axis=0)
        lbls = np.concatenate(labels, axis=0)
        self.images = torch.from_numpy(imgs[:, None, :, :])  # add channel dim
        self.labels = torch.from_numpy(lbls)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    total_samples = 0
    scaler = torch.amp.GradScaler(enabled=not args.no_accel and torch.cuda.is_available())

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=not args.no_accel and torch.cuda.is_available()):
            output = model(data)
            loss = F.nll_loss(F.log_softmax(output, dim=1), target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * data.size(0)
        total_samples += data.size(0)
        if args.dry_run:
            break
    return running_loss / max(1, total_samples)


def evaluate(model, device, loader, split_name="Test", quiet=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)
    acc = 100. * correct / len(loader.dataset)
    if not quiet:
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            split_name, test_loss, correct, len(loader.dataset), acc))
    return test_loss, acc


def main():
    parser = argparse.ArgumentParser(description='QuickDraw ResNet18')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='directory with QuickDraw .npy files (default: data)')
    parser.add_argument('--samples-per-class', type=int, default=None,
                        help='limit number of samples per class (default: all)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 25)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables CUDA')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true',
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_accel and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        accel_kwargs = {'num_workers': 1,
                        'persistent_workers': True,
                        'pin_memory': True,
                        'shuffle': True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    train_dataset = QuickDrawDataset(root=args.data_dir,
                                     samples_per_class=args.samples_per_class,
                                     split="train",
                                     seed=args.seed)
    val_dataset = QuickDrawDataset(root=args.data_dir,
                                   samples_per_class=args.samples_per_class,
                                   split="val",
                                   seed=args.seed)
    test_dataset = QuickDrawDataset(root=args.data_dir,
                                    samples_per_class=args.samples_per_class,
                                    split="test",
                                    seed=args.seed)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = build_model(num_classes=len(CLASSES)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        _, val_acc = evaluate(model, device, val_loader, split_name="Validation", quiet=True)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_acc:.2f}%")
        scheduler.step()

    evaluate(model, device, test_loader, split_name="Test")

    if args.save_model:
        torch.save(model.state_dict(), "quickdraw_resnet18.pt")


if __name__ == '__main__':
    main()
