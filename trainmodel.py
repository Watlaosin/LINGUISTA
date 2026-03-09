from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ========= SETTINGS =========
DATASET_DIR = Path("dataset_hands_pose")
MODEL_SAVE_PATH = "best_model.pt"
LABELS_SAVE_PATH = "label_names.npy"

SEQUENCE_LENGTH = 45

BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

TRAIN_RATIO = 0.8
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ============================


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_feature_dim(dataset_dir: Path, sequence_length: int) -> int:
    for label_dir in dataset_dir.iterdir():
        if label_dir.is_dir():
            for npy_file in label_dir.glob("*.npy"):
                arr = np.load(npy_file)
                if arr.ndim == 2 and arr.shape[0] == sequence_length:
                    return arr.shape[1]
                elif arr.ndim == 1 and arr.size % sequence_length == 0:
                    return arr.size // sequence_length
    raise ValueError("Could not detect feature dimension from dataset.")


class SignSequenceDataset(Dataset):
    def __init__(self, dataset_dir: Path, sequence_length: int, feature_dim: int):
        self.dataset_dir = dataset_dir
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim

        self.label_names = sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])
        if not self.label_names:
            raise ValueError(f"No label folders found in {dataset_dir}")

        self.label_to_idx = {label: i for i, label in enumerate(self.label_names)}
        self.samples = []

        for label in self.label_names:
            label_dir = dataset_dir / label
            for npy_file in sorted(label_dir.glob("*.npy")):
                self.samples.append((npy_file, self.label_to_idx[label]))

        if not self.samples:
            raise ValueError(f"No .npy files found under {dataset_dir}")

    def __len__(self):
        return len(self.samples)

    def _fix_sequence_shape(self, arr: np.ndarray) -> np.ndarray:
        arr = np.array(arr, dtype=np.float32)

        if arr.ndim == 2 and arr.shape == (self.sequence_length, self.feature_dim):
            return arr

        if arr.ndim == 1:
            expected = self.sequence_length * self.feature_dim
            if arr.size == expected:
                return arr.reshape(self.sequence_length, self.feature_dim)

        if arr.ndim == 2 and arr.shape[1] == self.feature_dim:
            frames = arr.shape[0]
            if frames > self.sequence_length:
                return arr[:self.sequence_length]
            elif frames < self.sequence_length:
                pad = np.zeros((self.sequence_length - frames, self.feature_dim), dtype=np.float32)
                return np.vstack([arr, pad])

        raise ValueError(
            f"Unexpected sample shape {arr.shape}. "
            f"Expected ({self.sequence_length}, {self.feature_dim})"
        )

    def __getitem__(self, idx):
        file_path, label_idx = self.samples[idx]
        arr = np.load(file_path)
        arr = self._fix_sequence_shape(arr)

        x = torch.tensor(arr, dtype=torch.float32)
        y = torch.tensor(label_idx, dtype=torch.long)
        return x, y


class SignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_timestep = out[:, -1, :]
        return self.classifier(last_timestep)


def calculate_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += calculate_accuracy(logits, y)

    return total_loss / len(loader), total_acc / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        total_acc += calculate_accuracy(logits, y)

    return total_loss / len(loader), total_acc / len(loader)


def main():
    set_seed(SEED)

    print(f"Using device: {DEVICE}")
    print(f"Loading dataset from: {DATASET_DIR.resolve()}")

    feature_dim = detect_feature_dim(DATASET_DIR, SEQUENCE_LENGTH)
    print(f"Detected feature dimension: {feature_dim}")

    dataset = SignSequenceDataset(
        dataset_dir=DATASET_DIR,
        sequence_length=SEQUENCE_LENGTH,
        feature_dim=feature_dim
    )

    np.save(LABELS_SAVE_PATH, np.array(dataset.label_names))
    print(f"Labels: {dataset.label_names}")
    print(f"Saved labels to: {LABELS_SAVE_PATH}")
    print(f"Total samples: {len(dataset)}")

    train_size = int(len(dataset) * TRAIN_RATIO)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SignLSTM(
        input_size=feature_dim,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=len(dataset.label_names),
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_names": dataset.label_names,
                    "sequence_length": SEQUENCE_LENGTH,
                    "feature_dim": feature_dim,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                },
                MODEL_SAVE_PATH
            )
            print(f"Saved best model to {MODEL_SAVE_PATH}")

    print(f"\nTraining finished. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()