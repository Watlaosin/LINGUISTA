from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ========= SETTINGS =========
DATASET_DIR = Path("dataset")
SEQUENCE_LENGTH = 45
FEATURE_DIM = 126
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "sign_lstm_model_best.pt"
LABEL_SAVE_PATH = "label_names.npy"
# ============================

# ---------- Load Dataset ----------
X = []
y = []

for word_dir in sorted(DATASET_DIR.iterdir()):
    if not word_dir.is_dir():
        continue

    label = word_dir.name

    for file in sorted(word_dir.glob("*.npy")):
        arr = np.load(file)

        if arr.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            print(f"Skipping {file} because shape is {arr.shape}")
            continue

        X.append(arr)
        y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

if len(X) == 0:
    raise ValueError("No valid .npy files found in dataset folder.")

# ---------- Encode Labels ----------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print("Classes:", list(label_encoder.classes_))

# Save label names
np.save(LABEL_SAVE_PATH, label_encoder.classes_)
print(f"Saved label names to {LABEL_SAVE_PATH}")

# ---------- Train / Validation Split ----------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Train set:", X_train.shape, y_train.shape)
print("Val set:", X_val.shape, y_val.shape)

# ---------- Dataset Class ----------
class SignDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SignDataset(X_train, y_train)
val_dataset = SignDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------- LSTM Model ----------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)      # out shape: (batch, seq_len, hidden_dim)
        out = out[:, -1, :]        # take last timestep
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = LSTMModel(FEATURE_DIM, 64, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------- Training Loop ----------
best_val_acc = 0.0
best_epoch = 0

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()

    train_loss /= len(train_dataset)
    train_acc = train_correct / len(train_dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()

    val_loss /= len(val_dataset)
    val_acc = val_correct / len(val_dataset)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}"
    )

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"New best model saved at epoch {best_epoch} with Val Acc: {best_val_acc:.3f}")

print("\nTraining finished.")
print(f"Best epoch: {best_epoch}")
print(f"Best validation accuracy: {best_val_acc:.3f}")
print(f"Best model saved to: {MODEL_SAVE_PATH}")
print(f"Label names saved to: {LABEL_SAVE_PATH}")