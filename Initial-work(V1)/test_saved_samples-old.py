from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

MODEL_PATH = "sign_lstm_model_best.pt"
LABEL_PATH = "label_names.npy"
DATASET_DIR = Path("dataset")
SEQUENCE_LENGTH = 45
FEATURE_DIM = 126

label_names = np.load(LABEL_PATH, allow_pickle=True)
num_classes = len(label_names)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc1 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(FEATURE_DIM, 64, num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

correct = 0
total = 0

for word_dir in sorted(DATASET_DIR.iterdir()):
    if not word_dir.is_dir():
        continue

    true_label = word_dir.name

    for file in sorted(word_dir.glob("*.npy")):
        arr = np.load(file).astype(np.float32)

        if arr.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            continue

        x = torch.tensor(arr).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_label = str(label_names[pred_idx])
            conf = float(probs[pred_idx])

        ok = pred_label == true_label
        if ok:
            correct += 1
        total += 1

        print(f"True: {true_label:10s} | Pred: {pred_label:10s} | Conf: {conf:.3f} | {'OK' if ok else 'WRONG'}")

print(f"\nAccuracy on saved dataset: {correct}/{total} = {correct/total:.3f}")