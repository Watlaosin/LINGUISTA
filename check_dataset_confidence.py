from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

MODEL_PATH = "best_model.pt"
LABEL_PATH = "label_names.npy"
DATASET_DIR = Path("dataset_hands_pose")


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


def format_top_k(probs, label_names, k=3):
    top_indices = np.argsort(probs)[::-1][:k]
    parts = []
    for idx in top_indices:
        parts.append(f"{label_names[idx]}={probs[idx]:.3f}")
    return " | ".join(parts)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    label_names = np.load(LABEL_PATH, allow_pickle=True)

    sequence_length = checkpoint["sequence_length"]
    feature_dim = checkpoint["feature_dim"]
    hidden_size = checkpoint["hidden_size"]
    num_layers = checkpoint["num_layers"]
    dropout = checkpoint["dropout"]
    num_classes = len(label_names)

    print(f"Device: {device}")
    print(f"Sequence length: {sequence_length}")
    print(f"Feature dim: {feature_dim}")
    print(f"Labels: {list(label_names)}")
    print()

    model = SignLSTM(
        input_size=feature_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    correct = 0
    total = 0

    for word_dir in sorted(DATASET_DIR.iterdir()):
        if not word_dir.is_dir():
            continue

        true_label = word_dir.name

        for file in sorted(word_dir.glob("*.npy")):
            arr = np.load(file).astype(np.float32)

            if arr.shape != (sequence_length, feature_dim):
                print(f"SKIP shape mismatch: {file} -> {arr.shape}")
                continue

            x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)

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

            top3 = format_top_k(probs, label_names, k=3)

            print(
                f"True: {true_label:10s} | "
                f"Pred: {pred_label:10s} | "
                f"Conf: {conf:.3f} | "
                f"{'OK' if ok else 'WRONG'} | "
                f"Top3: {top3}"
            )

    if total == 0:
        print("No valid samples found.")
    else:
        print(f"\nAccuracy on saved dataset: {correct}/{total} = {correct/total:.3f}")


if __name__ == "__main__":
    main()