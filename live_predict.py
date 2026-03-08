from collections import deque
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
import torch
import torch.nn as nn

# ========= SETTINGS =========
MODEL_PATH = "sign_lstm_model_best.pt"
LABEL_PATH = "label_names.npy"
SEQUENCE_LENGTH = 45
FEATURE_DIM = 126
CONFIDENCE_THRESHOLD = 0.70
CAMERA_INDEX = 0
# ============================


# ---------- Load labels ----------
if not Path(LABEL_PATH).exists():
    raise FileNotFoundError(f"Label file not found: {LABEL_PATH}")

label_names = np.load(LABEL_PATH, allow_pickle=True)
num_classes = len(label_names)

print("Loaded labels:", list(label_names))


# ---------- Model definition ----------
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


# ---------- Load model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = LSTMModel(FEATURE_DIM, 64, num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print(f"Loaded model from: {MODEL_PATH}")
print(f"Using device: {device}")


# ---------- MediaPipe setup ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def extract_keypoints(results):
    """
    Returns a (126,) vector:
    63 for left hand + 63 for right hand
    Missing hand => zeros
    """
    left = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            coords = np.array(coords, dtype=np.float32)

            label = handedness.classification[0].label  # "Left" or "Right"
            # MediaPipe label is from the person's perspective in mirrored/selfie view logic.
            # Keep it consistent with training. If predictions seem swapped, reverse this logic.
            if label == "Left":
                left = coords
            else:
                right = coords

    return np.concatenate([left, right])


def draw_prediction_panel(frame, predicted_word, confidence, frames_collected):
    cv2.rectangle(frame, (10, 10), (500, 130), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Prediction: {predicted_word}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"Frames: {frames_collected}/{SEQUENCE_LENGTH}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


# ---------- Webcam ----------
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

sequence = deque(maxlen=SEQUENCE_LENGTH)
predicted_word = "..."
confidence = 0.0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    print("Press Q to quit.")
    print("Press C to clear current sequence.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Extract and append keypoints every frame
        keypoints = extract_keypoints(results)
        if keypoints.shape[0] == FEATURE_DIM:
            sequence.append(keypoints)

        # Predict when enough frames are collected
        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.array(sequence, dtype=np.float32)
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                pred_conf = float(probs[pred_idx])
                print("Raw prediction:", label_names[pred_idx], "Confidence:", round(pred_conf, 3))

            if pred_conf >= CONFIDENCE_THRESHOLD:
                predicted_word = str(label_names[pred_idx])
                confidence = pred_conf
            else:
                predicted_word = "Unsure"
                confidence = pred_conf

        draw_prediction_panel(frame, predicted_word, confidence, len(sequence))

        cv2.imshow("Live Sign Prediction", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("c"):
            sequence.clear()
            predicted_word = "..."
            confidence = 0.0
            print("Sequence cleared.")

cap.release()
cv2.destroyAllWindows()