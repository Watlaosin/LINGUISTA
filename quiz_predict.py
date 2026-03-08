from pathlib import Path
import time
import random
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
CAMERA_INDEX = 0
COUNTDOWN_SECONDS = 3
CONFIDENCE_THRESHOLD = 0.55
# ============================


# ---------- Load labels ----------
if not Path(LABEL_PATH).exists():
    raise FileNotFoundError(f"Label file not found: {LABEL_PATH}")

label_names = np.load(LABEL_PATH, allow_pickle=True)
label_names = list(label_names)
num_classes = len(label_names)

print("Loaded labels:", label_names)


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
    left = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)
    hand_detected = False

    if results.multi_hand_landmarks and results.multi_handedness:
        hand_detected = True
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            coords = np.array(coords, dtype=np.float32)

            label = handedness.classification[0].label
            if label == "Left":
                left = coords
            else:
                right = coords

    return np.concatenate([left, right]), hand_detected


def draw_text_block(frame, lines, start_x=20, start_y=35, line_gap=35):
    cv2.rectangle(frame, (10, 10), (760, 180), (0, 0, 0), -1)
    y = start_y
    for line in lines:
        cv2.putText(
            frame,
            line,
            (start_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        y += line_gap


def run_prediction(sequence_data):
    input_data = np.array(sequence_data, dtype=np.float32)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_conf = float(probs[pred_idx])

    predicted_word = str(label_names[pred_idx])
    return predicted_word, pred_conf, probs


# ---------- Quiz state ----------
target_word = random.choice(label_names)
message = "Press S to start attempt"
predicted_word = "..."
confidence = 0.0
result_text = "No attempt yet"


# ---------- Webcam ----------
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    print("Press S to start countdown")
    print("Press N for a new target word")
    print("Press Q to quit")

    state = "idle"
    countdown_start = None
    collected_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        keypoints, hand_detected = extract_keypoints(results)

        # ---------- State machine ----------
        if state == "idle":
            lines = [
                f"Target: {target_word}",
                message,
                f"Last prediction: {predicted_word} | Confidence: {confidence:.2f}",
                result_text
            ]

        elif state == "countdown":
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN_SECONDS - int(elapsed)

            if elapsed >= COUNTDOWN_SECONDS:
                state = "collecting"
                collected_frames = []
                message = "Recording 45 frames..."
            else:
                lines = [
                    f"Target: {target_word}",
                    f"Get ready... {remaining}",
                    "Hold your sign clearly in front of the camera",
                    "Press Q to quit"
                ]

        if state == "collecting":
            if hand_detected and keypoints.shape[0] == FEATURE_DIM:
                collected_frames.append(keypoints)

            if len(collected_frames) >= SEQUENCE_LENGTH:
                predicted_word, confidence, probs = run_prediction(collected_frames[:SEQUENCE_LENGTH])

                print("Target:", target_word)
                print("Predicted:", predicted_word)
                print("Confidence:", round(confidence, 3))

                if confidence < CONFIDENCE_THRESHOLD:
                    result_text = "Result: Unsure"
                elif predicted_word == target_word:
                    result_text = "Result: Correct!"
                else:
                    result_text = f"Result: Wrong (predicted {predicted_word})"

                message = "Press S to try again or N for new word"
                state = "idle"

            lines = [
                f"Target: {target_word}",
                f"Recording... {len(collected_frames)}/{SEQUENCE_LENGTH}",
                "Keep signing steadily",
                "Press Q to quit"
            ]

        draw_text_block(frame, lines)

        cv2.imshow("Sign Quiz Prediction", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n"):
            target_word = random.choice(label_names)
            predicted_word = "..."
            confidence = 0.0
            result_text = "No attempt yet"
            message = "Press S to start attempt"
            state = "idle"
        elif key == ord("s") and state == "idle":
            countdown_start = time.time()
            predicted_word = "..."
            confidence = 0.0
            result_text = "Waiting for result..."
            state = "countdown"

cap.release()
cv2.destroyAllWindows()