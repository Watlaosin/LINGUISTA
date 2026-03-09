from collections import deque
import numpy as np
import cv2
import mediapipe as mp
import torch
import torch.nn as nn

# ========= SETTINGS =========
MODEL_PATH = "best_model.pt"
LABEL_PATH = "label_names.npy"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.5
SEQUENCE_LENGTH = 45
# ============================

# ===== DISPLAY LABEL MAP =====
thai_to_eng = {
    "พ่อ": "Father",
    "แม่": "Mother",
    "พี่": "OlderSibling",
    "น้อง": "YoungerSibling",
    "กิน": "Eat",
    "ดื่ม": "Drink",
    "หิว": "Hungry",
    "ง่วง": "Sleepy",
    "ใช่": "Yes",
    "ไม่ใช่": "No",
    "นอน": "Sleep"
}
# =============================


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


def extract_keypoints(results):
    # Pose: 33 * (x, y, z, visibility) = 132
    pose = np.zeros(33 * 4, dtype=np.float32)
    if results.pose_landmarks:
        pose_points = []
        for lm in results.pose_landmarks.landmark:
            pose_points.extend([lm.x, lm.y, lm.z, lm.visibility])
        pose = np.array(pose_points, dtype=np.float32)

    # Left hand: 21 * 3 = 63
    left_hand = np.zeros(21 * 3, dtype=np.float32)
    if results.left_hand_landmarks:
        left_points = []
        for lm in results.left_hand_landmarks.landmark:
            left_points.extend([lm.x, lm.y, lm.z])
        left_hand = np.array(left_points, dtype=np.float32)

    # Right hand: 21 * 3 = 63
    right_hand = np.zeros(21 * 3, dtype=np.float32)
    if results.right_hand_landmarks:
        right_points = []
        for lm in results.right_hand_landmarks.landmark:
            right_points.extend([lm.x, lm.y, lm.z])
        right_hand = np.array(right_points, dtype=np.float32)

    return np.concatenate([pose, left_hand, right_hand])  # total = 258


def format_top_k(probs, label_names, k=3):
    top_indices = np.argsort(probs)[::-1][:k]
    return " | ".join([
        f"{thai_to_eng.get(str(label_names[i]), str(label_names[i]))}={probs[i]:.2f}"
        for i in top_indices
    ])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    label_names = np.load(LABEL_PATH, allow_pickle=True)

    feature_dim = checkpoint["feature_dim"]
    hidden_size = checkpoint["hidden_size"]
    num_layers = checkpoint["num_layers"]
    dropout = checkpoint["dropout"]
    saved_seq_len = checkpoint["sequence_length"]

    if saved_seq_len != SEQUENCE_LENGTH:
        print(f"Warning: script SEQUENCE_LENGTH={SEQUENCE_LENGTH}, model expects {saved_seq_len}")

    model = SignLSTM(
        input_size=feature_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=len(label_names),
        dropout=dropout
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    sequence = deque(maxlen=saved_seq_len)
    prediction_history = deque(maxlen=10)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            # Draw pose
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # Draw left hand
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            # Draw right hand
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            keypoints = extract_keypoints(results)

            if keypoints.shape[0] == feature_dim:
                sequence.append(keypoints)

            predicted_text = "Collecting..."
            confidence_text = ""
            top3_text = ""

            if len(sequence) == saved_seq_len:
                x = np.array(sequence, dtype=np.float32)
                x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(x)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]

                pred_idx = int(np.argmax(probs))
                pred_label = str(label_names[pred_idx])
                conf = float(probs[pred_idx])

                prediction_history.append(pred_label)
                stable_pred = max(set(prediction_history), key=prediction_history.count)
                stable_pred_display = thai_to_eng.get(stable_pred, stable_pred)

                if conf >= CONFIDENCE_THRESHOLD:
                    predicted_text = stable_pred_display
                else:
                    predicted_text = "Uncertain"

                confidence_text = f"Confidence: {conf:.3f}"
                top3_text = format_top_k(probs, label_names, k=3)

            cv2.putText(frame, f"Frames: {len(sequence)}/{saved_seq_len}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"Prediction: {predicted_text}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.putText(frame, confidence_text, (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, top3_text, (20, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.imshow("Live Sign Prediction", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c"):
                sequence.clear()
                prediction_history.clear()
                print("Cleared sequence.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()