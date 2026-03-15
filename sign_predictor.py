from collections import deque
import numpy as np
import cv2
import mediapipe as mp
import torch
import torch.nn as nn


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
    "นอน": "Sleep",
    "idle": "Idle",
}


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


class SignLanguagePredictor:
    def __init__(self, model_path="best_model.pt", label_path="label_names.npy", confidence_threshold=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold

        checkpoint = torch.load(model_path, map_location=self.device)
        self.label_names = np.load(label_path, allow_pickle=True)

        self.feature_dim = checkpoint["feature_dim"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.dropout = checkpoint["dropout"]
        self.sequence_length = checkpoint["sequence_length"]

        self.model = SignLSTM(
            input_size=self.feature_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=len(self.label_names),
            dropout=self.dropout
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.sequence = deque(maxlen=self.sequence_length)
        self.prediction_history = deque(maxlen=10)

        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_keypoints(self, results):
        pose = np.zeros(33 * 4, dtype=np.float32)
        if results.pose_landmarks:
            pose_points = []
            for lm in results.pose_landmarks.landmark:
                pose_points.extend([lm.x, lm.y, lm.z, lm.visibility])
            pose = np.array(pose_points, dtype=np.float32)

        left_hand = np.zeros(21 * 3, dtype=np.float32)
        if results.left_hand_landmarks:
            left_points = []
            for lm in results.left_hand_landmarks.landmark:
                left_points.extend([lm.x, lm.y, lm.z])
            left_hand = np.array(left_points, dtype=np.float32)

        right_hand = np.zeros(21 * 3, dtype=np.float32)
        if results.right_hand_landmarks:
            right_points = []
            for lm in results.right_hand_landmarks.landmark:
                right_points.extend([lm.x, lm.y, lm.z])
            right_hand = np.array(right_points, dtype=np.float32)

        return np.concatenate([pose, left_hand, right_hand])

    def format_top_k(self, probs, k=3):
        top_indices = np.argsort(probs)[::-1][:k]
        return [
            {
                "label": thai_to_eng.get(str(self.label_names[i]), str(self.label_names[i])),
                "confidence": float(probs[i])
            }
            for i in top_indices
        ]

    def process_frame(self, frame, draw_landmarks=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        if draw_landmarks and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        if draw_landmarks and results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

        if draw_landmarks and results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

        keypoints = self.extract_keypoints(results)

        if keypoints.shape[0] == self.feature_dim:
            self.sequence.append(keypoints)

        prediction = {
            "frames_collected": len(self.sequence),
            "frames_needed": self.sequence_length,
            "prediction": "Collecting...",
            "raw_label": None,
            "display_label": None,
            "confidence": 0.0,
            "is_confident": False,
            "top3": []
        }

        if len(self.sequence) == self.sequence_length:
            try:
                x = np.array(self.sequence, dtype=np.float32)
                x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.model(x)
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]

                pred_idx = int(np.argmax(probs))
                pred_label = str(self.label_names[pred_idx])
                conf = float(probs[pred_idx])

                self.prediction_history.append(pred_label)
                stable_pred = max(set(self.prediction_history), key=self.prediction_history.count)

                pred_display = thai_to_eng.get(stable_pred, stable_pred)
                is_confident = conf >= self.confidence_threshold

                prediction["raw_label"] = stable_pred
                prediction["display_label"] = pred_display
                prediction["prediction"] = pred_display if is_confident else f"Uncertain ({pred_display})"
                prediction["confidence"] = conf
                prediction["is_confident"] = is_confident
                prediction["top3"] = self.format_top_k(probs)

            except Exception as e:
                print("Prediction error:", repr(e))
                prediction["prediction"] = "ERROR"

        return frame, prediction

    def clear_sequence(self):
        self.sequence.clear()
        self.prediction_history.clear()

    def close(self):
        self.holistic.close()