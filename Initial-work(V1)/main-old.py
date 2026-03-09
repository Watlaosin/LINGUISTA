import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

SEQUENCE_LENGTH = 45
COUNTDOWN_SECONDS = 3
DATASET_DIR = Path("dataset")

#Basic setup 10 words?
LABEL_KEYS = {
    ord("h"): "หิว",
    ord("t"): "ง่วง",
    ord("n"): "ไม่ใช่",
    ord("y"): "ใช่",
    ord("g"): "กิน"
}

def extract_frame_features(result) -> list[float]:
    left_hand = [0.0] * 63
    right_hand = [0.0] * 63

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(
            result.multi_hand_landmarks,
            result.multi_handedness
        ):
            hand_label = handedness.classification[0].label
            landmark_list = []

            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)
                landmark_list.append(lm.z)

            if hand_label == "Left":
                left_hand = landmark_list
            else:
                right_hand = landmark_list

    return left_hand + right_hand


def get_next_sample_path(label: str) -> Path:
    label_dir = DATASET_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(label_dir.glob("sample_*.npy"))
    next_index = len(existing) + 1
    return label_dir / f"sample_{next_index:03d}.npy"


# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Camera opened?", cap.isOpened())

state = "idle"   # idle, countdown, recording
current_label = None
current_sequence = []
countdown_start_time = None

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Draw landmarks
        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks,
                result.multi_handedness
            ):
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                hand_label = handedness.classification[0].label
                h, w, _ = frame.shape
                wrist = hand_landmarks.landmark[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)

                cv2.putText(
                    frame,
                    hand_label,
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        frame_features = extract_frame_features(result)

        # ----------------------------
        # State machine
        # ----------------------------
        if state == "idle":
            cv2.putText(
                frame,
                "Press Q=quit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

        elif state == "countdown":
            elapsed = time.time() - countdown_start_time
            remaining = COUNTDOWN_SECONDS - int(elapsed)

            if elapsed >= COUNTDOWN_SECONDS:
                state = "recording"
                current_sequence = []
                print(f"Started recording: {current_label}")
            else:
                cv2.putText(
                    frame,
                    f"Get ready: {remaining}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 255),
                    3
                )

        elif state == "recording":
            current_sequence.append(frame_features)

            cv2.putText(
                frame,
                f"Recording {current_label}: {len(current_sequence)}/{SEQUENCE_LENGTH}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            if len(current_sequence) == SEQUENCE_LENGTH:
                save_path = get_next_sample_path(current_label)
                np.save(save_path, np.array(current_sequence, dtype=np.float32))
                print(f"Saved: {save_path}")

                state = "idle"
                current_label = None
                current_sequence = []

        cv2.imshow("Sign Dataset Recorder", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if state == "idle" and key in LABEL_KEYS:
            current_label = LABEL_KEYS[key]
            countdown_start_time = time.time()
            state = "countdown"
            print(f"Get ready for: {current_label}")

cap.release()
cv2.destroyAllWindows()