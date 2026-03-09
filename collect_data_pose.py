from pathlib import Path
from collections import deque
import numpy as np
import cv2
import mediapipe as mp

# ========= SETTINGS =========
DATASET_DIR = Path("dataset_hands_pose")
SEQUENCE_LENGTH = 45
CAMERA_INDEX = 0
COUNTDOWN_SECONDS = 2

LABEL_KEYS = {
    ord("f"): "พ่อ",
    ord("m"): "แม่",
    ord("p"): "พี่",
    ord("n"): "น้อง",
    ord("e"): "กิน",
    ord("d"): "ดื่ม",
    ord("h"): "หิว",
    ord("s"): "ง่วง",
    ord("y"): "ใช่",
    ord("x"): "ไม่ใช่",
    ord("l"): "นอน",
    ord("z"): "idle",
}
# ============================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

DATASET_DIR.mkdir(exist_ok=True)

for label in LABEL_KEYS.values():
    (DATASET_DIR / label).mkdir(parents=True, exist_ok=True)


def get_next_sample_id(word_dir: Path) -> int:
    existing = []
    for f in word_dir.glob("*.npy"):
        try:
            existing.append(int(f.stem))
        except ValueError:
            pass
    return max(existing, default=0) + 1


def extract_keypoints(results):
    # Pose: 33 landmarks * (x, y, z, visibility) = 132
    pose = np.zeros(33 * 4, dtype=np.float32)
    if results.pose_landmarks:
        pose_points = []
        for lm in results.pose_landmarks.landmark:
            pose_points.extend([lm.x, lm.y, lm.z, lm.visibility])
        pose = np.array(pose_points, dtype=np.float32)

    # Left hand: 21 landmarks * (x, y, z) = 63
    left_hand = np.zeros(21 * 3, dtype=np.float32)
    if results.left_hand_landmarks:
        left_points = []
        for lm in results.left_hand_landmarks.landmark:
            left_points.extend([lm.x, lm.y, lm.z])
        left_hand = np.array(left_points, dtype=np.float32)

    # Right hand: 21 landmarks * (x, y, z) = 63
    right_hand = np.zeros(21 * 3, dtype=np.float32)
    if results.right_hand_landmarks:
        right_points = []
        for lm in results.right_hand_landmarks.landmark:
            right_points.extend([lm.x, lm.y, lm.z])
        right_hand = np.array(right_points, dtype=np.float32)

    return np.concatenate([pose, left_hand, right_hand])  # total = 258


cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

current_label = None
sequence = deque(maxlen=SEQUENCE_LENGTH)
recording = False
countdown_active = False
countdown_start_time = 0

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    print("Controls:")
    print("Press a label key to choose a word")
    print("Press R to start recording")
    print("Press Q to quit")

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

        # UI text
        cv2.putText(frame, f"Label: {current_label}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Rec: {recording}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Countdown logic
        if countdown_active:
            elapsed = (cv2.getTickCount() - countdown_start_time) / cv2.getTickFrequency()
            remaining = COUNTDOWN_SECONDS - int(elapsed)

            cv2.putText(frame, f"Starting in: {remaining}", (420, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)

            if elapsed >= COUNTDOWN_SECONDS:
                countdown_active = False
                recording = True
                print("Recording started!")

        # Recording logic
        if recording:
            keypoints = extract_keypoints(results)
            if keypoints.shape[0] == 258:
                sequence.append(keypoints)

            if len(sequence) == SEQUENCE_LENGTH:
                sample = np.array(sequence, dtype=np.float32)
                word_dir = DATASET_DIR / current_label
                sample_id = get_next_sample_id(word_dir)
                save_path = word_dir / f"{sample_id}.npy"
                np.save(save_path, sample)
                print(f"Saved: {save_path} shape={sample.shape}")
                recording = False
                sequence.clear()

        cv2.imshow("Collect Data Hands+Pose", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key in LABEL_KEYS:
            current_label = LABEL_KEYS[key]
            sequence.clear()
            recording = False
            countdown_active = False
            print(f"Selected label: {current_label}")

        elif key == ord("r"):
            if current_label is None:
                print("Choose a label first.")
            else:
                countdown_active = True
                countdown_start_time = cv2.getTickCount()
                sequence.clear()
                recording = False
                print(f"Countdown started for: {current_label}")

cap.release()
cv2.destroyAllWindows()