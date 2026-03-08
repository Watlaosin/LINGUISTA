import sys
import cv2

print("RUNNING test_cam.py")
print("Python exe:", sys.executable)
print("OpenCV:", cv2.__version__)
print("-----")

backends = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("ANY",  cv2.CAP_ANY),
]

for name, backend in backends:
    print(f"\nBackend: {name}")
    for i in range(6):
        cap = cv2.VideoCapture(i, backend)
        opened = cap.isOpened()
        ok, frame = cap.read() if opened else (False, None)
        print(f"  index {i}: opened={opened}, read={ok}")
        cap.release()

print("\nDONE")