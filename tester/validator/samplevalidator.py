from pathlib import Path
import numpy as np

DATASET_DIR = Path("dataset")
expected_shape = (45, 126)

total = 0
bad_files = []

for word_dir in DATASET_DIR.iterdir():
    if not word_dir.is_dir():
        continue

    count = 0
    for file in word_dir.glob("*.npy"):
        arr = np.load(file)
        if arr.shape != expected_shape:
            bad_files.append((str(file), arr.shape))
        count += 1
        total += 1

    print(f"{word_dir.name}: {count} samples")

print(f"\nTotal samples: {total}")

if bad_files:
    print("\nBad files found:")
    for path, shape in bad_files:
        print(path, shape)
else:
    print("\nAll files are correct yippie.")