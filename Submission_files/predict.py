import os
import csv
import random

TEST_DIR = "/data/test"
OUT_FILE = "/output/predictions.csv"
SEED = 42  # fixed seed for reproducibility


def main():
    os.makedirs("/output", exist_ok=True)

    # Collect samples (files only)
    samples = sorted([
        f for f in os.listdir(TEST_DIR)
        if os.path.isfile(os.path.join(TEST_DIR, f))
    ])

    # Seed the random number generator
    random.seed(SEED)

    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "prediction"])

        for fname in samples:
            sample_id = os.path.splitext(fname)[0]
            prediction = random.random()  # float between 0 and 1
            writer.writerow([sample_id, prediction])


if __name__ == "__main__":
    main()
