from pathlib import Path
import csv
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import (
    SISFALL_ROOT,
    find_recordings,
    get_subjects,
    split_subjects,
    recordings_for_subjects,
    load_recording,
    create_windows,
)


WINDOW_SIZE = 400
STRIDE = 200

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"

LABEL_TO_INT = {
    "normal": 0,
    "fall": 1,
}


def calculate_training_statistics(recordings):
    """Calculate per-channel mean/std using TRAINING recordings only."""
    channel_sum = np.zeros(6, dtype=np.float64)
    channel_sum_sq = np.zeros(6, dtype=np.float64)
    count = 0

    print("Calculating normalization statistics from TRAINING data...")

    for index, path in enumerate(recordings, start=1):
        data, _ = load_recording(path)

        channel_sum += data.sum(axis=0, dtype=np.float64)
        channel_sum_sq += np.square(data, dtype=np.float64).sum(axis=0)
        count += data.shape[0]

        if index % 250 == 0 or index == len(recordings):
            print(f"  Processed {index}/{len(recordings)} recordings")

    mean = channel_sum / count

    variance = (channel_sum_sq / count) - np.square(mean)
    variance = np.maximum(variance, 1e-12)

    std = np.sqrt(variance)

    return mean.astype(np.float32), std.astype(np.float32)


def save_split(split_name, recordings, mean, std):
    """Window and normalize one dataset split."""
    split_dir = OUTPUT_ROOT / split_name
    windows_dir = split_dir / "windows"

    windows_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = split_dir / "metadata.csv"

    total_windows = 0
    label_counts = {0: 0, 1: 0}

    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow([
            "window_id",
            "recording_file",
            "subject_id",
            "activity_code",
            "trial",
            "label",
            "label_id",
        ])

        for index, path in enumerate(recordings, start=1):
            data, metadata = load_recording(path)

            windows = create_windows(
                data,
                window_size=WINDOW_SIZE,
                stride=STRIDE,
            )

            if len(windows) == 0:
                continue

            # Normalize using statistics calculated ONLY from training data.
            windows = (
                (windows - mean.reshape(1, 1, -1))
                / std.reshape(1, 1, -1)
            ).astype(np.float32)

            recording_name = path.stem
            output_file = windows_dir / f"{recording_name}.npy"

            np.save(output_file, windows)

            label = metadata["label"]
            label_id = LABEL_TO_INT[label]

            for window_index in range(len(windows)):
                window_id = f"{recording_name}_W{window_index:04d}"

                writer.writerow([
                    window_id,
                    path.name,
                    metadata["subject_id"],
                    metadata["activity_code"],
                    metadata["trial"],
                    label,
                    label_id,
                ])

            total_windows += len(windows)
            label_counts[label_id] += len(windows)

            if index % 250 == 0 or index == len(recordings):
                print(
                    f"  {split_name}: "
                    f"{index}/{len(recordings)} recordings processed"
                )

    return total_windows, label_counts


def main():
    print("SisFall preprocessing")
    print("====================")
    print(f"Dataset root: {SISFALL_ROOT}")
    print(f"Output root:  {OUTPUT_ROOT}")
    print()
    print(f"Window size: {WINDOW_SIZE} samples")
    print(f"Stride:      {STRIDE} samples")
    print()

    recordings = find_recordings()

    if not recordings:
        raise RuntimeError(
            "No SisFall recordings found. Check SISFALL_ROOT in dataset.py."
        )

    subjects = get_subjects(recordings)

    train_subjects, val_subjects, test_subjects = split_subjects(subjects)

    train_recordings = recordings_for_subjects(
        recordings,
        set(train_subjects),
    )

    val_recordings = recordings_for_subjects(
        recordings,
        set(val_subjects),
    )

    test_recordings = recordings_for_subjects(
        recordings,
        set(test_subjects),
    )

    print(f"Total recordings: {len(recordings)}")
    print(f"Train recordings: {len(train_recordings)}")
    print(f"Val recordings:   {len(val_recordings)}")
    print(f"Test recordings:  {len(test_recordings)}")
    print()

    # ------------------------------------------------------------
    # STEP 1: Fit normalization ONLY on training recordings.
    # ------------------------------------------------------------
    mean, std = calculate_training_statistics(train_recordings)

    stats_path = OUTPUT_ROOT / "normalization_stats.npz"
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        stats_path,
        mean=mean,
        std=std,
    )

    print()
    print("Training normalization statistics:")
    print("Mean:", mean)
    print("Std: ", std)
    print()

    # ------------------------------------------------------------
    # STEP 2: Process each split independently.
    # ------------------------------------------------------------
    results = {}

    for split_name, split_recordings in [
        ("train", train_recordings),
        ("val", val_recordings),
        ("test", test_recordings),
    ]:
        print(f"Processing {split_name.upper()}...")
        print()

        total_windows, label_counts = save_split(
            split_name,
            split_recordings,
            mean,
            std,
        )

        results[split_name] = (total_windows, label_counts)

        print()
        print(
            f"{split_name.upper()} complete: "
            f"{total_windows} windows | "
            f"normal={label_counts[0]} | "
            f"fall={label_counts[1]}"
        )
        print()

    print("====================")
    print("PREPROCESSING COMPLETE")
    print("====================")

    total = sum(result[0] for result in results.values())

    print(f"Total windows: {total}")

    for split_name, (count, labels) in results.items():
        print(
            f"{split_name}: "
            f"{count} windows "
            f"(normal={labels[0]}, fall={labels[1]})"
        )


if __name__ == "__main__":
    main()
