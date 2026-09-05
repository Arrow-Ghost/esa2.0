from pathlib import Path
import re

import numpy as np


# SisFall dataset location on the local machine.
# Keep the dataset outside the Git repository.
SISFALL_ROOT = Path("D:/Downloads/SisFall_extracted/SisFall_dataset")


# SisFall raw files contain:
# 1-3: ADXL345 acceleration X/Y/Z
# 4-6: ITG3200 rotation X/Y/Z
# 7-9: MMA8451Q acceleration X/Y/Z
#
# For the first experiment we use:
# ADXL345 acceleration + ITG3200 rotation = 6 features.
FEATURE_COLUMNS = [0, 1, 2, 3, 4, 5]


def parse_filename(path: Path) -> dict:
    """
    Extract metadata from a SisFall recording filename.

    Example:
        D01_SA01_R01.txt
        F02_SE06_R03.txt
    """
    match = re.fullmatch(
        r"([DF]\d{2})_(S[AE]\d{2})_R(\d{2})\.txt",
        path.name,
    )

    if match is None:
        raise ValueError(f"Unexpected SisFall filename: {path.name}")

    activity_code = match.group(1)
    subject_id = match.group(2)
    trial = int(match.group(3))

    label = "fall" if activity_code.startswith("F") else "normal"

    return {
        "subject_id": subject_id,
        "activity_code": activity_code,
        "trial": trial,
        "label": label,
    }


def load_recording(path: Path) -> tuple[np.ndarray, dict]:
    """
    Load one SisFall recording.

    Returns:
        data:
            NumPy array with shape (timesteps, 6)
            containing acceleration XYZ + rotation XYZ.

        metadata:
            Recording metadata extracted from the filename.
    """
    metadata = parse_filename(path)

    rows = []

    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            # SisFall rows end with ';'
            line = line.rstrip(";").strip()

            values = [value.strip() for value in line.split(",")]

            if len(values) != 9:
                raise ValueError(
                    f"{path.name}: expected 9 values on line "
                    f"{line_number}, found {len(values)}"
                )

            try:
                row = [float(value) for value in values]
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}: non-numeric value on line "
                    f"{line_number}"
                ) from exc

            rows.append(row)

    if not rows:
        raise ValueError(f"No sensor data found in {path}")

    data = np.asarray(rows, dtype=np.float32)

    # Keep ADXL345 XYZ + ITG3200 XYZ.
    data = data[:, FEATURE_COLUMNS]

    return data, metadata


def find_recordings(root: Path = SISFALL_ROOT) -> list[Path]:
    """Find all valid SisFall recording files."""
    recordings = []

    for path in root.rglob("*.txt"):
        if re.fullmatch(r"[DF]\d{2}_S[AE]\d{2}_R\d{2}\.txt", path.name):
            recordings.append(path)

    return sorted(recordings)


if __name__ == "__main__":
    recordings = find_recordings()

    print(f"Found recordings: {len(recordings)}")

    if not recordings:
        raise SystemExit("No SisFall recordings found.")

    # Test one normal and one fall recording.
    normal_path = next(
        path for path in recordings
        if path.name.startswith("D")
    )

    fall_path = next(
        path for path in recordings
        if path.name.startswith("F")
    )

    for path in [normal_path, fall_path]:
        data, metadata = load_recording(path)

        print()
        print("File:", path.name)
        print("Metadata:", metadata)
        print("Shape:", data.shape)
        print("Dtype:", data.dtype)
        print("First 3 rows:")
        print(data[:3])


def create_windows(
    data: np.ndarray,
    window_size: int = 400,
    stride: int = 200,
) -> np.ndarray:
    """
    Split one recording into overlapping fixed-size windows.

    Args:
        data: Sensor data with shape (timesteps, features).
        window_size: Number of timesteps in each window.
        stride: Number of timesteps to move between windows.

    Returns:
        Array with shape (num_windows, window_size, features).
    """
    if data.ndim != 2:
        raise ValueError(
            f"Expected 2D data (timesteps, features), got shape {data.shape}"
        )

    num_samples = data.shape[0]

    if num_samples < window_size:
        return np.empty(
            (0, window_size, data.shape[1]),
            dtype=data.dtype,
        )

    windows = []

    for start in range(0, num_samples - window_size + 1, stride):
        end = start + window_size
        windows.append(data[start:end])

    return np.stack(windows).astype(np.float32)


def window_recording(
    path: Path,
    window_size: int = 400,
    stride: int = 200,
) -> tuple[np.ndarray, dict]:
    """
    Load a recording and divide it into overlapping windows.

    Returns:
        windows:
            Shape (num_windows, window_size, 6)

        metadata:
            Recording metadata.
    """
    data, metadata = load_recording(path)

    windows = create_windows(
        data,
        window_size=window_size,
        stride=stride,
    )

    return windows, metadata
