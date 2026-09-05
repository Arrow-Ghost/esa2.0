from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


PROCESSED_ROOT = Path("ml/data/processed")


class SisFallWindowDataset(Dataset):
    """
    PyTorch Dataset for preprocessed SisFall windows.

    Each sample:
        X: (6, 400)
        y: scalar class label

    Classes:
        0 = normal
        1 = fall
    """

    def __init__(self, split, cache_size=8):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.split = split
        self.split_root = PROCESSED_ROOT / split
        self.windows_root = self.split_root / "windows"
        self.metadata_path = self.split_root / "metadata.csv"

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}"
            )

        self.samples = pd.read_csv(self.metadata_path)

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split: {split}")

        self.cache_size = cache_size
        self.cache = {}
        self.cache_order = []

    def __len__(self):
        return len(self.samples)

    def _load_recording_windows(self, recording_file):
        """
        Load all windows for one recording.

        Uses a small cache to avoid repeatedly reading the same
        .npy file from disk.
        """
        if recording_file in self.cache:
            return self.cache[recording_file]

        path = self.windows_root / Path(recording_file).with_suffix(".npy")

        if not path.exists():
            raise FileNotFoundError(
                f"Processed recording not found: {path}"
            )

        windows = np.load(path)

        # Keep only a limited number of recording arrays in RAM.
        if recording_file in self.cache_order:
            self.cache_order.remove(recording_file)

        self.cache[recording_file] = windows
        self.cache_order.append(recording_file)

        while len(self.cache_order) > self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]

        return windows

    def __getitem__(self, index):
        row = self.samples.iloc[index]

        recording_file = row["recording_file"]

        # window_id format:
        # D01_SA01_R01_w0000
        match = re.search(r"_W(\d+)$", row["window_id"])

        if match is None:
            raise ValueError(
                f"Invalid window_id: {row['window_id']}"
            )

        window_index = int(match.group(1))

        windows = self._load_recording_windows(recording_file)

        if window_index >= len(windows):
            raise IndexError(
                f"Window {window_index} does not exist in "
                f"{recording_file}. Available: {len(windows)}"
            )

        # Original shape:
        # (400, 6)
        #
        # PyTorch model expects:
        # (6, 400)
        x = windows[window_index]

        x = torch.from_numpy(
            x.transpose(1, 0).copy()
        ).float()

        y = torch.tensor(
            int(row["label_id"]),
            dtype=torch.long
        )

        return x, y


def create_dataloader(
    split,
    batch_size=64,
    shuffle=None,
    num_workers=0,
    cache_size=8,
):
    """
    Create a DataLoader for a dataset split.
    """

    dataset = SisFallWindowDataset(
        split=split,
        cache_size=cache_size,
    )

    if shuffle is None:
        shuffle = split == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


if __name__ == "__main__":
    print("Testing SisFall DataLoader")
    print("=========================")

    for split in ["train", "val", "test"]:
        dataset = SisFallWindowDataset(split)

        print(
            f"{split}: {len(dataset)} samples"
        )

    print()

    loader = create_dataloader(
        "train",
        batch_size=8,
    )

    x, y = next(iter(loader))

    print("Batch shape:", tuple(x.shape))
    print("Labels shape:", tuple(y.shape))
    print("Labels:", y.tolist())
    print("Input dtype:", x.dtype)
    print("Label dtype:", y.dtype)

    print()
    print("Expected input shape: (batch, 6, 400)")
    print("Expected label shape: (batch,)")

    print()
    print("Cache test:")

    dataset = loader.dataset

    _ = dataset[0]
    print("After sample 0:", len(dataset.cache), "cached recording(s)")

    _ = dataset[1]
    print("After sample 1:", len(dataset.cache), "cached recording(s)")

    print()
    print("DataLoader test passed.")
