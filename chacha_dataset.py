import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_binary_u32(filepath: str) -> np.ndarray:
    """Load binary file of u32 little-endian values into numpy array."""
    data = np.fromfile(filepath, dtype=np.uint32)
    print(f"Loaded {len(data):,} u32 words from {filepath}")
    return data


def u32_to_bits(value: np.ndarray) -> np.ndarray:
    """Convert u32 array to bit representation. Shape: (*) -> (*, 32)."""
    bits = np.unpackbits(
        value.view(np.uint8).reshape(*value.shape, 4)[:, ::-1],  # big-endian bit order
        axis=-1,
    )
    return bits.astype(np.float32)


class ChaChaSequenceDataset(Dataset):
    """Sliding window dataset for next-u32 prediction.

    Input: previous `window_size` u32 words as bits -> shape (window_size, 32) or (window_size*32,)
    Target: next u32 word as bits -> shape (32,)
    """

    def __init__(self, data: np.ndarray, window_size: int = 4, flatten: bool = True):
        self.window_size = window_size
        self.flatten = flatten
        # Convert all data to bits once, store as tensor
        self.bits = torch.from_numpy(u32_to_bits(data))  # shape (N, 32)
        self.length = len(data) - window_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.bits[idx : idx + self.window_size]  # (window_size, 32)
        y = self.bits[idx + self.window_size]  # (32,)
        if self.flatten:
            x = x.reshape(-1)  # (window_size * 32,)
        return x, y


def get_dataloaders(
    filepath: str,
    window_size: int = 4,
    batch_size: int = 1024,
    train_ratio: float = 0.8,
    flatten: bool = True,
    num_workers: int = 4,
):
    """Load data and return train/val DataLoaders."""
    data = load_binary_u32(filepath)

    split = int(len(data) * train_ratio)
    train_data = data[:split]
    val_data = data[split:]

    train_ds = ChaChaSequenceDataset(train_data, window_size, flatten=flatten)
    val_ds = ChaChaSequenceDataset(val_data, window_size, flatten=flatten)

    print(f"Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
