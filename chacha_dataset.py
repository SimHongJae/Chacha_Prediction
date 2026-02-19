import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_sequences(filepath: str):
    """Load binary file with per-sequence length headers.

    Format: [len: u32] [word0..wordN: u32] [len: u32] [word0..wordN: u32] ...
    Returns list of numpy arrays, each shape (seq_len,) dtype uint32.
    """
    raw = np.fromfile(filepath, dtype=np.uint32)
    sequences = []
    i = 0
    while i < len(raw):
        seq_len = int(raw[i])
        i += 1
        if i + seq_len > len(raw):
            break
        sequences.append(raw[i : i + seq_len])
        i += seq_len

    total_words = sum(len(s) for s in sequences)
    print(f"Loaded {len(sequences):,} sequences ({total_words:,} total words) from {filepath}")
    return sequences


def u32_to_bits(value: np.ndarray) -> np.ndarray:
    """Convert u32 array to bit representation. Shape: (N,) -> (N, 32)."""
    bits = np.unpackbits(
        value.view(np.uint8).reshape(*value.shape, 4)[:, ::-1],
        axis=-1,
    )
    return bits.astype(np.float32)


class ChaChaSequenceDataset(Dataset):
    """Sliding window dataset for next-u32 prediction with multiple sequences.

    Each sequence is independent (no cross-sequence windows).
    Input: previous `window_size` u32 words as bits
    Target: next u32 word as bits -> shape (32,)
    """

    def __init__(self, sequences, window_size: int = 4, flatten: bool = True):
        self.window_size = window_size
        self.flatten = flatten

        # Build index: (seq_idx, position_in_seq) for each valid sample
        self.samples = []
        self.seq_bits = []

        for seq in sequences:
            bits = torch.from_numpy(u32_to_bits(seq))  # (seq_len, 32)
            self.seq_bits.append(bits)
            seq_idx = len(self.seq_bits) - 1
            # Valid positions: 0 to len(seq) - window_size - 1
            for pos in range(len(seq) - window_size):
                self.samples.append((seq_idx, pos))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_idx, pos = self.samples[idx]
        bits = self.seq_bits[seq_idx]
        x = bits[pos : pos + self.window_size]       # (window_size, 32)
        y = bits[pos + self.window_size]              # (32,)
        if self.flatten:
            x = x.reshape(-1)                         # (window_size * 32,)
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
    sequences = load_sequences(filepath)

    split = int(len(sequences) * train_ratio)
    train_seqs = sequences[:split]
    val_seqs = sequences[split:]

    train_ds = ChaChaSequenceDataset(train_seqs, window_size, flatten=flatten)
    val_ds = ChaChaSequenceDataset(val_seqs, window_size, flatten=flatten)

    print(f"Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
