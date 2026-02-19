import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_sequences(filepath: str):
    """Load binary file with per-sequence length headers.

    Format: [len: u32] [word0..wordN: u32] ...
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


def load_sequences_as_bytes(filepath: str):
    """Load binary file and convert u32 sequences to byte sequences.

    Returns list of numpy arrays, each shape (seq_len*4,) dtype uint8.
    """
    sequences = load_sequences(filepath)
    byte_sequences = []
    for seq in sequences:
        byte_sequences.append(seq.view(np.uint8))  # u32 LE -> 4 bytes each

    total_bytes = sum(len(s) for s in byte_sequences)
    print(f"Converted to bytes: {len(byte_sequences):,} sequences ({total_bytes:,} total bytes)")
    return byte_sequences


def u32_to_bits(value: np.ndarray) -> np.ndarray:
    """Convert u32 array to bit representation. Shape: (N,) -> (N, 32)."""
    bits = np.unpackbits(
        value.view(np.uint8).reshape(*value.shape, 4)[:, ::-1],
        axis=-1,
    )
    return bits.astype(np.float32)


# ── Block-level dataset (64-byte block -> next 64-byte block) ─────────

BLOCK_BYTES = 64  # ChaCha block = 16 x u32 = 64 bytes

class ChaChaBlockDataset(Dataset):
    """Block-to-block prediction dataset.

    Input: one 64-byte block as long tensor -> shape (64,)
    Target: next 64-byte block as long tensor -> shape (64,)
    Each byte is 0-255 (256-class per position).
    """

    def __init__(self, sequences, block_size: int = BLOCK_BYTES):
        self.block_size = block_size
        self.samples = []

        for seq in sequences:
            t = torch.from_numpy(seq.copy()).long()
            n_blocks = len(t) // block_size
            for b in range(n_blocks - 1):
                x = t[b * block_size : (b + 1) * block_size]
                y = t[(b + 1) * block_size : (b + 2) * block_size]
                self.samples.append((x, y))

        print(f"  Created {len(self.samples):,} block pairs (block_size={block_size})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]  # (x: (64,), y: (64,))


class ChaChaBlockToByteDataset(Dataset):
    """Block-to-next-byte prediction dataset.

    Input: one 64-byte block as long tensor -> shape (64,)
    Target: first byte of next block as long scalar (0-255)
    """

    def __init__(self, sequences, block_size: int = BLOCK_BYTES):
        self.block_size = block_size
        self.samples = []

        for seq in sequences:
            t = torch.from_numpy(seq.copy()).long()
            n_blocks = len(t) // block_size
            for b in range(n_blocks - 1):
                x = t[b * block_size : (b + 1) * block_size]          # (64,)
                y = t[(b + 1) * block_size]                            # scalar
                self.samples.append((x, y))

        print(f"  Created {len(self.samples):,} block->byte pairs (block_size={block_size})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]  # (x: (64,), y: scalar)


def get_block_dataloaders(
    filepath: str,
    mode: str = "block64",
    batch_size: int = 1024,
    train_ratio: float = 0.8,
    num_workers: int = 4,
):
    """Load data as bytes and return train/val DataLoaders.

    mode="block64": predict next 64-byte block
    mode="byte1":   predict first byte of next block
    """
    sequences = load_sequences_as_bytes(filepath)

    split = int(len(sequences) * train_ratio)
    train_seqs = sequences[:split]
    val_seqs = sequences[split:]

    ds_cls = ChaChaBlockDataset if mode == "block64" else ChaChaBlockToByteDataset

    print(f"Mode: {mode}")
    print("Train set:")
    train_ds = ds_cls(train_seqs)
    print("Val set:")
    val_ds = ds_cls(val_seqs)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


# ── Byte-level dataset (paper approach, kept for compatibility) ───────

class ChaChaByteDataset(Dataset):
    """Sliding window dataset for next-byte prediction (256-class)."""

    def __init__(self, sequences, window_size: int = 100):
        self.window_size = window_size
        self.samples = []
        self.seq_data = []

        for seq in sequences:
            t = torch.from_numpy(seq.copy()).long()
            self.seq_data.append(t)
            seq_idx = len(self.seq_data) - 1
            for pos in range(len(seq) - window_size):
                self.samples.append((seq_idx, pos))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_idx, pos = self.samples[idx]
        data = self.seq_data[seq_idx]
        x = data[pos : pos + self.window_size]
        y = data[pos + self.window_size]
        return x, y


def get_byte_dataloaders(
    filepath: str,
    window_size: int = 100,
    batch_size: int = 1024,
    train_ratio: float = 0.8,
    num_workers: int = 4,
):
    """Load data as bytes and return train/val DataLoaders for next-byte prediction."""
    sequences = load_sequences_as_bytes(filepath)

    split = int(len(sequences) * train_ratio)
    train_seqs = sequences[:split]
    val_seqs = sequences[split:]

    train_ds = ChaChaByteDataset(train_seqs, window_size)
    val_ds = ChaChaByteDataset(val_seqs, window_size)

    print(f"Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


# ── Bit-level dataset (old approach, kept for compatibility) ─────────

class ChaChaSequenceDataset(Dataset):
    def __init__(self, sequences, window_size: int = 4, flatten: bool = True):
        self.window_size = window_size
        self.flatten = flatten
        self.samples = []
        self.seq_bits = []

        for seq in sequences:
            bits = torch.from_numpy(u32_to_bits(seq))
            self.seq_bits.append(bits)
            seq_idx = len(self.seq_bits) - 1
            for pos in range(len(seq) - window_size):
                self.samples.append((seq_idx, pos))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_idx, pos = self.samples[idx]
        bits = self.seq_bits[seq_idx]
        x = bits[pos : pos + self.window_size]
        y = bits[pos + self.window_size]
        if self.flatten:
            x = x.reshape(-1)
        return x, y


def get_dataloaders(
    filepath: str,
    window_size: int = 4,
    batch_size: int = 1024,
    train_ratio: float = 0.8,
    flatten: bool = True,
    num_workers: int = 4,
):
    sequences = load_sequences(filepath)
    split = int(len(sequences) * train_ratio)
    train_ds = ChaChaSequenceDataset(sequences[:split], window_size, flatten=flatten)
    val_ds = ChaChaSequenceDataset(sequences[split:], window_size, flatten=flatten)
    print(f"Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
