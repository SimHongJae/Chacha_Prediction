import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ── Raw stream loaders ────────────────────────────────────────────────

def load_flat_bytes(filepath: str) -> np.ndarray:
    """Load a flat binary file as a single byte stream.

    Expects raw bytes with no length headers.
    Returns numpy array shape (N,) dtype uint8.
    """
    data = np.fromfile(filepath, dtype=np.uint8)
    print(f"Loaded flat stream: {len(data):,} bytes from {filepath}")
    return data


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
    """Load binary file and convert u32 sequences to byte sequences."""
    sequences = load_sequences(filepath)
    byte_sequences = [seq.view(np.uint8) for seq in sequences]
    total_bytes = sum(len(s) for s in byte_sequences)
    print(f"Converted to bytes: {len(byte_sequences):,} sequences ({total_bytes:,} total bytes)")
    return byte_sequences


def load_single_stream_as_bytes(filepath: str) -> np.ndarray:
    """Load multi-sequence binary file, concatenate all into one byte stream.

    Use this when the file contains one key's worth of data across multiple
    ChaCha blocks (each recorded as a separate sequence entry).
    Returns numpy array shape (N,) dtype uint8.
    """
    byte_sequences = load_sequences_as_bytes(filepath)
    stream = np.concatenate(byte_sequences)
    print(f"Concatenated into single stream: {len(stream):,} bytes")
    return stream


# ── Fixed-key stream dataset (main approach) ──────────────────────────

class ChaChaStreamDataset(Dataset):
    """Sliding-window next-byte prediction dataset from a single keystream.

    Input:  window of `window_size` consecutive bytes -> shape (window_size,) long
    Target: next byte after the window -> scalar long (0-255)

    Train/val split is done temporally (by position) before constructing
    this dataset, so pass the appropriate slice of the stream.
    """

    def __init__(self, stream: np.ndarray, window_size: int = 64):
        self.window_size = window_size
        # Keep as uint8 numpy array; convert to tensor on access to save RAM
        self.data = torch.from_numpy(stream.astype(np.int64))
        self.n_samples = max(0, len(stream) - window_size)
        print(f"  StreamDataset: {self.n_samples:,} samples "
              f"(stream={len(stream):,} bytes, window={window_size})")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]          # (window_size,)
        y = self.data[idx + self.window_size]                # scalar
        return x, y


def get_stream_dataloaders(
    filepath: str,
    window_size: int = 64,
    batch_size: int = 1024,
    train_ratio: float = 0.8,
    num_workers: int = 4,
    flat: bool = False,
):
    """Build train/val DataLoaders for fixed-key next-byte prediction.

    Args:
        filepath:    Path to the keystream binary file.
        window_size: Number of preceding bytes fed as input.
        batch_size:  Mini-batch size.
        train_ratio: Fraction of stream used for training (temporal split).
        num_workers: DataLoader worker processes.
        flat:        True  -> file is raw bytes (no length headers)
                     False -> file has u32 length-prefixed sequences
    """
    if flat:
        stream = load_flat_bytes(filepath)
    else:
        stream = load_single_stream_as_bytes(filepath)

    split = int(len(stream) * train_ratio)
    train_stream = stream[:split]
    val_stream = stream[split:]

    print("Train set:")
    train_ds = ChaChaStreamDataset(train_stream, window_size)
    print("Val set:")
    val_ds = ChaChaStreamDataset(val_stream, window_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ── Block-pair dataset (block N → block N+1 prediction) ──────────────

BLOCK_BYTES = 64


class ChaChaBlockPairDataset(Dataset):
    """Block-pair prediction dataset from a flat keystream.

    Input:  block N   -> shape (64,) long (byte values 0-255)
    Target: block N+1 -> shape (64,) long (byte values 0-255)

    Splits are done on block boundaries to avoid leakage.
    """

    def __init__(self, stream: np.ndarray):
        n_blocks = len(stream) // BLOCK_BYTES
        stream = stream[: n_blocks * BLOCK_BYTES]
        blocks = torch.from_numpy(stream.astype(np.int64)).view(n_blocks, BLOCK_BYTES)
        self.x = blocks[:-1]  # blocks 0 .. N-2
        self.y = blocks[1:]   # blocks 1 .. N-1
        print(f"  BlockPairDataset: {len(self.x):,} pairs from {n_blocks:,} blocks")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_block_pair_dataloaders(
    filepath: str,
    batch_size: int = 256,
    train_ratio: float = 0.8,
    num_workers: int = 4,
    flat: bool = True,
):
    """Build train/val DataLoaders for block-pair (block N → block N+1) prediction.

    Args:
        filepath:    Path to keystream binary file.
        batch_size:  Mini-batch size.
        train_ratio: Fraction of blocks used for training (split on block boundary).
        num_workers: DataLoader worker processes.
        flat:        True  -> file is raw bytes (no length headers)
                     False -> file has u32 length-prefixed sequences
    """
    if flat:
        stream = load_flat_bytes(filepath)
    else:
        stream = load_single_stream_as_bytes(filepath)

    n_blocks = len(stream) // BLOCK_BYTES
    split_block = int(n_blocks * train_ratio)
    split_byte = split_block * BLOCK_BYTES

    print("Train set:")
    train_ds = ChaChaBlockPairDataset(stream[:split_byte])
    print("Val set:")
    val_ds = ChaChaBlockPairDataset(stream[split_byte:])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ── Sweep datasets: fixed key, vary only counter or nonce ─────────────

class ChaChaCounterDataset(Dataset):
    """Counter-sweep dataset: input=counter(4 bytes), output=block(64 bytes).

    Reads a flat stream file (chacha{R}_stream.bin).
    Counter for block at position N is simply N — no extra storage needed.

    Input:  counter as 4 uint8 values (little-endian) -> shape (4,) long
    Target: 64-byte block output                       -> shape (64,) long
    """

    def __init__(self, stream: np.ndarray):
        n_blocks = len(stream) // BLOCK_BYTES
        stream = stream[: n_blocks * BLOCK_BYTES]
        blocks = torch.from_numpy(stream.astype(np.int64)).view(n_blocks, BLOCK_BYTES)
        # Build counter tensors: counter N -> 4 bytes little-endian
        counters = np.arange(n_blocks, dtype=np.uint32).view(np.uint8).reshape(n_blocks, 4)
        self.x = torch.from_numpy(counters.astype(np.int64))  # (N, 4)
        self.y = blocks                                         # (N, 64)
        print(f"  CounterDataset: {n_blocks:,} samples (input=4B counter, output=64B block)")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ChaChaNonceDataset(Dataset):
    """Nonce-sweep dataset: input=nonce(12 bytes), output=block(64 bytes).

    Reads a nonce-sweep file (chacha{R}_nonce_sweep.bin).
    Record format: [nonce: 12 bytes][block_output: 64 bytes] = 76 bytes each.

    Input:  nonce as 12 uint8 values -> shape (12,) long
    Target: 64-byte block output     -> shape (64,) long
    """

    RECORD = 76  # 12 + 64

    def __init__(self, data: np.ndarray):
        assert len(data) % self.RECORD == 0, "File size not a multiple of 76"
        n = len(data) // self.RECORD
        records = data.reshape(n, self.RECORD)
        self.x = torch.from_numpy(records[:, :12].astype(np.int64))   # (N, 12)
        self.y = torch.from_numpy(records[:, 12:].astype(np.int64))   # (N, 64)
        print(f"  NonceDataset: {n:,} samples (input=12B nonce, output=64B block)")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_counter_dataloaders(
    filepath: str,
    batch_size: int = 512,
    train_ratio: float = 0.8,
    num_workers: int = 0,
):
    """Train/val loaders for counter-sweep (reads flat stream file)."""
    stream = load_flat_bytes(filepath)
    n_blocks = len(stream) // BLOCK_BYTES
    split = int(n_blocks * train_ratio) * BLOCK_BYTES

    print("Train set:")
    train_ds = ChaChaCounterDataset(stream[:split])
    print("Val set:")
    val_ds = ChaChaCounterDataset(stream[split:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def get_nonce_dataloaders(
    filepath: str,
    batch_size: int = 512,
    train_ratio: float = 0.8,
    num_workers: int = 0,
):
    """Train/val loaders for nonce-sweep (reads nonce_sweep.bin file)."""
    data = np.fromfile(filepath, dtype=np.uint8)
    n = len(data) // 76
    split = int(n * train_ratio) * 76

    print("Train set:")
    train_ds = ChaChaNonceDataset(data[:split])
    print("Val set:")
    val_ds   = ChaChaNonceDataset(data[split:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# ── Legacy: block-level datasets (kept for compatibility) ─────────────


class ChaChaBlockDataset(Dataset):
    """Block-to-block prediction dataset (legacy)."""

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
        return self.samples[idx]


class ChaChaByteDataset(Dataset):
    """Sliding window next-byte prediction across multiple sequences (legacy)."""

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
    sequences = load_sequences_as_bytes(filepath)
    split = int(len(sequences) * train_ratio)
    train_ds = ChaChaByteDataset(sequences[:split], window_size)
    val_ds = ChaChaByteDataset(sequences[split:], window_size)
    print(f"Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
