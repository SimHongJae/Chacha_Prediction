"""
Generate a single continuous ChaCha20 keystream for CNN+LSTM cryptanalysis.

Paper: "Strengthening the No-Go Theorem for QRNGs" (arXiv:2503.18026v2, 2025)
Section III.3 Case III — raw/unprocessed ChaCha20 stream, Table 3.

Protocol (must match paper exactly):
  - Fixed 32-byte key  : 32 zero bytes
  - Fixed 12-byte nonce: 12 zero bytes (RFC 8439 format)
  - Counter            : starts at 0, increments block-by-block automatically
  - Stream length      : 625,000 bytes  =  5,000,000 bits
  - Output             : flat binary file, no headers (raw keystream bytes)

The `cryptography` library's ChaCha20 primitive accepts a 16-byte nonce
encoded as [counter(4B, little-endian)] + [nonce(12B)].  Setting all 16
bytes to zero starts the counter at 0 with a zero nonce, which is the
standard single-key continuous-counter mode described in the paper.

Usage:
    pip install cryptography numpy
    python generate_chacha_stream.py
    # → writes  chacha_raw_5M.bin  (625,000 bytes)
"""

import os
import struct
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.backends import default_backend

# ── Parameters (Section III.3: fixed key + fixed nonce, counter increments) ──
KEY_BYTES: int = 32
NONCE_BYTES: int = 12
TOTAL_BYTES: int = 625_000   # 5,000,000 bits ÷ 8
OUTPUT_PATH: str = "chacha_raw_5M.bin"

# Fixed key and nonce (all zeros — any fixed value works; zeros are canonical)
KEY   = bytes(KEY_BYTES)
NONCE = bytes(NONCE_BYTES)


def generate_chacha20_stream(key: bytes, nonce_12: bytes, n_bytes: int) -> bytes:
    """Return `n_bytes` bytes of ChaCha20 keystream.

    The cryptography library encodes the RFC 8439 16-byte nonce as:
        nonce[0:4]  = initial counter value (little-endian uint32)
        nonce[4:16] = the 12-byte RFC 8439 nonce

    We start the counter at 0, so the full 16-byte value is all zeros when
    the 12-byte nonce is also all zeros.  Each internal 64-byte ChaCha20
    block increments the counter by 1, producing a seamless keystream.
    """
    assert len(key)      == 32, "Key must be 32 bytes"
    assert len(nonce_12) == 12, "Nonce must be 12 bytes"

    # Build 16-byte nonce: counter=0 (LE) || nonce_12
    nonce_16 = struct.pack("<I", 0) + nonce_12   # counter starts at 0

    algorithm = algorithms.ChaCha20(key, nonce_16)
    cipher    = Cipher(algorithm, mode=None, backend=default_backend())
    enc       = cipher.encryptor()

    # Encrypt an all-zero plaintext to extract the raw keystream
    keystream = enc.update(bytes(n_bytes))
    return keystream


def main() -> None:
    print("=" * 60)
    print("ChaCha20 stream generator")
    print(f"  Paper  : arXiv:2503.18026v2, Section III.3 Case III")
    print(f"  Key    : {KEY.hex()}")
    print(f"  Nonce  : {NONCE.hex()}")
    print(f"  Counter: 0 -> auto-increment per 64-byte block")
    print(f"  Length : {TOTAL_BYTES:,} bytes  ({TOTAL_BYTES * 8:,} bits)")
    print(f"  Output : {OUTPUT_PATH}")
    print("=" * 60)

    keystream = generate_chacha20_stream(KEY, NONCE, TOTAL_BYTES)

    with open(OUTPUT_PATH, "wb") as fh:
        fh.write(keystream)

    size = os.path.getsize(OUTPUT_PATH)
    print(f"Written: {size:,} bytes  OK")

    # Sanity checks
    data   = np.frombuffer(keystream, dtype=np.uint8)
    counts = np.bincount(data, minlength=256)
    print(f"Byte stats : mean={data.mean():.3f} (ideal 127.5), "
          f"std={data.std():.3f} (ideal ~73.9)")
    print(f"Value dist : min_count={counts.min()}, "
          f"max_count={counts.max()}, "
          f"expected~{TOTAL_BYTES//256}")
    print("Done.")


if __name__ == "__main__":
    main()
