"""CNN+LSTM ChaCha keystream analysis.

Paper: "Strengthening the No-Go Theorem for QRNGs" (arXiv:2503.18026v2, 2025)
Architecture: Section II.3.1 "Extraction of features via CNN" + Figure 3.
Target result: ChaCha20 raw stream  →  P_ml ≈ 6.038%  (Table 3, Case III).

Model summary (paper Fig. 3):
  Input  : N=100 previous 8-bit integers, one-hot encoded → (batch, 100, 256)
  CNN-1  : Conv1d(in=256, out=64,  kernel=5) → ReLU → MaxPool1d(2)
  CNN-2  : Conv1d(in=64,  out=128, kernel=3) → ReLU → MaxPool1d(2)
  LSTM   : hidden=128, 1 layer, batch_first=True
  FC-1   : Linear(128 → 64)  + ReLU
  FC-2   : Linear(64  → 256) + softmax  (applied via CrossEntropyLoss during training)
  Output : 256-class prediction of next byte

Modes (--mode):
  next-byte : given W consecutive bytes, predict the next byte (256-class).
              Metric: P_ml vs P_g = 1/256 ≈ 0.3906%
  block     : given block N (64 bytes), predict block N+1 (64 bytes).
              Loss: mean CrossEntropy over 64 output positions.
              Metric: mean per-byte P_ml vs P_g

Data file formats:
  --flat : raw byte dump (no headers)  ← use this with chacha_raw_5M.bin
  default: u32 length-prefixed multi-sequence format (legacy)
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from chacha_dataset import get_stream_dataloaders, get_block_pair_dataloaders

# Optional vessl logging (gracefully disabled when not installed)
try:
    import vessl
    _VESSL = True
except ImportError:
    _VESSL = False


# ── Model ─────────────────────────────────────────────────────────────

BLOCK_SIZE = 64
VOCAB_SIZE = 256


class CNN_LSTM(nn.Module):
    """One-hot CNN → LSTM → FC  (paper arXiv:2503.18026v2, Section II.3.1).

    Paper architecture (next-byte prediction, N=100):
        Input  : (batch, window_size, VOCAB_SIZE=256)  one-hot float
        CNN-1  : Conv1d(256→64,  k=5, pad=2) → ReLU → MaxPool1d(2)
        CNN-2  : Conv1d(64→128,  k=3, pad=1) → ReLU → MaxPool1d(2)
        LSTM   : hidden=128, layers=1, batch_first=True
        FC-1   : Linear(128→64) + ReLU
        FC-2   : Linear(64→256)  ← logits; softmax applied by CrossEntropyLoss

    For block mode (block N → block N+1) the same CNN+LSTM backbone is
    reused; the FC head is replaced by Linear(128 → BLOCK_SIZE*256).

    Args:
        vocab_size   : 256 (byte alphabet, also = one-hot width)
        window_size  : input sequence length in bytes  (paper: N=100)
        cnn_channels : [64, 128] (paper Fig. 3)
        cnn_kernels  : [5, 3]    (paper Fig. 3)
        lstm_hidden  : 128       (paper Fig. 3)
        lstm_layers  : 1         (paper Fig. 3)
        lstm_dropout : dropout between stacked LSTM layers (ignored if layers=1)
        block_mode   : True → predict all 64 output bytes at once
    """

    def __init__(
        self,
        vocab_size   : int   = VOCAB_SIZE,
        window_size  : int   = 100,
        cnn_channels : list  = None,
        cnn_kernels  : list  = None,
        lstm_hidden  : int   = 128,
        lstm_layers  : int   = 1,
        lstm_dropout : float = 0.0,
        block_mode   : bool  = False,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [64, 128]     # paper Fig. 3
        if cnn_kernels is None:
            cnn_kernels = [5, 3]         # paper Fig. 3

        assert len(cnn_channels) == len(cnn_kernels), \
            "cnn_channels and cnn_kernels must have the same length"

        self.vocab_size = vocab_size
        self.block_mode = block_mode

        # ── CNN (Section II.3.1) ──────────────────────────────────────
        # Input to CNN: (batch, vocab_size=256, seq_len) after permute
        cnn_layers = []
        in_ch = vocab_size                        # one-hot width is the channel dim
        for out_ch, k in zip(cnn_channels, cnn_kernels):
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_out_ch = in_ch                   # = 128 after two layers

        # ── LSTM (Section II.3.1) ─────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size  = self.cnn_out_ch,        # 128
            hidden_size = lstm_hidden,             # 128 (paper)
            num_layers  = lstm_layers,             # 1   (paper)
            batch_first = True,
            dropout     = lstm_dropout if lstm_layers > 1 else 0.0,
        )

        # ── FC head (Section II.3.1) ─────────────────────────────────
        if block_mode:
            # Predict all 64 output bytes at once from the final hidden state
            self.fc = nn.Sequential(
                nn.Linear(lstm_hidden, 64),
                nn.ReLU(),
                nn.Linear(64, BLOCK_SIZE * vocab_size),
            )
        else:
            # Two-layer FC: lstm_hidden(128) → 64 → 256 (paper Fig. 3)
            self.fc = nn.Sequential(
                nn.Linear(lstm_hidden, 64),
                nn.ReLU(),
                nn.Linear(64, vocab_size),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len)  long  — raw byte indices 0-255

        Returns:
            logits: (batch, 256)         in next-byte mode
                    (batch, BLOCK_SIZE, 256)  in block mode
        """
        # One-hot encode: (batch, seq_len) → (batch, seq_len, vocab_size=256)
        # Paper Section II.3.1: "one-hot encoding of each 8-bit integer"
        x_oh = F.one_hot(x, self.vocab_size).float()   # (batch, seq_len, 256)

        # CNN expects (batch, channels, seq_len)
        x_oh = x_oh.permute(0, 2, 1)                   # (batch, 256, seq_len)
        x_cnn = self.cnn(x_oh)                          # (batch, 128, seq_len')
        x_cnn = x_cnn.permute(0, 2, 1)                  # (batch, seq_len', 128)

        # LSTM
        lstm_out, _ = self.lstm(x_cnn)                  # (batch, seq_len', hidden)
        h_last = lstm_out[:, -1, :]                      # (batch, hidden=128)

        # FC head
        logits = self.fc(h_last)

        if self.block_mode:
            # (batch, BLOCK_SIZE * vocab_size) → (batch, BLOCK_SIZE, vocab_size)
            return logits.view(logits.size(0), BLOCK_SIZE, self.vocab_size)
        return logits                                    # (batch, 256)


# ── Evaluate ──────────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)

            if model.block_mode:
                # logits: (batch, 64, 256)  y: (batch, 64)
                loss   = criterion(logits.permute(0, 2, 1), y)
                preds  = logits.argmax(dim=2)             # (batch, 64)
                total_correct += (preds == y).sum().item()
                total_samples += x.size(0) * BLOCK_SIZE
            else:
                # logits: (batch, 256)  y: (batch,)
                loss   = criterion(logits, y)
                preds  = logits.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += x.size(0)

            batch_denom = x.size(0) if not model.block_mode else x.size(0)
            total_loss += loss.item() * batch_denom

    n_batches = total_samples // BLOCK_SIZE if model.block_mode else total_samples
    return {
        "loss": total_loss / max(n_batches, 1),
        "pml" : total_correct / max(total_samples, 1),
    }


# ── Train ─────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, args):
    pg      = 1.0 / VOCAB_SIZE    # random-guess baseline  (1/256 ≈ 0.3906%)
    history = {"train_loss": [], "val_loss": [], "val_pml": []}

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_samples  = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", file=sys.stdout)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)

            if model.block_mode:
                loss = criterion(logits.permute(0, 2, 1), y)
            else:
                loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            n_samples  += x.size(0)
            pbar.set_postfix(loss=f"{epoch_loss/n_samples:.6f}")

        train_loss = epoch_loss / n_samples
        metrics    = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(metrics["loss"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(metrics["loss"])
        history["val_pml"].append(metrics["pml"])

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {metrics['loss']:.6f} | "
            f"Pml: {metrics['pml']:.4%} (Pg: {pg:.4%}) | "
            f"Pml/Pg: {metrics['pml']/pg:.4f}x | "
            f"LR: {lr:.2e}",
            flush=True,
        )

        if _VESSL:
            vessl.log(
                step=epoch + 1,
                payload={
                    "train_loss"  : train_loss,
                    "val_loss"    : metrics["loss"],
                    "pml"         : metrics["pml"],
                    "pml_over_pg" : metrics["pml"] / pg,
                    "lr"          : lr,
                },
            )

    return history


# ── Plot ──────────────────────────────────────────────────────────────

def plot_results(history, args):
    pg  = 1.0 / VOCAB_SIZE
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    mode_label = "block→block" if args.mode == "block" else f"next-byte (w={args.window_size})"
    axes[0].set_title(f"CNN+LSTM  ChaCha{args.rounds}  ({mode_label})")
    axes[0].legend()
    axes[0].grid(True)

    epochs  = range(1, len(history["val_pml"]) + 1)
    pml_pct = [p * 100 for p in history["val_pml"]]
    axes[1].plot(epochs, pml_pct, label="Pml", marker="o", markersize=3)
    axes[1].axhline(y=pg * 100, color="r", linestyle="--", label=f"Pg ({pg:.4%})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"CNN+LSTM ({args.mode})  ChaCha{args.rounds}")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    suffix = "block" if args.mode == "block" else f"w{args.window_size}"
    fname  = f"cnn_lstm_chacha{args.rounds}_{suffix}_results.png"
    plt.savefig(fname, dpi=150)
    print(f"Plot saved to {fname}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────

def main(args):
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block_mode = (args.mode == "block")

    print("=" * 70)
    print(f"CNN+LSTM ChaCha cryptanalysis  (arXiv:2503.18026v2)")
    print(f"  Device       : {device}")
    print(f"  Mode         : {args.mode}")
    print(f"  ChaCha rounds: {args.rounds}")
    print(f"  Data file    : {args.data}  (flat={args.flat})")
    print("=" * 70)

    if block_mode:
        print("Block mode: input=block_N (64 bytes), target=block_{N+1} (64 bytes)")
        train_loader, val_loader = get_block_pair_dataloaders(
            filepath    = args.data,
            batch_size  = args.batch_size,
            train_ratio = 0.8,
            num_workers = args.num_workers,
            flat        = args.flat,
        )
    else:
        print(f"Next-byte mode: window_size={args.window_size} bytes  (paper: N=100)")
        train_loader, val_loader = get_stream_dataloaders(
            filepath    = args.data,
            window_size = args.window_size,
            batch_size  = args.batch_size,
            train_ratio = 0.8,
            num_workers = args.num_workers,
            flat        = args.flat,
        )

    # ── Build model (paper Section II.3.1, Fig. 3) ────────────────────
    model = CNN_LSTM(
        vocab_size   = VOCAB_SIZE,
        window_size  = BLOCK_SIZE if block_mode else args.window_size,
        cnn_channels = args.cnn_channels,
        cnn_kernels  = args.cnn_kernels,
        lstm_hidden  = args.lstm_hidden,
        lstm_layers  = args.lstm_layers,
        lstm_dropout = args.lstm_dropout,
        block_mode   = block_mode,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters   : {n_params:,}")
    print(f"Random guess P_g   : {1/VOCAB_SIZE:.4%}")
    print(f"Paper target P_ml  : ~6.038%  (ChaCha20 raw, Table 3)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    history = train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, args)
    plot_results(history, args)

    suffix = "block" if args.mode == "block" else f"w{args.window_size}"
    fname  = f"cnn_lstm_chacha{args.rounds}_{suffix}.pt"
    torch.save(model.state_dict(), fname)
    print(f"Model saved to {fname}")

    best_pml = max(history["val_pml"])
    print(f"\nBest val P_ml : {best_pml:.4%}  (paper target: ~6.038%)")
    print(f"Pml / Pg      : {best_pml / (1/VOCAB_SIZE):.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CNN+LSTM next-byte prediction on ChaCha keystream "
                    "(arXiv:2503.18026v2, Section II.3.1)"
    )
    # ── Data / mode ───────────────────────────────────────────────────
    parser.add_argument(
        "--mode", choices=["next-byte", "block"], default="next-byte",
        help="next-byte: sliding window → predict 1 byte | "
             "block: block N → block N+1  (paper uses next-byte)")
    parser.add_argument(
        "--data", default="chacha_raw_5M.bin",
        help="Path to keystream binary file  (default: chacha_raw_5M.bin)")
    parser.add_argument(
        "--flat", action="store_true", default=True,
        help="File is raw bytes, no u32 length headers  (default: True)")
    parser.add_argument(
        "--rounds", type=int, default=20,
        help="Number of ChaCha rounds  (for labeling, default: 20)")
    parser.add_argument(
        "--window-size", type=int, default=100,
        help="Sliding window size in next-byte mode  (paper: N=100)")

    # ── Training ──────────────────────────────────────────────────────
    parser.add_argument("--batch-size",   type=int,   default=1024)
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--num-workers",  type=int,   default=0)

    # ── Model architecture (paper Fig. 3 defaults) ────────────────────
    parser.add_argument(
        "--cnn-channels", type=int, nargs="+", default=[64, 128],
        help="Output channels per CNN layer  (paper: 64 128)")
    parser.add_argument(
        "--cnn-kernels",  type=int, nargs="+", default=[5, 3],
        help="Kernel size per CNN layer  (paper: 5 3)")
    parser.add_argument(
        "--lstm-hidden",  type=int,   default=128,
        help="LSTM hidden units  (paper: 128)")
    parser.add_argument(
        "--lstm-layers",  type=int,   default=1,
        help="Stacked LSTM layers  (paper: 1)")
    parser.add_argument(
        "--lstm-dropout", type=float, default=0.0,
        help="LSTM inter-layer dropout  (ignored when lstm-layers=1)")

    args = parser.parse_args()
    main(args)
