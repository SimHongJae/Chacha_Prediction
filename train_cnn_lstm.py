"""CNN+LSTM fixed-key next-byte prediction for ChaCha keystream analysis.

Task: Given a fixed key, observe W consecutive keystream bytes and predict
      the next byte (256-class classification).

Model: Embedding -> CNN (Conv1d) -> LSTM -> FC -> 256 logits
Metric: Pml (prediction-at-most-likely), baseline Pg = 1/256 ≈ 0.39%

Data file formats supported:
  --flat : raw byte dump (no headers)  ← single fixed-key stream
  default: u32 length-prefixed multi-sequence format (all sequences
           concatenated into one stream before use)
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import vessl
from tqdm import tqdm
from chacha_dataset import get_stream_dataloaders


# ── Model ─────────────────────────────────────────────────────────────

class CNN_LSTM(nn.Module):
    """Embedding -> CNN -> LSTM -> FC for next-byte prediction.

    Args:
        vocab_size:     256 (byte alphabet)
        window_size:    number of input bytes (sequence length for CNN/LSTM)
        embed_dim:      byte embedding dimension
        cnn_channels:   list of output channels for successive Conv1d layers
        kernel_size:    conv kernel size (same for all layers)
        lstm_hidden:    LSTM hidden state size
        lstm_layers:    number of stacked LSTM layers
        lstm_dropout:   dropout between LSTM layers (only if lstm_layers > 1)
    """

    def __init__(
        self,
        vocab_size: int = 256,
        window_size: int = 64,
        embed_dim: int = 16,
        cnn_channels: list = None,
        kernel_size: int = 3,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [64, 128]

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Build CNN: each block = Conv1d -> ReLU -> MaxPool1d(2)
        cnn_layers = []
        in_ch = embed_dim
        for out_ch in cnn_channels:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_out_ch = in_ch

        self.lstm = nn.LSTM(
            input_size=self.cnn_out_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(lstm_hidden, vocab_size)

    def forward(self, x):
        # x: (batch, window_size)  long
        x = self.embedding(x)           # (batch, window_size, embed_dim)
        x = x.permute(0, 2, 1)          # (batch, embed_dim, window_size)
        x = self.cnn(x)                 # (batch, cnn_out_ch, seq_len')
        x = x.permute(0, 2, 1)          # (batch, seq_len', cnn_out_ch)
        out, _ = self.lstm(x)
        out = out[:, -1, :]             # (batch, lstm_hidden)
        return self.fc(out)             # (batch, 256)


# ── Evaluate ──────────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)                       # (batch, 256)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

    return {
        "loss": total_loss / total_samples,
        "pml": total_correct / total_samples,
    }


# ── Train ─────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, args):
    pg = 1.0 / 256
    history = {"train_loss": [], "val_loss": [], "val_pml": []}

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", file=sys.stdout)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
            pbar.set_postfix(loss=f"{epoch_loss/n_samples:.6f}")

        train_loss = epoch_loss / n_samples
        metrics = evaluate(model, val_loader, criterion, device)

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
        vessl.log(
            step=epoch + 1,
            payload={
                "train_loss": train_loss,
                "val_loss": metrics["loss"],
                "pml": metrics["pml"],
                "pml_over_pg": metrics["pml"] / pg,
                "lr": lr,
            },
        )

    return history


# ── Plot ──────────────────────────────────────────────────────────────

def plot_results(history, args):
    pg = 1.0 / 256
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title(f"CNN+LSTM - ChaCha{args.rounds} (window={args.window_size})")
    axes[0].legend()
    axes[0].grid(True)

    epochs = range(1, len(history["val_pml"]) + 1)
    pml_pct = [p * 100 for p in history["val_pml"]]
    axes[1].plot(epochs, pml_pct, label="Pml", marker="o", markersize=3)
    axes[1].axhline(y=pg * 100, color="r", linestyle="--", label=f"Pg ({pg:.2%})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"CNN+LSTM Next-Byte Prediction - ChaCha{args.rounds}")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    fname = f"cnn_lstm_chacha{args.rounds}_w{args.window_size}_results.png"
    plt.savefig(fname, dpi=150)
    print(f"Plot saved to {fname}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"ChaCha rounds: {args.rounds}")
    print(f"Window size: {args.window_size} bytes")
    print(f"Data file: {args.data}  (flat={args.flat})")

    train_loader, val_loader = get_stream_dataloaders(
        filepath=args.data,
        window_size=args.window_size,
        batch_size=args.batch_size,
        train_ratio=0.8,
        num_workers=args.num_workers,
        flat=args.flat,
    )

    model = CNN_LSTM(
        vocab_size=256,
        window_size=args.window_size,
        embed_dim=args.embed_dim,
        cnn_channels=args.cnn_channels,
        kernel_size=args.kernel_size,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        lstm_dropout=args.lstm_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Random guess baseline Pg = {1/256:.4%}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    history = train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, args)
    plot_results(history, args)

    fname = f"cnn_lstm_chacha{args.rounds}_w{args.window_size}.pt"
    torch.save(model.state_dict(), fname)
    print(f"Model saved to {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CNN+LSTM next-byte prediction on ChaCha fixed-key keystream"
    )
    # Data
    parser.add_argument("--data", default="dataset/chacha4_stream.bin",
                        help="Path to keystream binary file")
    parser.add_argument("--flat", action="store_true",
                        help="File is raw bytes (no u32 length headers)")
    parser.add_argument("--rounds", type=int, default=4,
                        help="Number of ChaCha rounds (for labeling only)")
    parser.add_argument("--window-size", type=int, default=64,
                        help="Sliding window size (bytes fed as input)")

    # Training
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)

    # Model architecture
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--cnn-channels", type=int, nargs="+", default=[64, 128],
                        help="Output channels for each CNN layer (e.g. --cnn-channels 64 128 256)")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--lstm-hidden", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--lstm-dropout", type=float, default=0.2)

    args = parser.parse_args()
    main(args)
