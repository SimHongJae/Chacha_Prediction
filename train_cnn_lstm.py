"""CNN+LSTM next-byte prediction model.

Based on: "Strengthening the No-Go Theorem for QRNGs" (arXiv:2503.18026)
- Input: 100 bytes (one-hot encoded to 256 dims)
- CNN: Conv1d(256,64,k=5) -> Pool(2) -> Conv1d(64,128,k=3) -> Pool(2)
- LSTM: hidden=128
- FC: 128 -> 64 (sigmoid) -> 256 (softmax/cross-entropy)
- Metric: Pml (prediction probability, baseline Pg = 1/256 = 0.39%)
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import vessl
from tqdm import tqdm
from chacha_dataset import get_byte_dataloaders


class CNN_LSTM(nn.Module):
    def __init__(self, vocab_size=256, seq_len=100, lstm_hidden=128):
        super().__init__()
        self.vocab_size = vocab_size

        # CNN feature extractor
        self.conv1 = nn.Conv1d(vocab_size, 64, kernel_size=5)    # -> (batch, 64, seq_len-4)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)           # -> (batch, 128, L-2)
        self.pool = nn.MaxPool1d(2)

        # Compute LSTM input size
        # After conv1: seq_len - 4 = 96, pool: 48
        # After conv2: 48 - 2 = 46, pool: 23
        self.lstm = nn.LSTM(128, lstm_hidden, batch_first=True)

        # Classifier
        self.fc1 = nn.Linear(lstm_hidden, 64)
        self.fc2 = nn.Linear(64, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len) long tensor of byte values 0-255
        x = F.one_hot(x, self.vocab_size).float()  # (batch, seq_len, 256)
        x = x.permute(0, 2, 1)                     # (batch, 256, seq_len) for Conv1d

        x = self.pool(F.relu(self.conv1(x)))        # (batch, 64, 48)
        x = self.pool(F.relu(self.conv2(x)))         # (batch, 128, 23)

        x = x.permute(0, 2, 1)                      # (batch, 23, 128) for LSTM
        out, _ = self.lstm(x)
        out = out[:, -1, :]                          # (batch, 128) last timestep

        out = torch.sigmoid(self.fc1(out))           # (batch, 64)
        out = self.fc2(out)                          # (batch, 256) logits
        return out


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    pml = total_correct / total_samples  # prediction probability
    return avg_loss, pml


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_byte_dataloaders(
        args.data,
        window_size=args.window,
        batch_size=args.batch_size,
    )

    model = CNN_LSTM(vocab_size=256, seq_len=args.window, lstm_hidden=128).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    pg = 1.0 / 256  # random guess probability
    print(f"Random guess baseline Pg = {pg:.4%}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
            pbar.set_postfix(loss=f"{epoch_loss/n_samples:.6f}")

        train_loss = epoch_loss / n_samples
        val_loss, val_pml = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pml"].append(val_pml)

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Pml: {val_pml:.4%} (Pg: {pg:.4%})",
            flush=True,
        )
        vessl.log(
            step=epoch + 1,
            payload={
                "cnn_lstm/train_loss": train_loss,
                "cnn_lstm/val_loss": val_loss,
                "cnn_lstm/pml": val_pml,
                "cnn_lstm/pml_over_pg": val_pml / pg,
                "cnn_lstm/lr": optimizer.param_groups[0]["lr"],
            },
        )

    # Save results
    plot_results(history, pg, args.rounds)
    torch.save(model.state_dict(), f"cnn_lstm_chacha{args.rounds}.pt")
    print(f"Model saved to cnn_lstm_chacha{args.rounds}.pt")


def plot_results(history, pg, rounds):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"CNN+LSTM - ChaCha{rounds} Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Pml curve
    epochs = range(1, len(history["val_pml"]) + 1)
    axes[1].plot(epochs, [p * 100 for p in history["val_pml"]], label="Pml", marker="o", markersize=3)
    axes[1].axhline(y=pg * 100, color="r", linestyle="--", label=f"Pg ({pg:.2%})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Prediction Probability (%)")
    axes[1].set_title(f"CNN+LSTM - ChaCha{rounds} Pml")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"cnn_lstm_chacha{rounds}_results.png", dpi=150)
    print(f"Plot saved to cnn_lstm_chacha{rounds}_results.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN+LSTM on ChaCha next-byte prediction")
    parser.add_argument("--data", default="dataset/chacha4_seq.bin")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--window", type=int, default=100, help="Input window size in bytes")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
