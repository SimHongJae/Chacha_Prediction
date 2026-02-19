"""CNN+LSTM block-level prediction model.

Two modes:
  --mode block64 : 64-byte block -> next 64-byte block (64 x 256-class)
  --mode byte1   : 64-byte block -> next 1 byte (256-class)

CNN: Conv1d(256,64,k=5) -> Pool(2) -> Conv1d(64,128,k=3) -> Pool(2)
LSTM: hidden=128
Metric: byte accuracy (baseline Pg = 1/256 = 0.39%)
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
from chacha_dataset import get_block_dataloaders


BLOCK_BYTES = 64


class CNN_LSTM(nn.Module):
    def __init__(self, vocab_size=256, block_size=64, lstm_hidden=128, output_bytes=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.output_bytes = output_bytes  # 64 or 1

        # CNN feature extractor
        self.conv1 = nn.Conv1d(vocab_size, 64, kernel_size=5)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.pool = nn.MaxPool1d(2)

        # After conv1: 64-4=60, pool: 30
        # After conv2: 30-2=28, pool: 14
        self.lstm = nn.LSTM(128, lstm_hidden, batch_first=True)

        # Output head
        self.fc1 = nn.Linear(lstm_hidden, 256)
        self.fc2 = nn.Linear(256, output_bytes * vocab_size)

    def forward(self, x):
        # x: (batch, 64) long tensor of byte values 0-255
        x = F.one_hot(x, self.vocab_size).float()  # (batch, 64, 256)
        x = x.permute(0, 2, 1)                     # (batch, 256, 64)

        x = self.pool(F.relu(self.conv1(x)))        # (batch, 64, 30)
        x = self.pool(F.relu(self.conv2(x)))        # (batch, 128, 14)

        x = x.permute(0, 2, 1)                     # (batch, 14, 128)
        out, _ = self.lstm(x)
        out = out[:, -1, :]                         # (batch, 128)

        out = F.relu(self.fc1(out))                 # (batch, 256)
        out = self.fc2(out)                         # (batch, output_bytes*256)

        if self.output_bytes == 1:
            return out                              # (batch, 256)
        else:
            return out.view(-1, self.output_bytes, self.vocab_size)  # (batch, 64, 256)


# ── block64 mode: predict next 64 bytes ──────────────────────────────

def evaluate_block64(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_bytes = 0
    total_exact = 0
    total_blocks = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)                              # (batch, 64, 256)
            loss = criterion(logits.view(-1, 256), y.view(-1))
            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=2)                   # (batch, 64)
            correct = (preds == y)
            total_correct += correct.sum().item()
            total_bytes += y.numel()
            total_exact += correct.all(dim=1).sum().item()
            total_blocks += x.size(0)

    return {
        "loss": total_loss / total_blocks,
        "byte_acc": total_correct / total_bytes,
        "exact_acc": total_exact / total_blocks,
    }


def train_block64(model, train_loader, val_loader, optimizer, criterion, device, args):
    pg = 1.0 / 256
    history = {"train_loss": [], "val_loss": [], "val_byte_acc": [], "val_exact_acc": []}

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", file=sys.stdout)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, 256), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
            pbar.set_postfix(loss=f"{epoch_loss/n_samples:.6f}")

        train_loss = epoch_loss / n_samples
        metrics = evaluate_block64(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(metrics["loss"])
        history["val_byte_acc"].append(metrics["byte_acc"])
        history["val_exact_acc"].append(metrics["exact_acc"])

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {metrics['loss']:.6f} | "
            f"Byte Acc: {metrics['byte_acc']:.4%} (Pg: {pg:.4%}) | "
            f"Exact Match: {metrics['exact_acc']:.6%}",
            flush=True,
        )
        vessl.log(
            step=epoch + 1,
            payload={
                "block64/train_loss": train_loss,
                "block64/val_loss": metrics["loss"],
                "block64/byte_acc": metrics["byte_acc"],
                "block64/byte_acc_over_pg": metrics["byte_acc"] / pg,
                "block64/exact_match": metrics["exact_acc"],
                "block64/lr": optimizer.param_groups[0]["lr"],
            },
        )

    return history


# ── byte1 mode: predict next 1 byte ──────────────────────────────────

def evaluate_byte1(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)                              # (batch, 256)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

    return {
        "loss": total_loss / total_samples,
        "pml": total_correct / total_samples,
    }


def train_byte1(model, train_loader, val_loader, optimizer, criterion, device, args):
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
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
            pbar.set_postfix(loss=f"{epoch_loss/n_samples:.6f}")

        train_loss = epoch_loss / n_samples
        metrics = evaluate_byte1(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(metrics["loss"])
        history["val_pml"].append(metrics["pml"])

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {metrics['loss']:.6f} | "
            f"Pml: {metrics['pml']:.4%} (Pg: {pg:.4%})",
            flush=True,
        )
        vessl.log(
            step=epoch + 1,
            payload={
                "byte1/train_loss": train_loss,
                "byte1/val_loss": metrics["loss"],
                "byte1/pml": metrics["pml"],
                "byte1/pml_over_pg": metrics["pml"] / pg,
                "byte1/lr": optimizer.param_groups[0]["lr"],
            },
        )

    return history


# ── Plotting ──────────────────────────────────────────────────────────

def plot_results(history, mode, pg, rounds):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"CNN+LSTM ({mode}) - ChaCha{rounds} Loss")
    axes[0].legend()
    axes[0].grid(True)

    epochs = range(1, len(history["val_loss"]) + 1)
    if mode == "block64":
        acc_data = [p * 100 for p in history["val_byte_acc"]]
        acc_label = "Byte Acc"
    else:
        acc_data = [p * 100 for p in history["val_pml"]]
        acc_label = "Pml"

    axes[1].plot(epochs, acc_data, label=acc_label, marker="o", markersize=3)
    axes[1].axhline(y=pg * 100, color="r", linestyle="--", label=f"Pg ({pg:.2%})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"CNN+LSTM ({mode}) - ChaCha{rounds}")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    fname = f"cnn_lstm_{mode}_chacha{rounds}_results.png"
    plt.savefig(fname, dpi=150)
    print(f"Plot saved to {fname}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")

    output_bytes = BLOCK_BYTES if args.mode == "block64" else 1

    train_loader, val_loader = get_block_dataloaders(
        args.data,
        mode=args.mode,
        batch_size=args.batch_size,
    )

    model = CNN_LSTM(
        vocab_size=256, block_size=BLOCK_BYTES,
        lstm_hidden=128, output_bytes=output_bytes,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    pg = 1.0 / 256
    print(f"Random guess baseline Pg = {pg:.4%}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == "block64":
        history = train_block64(model, train_loader, val_loader, optimizer, criterion, device, args)
    else:
        history = train_byte1(model, train_loader, val_loader, optimizer, criterion, device, args)

    plot_results(history, args.mode, pg, args.rounds)

    model_fname = f"cnn_lstm_{args.mode}_chacha{args.rounds}.pt"
    torch.save(model.state_dict(), model_fname)
    print(f"Model saved to {model_fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN+LSTM on ChaCha block prediction")
    parser.add_argument("--data", default="dataset/chacha4_seq.bin")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--mode", choices=["block64", "byte1"], default="block64",
                        help="block64: predict next 64-byte block, byte1: predict next 1 byte")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
