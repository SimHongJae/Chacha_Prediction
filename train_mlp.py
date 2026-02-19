import argparse
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from chacha_dataset import get_dataloaders


class MLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dims=(256, 256, 128), output_dim=32):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct_bits = 0
    total_bits = 0
    total_exact = 0
    total_samples = 0
    bit_correct = np.zeros(32)
    bit_total = np.zeros(32)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct = (preds == y)
            total_correct_bits += correct.sum().item()
            total_bits += y.numel()
            total_exact += correct.all(dim=1).sum().item()
            total_samples += x.size(0)

            bit_correct += correct.sum(dim=0).cpu().numpy()
            bit_total += x.size(0)

    avg_loss = total_loss / total_samples
    bit_acc = total_correct_bits / total_bits
    exact_acc = total_exact / total_samples
    per_bit_acc = bit_correct / bit_total
    return avg_loss, bit_acc, exact_acc, per_bit_acc


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(
        args.data, window_size=args.window, batch_size=args.batch_size, flatten=True
    )

    input_dim = args.window * 32
    model = MLP(input_dim=input_dim).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "val_bit_acc": [], "val_exact_acc": []}

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
        val_loss, val_bit_acc, val_exact_acc, per_bit_acc = evaluate(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_bit_acc"].append(val_bit_acc)
        history["val_exact_acc"].append(val_exact_acc)

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Bit Acc: {val_bit_acc:.4f} | "
            f"Exact Acc: {val_exact_acc:.6f}",
            flush=True,
        )

    # Save results
    plot_results(history, per_bit_acc, "MLP", args.rounds)
    torch.save(model.state_dict(), f"mlp_chacha{args.rounds}.pt")
    print(f"Model saved to mlp_chacha{args.rounds}.pt")


def plot_results(history, per_bit_acc, model_name, rounds):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curve
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} - ChaCha{rounds} Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Bit accuracy curve
    axes[1].plot(history["val_bit_acc"], label="Bit Acc")
    axes[1].axhline(y=0.5, color="r", linestyle="--", label="Random (50%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_name} - Bit Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    # Per-bit accuracy heatmap
    bit_acc_2d = per_bit_acc.reshape(4, 8)  # 32 bits as 4x8
    im = axes[2].imshow(bit_acc_2d, cmap="RdYlGn", vmin=0.45, vmax=0.55, aspect="auto")
    axes[2].set_title(f"{model_name} - Per-Bit Accuracy")
    axes[2].set_xlabel("Bit (within byte)")
    axes[2].set_ylabel("Byte")
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig(f"{model_name.lower()}_chacha{rounds}_results.png", dpi=150)
    print(f"Plot saved to {model_name.lower()}_chacha{rounds}_results.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on ChaCha sequence prediction")
    parser.add_argument("--data", default="dataset/chacha4_seq.bin", help="Path to binary data file")
    parser.add_argument("--rounds", type=int, default=4, help="ChaCha round count (for labeling)")
    parser.add_argument("--window", type=int, default=4, help="Sliding window size")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
