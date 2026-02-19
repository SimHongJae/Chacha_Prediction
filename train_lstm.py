import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from chacha_dataset import get_dataloaders


class LSTMModel(nn.Module):
    def __init__(self, input_dim=32, hidden_size=128, num_layers=2, output_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, 32)
        out, _ = self.lstm(x)
        # Use last timestep
        out = out[:, -1, :]  # (batch, hidden_size)
        return self.fc(out)


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

    # flatten=False -> shape (window_size, 32) for LSTM
    train_loader, val_loader = get_dataloaders(
        args.data, window_size=args.window, batch_size=args.batch_size, flatten=False
    )

    model = LSTMModel(input_dim=32, hidden_size=args.hidden, num_layers=args.layers).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "val_bit_acc": [], "val_exact_acc": []}

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_samples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            n_samples += x.size(0)

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
            f"Exact Acc: {val_exact_acc:.6f}"
        )

    # Save results
    plot_results(history, per_bit_acc, "LSTM", args.rounds)
    torch.save(model.state_dict(), f"lstm_chacha{args.rounds}.pt")
    print(f"Model saved to lstm_chacha{args.rounds}.pt")


def plot_results(history, per_bit_acc, model_name, rounds):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} - ChaCha{rounds} Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["val_bit_acc"], label="Bit Acc")
    axes[1].axhline(y=0.5, color="r", linestyle="--", label="Random (50%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_name} - Bit Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    bit_acc_2d = per_bit_acc.reshape(4, 8)
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
    parser = argparse.ArgumentParser(description="Train LSTM on ChaCha sequence prediction")
    parser.add_argument("--data", default="dataset/chacha4_seq.bin")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()
    train(args)
