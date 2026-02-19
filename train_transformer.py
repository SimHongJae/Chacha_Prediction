import argparse
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from chacha_dataset import get_dataloaders


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=32, d_model=128, n_heads=4, n_layers=4, d_ff=512, max_len=512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, 32)
        seq_len = x.size(1)

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x, mask=mask, is_causal=True)
        # Use last position
        x = x[:, -1, :]  # (batch, d_model)
        return self.output_head(x)  # (batch, 32)


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


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # flatten=False -> shape (window_size, 32)
    train_loader, val_loader = get_dataloaders(
        args.data, window_size=args.window, batch_size=args.batch_size, flatten=False
    )

    model = TransformerPredictor(
        input_dim=32,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.window + 1,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = len(train_loader) * 2  # 2 epochs warmup
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

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

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Bit Acc: {val_bit_acc:.4f} | "
            f"Exact Acc: {val_exact_acc:.6f} | "
            f"LR: {current_lr:.2e}"
        )

    # Save results
    plot_results(history, per_bit_acc, "Transformer", args.rounds)
    torch.save(model.state_dict(), f"transformer_chacha{args.rounds}.pt")
    print(f"Model saved to transformer_chacha{args.rounds}.pt")


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
    parser = argparse.ArgumentParser(description="Train Transformer on ChaCha sequence prediction")
    parser.add_argument("--data", default="dataset/chacha4_seq.bin")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    args = parser.parse_args()
    train(args)
