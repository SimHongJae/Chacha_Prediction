"""ChaCha function learning via counter/nonce sweep.

Modes (--mode):
  counter : input=counter(4 bytes), output=block(64 bytes)
            data: chacha{R}_stream.bin  (flat, block N -> counter N)
  nonce   : input=nonce(12 bytes),   output=block(64 bytes)
            data: chacha{R}_nonce_sweep.bin  (records of 76 bytes)

Model: byte Embedding -> MLP -> 64 x 256 logits
Loss : CrossEntropy averaged over 64 output positions
Metric: per-byte Pml vs Pg = 1/256 ≈ 0.39%

Train/test split is positional (counter mode) or random-index (nonce mode),
so test counters/nonces are never seen during training.
"""

import argparse
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from chacha_dataset import get_counter_dataloaders, get_nonce_dataloaders

# Optional vessl logging (gracefully disabled when not installed)
try:
    import vessl
    _VESSL = True
except ImportError:
    _VESSL = False

BLOCK_SIZE = 64
VOCAB_SIZE = 256


# ── Model ─────────────────────────────────────────────────────────────

class SweepMLP(nn.Module):
    """Byte Embedding -> MLP -> 64 x 256 logits.

    Each input byte is embedded independently, embeddings are concatenated,
    then passed through an MLP to predict all 64 output bytes at once.

    Args:
        input_bytes:  4 (counter mode) or 12 (nonce mode)
        embed_dim:    embedding dimension per byte
        hidden_dims:  MLP hidden layer sizes
    """

    def __init__(
        self,
        input_bytes: int,
        embed_dim: int = 32,
        hidden_dims: tuple = (512, 512, 512),
    ):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim)
        in_dim = input_bytes * embed_dim
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, BLOCK_SIZE * VOCAB_SIZE))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, input_bytes) long
        emb = self.embedding(x)                          # (batch, input_bytes, embed_dim)
        emb = emb.flatten(1)                             # (batch, input_bytes * embed_dim)
        out = self.mlp(emb)                              # (batch, 64 * 256)
        return out.view(out.size(0), BLOCK_SIZE, VOCAB_SIZE)  # (batch, 64, 256)


# ── Evaluate ──────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_bytes = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)                         # (batch, 64, 256)
            loss = criterion(logits.permute(0, 2, 1), y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=2)              # (batch, 64)
            total_correct += (preds == y).sum().item()
            total_bytes += x.size(0) * BLOCK_SIZE

    return {
        "loss": total_loss / (total_bytes // BLOCK_SIZE),
        "pml":  total_correct / total_bytes,
    }


# ── Train ─────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, optimizer, scheduler, device, args):
    pg = 1.0 / VOCAB_SIZE
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_pml": []}

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", file=sys.stdout)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)                         # (batch, 64, 256)
            loss = criterion(logits.permute(0, 2, 1), y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
            pbar.set_postfix(loss=f"{epoch_loss/n_samples:.6f}")

        train_loss = epoch_loss / n_samples
        metrics = evaluate(model, val_loader, device)

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
    pg = 1.0 / VOCAB_SIZE
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title(f"MLP {args.mode}-sweep - ChaCha{args.rounds}")
    axes[0].legend()
    axes[0].grid(True)

    epochs = range(1, len(history["val_pml"]) + 1)
    pml_pct = [p * 100 for p in history["val_pml"]]
    axes[1].plot(epochs, pml_pct, label="Pml", marker="o", markersize=3)
    axes[1].axhline(y=pg * 100, color="r", linestyle="--", label=f"Pg ({pg:.2%})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Per-byte Accuracy (%)")
    axes[1].set_title(f"MLP {args.mode}-sweep - ChaCha{args.rounds}")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    fname = f"sweep_{args.mode}_chacha{args.rounds}_results.png"
    plt.savefig(fname, dpi=150)
    print(f"Plot saved to {fname}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}-sweep  |  ChaCha rounds: {args.rounds}")

    if args.mode == "counter":
        input_bytes = 4
        train_loader, val_loader = get_counter_dataloaders(
            args.data, batch_size=args.batch_size, num_workers=args.num_workers,
        )
    else:  # nonce
        input_bytes = 12
        train_loader, val_loader = get_nonce_dataloaders(
            args.data, batch_size=args.batch_size, num_workers=args.num_workers,
        )

    model = SweepMLP(
        input_bytes=input_bytes,
        embed_dim=args.embed_dim,
        hidden_dims=tuple(args.hidden_dims),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Input bytes: {input_bytes}  |  Random guess Pg = {1/VOCAB_SIZE:.4%}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    history = train(model, train_loader, val_loader, optimizer, scheduler, device, args)
    plot_results(history, args)

    fname = f"sweep_{args.mode}_chacha{args.rounds}.pt"
    torch.save(model.state_dict(), fname)
    print(f"Model saved to {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learn ChaCha block function via counter or nonce sweep"
    )
    parser.add_argument("--mode", choices=["counter", "nonce"], default="counter",
                        help="counter: vary counter(4B) | nonce: vary nonce(12B)")
    parser.add_argument("--data", default=None,
                        help="Path to data file (auto-detected if omitted)")
    parser.add_argument("--rounds", type=int, default=4)

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 512, 512])

    args = parser.parse_args()

    if args.data is None:
        if args.mode == "counter":
            args.data = f"dataset/chacha{args.rounds}_stream.bin"
        else:
            args.data = f"dataset/chacha{args.rounds}_nonce_sweep.bin"

    main(args)
