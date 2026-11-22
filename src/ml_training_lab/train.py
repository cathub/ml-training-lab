"""Train on CPU, select on validation BCE, and score the test set once."""

import argparse
import json
from pathlib import Path

import matplotlib
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import NUM_MOVIES, NUM_USERS, SEED, load_ratings, split_ratings, split_stats, to_dataset
from .evaluate import evaluate
from .model import MatrixFactorization

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def train_epoch(model, loader, loss_fn, optimizer):
    model.train()
    loss_sum = 0.0
    for users, movies, targets in loader:
        optimizer.zero_grad()
        logits = model(users, movies)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * targets.numel()
    return loss_sum / len(loader.dataset)


def plot_history(history, best_epoch, path):
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(epochs, [row["train_bce"] for row in history], label="Train")
    ax.plot(epochs, [row["validation_bce"] for row in history], label="Validation")
    ax.axvline(best_epoch, color="0.5", linestyle="--", label=f"Selected epoch {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="Mean BCE", title="MovieLens 100K matrix factorization")
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/ml-100k.zip"))
    parser.add_argument("--output", type=Path, default=Path("outputs/seed42"))
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if min(args.embedding_dim, args.batch_size, args.epochs, args.patience) < 1:
        parser.error("embedding dimension, batch size, epochs and patience must be positive")
    if not 0 < args.learning_rate < float("inf"):
        parser.error("learning rate must be finite and positive")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    ratings = load_ratings(args.data)
    train, validation, test = split_ratings(ratings, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    config = vars(args) | {
        "data": str(args.data), "output": str(args.output), "device": "cpu",
        "optimizer": "Adam", "loss": "BCEWithLogitsLoss", "torch_threads": 1,
    }
    save_json(args.output / "config.json", config)
    train_loader = DataLoader(
        to_dataset(train), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(to_dataset(validation), batch_size=args.batch_size)
    model = MatrixFactorization(NUM_USERS, NUM_MOVIES, args.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    checkpoint = args.output / "best.pt"
    best_loss, best_epoch = float("inf"), 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_bce = train_epoch(model, train_loader, loss_fn, optimizer)
        val = evaluate(model, validation_loader, loss_fn)
        row = {
            "epoch": epoch, "train_bce": train_bce,
            "validation_bce": val["bce"], "validation_roc_auc": val["roc_auc"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        save_json(args.output / "history.json", history)
        if val["bce"] < best_loss:
            best_loss, best_epoch = val["bce"], epoch
            torch.save(model.state_dict(), checkpoint)
        if epoch - best_epoch >= args.patience:
            break

    plot_history(history, best_epoch, args.output / "loss.png")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    test_loader = DataLoader(to_dataset(test), batch_size=args.batch_size)
    report = {
        "selected_epoch": best_epoch,
        "selected_validation": history[best_epoch - 1],
        "test": evaluate(model, test_loader, loss_fn),
        "dataset": {
            "ratings": len(ratings), "users": NUM_USERS, "movies": NUM_MOVIES,
            "splits": {
                name: split_stats(frame, train)
                for name, frame in [("train", train), ("validation", validation), ("test", test)]
            },
        },
    }
    save_json(args.output / "metrics.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
