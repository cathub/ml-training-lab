"""Run the fixed Phase 2A protocol using only train and validation metrics."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from shutil import copy2

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import NUM_MOVIES, NUM_USERS, SEED, load_ratings, split_ratings, to_dataset
from .evaluate import evaluate
from .model import MatrixFactorization
from .train import plot_history, plt, save_json, train_epoch

CONFIG = {
    "learning_rate": 0.01, "batch_size": 1024, "epochs": 40, "patience": 5,
    "seed": SEED, "device": "cpu", "optimizer": "Adam", "weight_decay": 0.0,
    "loss": "BCEWithLogitsLoss", "torch_threads": 1,
}


class BiasOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_bias = nn.Embedding(NUM_USERS, 1)
        self.movie_bias = nn.Embedding(NUM_MOVIES, 1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, users, movies):
        return self.user_bias(users).squeeze(-1) + self.movie_bias(movies).squeeze(-1)


def verify_phase1(project, archive, baseline):
    """Fail closed if the archived sources or local Phase 1 artifacts differ."""
    evidence = project / "research-log/evidence/014-ml-training-lab-v1"
    manifest = {}
    for line in (evidence / "sources-and-local.sha256").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        manifest[path] = digest

    def verify(path, expected):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Phase 1 provenance mismatch for {path}")

    for name in ("data.py", "model.py", "train.py", "evaluate.py"):
        relative = f"src/ml_training_lab/{name}"
        verify(project / relative, manifest[f"ml-training-lab/{relative}"])
    verify(archive, manifest["ml-training-lab/data/ml-100k.zip"])
    verify(baseline / "best.pt", manifest["ml-training-lab/outputs/seed42/best.pt"])
    for line in (evidence / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if name in {"config.json", "history.json", "loss.png"}:
            verify(baseline / name, digest)
    historical = json.loads((baseline / "config.json").read_text())
    expected = {key: value for key, value in CONFIG.items() if key != "weight_decay"}
    if any(historical.get(key) != value for key, value in (expected | {"embedding_dim": 32}).items()):
        raise ValueError("Phase 1 hyperparameters differ from the ablation")


def fit(model, train, validation_loader, loss_fn, run_dir):
    train_loader = DataLoader(
        train, batch_size=CONFIG["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]
    )
    history = []
    best_loss, best_epoch = float("inf"), 0
    for epoch in range(1, CONFIG["epochs"] + 1):
        train_bce = train_epoch(model, train_loader, loss_fn, optimizer)
        val = evaluate(model, validation_loader, loss_fn)
        history.append({
            "epoch": epoch, "train_bce": train_bce,
            "validation_bce": val["bce"], "validation_roc_auc": val["roc_auc"],
        })
        save_json(run_dir / "history.json", history)
        if val["bce"] < best_loss:
            best_loss, best_epoch = val["bce"], epoch
            torch.save(model.state_dict(), run_dir / "best.pt")
        if epoch - best_epoch >= CONFIG["patience"]:
            break
    plot_history(history, best_epoch, run_dir / "loss.png")
    return history


def plot_comparison(histories, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for index, (key, label) in enumerate([
        ("train_bce", "Online training BCE"), ("validation_bce", "Validation BCE"),
        ("validation_roc_auc", "Validation ROC-AUC"),
    ]):
        ax = axes.flat[index]
        for name, history in histories.items():
            (line,) = ax.plot([row["epoch"] for row in history], [row[key] for row in history], label=name)
            selected = min(history, key=lambda row: row["validation_bce"])
            ax.scatter(selected["epoch"], selected[key], color=line.get_color(), s=20)
        ax.set(xlabel="Epoch", ylabel=label, title=label)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[3].axis("off")
    axes.flat[3].legend(handles, labels, title="Dots mark minimum validation BCE", loc="center")
    fig.suptitle("MovieLens 100K embedding dimension ablation · seed 42")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/ml-100k.zip"))
    parser.add_argument("--output", type=Path, default=Path("outputs/capacity-ablation"))
    parser.add_argument("--phase1", type=Path, default=Path("outputs/seed42"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    verify_phase1(project, args.data, args.phase1)
    args.output.mkdir(parents=True, exist_ok=False)

    torch.set_num_threads(CONFIG["torch_threads"])
    torch.use_deterministic_algorithms(True)
    train_frame, validation_frame, _ = split_ratings(load_ratings(args.data), SEED)
    train = to_dataset(train_frame)
    train_eval_loader = DataLoader(train, batch_size=CONFIG["batch_size"])
    validation_loader = DataLoader(to_dataset(validation_frame), batch_size=CONFIG["batch_size"])
    loss_fn = nn.BCEWithLogitsLoss()
    histories, results = {}, []
    for dim in (0, 2, 8, 32, 128):
        name = f"{dim}D" if dim else "Bias only"
        run_dir = args.output / (f"dim-{dim}" if dim else "bias-only")
        run_dir.mkdir()
        torch.manual_seed(SEED)
        model = MatrixFactorization(NUM_USERS, NUM_MOVIES, dim) if dim else BiasOnly()
        checkpoint = run_dir / "best.pt"
        if dim == 32:
            for artifact in ("history.json", "loss.png", "best.pt"):
                copy2(args.phase1 / artifact, run_dir / artifact)
            history = json.loads((run_dir / "history.json").read_text())
        else:
            history = fit(model, train, validation_loader, loss_fn, run_dir)
        histories[name] = history

        parameters = sum(parameter.numel() for parameter in model.parameters())
        save_json(run_dir / "config.json", CONFIG | {
            "embedding_dim": dim, "parameters": parameters, "data": str(args.data),
            "initialization": "N(0, 0.05^2) embeddings, zero biases" if dim else "zero biases",
            "reused_phase1": dim == 32,
            "test_evaluated_in_ablation": False,
        })
        selected = min(history, key=lambda row: row["validation_bce"])
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        train_bce = evaluate(model, train_eval_loader, loss_fn)["bce"]
        result = {
            "model": name, "parameters": parameters, "epochs_run": len(history),
            "selected_epoch": selected["epoch"], "checkpoint_train_bce": train_bce,
            "validation_bce": selected["validation_bce"],
            "validation_roc_auc": selected["validation_roc_auc"],
            "gap": selected["validation_bce"] - train_bce,
            "reused_phase1": dim == 32,
        }
        save_json(run_dir / "selection.json", result)
        print(json.dumps(result), flush=True)
        results.append(result)

    with (args.output / "comparison.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    plot_comparison(histories, args.output / "learning-curves.png")


if __name__ == "__main__":
    main()
