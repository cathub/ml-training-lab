"""Score frozen MovieLens checkpoints by the user's training history length."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from .data import NUM_MOVIES, NUM_USERS, SEED, load_ratings, split_ratings, to_dataset
from .model import MatrixFactorization
from .train import plt, save_json

# Merge an undersized bin into its right neighbor, or left at the upper edge.
MIN_USERS = 30
MIN_EXAMPLES = 100


def history_bins(users, validation):
    bins = [(0, 20), (21, 50), (51, 100), (101, 200), (201, np.inf)]
    index = 0
    while len(bins) > 1 and index < len(bins):
        lower, upper = bins[index]
        user_count = users["training_ratings"].between(lower, upper).sum()
        example_count = validation["training_ratings"].between(lower, upper).sum()
        if user_count >= MIN_USERS and example_count >= MIN_EXAMPLES:
            index += 1
        elif index + 1 < len(bins):
            bins[index:index + 2] = [(lower, bins[index + 1][1])]
        else:
            bins[index - 1:] = [(bins[index - 1][0], upper)]
            break
    return bins


@torch.inference_mode()
def validation_scores(model, validation):
    users, movies, labels = to_dataset(validation).tensors
    logits = model(users, movies)
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return validation.assign(logit=logits.numpy(), bce=bce.numpy().astype("float64"))


def summarize_bins(dim, scores, users, bins):
    rows = []
    for lower, upper in bins:
        group = scores.loc[scores["training_ratings"].between(lower, upper)]
        members = users.loc[users["training_ratings"].between(lower, upper)]
        auc = roc_auc_score(group["label"], group["logit"]) if group["label"].nunique() == 2 else None
        rows.append({
            "embedding_dim": dim,
            "history_bin": f"{lower}+" if np.isinf(upper) else f"{lower}–{upper}",
            "median_training_ratings": members["training_ratings"].median(),
            "users": len(members),
            "users_with_validation": group["user"].nunique(),
            "validation_examples": len(group),
            "validation_positives": int(group["label"].sum()),
            "validation_bce": group["bce"].mean(),
            "user_mean_validation_bce": group.groupby("user")["bce"].mean().mean(),
            "validation_roc_auc": auc,
        })
    return rows


def rank_diagnostic(model, dim, user, train, validation):
    weights = model.movie_embedding.weight.detach().numpy().astype("float64")
    train_ids = train.loc[train["user"] == user, "movie"].sort_values().to_numpy()
    validation_ids = validation.loc[validation["user"] == user, "movie"].to_numpy()
    q = weights[train_ids]
    _, singular_values, vh = np.linalg.svd(q, full_matrices=True)
    tolerance = max(q.shape) * np.finfo(np.float64).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    direction = vh[rank] if rank < dim else None
    return {
        "movielens_user_id": user + 1,
        "movielens_training_movie_ids": (train_ids + 1).tolist(),
        "rows": len(q),
        "embedding_dim": dim,
        "rank": rank,
        "nullity": dim - rank,
        "tolerance": tolerance,
        "singular_values": singular_values.tolist(),
        "max_abs_training_logit_change_for_unit_null_direction": (
            float(np.max(np.abs(q @ direction))) if direction is not None else None
        ),
        "max_abs_validation_logit_change_for_unit_null_direction": (
            float(np.max(np.abs(weights[validation_ids] @ direction)))
            if direction is not None and len(validation_ids) else None
        ),
    }


def plot_history_length(summary, path):
    fig, (bce, delta) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    baseline = summary.loc[summary["embedding_dim"] == 32, "validation_bce"].to_numpy()
    for dim, group in summary.groupby("embedding_dim"):
        x = group["median_training_ratings"]
        bce.plot(x, group["validation_bce"], marker="o", label=f"{dim}D")
        delta.plot(x, group["validation_bce"].to_numpy() - baseline, marker="o", label=f"{dim}D")
    delta.axhline(0, color="0.65", linewidth=0.8)
    bce.set(ylabel="Validation BCE per rating", title="Validation loss by history bin")
    delta.set(ylabel="Validation BCE minus 32D BCE", title="Difference on the same examples")
    for ax in (bce, delta):
        ax.set(xlabel="Median training ratings per user in bin", xscale="log")
    bce.legend()
    fig.suptitle("MovieLens 100K · seed-42 split · frozen checkpoints")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/ml-100k.zip"))
    parser.add_argument("--baseline", type=Path, default=Path("outputs/seed42"))
    parser.add_argument("--ablation", type=Path, default=Path("outputs/capacity-ablation"))
    parser.add_argument("--output", type=Path, default=Path("outputs/history-analysis"))
    args = parser.parse_args()

    torch.set_num_threads(1)
    train, validation, _ = split_ratings(load_ratings(args.data), SEED)
    counts = train.groupby("user").size().reindex(range(NUM_USERS), fill_value=0)
    users = counts.rename("training_ratings").rename_axis("user").reset_index()
    validation = validation.assign(training_ratings=validation["user"].map(counts))
    bins = history_bins(users, validation)
    diagnostic_user = int(counts.idxmin())

    checkpoints = {32: args.baseline}
    for dim in (2, 8, 128):
        directory = args.ablation / f"dim-{dim}"
        if (directory / "best.pt").is_file():
            checkpoints[dim] = directory

    rows, diagnostics, sources = [], {}, {}
    for dim, directory in sorted(checkpoints.items()):
        config = json.loads((directory / "config.json").read_text())
        if config["embedding_dim"] != dim or config["seed"] != SEED:
            raise ValueError(f"Expected {dim}D and seed {SEED} in {directory}")
        history = json.loads((directory / "history.json").read_text())
        selected = min(history, key=lambda row: row["validation_bce"])
        checkpoint = directory / "best.pt"
        model = MatrixFactorization(NUM_USERS, NUM_MOVIES, dim)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        scores = validation_scores(model, validation)
        if not np.isclose(scores["bce"].mean(), selected["validation_bce"], atol=1e-7, rtol=0):
            raise ValueError(f"Validation BCE disagrees with the recorded minimum in {directory}")
        rows.extend(summarize_bins(dim, scores, users, bins))
        diagnostics[str(dim)] = rank_diagnostic(model, dim, diagnostic_user, train, validation)
        sources[str(dim)] = {"checkpoint": str(checkpoint), "selected_epoch": selected["epoch"]}

    args.output.mkdir(parents=True, exist_ok=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "by-history.csv", index=False)
    users.assign(movielens_user_id=users["user"] + 1).drop(columns="user").to_csv(
        args.output / "user-training-counts.csv", index=False
    )
    save_json(args.output / "rank-diagnostic.json", {
        "user_selection": "Fewest training ratings, breaking ties by lowest MovieLens user ID",
        "tolerance_rule": "max(Q.shape) * float64 epsilon * largest singular value",
        "models": diagnostics,
    })
    save_json(args.output / "config.json", {
        "data": str(args.data), "seed": SEED, "minimum_users_per_bin": MIN_USERS,
        "minimum_validation_examples_per_bin": MIN_EXAMPLES, "checkpoints": sources,
    })
    plot_history_length(summary, args.output / "validation-by-history.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
