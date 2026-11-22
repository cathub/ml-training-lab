"""Read the official archive and split observed ratings by user."""

from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

NUM_USERS = 943
NUM_MOVIES = 1682
SEED = 42


def load_ratings(archive):
    with ZipFile(archive) as source, source.open("ml-100k/u.data") as ratings:
        frame = pd.read_csv(
            ratings, sep="\t", header=None, names=["user", "movie", "rating", "timestamp"]
        )
    if (
        len(frame) != 100_000
        or frame["user"].nunique() != NUM_USERS
        or frame["movie"].nunique() != NUM_MOVIES
        or not frame["user"].between(1, NUM_USERS).all()
        or not frame["movie"].between(1, NUM_MOVIES).all()
        or not frame["rating"].isin([1, 2, 3, 4, 5]).all()
        or frame.duplicated(["user", "movie"]).any()
    ):
        raise ValueError("Expected the original MovieLens 100K observed ratings")
    # The published ID ranges define the embedding slots. No mapping is fitted.
    frame["user"] -= 1
    frame["movie"] -= 1
    frame["label"] = (frame["rating"] >= 4).astype("float32")
    return frame


def split_ratings(frame, seed=SEED):
    rng = np.random.default_rng(seed)
    indices = [[], [], []]
    for positions in frame.groupby("user", sort=True).indices.values():
        shuffled = rng.permutation(positions)
        train_end = len(shuffled) * 70 // 100
        validation_end = len(shuffled) * 85 // 100
        for split, part in zip(indices, np.split(shuffled, [train_end, validation_end])):
            split.extend(part)
    return tuple(frame.iloc[split].reset_index(drop=True) for split in indices)


def to_dataset(frame):
    return TensorDataset(
        torch.tensor(frame["user"].to_numpy(), dtype=torch.long),
        torch.tensor(frame["movie"].to_numpy(), dtype=torch.long),
        torch.tensor(frame["label"].to_numpy(), dtype=torch.float32),
    )


def split_stats(frame, train):
    positives = int(frame["label"].sum())
    return {
        "ratings": len(frame),
        "users": int(frame["user"].nunique()),
        "movies": int(frame["movie"].nunique()),
        "positive": positives,
        "negative": len(frame) - positives,
        "positive_rate": positives / len(frame),
        "ratings_with_unseen_movie": int((~frame["movie"].isin(train["movie"])).sum()),
    }
