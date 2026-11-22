import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml_training_lab.data import split_ratings
from ml_training_lab.evaluate import evaluate
from ml_training_lab.model import MatrixFactorization


def test_logits_include_dot_product_and_both_biases():
    model = MatrixFactorization(2, 2, embedding_dim=2)
    with torch.no_grad():
        model.user_embedding.weight.copy_(torch.tensor([[1., 2.], [2., 3.]]))
        model.movie_embedding.weight.copy_(torch.tensor([[3., 4.], [4., 5.]]))
        model.user_bias.weight.copy_(torch.tensor([[1.], [2.]]))
        model.movie_bias.weight.copy_(torch.tensor([[4.], [8.]]))
    torch.testing.assert_close(
        model(torch.tensor([0, 1]), torch.tensor([1, 0])), torch.tensor([23., 24.])
    )
    assert model(torch.tensor([0]), torch.tensor([0])).shape == (1,)


def test_single_class_auc_is_undefined():
    model = MatrixFactorization(1, 1)
    ids = torch.zeros(3, dtype=torch.long)
    loader = DataLoader(TensorDataset(ids, ids, torch.ones(3)), batch_size=2)
    result = evaluate(model, loader, nn.BCEWithLogitsLoss())
    assert result["roc_auc"] is None
    assert result["bce"] > 0


def test_per_user_split_is_complete_disjoint_and_deterministic():
    frame = pd.DataFrame([(u, i) for u in range(2) for i in range(20)], columns=["user", "movie"])
    splits = split_ratings(frame)
    for split, repeated, count in zip(splits, split_ratings(frame), [14, 3, 3]):
        pd.testing.assert_frame_equal(split, repeated)
        assert split.groupby("user").size().tolist() == [count, count]
    combined = pd.concat(splits)
    assert len(combined) == len(frame)
    assert not combined.duplicated(["user", "movie"]).any()
