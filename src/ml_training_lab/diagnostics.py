"""Compare autograd with the derivative of BCE for one observed rating."""

import torch
from torch.nn import functional as F

from .data import SEED
from .model import MatrixFactorization


def gradient_step():
    model = MatrixFactorization(1, 1, embedding_dim=2).double()
    users = movies = torch.tensor([0])
    label = torch.tensor([1.0], dtype=torch.float64)
    with torch.no_grad():
        model.user_embedding.weight[0] = torch.tensor([0.25, -0.5])
        model.movie_embedding.weight[0] = torch.tensor([0.5, 0.25])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    logits = model(users, movies)
    F.binary_cross_entropy_with_logits(logits, label).backward()

    # For BCE with logits, dL/dz = sigmoid(z) - y.
    # The dot product contributes dz/du = v and dz/dv = u.
    user = model.user_embedding.weight[0].detach().clone()
    movie = model.movie_embedding.weight[0].detach().clone()
    dloss_dlogit = (logits.detach().sigmoid() - label).item()
    gradients = {}
    for name, weight, before, partner in [
        ("user", model.user_embedding.weight, user, movie),
        ("movie", model.movie_embedding.weight, movie, user),
    ]:
        actual = weight.grad[0].clone()
        expected = dloss_dlogit * partner
        if not torch.isfinite(actual).all() or torch.count_nonzero(actual) == 0:
            raise ValueError(f"Expected a finite nonzero {name} gradient")
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
        gradients[name] = {"before": before, "autograd": actual, "analytical": expected}

    optimizer.step()
    gradients["user"]["after"] = model.user_embedding.weight[0].detach().clone()
    gradients["movie"]["after"] = model.movie_embedding.weight[0].detach().clone()
    return gradients


def main():
    torch.manual_seed(SEED)
    for name, result in gradient_step().items():
        print(name)
        for field, value in result.items():
            print(f"  {field:10} {value.tolist()}")
    print("Both embedding gradients are finite, nonzero, and match the analytical values.")


if __name__ == "__main__":
    main()
