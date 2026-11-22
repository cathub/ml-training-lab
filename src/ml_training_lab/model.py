"""Biased matrix factorization with one logit per observed rating."""

from torch import nn


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.movie_embedding.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, users, movies):
        interaction = (self.user_embedding(users) * self.movie_embedding(movies)).sum(dim=-1)
        return interaction + self.user_bias(users).squeeze(-1) + self.movie_bias(movies).squeeze(-1)
