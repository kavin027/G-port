from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class SparseRidgeProblem:
    x: sparse.csr_matrix
    y: np.ndarray
    shard_slices: list[slice]
    l2: float

    @property
    def n_samples(self) -> int:
        return self.x.shape[0]

    @property
    def n_features(self) -> int:
        return self.x.shape[1]

    @property
    def n_shards(self) -> int:
        return len(self.shard_slices)

    def shard_costs(self) -> np.ndarray:
        costs = []
        for shard_slice in self.shard_slices:
            costs.append(float(self.x[shard_slice].nnz))
        costs_array = np.asarray(costs, dtype=float)
        return costs_array / np.mean(costs_array)

    def shard_gradients(self, weights: np.ndarray) -> np.ndarray:
        gradients = np.zeros((self.n_shards, self.n_features), dtype=float)
        scale = 1.0 / self.n_samples
        for shard_id, shard_slice in enumerate(self.shard_slices):
            x_shard = self.x[shard_slice]
            residual = x_shard @ weights - self.y[shard_slice]
            gradients[shard_id] = np.asarray(x_shard.T @ residual).ravel() * scale
        return gradients

    def full_gradient(self, weights: np.ndarray) -> np.ndarray:
        return self.shard_gradients(weights).sum(axis=0) + self.l2 * weights

    def loss(self, weights: np.ndarray) -> float:
        residual = self.x @ weights - self.y
        data_loss = 0.5 * float(np.dot(residual, residual)) / self.n_samples
        reg_loss = 0.5 * self.l2 * float(np.dot(weights, weights))
        return data_loss + reg_loss


def make_sparse_ridge_problem(
    n_samples: int,
    n_features: int,
    density: float,
    n_shards: int,
    l2: float,
    seed: int,
) -> SparseRidgeProblem:
    rng = np.random.default_rng(seed)
    entry_scale = 1.0 / np.sqrt(max(1.0, n_features * density))

    x = sparse.random(
        n_samples,
        n_features,
        density=density,
        format="csr",
        random_state=seed,
        data_rvs=lambda size: rng.normal(0.0, entry_scale, size=size),
    )

    true_weights = np.zeros(n_features, dtype=float)
    active = rng.choice(n_features, size=max(1, n_features // 20), replace=False)
    true_weights[active] = rng.normal(0.0, 1.0, size=active.size)
    y = np.asarray(x @ true_weights).ravel() + rng.normal(0.0, 0.05, size=n_samples)

    boundaries = np.linspace(0, n_samples, n_shards + 1, dtype=int)
    shard_slices = [
        slice(int(boundaries[i]), int(boundaries[i + 1])) for i in range(n_shards)
    ]
    return SparseRidgeProblem(x=x, y=y, shard_slices=shard_slices, l2=l2)


@dataclass
class SparseEmbeddingProblem:
    users: np.ndarray
    items: np.ndarray
    y: np.ndarray
    x: sparse.csr_matrix
    shard_slices: list[slice]
    sample_costs: np.ndarray
    n_users: int
    n_items: int
    embedding_dim: int
    l2: float

    @property
    def n_samples(self) -> int:
        return self.users.size

    @property
    def n_features(self) -> int:
        return (self.n_users + self.n_items) * self.embedding_dim

    @property
    def n_shards(self) -> int:
        return len(self.shard_slices)

    def shard_costs(self) -> np.ndarray:
        costs = []
        for shard_slice in self.shard_slices:
            shard_users = np.unique(self.users[shard_slice]).size
            shard_items = np.unique(self.items[shard_slice]).size
            touch_cost = 0.20 * (shard_users + shard_items)
            costs.append(float(self.sample_costs[shard_slice].sum() + touch_cost))
        costs_array = np.asarray(costs, dtype=float)
        return costs_array / max(float(np.mean(costs_array)), 1e-12)

    def shard_gradients(self, weights: np.ndarray) -> np.ndarray:
        gradients = np.zeros((self.n_shards, self.n_features), dtype=float)
        user_weights, item_weights = self._split(weights)
        scale = 1.0 / self.n_samples
        for shard_id, shard_slice in enumerate(self.shard_slices):
            shard_users = self.users[shard_slice]
            shard_items = self.items[shard_slice]
            user_vecs = user_weights[shard_users]
            item_vecs = item_weights[shard_items]
            residual = np.einsum("ij,ij->i", user_vecs, item_vecs) - self.y[shard_slice]
            user_grad = np.zeros_like(user_weights)
            item_grad = np.zeros_like(item_weights)
            np.add.at(user_grad, shard_users, residual[:, None] * item_vecs * scale)
            np.add.at(item_grad, shard_items, residual[:, None] * user_vecs * scale)
            gradients[shard_id] = np.concatenate([user_grad.ravel(), item_grad.ravel()])
        return gradients

    def full_gradient(self, weights: np.ndarray) -> np.ndarray:
        return self.shard_gradients(weights).sum(axis=0) + self.l2 * weights

    def loss(self, weights: np.ndarray) -> float:
        user_weights, item_weights = self._split(weights)
        pred = np.einsum("ij,ij->i", user_weights[self.users], item_weights[self.items])
        residual = pred - self.y
        data_loss = 0.5 * float(np.dot(residual, residual)) / self.n_samples
        reg_loss = 0.5 * self.l2 * float(np.dot(weights, weights))
        return data_loss + reg_loss

    def _split(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        user_size = self.n_users * self.embedding_dim
        user_weights = weights[:user_size].reshape(self.n_users, self.embedding_dim)
        item_weights = weights[user_size:].reshape(self.n_items, self.embedding_dim)
        return user_weights, item_weights


def make_sparse_embedding_problem(
    n_interactions: int,
    n_users: int,
    n_items: int,
    embedding_dim: int,
    n_shards: int,
    l2: float,
    seed: int,
    zipf_exponent: float = 1.15,
    shard_cost_skew: float = 1.75,
) -> SparseEmbeddingProblem:
    rng = np.random.default_rng(seed)
    true_users = rng.normal(0.0, 0.35, size=(n_users, embedding_dim))
    true_items = rng.normal(0.0, 0.35, size=(n_items, embedding_dim))
    item_rank = np.arange(1, n_items + 1, dtype=float)
    item_probs = item_rank ** (-zipf_exponent)
    item_probs /= item_probs.sum()
    item_hotness = item_probs / item_probs.mean()
    shard_multipliers = np.geomspace(shard_cost_skew, 1.0 / shard_cost_skew, n_shards)

    boundaries = np.linspace(0, n_interactions, n_shards + 1, dtype=int)
    shard_slices = [
        slice(int(boundaries[i]), int(boundaries[i + 1])) for i in range(n_shards)
    ]
    users = np.empty(n_interactions, dtype=np.int32)
    items = np.empty(n_interactions, dtype=np.int32)
    sample_costs = np.empty(n_interactions, dtype=float)

    for shard_id, shard_slice in enumerate(shard_slices):
        size = shard_slice.stop - shard_slice.start
        users[shard_slice] = rng.integers(0, n_users, size=size, dtype=np.int32)
        mix = 0.20 + 0.65 * shard_id / max(n_shards - 1, 1)
        shard_probs = (1.0 - mix) * item_probs + mix / n_items
        shard_probs /= shard_probs.sum()
        shard_items = rng.choice(n_items, size=size, p=shard_probs)
        items[shard_slice] = shard_items.astype(np.int32, copy=False)
        sample_costs[shard_slice] = shard_multipliers[shard_id] * (
            1.0 + 0.08 * np.log1p(item_hotness[shard_items])
        )

    clean = np.einsum("ij,ij->i", true_users[users], true_items[items])
    y = clean + rng.normal(0.0, 0.03, size=n_interactions)
    row_ids = np.repeat(np.arange(n_interactions), 2)
    col_ids = np.empty(2 * n_interactions, dtype=np.int32)
    col_ids[0::2] = users
    col_ids[1::2] = n_users + items
    x = sparse.csr_matrix(
        (np.ones(2 * n_interactions, dtype=float), (row_ids, col_ids)),
        shape=(n_interactions, n_users + n_items),
    )
    return SparseEmbeddingProblem(
        users=users,
        items=items,
        y=y,
        x=x,
        shard_slices=shard_slices,
        sample_costs=sample_costs,
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        l2=l2,
    )
