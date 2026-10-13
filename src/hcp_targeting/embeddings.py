"""Unsupervised node embeddings: node2vec (second-order walks + skip-gram) and GraphSAGE.

Both are trained without labels, once per graph, so they cannot leak the outcome into a
cross-validation fold.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import sparse
from torch import nn
from torch_geometric.nn import SAGEConv


def node2vec_walks(
    A: sparse.csr_matrix,
    walks_per_node: int,
    walk_length: int,
    p: float,
    q: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Second-order biased walks, all walkers stepped together.

    Each step proposes a uniform neighbour and accepts it with the node2vec weight
    (1/p to return, 1 to stay near the previous node, 1/q to move outward) divided by the
    largest weight. Rejection sampling avoids precomputing per-edge alias tables, which for
    tens of thousands of nodes would cost more memory than the walks themselves.
    """
    n = A.shape[0]
    indptr, indices = A.indptr, A.indices
    deg = np.diff(indptr)
    row = np.repeat(np.arange(n, dtype=np.int64), deg)
    edge_keys = np.sort(row * n + indices.astype(np.int64))

    def is_edge(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        keys = u.astype(np.int64) * n + v.astype(np.int64)
        pos = np.searchsorted(edge_keys, keys)
        pos = np.minimum(pos, len(edge_keys) - 1)
        return edge_keys[pos] == keys

    def uniform_neighbour(cur: np.ndarray) -> np.ndarray:
        return indices[indptr[cur] + (rng.random(len(cur)) * deg[cur]).astype(np.int64)]

    starts = np.tile(np.flatnonzero(deg > 0), walks_per_node)
    walks = np.empty((len(starts), walk_length), dtype=np.int64)
    walks[:, 0] = starts
    walks[:, 1] = uniform_neighbour(starts)
    w_max = max(1.0 / p, 1.0, 1.0 / q)
    for t in range(2, walk_length):
        prev, cur = walks[:, t - 2], walks[:, t - 1]
        nxt = np.empty(len(cur), dtype=np.int64)
        pending = np.arange(len(cur))
        while len(pending):
            cand = uniform_neighbour(cur[pending])
            pv = prev[pending]
            w = np.where(cand == pv, 1.0 / p, np.where(is_edge(pv, cand), 1.0, 1.0 / q))
            ok = rng.random(len(pending)) * w_max < w
            nxt[pending[ok]] = cand[ok]
            pending = pending[~ok]
        walks[:, t] = nxt
    return walks


def skipgram(
    walks: np.ndarray,
    n_nodes: int,
    dim: int,
    window: int,
    negatives: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Skip-gram with negative sampling over walk co-occurrences."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    centres, contexts = [], []
    for o in range(1, window + 1):
        centres += [walks[:, :-o].ravel(), walks[:, o:].ravel()]
        contexts += [walks[:, o:].ravel(), walks[:, :-o].ravel()]
    c = torch.from_numpy(np.concatenate(centres))
    x = torch.from_numpy(np.concatenate(contexts))
    freq = np.bincount(walks.ravel(), minlength=n_nodes).astype(float) ** 0.75
    noise = torch.from_numpy(freq / freq.sum())

    emb_in = nn.Embedding(n_nodes, dim, sparse=True)
    emb_out = nn.Embedding(n_nodes, dim, sparse=True)
    nn.init.uniform_(emb_in.weight, -0.5 / dim, 0.5 / dim)
    nn.init.zeros_(emb_out.weight)
    opt = torch.optim.SparseAdam([*emb_in.parameters(), *emb_out.parameters()], lr=0.01)
    for _ in range(epochs):
        perm = torch.from_numpy(rng.permutation(len(c)))
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            u = emb_in(c[idx])
            v = emb_out(x[idx])
            neg = emb_out(torch.multinomial(noise, len(idx) * negatives, replacement=True)).view(
                len(idx), negatives, dim
            )
            pos_loss = nn.functional.logsigmoid((u * v).sum(-1))
            neg_loss = nn.functional.logsigmoid(-(neg @ u.unsqueeze(-1)).squeeze(-1)).sum(-1)
            loss = -(pos_loss + neg_loss).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return emb_in.weight.detach().numpy().copy()


def node2vec_embeddings(
    A: sparse.csr_matrix,
    dim: int = 32,
    walks_per_node: int = 5,
    walk_length: int = 16,
    window: int = 3,
    p: float = 1.0,
    q: float = 0.5,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    walks = node2vec_walks(A, walks_per_node, walk_length, p, q, rng)
    return skipgram(walks, A.shape[0], dim, window, negatives=5, epochs=1, batch_size=8192, seed=seed)


class SAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)


def graphsage_embeddings(
    A: sparse.csr_matrix,
    node_x: np.ndarray,
    dim: int = 32,
    hidden: int = 64,
    epochs: int = 60,
    seed: int = 7,
) -> np.ndarray:
    """Two-layer GraphSAGE trained to tell real ties from random pairs (no outcome labels)."""
    torch.manual_seed(seed)
    coo = A.tocoo()
    edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    x = torch.tensor(node_x, dtype=torch.float32)
    n = x.shape[0]
    model = SAGE(x.shape[1], hidden, dim)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        z = model(x, edge_index)
        src, dst = edge_index
        neg_dst = torch.randint(0, n, (src.numel(),), generator=gen)
        pos = (z[src] * z[dst]).sum(-1)
        neg = (z[src] * z[neg_dst]).sum(-1)
        loss = -(nn.functional.logsigmoid(pos).mean() + nn.functional.logsigmoid(-neg).mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return model(x, edge_index).numpy().copy()
