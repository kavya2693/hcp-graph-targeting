"""The label-derived graph features must never let an HCP see its own outcome."""

import numpy as np
from scipy import sparse

from hcp_targeting.features import label_graph_features


def _small_graph(n: int, seed: int) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n, size=4 * n)
    dst = rng.integers(0, n, size=4 * n)
    keep = src != dst
    A = sparse.coo_matrix((np.ones(keep.sum()), (src[keep], dst[keep])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float).tocsr()
    communities = np.column_stack([rng.integers(0, 5, size=n), rng.integers(0, 30, size=n)])
    y = (rng.random(n) < 0.2).astype(int)
    return A, communities, y


def test_flipping_an_hcps_own_label_never_changes_its_own_features() -> None:
    A, comm, y = _small_graph(400, 1)
    train_idx = np.arange(0, 320)
    base = label_graph_features(A, comm, y, train_idx, prior_weight=5.0, seed=3).to_numpy()
    # Pick HCPs from both the training and the held-out side.
    for i in (0, 17, 150, 319, 320, 399):
        flipped = y.copy()
        flipped[i] = 1 - flipped[i]
        out = label_graph_features(A, comm, flipped, train_idx, prior_weight=5.0, seed=3).to_numpy()
        np.testing.assert_allclose(out[i], base[i])


def test_held_out_labels_never_reach_any_feature() -> None:
    A, comm, y = _small_graph(400, 2)
    train_idx = np.arange(0, 300)
    base = label_graph_features(A, comm, y, train_idx, prior_weight=5.0, seed=3).to_numpy()
    scrambled = y.copy()
    scrambled[300:] = 1 - scrambled[300:]
    out = label_graph_features(A, comm, scrambled, train_idx, prior_weight=5.0, seed=3).to_numpy()
    np.testing.assert_allclose(out, base)


def test_leave_one_out_encoding_would_have_leaked() -> None:
    """Documents the bug the inner-fold scheme replaces: LOO rate plus size decodes the label."""
    _, comm, y = _small_graph(400, 4)
    level = comm[:, 1]
    size = np.bincount(level)[level]
    pos = np.bincount(level, weights=y)[level]
    loo = (pos - y) / (size - 1)
    # Inside every mixed community the LOO rate is lower exactly for the converters.
    checked = 0
    for c in np.unique(level):
        m = level == c
        if 0 < y[m].sum() < m.sum():
            assert loo[m & (y == 1)].max() < loo[m & (y == 0)].min()
            checked += 1
    assert checked > 10
