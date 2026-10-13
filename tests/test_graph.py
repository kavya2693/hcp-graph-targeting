import numpy as np
from sklearn.metrics import roc_auc_score

from hcp_targeting.data import generate
from hcp_targeting.embeddings import graphsage_embeddings, node2vec_walks
from hcp_targeting.features import adjacency, build_graphs


def test_generator_is_seeded_and_keeps_hidden_drivers_out_of_the_files() -> None:
    a = generate(1500, seed=3)
    b = generate(1500, seed=3)
    assert a.hcps.equals(b.hcps)
    assert a.edges.equals(b.edges)
    assert set(a.hcps.columns) == {
        "hcp_id", "specialty", "region", "years_in_practice", "annual_class_rx", "decile", "converted",
    }  # fmt: skip
    assert 0.04 < a.hcps["converted"].mean() < 0.15
    assert a.hcps["years_in_practice"].isna().mean() > 0.02


def test_node2vec_walks_only_follow_real_edges() -> None:
    ds = generate(800, seed=5)
    _, ug = build_graphs(ds)
    A = adjacency(ug)
    walks = node2vec_walks(A, 2, 10, p=1.0, q=0.5, rng=np.random.default_rng(0))
    steps = np.stack([walks[:, :-1].ravel(), walks[:, 1:].ravel()])
    assert np.all(A[steps[0], steps[1]] > 0)


def test_low_return_parameter_makes_walks_backtrack() -> None:
    ds = generate(800, seed=5)
    _, ug = build_graphs(ds)
    A = adjacency(ug)
    sticky = node2vec_walks(A, 1, 8, p=0.05, q=1.0, rng=np.random.default_rng(0))
    free = node2vec_walks(A, 1, 8, p=20.0, q=1.0, rng=np.random.default_rng(0))

    def backtrack(w: np.ndarray) -> float:
        return float((w[:, 2:] == w[:, :-2]).mean())

    assert backtrack(sticky) > 0.6 > backtrack(free)


def test_graphsage_embeddings_separate_ties_from_random_pairs() -> None:
    ds = generate(1500, seed=5)
    _, ug = build_graphs(ds)
    A = adjacency(ug)
    x = np.random.default_rng(0).normal(size=(A.shape[0], 8))
    z = graphsage_embeddings(A, x, dim=16, hidden=32, epochs=40, seed=0)
    coo = A.tocoo()
    rng = np.random.default_rng(1)
    neg = rng.integers(0, A.shape[0], size=(2, coo.nnz))
    pos_s = (z[coo.row] * z[coo.col]).sum(1)
    neg_s = (z[neg[0]] * z[neg[1]]).sum(1)
    labels = np.r_[np.ones(len(pos_s)), np.zeros(len(neg_s))]
    assert roc_auc_score(labels, np.r_[pos_s, neg_s]) > 0.8
