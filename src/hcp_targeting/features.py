"""Feature blocks for the ablation: profile and decile, graph structure, and fold-safe label features."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from .data import Dataset


def profile_features(ds: Dataset) -> pd.DataFrame:
    """What a decile-driven targeting list already has: decile, volume, trend, specialty, region."""
    h = ds.hcps
    rx = ds.monthly_rx.drop(columns="hcp_id").to_numpy(dtype=float)
    last3 = np.nanmean(rx[:, -3:], axis=1)
    prior3 = np.nanmean(rx[:, -6:-3], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        trend = np.log1p(last3) - np.log1p(prior3)
    out = pd.DataFrame(
        {
            "decile": h["decile"].to_numpy(),
            "log_annual_rx": np.log1p(h["annual_class_rx"].to_numpy()),
            "rx_trend_3m": trend,
            "months_reported": np.sum(~np.isnan(rx), axis=1),
            "years_in_practice": h["years_in_practice"].to_numpy(),
        }
    )
    spec = pd.get_dummies(h["specialty"], prefix="spec", dtype=float)
    region = pd.get_dummies(h["region"], prefix="region", dtype=float)
    return pd.concat([out, spec, region], axis=1)


def build_graphs(ds: Dataset) -> tuple[nx.DiGraph, nx.Graph]:
    """Directed graph (referrals point at the specialist) and its undirected weighted view."""
    n = len(ds.hcps)
    e = ds.edges
    dg = nx.DiGraph()
    dg.add_nodes_from(range(n))
    for s, d, t, w in e[["src", "dst", "edge_type", "shared_patients"]].itertuples(index=False):
        dg.add_edge(int(s), int(d), weight=float(w))
        if t == "co_treatment":
            dg.add_edge(int(d), int(s), weight=float(w))
    ug = nx.Graph()
    ug.add_nodes_from(range(n))
    for s, d, w in e[["src", "dst", "shared_patients"]].itertuples(index=False):
        if ug.has_edge(int(s), int(d)):
            ug[int(s)][int(d)]["weight"] += float(w)
        else:
            ug.add_edge(int(s), int(d), weight=float(w))
    return dg, ug


def adjacency(ug: nx.Graph) -> sparse.csr_matrix:
    n = ug.number_of_nodes()
    return nx.to_scipy_sparse_array(ug, nodelist=range(n), weight=None, format="csr").astype(float)


def louvain_communities(ug: nx.Graph, seed: int, resolutions: tuple[float, ...] = (1.0, 16.0)) -> np.ndarray:
    """One column of community labels per resolution.

    Resolution 1 finds a few dozen regional blocks. Practice groups are far smaller than the
    resolution limit lets default Louvain see, so a second, finer level is added; 16 was picked
    because it yields communities about the size of a practice group (~50 HCPs), not by
    looking at conversion.
    """
    levels = []
    for r in resolutions:
        comms = nx.community.louvain_communities(ug, weight="weight", resolution=r, seed=seed)
        labels = np.empty(ug.number_of_nodes(), dtype=int)
        for i, c in enumerate(sorted(comms, key=len, reverse=True)):
            labels[list(c)] = i
        levels.append(labels)
    return np.column_stack(levels)


def structural_features(
    ds: Dataset, dg: nx.DiGraph, ug: nx.Graph, communities: np.ndarray, betweenness_k: int, seed: int
) -> pd.DataFrame:
    """Label-free graph features: centrality, local structure, community context."""
    n = len(ds.hcps)
    A = adjacency(ug)
    deg = np.asarray(A.sum(axis=1)).ravel()
    in_deg = np.array([dg.in_degree(i) for i in range(n)], dtype=float)
    out_deg = np.array([dg.out_degree(i) for i in range(n)], dtype=float)
    wdeg = np.array([d for _, d in ug.degree(weight="weight")], dtype=float)
    pr_dict = nx.pagerank(dg, alpha=0.85, weight="weight")
    pagerank = np.array([pr_dict[i] for i in range(n)])
    # Exact betweenness is O(nm); sampling source nodes keeps it to seconds and the ranking holds.
    k = min(betweenness_k, n)
    bt_dict = nx.betweenness_centrality(ug, k=k, seed=seed, normalized=True)
    betweenness = np.array([bt_dict[i] for i in range(n)])
    clust_dict = nx.clustering(ug)
    clustering = np.array([clust_dict[i] for i in range(n)])

    decile = ds.hcps["decile"].to_numpy(dtype=float)
    safe_deg = np.maximum(deg, 1.0)
    nbr_mean_decile = (A @ decile) / safe_deg
    nbr_mean_pr = (A @ pagerank) / safe_deg
    # Max neighbour PageRank: is this HCP one hop from a hub, whoever that hub is.
    A_coo = A.tocoo()
    nbr_max_pr = np.zeros(n)
    np.maximum.at(nbr_max_pr, A_coo.row, pagerank[A_coo.col])
    hub_cut = np.quantile(pagerank, 0.99)
    hub_ties = A @ (pagerank >= hub_cut).astype(float)

    comm_cols = {}
    for i, level in enumerate(communities.T):
        size = np.bincount(level)
        comm_cols[f"community_size_l{i}"] = size[level].astype(float)
        comm_cols[f"community_mean_decile_l{i}"] = (np.bincount(level, weights=decile) / size)[level]

    return pd.DataFrame(
        {
            "degree": deg,
            "referrals_in": in_deg,
            "referrals_out": out_deg,
            "shared_patients": wdeg,
            "pagerank": pagerank,
            "betweenness": betweenness,
            "clustering": clustering,
            "nbr_mean_decile": nbr_mean_decile,
            "nbr_mean_pagerank": nbr_mean_pr,
            "nbr_max_pagerank": nbr_max_pr,
            "hub_ties": hub_ties,
            **comm_cols,
        }
    )


def _rates(
    A: sparse.csr_matrix, communities: np.ndarray, y: np.ndarray, known_idx: np.ndarray, prior_weight: float
) -> np.ndarray:
    """Smoothed community conversion rate (one column per level), neighbour rate and count."""
    n = A.shape[0]
    known = np.zeros(n)
    known[known_idx] = 1.0
    pos = np.zeros(n)
    pos[known_idx] = y[known_idx]
    base = y[known_idx].mean()
    cols = []
    for level in communities.T:
        c_n = np.bincount(level, weights=known)
        c_pos = np.bincount(level, weights=pos)
        cols.append((c_pos[level] + prior_weight * base) / (c_n[level] + prior_weight))
    nbr_known = A @ known
    nbr_pos = A @ pos
    cols.append((nbr_pos + prior_weight * base) / (nbr_known + prior_weight))
    cols.append(nbr_pos)
    return np.column_stack(cols)


def label_graph_features(
    A: sparse.csr_matrix,
    communities: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    prior_weight: float,
    inner_folds: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Community and neighbour conversion rates, built so no HCP ever sees its own outcome.

    Test HCPs get rates from every training label. Training HCPs get rates from the other
    inner folds of the training set. A leave-one-out rate looked equivalent and was not: with
    community size also in the model, (sum - own label) / (size - 1) lets the trees read the
    HCP's own label back, which inflated training fit and cost about 0.05 AUC on held-out HCPs.
    """
    if communities.ndim == 1:
        communities = communities[:, None]
    n = A.shape[0]
    out = np.zeros((n, communities.shape[1] + 2))
    in_train = np.zeros(n, dtype=bool)
    in_train[train_idx] = True
    out[~in_train] = _rates(A, communities, y, train_idx, prior_weight)[~in_train]
    rng = np.random.default_rng(seed)
    inner = rng.integers(0, inner_folds, size=len(train_idx))
    for k in range(inner_folds):
        target = train_idx[inner == k]
        known = train_idx[inner != k]
        out[target] = _rates(A, communities, y, known, prior_weight)[target]
    names = [f"community_conv_rate_l{i}" for i in range(communities.shape[1])]
    return pd.DataFrame(out, columns=[*names, "nbr_conv_rate", "nbr_converted"])
