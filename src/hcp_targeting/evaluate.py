"""Targeting metrics and the cross-validated XGBoost ablation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from .features import label_graph_features


def top_decile_lift(y: np.ndarray, score: np.ndarray) -> float:
    """Conversion rate among the top 10% by score, divided by the overall rate."""
    k = max(int(np.ceil(0.10 * len(y))), 1)
    top = np.argsort(-score, kind="stable")[:k]
    return float(y[top].mean() / y.mean())


def captured_at_budget(y: np.ndarray, score: np.ndarray, calls: int) -> int:
    """Conversions among the `calls` highest-scored HCPs: what a fixed call plan actually wins."""
    top = np.argsort(-score, kind="stable")[:calls]
    return int(y[top].sum())


def summarise(y: np.ndarray, score: np.ndarray, budgets: dict[str, int]) -> dict[str, float]:
    out = {"auc": float(roc_auc_score(y, score)), "top_decile_lift": top_decile_lift(y, score)}
    for label, calls in budgets.items():
        out[label] = float(captured_at_budget(y, score, calls))
    return out


def captured_share(y: np.ndarray, score: np.ndarray, share: float) -> float:
    return float(captured_at_budget(y, score, max(round(share * len(y)), 1)))


def bootstrap_delta(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """metric(a) - metric(b) on the same HCPs, with a paired 95% bootstrap interval."""
    rng = np.random.default_rng(seed)
    point = metric(y, a) - metric(y, b)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if np.min(y[idx]) == np.max(y[idx]):
            continue
        deltas.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(point), float(lo), float(hi)


@dataclass
class Arm:
    name: str
    blocks: list[str]
    # Label-dependent blocks are rebuilt inside every fold from training labels only.
    fold_blocks: list[str] = field(default_factory=list)


ARMS = [
    Arm("profile + decile (XGBoost)", ["profile"]),
    Arm("+ centrality and community structure", ["profile", "structure"]),
    Arm("+ community and neighbour conversion rates", ["profile", "structure"], ["label_graph"]),
    Arm("graph features + node2vec", ["profile", "structure", "node2vec"], ["label_graph"]),
    Arm("graph features + GraphSAGE", ["profile", "structure", "graphsage"], ["label_graph"]),
    Arm(
        "graph features + node2vec + GraphSAGE",
        ["profile", "structure", "node2vec", "graphsage"],
        ["label_graph"],
    ),
    Arm("profile + embeddings only (no hand-built graph features)", ["profile", "node2vec", "graphsage"]),
]


def make_model(seed: int, n_jobs: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        tree_method="hist",
        eval_metric="auc",
        random_state=seed,
        n_jobs=n_jobs,
    )


@dataclass
class ArmResult:
    name: str
    oof: np.ndarray
    fold_auc: list[float]
    importance: pd.Series


def run_arm(
    arm: Arm,
    blocks: dict[str, pd.DataFrame],
    y: np.ndarray,
    A: sparse.csr_matrix,
    communities: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    n_jobs: int,
    label_fn: Callable[..., pd.DataFrame] = label_graph_features,
) -> ArmResult:
    static = pd.concat([blocks[b] for b in arm.blocks], axis=1)
    oof = np.zeros(len(y))
    fold_auc: list[float] = []
    gains: list[pd.Series] = []
    for train_idx, test_idx in folds:
        X = static
        if "label_graph" in arm.fold_blocks:
            X = pd.concat(
                [static, label_fn(A, communities, y, train_idx, prior_weight=20.0, seed=seed)], axis=1
            )
        model = make_model(seed, n_jobs)
        model.fit(X.iloc[train_idx], y[train_idx])
        p = model.predict_proba(X.iloc[test_idx])[:, 1]
        oof[test_idx] = p
        fold_auc.append(float(roc_auc_score(y[test_idx], p)))
        gain = model.get_booster().get_score(importance_type="total_gain")
        gains.append(pd.Series(gain, dtype=float))
    imp = pd.concat(gains, axis=1).fillna(0.0).mean(axis=1).sort_values(ascending=False)
    return ArmResult(arm.name, oof, fold_auc, imp / imp.sum())


def make_folds(y: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))
