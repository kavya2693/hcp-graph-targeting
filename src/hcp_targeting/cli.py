"""One command: build features, run the ablation, write the report to outputs/."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from . import data as datamod
from .embeddings import graphsage_embeddings, node2vec_embeddings
from .evaluate import (
    ARMS,
    ArmResult,
    bootstrap_delta,
    captured_share,
    make_folds,
    run_arm,
    summarise,
)
from .features import adjacency, build_graphs, louvain_communities, profile_features, structural_features


def _standardise(df: pd.DataFrame) -> np.ndarray:
    X = df.copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col + "_missing"] = X[col].isna().astype(float)
            X[col] = X[col].fillna(X[col].median())
    arr = X.to_numpy(dtype=float)
    sd = arr.std(axis=0)
    return (arr - arr.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def run(args: argparse.Namespace) -> dict[str, object]:
    t0 = time.time()
    log: list[str] = []

    def note(msg: str) -> None:
        line = f"[{time.time() - t0:6.1f}s] {msg}"
        print(line, flush=True)
        log.append(line)

    torch.set_num_threads(args.threads)
    data_dir: Path = args.data_dir
    if args.refresh or not (data_dir / "hcps.csv").exists():
        datamod.save(datamod.generate(args.n_hcps, args.seed), data_dir)
    ds = datamod.load(data_dir)
    y = ds.hcps["converted"].to_numpy()
    n = len(y)
    note(f"loaded {n} HCPs, {len(ds.edges)} edges, conversion rate {y.mean():.3f}")

    profile = profile_features(ds)
    dg, ug = build_graphs(ds)
    A = adjacency(ug)
    communities = louvain_communities(ug, args.seed)
    for i, level in enumerate(communities.T):
        n_comm, largest = int(np.max(level)) + 1, int(np.max(np.bincount(level)))
        note(f"Louvain level {i}: {n_comm} communities, largest {largest}")
    structure = structural_features(ds, dg, ug, communities, args.betweenness_k, args.seed)
    note("centrality and structure features built")
    n2v = node2vec_embeddings(A, dim=args.dim, seed=args.seed)
    note("node2vec embeddings trained")
    sage = graphsage_embeddings(A, _standardise(profile), dim=args.dim, seed=args.seed)
    note("GraphSAGE embeddings trained")

    blocks = {
        "profile": profile,
        "structure": structure,
        "node2vec": pd.DataFrame(n2v, columns=[f"n2v_{i}" for i in range(n2v.shape[1])]),
        "graphsage": pd.DataFrame(sage, columns=[f"sage_{i}" for i in range(sage.shape[1])]),
    }
    folds = make_folds(y, args.folds, args.seed)
    budgets = {
        f"captured_{round(100 * s)}pct": max(round(s * n), 1) for s in (args.budget_small, args.budget_large)
    }

    rows: list[dict[str, object]] = []
    # Today's practice: call down the decile list, heaviest writers first within a decile.
    rule = ds.hcps["decile"].to_numpy() * 1e6 + ds.hcps["annual_class_rx"].to_numpy()
    rows.append({"arm": "decile list (no model)", **summarise(y, rule, budgets), "auc_fold_sd": np.nan})

    results: dict[str, ArmResult] = {}
    for arm in ARMS:
        res = run_arm(arm, blocks, y, A, communities, folds, args.seed, args.threads)
        results[arm.name] = res
        rows.append(
            {"arm": arm.name, **summarise(y, res.oof, budgets), "auc_fold_sd": float(np.std(res.fold_auc))}
        )
        note(f"{arm.name}: AUC {rows[-1]['auc']:.3f}")

    # Control: move every HCP to a random node of the same graph. Structure is unchanged, the
    # link between a person and their position is broken, so any lift that survives is not
    # coming from the network.
    sigma = np.random.default_rng(args.seed + 1).permutation(n)
    shuffled_blocks = {
        k: (v if k == "profile" else v.iloc[sigma].reset_index(drop=True)) for k, v in blocks.items()
    }
    A_shuf = A[sigma][:, sigma].tocsr()
    for arm in (ARMS[2], ARMS[5]):
        res = run_arm(arm, shuffled_blocks, y, A_shuf, communities[sigma], folds, args.seed, args.threads)
        rows.append(
            {
                "arm": f"control, HCPs shuffled across the graph: {arm.name}",
                **summarise(y, res.oof, budgets),
                "auc_fold_sd": float(np.std(res.fold_auc)),
            }
        )
        note(f"shuffled control {arm.name}: AUC {rows[-1]['auc']:.3f}")

    if not np.isnan(ds.true_logit).any():
        rows.append({"arm": "ceiling: true propensity (answer key)", **summarise(y, ds.true_logit, budgets)})

    table = pd.DataFrame(rows)
    base = results[ARMS[0].name].oof
    graph = results[ARMS[2].name].oof
    deltas = []
    small = args.budget_small
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "auc": lambda yy, ss: float(roc_auc_score(yy, ss)),
        f"captured_{round(100 * small)}pct": lambda yy, ss: captured_share(yy, ss, small),
    }
    for name, res in results.items():
        for ref_name, ref in (("profile + decile", base), ("graph features", graph)):
            if res.oof is ref:
                continue
            for metric_name, fn in metrics.items():
                d, lo, hi = bootstrap_delta(y, res.oof, ref, fn, args.n_boot, args.seed)
                deltas.append(
                    {
                        "arm": name,
                        "vs": ref_name,
                        "metric": metric_name,
                        "delta": d,
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
    delta_df = pd.DataFrame(deltas)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "ablation.csv", index=False)
    delta_df.to_csv(out / "deltas.csv", index=False)
    full = results[ARMS[5].name].importance
    full.head(25).rename("share_of_gain").to_csv(out / "feature_importance.csv")
    block_share = {
        "profile": float(full[full.index.isin(profile.columns)].sum()),
        "structure": float(full[full.index.isin(structure.columns)].sum()),
        "label_graph": float(full[full.index.str.contains("conv_rate|nbr_converted")].sum()),
        "node2vec": float(full[full.index.str.startswith("n2v_")].sum()),
        "graphsage": float(full[full.index.str.startswith("sage_")].sum()),
    }
    summary = {
        "n_hcps": n,
        "n_edges": len(ds.edges),
        "conversion_rate": float(y.mean()),
        "conversions": int(y.sum()),
        "communities_per_level": [int(np.max(c)) + 1 for c in communities.T],
        "budgets": budgets,
        "folds": args.folds,
        "seed": args.seed,
        "gain_share_by_block_full_model": block_share,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    md = ["| arm | AUC | fold sd | top-decile lift | " + " | ".join(budgets) + " |"]
    md.append("|---|---|---|---|" + "---|" * len(budgets))
    for r in rows:
        sd = r.get("auc_fold_sd")
        sd_txt = "" if sd is None or (isinstance(sd, float) and np.isnan(sd)) else f"{sd:.3f}"
        caps = " | ".join(f"{int(r[b])}" for b in budgets)  # type: ignore[call-overload]
        md.append(f"| {r['arm']} | {r['auc']:.3f} | {sd_txt} | {r['top_decile_lift']:.2f} | {caps} |")
    (out / "ablation.md").write_text("\n".join(md) + "\n")
    note(f"wrote {out}/ablation.md, deltas.csv, feature_importance.csv, summary.json")
    (out / "run_log.txt").write_text("\n".join(log) + "\n")
    print("\n".join(md))
    print(delta_df.round(4).to_string(index=False))
    print(json.dumps(block_share, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="HCP targeting ablation: deciles vs graph features vs embeddings")
    p.add_argument("--n-hcps", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--refresh", action="store_true", help="regenerate the dataset first")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--betweenness-k", type=int, default=256)
    p.add_argument("--budget-small", type=float, default=0.05, help="call budget as a share of HCPs")
    p.add_argument("--budget-large", type=float, default=0.20)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--threads", type=int, default=4)
    run(p.parse_args())


if __name__ == "__main__":
    main()
