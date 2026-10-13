"""Rerun the ablation on freshly generated networks to see which gaps survive a new seed.

The seed-7 run in outputs/ is reused; each extra seed regenerates data, embeddings and folds.
Betweenness uses 64 sampled sources here instead of 256 to keep the sweep affordable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hcp_targeting.cli import run

KEY = {
    "profile + decile (XGBoost)": "profile",
    "+ community and neighbour conversion rates": "graph features",
    "graph features + node2vec": "+ node2vec",
    "graph features + GraphSAGE": "+ GraphSAGE",
    "graph features + node2vec + GraphSAGE": "+ both",
    "profile + embeddings only (no hand-built graph features)": "embeddings only",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 23])
    p.add_argument("--n-hcps", type=int, default=20_000)
    p.add_argument("--out", type=Path, default=Path("outputs"))
    a = p.parse_args()
    tables = {}
    main_table = a.out / "ablation.csv"
    if main_table.exists():
        tables[7] = pd.read_csv(main_table)
    for seed in a.seeds:
        out = a.out / "seeds" / f"seed_{seed}"
        ns = argparse.Namespace(
            n_hcps=a.n_hcps, seed=seed, data_dir=Path("data") / "seeds" / f"seed_{seed}", out=out,
            refresh=True, folds=5, dim=32, betweenness_k=64, budget_small=0.05, budget_large=0.2,
            n_boot=50, threads=4,
        )  # fmt: skip
        run(ns)
        tables[seed] = pd.read_csv(out / "ablation.csv")
    rows = []
    for seed, t in sorted(tables.items()):
        t = t[t["arm"].isin(KEY)].set_index("arm")
        for arm, short in KEY.items():
            rows.append({"seed": seed, "arm": short, **t.loc[arm, ["auc", "captured_5pct"]].to_dict()})
    long = pd.DataFrame(rows)
    wide = long.pivot(index="seed", columns="arm", values="auc")[list(KEY.values())]
    cap = long.pivot(index="seed", columns="arm", values="captured_5pct")[list(KEY.values())]
    long.to_csv(a.out / "seed_sweep.csv", index=False)
    text = (
        "AUC by seed\n"
        + wide.round(3).to_string()
        + "\n\nConversions captured with a 5% call budget, by seed\n"
        + cap.astype(int).to_string()
        + "\n\nAUC gain over graph features, by seed\n"
        + wide.sub(wide["graph features"], axis=0).round(4).to_string()
        + "\n"
    )
    (a.out / "seed_sweep.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
