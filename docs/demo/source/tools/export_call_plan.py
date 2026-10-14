"""Re-run the repo's own pipeline for one arm and export the launch call plan + a network sample.

Same data (data/, seed 7), Louvain levels, structural features (betweenness on 256 sampled
sources), 5 stratified folds and XGBoost settings as `make run`, for the arm
"+ community and neighbour conversion rates" (hand-built graph features, no embeddings).
Its out-of-fold scores rank all 20,000 HCPs; the top 1,000 (the 5% call budget) are the call plan.
The script asserts that AUC and converters captured match outputs/ablation.csv.

It also exports, for the video only:
  - the 8 largest practice-size (level-1) communities inside one regional (level-0) block, with a
    seeded spring layout, for slide 06;
  - the level-1 community holding the most planned calls, with layout and the plan rank of each
    member, for the slide 07 scene.

Run:  cd <repo> && .venv/bin/python <this file> <out.json>
"""

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from hcp_targeting import data as datamod
from hcp_targeting.evaluate import ARMS, captured_at_budget, make_folds, run_arm
from hcp_targeting.features import (
    adjacency,
    build_graphs,
    label_graph_features,
    louvain_communities,
    profile_features,
    structural_features,
)

SEED, FOLDS, BETWEENNESS_K, THREADS, CALLS = 7, 5, 256, 4, 1000

torch.set_num_threads(THREADS)
ds = datamod.load(Path("data"))
h = ds.hcps
y = h["converted"].to_numpy()
n = len(y)
profile = profile_features(ds)
dg, ug = build_graphs(ds)
A = adjacency(ug)
comms = louvain_communities(ug, SEED)
structure = structural_features(ds, dg, ug, comms, BETWEENNESS_K, SEED)
blocks = {"profile": profile, "structure": structure}
folds = make_folds(y, FOLDS, SEED)
arm = ARMS[2]
res = run_arm(arm, blocks, y, A, comms, folds, SEED, THREADS)
score = res.oof
auc = float(roc_auc_score(y, score))
captured = captured_at_budget(y, score, CALLS)
rule = h["decile"].to_numpy() * 1e6 + h["annual_class_rx"].to_numpy()
captured_rule = captured_at_budget(y, rule, CALLS)
ref = pd.read_csv("outputs/ablation.csv").set_index("arm")
assert round(auc, 3) == round(ref.at[arm.name, "auc"], 3), (auc, ref.at[arm.name, "auc"])
assert captured == int(ref.at[arm.name, "captured_5pct"]), captured
assert captured_rule == int(ref.at["decile list (no model)", "captured_5pct"]), captured_rule

# Fold-built label features as each HCP saw them when it was held out (same call as run_arm).
lab = np.zeros((n, 4))
for train_idx, test_idx in folds:
    f = label_graph_features(A, comms, y, train_idx, prior_weight=20.0, seed=SEED).to_numpy()
    lab[test_idx] = f[test_idx]

order = np.argsort(-score, kind="stable")
rank = np.empty(n, dtype=int)
rank[order] = np.arange(1, n + 1)
rule_rank = np.empty(n, dtype=int)
rule_rank[np.argsort(-rule, kind="stable")] = np.arange(1, n + 1)
plan = []
for r, i in enumerate(order[:CALLS], start=1):
    plan.append(
        {
            "rank": r,
            "hcp": int(i),
            "specialty": h.at[i, "specialty"],
            "region": h.at[i, "region"],
            "decile": int(h.at[i, "decile"]),
            "decile_rank": int(rule_rank[i]),
            "score": round(float(score[i]), 3),
            "practice_rate": round(float(lab[i, 1]), 3),
            "nbr_converted": int(lab[i, 3]),
            "converted": int(y[i]),
        }
    )


def layout(nodes: list[int], seed: int) -> dict[int, list[float]]:
    sub = ug.subgraph(nodes)
    pos = nx.spring_layout(sub, seed=seed, iterations=200, weight=None)
    xy = np.array([pos[v] for v in nodes])
    xy = (xy - xy.min(axis=0)) / np.maximum(xy.max(axis=0) - xy.min(axis=0), 1e-9)
    return {int(v): [round(float(a), 4), round(float(b), 4)] for v, (a, b) in zip(nodes, xy, strict=True)}


# Slide 06 sample: one regional block (level 0), its 8 largest practice-size communities (level 1).
l0, l1 = comms[:, 0], comms[:, 1]
block = int(pd.Series(l0).value_counts().index[4])
in_block = np.flatnonzero(l0 == block)
top_l1 = pd.Series(l1[in_block]).value_counts().index[:8].tolist()
nodes = [int(v) for v in in_block if l1[v] in top_l1]
pos = layout(nodes, SEED)
sample = {
    "l0_block": block,
    "l0_size": len(in_block),
    "communities": [
        {
            "l1": int(c),
            "size": int((l1 == c).sum()),
            "converted": int(y[l1 == c].sum()),
            "rate": round(float(y[l1 == c].mean()), 3),
        }
        for c in top_l1
    ],
    "nodes": [[v, top_l1.index(int(l1[v])), int(y[v]), *pos[v]] for v in nodes],
    "edges": [[int(a), int(b)] for a, b in ug.subgraph(nodes).edges()],
}

# Slide 07 scene: the level-1 community with the most planned calls.
in_plan = rank <= CALLS
best = int(pd.Series(l1[in_plan]).value_counts().index[0])
members = [int(v) for v in np.flatnonzero(l1 == best)]
pos2 = layout(members, SEED)
scene = {
    "l1": best,
    "size": len(members),
    "planned": int(in_plan[members].sum()),
    "planned_converted": int((in_plan & (y == 1))[members].sum()),
    "nodes": [[v, int(rank[v]) if rank[v] <= CALLS else 0, int(y[v]), *pos2[v]] for v in members],
    "edges": [[int(a), int(b)] for a, b in ug.subgraph(members).edges()],
}

out = {
    "arm": arm.name,
    "n_hcps": n,
    "calls": CALLS,
    "auc": round(auc, 4),
    "captured": captured,
    "captured_decile_list": captured_rule,
    "plan_deciles": {
        int(k): int(v) for k, v in pd.Series([p["decile"] for p in plan]).value_counts().items()
    },
    "plan_from_outside_decile_top_1000": int(sum(p["decile_rank"] > CALLS for p in plan)),
    "plan": plan,
    "network_sample": sample,
    "scene": scene,
}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if k not in ("plan", "network_sample", "scene")})
print(
    "sample",
    sample["l0_block"],
    sample["l0_size"],
    len(sample["nodes"]),
    len(sample["edges"]),
    sample["communities"],
)
print("scene", {k: v for k, v in scene.items() if k not in ("nodes", "edges")}, len(scene["edges"]))
print(pd.DataFrame(plan[:12]).to_string())
