| arm | AUC | fold sd | top-decile lift | captured_5pct | captured_20pct |
|---|---|---|---|---|---|
| decile list (no model) | 0.617 |  | 1.52 | 140 | 591 |
| profile + decile (XGBoost) | 0.620 | 0.008 | 1.63 | 166 | 593 |
| + centrality and community structure | 0.650 | 0.008 | 1.93 | 217 | 686 |
| + community and neighbour conversion rates | 0.659 | 0.010 | 2.06 | 249 | 716 |
| graph features + node2vec | 0.666 | 0.010 | 2.18 | 254 | 705 |
| graph features + GraphSAGE | 0.658 | 0.009 | 2.10 | 230 | 700 |
| graph features + node2vec + GraphSAGE | 0.666 | 0.010 | 2.12 | 253 | 714 |
| profile + embeddings only (no hand-built graph features) | 0.647 | 0.011 | 1.88 | 204 | 672 |
| control, HCPs shuffled across the graph: + community and neighbour conversion rates | 0.619 | 0.009 | 1.73 | 187 | 586 |
| control, HCPs shuffled across the graph: graph features + node2vec + GraphSAGE | 0.614 | 0.005 | 1.49 | 155 | 575 |
| ceiling: true propensity (answer key) | 0.750 |  | 3.11 | 371 | 962 |
