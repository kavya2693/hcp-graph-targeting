| arm | AUC | fold sd | top-decile lift | captured_5pct | captured_20pct |
|---|---|---|---|---|---|
| decile list (no model) | 0.623 |  | 1.60 | 126 | 511 |
| profile + decile (XGBoost) | 0.641 | 0.016 | 2.03 | 166 | 552 |
| + centrality and community structure | 0.658 | 0.020 | 2.07 | 188 | 566 |
| + community and neighbour conversion rates | 0.668 | 0.014 | 2.10 | 189 | 594 |
| graph features + node2vec | 0.676 | 0.014 | 2.26 | 202 | 631 |
| graph features + GraphSAGE | 0.666 | 0.016 | 2.15 | 194 | 592 |
| graph features + node2vec + GraphSAGE | 0.674 | 0.014 | 2.32 | 212 | 627 |
| profile + embeddings only (no hand-built graph features) | 0.658 | 0.020 | 2.22 | 213 | 603 |
| control, HCPs shuffled across the graph: + community and neighbour conversion rates | 0.633 | 0.016 | 1.84 | 146 | 550 |
| control, HCPs shuffled across the graph: graph features + node2vec + GraphSAGE | 0.633 | 0.017 | 1.81 | 160 | 526 |
| ceiling: true propensity (answer key) | 0.761 |  | 3.24 | 322 | 834 |
