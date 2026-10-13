| arm | AUC | fold sd | top-decile lift | captured_5pct | captured_20pct |
|---|---|---|---|---|---|
| decile list (no model) | 0.626 |  | 1.58 | 126 | 503 |
| profile + decile (XGBoost) | 0.638 | 0.013 | 1.58 | 119 | 520 |
| + centrality and community structure | 0.659 | 0.010 | 2.11 | 180 | 593 |
| + community and neighbour conversion rates | 0.663 | 0.012 | 2.14 | 192 | 584 |
| graph features + node2vec | 0.665 | 0.009 | 2.18 | 197 | 602 |
| graph features + GraphSAGE | 0.662 | 0.011 | 2.15 | 186 | 576 |
| graph features + node2vec + GraphSAGE | 0.666 | 0.009 | 2.22 | 203 | 591 |
| profile + embeddings only (no hand-built graph features) | 0.652 | 0.012 | 1.96 | 153 | 549 |
| control, HCPs shuffled across the graph: + community and neighbour conversion rates | 0.636 | 0.014 | 1.63 | 128 | 523 |
| control, HCPs shuffled across the graph: graph features + node2vec + GraphSAGE | 0.629 | 0.013 | 1.56 | 123 | 495 |
| ceiling: true propensity (answer key) | 0.754 |  | 3.09 | 291 | 805 |
