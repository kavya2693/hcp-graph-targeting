# ADR-0001: Label-derived graph features are built inside each fold; embeddings never see labels

## Status
Accepted

## Context
The question is whether network position adds anything to a decile-based call list, and
whether learned node embeddings add anything on top of hand-built graph features. Both
answers are easy to inflate. Graph features that use outcomes (how many of my neighbours
converted, how my community converted) leak the label if a held-out HCP's own outcome, or
its training-time equivalent, reaches its own row. Embeddings trained with a supervised
objective on the whole graph leak in the same way.

The first version used a leave-one-out community conversion rate. It looked safe and was
not: with community size also in the model, (community converters - own label) /
(size - 1) is lower for exactly the converters, so trees read the label back. Training fit
went up and held-out AUC fell from 0.641 to 0.589, below the plain profile model.

## Decision
- Label-derived features (community conversion rate at two Louvain levels, neighbour
  conversion rate, number of converted neighbours) are rebuilt inside every outer fold.
  Held-out HCPs get rates from all training labels; training HCPs get rates from the other
  inner folds of the training set. A test asserts that flipping any HCP's own label leaves
  its own feature row unchanged.
- node2vec and GraphSAGE are trained once per graph with unsupervised objectives (skip-gram
  over walks; tie versus random-pair discrimination). No outcome enters either.
- node2vec walks and skip-gram are written here (numpy + torch). PyTorch Geometric's own
  Node2Vec needs torch-cluster or pyg-lib, which have no plain PyPI wheels for CPU; SAGEConv
  from PyTorch Geometric installs cleanly and is used as is.
- Every arm is scored on the same 5-fold out-of-fold predictions, and arm-to-arm gaps get a
  paired bootstrap interval, so a 0.005 AUC difference is not reported as a win.
- A control arm moves every HCP to a random node of the same graph. Any lift that survives
  it is not coming from the network.

## Alternatives rejected
- **Leave-one-out target encoding.** Leaks as described above.
- **Supervised GraphSAGE as the classifier.** Would compete with XGBoost rather than test
  whether embeddings add to it, and would need its own fold-safe training loop per fold.
- **Single train/test split.** At an 8% conversion rate and 4,000 held-out HCPs, the
  5%-budget capture count moves by 10+ between splits; out-of-fold scoring on all 20,000
  HCPs is steadier.

## Consequences
- Each arm costs five XGBoost fits; the full run takes several minutes on a laptop CPU.
- Fold-built features cannot be precomputed once for serving. In production they would be
  computed from the last campaign's outcomes, which is the same shape of computation.
