# hcp-graph-targeting

[![ci](https://github.com/kavya2693/hcp-graph-targeting/actions/workflows/ci.yml/badge.svg)](https://github.com/kavya2693/hcp-graph-targeting/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Which doctors a launch sales team should call when prescription-volume deciles stop telling
converters from non-converters, and how much of the answer comes from the referral network.

[![88-second silent walkthrough: the industry problem, how it works, the architecture, the network, the call plan and the results](docs/demo/demo.gif)](https://kavya2693.github.io/work-demos/hcp-graph-targeting.mp4)

*88-second silent walkthrough. [Play the full video](https://kavya2693.github.io/work-demos/hcp-graph-targeting.mp4) · [download the MP4, 7 MB](docs/demo/demo.mp4).*

📄 **Audit report:** every figure and assumption in the video, traced to its source — [PDF](docs/demo/AUDIT_REPORT.pdf) · [Word](docs/demo/AUDIT_REPORT.docx) · [Markdown](docs/demo/AUDIT_REPORT.md)

## At a glance

| | |
|---|---|
| **Problem** | A launch sales team ranks 20,000 HCPs by prescription-volume decile and calls from the top. Deciles barely separate converters from non-converters (AUC 0.623), and they ignore the referral and co-treatment ties through which doctors actually adopt. |
| **Approach** | One XGBoost model per feature set, from deciles alone up to deciles plus centrality, Louvain communities, fold-built neighbour conversion rates, node2vec and GraphSAGE. All arms are scored on the same 5-fold out-of-fold predictions under a fixed call budget, with paired bootstrap intervals and a shuffled-graph control. |
| **Result** | Graph features lift AUC from **0.641 to 0.668** (+0.027, 95% CI +0.018 to +0.037) and capture **189 converters instead of 166** with 1,000 calls. The best model captures **212 against 126** for the decile list, closing 44% of the gap to the answer-key ceiling (322). Shuffling HCPs across the graph drops both graph arms to 0.633. |
| **What didn't win** | Embeddings. On top of the hand-built graph features, node2vec adds +0.008 AUC (interval touches zero), GraphSAGE adds nothing on any of three seeds, and both together add +0.006, yet they take 61% of split gain in the full model. The one run where embeddings alone captured the most converters (213 vs 189) reversed on two new networks. |

## Architecture

![Architecture walkthrough: sources, graph and profile features, fold-built conversion rates, embeddings, the per-arm XGBoost ablation and the evaluation that scores it](docs/demo/architecture.gif)

```mermaid
flowchart LR
    subgraph SRC["Sources"]
        GEN["Seeded generator<br/>scripts/generate_data.py<br/>data.generate"]
        HCPS[("hcps.csv<br/>20,000 HCPs, labels")]
        RX[("monthly_class_rx.csv<br/>24 months, gappy")]
        EDGES[("edges.csv<br/>75,155 ties")]
        KEY[("propensity_answer_key.npy<br/>ceiling only")]
    end

    subgraph FEAT["Data and features"]
        LOAD["Loader<br/>data.load"]
        PROF["Profile + decile<br/>features.profile_features"]
        GRAPH["Directed + undirected graph<br/>features.build_graphs · adjacency"]
        LOUV["Louvain, resolution 1 and 16<br/>features.louvain_communities"]
        STRUCT["Centrality and structure<br/>features.structural_features"]
        LBL["Fold-built conversion rates<br/>features.label_graph_features"]
    end

    subgraph MOD["Models"]
        N2V["node2vec, label-free<br/>embeddings.node2vec_embeddings"]
        SAGE["GraphSAGE, label-free<br/>embeddings.graphsage_embeddings"]
        XGB["XGBoost per arm, 5-fold OOF<br/>evaluate.run_arm"]
    end

    subgraph DEC["Decision"]
        RULE["Decile list, today's practice<br/>cli.run"]
        PLAN["Call plan at 5% / 20% budget<br/>evaluate.captured_at_budget"]
    end

    subgraph EVAL["Evaluation"]
        MET["AUC, top-decile lift, captured<br/>evaluate.summarise"]
        BOOT["Paired bootstrap deltas<br/>evaluate.bootstrap_delta"]
        SHUF["Shuffled-graph control<br/>cli.run"]
        CEIL["Ceiling: true propensity<br/>cli.run"]
        SWEEP["Seed sweep, seeds 11 and 23<br/>scripts/seed_sweep.py"]
    end

    subgraph OUT["Outputs"]
        REPORT[("outputs/<br/>ablation.md · ablation.csv · deltas.csv<br/>feature_importance.csv · summary.json")]
    end

    GEN --> HCPS & RX & EDGES & KEY
    HCPS --> LOAD
    RX --> LOAD
    EDGES --> LOAD
    LOAD -->|"decile, volume, specialty, region"| PROF
    LOAD -->|"src, dst, edge_type, shared_patients"| GRAPH
    GRAPH -->|"weighted undirected graph"| LOUV
    GRAPH -->|"DiGraph + Graph"| STRUCT
    LOUV -->|"community id x 2 levels"| STRUCT
    LOUV -->|"community id x 2 levels"| LBL
    GRAPH -->|"sparse adjacency"| N2V
    GRAPH -->|"sparse adjacency"| SAGE
    PROF -->|"standardised node features"| SAGE
    PROF -->|"profile block"| XGB
    STRUCT -->|"structure block"| XGB
    N2V -->|"32-dim vectors"| XGB
    SAGE -->|"32-dim vectors"| XGB
    LOAD -->|"converted, training folds only"| LBL
    LBL -->|"rates per fold"| XGB
    LOAD -->|"decile, annual_class_rx"| RULE
    XGB -->|"out-of-fold score per HCP"| PLAN
    RULE -->|"rank"| PLAN
    PLAN --> MET
    XGB -->|"OOF scores, two arms"| BOOT
    KEY --> CEIL
    CEIL --> MET
    SHUF -.->|"HCPs permuted across nodes, re-fit"| XGB
    SWEEP -.->|"regenerate market, rerun everything"| GEN
    RX -.->|"missing months: nanmean, months_reported"| PROF
    LBL -.->|"thin community: shrink to base rate"| LBL
    MET --> REPORT
    BOOT --> REPORT
    SHUF --> REPORT
    XGB -->|"gain share by block"| REPORT

    classDef data fill:#7fa8ff,stroke:#3d6fd9,color:#0b1020
    classDef model fill:#b9a2ff,stroke:#7a5fd6,color:#0b1020
    classDef decision fill:#3fd8a8,stroke:#1f9c76,color:#0b1020
    classDef evaluation fill:#f2c46d,stroke:#b8902f,color:#0b1020
    classDef risk fill:#ff8a73,stroke:#cc5640,color:#0b1020

    class GEN,HCPS,RX,EDGES,LOAD,PROF,GRAPH,LOUV,STRUCT,REPORT data
    class N2V,SAGE,XGB model
    class LBL,KEY risk
    class RULE,PLAN decision
    class MET,BOOT,SHUF,CEIL,SWEEP evaluation
```

Solid edges are the data flow of one `make run`; dotted edges are the controls, the fallbacks
and the sweep that reruns everything. Red nodes are where an outcome can leak into a feature:
the fold-built conversion rates, and the answer key that only the ceiling row may read.

### Components

| Component | Responsibility | File |
|---|---|---|
| Data generator | 20,000 HCPs, 24 months of class prescriptions with gaps and spikes, co-treatment and referral ties, conversion outcomes, and the true propensity written to a separate answer key | [`scripts/generate_data.py`](scripts/generate_data.py), [`src/hcp_targeting/data.py`](src/hcp_targeting/data.py) |
| Profile features | What a decile list already has: decile, log volume, 3-month trend, months reported, years in practice, specialty, region | [`src/hcp_targeting/features.py`](src/hcp_targeting/features.py) (`profile_features`) |
| Graph and structure features | Directed and undirected graphs, PageRank, sampled betweenness, clustering, neighbour decile and PageRank, hub ties, Louvain at two resolutions | [`src/hcp_targeting/features.py`](src/hcp_targeting/features.py) (`build_graphs`, `louvain_communities`, `structural_features`) |
| Fold-built conversion rates | Community and neighbour conversion rates rebuilt per outer fold from training labels, with an inner 5-fold split for training rows and shrinkage to the base rate | [`src/hcp_targeting/features.py`](src/hcp_targeting/features.py) (`label_graph_features`) |
| Embeddings | node2vec walks and skip-gram written in numpy and torch; two-layer GraphSAGE (PyTorch Geometric `SAGEConv`) trained on tie vs random pair; neither sees a label | [`src/hcp_targeting/embeddings.py`](src/hcp_targeting/embeddings.py) |
| Ablation and metrics | The seven arms, XGBoost settings, stratified folds, AUC, top-decile lift, captured at budget, paired bootstrap | [`src/hcp_targeting/evaluate.py`](src/hcp_targeting/evaluate.py) |
| Orchestrator | One command end to end: data, features, embeddings, every arm, decile baseline, shuffled control, ceiling, report | [`src/hcp_targeting/cli.py`](src/hcp_targeting/cli.py) |
| Seed sweep | Regenerates the market with seeds 11 and 23 and reruns the ablation | [`scripts/seed_sweep.py`](scripts/seed_sweep.py) |

### Design decisions

| Decision | Options considered | Chosen | Trade-off |
|---|---|---|---|
| How outcome-based graph features are built | Global rates on all labels; leave-one-out target encoding; rebuilt inside each fold | Rebuilt per outer fold, inner 5-fold split for training rows ([ADR-0001](docs/adr/0001-graph-features-built-fold-safe-embeddings-label-free.md)) | No leak (LOO dropped held-out AUC to 0.589 here), but the features cannot be precomputed once and need last campaign's outcomes at scoring time |
| How embeddings are trained | Supervised GraphSAGE as the classifier; unsupervised embeddings fed to XGBoost | Unsupervised node2vec and GraphSAGE, trained once per graph, fed to the same XGBoost | Tests whether embeddings add to the hand-built features rather than competing with XGBoost; leaves a fold-safe supervised GraphSAGE untested |
| node2vec implementation | PyTorch Geometric `Node2Vec`; own walks and skip-gram | Own walks and skip-gram in numpy and torch; `SAGEConv` from PyTorch Geometric | PyG's `Node2Vec` needs torch-cluster or pyg-lib, which have no plain PyPI CPU wheels; the own version is tested for edge-following and backtracking |
| How arms are compared | Single train/test split; 5-fold out-of-fold on all HCPs with paired bootstrap intervals | 5-fold OOF, same folds for every arm, 200 paired bootstrap resamples | Five fits per arm, but a 0.005 AUC gap is no longer reported as a win; a single split moved the 5% capture count by 10+ |
| Proof the lift is the network | None; random feature columns; HCPs shuffled across the same graph | Shuffled-graph control on the graph-features and full arms | Keeps the graph's structure and breaks the person-to-position link; costs two more arms per run |
| Community granularity | Default Louvain (resolution 1); a finer level added | Both, resolution 1 (35 blocks) and 16 (408 communities) | Resolution 16 matches practice size; picked on structure, not tuned on conversion |

### Data contracts

| Stage | Input | Output |
|---|---|---|
| Generator to loader | `n_hcps`, `seed` | `hcps.csv`: `hcp_id` (int), `specialty`, `region` (str), `years_in_practice` (float, 7% missing), `annual_class_rx` (float), `decile` (int 1-10, equal volume shares), `converted` (0/1); `monthly_class_rx.csv`: `hcp_id` plus one float column per month `2024-07` to `2026-06` (1% missing); `edges.csv`: `src, dst` (int), `edge_type` (`co_treatment` or `referral`), `shared_patients` (int); `propensity_answer_key.npy`: float logit per HCP |
| Profile | `hcps`, monthly Rx | `decile, log_annual_rx, rx_trend_3m, months_reported, years_in_practice` plus one-hot `spec_*`, `region_*` (float) |
| Structure | directed and undirected graphs, community ids `(n, 2)` | `degree, referrals_in, referrals_out, shared_patients, pagerank, betweenness, clustering, nbr_mean_decile, nbr_mean_pagerank, nbr_max_pagerank, hub_ties, community_size_l0/l1, community_mean_decile_l0/l1` (float) |
| Fold-built rates | adjacency, community ids, `converted`, training indices | `community_conv_rate_l0, community_conv_rate_l1, nbr_conv_rate, nbr_converted` (float), one frame per fold |
| Embeddings | sparse adjacency (CSR); standardised profile matrix for GraphSAGE | `n2v_0..31`, `sage_0..31` (float) |
| Ablation | feature blocks per arm, `converted`, 5 stratified folds | per arm: OOF score per HCP, fold AUCs, gain share per column; `ablation.csv`: `arm, auc, top_decile_lift, captured_5pct, captured_20pct, auc_fold_sd`; `deltas.csv`: `arm, vs, metric, delta, ci_low, ci_high` |

### Failure modes and fallbacks

| Failure | What the code does |
|---|---|
| Prescription months missing for an HCP (1% of HCP-months) | Volume and trend use `nanmean` over the months present, and `months_reported` goes into the model as its own feature, so a gap is visible rather than read as zero. |
| `years_in_practice` missing (7% of HCPs) | XGBoost receives the NaN and learns its own split direction. For the GraphSAGE input the column is median-filled and a `_missing` indicator is added before standardising. |
| Community or neighbourhood with few labelled HCPs | Conversion rates are shrunk toward the training base rate with a prior weight of 20, so a community with two known HCPs does not get a rate of 0 or 1. An HCP with no ties gets the base rate and zero converted neighbours. |
| Answer key absent | `data.load` fills the true propensity with NaN and the ceiling row is skipped; every model arm still runs. |
| Betweenness too slow on a large graph | Computed from 256 sampled sources (64 in the seed sweep and CI) rather than exactly. |
| Outcome leaking into a feature | Guarded by construction and pinned by `tests/test_leakage.py`, which flips each HCP's own label and asserts its own feature row does not move. |
| Model degrades or the market drifts | Not handled. Every run retrains from scratch on one snapshot; there is no drift monitoring or champion/challenger. Listed under limitations. |
| Solver or LLM timeout | Not applicable: there is no solver, LLM, model server or network call in the pipeline. |

### Path to production

This is a mapping of each local piece to what would replace it in a deployed system, not a
description of one that exists.

| Here | Production equivalent |
|---|---|
| Seeded generator writing CSVs to `data/` | Licensed prescription and claims feeds landed in a warehouse (BigQuery, Snowflake or Redshift); shared-patient and referral ties derived there with SQL, refreshed monthly |
| NetworkX graph and Louvain in memory | Same code on a larger machine up to a few hundred thousand HCPs; beyond that, a graph engine (Neo4j GDS or Spark GraphFrames) for PageRank, betweenness and community detection |
| `make run` on a laptop | A monthly pipeline in Airflow or Step Functions: refresh ties, recompute structure and embeddings, rebuild conversion rates from the last campaign's outcomes, score, publish |
| XGBoost retrained per run, nothing stored | Versioned model and feature definitions in a registry (MLflow, SageMaker or Vertex AI), promoted only if they beat the incumbent on a held-out cycle, with the shuffled-graph control kept as a release check |
| Out-of-fold scores in `outputs/` | Batch scores written back to the CRM call-planning table, capped by rep capacity per territory |
| `outputs/ablation.md` and `deltas.csv` | A campaign dashboard comparing the model's call list against the decile list, with a holdout of territories that keep the decile list so the lift is measured, not assumed |
| Monitoring (not built) | Conversion rate by score decile, predicted vs observed, per cycle; feature drift on degree, community size and prescription trend; share of HCPs with no ties or no label history; an alert when top-decile lift falls below the decile list's |

## The problem

A pharmaceutical brand launching into a crowded therapy class has a few hundred field reps
and a fixed number of calls per cycle. The call list is almost always built from
prescription-volume deciles: rank healthcare professionals (HCPs) by how much they write in
the class, call the top deciles first. It is simple and defensible, and it misses the reason
many HCPs actually adopt a new therapy. Doctors adopt when the people they share patients
with adopt: the practice they work in, the specialist they refer to, the key opinion leader
down the corridor. Heavy writers are also often the most entrenched on the incumbent drug.

Every call spent on a high-decile HCP who will not move is a call not spent on a mid-decile
HCP whose practice is already converting. Over a launch year that is the difference between
a campaign that finds its early adopters and one that spends its budget confirming the
market's biggest loyalists.

## Why it is hard

The network is available (shared-patient and referral links come out of claims data), so the
obstacle is not access. It is that graph features are easy to get wrong in a way that looks
like success. Any feature built from outcomes (how many of my neighbours converted) leaks the
label if it is computed carelessly, and the first version here did exactly that, losing 0.05
AUC on held-out HCPs while training fit improved. The second obstacle is cost: node
embeddings are expensive to train and explain, and a commercial team needs to know whether
they add anything over a PageRank score and a community conversion rate before paying for them.

## The approach

On a 20,000-HCP network with 75,155 co-treatment and referral ties, one XGBoost model is
trained per feature set, from deciles alone up to deciles plus hand-built graph features plus
node2vec and GraphSAGE embeddings. Every arm is scored on the same 5-fold out-of-fold
predictions, against the plain decile list reps use today, under a fixed call budget.

The key decision: features that use outcomes (community and neighbour conversion rates) are
rebuilt inside every fold from training labels only, with an inner fold split so no HCP ever
sees its own outcome, and the embeddings are trained without labels at all. A shuffled-graph
control (every HCP moved to a random node of the same graph) checks that whatever lift
remains really comes from the network. [ADR-0001](docs/adr/0001-graph-features-built-fold-safe-embeddings-label-free.md)
records why.

## The result

20,000 HCPs, 1,617 conversions (8.1%), 5-fold out-of-fold scores, seed 7. "Captured" is the
number of converters inside a call plan of 1,000 HCPs (5%) or 4,000 HCPs (20%).

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

Paired bootstrap 95% intervals on the same HCPs (`outputs/deltas.csv`):

| arm | vs | AUC delta [95% CI] | captured @ 5% delta [95% CI] |
|---|---|---|---|
| + centrality and community structure | profile + decile | +0.0164 [+0.0084, +0.0260] | +22 [-4, +43] |
| + centrality and community structure | graph features | -0.0105 [-0.0177, -0.0028] | -1 [-22, +19] |
| + community and neighbour conversion rates | profile + decile | +0.0269 [+0.0181, +0.0370] | +23 [+1, +50] |
| graph features + node2vec | profile + decile | +0.0347 [+0.0238, +0.0451] | +36 [+13, +68] |
| graph features + node2vec | graph features | +0.0078 [-0.0001, +0.0141] | +13 [-7, +34] |
| graph features + GraphSAGE | profile + decile | +0.0244 [+0.0140, +0.0353] | +28 [+7, +55] |
| graph features + GraphSAGE | graph features | -0.0025 [-0.0075, +0.0026] | +5 [-15, +23] |
| graph features + node2vec + GraphSAGE | profile + decile | +0.0324 [+0.0217, +0.0437] | +46 [+16, +74] |
| graph features + node2vec + GraphSAGE | graph features | +0.0055 [-0.0019, +0.0119] | +23 [-5, +42] |
| profile + embeddings only (no hand-built graph features) | profile + decile | +0.0170 [+0.0063, +0.0268] | +47 [+22, +75] |
| profile + embeddings only (no hand-built graph features) | graph features | -0.0099 [-0.0201, -0.0008] | +24 [-4, +49] |

What this says:

- **Graph features help, and the control says it is the network.** Adding centrality,
  community and neighbour conversion rates to the profile-and-decile model lifts AUC from
  0.641 to 0.668 (+0.027, interval +0.018 to +0.037) and captures 189 converters instead of
  166 with 1,000 calls. Shuffling HCPs across the same graph drops both graph arms to 0.633,
  below the profile model, so the gain is not an artefact of extra columns.
- **Embeddings add little.** On top of the hand-built graph features, node2vec adds +0.008
  AUC with an interval that touches zero, GraphSAGE adds nothing (-0.003), and both together
  add +0.006. Their capture gains at the 5% budget (+13, +5, +23) all have intervals that
  include zero. Yet in the full model the two embedding blocks take 61% of total split gain.
  Gain share measures how often the trees split on a column, not what the column added: the
  embeddings mostly re-encode community membership that the hand-built features already carry.
- **The metrics disagreed once, and it did not hold.** On seed 7, embeddings without any
  hand-built graph features scored significantly lower AUC than the graph-features model
  (-0.010, interval -0.020 to -0.001) but captured the most converters at the 5% budget
  (213 against 189). Rerunning on two freshly generated networks (below) reversed it: 153
  against 192 and 204 against 249. At 1,000 calls one run's capture ranking is noise; the
  AUC ordering was the stable one.
- **Most of the gap to the ceiling is still open.** The generator's true propensity reaches
  AUC 0.761 and would capture 322 converters with the same 1,000 calls. Going from the decile
  list (126) to the best model (212) closes 44% of that gap.

### Does it survive a new network?

`scripts/seed_sweep.py` regenerates the market with seeds 11 and 23 and reruns everything
(betweenness on 64 sampled sources instead of 256 to save time). From `outputs/seed_sweep.txt`:

```
AUC by seed
arm   profile  graph features  + node2vec  + GraphSAGE  + both  embeddings only
seed                                                                           
7       0.641           0.668       0.676        0.666   0.674            0.658
11      0.638           0.663       0.665        0.662   0.666            0.652
23      0.620           0.659       0.666        0.658   0.666            0.647

Conversions captured with a 5% call budget, by seed
arm   profile  graph features  + node2vec  + GraphSAGE  + both  embeddings only
seed                                                                           
7         166             189         202          194     212              213
11        119             192         197          186     203              153
23        166             249         254          230     253              204

AUC gain over graph features, by seed
arm   profile  graph features  + node2vec  + GraphSAGE  + both  embeddings only
seed                                                                           
7     -0.0269             0.0      0.0078      -0.0025  0.0055          -0.0099
11    -0.0247             0.0      0.0016      -0.0009  0.0034          -0.0109
23    -0.0388             0.0      0.0069      -0.0005  0.0074          -0.0117
```

Graph features beat the profile model by 0.025 to 0.039 AUC on every seed. node2vec adds a
small, consistent +0.002 to +0.008; GraphSAGE adds nothing on any seed. With three seeds that
is consistent, not proven.

## How it works

```
data/hcps.csv, edges.csv, monthly_class_rx.csv
      │
      ├─ profile:   decile (volume-share deciles), annual class Rx, 3-month trend,
      │             months reported, years in practice, specialty, region
      ├─ structure: degree, referrals in/out, PageRank (directed, weighted),
      │             betweenness (256 sampled sources), clustering, neighbour decile and
      │             PageRank, ties to top-1% hubs, Louvain at two resolutions
      ├─ fold-built: community conversion rate (both levels), neighbour conversion rate
      │             and count — training labels only, inner 5-fold for training rows
      ├─ node2vec:  p=1, q=0.5, 5 walks × 16 steps, skip-gram window 3, 32 dims
      └─ GraphSAGE: 2 SAGEConv layers (PyTorch Geometric), tie-vs-random-pair loss, 32 dims
      │
      ▼
  XGBoost per arm, 5-fold stratified CV ─► AUC, top-decile lift, captured @ 5% / 20%
      │                                    paired bootstrap deltas, shuffled-graph control
      ▼
  outputs/ablation.md, ablation.csv, deltas.csv, feature_importance.csv, summary.json
```

Louvain runs at resolution 1 (35 regional blocks) and 16 (408 communities of about 50 HCPs).
Default Louvain cannot see communities as small as a practice group, so the finer level was
added; 16 was picked to match practice size, not tuned on conversion.

## Running it

```bash
uv sync
make data                 # writes data/ (seeded; 20,000 HCPs)
make run                  # regenerates data and runs the full ablation into outputs/
uv run python scripts/seed_sweep.py   # same ablation on two more generated networks
make check                # ruff, format, mypy, pytest
uv run hcp-targeting --n-hcps 3000 --folds 3 --betweenness-k 64   # quick run, about a minute
docker build -t hcp-targeting . && docker run --rm hcp-targeting
```

No API key, model server or GPU is needed. On Linux, torch resolves to the CPU wheel index.
A full run took 403 s on this 10-core laptop in an earlier session; the captured run in `outputs/`
took 3.4 hours because the machine was shared with other heavy jobs at the time.

## The data

The dataset is generated by the seeded script in `scripts/generate_data.py` (`make data`),
sized to the production problem of a regional HCP universe; real HCP prescribing and claims
data is proprietary and is not used here.

It has 20,000 HCPs in eight specialties across 12 regions and about 440 practice groups,
24 months of therapeutic-class prescriptions (winter seasonality, December and August dips,
new-to-practice gaps, 1% missing months, a few six-fold spikes), and 75,155 ties: symmetric
co-treatment ties inside practices, directed referrals from generalists to specialists in the
same region, and a thin layer of cross-region ties that doubles as claims-matching noise.
Years in practice is missing for 7% of HCPs. Deciles follow the industry convention of equal
volume shares, so decile 10 holds a few hundred HCPs and decile 1 several thousand.

Conversion depends on volume (with a bend at the top for entrenched heavy writers),
specialty, region, years in practice, a hidden practice-group culture and hidden exposure to
key opinion leaders who advocate the therapy. Neither hidden driver is written to the
modelling files; the true propensity goes to a separate answer-key file that only the
ceiling row reads. Because the generator decides how much the network matters, these results
show that the pipeline finds network signal when it exists and does not invent it when it is
shuffled away. They do not show how much network signal a real market has.

## Tests

`tests/test_leakage.py` is the important one. It flips each sampled HCP's own outcome, on both
the training and held-out side, and asserts that the HCP's own label-derived features do not
change; it also scrambles every held-out label and asserts no feature moves. A third test
documents the leave-one-out bug: inside every mixed community, the LOO rate is lower for
exactly the converters.

`tests/test_graph.py` checks that every node2vec step follows a real edge, that a low return
parameter makes walks backtrack and a high one stops it, and that GraphSAGE embeddings tell
real ties from random pairs. `tests/test_metrics.py` pins capture counts and lift on
hand-built inputs and checks that deciles split volume, not headcount.
`tests/test_pipeline.py` runs the whole ablation at 1,500 HCPs.

## Known limitations

- **The biggest gap: one generated market.** The size of the graph lift is set by how the
  generator wires practice culture and opinion-leader exposure into conversion. A real
  market could have more network signal or almost none.
- Three seeds is a small sweep. Arm gaps under 0.01 AUC and under ~25 captures at the 5%
  budget are inside run-to-run noise, and the seed-7 capture ranking already flipped once.
- Static snapshot. Real adoption spreads over time; a production model would use neighbours'
  conversions from earlier months, not the same campaign window.
- No call response model. The budget metric assumes a call is what converts the HCP; it
  ranks propensity, not uplift from calling.
- Fold-built features need the previous campaign's outcomes at scoring time.
- The Docker image is built and run in CI only; the local Docker daemon failed with a
  storage I/O error, so it was never built on this machine.

## Insights and learning

I picked this because decile targeting is the default in pharma commercial teams and graph
ML is the most common proposed replacement, usually with embeddings at the centre. The thing
I wanted to know was not whether a graph helps but which part of it pays for itself.

The surprise was how badly a leave-one-out community rate leaks. It is the textbook way to
target-encode, and on its own it would be fine; with community size beside it, the trees
reconstruct the HCP's own label and held-out AUC collapsed below the no-graph model. The
inner-fold scheme fixed it, and a test now pins the property instead of the implementation.

What I got wrong at first was reading feature importance as contribution. The embeddings
dominate split gain in the full model and add almost nothing measurable. If I took this
further I would replace the static snapshot with a time-stamped adoption cascade, add an
uplift model so the call budget goes to persuadable HCPs rather than likely ones, and test
whether a supervised GraphSAGE trained inside each fold beats the hand-built features.

## License

MIT
