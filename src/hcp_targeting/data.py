"""Seeded generator for an HCP co-treatment and referral network with a launch outcome.

The generator is written so that the outcome depends on things a decile cannot see
(practice-group culture, exposure to advocate key opinion leaders) as well as on volume,
and none of those hidden drivers is written to disk. Only what a commercial team would
actually hold goes into the modelling files: profile, monthly class prescriptions,
shared-patient edges and whether the HCP converted during the campaign window. The true
propensity is saved to a separate answer-key file that only the report's ceiling row reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# share of HCPs, mean log monthly class Rx, relevance to the launched therapy, refers out
SPECIALTIES: dict[str, tuple[float, float, float, bool]] = {
    "primary_care": (0.33, 2.9, 0.0, True),
    "internal_medicine": (0.16, 3.0, 0.1, True),
    "nurse_practitioner": (0.12, 2.4, -0.1, True),
    "cardiology": (0.10, 3.3, 0.4, False),
    "endocrinology": (0.07, 3.5, 0.6, False),
    "nephrology": (0.05, 3.1, 0.3, False),
    "geriatrics": (0.05, 2.8, -0.2, True),
    "other": (0.12, 1.7, -0.8, True),
}
SPECIALISTS = ("cardiology", "endocrinology", "nephrology")
N_REGIONS = 12
N_MONTHS = 24
LAST_MONTH = "2026-06"


@dataclass
class Dataset:
    hcps: pd.DataFrame
    monthly_rx: pd.DataFrame
    edges: pd.DataFrame
    # Answer key for the ceiling row in the report; no feature is built from it.
    true_logit: np.ndarray


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def volume_deciles(volume: np.ndarray) -> np.ndarray:
    """Industry-style deciles: each decile holds about a tenth of total volume, not of HCPs.

    Decile 10 is the small group of heaviest writers; HCPs with no class volume get 0.
    """
    vol = np.nan_to_num(volume, nan=0.0)
    order = np.argsort(vol, kind="stable")
    cum = np.cumsum(vol[order])
    total = cum[-1] if cum[-1] > 0 else 1.0
    dec_sorted = np.clip(np.ceil(cum / total * 10), 1, 10).astype(int)
    dec = np.empty_like(dec_sorted)
    dec[order] = dec_sorted
    dec[vol <= 0] = 0
    return dec


def _sample_partners(
    rng: np.random.Generator, members: np.ndarray, weights: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each member draw counts[i] partners from `members`, weighted by attractiveness."""
    total = int(counts.sum())
    if total == 0 or len(members) < 2:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    p = weights / weights.sum()
    src = np.repeat(members, counts)
    dst = rng.choice(members, size=total, p=p)
    return src, dst


def generate(n_hcps: int = 20_000, seed: int = 7) -> Dataset:
    rng = np.random.default_rng(seed)
    n = n_hcps
    names = list(SPECIALTIES)
    shares = np.array([SPECIALTIES[s][0] for s in names])
    spec_idx = rng.choice(len(names), size=n, p=shares / shares.sum())
    specialty = np.array(names)[spec_idx]
    is_specialist = np.isin(specialty, SPECIALISTS)
    refers_out = np.array([SPECIALTIES[s][3] for s in specialty])

    # Practice groups (hospital systems, multi-site practices) nested in regions, heavy-tailed size.
    n_groups = max(n // 45, 4)
    group_region = rng.integers(0, N_REGIONS, size=n_groups)
    group_size_w = rng.lognormal(0.0, 0.8, size=n_groups)
    group = rng.choice(n_groups, size=n, p=group_size_w / group_size_w.sum())
    region = group_region[group]

    # Hidden drivers: key opinion leaders, which of them advocate the therapy, group culture.
    is_kol = is_specialist & (rng.random(n) < 0.08)
    is_advocate = is_kol & (rng.random(n) < 0.5)
    group_openness = rng.normal(0.0, 0.75, size=n_groups)
    region_effect = rng.normal(0.0, 0.2, size=N_REGIONS)

    years = np.clip(rng.gamma(3.0, 5.5, size=n), 0.5, 45.0).round(1)

    # Monthly therapeutic-class prescriptions: seasonality, holiday dips, gaps and spikes.
    log_mu = np.array([SPECIALTIES[s][1] for s in specialty])
    base = np.exp(rng.normal(log_mu + 0.6 * is_kol, 0.75))
    months = pd.period_range(end=LAST_MONTH, periods=N_MONTHS, freq="M")
    moy = np.array([m.month for m in months])
    season = 1.0 + 0.08 * np.cos(2 * np.pi * (moy - 1) / 12)
    season = season * np.where(moy == 12, 0.85, 1.0) * np.where(moy == 8, 0.92, 1.0)
    trend = np.exp(rng.normal(0.0, 0.15, size=n))[:, None] ** (np.arange(N_MONTHS) / N_MONTHS)
    lam = base[:, None] * season[None, :] * trend
    rx = rng.poisson(lam).astype(float)
    # New-to-practice HCPs have no history before they started.
    start = np.clip(((2.0 - years) * 12).round().astype(int), 0, N_MONTHS)
    rx[np.arange(N_MONTHS)[None, :] < start[:, None]] = np.nan
    spikes = rng.random(n) < 0.003
    rx[spikes, rng.integers(0, N_MONTHS, size=int(spikes.sum()))] *= 6
    rx[rng.random(rx.shape) < 0.01] = np.nan

    trailing = np.nansum(rx[:, -12:], axis=1)
    observed = np.sum(~np.isnan(rx[:, -12:]), axis=1)
    annual = np.where(observed > 0, trailing * 12 / np.maximum(observed, 1), 0.0)
    decile = volume_deciles(annual)

    # Edges. Co-treatment inside practice groups, referrals to specialists inside a region,
    # and a thin layer of cross-region ties that also stands in for claims-matching noise.
    attract = np.where(is_kol, 6.0, np.where(is_specialist, 2.0, 1.0))
    srcs, dsts, kinds = [], [], []
    order = np.argsort(group, kind="stable")
    bounds = np.searchsorted(group[order], np.arange(n_groups + 1))
    for g in range(n_groups):
        members = order[bounds[g] : bounds[g + 1]]
        s, d = _sample_partners(rng, members, attract[members], rng.poisson(2.6, size=len(members)))
        srcs.append(s)
        dsts.append(d)
        kinds.append(np.zeros(len(s), dtype=int))
    ref_attract = np.where(is_kol, 15.0, 1.0)
    for r in range(N_REGIONS):
        in_r = region == r
        senders = np.flatnonzero(in_r & refers_out)
        targets = np.flatnonzero(in_r & is_specialist)
        if len(targets) == 0 or len(senders) == 0:
            continue
        counts = rng.poisson(1.6, size=len(senders))
        p = ref_attract[targets] / ref_attract[targets].sum()
        srcs.append(np.repeat(senders, counts))
        dsts.append(rng.choice(targets, size=int(counts.sum()), p=p))
        kinds.append(np.ones(int(counts.sum()), dtype=int))
    k = rng.poisson(0.15, size=n)
    srcs.append(np.repeat(np.arange(n), k))
    dsts.append(rng.integers(0, n, size=int(k.sum())))
    kinds.append(np.full(int(k.sum()), 2))

    src = np.concatenate(srcs)
    dst = np.concatenate(dsts)
    kind = np.concatenate(kinds)
    keep = src != dst
    src, dst, kind = src[keep], dst[keep], kind[keep]
    # Co-treatment ties are symmetric, so collapse both orientations; referrals stay directed.
    sym = kind != 1
    a = np.where(sym, np.minimum(src, dst), src)
    b = np.where(sym, np.maximum(src, dst), dst)
    edges = pd.DataFrame({"src": a, "dst": b, "kind": kind}).drop_duplicates(["src", "dst"])
    edge_type = np.array(["co_treatment", "referral", "co_treatment"])
    edges["edge_type"] = edge_type[edges["kind"].to_numpy()]
    base_w = np.where(edges["kind"].to_numpy() == 1, 5, 3)
    edges["shared_patients"] = 1 + rng.poisson(base_w)
    edges = edges.drop(columns="kind").reset_index(drop=True)

    # Exposure to advocate KOLs through any tie, in either direction.
    e_src = edges["src"].to_numpy()
    e_dst = edges["dst"].to_numpy()
    adv_ties = np.bincount(e_src, weights=is_advocate[e_dst], minlength=n) + np.bincount(
        e_dst, weights=is_advocate[e_src], minlength=n
    )
    exposure = 1.0 - np.exp(-0.9 * adv_ties)

    z = (np.log1p(annual) - np.log1p(annual).mean()) / np.log1p(annual).std()
    # Heaviest writers are often entrenched on the incumbent therapy, so the volume effect bends.
    vol_effect = 0.45 * z - 0.30 * np.clip(z - 0.8, 0, None) ** 2
    relevance = np.array([SPECIALTIES[s][2] for s in specialty])
    logit = (
        -3.05
        + vol_effect
        + relevance
        + group_openness[group]
        + region_effect[region]
        + 1.3 * exposure
        + 0.9 * is_advocate
        - 0.012 * (years - 15)
    )
    converted = (rng.random(n) < _sigmoid(logit)).astype(int)

    # Years in practice is missing for a slice of records, as in most licensing feeds.
    years_obs = np.where(rng.random(n) < 0.07, np.nan, years)
    hcps = pd.DataFrame(
        {
            "hcp_id": np.arange(n),
            "specialty": specialty,
            "region": [f"R{r:02d}" for r in region],
            "years_in_practice": years_obs,
            "annual_class_rx": annual.round(1),
            "decile": decile,
            "converted": converted,
        }
    )
    monthly = pd.DataFrame(rx, columns=[str(m) for m in months])
    monthly.insert(0, "hcp_id", np.arange(n))
    return Dataset(hcps=hcps, monthly_rx=monthly, edges=edges, true_logit=logit)


def save(ds: Dataset, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.hcps.to_csv(out_dir / "hcps.csv", index=False)
    ds.monthly_rx.to_csv(out_dir / "monthly_class_rx.csv", index=False)
    ds.edges.to_csv(out_dir / "edges.csv", index=False)
    np.save(out_dir / "propensity_answer_key.npy", ds.true_logit)


def load(data_dir: Path) -> Dataset:
    hcps = pd.read_csv(data_dir / "hcps.csv")
    monthly = pd.read_csv(data_dir / "monthly_class_rx.csv")
    edges = pd.read_csv(data_dir / "edges.csv")
    logit_path = data_dir / "propensity_answer_key.npy"
    logit = np.load(logit_path) if logit_path.exists() else np.full(len(hcps), np.nan)
    return Dataset(hcps=hcps, monthly_rx=monthly, edges=edges, true_logit=logit)
