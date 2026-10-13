import numpy as np

from hcp_targeting.data import volume_deciles
from hcp_targeting.evaluate import captured_at_budget, top_decile_lift


def test_captured_counts_conversions_in_the_call_plan() -> None:
    y = np.array([1, 0, 1, 0, 0, 1])
    score = np.array([0.9, 0.8, 0.1, 0.7, 0.2, 0.6])
    assert captured_at_budget(y, score, 2) == 1
    assert captured_at_budget(y, score, 4) == 2
    assert captured_at_budget(y, score, 6) == 3


def test_top_decile_lift_is_one_for_random_and_max_for_perfect() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(20_000) < 0.1).astype(int)
    assert abs(top_decile_lift(y, rng.random(len(y))) - 1.0) < 0.15
    assert abs(top_decile_lift(y, y + rng.random(len(y)) * 0.01) - 1 / y.mean()) < 0.2


def test_deciles_split_volume_not_headcount() -> None:
    rng = np.random.default_rng(0)
    vol = rng.lognormal(3, 1.2, size=10_000)
    vol[:50] = 0
    dec = volume_deciles(vol)
    assert (dec[:50] == 0).all()
    shares = np.array([vol[dec == d].sum() for d in range(1, 11)]) / vol.sum()
    assert np.all(np.abs(shares - 0.1) < 0.03)
    # Volume deciles put far fewer HCPs in decile 10 than in decile 1.
    assert (dec == 10).sum() * 5 < (dec == 1).sum()
