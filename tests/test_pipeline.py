import argparse
import json
from pathlib import Path

import pandas as pd

from hcp_targeting.cli import run


def test_end_to_end_run_writes_every_report_file(tmp_path: Path) -> None:
    args = argparse.Namespace(
        n_hcps=1500, seed=3, data_dir=tmp_path / "data", out=tmp_path / "out", refresh=True,
        folds=3, dim=8, betweenness_k=32, budget_small=0.05, budget_large=0.2, n_boot=20, threads=2,
    )  # fmt: skip
    summary = run(args)
    out = tmp_path / "out"
    for name in ("ablation.csv", "ablation.md", "deltas.csv", "feature_importance.csv", "summary.json"):
        assert (out / name).exists(), name
    table = pd.read_csv(out / "ablation.csv")
    assert table["arm"].str.startswith("control").sum() == 2
    assert table["auc"].between(0.0, 1.0).all()
    saved = json.loads((out / "summary.json").read_text())
    assert saved["n_hcps"] == summary["n_hcps"] == 1500
