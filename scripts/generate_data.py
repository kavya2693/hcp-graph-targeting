"""Write the HCP network dataset to data/ (seeded, so every run gives the same files)."""

from __future__ import annotations

import argparse
from pathlib import Path

from hcp_targeting.data import generate, save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-hcps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    ds = generate(args.n_hcps, args.seed)
    save(ds, args.out)
    h = ds.hcps
    print(
        f"hcps={len(h)} edges={len(ds.edges)} conversion_rate={h['converted'].mean():.3f} "
        f"isolates={len(h) - len(set(ds.edges['src']) | set(ds.edges['dst']))}"
    )


if __name__ == "__main__":
    main()
