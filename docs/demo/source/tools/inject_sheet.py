"""Inline the exported call plan and network sample into the compositions (renders cannot fetch files).

Writes the SHEET block of compositions/value.html (call-plan rows + the slide 07 community scene)
and the NET block of compositions/s04-routes.html and s07-end.html (the slide 06 network sample).

Usage: python3 tools/inject_sheet.py assets/call_plan.json compositions --rows 1,2,3,4,5,6
"""

import argparse
import json
import re
from pathlib import Path

ROW_KEYS = (
    "rank",
    "hcp",
    "specialty",
    "region",
    "decile",
    "decile_rank",
    "score",
    "practice_rate",
    "nbr_converted",
)

ap = argparse.ArgumentParser()
ap.add_argument("plan")
ap.add_argument("compositions")
ap.add_argument("--rows", default="1,2,3,4,5,6", help="plan ranks shown as sheet rows, in order")
a = ap.parse_args()

d = json.loads(Path(a.plan).read_text())
by_rank = {p["rank"]: p for p in d["plan"]}
ranks = [int(r) for r in a.rows.split(",")]
sheet = {
    "n_hcps": d["n_hcps"],
    "calls": d["calls"],
    "captured": d["captured"],
    "captured_decile_list": d["captured_decile_list"],
    "outside_decile_top": d["plan_from_outside_decile_top_1000"],
    "rows": [{k: by_rank[r][k] for k in ROW_KEYS} for r in ranks],
    "scene": d["scene"],
}
net = d["network_sample"]


def inject(path: Path, marker: str, name: str, obj: object) -> None:
    html = path.read_text()
    block = (
        f"/* {marker}:BEGIN */ const {name} = "
        + json.dumps(obj, separators=(",", ":"))
        + f"; /* {marker}:END */"
    )
    html, n = re.subn(rf"/\* {marker}:BEGIN \*/.*?/\* {marker}:END \*/", lambda _: block, html, flags=re.S)
    assert n == 1, f"{marker} marker not found in {path}"
    path.write_text(html)


comp = Path(a.compositions)
inject(comp / "value.html", "SHEET", "SHEET", sheet)
inject(comp / "s04-routes.html", "NET", "NET", net)
inject(comp / "s07-end.html", "NET", "NET", net)
print(
    "injected",
    len(sheet["rows"]),
    "rows,",
    len(sheet["scene"]["nodes"]),
    "scene nodes,",
    len(net["nodes"]),
    "network nodes",
)
