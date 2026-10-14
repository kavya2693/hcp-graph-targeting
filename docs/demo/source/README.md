# Video and audit report sources

Everything needed to rebuild the silent walkthrough (`docs/demo/demo.mp4`) and the audit report
(`docs/demo/AUDIT_REPORT.*`) from this repository alone. Paths below are relative to the repo root.

## What is here

| Path | What it is |
|---|---|
| `index.html` | Root of the video: theme (colours, type), and the timing of every slide |
| `compositions/bg.html`, `compositions/chrome.html` | Background grid, and the persistent project label, slide counter and progress bar |
| `compositions/s01-context.html` | Slide 01, the industry and four everyday problems |
| `compositions/s02-hook.html` | Slide 02, the project in one sentence |
| `compositions/explainer.html` | Slides 03-04, the white "how it works" chapter (edit its `CONFIG` block) |
| `compositions/s03-architecture.html` | Slide 05, the architecture diagram (`FRAMES`, `N` components and `W` edges) |
| `compositions/s04-routes.html` | Slide 06, the network sample and the ablation ladder (`NET` is injected, `LADDER` is from `outputs/ablation.csv`) |
| `compositions/value.html` | Slide 07, the call plan and the field-rep scene (data injected by `tools/inject_sheet.py`) |
| `compositions/s05-results.html`, `s06-twist.html`, `s07-end.html` | Slides 08-10, results, what didn't win, end card |
| `assets/call_plan.json` | Call plan (top 1,000 HCPs), slide 06 network sample and slide 07 community, exported by `tools/export_call_plan.py` |
| `audit/facts.yaml` | Every figure (F-xx), assumption (A-xx), slide text and check expression used by the report |
| `tools/build_audit_report.py` | Builds the slide stills, re-checks every figure against the repo outputs, writes MD, DOCX and PDF |
| `tools/export_call_plan.py` | Re-runs the repo's own graph-features arm (same data, seed, folds, settings as `make run`), asserts AUC and captured counts match `outputs/ablation.csv`, and exports the JSON |
| `tools/inject_sheet.py` | Inlines that JSON into `compositions/value.html`, `s04-routes.html` and `s07-end.html` (renders cannot fetch files) |
| `package.json`, `hyperframes.json`, `meta.json` | HyperFrames project files (CLI pinned to 0.8.156) |

The repo writes no plots, so there is no image-conversion step: every chart in the video is drawn from
numbers in `outputs/` or from the exported JSON. Fonts (Inter, JetBrains Mono, Montserrat) are referenced
by name and embedded by the HyperFrames renderer; all three are SIL Open Font License.

## Requirements

- Node 22 and `npx` (HyperFrames 0.8.156 is fetched by `npx`), plus Chrome/Chromium for rendering.
- `ffmpeg` for compressing the video and extracting slide stills.
- LibreOffice (`soffice`) to convert the report to PDF.
- The repo's own environment (`uv sync`) for `make run` and `export_call_plan.py`.
- For `build_audit_report.py` and `inject_sheet.py` (any Python 3.11+, not added to the repo's
  `pyproject.toml`): `pip install pyyaml pandas python-docx pillow numpy matplotlib`.

## Rebuild the numbers, the video and the report

```bash
# 1. repo outputs
make run                                   # data/ + outputs/ablation.csv, deltas.csv, summary.json
uv run python scripts/seed_sweep.py        # outputs/seed_sweep.csv (slide 08 footnote)

# 2. inputs for the video (about 30 s)
.venv/bin/python docs/demo/source/tools/export_call_plan.py docs/demo/source/assets/call_plan.json
python3 docs/demo/source/tools/inject_sheet.py docs/demo/source/assets/call_plan.json \
  docs/demo/source/compositions --rows 1,2,3,4,5,6

# 3. render (90 s, 1920x1080, 30 fps, silent) and compress
cd docs/demo/source
npx --yes hyperframes@0.8.156 check
npx --yes hyperframes@0.8.156 render -f 30 -q looks -o renders/hcp-graph-targeting-master.mp4
cd -
ffmpeg -y -i docs/demo/source/renders/hcp-graph-targeting-master.mp4 -c:v libx264 -preset slow -crf 25 \
  -pix_fmt yuv420p -movflags +faststart -an docs/demo/demo.mp4

# GIFs: the end-user value slide (12 s) and the architecture data flow (8 s)
F="fps=12,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
ffmpeg -y -ss 56.9 -t 12 -i docs/demo/source/renders/hcp-graph-targeting-master.mp4 -vf "$F" -loop 0 docs/demo/demo.gif
ffmpeg -y -ss 32.6 -t 8 -i docs/demo/source/renders/hcp-graph-targeting-master.mp4 -vf "$F" -loop 0 docs/demo/architecture.gif

# 4. audit report: slide stills, automated checks, AUDIT_REPORT.md/.docx/.pdf
python3 docs/demo/source/tools/build_audit_report.py docs/demo/source/audit/facts.yaml
```

`build_audit_report.py` exits with code 1 if any figure on screen no longer matches the repo outputs.
`export_call_plan.py` stops with an assertion error if its re-run of the graph-features arm does not
reproduce `outputs/ablation.csv` (AUC 0.668, 189 converters captured, 126 for the decile list).

The `renders/` folder (about 20 MB) is not committed; step 3 recreates it.
