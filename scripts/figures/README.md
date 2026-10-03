# Figure scripts

`simbench_style_figures.py` draws the three figures in the paper from the leaderboard CSVs. It
performs no analysis of its own: every plotted value is read from a CSV that `reproduce.sh` writes
earlier, so each figure traces back to a file in `results/leaderboard/`.

| Figure | Output | Inputs |
|---|---|---|
| Internal validity by psychological state, leading models | `results/figures/fig_bycriterion.{pdf,png}` | `results/leaderboard/leaderboard.csv` |
| Both validity axes against model scale | `results/figures/fig_scaling.{pdf,png}` | `leaderboard.csv`, `nomological_leaderboard.csv`, `data/seeds/model_roster_scaling_live.csv` |
| Internal against external validity, one point per model | `results/figures/fig_tradeoff.{pdf,png}` | `leaderboard.csv`, `nomological_leaderboard.csv` |

Model identifiers in the roster (`google/gemma-3-27b-it`) are normalised to the leaderboard form
(`GEMMA_3_27B_IT`) inside the script. The palette is colorblind-safe (Wong, 2011).

```bash
.venv/bin/python scripts/figures/simbench_style_figures.py
```
