<h1 align="center">PsyDyBench</h1>

<p align="center"><b>Benchmarking Psychological Dynamics in Generative Agents</b></p>

<p align="center">
  <a href="#citation">📝 Paper</a> •
  <a href="#leaderboard">🏆 Leaderboard</a> •
  <a href="docs/DATA_CARD.md">💽 Data</a> •
  <a href="docs/METHOD.md">📖 Method</a> •
  <a href="docs/PROVENANCE.md">🔎 Provenance</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/code-MIT-blue.svg" alt="Code: MIT"></a>
  <a href="data/LICENSE"><img src="https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg" alt="Data: CC BY 4.0"></a>
  <a href="requirements.txt"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9+"></a>
  <img src="https://img.shields.io/badge/models-36-green.svg" alt="36 models">
  <img src="https://img.shields.io/badge/persona--days-1%2C800-green.svg" alt="1,800 persona-days">
  <img src="https://img.shields.io/badge/COLM%202026-AIMS%20Workshop-purple.svg" alt="COLM 2026 AIMS Workshop">
</p>

PsyDyBench is the code, data, and results release for **Benchmarking Psychological Dynamics in
Generative Agents** (Vaid and Whillans), published at the AI Measurement Science Workshop at COLM 2026.

## 📰 News

- **[2026-10-02]** PsyDyBench v1.0.0 released: 36 model corpora with raw responses, every leaderboard and figure in the paper, and a one-command reproduction that recomputes each reported number.

## 👋 Overview

Large language models are increasingly deployed to simulate human behavior, acting as computational
replicas of human subjects. PsyDyBench is a psychometric benchmark for such replicas: personas that
carry a fixed identity through an evolving sequence of events. Built entirely from published norms
and meta-analytic effects, it scores two dimensions of psychological realism.

**Design.** We initialize 300 synthetic personas from published norms. Each persona's Big Five traits
are drawn from a multivariate normal grounded in population norms and meta-analytic intercorrelations,
and its state baselines are linear functions of those traits using established coefficients. Daily
contextual indicators are sampled forward for six days while the state columns stay unpopulated, so any
recovered indicator-state relation is emergent rather than seeded. A fixed event model then writes an
identity portrait for each persona and a narrative for each persona-day. This happens exactly once.
Every benchmarked model processes these identical 2,100 texts under a uniform first-person prompt and
reports four workplace states at the end of each day: work engagement, psychological safety, workplace
creativity, and team trust. Because all models rate identical narratives, differences in ratings
reflect the model rather than the events.

**Internal validity** compares the synthetic corpus to published human norms on the statistical
properties of experience-sampling data, using two one-sided equivalence tests with persona-level
resampling. The criteria are the mean and spread of each state, the between- and within-person
variance partition, cross-state correlations, the within-person autoregressive coefficient, latent
dimensionality, and scale-range use. The specification is in [docs/INTERNAL_VALIDITY.md](docs/INTERNAL_VALIDITY.md).

**External validity** examines whether the relationships in the synthetic data match those established
in human research. Between persons, we test whether a persona's traits predict its average states, as
meta-analyses report. Within persons, we test whether a persona's daily context predicts the state it
reports that day, against 118 published meta-analytic indicator-state effects. The anchor library and
its audit are in [docs/ANCHOR_SOURCES.md](docs/ANCHOR_SOURCES.md).

**Findings.** Across 36 open-weight and proprietary models from nine developers (1B to 671B
parameters), most recover the *direction* of established relations (84.3% mean trait-state agreement)
and the variance partition (26 of 34), yet the *within-person correlation* and distributional structure
elude recovery: mean within-person |r| is 0.023 against a human band of 0.20 to 0.70. Internal validity
is independent of scale and capability. Gemma-2-27B has the smallest within-person variance error
(0.043), GPT-5.5 is at 0.061, and Gemma-3-27B strikes the best trade-off between the two dimensions.

## 🏆 Leaderboard

Thirty-four models ranked by within-person variance error: the absolute log-ratio of the synthetic
within-person standard deviation to the target human standard deviation, averaged over the four states.
Lower is closer to the human within-person variance structure. Two further corpora ship in the release
and are excluded here as degenerate: `mistral-small-24b-instruct-2501` produced one usable row, and
`qwen3.5-9b` has a variance error of 3.33.

Sources: `results/leaderboard/leaderboard.csv`, `nomological_leaderboard.csv`, `anchoring_leaderboard.csv`.

| Rank | Model | Params (B) | Within-person variance error (lower is better) | Criteria passed (of 6) | Trait→state direction agreement (%) | Indicator→state within-person direction agreement (%) |
|---|---|---|---|---|---|---|
| 1 | `google/gemma-2-27b-it` | 27 | 0.043 | 2 | 80 | 72.9 |
| 2 | `gpt-5.5` | — | 0.061 | 2 | 80 | 78.8 |
| 3 | `google/gemma-3-27b-it` | 27 | 0.065 | 3 | 90 | 71.2 |
| 4 | `qwen/qwen-2.5-coder-32b-instruct` | 32 | 0.090 | 2 | 80 | 78.8 |
| 5 | `anthropic/claude-sonnet-4` | — | 0.115 | 2 | 80 | 74.6 |
| 6 | `qwen/qwen3.5-27b` | 27 | 0.136 | 3 | — | 66.1 |
| 7 | `gpt-4.1` | — | 0.140 | 2 | 85 | 69.5 |
| 8 | `meta-llama/llama-3.1-8b-instruct` | 8 | 0.145 | 2 | 85 | 75.4 |
| 9 | `qwen/qwen-2.5-72b-instruct` | 72 | 0.146 | 2 | 90 | 77.1 |
| 10 | `google/gemma-3-12b-it` | 12 | 0.147 | 3 | 90 | 77.1 |
| 11 | `mistralai/mistral-small-3.2-24b-instruct` | 24 | 0.150 | 3 | 75 | 72.9 |
| 12 | `qwen/qwen3-14b` | 14 | 0.160 | 2 | 80 | 68.6 |
| 13 | `deepseek/deepseek-v3.2` | 671 | 0.161 | 3 | 85 | 71.2 |
| 14 | `qwen/qwen3-30b-a3b-thinking-2507` | 30 | 0.175 | 2 | 90 | 72.0 |
| 15 | `deepseek/deepseek-r1-0528` | 671 | 0.186 | 3 | 85 | 77.1 |
| 16 | `google/gemini-2.5-flash` | — | 0.196 | 2 | 85 | 76.3 |
| 17 | `qwen/qwen3-8b` | 8 | 0.206 | 2 | 90 | 71.2 |
| 18 | `deepseek/deepseek-r1-distill-llama-70b` | 70 | 0.210 | 2 | 90 | 74.6 |
| 19 | `meta-llama/llama-4-maverick` | 400 | 0.216 | 2 | 80 | 70.3 |
| 20 | `qwen/qwen3-30b-a3b-instruct-2507` | 30 | 0.232 | 2 | 85 | 74.6 |
| 21 | `qwen/qwen3-32b` | 32 | 0.236 | 2 | 85 | 77.1 |
| 22 | `meta-llama/llama-3.2-3b-instruct` | 3 | 0.240 | 2 | 80 | 78.0 |
| 23 | `meta-llama/llama-3-8b-instruct` | 8 | 0.242 | 2 | 85 | 76.3 |
| 24 | `google/gemini-2.5-pro` | — | 0.247 | 2 | 85 | 66.9 |
| 25 | `qwen/qwen3-235b-a22b-2507` | 235 | 0.250 | 1 | 90 | 68.6 |
| 26 | `openai/gpt-oss-20b` | 21 | 0.270 | 1 | 75 | 73.7 |
| 27 | `openai/gpt-oss-120b` | 117 | 0.287 | 1 | 85 | 76.3 |
| 28 | `meta-llama/llama-3.2-1b-instruct` | 1 | 0.309 | 3 | 75 | 71.2 |
| 29 | `ibm-granite/granite-4.1-8b` | 8 | 0.315 | 2 | 90 | 77.1 |
| 30 | `qwen/qwen-2.5-7b-instruct` | 7 | 0.325 | 1 | 85 | 76.3 |
| 31 | `meta-llama/llama-3.1-70b-instruct` | 70 | 0.354 | 2 | 80 | 74.6 |
| 32 | `microsoft/phi-4` | 14 | 0.362 | 1 | 85 | 72.0 |
| 33 | `meta-llama/llama-3.3-70b-instruct` | 70 | 0.363 | 1 | 80 | 73.7 |
| 34 | `google/gemma-3-4b-it` | 4 | 0.427 | 3 | 100 | 71.2 |

Proprietary models lack a public parameter count. Mixture-of-experts models are listed by total
parameters. `data/seeds/model_roster_scaling_live.csv` also lists active parameters.

## 🚀 Set Up

PsyDyBench runs on Python 3.9 or later on a CPU.

```bash
git clone https://github.com/sumervaid/PsyDyBench.git
cd PsyDyBench
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins the package versions that produced the released results.

## 💽 Usage

### Reproduce every result from the released corpora

The release contains the ratings of all 36 models, so every table, figure, and reported number can be
regenerated offline and for free. The run takes about 90 minutes on a laptop, almost all of it in the
1,000-resample bootstrap.

```bash
PYTHON=.venv/bin/python ./reproduce.sh
```

`reproduce.sh` runs the stages in dependency order and finishes with `scripts/verify_paper_values.py`,
which recomputes every macro in the manuscript (`results/paper/values.tex`) from the regenerated files
and fails on any disagreement. The resulting number-by-number table is
[docs/PROVENANCE.md](docs/PROVENANCE.md).

### Score a set of corpora

```bash
.venv/bin/python scripts/validation/run_validation.py results/generation/full_v2       # internal validity, pass/fail and variance error
.venv/bin/python scripts/validation/run_validation_v2.py results/generation/full_v2    # equivalence tests with bootstrap CIs
.venv/bin/python scripts/analysis/anchoring_leaderboard.py results/generation/full_v2  # external validity
```

### Generate new corpora

> [!WARNING]
> This stage calls paid APIs. It is needed only to add a model or to regenerate the corpora from
> scratch. Keys are read from the environment and are kept out of every file.

```bash
export OPENROUTER_API_KEY=...   # open-weight models
export OPENAI_API_KEY=...       # OpenAI models
./scripts/run_benchmark.sh --mock --limit 3   # dry run on a stub client, offline
./scripts/run_benchmark.sh                    # full roster, resumable
```

The orchestrator halts at the per-provider spend limits set in `scripts/run_benchmark.sh`. The spend
caps and estimates recorded during the original runs are in `docs/DECISIONS.md` (D-017, D-020).

### Add a model

1. Add a row to `data/seeds/model_roster_scaling_live.csv` (`model_id`, `provider`, `family`, `params_b`).
2. Run `./scripts/run_benchmark.sh`. The new corpus lands in `results/generation/full_v2/<MODEL>/`.
3. Run `./reproduce.sh`. The leaderboards, figures, and ranking table pick the model up automatically.

## 📦 Data

All data are synthetic and released under CC BY 4.0. See [docs/DATA_CARD.md](docs/DATA_CARD.md).

| Set | Location | Size |
|---|---|---|
| Persona seeds | `data/seeds/personas_full.csv` | 300 personas × 6 days |
| Stimulus set | `results/generation/full_v2/_shared_events/` | 300 portraits + 1,800 day narratives |
| Model ratings | `results/generation/full_v2/<MODEL>/states.csv` | 36 models × 1,800 persona-days |
| Raw responses | `results/generation/full_v2/<MODEL>/ratings.tar.gz` | each model's reasoning text per persona-day |
| Cultural variant | `results/generation/cultural/` | 6 nations × 3 models |
| Anchor library | `scripts/validation/meta_analysis_validation.py`, `data/anchors/` | 118 indicator-state pairs with source audit |

Publisher PDFs consulted during anchor verification are copyrighted and stay out of the release.

## 🗂️ Repository Structure

```
reproduce.sh                  regenerate all results offline
scripts/
  seeding/                    personas from published norms
  generation/                 stimulus set (run once) and first-person rating (per model)
  validation/                 internal-validity battery; the 118 meta-analytic anchors
  analysis/                   external validity, scaling, sensitivity
  magnitude/                  appendix probe of within-person compression
  scraping/                   locating and verifying the anchors
  figures/                    the three paper figures
  verify_paper_values.py      every manuscript number, recomputed
data/
  seeds/                      personas, norms, model roster, cross-state targets
  anchors/                    per-pair anchor audit
results/
  generation/                 corpora (see Data)
  leaderboard/                leaderboard CSVs and per-model JSON reports
  magnitude/                  appendix probe outputs
  figures/                    fig_scaling, fig_bycriterion, fig_tradeoff
  paper/                      values.tex (all reported numbers), model_ranking_table.tex
docs/                         see Documentation
```

## 📖 Documentation

| Document | Content |
|---|---|
| [METHOD.md](docs/METHOD.md) | Map from the paper's methods to the code |
| [INTERNAL_VALIDITY.md](docs/INTERNAL_VALIDITY.md) | The internal-validity battery, criterion by criterion |
| [DATA_CARD.md](docs/DATA_CARD.md) | What the data are, intended use, limitations |
| [PROVENANCE.md](docs/PROVENANCE.md) | Number-by-number audit of the manuscript |
| [ANCHOR_SOURCES.md](docs/ANCHOR_SOURCES.md) | How the 118 anchors were located and verified |
| [DECISIONS.md](docs/DECISIONS.md) | Dated ledger of every results-changing decision |
| [PREREGISTRATION.md](docs/PREREGISTRATION.md) | Design and analysis plan, fixed before any run |

The preregistration is anchored with OpenTimestamps (`docs/PREREGISTRATION.md.ots`):

```bash
ots verify docs/PREREGISTRATION.md.ots
```

## 💫 Contributing

New models, new anchors, and corrections are welcome.

- To add a model, follow *Add a model* above and open a pull request with the new corpus and the regenerated leaderboards.
- To correct an anchor, open an issue with the DOI, the table, and the value.
- `docs/PREREGISTRATION.md` is hash-anchored and stays as written.

## ✍️ Citation

```bibtex
@inproceedings{vaid2026psydybench,
  title     = {Benchmarking Psychological Dynamics in Generative Agents},
  author    = {Vaid, Sumer S. and Whillans, Ashley V.},
  booktitle = {AI Measurement Science Workshop at the Conference on Language Modeling (COLM)},
  year      = {2026},
  url       = {https://github.com/sumervaid/PsyDyBench}
}
```

## 📄 License

Code is released under the MIT License ([LICENSE](LICENSE)). Data are released under CC BY 4.0 ([data/LICENSE](data/LICENSE)).

## ✉️ Contact

Sumer S. Vaid, AI Institute, Harvard Business School. svaid@hbs.edu
