# Power Sampling

MIT 6.8610 (NLP) final project. Reproduces and extends Karan & Du, *"Reasoning with Sampling: Your Base Model is Smarter Than You Think"* ([arXiv 2510.14901](https://arxiv.org/abs/2510.14901)) — power-sampling MCMC over base language models for math reasoning.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                  # install deps + create .venv (CPU / MPS)
uv sync --extra gpu      # on Linux GPU pods (adds bitsandbytes for 4/8-bit)
```

## Running experiments (legacy entry points)

```bash
uv run python scripts/run_global_power_ablation.py     --max-problems 5
uv run python scripts/run_local_power_ablation.py      --max-problems 5
uv run python scripts/run_speculative_power_decoding.py --max-problems 5
```

Outputs land in `results/<model>/`.

A unified `scripts/run.py` (YAML-driven, per-owner config dirs, self-describing run folders) will replace these in a later step. Until then, the legacy scripts are the source of truth.

## Layout

- `power_sampling/` — core library (MCMC kernel, samplers, model loading, MATH parsing)
- `scripts/` — CLI entry points
- `notebooks/` — analysis only (run experiments via scripts, not notebooks)
- `docs/` — project proposal
- `results/` — gitignored; per-experiment output folders

## Reproducibility notes

- Python pinned via `requires-python = ">=3.11"`; deps locked in `uv.lock` (committed).
- HF model + dataset revisions are pinned in `power_sampling/` (see `MODEL_REGISTRY`).
- Each run writes its config + env metadata alongside its outputs.

## Team workflow

- Shared core under `power_sampling/` — change via PR.
- Per-person experiment configs under `configs/<owner>/` (added in a later step).
- Per-person results under `results/<owner>/<run-id>/` — gitignored; share via the run folder, not the repo.
- Cross-team comparison via `results/INDEX.md` (one row per run, committed).
