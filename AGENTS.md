# Repository Guidelines

## Project Structure & Modules
- Core model code lives in `model.py`, `vocab.py`, and `data_module.py`; dataset loading and filtering sit in `dataset.py` (caches under `data/`).
- Training entry point is `train.py` (CPU by default). Quick subset training lives in `demo.py`. Inference against a saved checkpoint is in `demo_sentences.py`.
- Helper diagnostics include `check_max_candidates.py` and `analyze_pinyin_hanzi_mapping.py`. Checkpoints land in `checkpoints/`; cached vocab/mapping files stay in `data/`.

## Setup, Build, and Run
- Target Python 3.10+ with minimal dependencies (PyTorch and `tqdm`; rest is stdlib). Keep the environment lightweight—this is a small project.
- Data is auto-downloaded on first use if `data/pinyin2hanzi.csv` is absent; otherwise the cached copy is reused.
- Training runs on CPU by default and writes checkpoints under `checkpoints/`. Subset training and quick diagnostics are available via existing scripts (see file headers).

## Coding Style & Naming
- Python throughout; 4-space indents, type hints where practical, and small, direct functions. Prefer straightforward tensor ops over abstractions.
- Use `snake_case` for functions/vars, `CamelCase` for classes. Keep modules slim and aligned with existing file roles.
- Avoid unnecessary dependencies or framework layers; favor explicit control flow and comments only when logic is non-obvious.

## Data & Checkpoints
- Cached vocab/mapping and dataset CSVs live in `data/`; delete them only if you want a fresh download (network required).
- Training writes `.pt` files under `checkpoints/`; include the path when sharing results so others can reproduce inference.

## Commit & PR Guidelines
- Git history uses short, imperative, lower-case summaries (e.g., `filter valid hanzi`); follow that pattern and keep commits scoped.
- PRs: brief description of what changed and why, commands/logs you ran, and any data/regression considerations. Link issues if relevant, but keep process light to avoid overengineering.
