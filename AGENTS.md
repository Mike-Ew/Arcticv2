# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the core training code: `train.py` (CLI entrypoint), `dataset.py` (NetCDF/NPZ preprocessing + dataloaders), and `model.py` (U-Net via `segmentation_models_pytorch`).
- `utils/` contains helper scripts like `check_gpu.py` (CUDA validation) and `extract_rtt.py` (split archive extraction).
- `data/` stores raw/processed datasets (gitignored); `outputs/` and `checkpoints/` hold training artifacts (gitignored).
- `notebooks/` is for exploratory analysis; `igarss_2026/` contains paper assets; `configs/` is reserved for configuration files.

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt` installs dependencies.
- `python src/train.py --data_dir "data/ai4arctic_hugging face" --output_dir outputs` runs training; quote paths with spaces.
- `python src/model.py` runs a quick model smoke test.
- `python utils/check_gpu.py` verifies CUDA setup and benchmarks.
- `python utils/extract_rtt.py` extracts split `.tar.gz*` archives into `data/` (edit `base_dir` first).

## Coding Style & Naming Conventions
- Python with 4-space indentation; prefer PEP 8 naming (`snake_case` for functions/vars, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants).
- Keep module boundaries clean: dataset logic in `src/dataset.py`, models in `src/model.py`, and training orchestration in `src/train.py`.
- No formatter or linter is configured; keep lines readable (~100 chars) and use docstrings for non-obvious behavior.

## Testing Guidelines
- No formal unit test suite is present. Use smoke checks:
  - `python src/model.py` for forward-pass validation.
  - `notebooks/02_test_dataset.ipynb` for dataset sanity checks.
- If you add tests, use `tests/test_*.py` naming and document how to run them.

## Commit & Pull Request Guidelines
- Commit messages are short and imperative; an optional type prefix like `feat:` appears in history.
  - Example: `feat: add class-weighted loss` or `clean up training defaults`.
- PRs should include a concise summary, data or training changes, and any metrics/plots. Do not commit `data/` or `outputs/` artifacts (see `.gitignore`).

## Configuration & Data Notes
- Training defaults expect `data/ai4arctic_hugging face` with preprocessed `.npz` patches; training will preprocess from NetCDF if missing.
- Outputs, checkpoints, and logs should stay in `outputs/` or `checkpoints/` and remain untracked.
