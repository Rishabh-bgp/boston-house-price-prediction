# Contributing
1. `pip install -r requirements-dev.txt`; `python -m src.train && ruff check . && pytest -q` must pass.
2. Do not add `B` (or any race/ethnicity feature) to the model — a test enforces this.
3. Re-execute the notebook if model settings change, so the docs stay accurate. MIT licence applies.
