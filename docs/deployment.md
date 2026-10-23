# Deployment
* Docker / compose: API on :8000, nginx-served web app on :8080 (written, not built in the authoring environment — no Docker there).
* CI: `.github/workflows/ci.yml` (ruff + train + pytest on Python 3.12/3.13).
* GitHub Pages: `gh-pages` branch holds a copy of `web/`; copy and push to update.
