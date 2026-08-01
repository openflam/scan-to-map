# Repository Guidelines

## Project Structure & Module Organization

The repository is split into independently runnable components. `segment3d/` contains the Python scan-to-map pipeline, with implementation under `segment3d/src/` and pytest coverage under `segment3d/tests/`. `search-server/` is the Flask search API; semantic providers, reasoning tools, routing, and PostGIS helpers live in dedicated subpackages. `semantic-3d-search-demo/` is the React/TypeScript Vite client. Supporting workflows live in `data-processor/`, `QA-data-generator/`, and `benchmark-eval/`. Architecture notes belong in `docs/`. Generated `data/`, `outputs/`, `benchmark/`, and model checkpoints are intentionally ignored; do not commit them.

## Build, Test, and Development Commands

- `docker compose up --build --detach` builds and starts PostGIS and the GPU-enabled search server; use `docker compose logs -f` to inspect it and `docker compose down` to stop it.
- `cd search-server && pip install -r requirements.txt && python app.py` runs the API directly on port 5000.
- `cd semantic-3d-search-demo && npm ci && npm run dev` installs locked dependencies and starts Vite. Run `npm run build` for a production/type-check build and `npm run lint` for ESLint.
- `cd segment3d && ./run.sh <dataset_name>` runs the full mapping pipeline. For selective execution, use `python main.py --dataset <dataset_name> --skip-sam`.
- `python -m pytest segment3d/tests` runs the Python tests; ensure the local dataset configuration expected by `segment3d/config.py` is available.

## Coding Style & Naming Conventions

Use four spaces, `snake_case` functions/modules, `PascalCase` classes, type hints, and focused docstrings in Python. Keep pipeline stages callable from small CLI entry points. TypeScript uses two-space indentation, double quotes, semicolons, `PascalCase` React components, and `camelCase` hooks/helpers. Follow the checked-in ESLint configuration; keep imports explicit and shared types in `src/types/`.

## Testing Guidelines

Pytest files follow `test_*.py`, with shared setup in `conftest.py`. Add unit tests beside the existing `segment3d/tests` suite for configuration, geometry, and file parsing changes. Frontend changes currently have no test runner, so treat `npm run lint` and `npm run build` as required checks. Document any GPU- or dataset-dependent validation in the PR.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries such as `Add shave borders script` and `Update plots`. Keep each commit scoped to one concern. PRs should explain the behavior change, list commands run, link relevant issues, and call out required datasets, models, environment variables, or migrations. Include screenshots or recordings for UI/3D-viewer changes and never commit `.env` files or API keys.
