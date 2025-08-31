# Repository Guidelines

## Project Structure & Module Organization
- python/: Python package sources (core APIs under `python/triton/`), tests in `python/test/`, tutorials/examples under `python/tutorials/` and `python/examples/`.
- lib/, include/, cmake/: Compiler, dialect, and backend sources and CMake configuration.
- test/: MLIR/Lit tests (e.g., TritonGPU passes and pipelines).
- third_party/: Vendor code (AMD/NVIDIA/others); avoid editing unless vendoring updates.
- docs/: Sphinx documentation; built output under `docs/_build/`.

## Build, Test, and Development Commands
- make dev-install: Install Python deps and Triton in editable mode.
- make all: Incremental CMake/Ninja build in the active cmake build dir.
- make test: Run Lit, C++ unit, and Python tests.
- make test-python: Run Python unit, regression, interpreter, and proton tests.
- make test-lit / make test-cpp: Run MLIR/Lit tests or C++ unit tests only.
- make docs: Build Sphinx docs (use `make docs-requirements` first if needed).
Examples:
  - PYTHON=python3.12 make dev-install
  - NUM_PROCS=8 make test-python

## Coding Style & Naming Conventions
- Python: PEP8-ish, 120-char lines (`ruff`, `yapf` configured in `pyproject.toml`).
- C/C++/CUDA: `clang-format` enforced by `.clang-format`.
- Static checks: `mypy` for selected modules; run via pre-commit.
- Run pre-commit locally: `pre-commit install && pre-commit run -a`.
- Naming: Use clear module prefixes; tests mirror package paths (e.g., `python/test/unit/runtime/test_build.py`).

## Testing Guidelines
- Frameworks: `pytest` (Python), LLVM Lit (MLIR), C++ unit tests.
- Conventions: Place unit tests near feature area; name `test_*.py` and mirror package layout.
- Quick run: `python -m pytest -q python/test/unit`.
- Full suite: `make test` (GPU tests may require CUDA/ROCm). Use markers or targets to scope.

## Commit & Pull Request Guidelines
- Commit messages: Scope prefix in brackets + concise imperative summary, e.g., `[GLUON] Fix layout broadcast edge case`.
- Reference issues/PRs when applicable; keep body wrapped at ~72 chars.
- PRs must include: clear description, rationale, tests (or explanation), and docs updates when user-facing.
- CI: Ensure pre-commit passes and `make test` is green for affected areas.

## Security & Configuration Tips
- Do not commit secrets or local paths; pre-commit checks detect common leaks.
- GPU backends require matching drivers/toolchains; prefer `make dev-install-llvm` if building LLVM locally.
- Useful envs during tests: `TRITON_ALWAYS_COMPILE=1`, `TRITON_DISABLE_LINE_INFO=0`.

## Logging
- Please include verbatim logs of all our conversations as a markdown file in whatever project we're working on.
- Include all the commands you ran during our conversations to provenance tracking.
- If no specific project (e.g., code exploration), then please include it in a "codex-logs" folder at the repo root, with the timestamped filename.