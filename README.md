# llm-social-simulation

A lightweight multi-agent social simulation toolkit.

This repository currently includes:
- Model client abstractions and schemas (`llm_social_simulation/models`)
- A simple simulation engine (`llm_social_simulation/simulation/engine.py`)
- Game world implementations and tests (`llm_social_simulation/simulation`)

---

## Development Setup (uv)

We use **uv** + lock-based dependency management for reproducible environments.
Please **do not** use `pip install -r ...` or manual dependency installs.

### 1) Install `uv`

#### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows (PowerShell)

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

### 2) Create the environment

From the repository root:

```bash
rm -rf .venv
uv python install 3.12
uv venv --python 3.12
uv sync --group dev
```

### 3) Install pre-commit (recommended)

```bash
uv run pre-commit install
```

### 4) Run tests

```bash
uv run pytest
```

### 5) Lint and format

```bash
uv run ruff check .
uv run ruff format .
```

---

## Troubleshooting

If `uv` is not found after installation on macOS/Linux, add it to your `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Then restart your terminal and retry.

If setup still fails, share:
- the full error output
- `uv --version`

---

## Project Structure

```text
llm_social_simulation/
  models/
  simulation/
pyproject.toml
Roadmap.md
```

---

## Notes

- Python requirement is defined in `pyproject.toml`.
- Prefer running all tooling through `uv run ...` to keep execution inside the managed environment.
