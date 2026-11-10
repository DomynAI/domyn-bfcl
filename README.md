<div align="center">

<!-- TODO: add the logos to static/ and drop the title below.
<img src="static/domyn-bfcl-logo-primary.svg#gh-light-mode-only" alt="Domyn BFCL" height="100">
<img src="static/domyn-bfcl-logo-white.svg#gh-dark-mode-only" alt="Domyn BFCL" height="100">
-->

# domyn-bfcl

**The Berkeley Function Calling Leaderboard, with Domyn's sampling parameters, token budget and models.**

domyn-bfcl wraps the unmodified PyPI package [`bfcl-eval`](https://pypi.org/project/bfcl-eval/):
its `domyn-bfcl` CLI is upstream's `bfcl` plus `--generation-config '<json>'`. domyn-eval runs it
as its `bfcl` suite. Every divergence from upstream is documented in [BFCL.md](BFCL.md).

[![CI](https://github.com/DomynAI/domyn-bfcl/actions/workflows/ci.yaml/badge.svg)](https://github.com/DomynAI/domyn-bfcl/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-brightgreen)](pyproject.toml)
[![bfcl-eval](https://img.shields.io/badge/bfcl--eval-2026.3.23-orange)](https://pypi.org/project/bfcl-eval/2026.3.23/)
[![License - Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://github.com/microsoft/pyright)

<a href="#quick-start">Quick start</a> •
<a href="BFCL.md">Divergences from upstream</a> •
<a href="#layout">Layout</a> •
<a href="#development">Development</a>

</div>

## Quick start

```sh
uv sync
export REMOTE_OPENAI_BASE_URL=http://localhost:8000/v1 API_KEY=...
uv run domyn-bfcl generate --model Domyn/Domyn-Edge-FC --test-category single_turn \
    --skip-server-setup --generation-config '{"temperature": 0.6, "top_p": 0.95}'
uv run domyn-bfcl evaluate --model Domyn/Domyn-Edge-FC --test-category single_turn
```

Every `bfcl` command and option works. `--generation-config` takes the run's sampling
parameters, thinking mode (`reasoning`, `chat_template_kwargs`), completion budget
(`max_tokens`, `max_model_len`, `token_safety_margin`) and `request_timeout`; unset sampling
parameters are left to the server. The endpoint's key is `$REMOTE_OPENAI_API_KEY`, else `$API_KEY`.
Requests name the model by the HF id or local path that BFCL loads the tokenizer from; set
`$REMOTE_OPENAI_MODEL` when the endpoint serves it under another name.

> **Scores are not leaderboard-comparable.** Sampling, completion budget, web search and
> failed-entry handling differ from the official setup; see [BFCL.md](BFCL.md).

## Layout

- `domyn_bfcl/install.py`: the only module that touches upstream (registrations + patches).
- `domyn_bfcl/config.py`: the validated `--generation-config`.
- `domyn_bfcl/oss.py`: sampling and token budget for OSS handlers.
- `domyn_bfcl/generation.py`, `web_search.py`, `categories.py`, `models.py`, `handlers/`.

## Development

```sh
uv sync
uv run pytest tests
uv tool install pre-commit && pre-commit install   # ruff and pyright on every commit
```

Bumping upstream: change `bfcl-eval==…` and move `[tool.uv] exclude-newer` (it pins
the resolution date, so a plain `uv sync` reproduces the lock), `uv lock`, run the tests
(`tests/test_install.py` pins every upstream name the patches rely on).
