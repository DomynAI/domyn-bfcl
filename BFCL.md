# BFCL in domyn-eval

BFCL runs upstream's code unmodified: [`bfcl-eval`](https://pypi.org/project/bfcl-eval/)
**2026.3.23** from PyPI (= gorilla `6ea5797`, upstream's latest). Everything Domyn adds or
changes lives in one wrapper package, [domyn-bfcl](https://github.com/DomynAI/domyn-bfcl)
(submodule `suites/domyn-bfcl`, Python package `domyn_bfcl`), run through its `domyn-bfcl` CLI: upstream's `bfcl` plus a
`--generation-config '<json>'` option that the driver fills from `[generation]`
(see [Generation config](#generation-config)).

> **Scores are not leaderboard-comparable.** Sampling, completion budget, web search
> backend and failed-entry handling all differ from the official setup (table below).
> Compare BFCL numbers only with other domyn-eval runs.

The suite owns its env (`suites/domyn-bfcl/.venv`, from its `uv.lock`), like
domyn-inspect. `BFCL_PROJECT_ROOT` is set per run by the driver to the run's results
folder, where BFCL keeps its scratch and lock files.

## Divergences from upstream

Every row is a deliberate choice; nothing else differs.

| What | Upstream | domyn-bfcl | Where | Changes scores? |
| --- | --- | --- | --- | --- |
| Sampling | `temperature` only (`--temperature`, default 0.001) | `temperature`, `top_p`, `presence_penalty` + vLLM extras `top_k`, `min_p`, `repetition_penalty`, `frequency_penalty`, `chat_template_kwargs` from `[generation]`; unset values are not sent (server defaults). Upstream's handlers get `temperature` as `generate --temperature` | `config.py`, `oss.py`, `__main__.py` | Yes |
| Completion budget (OSS handlers) | `min(4096, window − prompt)` | largest completion that fits the served window (`fit_max_tokens`, honours `max_tokens` / `max_model_len` / `token_safety_margin`) | `oss.py` | Yes (no truncation at 4096) |
| Failed entries on resume | kept; score as wrong (or as correct abstention in irrelevance) | regenerated (`"Error during inference..."` results are dropped first) | `generation.py`, `install.py` (collect_test_cases) | Yes, on resumed runs |
| `max_samples` | — | first N per category (+ memory prerequisites), scored with `--partial-eval` | `generation.py` | Smoke runs only |
| Web search | SerpAPI (DuckDuckGo engine) | Brave Search API (`BRAVE_API_KEY`), shared rate limit, 429 backoff capped at 900 s, region → Brave locale | `web_search.py` | Yes (different engine) |
| Request timeout and retries | 72000 s, 2 SDK retries | `request_timeout` (default 3600 s); no retries, except `DomynHandler` / `ChatFCHandler` (5 attempts on 429, 5xx and connection errors; never on a timeout) | `oss.py`, `handlers/domyn.py` | Only for very slow or failing requests |
| Served model id | request names the HF id / local path | `$REMOTE_OPENAI_MODEL` when set (the driver sets it to the served id); unset, upstream's. Needed when the endpoint serves the model under another name, e.g. with `--local-model-path` | `oss.py` | No |
| Readiness check | no API key (hangs on an auth-protected endpoint) | sends `$REMOTE_OPENAI_API_KEY` | `install.py` | No |
| Endpoint key | `$REMOTE_OPENAI_API_KEY` | `$REMOTE_OPENAI_API_KEY`, else `$API_KEY` | `__main__.py` | No |
| Memory encoder | `all-MiniLM-L6-v2` by name (HF hub/cache) | `$BFCL_ENCODER_MODEL` when set (offline compute nodes); same model | `install.py` (import hook on `memory_vector`) | No (without it, offline memory_vector scores 0) |
| Lone UTF-16 surrogates in output | `UnicodeEncodeError` aborts the run | written as `?` | `install.py` | No (prevents a crash) |
| Extra test collections | — | `all_scoring_no_mem`, `all_scoring_no_websearch`, `single_multi`, `core` | `categories.py` | No |
| Extra models | — | see below | `models.py`, `handlers/` | n/a |

`domyn_bfcl/install.py` is the only module that touches upstream: three registrations
(`MODEL_CONFIG_MAPPING`, `TEST_COLLECTION_MAPPING`, `CLASS_FILE_PATH_MAPPING["WebSearchAPI"]`)
and four patches (`_llm_response_generation.collect_test_cases`, `base_oss_handler.requests`,
`utils.make_json_serializable`, and `sentence_transformers.SentenceTransformer` only
while `memory_vector` is imported, when `BFCL_ENCODER_MODEL` is set). Upstream handlers are reused through composed classes,
e.g. `DomynQwenFCHandler(DomynOSSMixin, QwenFCHandler)`.

### Generation config

`--generation-config` is validated into `domyn_bfcl.config.GenerationConfig` when the CLI
starts; an unknown key or a wrong type fails there, before any request. Upstream builds
the handlers unchanged, and each Domyn handler reads the validated config
(`config.GENERATION`) when it is built. Upstream's own handlers read only `temperature`,
which `generate` gets as `--temperature` (an explicit `--temperature` must match it).

| Key | Default | Use |
| --- | --- | --- |
| `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty`, `frequency_penalty` | unset | sampling; unset is not sent |
| `chat_template_kwargs` | unset | sent to the chat template as given, with `enable_thinking` set from the thinking mode (`DomynHandler` also defaults `preserve_thinking` to on) |
| `reasoning` | `true` | the thinking mode when `chat_template_kwargs` has no `enable_thinking` |
| `max_model_len`, `max_tokens`, `token_safety_margin` | unset, unset, 256 | completion budget |
| `request_timeout` | 3600 | seconds per request |
| `max_samples` | -1 (all) | first N per category |

Every handler with a thinking toggle (`DomynHandler`, `ChatFCHandler`, `NemotronOSSHandler`)
uses one rule: `chat_template_kwargs["enable_thinking"]` if set, else `reasoning`.

### Added models

| Registry name | Handler |
| --- | --- |
| `Domyn/Domyn-Edge-FC`, `Domyn/Domyn-Small-FC` | `DomynHandler` (chat API + tools; the server applies each model's template and tool parser) |
| `Qwen/Qwen3-4B-FC`, `Qwen/Qwen3.5-4B-FC`, `Qwen/Qwen3.5-4B-Base-FC`, `Qwen/Qwen3.5-9B-FC`, `VibeThinker-3B-FC` | upstream `QwenFCHandler` + mixin |
| `Nanbeige4.1-3B-FC` | upstream `NanbeigeFCHandler` + mixin |
| `mistralai/Ministral-3-8B-Reasoning-2512`, `Ministral-3-3B-Reasoning-2512-FC` | `MinistralFCHandler` |
| `allenai/Olmo-3-7B-Think-FC` | `OlmoFCHandler` |
| `nvidia/Llama-3.1-Nemotron-Nano-8B-v1-FC` | `NemotronOSSHandler` |
| `HuggingFaceTB/SmolLM3-3B-FC`, `allenai/Olmo-3-7B-Instruct-FC`, `mistralai/Ministral-3-8B-Instruct-2512-FC`, `nvidia/NVIDIA-Nemotron-Nano-9B-v2-FC`, `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16-FC`, `tencent/Youtu-LLM-2B-FC`, `tencent/Youtu-LLM-2B-Base-FC` | `ChatFCHandler` (generic chat API + tools; needs a `--tool-call-parser` in the swarm) |
| `mock-FC` | `MockHandler` (tests/mock_model) |

Prompts: models upstream supports use upstream's handler, which renders the model's
template itself (upstream's design for local models). Our own handlers cover only
models upstream lacks: `DomynHandler` and `ChatFCHandler` use the
chat API, so the server applies the template; Ministral 3, Olmo 3 and Nemotron Nano keep a
client-side template (`*.jinja`), like upstream's local handlers. Upstream's
`MistralFCHandler` (Ministral-8B-Instruct-2410, no `[THINK]` handling) and
`NemotronHandler` (Nemotron Ultra on NVIDIA's API, no thinking toggle) do not fit
Ministral 3 and Nemotron Nano.

`ChatFCHandler` is the fallback for models with no fitting handler: `DomynHandler`'s
requests, but past tool calls go back as native `tool_calls` (with `tool_call_id`), so
the server renders them in each model's own format. Its swarms run vLLM's parser for
the model: `hermes` (SmolLM3), `olmo3`, `mistral`, `qwen3_coder` (Nemotron 3 Nano), `hermes` (Youtu-LLM),
NVIDIA's `nemotron_json` plugin (Nemotron Nano 9B v2). `configs/HuggingFaceTB/smollm3_base.toml`
has no BFCL handler: its swarm's chat template does not render tools.

## Working on it

- Tests (in the BFCL venv, against the real upstream):
  `suites/domyn-bfcl/.venv/bin/python -m pytest suites/domyn-bfcl/tests` (also run in CI).
  `tests/test_install.py` pins the upstream names and signatures `install.py` relies on.
- Bumping upstream: change `bfcl-eval==…` in domyn-bfcl's `pyproject.toml`,
  `uv lock` + `uv sync`, run the tests (the contract tests fail first if a patch target
  moved).
- A new divergence goes in its own module (or `install.py` if it must patch upstream),
  with a row in the table above.
