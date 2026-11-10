"""domyn_bfcl.install: the upstream names it relies on, its registrations and its patches."""

import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import bfcl_eval.__main__ as bfcl_cli
import bfcl_eval._llm_response_generation as generation_module
import bfcl_eval.model_handler.local_inference.base_oss_handler as base_oss_handler
import bfcl_eval.utils as bfcl_utils
import pytest
from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING
from bfcl_eval.constants.executable_backend_config import CLASS_FILE_PATH_MAPPING
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search import (
    WebSearchAPI as UpstreamWebSearchAPI,
)
from pydantic import ValidationError

from domyn_bfcl import config
from domyn_bfcl import install as install_module
from domyn_bfcl.__main__ import pass_temperature, pop_generation_config, use_api_key
from domyn_bfcl.categories import EXTRA_TEST_COLLECTIONS
from domyn_bfcl.install import install
from domyn_bfcl.models import MODEL_CONFIGS

# -- Upstream contract: what install() patches must still exist and be used as assumed.
# A failure here after bumping bfcl-eval means re-checking domyn_bfcl.install by hand.


def test_patched_functions_keep_their_signatures():
    assert list(inspect.signature(install_module.upstream_collect_test_cases).parameters) == [
        "args",
        "model_name",
        "all_test_categories",
        "all_test_entries_involved",
    ]


def test_patched_names_are_looked_up_at_call_time():
    # main() resolves collect_test_cases through the module global install() replaces.
    assert "collect_test_cases(" in inspect.getsource(generation_module.main)
    spin_up = inspect.getsource(base_oss_handler.OSSHandler.spin_up_local_server)
    assert 'requests.get(f"{self.base_url}/models")' in spin_up
    writer = inspect.getsource(bfcl_utils.write_list_of_dicts_to_file)
    assert "make_json_serializable(entry)" in writer


def test_memory_vector_loads_its_encoder_by_name_at_import():
    origin = importlib.util.find_spec(install_module.MEMORY_VECTOR_MODULE).origin
    source = Path(origin).read_text()
    assert "from sentence_transformers import SentenceTransformer" in source
    assert f'ENCODER = SentenceTransformer("{install_module.UPSTREAM_ENCODER}"' in source


# -- install()


@pytest.fixture
def installed():
    install({})
    yield
    install({})


def test_registers_models_categories_and_web_search(installed):
    for name, model_config in MODEL_CONFIGS.items():
        assert MODEL_CONFIG_MAPPING[name] is model_config
    for name, categories in EXTRA_TEST_COLLECTIONS.items():
        assert TEST_COLLECTION_MAPPING[name] == categories
    module = importlib.import_module(CLASS_FILE_PATH_MAPPING["WebSearchAPI"])
    assert issubclass(module.WebSearchAPI, UpstreamWebSearchAPI)


def test_registration_refuses_to_shadow_upstream():
    with pytest.raises(RuntimeError, match="now defines models"):
        install_module._register({"a": 1}, {"a": 2}, "models")


def test_handlers_built_by_upstream_get_the_generation_config(installed):
    install({"temperature": 0.3, "top_p": 0.9})
    handler = generation_module.build_handler("Qwen/Qwen3.5-4B-FC", 0.001)
    assert handler.generation_config == config.GenerationConfig(temperature=0.3, top_p=0.9)


@pytest.mark.parametrize("generation_config", [{"enable_thinking": False}, {"token_safety_margin": "many"}])
def test_install_rejects_an_invalid_generation_config(installed, generation_config):
    with pytest.raises(ValidationError):
        install(generation_config)


def _result_file(result_dir, model_name, category):
    path = (
        result_dir
        / model_name.replace("/", "_")
        / bfcl_utils.get_directory_structure_by_category(category)
        / bfcl_utils.get_file_name_by_category(category, is_result_file=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_collect_test_cases_regenerates_failed_entries_and_subsamples(installed, tmp_path, monkeypatch):
    install({"max_samples": 1})
    result_file = _result_file(tmp_path, "mock-FC", "simple_python")
    entries = [
        {"id": "simple_python_0", "result": "[f(a=1)]"},
        {"id": "simple_python_1", "result": "Error during inference: timeout"},
    ]
    result_file.write_text("".join(json.dumps(e) + "\n" for e in entries))
    seen = {}

    def fake_upstream(args, model_name, categories, involved):
        seen["ids"] = [e["id"] for e in bfcl_utils.load_file(result_file)]
        return [{"id": "simple_python_1"}, {"id": "simple_python_2"}]

    monkeypatch.setattr(install_module, "upstream_collect_test_cases", fake_upstream)
    args = SimpleNamespace(result_dir=tmp_path, allow_overwrite=False)
    to_generate = generation_module.collect_test_cases(args, "mock-FC", ["simple_python"], [])

    assert seen["ids"] == ["simple_python_0"]
    assert to_generate == [{"id": "simple_python_1"}]


def test_readiness_check_sends_the_api_key(installed, monkeypatch):
    calls = []
    monkeypatch.setenv("REMOTE_OPENAI_API_KEY", "secret")
    monkeypatch.setattr(install_module.requests, "get", lambda url, **kw: calls.append((url, kw)))
    base_oss_handler.requests.get("http://x/v1/models")
    assert calls == [("http://x/v1/models", {"headers": {"Authorization": "Bearer secret"}})]
    assert base_oss_handler.requests.exceptions.ConnectionError is install_module.requests.exceptions.ConnectionError


def test_lone_surrogates_are_written_replaced(installed, tmp_path):
    path = tmp_path / "out.json"
    bfcl_utils.write_list_of_dicts_to_file(path, [{"id": "x", "result": ["a\ud83db", 1]}])
    assert json.loads(path.read_text()) == {"id": "x", "result": ["a?b", 1]}


# -- CLI


@pytest.mark.parametrize(
    "argv",
    [
        ["generate", "--generation-config", '{"top_k": 20}', "--model", "m"],
        ["generate", '--generation-config={"top_k": 20}', "--model", "m"],
    ],
)
def test_pop_generation_config(argv):
    assert pop_generation_config(argv) == ({"top_k": 20}, ["generate", "--model", "m"])


@pytest.mark.parametrize(
    ("args", "temperature", "expected"),
    [
        (["generate", "-m", "m"], 0.6, ["generate", "-m", "m", "--temperature", "0.6"]),
        (["generate", "--temperature", "0.6"], 0.6, ["generate", "--temperature", "0.6"]),
        (["generate", "--temperature=0.6"], 0.6, ["generate", "--temperature=0.6"]),
        (["generate", "--model", "m"], None, ["generate", "--model", "m"]),
        (["evaluate", "--model", "m"], 0.6, ["evaluate", "--model", "m"]),
        ([], 0.6, []),
    ],
)
def test_generate_gets_the_config_temperature(args, temperature, expected):
    assert pass_temperature(args, temperature) == expected


@pytest.mark.parametrize("args", [["generate", "--temperature", "0.2"], ["generate", "--temperature=0.2"]])
def test_a_conflicting_temperature_fails(args):
    with pytest.raises(SystemExit, match="conflicts"):
        pass_temperature(args, 0.6)


def test_only_generate_takes_a_temperature():
    # pass_temperature relies on these: bfcl has no top-level options, and only
    # generate has --temperature.
    assert bfcl_cli.cli.registered_callback is None
    assert "temperature" in inspect.signature(bfcl_cli.generate).parameters
    assert "temperature" not in inspect.signature(bfcl_cli.evaluate).parameters


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"API_KEY": "k"}, "k"),
        ({"API_KEY": "k", "REMOTE_OPENAI_API_KEY": "remote"}, "remote"),
        ({}, None),
    ],
)
def test_the_endpoint_key_falls_back_to_api_key(monkeypatch, env, expected):
    monkeypatch.setattr(os, "environ", dict(env))
    use_api_key()
    assert os.environ.get("REMOTE_OPENAI_API_KEY") == expected


_ENCODER_PROBE = """
import sentence_transformers
from domyn_bfcl.install import MEMORY_VECTOR_MODULE, install
install({})
import importlib
memory_vector = importlib.import_module(MEMORY_VECTOR_MODULE)
print(memory_vector.ENCODER.name)
print(sentence_transformers.SentenceTransformer.__module__)
"""

_STUB_SENTENCE_TRANSFORMERS = """
class SentenceTransformer:
    def __init__(self, name, device=None):
        self.name = name
    def get_sentence_embedding_dimension(self):
        return 384
"""


@pytest.mark.parametrize("configured", [True, False])
def test_memory_encoder_comes_from_bfcl_encoder_model(tmp_path, configured):
    # A fresh interpreter: memory_vector loads its encoder once, at import.
    (tmp_path / "sentence_transformers").mkdir()
    (tmp_path / "sentence_transformers" / "__init__.py").write_text(_STUB_SENTENCE_TRANSFORMERS)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), *sys.path])}
    env.pop("BFCL_ENCODER_MODEL", None)
    if configured:
        env["BFCL_ENCODER_MODEL"] = "/models/all-MiniLM-L6-v2"
    out = subprocess.run(
        [sys.executable, "-c", _ENCODER_PROBE], env=env, capture_output=True, text=True, check=True
    ).stdout.split()
    assert out[0] == ("/models/all-MiniLM-L6-v2" if configured else "all-MiniLM-L6-v2")
    assert out[1] == "sentence_transformers"  # restored after memory_vector's import
