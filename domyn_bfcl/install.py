# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""The one place where domyn_bfcl touches upstream ``bfcl_eval``.

Registrations (upstream's module-level dicts, extended in place):
  - ``MODEL_CONFIG_MAPPING``: the models in ``domyn_bfcl.models``.
  - ``TEST_COLLECTION_MAPPING``: the collections in ``domyn_bfcl.categories``.
  - ``CLASS_FILE_PATH_MAPPING["WebSearchAPI"]``: Brave Search (``domyn_bfcl.web_search``).

The run's ``--generation-config`` is validated into ``domyn_bfcl.config.GENERATION``,
which Domyn handlers read when built.

Patches (module attributes upstream looks up at call time):
  1. ``_llm_response_generation.collect_test_cases``: regenerates failed entries on
     resume and applies ``max_samples`` (``domyn_bfcl.generation``).
  2. ``base_oss_handler.requests``: the server-readiness check in the final
     ``OSSHandler.spin_up_local_server`` sends ``$REMOTE_OPENAI_API_KEY`` (upstream sends
     none, and spins forever on an endpoint that requires it).
  3. ``utils.make_json_serializable``: strings with lone UTF-16 surrogates (seen in model
     output) are written with ``?`` in their place; upstream's
     ``write_list_of_dicts_to_file`` raises ``UnicodeEncodeError`` on them, aborting the
     whole generate/evaluate run.
  4. ``sentence_transformers.SentenceTransformer``, only while upstream's
     ``memory_vector`` module is imported and only when ``$BFCL_ENCODER_MODEL`` is set:
     its module-level encoder, which upstream loads by name ("all-MiniLM-L6-v2", from the
     HF hub or cache), is loaded from that path instead -- compute nodes are offline.

``tests/test_install.py`` pins the upstream names and signatures relied on here.
"""

import importlib.abc
import importlib.util
import os
import sys

import bfcl_eval._llm_response_generation as generation_module
import bfcl_eval.model_handler.local_inference.base_oss_handler as base_oss_handler
import bfcl_eval.utils as bfcl_utils
import requests
from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING
from bfcl_eval.constants.executable_backend_config import CLASS_FILE_PATH_MAPPING
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from tqdm import tqdm

from domyn_bfcl import categories, config, generation, models

upstream_collect_test_cases = generation_module.collect_test_cases
upstream_make_json_serializable = bfcl_utils.make_json_serializable


def collect_test_cases(args, model_name, all_test_categories, all_test_entries_involved):
    """Patch 1: upstream's, after purging failed results and before applying ``max_samples``."""
    generation.purge_failed_results(args, model_name, all_test_categories)
    test_cases = upstream_collect_test_cases(args, model_name, all_test_categories, all_test_entries_involved)
    max_samples = config.GENERATION.max_samples
    if max_samples > 0 and test_cases:
        test_cases = generation.subsample_test_cases(test_cases, max_samples)
        tqdm.write(
            f"⚠️  Warning: max_samples is set to {max_samples}. Only the first "
            f"{max_samples} test cases per test category will be generated for "
            f"{model_name} ({len(test_cases)} in total)."
        )
    return test_cases


class AuthorizedRequests:
    """``requests`` as seen by ``base_oss_handler``: its GETs carry the endpoint's API key."""

    def __getattr__(self, name):
        """Everything but ``get`` is ``requests``'."""
        return getattr(requests, name)

    def get(self, url, **kwargs):
        """``requests.get`` with a bearer token from ``$REMOTE_OPENAI_API_KEY``."""
        headers = {
            "Authorization": f"Bearer {os.getenv('REMOTE_OPENAI_API_KEY', 'EMPTY')}",
            **(kwargs.pop("headers", None) or {}),
        }
        return requests.get(url, headers=headers, **kwargs)


def make_json_serializable(value):
    """Patch 3: upstream's, with lone surrogates in strings replaced by ``?``."""
    # Upstream recurses through the module-global name, so every leaf passes through here.
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    return upstream_make_json_serializable(value)


MEMORY_VECTOR_MODULE = "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_vector"
UPSTREAM_ENCODER = "all-MiniLM-L6-v2"


class MemoryEncoderFinder(importlib.abc.MetaPathFinder):
    """Import hook for patch 4; sentence_transformers is imported only if memory_vector is."""

    def find_spec(self, fullname, path, target=None):
        """Upstream's spec for memory_vector, executed with the configured encoder."""
        if fullname != MEMORY_VECTOR_MODULE:
            return None
        sys.meta_path.remove(self)  # one-shot, and lets the real finders resolve it below
        spec = importlib.util.find_spec(fullname)
        assert spec is not None
        assert spec.loader is not None
        exec_module = spec.loader.exec_module

        def exec_with_configured_encoder(module):
            import sentence_transformers

            upstream = sentence_transformers.SentenceTransformer

            def SentenceTransformer(model_name_or_path=None, *args, **kwargs):
                if model_name_or_path == UPSTREAM_ENCODER:
                    model_name_or_path = os.environ["BFCL_ENCODER_MODEL"]
                return upstream(model_name_or_path, *args, **kwargs)

            sentence_transformers.SentenceTransformer = SentenceTransformer
            try:
                exec_module(module)
            finally:
                sentence_transformers.SentenceTransformer = upstream

        spec.loader.exec_module = exec_with_configured_encoder
        return spec


def _register(mapping, additions, what):
    clashes = sorted(set(additions) & set(mapping))
    if clashes:
        msg = (
            f"domyn_bfcl: upstream bfcl_eval now defines {what} {clashes}; "
            "drop or rename ours (domyn_bfcl) after checking they match"
        )
        raise RuntimeError(msg)
    mapping.update(additions)


_installed = False


def install(generation_config: dict | None = None) -> None:
    """Register Domyn models, categories and Brave search; patch upstream (idempotent)."""
    global _installed
    config.configure(generation_config or {})
    if _installed:
        return
    _register(MODEL_CONFIG_MAPPING, models.MODEL_CONFIGS, "models")
    _register(TEST_COLLECTION_MAPPING, categories.EXTRA_TEST_COLLECTIONS, "test collections")
    CLASS_FILE_PATH_MAPPING["WebSearchAPI"] = "domyn_bfcl.web_search"
    generation_module.collect_test_cases = collect_test_cases
    base_oss_handler.requests = AuthorizedRequests()
    bfcl_utils.make_json_serializable = make_json_serializable
    if os.getenv("BFCL_ENCODER_MODEL") and MEMORY_VECTOR_MODULE not in sys.modules:
        sys.meta_path.insert(0, MemoryEncoderFinder())
    _installed = True
