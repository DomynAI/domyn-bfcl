"""DomynOSSMixin: request construction for BFCL's local (OSS) handlers."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler

from domyn_bfcl import config, models
from domyn_bfcl.models import DomynQwenFCHandler
from domyn_bfcl.oss import DomynOSSMixin, fit_max_tokens

GENERATION_CONFIG = {
    "temperature": 0.6,
    "top_p": 0.95,
    "presence_penalty": None,
    "top_k": 20,
    "min_p": None,
    "repetition_penalty": 1.0,
    "chat_template_kwargs": {"enable_thinking": True},
    "max_samples": 2,
    "max_tokens": None,
    "max_model_len": 1000,
    "token_safety_margin": 256,
    "request_timeout": 123,
}


def _handler(generation_config=GENERATION_CONFIG):
    config.configure(generation_config)
    handler = DomynQwenFCHandler(
        model_name="Qwen/Qwen3.5-4B",
        temperature=0.001,
        registry_name="Qwen/Qwen3.5-4B-FC",
        is_fc_model=True,
    )
    handler.tokenizer = SimpleNamespace(
        tokenize=lambda text: text.split(), decode=lambda ids, **kw: f"<{len(ids)} ids>"
    )
    handler.max_context_length = 4096
    handler.model_path_or_id = "Qwen/Qwen3.5-4B"
    handler.client = MagicMock()
    return handler


def _query(handler, prompt="a b c"):
    handler._format_prompt = lambda messages, function: prompt
    inference_data = {"message": [], "function": []}
    handler._query_prompting(inference_data)
    return handler.client.completions.create.call_args_list, inference_data


def test_every_registered_oss_handler_uses_the_mixin():
    for name, model_config in models.MODEL_CONFIGS.items():
        if issubclass(model_config.model_handler, OSSHandler):
            assert issubclass(model_config.model_handler, DomynOSSMixin), name


def test_client_uses_request_timeout_without_retries():
    handler = _handler()
    handler.__init__("Qwen/Qwen3.5-4B", 0.6, "Qwen/Qwen3.5-4B-FC", True)
    assert handler.client.timeout == 123
    assert handler.client.max_retries == 0


def test_handler_without_a_run_config_uses_the_defaults():
    handler = DomynQwenFCHandler("Qwen/Qwen3.5-4B", 0, "Qwen/Qwen3.5-4B-FC", True)
    assert handler.generation_config == config.GenerationConfig()


def test_sampling_comes_from_generation_config():
    (call,), _ = _query(_handler())
    kwargs = call.kwargs
    assert kwargs["temperature"] == 0.6
    assert kwargs["top_p"] == 0.95
    assert "presence_penalty" not in kwargs  # None: left to the server
    assert kwargs["extra_body"] == {
        "top_k": 20,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert "timeout" not in kwargs  # the client's request_timeout applies


def test_max_tokens_is_the_budget_that_fits_the_window():
    (call,), _ = _query(_handler(), prompt="a b c")
    # 1000 window - 3 prompt tokens - the margin
    assert call.kwargs["max_tokens"] == 997 - GENERATION_CONFIG["token_safety_margin"]


@pytest.mark.parametrize(
    ("max_tokens", "prompt_tokens", "expected"),
    [
        (None, 100, 644),  # unset: the whole window the prompt leaves
        (0, 100, 644),
        (500, 100, 500),  # fits: sent as is
        (900, 100, 644),  # cut to the room left
        (500, 1000, 1),  # the prompt alone fills the window
    ],
)
def test_fit_max_tokens(max_tokens, prompt_tokens, expected):
    assert fit_max_tokens(max_tokens, prompt_tokens, 1000, margin=256) == expected


def test_token_id_prompts_are_counted_and_logged():
    (call,), inference_data = _query(_handler(), prompt=[1, 2, 3, 4])
    assert call.kwargs["prompt"] == [1, 2, 3, 4]
    assert inference_data["inference_input_log"] == {"formatted_prompt": "<4 ids>"}


def test_context_overflow_is_not_retried():
    """max_tokens is budgeted up front; a rejection propagates (upstream records it)."""
    handler = _handler()
    error = Exception(
        "This model's maximum context length is only 150 tokens. However, you "
        "requested 900 tokens (passed 100 input tokens and 800 output tokens)"
    )
    handler.client.completions.create.side_effect = [error, "ok"]
    with pytest.raises(Exception, match="maximum context length"):
        _query(handler)
    assert handler.client.completions.create.call_count == 1


def test_requests_name_the_served_model(monkeypatch):
    handler = _handler()
    monkeypatch.setattr(OSSHandler, "inference", lambda self, *args: (self.model_path_or_id, {}))
    monkeypatch.setenv("REMOTE_OPENAI_MODEL", "served-id")
    assert handler.inference({"id": "simple_python_0"}, False, False)[0] == "served-id"


def _finishing(*reasons):
    return [SimpleNamespace(choices=[SimpleNamespace(finish_reason=r)]) for r in reasons]


def _requests(count, barrier=None):
    """An upstream inference that sends ``count`` requests, waiting at ``barrier`` between them."""

    def inference(self, *args):
        for _ in range(count):
            self._query_prompting({"message": [], "function": []})
            if barrier:
                barrier.wait(timeout=5)
        return "result", {}

    return inference


def test_entries_record_the_finish_reason_of_every_request(monkeypatch):
    handler = _handler()
    handler._format_prompt = lambda messages, function: "a b c"
    handler.client.completions.create.side_effect = _finishing("stop", "length", "stop")
    monkeypatch.setattr(OSSHandler, "inference", _requests(2))
    assert handler.inference({}, False, False) == ("result", {"finish_reason": ["stop", "length"]})
    monkeypatch.setattr(OSSHandler, "inference", _requests(1))
    assert handler.inference({}, False, False)[1] == {"finish_reason": ["stop"]}


def test_entries_in_parallel_threads_keep_their_own_finish_reasons(monkeypatch):
    handler = _handler()
    handler._format_prompt = lambda messages, function: "a b c"
    handler.client.completions.create.side_effect = _finishing("stop", "length", "stop", "length")
    monkeypatch.setattr(OSSHandler, "inference", _requests(2, threading.Barrier(2)))
    with ThreadPoolExecutor(2) as pool:
        entries = list(pool.map(lambda _: handler.inference({}, False, False)[1], range(2)))
    reasons = [entry["finish_reason"] for entry in entries]
    assert sorted(len(r) for r in reasons) == [2, 2]
    assert sorted(r for entry_reasons in reasons for r in entry_reasons) == ["length", "length", "stop", "stop"]
