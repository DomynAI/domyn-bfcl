"""Session routing: one session ID per inference call, sent in the header and the body of each request."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import MagicMock

import httpx
import pytest
import tenacity.nap
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from openai import APIConnectionError

from domyn_bfcl import models
from domyn_bfcl.config import GenerationConfig
from domyn_bfcl.handlers.domyn import DomynHandler
from domyn_bfcl.oss import DomynOSSMixin


@pytest.fixture
def handler():
    handler = DomynHandler.__new__(DomynHandler)
    handler.generation_config = GenerationConfig(session_routing=True, temperature=0)
    handler.model_path_or_id = "m"
    handler.max_context_length = 100
    handler.tokenizer = MagicMock()
    handler.tokenizer.tokenize.return_value = [1]
    handler.tokenizer.apply_chat_template.return_value = [1]
    handler._format_prompt = lambda *_: "prompt"
    handler.client = MagicMock()
    return handler


@pytest.mark.parametrize("chat", [False, True])
def test_sessions_follow_inference_and_remain_isolated_across_threads(monkeypatch, chat, handler):
    query = DomynHandler._query_prompting if chat else DomynOSSMixin._query_prompting
    barrier = Barrier(2)
    captured = []

    def create(**kwargs):
        captured.append(kwargs)
        return MagicMock()

    handler.client.chat.completions.create.side_effect = create
    handler.client.completions.create.side_effect = create

    def inference(self, entry, include_input_log, exclude_state_log):
        barrier.wait()
        data = self._pre_query_processing_prompting(entry)
        query(self, data)
        query(self, data)
        return entry["id"], {}

    monkeypatch.setattr(OSSHandler, "inference", inference)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(
            pool.map(lambda n: handler.inference({"id": f"multi_turn_base_{n}", "function": []}, False, False), [1, 2])
        )
    ids = [call["extra_headers"]["X-Session-ID"] for call in captured]
    assert len(set(ids)) == 2
    assert all(ids.count(sid) == 2 for sid in set(ids))
    assert all(
        call["extra_body"]["session_params"]["session_id"] == call["extra_headers"]["X-Session-ID"] for call in captured
    )
    assert all("X-Dynamo-Session-Final" not in call["extra_headers"] for call in captured)
    query(handler, {"message": [], "function": []})
    assert "extra_headers" not in captured[-1]


def test_single_turn_session_is_final_and_does_not_leak_after_error(monkeypatch):
    handler = DomynHandler.__new__(DomynHandler)
    handler.generation_config = GenerationConfig(session_routing=True)
    captured = []

    def inference(self, entry, *args):
        data = self._pre_query_processing_prompting(entry)
        captured.append(self._session_request_options(data, {}))
        msg = "failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(OSSHandler, "inference", inference)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="failed"):
            handler.inference({"id": "simple_python_0", "function": []}, False, False)
    assert captured[0]["extra_headers"]["X-Dynamo-Session-Final"] == "true"
    assert captured[0]["extra_headers"]["X-Session-ID"] != captured[1]["extra_headers"]["X-Session-ID"]
    assert handler._session_request_options({}, {"top_k": 3}) == {"extra_body": {"top_k": 3}}


def test_session_is_reused_after_a_connection_failure(monkeypatch, handler):
    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda _seconds: None)
    handler.client.chat.completions.create.side_effect = [
        APIConnectionError(request=httpx.Request("POST", "http://swarm/v1/chat/completions")),
        MagicMock(),
    ]

    def inference(self, entry, include_input_log, exclude_state_log):
        return self._query_prompting(self._pre_query_processing_prompting(entry)), {}

    monkeypatch.setattr(OSSHandler, "inference", inference)
    handler.inference({"id": "multi_turn_base_1", "function": []}, False, False)

    calls = handler.client.chat.completions.create.call_args_list
    assert len(calls) == 2
    first, retry = (call.kwargs for call in calls)
    session_id = first["extra_headers"]["X-Session-ID"]
    assert retry["extra_headers"]["X-Session-ID"] == session_id
    assert all(call.kwargs["extra_body"]["session_params"]["session_id"] == session_id for call in calls)


@pytest.mark.parametrize(
    "handler_cls",
    sorted(
        {
            entry.model_handler
            for entry in models.MODEL_CONFIGS.values()
            if issubclass(entry.model_handler, DomynOSSMixin)
        },
        key=lambda cls: cls.__name__,
    ),
)
@pytest.mark.parametrize("enabled", [False, True])
def test_registered_handlers_prepare_their_own_session_data(handler_cls, enabled):
    handler = handler_cls.__new__(handler_cls)
    handler.generation_config = GenerationConfig(session_routing=enabled)
    entry = {"id": "multi_turn_base_1", "function": [], "question": [[{"role": "user", "content": "question"}]]}

    data = handler._pre_query_processing_prompting(entry)
    options = handler._session_request_options(data, {})

    if enabled:
        session_id = options["extra_headers"]["X-Session-ID"]
        assert options["extra_body"]["session_params"]["session_id"] == session_id
    else:
        assert options == {}
