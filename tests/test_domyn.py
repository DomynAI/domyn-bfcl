"""DomynHandler: request construction, response parsing and tool-call decoding."""

from unittest.mock import MagicMock

import httpx
import pytest
import tenacity.nap
from bfcl_eval.constants.enums import ModelStyle, ReturnFormat
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.utils import convert_to_tool
from openai import (
    APIConnectionError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from domyn_bfcl.config import GenerationConfig
from domyn_bfcl.handlers.domyn import (
    DomynHandler,
)
from domyn_bfcl.oss import estimate_prompt_tokens

# ══════════════════════════════════════════════════════════════════════════════
# DomynHandler — request construction & response parsing
# ══════════════════════════════════════════════════════════════════════════════

_SAMPLE_MESSAGES = [
    {"role": "user", "content": "What's the weather like in New York?"},
    {
        "role": "assistant",
        "content": (
            "<think>\nThe user is asking about the weather in New York. I should "
            "use the weather tool to get this information.\n</think>\n"
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nNew York\n"
            "</parameter>\n</function>\n</tool_call>"
        ),
    },
    {
        "role": "tool",
        "content": '{"temperature": 22, "condition": "Sunny", "humidity": 45, "wind_speed": 10}',
    },
    {
        "role": "assistant",
        "content": (
            "The weather in New York is currently sunny with a temperature of "
            "22°C. The humidity is at 45% with a wind speed of 10 km/h. It's a "
            "great day to be outside!"
        ),
    },
    {"role": "user", "content": "Thanks! What about Boston?"},
]

_FUNCTIONS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "dict",
            "properties": {"city": {"type": "string", "description": "City name."}},
            "required": ["city"],
        },
    }
]

_GEN_CONFIG = {
    "token_safety_margin": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "chat_template_kwargs": {"enable_thinking": False},
}


def _make_handler(gen_config=None):
    """A DomynHandler with server-dependent __init__ bypassed."""
    h = DomynHandler.__new__(DomynHandler)
    h.model_path_or_id = "my-domyn-edge-model"
    h.generation_config = GenerationConfig(**(gen_config if gen_config is not None else _GEN_CONFIG))
    h.max_context_length = 131072  # the architecture's, set by spin_up_local_server
    # Every prompt renders as 50 tokens, recording what it was asked to render
    h.tokenizer = MagicMock()
    h.tokenizer.apply_chat_template.return_value = list(range(50))
    h.client = MagicMock()
    return h


def _captured_kwargs(handler):
    """Run _query_prompting on the sample and return the create() kwargs."""
    inference_data = {"message": _SAMPLE_MESSAGES, "function": _FUNCTIONS}
    handler._query_prompting(inference_data)
    return handler.client.chat.completions.create.call_args.kwargs


def _tool_call(name, arguments):
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _mock_response(tool_calls=None, content=None, usage=(11, 7), finish_reason="stop", reasoning=None):
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.tool_calls = tool_calls
    choice.message.content = content
    choice.message.reasoning_content = reasoning
    choice.message.reasoning = None
    resp = MagicMock()
    resp.choices = [choice]
    if usage is None:
        resp.usage = None
    else:
        resp.usage.prompt_tokens, resp.usage.completion_tokens = usage
    return resp


# ── _query_prompting: what is sent to chat.completions ────────────────────────


def test_query_forwards_sampling_params():
    kwargs = _captured_kwargs(_make_handler())
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.9


def test_query_leaves_unset_sampling_params_to_the_server():
    kwargs = _captured_kwargs(_make_handler({"top_k": 20}))
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert set(kwargs["extra_body"]) == {"top_k", "chat_template_kwargs", "add_generation_prompt"}


def test_query_without_max_model_len_budgets_against_the_architecture_window():
    """As the completions handlers do (DomynOSSMixin._budget_completion)."""
    # 131072 max_context_length - 50 prompt (stubbed tokenizer) - 256 margin
    assert _captured_kwargs(_make_handler())["max_tokens"] == 130766


def test_query_the_smaller_window_binds():
    cfg = {**_GEN_CONFIG, "max_model_len": 262144}
    assert _captured_kwargs(_make_handler(cfg))["max_tokens"] == 130766


def test_query_max_tokens_fills_the_window_by_default():
    cfg = {**_GEN_CONFIG, "max_model_len": 8192}
    # 8192 window - 50 prompt (stubbed tokenizer) - 256 margin
    assert _captured_kwargs(_make_handler(cfg))["max_tokens"] == 7886


def test_query_estimates_a_prompt_the_template_cannot_render(capsys):
    handler = _make_handler({**_GEN_CONFIG, "max_model_len": 32768})
    handler.tokenizer.apply_chat_template.side_effect = ValueError("bad template")
    kwargs = _captured_kwargs(handler)
    estimate = estimate_prompt_tokens(kwargs["messages"], kwargs["tools"])
    # an estimate keeps half its size free when that exceeds the configured margin
    margin = max(256, estimate // 2)
    assert kwargs["max_tokens"] == 32768 - estimate - margin
    assert "chars//4 estimate" in capsys.readouterr().out


def test_query_counts_the_prompt_as_sent():
    handler = _make_handler({**_GEN_CONFIG, "max_model_len": 8192})
    kwargs = _captured_kwargs(handler)
    render = handler.tokenizer.apply_chat_template.call_args.kwargs
    assert render["conversation"] == kwargs["messages"]
    assert render["tools"] == kwargs["tools"]
    assert render["add_generation_prompt"]
    assert render["tokenize"]
    assert render["return_dict"] is False
    # rendered with the template flags the server renders with
    flags = kwargs["extra_body"]["chat_template_kwargs"]
    assert {k: render[k] for k in flags} == flags


def test_query_tools_sent_once_top_level():
    kwargs = _captured_kwargs(_make_handler())
    expected = convert_to_tool(_FUNCTIONS, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)
    assert kwargs["tools"] == expected
    assert "tools" not in kwargs["extra_body"]


def test_query_extra_body_thinking_flags_match_config():
    """enable_thinking from the config must reach the chat template via
    chat_template_kwargs, preserve_thinking defaults to on, and
    add_generation_prompt is always set."""
    extra = _captured_kwargs(_make_handler())["extra_body"]
    assert extra["chat_template_kwargs"]["preserve_thinking"] is True
    assert extra["chat_template_kwargs"]["enable_thinking"] is False
    assert extra["add_generation_prompt"] is True


@pytest.mark.parametrize("reasoning", [True, False])
def test_query_thinking_follows_reasoning_without_enable_thinking(reasoning):
    kwargs = _captured_kwargs(_make_handler({"reasoning": reasoning}))
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is reasoning


def test_query_keeps_every_chat_template_kwarg():
    cfg = {"chat_template_kwargs": {"preserve_thinking": False, "custom": "x"}}
    handler = _make_handler(cfg)
    flags = _captured_kwargs(handler)["extra_body"]["chat_template_kwargs"]
    assert flags == {"preserve_thinking": False, "custom": "x", "enable_thinking": True}
    assert handler.tokenizer.apply_chat_template.call_args.kwargs["custom"] == "x"


def test_query_without_functions_omits_tools():
    handler = _make_handler()
    handler._query_prompting({"message": _SAMPLE_MESSAGES, "function": []})
    kwargs = handler.client.chat.completions.create.call_args.kwargs
    assert "tools" not in kwargs
    assert "tools" not in kwargs["extra_body"]


def test_inference_records_the_finish_reason():
    handler = _make_handler()
    handler.client.chat.completions.create.return_value = _mock_response(content="cut", finish_reason="length")
    entry = {"id": "simple_python_0", "question": [_SAMPLE_MESSAGES[:1]], "function": _FUNCTIONS}
    result, metadata = handler.inference(entry, False, False)
    assert result == "cut"
    assert metadata["finish_reason"] == ["length"]


# ── generate_with_backoff: retrying transient endpoint failures ───────────────


def _api_error(cls, status=500):
    """The exception the OpenAI SDK raises for an HTTP `status` from the endpoint."""
    request = httpx.Request("POST", "https://gateway/v1/chat/completions")
    if cls is APIConnectionError:
        return APIConnectionError(request=request)
    return cls("boom", response=httpx.Response(status, request=request), body=None)


@pytest.fixture
def no_backoff_sleep(monkeypatch):
    """Retries wait 6-120s between attempts; drop the wait, keep the counting."""
    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda _seconds: None)


def test_query_retries_rate_limit_until_it_succeeds(no_backoff_sleep):
    """A LiteLLM router answers 429 while its replicas are in cooldown. Without a
    retry, _llm_response_generation records that 429 as the model's answer and
    scores it as a wrong response, so the run reports a number built on noise."""
    handler = _make_handler()
    sentinel = _mock_response(content="ok")
    handler.client.chat.completions.create.side_effect = [
        _api_error(RateLimitError, 429),
        _api_error(RateLimitError, 429),
        sentinel,
    ]

    api_response, _ = handler._query_prompting({"message": _SAMPLE_MESSAGES, "function": _FUNCTIONS})

    assert api_response is sentinel
    assert handler.client.chat.completions.create.call_count == 3


@pytest.mark.parametrize("error_cls", [APIConnectionError, InternalServerError])
def test_query_retries_connection_and_server_errors(no_backoff_sleep, error_cls):
    handler = _make_handler()
    sentinel = _mock_response(content="ok")
    handler.client.chat.completions.create.side_effect = [
        _api_error(error_cls),
        sentinel,
    ]

    api_response, _ = handler._query_prompting({"message": _SAMPLE_MESSAGES, "function": _FUNCTIONS})

    assert api_response is sentinel
    assert handler.client.chat.completions.create.call_count == 2


def test_query_gives_up_after_five_attempts(no_backoff_sleep):
    """retry_with_backoff sets no default stop, so the explicit stop matters: an
    endpoint that never recovers must fail the entry, not hang the whole run."""
    handler = _make_handler()
    handler.client.chat.completions.create.side_effect = [_api_error(RateLimitError, 429) for _ in range(20)]

    with pytest.raises(RateLimitError):
        handler._query_prompting({"message": _SAMPLE_MESSAGES, "function": _FUNCTIONS})

    assert handler.client.chat.completions.create.call_count == 5


def test_query_does_not_retry_context_window_error(no_backoff_sleep):
    """A prompt that overflows the context window fails identically every time,
    so it must surface at once instead of burning five attempts on it."""
    handler = _make_handler()
    handler.client.chat.completions.create.side_effect = _api_error(BadRequestError, 400)

    with pytest.raises(BadRequestError):
        handler._query_prompting({"message": _SAMPLE_MESSAGES, "function": _FUNCTIONS})

    assert handler.client.chat.completions.create.call_count == 1


# ── _parse_query_response_prompting: parsing the reply ────────────────────────


def test_parse_tool_call_serialised_to_xml():
    """A Boston tool call must serialise to the <tool_call>{json}</tool_call> shape
    the shared decode_ast / decode_execute expect, with arguments as a real dict."""
    resp = _mock_response(tool_calls=[_tool_call("get_weather", '{"city": "Boston"}')])
    out = _make_handler()._parse_query_response_prompting(resp)
    assert out["model_responses"] == (
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Boston"}}\n</tool_call>'
    )
    assert out["model_responses_message_for_chat_history"] == {
        "role": "assistant",
        "content": out["model_responses"],
    }
    assert out["input_token"] == 11
    assert out["output_token"] == 7


def test_parse_multiple_tool_calls_newline_joined():
    resp = _mock_response(
        tool_calls=[
            _tool_call("get_weather", '{"city": "Boston"}'),
            _tool_call("get_weather", '{"city": "Chicago"}'),
        ]
    )
    out = _make_handler()._parse_query_response_prompting(resp)
    assert out["model_responses"].count("<tool_call>") == 2
    assert '"city": "Boston"' in out["model_responses"]
    assert '"city": "Chicago"' in out["model_responses"]
    # the two calls are separated by a newline join, not concatenated
    assert "</tool_call>\n<tool_call>" in out["model_responses"]


def test_parse_plain_text_response():
    """No tool calls → the assistant's natural-language answer is passed through."""
    answer = "It's sunny in Boston, 20°C."
    resp = _mock_response(tool_calls=None, content=answer)
    out = _make_handler()._parse_query_response_prompting(resp)
    assert out["model_responses"] == answer
    assert out["model_responses_message_for_chat_history"]["content"] == answer


def test_parse_invalid_json_arguments_falls_back_to_raw_string():
    """Malformed argument JSON must not crash — it is embedded as a raw string."""
    resp = _mock_response(tool_calls=[_tool_call("get_weather", "not-json")])
    out = _make_handler()._parse_query_response_prompting(resp)
    assert out["model_responses"] == ('<tool_call>\n{"name": "get_weather", "arguments": "not-json"}\n</tool_call>')


def test_parse_missing_usage_yields_zero_tokens():
    resp = _mock_response(content="ok", usage=None)
    out = _make_handler()._parse_query_response_prompting(resp)
    assert out["input_token"] == 0
    assert out["output_token"] == 0


def test_parse_keeps_out_of_band_reasoning_content():
    """vLLM with a reasoning parser puts thinking on message.reasoning_content;
    dropping it loses the trace from multi-turn history and debug output."""
    resp = _mock_response(content="It's sunny.", reasoning="Boston is coastal.")
    out = _make_handler()._parse_query_response_prompting(resp)
    assert out["reasoning_content"] == "Boston is coastal."
    assert out["model_responses"] == "It's sunny."


def test_parse_splits_inline_think_block_out_of_content():
    """Without a server-side reasoning parser the thinking stays inline; it must
    be split off so it does not end up in the scored response."""
    resp = _mock_response(content="<think>\nBoston is coastal.\n</think>\n\nIt's sunny.")
    out = _make_handler()._parse_query_response_prompting(resp)
    assert out["reasoning_content"] == "Boston is coastal."
    assert out["model_responses"] == "It's sunny."


# ── decode_ast: parsing the XML tool-call format the model emits ──────────────


def _decode(raw, language=ReturnFormat.PYTHON):
    return DomynHandler.decode_ast(_make_handler(), raw, language=language, has_tool_call_tag=False)


def test_decode_xml_call_coerces_argument_types():
    """The template emits strings raw and everything else as JSON, so the inverse
    must yield a real int here. BFCL's type_checker is strict: "7890" fails."""
    raw = (
        "\n\n<tool_call>\n<function=get_user_info>\n"
        "<parameter=user_id>\n7890\n</parameter>\n"
        "<parameter=special>\nblack\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert _decode(raw) == [{"get_user_info": {"user_id": 7890, "special": "black"}}]


def test_decode_xml_call_coerces_booleans():
    raw = (
        "<tool_call>\n<function=github_star>\n"
        "<parameter=repos>\nShishirPatil/gorilla,gorilla-llm/gorilla-cli\n</parameter>\n"
        "<parameter=aligned>\ntrue\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    args = _decode(raw)[0]["github_star"]
    assert args["aligned"] is True
    # a comma/slash-laden string must survive untouched
    assert args["repos"] == "ShishirPatil/gorilla,gorilla-llm/gorilla-cli"


def test_decode_xml_nested_json_argument():
    """A dict-valued argument goes out via tojson, so it must come back a dict."""
    raw = (
        "<tool_call>\n<function=ChaDri_change_drink>\n"
        "<parameter=drink_id>\nlatte\n</parameter>\n"
        '<parameter=new_preferences>\n{"size": "large", "hot": true}\n</parameter>\n'
        "</function>\n</tool_call>"
    )
    args = _decode(raw)[0]["ChaDri_change_drink"]
    assert args["new_preferences"] == {"size": "large", "hot": True}


@pytest.mark.parametrize("language", [ReturnFormat.JAVA, ReturnFormat.JAVASCRIPT, "java", "JavaScript"])
def test_decode_java_and_js_keep_every_value_as_string(language):
    """ast_checker.py:387-425 rejects any Java/JS argument that is not a str and
    runs its own java_type_converter / js_type_converter, so coercing here would
    turn every number and boolean into a type_error:java / type_error:js."""
    raw = (
        "<tool_call>\n<function=SQLCompletionAnalyzer_makeProposalsFromObject>\n"
        "<parameter=object>\nCustomers\n</parameter>\n"
        '<parameter=params>\n{"limit": "50", "schema": "public"}\n</parameter>\n'
        "<parameter=useShortName>\ntrue\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    args = _decode(raw, language)[0]["SQLCompletionAnalyzer_makeProposalsFromObject"]
    assert all(isinstance(v, str) for v in args.values()), args
    assert args["useShortName"] == "true"
    assert args["params"] == '{"limit": "50", "schema": "public"}'


def test_decode_two_tool_call_blocks_yield_two_calls():
    raw = (
        "\n\n<tool_call>\n<function=get_current_weather>\n"
        "<parameter=location>\nBeijing, China\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=get_current_weather>\n"
        "<parameter=location>\nShanghai, China\n</parameter>\n</function>\n</tool_call>"
    )
    assert _decode(raw) == [
        {"get_current_weather": {"location": "Beijing, China"}},
        {"get_current_weather": {"location": "Shanghai, China"}},
    ]


def test_decode_xml_preserves_multiline_string_parameter():
    """The call-format instructions advertise multi-line values, so only the
    single delimiter newline on each side may be stripped."""
    raw = (
        "<tool_call>\n<function=write_note>\n"
        "<parameter=body>\nline one\n\n  indented two\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert _decode(raw)[0]["write_note"]["body"] == "line one\n\n  indented two"


def test_decode_prose_without_tool_call_stays_empty():
    """Irrelevance detection depends on this: abstaining must decode to []."""
    assert _decode("I don't have a tool that can do that.") == []


def test_decode_legacy_json_tool_call_still_parses():
    """Domyn Small and DomynHandler's native-tool_calls branch both produce JSON
    inside <tool_call>; the XML support must not regress it."""
    raw = '<tool_call>\n{"name": "spotify.play", "arguments": {"artist": "Muse"}}\n</tool_call>'
    assert _decode(raw) == [{"spotify.play": {"artist": "Muse"}}]


def test_decode_recovers_call_truncated_before_closing_tool_call():
    """finish_reason="length" can cut the closing </tool_call>; the whole-string
    fallback keeps that from scoring as a silent zero."""
    raw = "<tool_call>\n<function=get_user_info>\n<parameter=user_id>\n7890\n</parameter>\n</function>"
    assert _decode(raw) == [{"get_user_info": {"user_id": 7890}}]


def test_decode_execute_renders_xml_call_as_function_string():
    calls = DomynHandler.decode_execute(
        _make_handler(),
        "<tool_call>\n<function=get_user_info>\n<parameter=user_id>\n7890\n</parameter>\n</function>\n</tool_call>",
        has_tool_call_tag=False,
    )
    assert calls == ["get_user_info(user_id=7890)"]
