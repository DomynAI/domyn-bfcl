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
from domyn_bfcl.oss import ContextWindowFullError

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


def _tool_call(name, arguments, call_id="call_0"):
    tc = MagicMock()
    tc.id = call_id
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
    # 131072 max_context_length - 50 prompt (stubbed tokenizer)
    assert _captured_kwargs(_make_handler())["max_tokens"] == 131022


def test_query_the_smaller_window_binds():
    cfg = {**_GEN_CONFIG, "max_model_len": 262144}
    assert _captured_kwargs(_make_handler(cfg))["max_tokens"] == 131022


def test_query_max_tokens_fills_the_window_by_default():
    cfg = {**_GEN_CONFIG, "max_model_len": 8192}
    # 8192 window - 50 prompt (stubbed tokenizer)
    assert _captured_kwargs(_make_handler(cfg))["max_tokens"] == 8142


def test_query_a_prompt_that_fills_the_window_is_not_sent():
    handler = _make_handler({**_GEN_CONFIG, "max_model_len": 50})  # the 50-token prompt fills it
    with pytest.raises(ContextWindowFullError):
        _captured_kwargs(handler)
    handler.client.chat.completions.create.assert_not_called()


def test_query_a_prompt_the_template_cannot_render_is_an_error():
    handler = _make_handler({**_GEN_CONFIG, "max_model_len": 32768})
    handler.tokenizer.apply_chat_template.side_effect = ValueError("bad template")
    with pytest.raises(ValueError, match="bad template"):
        _captured_kwargs(handler)
    handler.client.chat.completions.create.assert_not_called()


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
    chat_template_kwargs, preserve_thinking defaults to off, and
    add_generation_prompt is always set."""
    extra = _captured_kwargs(_make_handler())["extra_body"]
    assert extra["chat_template_kwargs"]["preserve_thinking"] is False
    assert extra["chat_template_kwargs"]["enable_thinking"] is False
    assert extra["add_generation_prompt"] is True


@pytest.mark.parametrize("reasoning", [True, False])
def test_query_thinking_follows_reasoning_without_enable_thinking(reasoning):
    kwargs = _captured_kwargs(_make_handler({"reasoning": reasoning}))
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is reasoning


def test_query_keeps_every_chat_template_kwarg():
    cfg = {"chat_template_kwargs": {"preserve_thinking": True, "custom": "x"}}
    handler = _make_handler(cfg)
    flags = _captured_kwargs(handler)["extra_body"]["chat_template_kwargs"]
    assert flags == {"preserve_thinking": True, "custom": "x", "enable_thinking": True}
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


# ── chat history: past calls replay natively, results carry their ids ─────────


def _history_after(handler, response, results):
    parsed = handler._parse_query_response_prompting(response)
    data = {"message": []}
    handler._add_assistant_message_prompting(data, parsed)
    handler._add_execution_results_prompting(data, results, parsed)
    return data["message"]


def test_history_replays_native_tool_calls_with_ids():
    """The template must render the model's own call format, not <tool_call> JSON text."""
    response = _mock_response(
        tool_calls=[
            _tool_call("get_weather", '{"city": "Boston"}', "call_a"),
            _tool_call("get_weather", '{"city": "Chicago"}', "call_b"),
        ]
    )
    assistant, *tools = _history_after(_make_handler(), response, ["sunny", "rainy"])
    assert assistant == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_a",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Boston"}'},
            },
            {
                "id": "call_b",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Chicago"}'},
            },
        ],
    }
    assert tools == [
        {"role": "tool", "content": "sunny", "tool_call_id": "call_a"},
        {"role": "tool", "content": "rainy", "tool_call_id": "call_b"},
    ]


def test_history_keeps_text_before_the_call():
    response = _mock_response(
        tool_calls=[_tool_call("get_weather", '{"city": "Boston"}')],
        content="Let me check.",
        reasoning="Need the weather.",
    )
    assistant, _ = _history_after(_make_handler(), response, ["sunny"])
    assert assistant["content"] == "Let me check."


def test_history_without_server_ids_still_pairs_results():
    response = _mock_response(tool_calls=[_tool_call("get_weather", '{"city": "Boston"}', call_id=None)])
    assistant, tool = _history_after(_make_handler(), response, ["sunny"])
    assert assistant["tool_calls"][0]["id"] == tool["tool_call_id"] == "call_0"


def test_history_with_unparsable_arguments_falls_back_to_text():
    """The server decodes replayed arguments as a JSON object; anything else would be a 400."""
    response = _mock_response(tool_calls=[_tool_call("get_weather", "not-json")])
    handler = _make_handler()
    assistant, tool = _history_after(handler, response, ["error"])
    assert assistant == {
        "role": "assistant",
        "content": handler._parse_query_response_prompting(response)["model_responses"],
    }
    assert tool == {"role": "tool", "content": "error"}


def test_history_of_raw_xml_content_stays_text_without_ids():
    """No native calls (the server's parser missed them): the content is already the model's format."""
    raw = "<tool_call>\n<function=get_weather>\n<parameter=city>\nBoston\n</parameter>\n</function>\n</tool_call>"
    assistant, tool = _history_after(_make_handler(), _mock_response(content=raw), ["sunny"])
    assert assistant == {"role": "assistant", "content": raw}
    assert tool == {"role": "tool", "content": "sunny"}


def test_query_counts_replayed_calls_with_decoded_arguments():
    """The server gets OpenAI's JSON-string arguments; the local count renders them as a
    dict, as vLLM does before applying a template that iterates them."""
    handler = _make_handler()
    response = _mock_response(tool_calls=[_tool_call("get_weather", '{"city": "Boston"}')])
    messages = [{"role": "user", "content": "Weather in Boston?"}]
    messages += _history_after(handler, response, ["sunny"])
    handler._query_prompting({"message": messages, "function": _FUNCTIONS})

    sent = handler.client.chat.completions.create.call_args.kwargs["messages"]
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == '{"city": "Boston"}'
    rendered = handler.tokenizer.apply_chat_template.call_args.kwargs["conversation"]
    assert rendered[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Boston"}
    assert rendered[2] == sent[2]
    # the history itself is left as sent
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == '{"city": "Boston"}'


# ── multi_turn_miss_func: held-out functions join the tools on their turn ─────

_HELD_OUT = {
    "name": "get_forecast",
    "description": "Get the forecast for a city.",
    "parameters": {
        "type": "dict",
        "properties": {"city": {"type": "string", "description": "City name."}},
        "required": ["city"],
    },
}


def _miss_func_entry():
    return {
        "id": "multi_turn_miss_func_0",
        "function": list(_FUNCTIONS),
        "missed_function": {"2": [_HELD_OUT]},
    }


def _tool_names(handler, inference_data):
    handler._query_prompting(inference_data)
    return [t["function"]["name"] for t in handler.client.chat.completions.create.call_args.kwargs["tools"]]


def test_held_out_function_joins_the_tools_on_its_turn():
    handler = _make_handler()
    entry = _miss_func_entry()
    data = handler._pre_query_processing_prompting(entry)

    data = handler.add_first_turn_message_prompting(data, [{"role": "user", "content": "Weather in Boston?"}])
    assert _tool_names(handler, data) == ["get_weather"]
    data = handler._add_next_turn_user_message_prompting(data, [{"role": "user", "content": "And Chicago?"}])
    assert _tool_names(handler, data) == ["get_weather"]

    # Upstream's prompting loop hands the holdout turn a user message listing the functions as text
    announced = [{"role": "user", "content": f"{[_HELD_OUT]}\nI have updated some more functions..."}]
    data = handler._add_next_turn_user_message_prompting(data, announced)
    assert _tool_names(handler, data) == ["get_weather", "get_forecast"]
    # ...which becomes upstream's FC message, since the functions now arrive as tools
    assert data["message"][-1] == {
        "role": "user",
        "content": "I have updated some more functions you can choose from. What about now?",
    }

    data = handler._add_next_turn_user_message_prompting(data, [{"role": "user", "content": "Thanks."}])
    assert _tool_names(handler, data) == ["get_weather", "get_forecast"]
    # the test entry itself is not changed
    assert entry["function"] == _FUNCTIONS


def test_entries_without_held_out_functions_keep_their_messages():
    handler = _make_handler()
    data = handler._pre_query_processing_prompting({"id": "multi_turn_base_0", "function": list(_FUNCTIONS)})
    data = handler.add_first_turn_message_prompting(data, [{"role": "user", "content": "a"}])
    data = handler._add_next_turn_user_message_prompting(data, [{"role": "user", "content": "b"}])
    assert data["message"] == [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    assert data["function"] == _FUNCTIONS


def test_history_replays_the_thinking_as_vllm_takes_it():
    """vLLM reads an assistant message's ``reasoning`` (not ``reasoning_content``) into the
    template; dropping it leaves the current turn's steps, and preserve_thinking, with empty
    <think> blocks."""
    response = _mock_response(tool_calls=[_tool_call("get_weather", '{"city": "Boston"}')], reasoning="Need it.")
    assistant, _ = _history_after(_make_handler(), response, ["sunny"])
    assert assistant["reasoning"] == "Need it."
    assert "reasoning_content" not in assistant

    answer, *_ = _history_after(_make_handler(), _mock_response(content="Sunny."), [])
    assert "reasoning" not in answer


def test_query_counts_the_thinking_as_the_template_reads_it():
    handler = _make_handler()
    messages = [
        {"role": "user", "content": "Weather in Boston?"},
        {"role": "assistant", "content": "Sunny.", "reasoning": "Checked."},
        {"role": "user", "content": "Thanks!"},
    ]
    handler._query_prompting({"message": messages, "function": _FUNCTIONS})
    rendered = handler.tokenizer.apply_chat_template.call_args.kwargs["conversation"]
    assert rendered[1]["reasoning_content"] == "Checked."
    assert "reasoning_content" not in handler.client.chat.completions.create.call_args.kwargs["messages"][1]


@pytest.mark.parametrize("model", ["Qwen/Qwen3.5-4B-FC", "Qwen/Qwen3.5-4B-Base-FC", "Qwen/Qwen3.5-9B-FC"])
def test_qwen3_5_uses_its_own_chat_template(model):
    """QwenFCHandler hard-codes Qwen3's JSON tool format; Qwen3.5's template uses <function=...> XML."""
    from domyn_bfcl.handlers.chat_fc import ChatFCHandler
    from domyn_bfcl.models import MODEL_CONFIGS

    assert MODEL_CONFIGS[model].model_handler is ChatFCHandler
    assert MODEL_CONFIGS[model].underscore_to_dot
