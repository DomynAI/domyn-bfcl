"""MinistralFCHandler: prompt building and tool-call parsing.

With tools the prompt is token ids in Mistral's native format; without, the
handler renders ministral_fc.jinja. Parsing covers the v1
([TOOL_CALLS]name[ARGS]{json}) and v2 ([TOOL_CALLS]name{json}) call shapes.
"""

from unittest.mock import MagicMock

from domyn_bfcl.handlers.ministral_fc import MinistralFCHandler

_TOOL = [
    {
        "name": "calculate_triangle_area",
        "description": "Calculate triangle area.",
        "parameters": {
            "type": "dict",
            "properties": {
                "base": {"type": "integer"},
                "height": {"type": "integer"},
            },
            "required": ["base", "height"],
        },
    }
]

_USER_MSG = [{"role": "user", "content": "Find the area of a triangle with base 10, height 5."}]


def _make_handler_with_tokenizer():
    """Return a MinistralFCHandler whose tokenizer is a minimal mock."""
    # Special tokens that the HF tokenizer encodes as single IDs
    special = {
        "[AVAILABLE_TOOLS]": 5,
        "[/AVAILABLE_TOOLS]": 6,
        "[INST]": 3,
        "[/INST]": 4,
        "[TOOL_CALLS]": 9,
        "[TOOL_RESULTS]": 12,
        "[/TOOL_RESULTS]": 13,
        "<s>": 1,
        "</s>": 2,
    }

    def _encode(text, add_special_tokens=True):
        ids = [1] if add_special_tokens else []  # BOS
        i = 0
        while i < len(text):
            matched = False
            for tok, tid in special.items():
                if text[i:].startswith(tok):
                    ids.append(tid)
                    i += len(tok)
                    matched = True
                    break
            if not matched:
                ids.append(ord(text[i]) % 1000 + 100)
                i += 1
        return ids

    tokenizer = MagicMock()
    tokenizer.eos_token = "</s>"
    tokenizer.encode.side_effect = _encode

    handler = MinistralFCHandler.__new__(MinistralFCHandler)
    handler.tokenizer = tokenizer
    return handler


# ── _format_prompt ────────────────────────────────────────────────────────────


def test_format_prompt_without_tools_renders_the_template():
    prompt = _make_handler_with_tokenizer()._format_prompt(_USER_MSG, None)
    system = prompt[prompt.index("[SYSTEM_PROMPT]") : prompt.index("[/SYSTEM_PROMPT]")]
    assert "HOW YOU SHOULD THINK" in system
    assert "[AVAILABLE_TOOLS]" not in prompt
    assert "triangle" in prompt[prompt.index("[INST]") : prompt.index("[/INST]")]
    assert prompt.rstrip().endswith("[/INST]")


def test_format_prompt_token_ids_start_with_bos_available_tools():
    """Token sequence must begin BOS(1) → [AVAILABLE_TOOLS](5)."""
    handler = _make_handler_with_tokenizer()
    ids = handler._format_prompt(_USER_MSG, _TOOL)
    assert ids[0] == 1, "First token must be BOS"
    assert ids[1] == 5, "Second token must be [AVAILABLE_TOOLS]"


def test_format_prompt_with_tools_has_the_tools_and_no_system_prompt():
    """The thinking-instructions system prompt suppresses tool calling, so it is left out."""
    handler = _make_handler_with_tokenizer()
    handler._format_prompt(_USER_MSG, _TOOL)
    prompt_str = handler.tokenizer.encode.call_args[0][0]
    assert "calculate_triangle_area" in prompt_str
    assert "[SYSTEM_PROMPT]" not in prompt_str


# ── Parsing ───────────────────────────────────────────────────────────────────


def test_parse_native_v1_single_call():
    """v1 format: [TOOL_CALLS]name[ARGS]{json}"""
    text = '[TOOL_CALLS]calculate_triangle_area[ARGS]{"base": 10, "height": 5}'
    assert MinistralFCHandler._parse_tool_calls(text) == [
        {"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}
    ]


def test_parse_native_v2_single_call():
    """v2 format: [TOOL_CALLS]name{json}  (no [ARGS] separator — model native output)."""
    text = '[TOOL_CALLS]calculate_triangle_area{"base": 10, "height": 5}'
    assert MinistralFCHandler._parse_tool_calls(text) == [
        {"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}
    ]


def test_parse_native_v2_after_think_block():
    """v2 tool call that comes after a [THINK]…[/THINK] block must be parsed."""
    text = (
        "[THINK]I should call calculate_triangle_area.[/THINK]"
        '[TOOL_CALLS]calculate_triangle_area{"base": 10, "height": 5}'
    )
    result = MinistralFCHandler._parse_tool_calls(text)
    assert len(result) == 1
    assert result[0] == {
        "name": "calculate_triangle_area",
        "arguments": {"base": 10, "height": 5},
    }


def test_parse_native_inside_think_block():
    """Tool call embedded inside [THINK]…[/THINK] must still be found."""
    text = '[THINK][TOOL_CALLS]calculate_triangle_area[ARGS]{"base": 10, "height": 5}[/THINK]'
    result = MinistralFCHandler._parse_tool_calls(text)
    assert len(result) == 1
    assert result[0]["name"] == "calculate_triangle_area"


def test_parse_native_with_trailing_eos():
    """</s> after the JSON arguments must not corrupt parsing."""
    text = '[TOOL_CALLS]func[ARGS]{"x": 1}</s>'
    assert MinistralFCHandler._parse_tool_calls(text) == [{"name": "func", "arguments": {"x": 1}}]


def test_parse_native_multiple_calls_v2():
    text = '[TOOL_CALLS]func1{"a": 1}[TOOL_CALLS]func2{"b": 2}'
    result = MinistralFCHandler._parse_tool_calls(text)
    assert result == [
        {"name": "func1", "arguments": {"a": 1}},
        {"name": "func2", "arguments": {"b": 2}},
    ]


def test_parse_no_tool_calls_returns_empty():
    text = "I cannot find a suitable function for this request."
    assert MinistralFCHandler._parse_tool_calls(text) == []
