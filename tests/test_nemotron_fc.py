"""NemotronOSSHandler prompt formatting."""

import pytest

from domyn_bfcl.config import GenerationConfig
from domyn_bfcl.handlers.nemotron_fc import NemotronOSSHandler

_FUNCTIONS = [{"name": "f", "description": "d", "parameters": {"type": "dict", "properties": {}}}]
_MESSAGES = [{"role": "user", "content": "hi"}]


def _render(generation_config):
    handler = NemotronOSSHandler.__new__(NemotronOSSHandler)
    handler.generation_config = GenerationConfig(**generation_config)
    return handler._format_prompt(_MESSAGES, _FUNCTIONS)


@pytest.mark.parametrize(
    ("generation_config", "mode"),
    [
        ({"reasoning": True}, "on"),
        ({"reasoning": False}, "off"),
        # chat_template_kwargs is what the driver resolves per think/nothink mode
        ({"reasoning": True, "chat_template_kwargs": {"enable_thinking": False}}, "off"),
        ({"reasoning": False, "chat_template_kwargs": {"enable_thinking": True}}, "on"),
        ({}, "on"),
    ],
)
def test_thinking_follows_generation_config(generation_config, mode):
    prompt = _render(generation_config)
    assert f"detailed thinking {mode}" in prompt
    assert prompt.count("detailed thinking") == 1
