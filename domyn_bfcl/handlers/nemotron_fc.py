# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
import re
from pathlib import Path
from typing import Any

from bfcl_eval.model_handler.utils import (
    combine_consecutive_user_prompts,
    convert_system_prompt_into_user_prompt,
    decoded_output_to_execution_list,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
)
from jinja2 import Template
from overrides import override

from domyn_bfcl.oss import DomynOSSMixin

# Maps shell command names to the ordered GorillaFileSystem API parameter names.
# Flags (args starting with "-") are filtered out before positional mapping.
_SHELL_PARAM_MAP: dict[str, list[str]] = {
    "mkdir": ["dir_name"],
    "mv": ["source", "destination"],
    "cp": ["source", "destination"],
    "rm": ["file_name"],
    "rmdir": ["dir_name"],
    "touch": ["file_name"],
    "cat": ["file_name"],
    "echo": ["content", "file"],
    "grep": ["file_name", "pattern"],
    "sort": ["file_name"],
    "diff": ["file_name1", "file_name2"],
    "wc": ["file_name"],
    "tail": ["file_name"],
    "head": ["file_name"],
    "find": ["path"],
    "ls": [],
}


def _parse_shell_call_list(call: list) -> tuple[str, dict] | None:
    """Convert a shell-style list [cmd, arg1, ...] to (cmd_name, args_dict).

    Flags (entries starting with '-') are silently dropped.
    Returns None if the call is malformed.
    """
    if not isinstance(call, list) or not call or not isinstance(call[0], str):
        return None
    cmd = call[0]
    positional = [a for a in call[1:] if isinstance(a, str) and not a.startswith("-")]
    param_names = _SHELL_PARAM_MAP.get(cmd, [])
    arguments = {param_names[i]: positional[i] for i in range(min(len(param_names), len(positional)))}
    return cmd, arguments


def _parse_shell_list_of_lists(calls: list) -> list[dict]:
    """Convert a list of shell-style call lists to BFCL AST dicts, skipping bad entries."""
    result = []
    for call in calls:
        parsed = _parse_shell_call_list(call)
        if parsed:
            name, args = parsed
            result.append({name: args})
    return result


def _salvage_inner_arrays(func_call_str: str) -> list[dict] | None:
    """Outer JSON is broken — try to salvage each inner array independently."""
    result = []
    for m in re.finditer(r"\[[^\[\]]*\]", func_call_str):
        try:
            call = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        parsed = _parse_shell_call_list(call)
        if parsed:
            name, args = parsed
            result.append({name: args})
    return result or None


def _parse_nemotron_json(func_call_str: str) -> list[dict] | None:
    """Parse Nemotron's native JSON tool call format.

    Accepts:
      - [{"name": "func", "arguments": {...}}, ...]  (standard JSON)
      - [["cmd", "arg1", "arg2"], ...]               (shell-style list of lists)
    Falls back to extracting individual inner arrays when the outer JSON is invalid
    (e.g. the model injects Python expressions like "%s" % p[0] into the list).
    Returns: [{"func": {args}}, ...] or None if nothing could be parsed.
    """
    try:
        data = json.loads(func_call_str.strip())
    except json.JSONDecodeError:
        return _salvage_inner_arrays(func_call_str)

    if not isinstance(data, list) or not data:
        return None

    # Shell-style list of lists: [["mkdir", "temp"], ["mv", "src", "dst"], ...]
    if isinstance(data[0], list):
        result = _parse_shell_list_of_lists(data)
        return result or None

    # Standard JSON: [{"name": "func", "arguments": {...}}, ...]
    result = []
    for call in data:
        if not isinstance(call, dict) or "name" not in call:
            return None
        args = call.get("arguments", {})
        if isinstance(args, str):
            with contextlib.suppress(json.JSONDecodeError):
                args = json.loads(args)
        result.append({call["name"]: args if isinstance(args, dict) else {}})
    return result or None


class NemotronOSSHandler(DomynOSSMixin):
    """Handler for Llama-3.1-Nemotron-Nano-8B-v1 behind a completions endpoint.

    Renders the model's chat template locally (nemotron_fc.jinja), with the
    "detailed thinking on/off" system line following the run's thinking mode.
    The model's XML-based function calling format:
    - <TOOLCALL>[function_calls]</TOOLCALL> for function calls
    - <AVAILABLE_TOOLS>{functions}</AVAILABLE_TOOLS> for function documentation
    """

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model=True,
        **kwargs,
    ) -> None:
        """Upstream's handler, in FC mode by default."""
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)

    # Although Nemotron is a FC model, its endpoint does not take in function docs,
    # but instead have them as part of the system prompt.
    # So we use the _query_prompting method for FC inference.
    # _query_FC = OpenAICompletionsHandler._query_prompting

    def _format_system_prompt(self, prompt, function_docs):
        """Format the system prompt in the Nemotron-specific XML format."""
        # print(
        #     "function_docs in format_system_prompt:\n",
        #     json.dumps(function_docs).replace("'", '"'),
        #     "\ndtype:",
        #     type(function_docs),
        # )

        return """You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
also point it out. You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of <TOOLCALL>[{"name": "func_name1", "arguments": {"param1": value1, "param2": value2}}, {"name": "func_name2", "arguments": {...}}]</TOOLCALL>

You SHOULD NOT include any other text in the response.
The available functions are defined between <AVAILABLE_TOOLS> and </AVAILABLE_TOOLS> tags.
"""  # noqa: E501  # prompt text

    @override
    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        """Process the input query and format it for the Nemotron model."""
        functions: list = test_entry["function"]

        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = convert_system_prompt_into_user_prompt(
                test_entry["question"][round_idx]
            )
            test_entry["question"][round_idx] = combine_consecutive_user_prompts(test_entry["question"][round_idx])

        test_entry["question"][0] = self._format_system_prompt(test_entry["question"][0], functions)

        inference_data["message"] = []
        return inference_data

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        toolcall_match = re.search(r"<TOOLCALL>(.*?)</TOOLCALL>", result, re.DOTALL)
        if not toolcall_match:
            return []
        func_call_str = toolcall_match.group(1)

        parsed = _parse_nemotron_json(func_call_str)
        if parsed is not None:
            return parsed

        try:
            decoded = default_decode_ast_prompting(func_call_str, language, has_tool_call_tag)
            if decoded and all(isinstance(c, dict) for c in decoded):
                return decoded
        except Exception:  # noqa: BLE001  # unparseable output means no calls
            return []
        return []

    @override
    def decode_execute(self, result, has_tool_call_tag):
        toolcall_match = re.search(r"<TOOLCALL>(.*?)</TOOLCALL>", result, re.DOTALL)
        if not toolcall_match:
            return []
        func_call_str = toolcall_match.group(1)

        parsed = _parse_nemotron_json(func_call_str)
        if parsed is not None:
            return decoded_output_to_execution_list(parsed)

        try:
            return default_decode_execute_prompting(func_call_str, has_tool_call_tag)
        except Exception:  # noqa: BLE001  # unparseable output means no calls
            return []

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        # Skip generic BFCL system prompt injection; _format_prompt handles it
        # via the Nemotron-specific system prompt + <AVAILABLE_TOOLS> from the jinja template.
        return {"message": [], "function": test_entry["function"]}

    @override
    def _format_prompt(self, messages, function):
        messages = list(messages)  # shallow copy — never mutate the caller's list

        # Replace any existing system message with the Nemotron-specific one.
        if messages and messages[0]["role"] == "system":
            messages.pop(0)
        messages.insert(0, {"role": "system", "content": self._format_system_prompt("", function)})

        # Find last real user message — reasoning content is only injected for
        # assistant messages after this index, keeping older turns clean.
        last_query_index = 0
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                last_query_index = i
                break

        # Build render-only copies — never mutate the stored message dicts
        render_messages = []
        for idx, message in enumerate(messages):
            msg = dict(message)
            if msg["role"] == "assistant":
                rc = msg.get("reasoning_content", "")
                if rc and rc.strip() and idx > last_query_index:
                    msg["content"] = f"<think>{rc}</think>\n{msg.get('content', '')}"
            render_messages.append(msg)

        # Apply chat template
        template = Template((Path(__file__).parent / "nemotron_fc.jinja").read_text())
        return template.render(
            messages=render_messages,
            add_generation_prompt=True,
            tools=function,
            enable_thinking=self.generation_config.enable_thinking,
        )

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        model_response = api_response.choices[0].text

        reasoning_content = ""
        cleaned_response = model_response
        if "</think>" in model_response:
            parts = model_response.split("</think>")
            reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            cleaned_response = parts[-1].lstrip("\n")

        return {
            "model_responses": cleaned_response,
            "reasoning_content": reasoning_content,
            "model_responses_message_for_chat_history": {
                "role": "assistant",
                "content": cleaned_response,
                "reasoning_content": reasoning_content,
            },
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["message"].append(model_response_data["model_responses_message_for_chat_history"])
        return inference_data
