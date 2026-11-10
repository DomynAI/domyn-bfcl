# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import re
from pathlib import Path
from typing import Any

from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
)
from jinja2 import Template
from overrides import override

from domyn_bfcl.oss import DomynOSSMixin

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that can call functions to answer user queries. When you want to call a function, respond with the <function_calls> tag, and provide a list of function calls in the following format:
<function_calls>[
{"name": "function_name1", "arguments": {"arg1": "value1", "arg2": "value2"}},
{"name": "function_name2", "arguments": {"param1": "value1", "param2": "value2"}}
]</function_calls>
Make sure to use the <function_calls> tag and provide the function calls in the specified format. If you want to provide a direct answer without calling any function, simply respond with the answer text without using the <function_calls> tag.
The available functions you can call are defined within <functions> tags. Each function has a name and a list of parameters with their types. You can call any of these functions by including them in the <function_calls> tag as described above.
"""  # noqa: E501  # prompt text


class OlmoFCHandler(DomynOSSMixin):
    """Olmo 3 behind a completions endpoint, calling functions in ``<function_calls>`` tags."""

    def __init__(self, model_name, temperature, registry_name, is_fc_model=True, **kwargs) -> None:
        """Upstream's handler, always in FC mode."""
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            registry_name=registry_name,
            is_fc_model=is_fc_model,
            **kwargs,
        )
        self.is_fc_model = True
        self.model_name_huggingface = model_name.replace("-FC", "")

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if isinstance(tool_calls, list) and all(isinstance(c, dict) and "name" in c for c in tool_calls) and tool_calls:
            return [{call["name"]: call.get("arguments", call.get("parameters", {}))} for call in tool_calls]
        # Fallback: Python function-call syntax, e.g. [func_name(param=val)]
        try:
            return default_decode_ast_prompting(result, language, has_tool_call_tag)
        except Exception:  # noqa: BLE001  # unparseable output means no calls
            return []

    @override
    def decode_execute(self, result, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        decoded_result = []
        for item in tool_calls:
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except json.JSONDecodeError:
                    continue
            if not isinstance(item, dict) or "name" not in item:
                continue
            decoded_result.append({item["name"]: item.get("arguments", item.get("parameters", {}))})
        if decoded_result:
            return convert_to_function_call(decoded_result)
        # Fallback: Python function-call syntax
        try:
            return default_decode_execute_prompting(result, has_tool_call_tag)
        except Exception:  # noqa: BLE001  # unparseable output means no calls
            return []

    @override
    def _format_prompt(self, messages, function):
        messages = list(messages)  # shallow copy — never mutate the caller's list

        # Build system prompt
        original_prompt = DEFAULT_SYSTEM_PROMPT
        if messages[0]["role"] == "system":
            original_prompt = DEFAULT_SYSTEM_PROMPT + "\n\n" + messages.pop(0)["content"].strip()
        messages.insert(
            0,
            {
                "role": "system",
                "content": original_prompt,
                "functions": json.dumps(function) if function else None,
            },
        )

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
                if "function_calls" in msg and msg["function_calls"] and not isinstance(msg["function_calls"], str):
                    msg["function_calls"] = json.dumps(msg["function_calls"])
            render_messages.append(msg)

        # Apply chat template
        template = Template((Path(__file__).parent / "olmo_fc.jinja").read_text())
        return template.render(messages=render_messages, add_generation_prompt=True)

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        # FC models use its own system prompt, so no need to add any message
        return {"message": [], "function": functions}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        model_response = api_response.choices[0].text
        extracted_tool_calls = self._extract_tool_calls(model_response)

        reasoning_content = ""
        cleaned_response = model_response
        if "</think>" in model_response:
            parts = model_response.split("</think>")
            reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            cleaned_response = parts[-1].lstrip("\n")

        if len(extracted_tool_calls) > 0:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": "",
                "function_calls": extracted_tool_calls,
            }

        else:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": cleaned_response,
            }

        model_responses_message_for_chat_history["reasoning_content"] = reasoning_content

        return {
            "model_responses": cleaned_response,
            "reasoning_content": reasoning_content,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"],
        )
        return inference_data

    @staticmethod
    def _extract_tool_calls(input_string):
        pattern = r"<function_calls>(.*?)</function_calls>"
        matches = re.findall(pattern, input_string, re.DOTALL)

        # Process matches into a list of dictionaries
        result = []
        for match in matches:
            match = match.strip()
            try:
                parsed = ast.literal_eval(match)
            except (SyntaxError, ValueError, TypeError):
                try:
                    parsed = json.loads(match)
                except json.JSONDecodeError:
                    continue
            # print(match)
            if isinstance(parsed, list):
                result.extend(parsed)  # ✅ keep ALL function calls
            else:
                result.append(parsed)
        return result
