# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import json
import random
import re
import string
from pathlib import Path
from typing import Any

from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
)
from jinja2 import Environment
from overrides import override

from domyn_bfcl.oss import DomynOSSMixin


class MinistralFCHandler(DomynOSSMixin):
    """Ministral 3 behind a completions endpoint, prompted in Mistral's native tool format."""

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model=False,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        """Upstream's handler, keeping special tokens in the output."""
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.skip_special_tokens = False

    @staticmethod
    def _parse_native_tool_calls(text: str) -> list[dict]:
        """Parse ``[TOOL_CALLS]name[ARGS]{json}`` (v1) and ``[TOOL_CALLS]name{json}`` (v2) calls."""
        parsed = []
        for part in text.split("[TOOL_CALLS]"):
            part = part.strip()
            if not part:
                continue
            if "[ARGS]" in part:
                # v1 format: name[ARGS]{json}
                name, args_str = part.split("[ARGS]", 1)
            else:
                # v2 format: name{json}  (function name runs directly into '{')
                brace_pos = part.find("{")
                if brace_pos == -1:
                    continue
                name = part[:brace_pos]
                args_str = part[brace_pos:]
            name = name.strip()
            args_str = args_str.strip()
            if not name:
                continue
            try:
                decoder = json.JSONDecoder()
                arguments, _ = decoder.raw_decode(args_str)
                parsed.append({"name": name, "arguments": arguments})
            except json.JSONDecodeError:
                continue
        return parsed

    @staticmethod
    def _parse_tool_calls(text: str) -> list[dict]:
        """Parse tool calls from model output, trying multiple formats.

        Supports:
        - Native v1: [TOOL_CALLS]func_name[ARGS]{json}
        - Native v2: [TOOL_CALLS]func_name{json}  (no [ARGS] separator)
        - JSON array: [{...}]
        - Python function-call syntax

        Returns a list of {"name": str, "arguments": dict} dicts, or [] on failure.
        """
        if "[TOOL_CALLS]" in text and (parsed := MinistralFCHandler._parse_native_tool_calls(text)):
            return parsed

        # Format 2: JSON array [{"name": ..., "arguments": ...}]
        json_match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if isinstance(data, list) and all(isinstance(c, dict) and "name" in c for c in data):
                    return [
                        {
                            "name": c["name"],
                            "arguments": c.get("arguments", c.get("parameters", {})),
                        }
                        for c in data
                    ]
            except json.JSONDecodeError:
                pass

        # Format 3: Python function-call syntax [func(param=val)]
        try:
            decoded = default_decode_ast_prompting(text)
            if decoded:
                return [{"name": next(iter(c)), "arguments": next(iter(c.values()))} for c in decoded]
        except Exception:  # noqa: BLE001  # unparseable output means no calls
            pass

        return []

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        if isinstance(result, list):
            return result
        parsed = self._parse_tool_calls(result)
        return [{c["name"]: c["arguments"]} for c in parsed]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        if isinstance(result, list):
            return convert_to_function_call(result)
        parsed = self._parse_tool_calls(result)
        if parsed:
            return convert_to_function_call([{c["name"]: c["arguments"]} for c in parsed])
        try:
            return default_decode_execute_prompting(result, has_tool_call_tag)
        except Exception:  # noqa: BLE001  # unparseable output means no calls
            return []

    @override
    def _format_prompt(self, messages, function) -> list[int] | str:
        """Token ids in Mistral's native tool format (no system prompt) with tools, else the rendered template.

        Passing token IDs is necessary because vLLM uses the MistralTokenizer to encode
        raw strings — that tokenizer does NOT recognise [AVAILABLE_TOOLS] etc. as special
        tokens, so the model never sees the correct input IDs.  Bypassing string encoding
        by passing IDs directly (which the OpenAI completions API supports) gives the
        model the exact token sequence produced by encode_chat_completion.
        """
        if function:
            # Native Mistral format: [AVAILABLE_TOOLS]{tools}[/AVAILABLE_TOOLS][INST]…[/INST]
            # No system prompt — the model was trained to call tools in this exact format.
            # The thinking-instructions system message (ending in "self-contained response")
            # suppresses tool-calling behaviour; omitting it lets the model use [TOOL_CALLS].
            prompt = f"[AVAILABLE_TOOLS]{json.dumps(function)}[/AVAILABLE_TOOLS]"
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content") or ""
                if role == "user":
                    prompt += f"[INST]{content}[/INST]"
                elif role == "assistant":
                    # Re-add previous assistant turn (may contain [TOOL_CALLS]…)
                    prompt += content + self.tokenizer.eos_token
                elif role == "tool":
                    prompt += f"[TOOL_RESULTS]{content}[/TOOL_RESULTS]"
            # HF tokenizer correctly maps [AVAILABLE_TOOLS] → token 5, [INST] → 3, etc.
            return self.tokenizer.encode(prompt, add_special_tokens=True)
        env = Environment()
        env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
        template = env.from_string((Path(__file__).parent / "ministral_fc.jinja").read_text())
        return template.render(
            messages=messages,
            tools=None,
            add_generation_prompt=True,
            bos_token="",
            eos_token=self.tokenizer.eos_token,
        )

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        return {"message": [], "function": functions}

    @staticmethod
    def generate_random_string() -> str:
        characters = string.ascii_letters + string.digits
        return "".join(random.choices(characters, k=9))

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        raw_response = api_response.choices[0].text

        # print("Raw model response:", raw_response, "--- End of response ---")

        reasoning_content = ""
        think_match = re.search(r"\[THINK\](.*?)\[/THINK\]", raw_response, re.DOTALL)
        if think_match:
            reasoning_content = think_match.group(1).strip()

        parsed_calls = self._parse_tool_calls(raw_response)

        tool_call_ids = []
        if parsed_calls:
            model_responses = [{c["name"]: c["arguments"]} for c in parsed_calls]
            tool_call_ids = [self.generate_random_string() for _ in parsed_calls]
        else:
            model_responses = raw_response

        if think_match:
            history_content = raw_response[think_match.end() :].strip()
            if not history_content:
                history_content = raw_response
        else:
            history_content = raw_response

        return {
            "model_responses": model_responses,
            "reasoning_content": reasoning_content,
            "model_responses_message_for_chat_history": history_content,
            "tool_call_ids": tool_call_ids,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["message"].append(
            {
                "role": "assistant",
                "content": model_response_data["model_responses_message_for_chat_history"],
            }
        )
        return inference_data

    @override
    def _add_execution_results_prompting(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        tool_call_ids = model_response_data.get("tool_call_ids", [])
        for i, execution_result in enumerate(execution_results):
            tool_call_id = tool_call_ids[i] if i < len(tool_call_ids) else self.generate_random_string()
            inference_data["message"].append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": execution_result,
                }
            )
        return inference_data
