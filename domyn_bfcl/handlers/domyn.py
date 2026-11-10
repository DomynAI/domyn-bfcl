# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import re
import time
from typing import Any

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    convert_to_tool,
    retry_with_backoff,
)
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion
from overrides import override
from tenacity import stop_after_attempt
from tqdm import tqdm

from domyn_bfcl.oss import (
    CHARS_PER_TOKEN,
    DomynOSSMixin,
    estimate_prompt_tokens,
)

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)


def _coerce_param_value(raw: str):
    """Invert the chat template's argument encoding.

    Domyn Edge's chat template renders each argument as
    ``args_value | string if args_value is string else args_value | tojson``,
    i.e. strings go out raw and everything else goes out as JSON. So a value
    that parses as JSON is that JSON value -- unless it parses as a *string*,
    which can only happen when the text carried literal quotes the template
    never adds, in which case the literal text is what the model meant.
    """
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw
    return raw if isinstance(parsed, str) else parsed


def _keeps_raw_strings(language) -> bool:
    """Whether decode_ast must hand the checker unconverted strings.

    The Java and JavaScript AST checkers reject any argument that is not a
    ``str`` and run java_type_converter / js_type_converter themselves
    (ast_checker.py:387-425), so the JSON coercion the Python categories need
    would turn every numeric or boolean argument into a type_error there.
    """
    value = getattr(language, "value", language)
    return isinstance(value, str) and value.lower() in ("java", "javascript")


def _parse_xml_tool_calls(text: str, coerce: bool = True) -> list[dict]:
    """Parse ``<function=NAME><parameter=KEY>value</parameter></function>`` blocks.

    Values are stripped of the single delimiter newline the template puts on
    each side, but not of inner whitespace -- multi-line string parameters are
    an explicitly supported case.
    """
    convert = _coerce_param_value if coerce else (lambda raw: raw)
    return [
        {
            "name": name,
            "arguments": {key: convert(value.strip("\n")) for key, value in _PARAMETER_RE.findall(body)},
        }
        for name, body in _FUNCTION_RE.findall(text)
    ]


def split_reasoning(text: str) -> tuple[str, str]:
    """Split an inline ``<think>...</think>`` prefix off a response.

    Returns ``(reasoning_content, cleaned_response)``. When the server has a
    reasoning parser enabled the thinking arrives out-of-band instead and the
    text passes through unchanged.
    """
    if "</think>" not in text:
        return "", text
    segments = text.split("</think>")
    reasoning = segments[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
    return reasoning, segments[-1].lstrip("\n")


class DomynHandler(DomynOSSMixin):
    """Domyn handler using chat completions with native tools= parameter.

    Model-agnostic: the server applies the model's chat template and tool parser
    (Domyn Edge's <function=...> XML, Domyn Small's <tool_call> JSON), and the run's
    chat_template_kwargs, with its thinking mode, reach that template. Tool call responses
    are serialised back to <tool_call> JSON for decode_ast / decode_execute, which
    also read raw <function=...> XML left in the content.
    """

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model=True,
        **kwargs,
    ) -> None:
        """Upstream's handler, always in FC mode."""
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            registry_name=registry_name,
            is_fc_model=True,
            **kwargs,
        )
        self.is_fc_model = True
        self.model_name_huggingface = model_name.replace("-FC", "")

    # -------------------------------------------------------------------------
    # Token budgeting
    # -------------------------------------------------------------------------

    def _chat_template_kwargs(self) -> dict:
        """The run's chat_template_kwargs, with its thinking mode; preserve_thinking defaults to on."""
        generation = self.generation_config
        return {
            "preserve_thinking": True,
            **(generation.chat_template_kwargs or {}),
            "enable_thinking": generation.enable_thinking,
        }

    def _count_prompt(self, messages: list[dict], tools: list[dict]) -> tuple[int, int]:
        """``(prompt tokens, safety margin)``, rendered as the server will render it.

        A prompt the template cannot render is estimated from its characters instead;
        the estimate misses the template's own tokens, so its margin grows with it.
        """
        margin = self.generation_config.token_safety_margin
        try:
            prompt = self.tokenizer.apply_chat_template(
                conversation=messages,
                tools=tools or None,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                **self._chat_template_kwargs(),
            )
        except Exception as exc:  # noqa: BLE001
            estimate = estimate_prompt_tokens(messages, tools)
            tqdm.write(
                f"⚠️  Warning: could not render the prompt with the chat template "
                f"({type(exc).__name__}: {exc}); budgeting max_tokens on a "
                f"chars//{CHARS_PER_TOKEN} estimate of {estimate} tokens"
            )
            return estimate, max(margin, estimate // 2)
        return len(prompt), margin

    # -------------------------------------------------------------------------
    # Inference: chat completions with tools
    # -------------------------------------------------------------------------

    @retry_with_backoff(
        error_type=[RateLimitError, APIConnectionError, InternalServerError],
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def generate_with_backoff(self, **kwargs):
        start = time.time()
        try:
            api_response = self.client.chat.completions.create(**kwargs)
        except APITimeoutError as exc:
            msg = f"generation exceeded request_timeout ({self.generation_config.request_timeout}s)"
            raise TimeoutError(msg) from exc
        return api_response, time.time() - start

    @override
    def _query_prompting(self, inference_data: dict) -> tuple[ChatCompletion, float]:
        functions: list[dict] = inference_data["function"]

        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS) if functions else []

        messages: list[dict] = list(inference_data["message"])

        sampling_kwargs, extra_body = self._sampling_kwargs()
        kwargs: dict = {
            "model": self.model_path_or_id,
            "messages": messages,
            **sampling_kwargs,
            "extra_body": {
                **extra_body,
                "chat_template_kwargs": self._chat_template_kwargs(),
                "add_generation_prompt": True,
            },
        }
        if tools:
            kwargs["tools"] = tools

        # Keep prompt + completion inside the context window, counted with the tokenizer
        # BFCL loaded at setup and the template flags the request renders with. No local
        # default: an unset max_tokens means "as much as the window allows".
        kwargs["max_tokens"] = self._budget_completion(*self._count_prompt(messages, tools))

        api_response, latency = self.generate_with_backoff(**kwargs)
        self._record_finish(api_response)

        return api_response, latency

    # -------------------------------------------------------------------------
    # Response parsing
    # -------------------------------------------------------------------------

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        choice = api_response.choices[0]
        message = choice.message
        tool_calls = message.tool_calls or []

        # Serialise to <tool_call> XML — compatible with decode_ast / decode_execute
        parts = []
        for tc in tool_calls:
            fn = tc.function
            try:
                args = json.loads(fn.arguments)
            except json.JSONDecodeError:
                args = fn.arguments
            parts.append(f'<tool_call>\n{{"name": "{fn.name}", "arguments": {json.dumps(args)}}}\n</tool_call>')

        # vLLM surfaces the thinking out-of-band when a reasoning parser is
        # enabled; without one it stays inline in the content as <think>...</think>.
        reasoning_content = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""
        content = message.content or ""
        if not reasoning_content:
            reasoning_content, content = split_reasoning(content)

        # When the server did not parse tool calls natively, the content still
        # holds the raw <function=...> XML; decode_ast parses either shape.
        model_responses = "\n".join(parts) if parts else content

        return {
            "model_responses": model_responses,
            "reasoning_content": reasoning_content,
            "model_responses_message_for_chat_history": {
                "role": "assistant",
                "content": model_responses,
            },
            "input_token": api_response.usage.prompt_tokens if api_response.usage else 0,
            "output_token": api_response.usage.completion_tokens if api_response.usage else 0,
        }

    # -------------------------------------------------------------------------
    # History management
    # -------------------------------------------------------------------------

    @override
    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        # Store the assistant response as XML content, not as a native tool_calls
        # object.
        inference_data["message"].append(
            {
                "role": "assistant",
                "content": model_response_data["model_responses"],
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
        for execution_result in execution_results:
            inference_data["message"].append(
                {
                    "role": "tool",
                    "content": str(execution_result),
                }
            )
        return inference_data

    # -------------------------------------------------------------------------
    # Decoding
    # -------------------------------------------------------------------------

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        """Decode the model's tool calls, formatted as below.

        <tool_call>
        {"name": "spotify.play", "arguments": {"artist": "Taylor Swift", "duration": 20}}
        </tool_call>
        <tool_call>
        {"name": "spotify.play", "arguments": {"artist": "Maroon 5", "duration": 15}}
        </tool_call>
        """
        tool_calls = self._extract_tool_calls(result, coerce=not _keeps_raw_strings(language))
        if not isinstance(tool_calls, list) or any(not isinstance(item, dict) for item in tool_calls):
            msg = f"Model did not return a list of function calls: {result}"
            raise ValueError(msg)

        return [{call["name"]: dict(call.get("arguments", call.get("parameters", {})).items())} for call in tool_calls]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if not isinstance(tool_calls, list) or any(not isinstance(item, dict) for item in tool_calls):
            msg = f"Model did not return a list of function calls: {result}"
            raise ValueError(msg)
        decoded_result = []
        for item in tool_calls:
            if isinstance(item, str):
                item = eval(item)
            decoded_result.append({item["name"]: item["arguments"]})
        return convert_to_function_call(decoded_result)

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        # FC models use its own system prompt, so no need to add any message
        return {"message": [], "function": functions}

    @override
    def _format_prompt(self, messages, function):
        msg = (
            f"{type(self).__name__} does not render prompts; it calls "
            "chat.completions in _query_prompting and the server applies the template."
        )
        raise NotImplementedError(msg)

    @staticmethod
    def _extract_tool_calls(input_string, coerce: bool = True):
        """Extract tool calls from either shape the Domyn models emit.

        Domyn Small's chat template asks for JSON inside <tool_call>, while Domyn
        Edge's asks for a nested <function=...><parameter=...> block (left raw in
        the content when the server's tool parser does not pick it up).
        JSON is tried first; the XML parse is a fallback, and cannot fire on a
        JSON payload because XML tags never parse as JSON.

        ``coerce`` turns XML argument text into the JSON value it encodes; pass
        False for the Java/JavaScript categories, which need raw strings.
        """
        result = []
        for match in _TOOL_CALL_RE.findall(input_string):
            match = match.strip()
            try:
                result.append(ast.literal_eval(match))
                continue
            except (SyntaxError, ValueError):
                pass
            try:
                result.append(json.loads(match))
                continue
            except json.JSONDecodeError:
                pass
            result.extend(_parse_xml_tool_calls(match, coerce))

        # Recover output the <tool_call> scan missed: calls the model emitted
        # unwrapped, or a response truncated (finish_reason="length") before
        # its closing </tool_call>. Without this those score as a silent zero.
        if not result and "<function=" in input_string:
            result = _parse_xml_tool_calls(input_string, coerce)
        return result
