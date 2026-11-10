# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Domyn behaviour for BFCL's local (OSS) handlers, layered over upstream ``OSSHandler``.

Divergences from upstream ``bfcl_eval.model_handler.local_inference.base_oss_handler``:

- Completion budget: ``max_tokens`` is the largest completion that fits the served
  window (``fit_max_tokens``; upstream caps it at 4096).
- Sampling: temperature, top_p and presence_penalty come from ``config.GENERATION``,
  plus an allowlist of vLLM sampling extras; unset (None) values are not sent, so the
  server's defaults apply (upstream sends temperature only, default 0.001).
- Timeouts: ``request_timeout`` for client and request, no client retries (upstream: a
  72000 s request timeout and the SDK's 2 retries).
- Served model id: ``$REMOTE_OPENAI_MODEL``, when set, names the model in requests
  (upstream, and unset, send the local path or HF id it loaded the tokenizer from).
- Finish reasons: each result entry gets ``finish_reason``, the server's reason for
  every request of the entry, in order (upstream records none, so a completion cut by
  ``max_tokens`` cannot be told from one that stopped).

Upstream handlers are used through a composed class, e.g.
``class DomynQwenFCHandler(DomynOSSMixin, QwenFCHandler)``.
"""

import json
import os
import threading
import time
from typing import Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from openai import OpenAI
from openai.types import Completion
from openai.types.chat import ChatCompletion
from overrides import override

from domyn_bfcl import config

# generation_config keys forwarded to vLLM as extra sampling parameters.
SAMPLING_EXTRA_KEYS = (
    "top_k",
    "min_p",
    "repetition_penalty",
    "frequency_penalty",
    "chat_template_kwargs",
)


# Rough characters per token, for a prompt the tokenizer cannot count
CHARS_PER_TOKEN = 4

# The entry each thread is generating: BFCL runs entries in parallel threads on one handler.
_entry = threading.local()


def estimate_prompt_tokens(messages: list[dict], tools: list[dict] | None) -> int:
    """The prompt's length from its characters (message contents and tool schemas)."""
    chars = sum(len(str(message.get("content") or "")) for message in messages)
    chars += len(json.dumps(tools)) if tools else 0
    return chars // CHARS_PER_TOKEN


def fit_max_tokens(max_tokens: int | None, prompt_tokens: int, window: int, margin: int) -> int:
    """``max_tokens`` (unset: the whole window) cut to what the prompt leaves of the window, at least 1."""
    room = window - prompt_tokens - margin
    return max(1, min(max_tokens or room, room))


class DomynOSSMixin(OSSHandler):
    """Domyn budget, sampling, timeouts and finish reasons over an upstream ``OSSHandler``."""

    # Set by some handlers only; sent to vLLM when present
    stop_token_ids: list[int]
    skip_special_tokens: bool

    def __init__(
        self,
        model_name: str,
        temperature: float,
        registry_name: str,
        is_fc_model: bool,
        **kwargs: Any,  # noqa: ANN401  # forwarded to upstream's untyped __init__
    ) -> None:
        """Upstream's handler, with a client that uses ``request_timeout`` and does not retry."""
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            registry_name=registry_name,
            is_fc_model=is_fc_model,
            **kwargs,
        )
        self.generation_config = config.GENERATION
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.generation_config.request_timeout,
            max_retries=0,
        )

    @override
    def inference(self, test_entry: dict, include_input_log: bool, exclude_state_log: bool) -> tuple[Any, dict]:
        """Upstream's, on the served model id, with the entry's finish reasons in its metadata."""
        # spin_up_local_server (final upstream) points model_path_or_id at the local path
        # or HF id it loads the tokenizer from; requests must name what the server serves.
        if served_model := os.getenv("REMOTE_OPENAI_MODEL"):
            self.model_path_or_id = served_model
        _entry.finish_reasons = finish_reasons = []
        result, metadata = super().inference(test_entry, include_input_log, exclude_state_log)
        metadata["finish_reason"] = finish_reasons
        return result, metadata

    @staticmethod
    def _record_finish(api_response: Completion | ChatCompletion) -> None:
        """Keep the finish reason of a request of the entry being generated, if any."""
        if (finish_reasons := getattr(_entry, "finish_reasons", None)) is not None:
            finish_reasons.append(api_response.choices[0].finish_reason)

    def _budget_completion(self, input_token_count: int, margin: int) -> int:
        """Completion budget that fits the served context window.

        ``max_context_length`` is the architecture's limit (from the model config);
        ``max_model_len`` is what the server will actually accept. The smaller binds.
        Shared by the completions requests here and DomynHandler's chat completions.
        """
        window = min(
            self.generation_config.max_model_len or self.max_context_length,
            self.max_context_length,
        )
        return fit_max_tokens(
            self.generation_config.max_tokens,
            input_token_count,
            window,
            margin,
        )

    def _sampling_kwargs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """``(request kwargs, extra_body)`` sampling parameters, None values left out."""
        generation = self.generation_config
        kwargs = {
            key: value
            for key in ("temperature", "top_p", "presence_penalty")
            if (value := getattr(generation, key)) is not None
        }
        extra_body = {key: value for key in SAMPLING_EXTRA_KEYS if (value := getattr(generation, key)) is not None}
        return kwargs, extra_body

    @override
    def _query_prompting(self, inference_data: dict) -> tuple[Any, float]:
        # We use the OpenAI Completions API
        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]

        # A str, or token ids for handlers that build them directly (MinistralFCHandler)
        formatted_prompt: str | list[int] = self._format_prompt(message, function)
        if isinstance(formatted_prompt, list):
            prompt_for_log = self.tokenizer.decode(formatted_prompt, skip_special_tokens=False)
            input_token_count = len(formatted_prompt)
        else:
            prompt_for_log = formatted_prompt
            input_token_count = len(self.tokenizer.tokenize(formatted_prompt))
        inference_data["inference_input_log"] = {"formatted_prompt": prompt_for_log}

        leftover_tokens_count = self._budget_completion(input_token_count, self.generation_config.token_safety_margin)

        sampling_kwargs, extra_body = self._sampling_kwargs()
        if hasattr(self, "stop_token_ids"):
            extra_body["stop_token_ids"] = self.stop_token_ids
        if hasattr(self, "skip_special_tokens"):
            extra_body["skip_special_tokens"] = self.skip_special_tokens

        start_time = time.time()
        api_response = self.client.completions.create(
            model=self.model_path_or_id,
            prompt=formatted_prompt,
            max_tokens=leftover_tokens_count,
            extra_body=extra_body or None,
            **sampling_kwargs,
        )
        end_time = time.time()
        self._record_finish(api_response)

        return api_response, end_time - start_time
