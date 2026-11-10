# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""The run's generation settings: ``domyn-bfcl --generation-config '<json>'``.

``install`` validates them into ``GENERATION``; every Domyn handler reads it when built.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict


class GenerationConfig(BaseModel):
    """Sampling, thinking mode, completion budget and timeout of every request.

    Unset (None) sampling params are not sent, so the server's defaults apply.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    reasoning: bool = True
    max_model_len: int | None = None
    max_tokens: int | None = None
    token_safety_margin: int = 256
    request_timeout: float = 3600
    max_samples: int = -1

    @property
    def enable_thinking(self) -> bool:
        """The thinking mode: ``chat_template_kwargs["enable_thinking"]``, else ``reasoning``."""
        return bool((self.chat_template_kwargs or {}).get("enable_thinking", self.reasoning))


GENERATION = GenerationConfig()


def configure(generation_config: dict) -> None:
    """Validate the run's settings into ``GENERATION``."""
    global GENERATION
    GENERATION = GenerationConfig.model_validate(generation_config)
