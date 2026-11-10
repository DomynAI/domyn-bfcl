# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import os

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.eval_config import LOCAL_SERVER_PORT
from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)


class MockHandler(OpenAICompletionsHandler):
    """Talks to an already-running OpenAI-compatible endpoint.

    Deliberately *not* an ``OSSHandler``: BFCL calls ``spin_up_local_server`` for
    those, which imports transformers and demands a full model directory
    (``config.json``) even under ``--skip-server-setup``. The endpoint is already
    serving here, so the handler only needs its URL.

    ``_build_client_kwargs`` ignores upstream's ``OPENAI_*`` variables: ``bfcl`` loads
    ``.env`` with ``override=True``, and an ``OPENAI_BASE_URL`` there would otherwise
    redirect this run to the real OpenAI API.
    """

    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs) -> None:
        """Upstream's handler on the completions style, named as the endpoint serves the model."""
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        # The endpoint serves the model under its own id, which need not match
        # the bfcl registry name that selected this handler.
        if served_model := os.getenv("REMOTE_OPENAI_MODEL"):
            self.model_name = served_model

    def _build_client_kwargs(self):
        endpoint = os.getenv("LOCAL_SERVER_ENDPOINT", "localhost")
        port = os.getenv("LOCAL_SERVER_PORT", LOCAL_SERVER_PORT)
        return {
            "base_url": os.getenv("REMOTE_OPENAI_BASE_URL", f"http://{endpoint}:{port}/v1"),
            "api_key": os.getenv("REMOTE_OPENAI_API_KEY", "EMPTY"),
        }
