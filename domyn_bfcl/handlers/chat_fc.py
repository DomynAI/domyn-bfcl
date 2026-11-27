# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from domyn_bfcl.handlers.domyn import DomynHandler


class ChatFCHandler(DomynHandler):
    """Generic chat-API handler for models no upstream or Domyn handler covers.

    Everything is DomynHandler's: chat completions with tools= and the run's
    chat_template_kwargs, so the server applies the model's own chat template and
    vLLM's --tool-call-parser turns its output into tool_calls (the swarm must run
    a parser for the model); past calls go back as native tool_calls, with
    tool_call_id on the results, so the template renders them in the model's own
    format. Its own name keeps the generic models apart in the registry.
    """
