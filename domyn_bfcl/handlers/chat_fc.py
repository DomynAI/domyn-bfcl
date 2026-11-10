# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from overrides import override

from domyn_bfcl.handlers.domyn import DomynHandler


class ChatFCHandler(DomynHandler):
    """Generic chat-API handler for models no upstream or Domyn handler covers.

    Requests are DomynHandler's: chat completions with tools= and the run's
    chat_template_kwargs, so the server applies the model's own chat template and
    vLLM's --tool-call-parser turns its output into tool_calls. The swarm must run
    a parser for the model.

    History differs: past calls go back as native tool_calls, with tool_call_id on
    the results, so the server's template renders them in the model's own format
    (DomynHandler writes them as <tool_call> JSON text, which is Domyn's format).
    """

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        parsed = super()._parse_query_response_prompting(api_response)
        message = api_response.choices[0].message
        tool_calls = message.tool_calls or []
        if tool_calls:
            parsed["model_responses_message_for_chat_history"] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        parsed["_tool_call_ids"] = [tc.id for tc in tool_calls]
        return parsed

    @override
    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["message"].append(model_response_data["model_responses_message_for_chat_history"])
        return inference_data

    @override
    def _add_execution_results_prompting(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        ids = model_response_data.get("_tool_call_ids", [])
        for i, execution_result in enumerate(execution_results):
            message: dict = {"role": "tool", "content": str(execution_result)}
            if i < len(ids):
                message["tool_call_id"] = ids[i]
            inference_data["message"].append(message)
        return inference_data
