"""ChatFCHandler: history goes back as native tool_calls, not Domyn's <tool_call> text."""

from unittest.mock import MagicMock

from domyn_bfcl.handlers.chat_fc import ChatFCHandler


def _response(tool_calls=None, content=None):
    message = MagicMock(content=content, tool_calls=tool_calls, reasoning_content=None)
    return MagicMock(
        choices=[MagicMock(message=message)],
        usage=MagicMock(prompt_tokens=10, completion_tokens=5),
    )


def _tool_call(call_id, name, arguments):
    tc = MagicMock(id=call_id)
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _handler():
    return ChatFCHandler.__new__(ChatFCHandler)


def test_tool_calls_go_back_native_with_matching_ids():
    handler = _handler()
    response = _response(tool_calls=[_tool_call("call_1", "f", '{"x": 1}')])
    parsed = handler._parse_query_response_prompting(response)

    # Scoring still sees the <tool_call> form the shared decoder reads.
    assert handler.decode_ast(parsed["model_responses"], "python", False) == [{"f": {"x": 1}}]

    data = {"message": []}
    handler._add_assistant_message_prompting(data, parsed)
    handler._add_execution_results_prompting(data, ["42"], parsed)
    assistant, tool = data["message"]
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["function"] == {"name": "f", "arguments": '{"x": 1}'}
    assert "<tool_call>" not in (assistant["content"] or "")
    assert tool == {"role": "tool", "content": "42", "tool_call_id": "call_1"}


def test_plain_answer_goes_back_as_text():
    handler = _handler()
    parsed = handler._parse_query_response_prompting(_response(content="No tool fits."))
    data = {"message": []}
    handler._add_assistant_message_prompting(data, parsed)
    assert data["message"] == [{"role": "assistant", "content": "No tool fits."}]
