import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from ai_assistant import AISystemsAssistant, build_tool_output


class FakeResponsesAPI:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if not self._responses:
            raise AssertionError("No fake response remains.")

        return self._responses.pop(0)


class FakeOpenAIClient:
    def __init__(self, responses):
        self.responses = FakeResponsesAPI(responses)


def make_response(
    response_id,
    output=None,
    output_text="",
):
    return SimpleNamespace(
        id=response_id,
        output=output or [],
        output_text=output_text,
    )


def make_function_call(
    name,
    arguments,
    call_id="call_123",
):
    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=arguments,
        call_id=call_id,
    )


def test_assistant_returns_normal_text_response():
    response = make_response(
        response_id="response_1",
        output_text="Use hybrid retrieval followed by reranking.",
    )

    client = FakeOpenAIClient([response])

    assistant = AISystemsAssistant(
        client=client,
        model="test-model",
        tools=[],
    )

    result = assistant.ask(
        "How should I design retrieval for a production RAG system?"
    )

    assert result == "Use hybrid retrieval followed by reranking."
    assert assistant.previous_response_id == "response_1"

    assert len(client.responses.calls) == 1

    request = client.responses.calls[0]

    assert request["model"] == "test-model"
    assert (
        request["input"]
        == "How should I design retrieval for a production RAG system?"
    )
    assert request["tools"] == []
    assert "previous_response_id" not in request


def test_assistant_executes_function_call_and_returns_final_response():
    function_call = make_function_call(
        name="design_rag_pipeline",
        arguments=json.dumps(
            {
                "use_case": "enterprise knowledge assistant",
            }
        ),
        call_id="call_rag_1",
    )

    first_response = make_response(
        response_id="response_tool_call",
        output=[function_call],
    )

    final_response = make_response(
        response_id="response_final",
        output_text="The RAG architecture has been designed.",
    )

    client = FakeOpenAIClient(
        [
            first_response,
            final_response,
        ]
    )

    assistant = AISystemsAssistant(
        client=client,
        model="test-model",
        tools=[],
    )

    result = assistant.ask(
        "Design a grounded RAG architecture."
    )

    assert result == "The RAG architecture has been designed."
    assert assistant.previous_response_id == "response_final"

    assert len(client.responses.calls) == 2

    second_request = client.responses.calls[1]

    assert second_request["previous_response_id"] == "response_tool_call"

    tool_outputs = second_request["input"]

    assert len(tool_outputs) == 1

    tool_output = tool_outputs[0]

    assert tool_output["type"] == "function_call_output"
    assert tool_output["call_id"] == "call_rag_1"

    parsed_output = json.loads(tool_output["output"])

    assert parsed_output["status"] == "completed"
    assert parsed_output["tool"] == "design_rag_pipeline"
    assert (
        parsed_output["execution_mode"]
        == "local_ai_engineering_runtime"
    )


def test_build_tool_output_handles_invalid_json_arguments():
    function_call = make_function_call(
        name="design_rag_pipeline",
        arguments="{invalid-json",
        call_id="call_invalid_1",
    )

    result = build_tool_output(function_call)

    assert result["type"] == "function_call_output"
    assert result["call_id"] == "call_invalid_1"

    output = json.loads(result["output"])

    assert output["status"] == "error"
    assert output["tool"] == "design_rag_pipeline"
    assert output["error_type"] == "invalid_json_arguments"


def test_assistant_rejects_empty_input():
    client = FakeOpenAIClient([])

    assistant = AISystemsAssistant(
        client=client,
        model="test-model",
        tools=[],
    )

    with pytest.raises(
        ValueError,
        match="user_input must not be empty",
    ):
        assistant.ask("   ")


def test_reset_clears_conversation_state():
    response = make_response(
        response_id="response_1",
        output_text="Response.",
    )

    client = FakeOpenAIClient([response])

    assistant = AISystemsAssistant(
        client=client,
        model="test-model",
        tools=[],
    )

    assistant.ask("Hello")

    assert assistant.previous_response_id == "response_1"

    assistant.reset()

    assert assistant.previous_response_id is None