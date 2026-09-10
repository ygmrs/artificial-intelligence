import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]


BASE_DIR = Path(__file__).resolve().parent

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
TOOL_FUNCTIONS_FILE = Path(
    os.getenv("TOOL_FUNCTIONS_FILE", str(BASE_DIR / "tool_functions.json"))
)
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "5"))


SYSTEM_INSTRUCTIONS = """
You are an AI Systems Engineering Assistant.

Your role is to help engineers design, evaluate, and improve modern AI systems
with a focus on:

- RAG pipelines
- Agentic workflows
- Tool and function orchestration
- Grounded generation
- Answer verification
- AI safety guardrails
- Evaluation
- Observability and tracing
- Reliability, latency, cost, and scalability
- Secure AI integration patterns

Operate like a senior AI engineering advisor rather than a generic chatbot.

Engineering principles:

1. Production-minded engineering
   - Give practical and implementation-oriented answers.
   - Explain architecture, trade-offs, failure modes, testing, monitoring,
     deployment, and maintainability when relevant.

2. Grounded responses
   - Never fabricate facts, tool results, logs, metrics, or external behavior.
   - Clearly state assumptions.
   - If evidence or context is insufficient, say so directly.

3. Tool execution discipline
   - Use tools only when relevant.
   - Treat tool outputs as evidence.
   - Never claim that a tool executed unless its returned result confirms it.
   - If a real external integration is unavailable, explain that an adapter is
     required rather than pretending execution occurred.
   - Never expose secrets, tokens, credentials, or private implementation data.

4. AI system design
   For RAG systems, consider:
   - ingestion and normalization
   - chunking and metadata
   - embeddings
   - hybrid retrieval
   - reranking
   - context assembly
   - citations
   - claim verification
   - refusal logic
   - evaluation
   - observability

   For agentic systems, consider:
   - tool contracts
   - execution boundaries
   - state management
   - authorization
   - auditability
   - safety controls
   - deterministic fallbacks
   - human approval where appropriate

5. Security
   - Avoid unsafe tool execution.
   - Recommend least privilege, scoped permissions, validation, audit logging,
     and secure secret management.
   - Consider prompt injection, data leakage, tool misuse, and unsupported
     generation.

6. Response style
   - Be concise and technically strong.
   - Use clear engineering language.
   - Avoid hype and unsupported claims.
   - When providing code, make it runnable and maintainable.
""".strip()


def get_client() -> Any:
    """
    Create an OpenAI client.

    Keeping client creation separate makes the assistant easy to test by
    injecting a fake client.
    """
    if OpenAI is None:
        raise RuntimeError(
            "The OpenAI Python SDK is not installed. "
            "Install dependencies before running this script."
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set it in your environment before running the assistant."
        )

    return OpenAI()


def _normalize_function_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize legacy Assistants-style function definitions to the Responses
    API function-tool shape.

    Legacy:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {...}
            }
        }

    Responses:
        {
            "type": "function",
            "name": "...",
            "description": "...",
            "parameters": {...},
            "strict": ...
        }

    This compatibility layer can be removed after tool_functions.json has been
    migrated completely to the Responses API format.
    """
    if tool.get("type") != "function":
        return tool

    function_definition = tool.get("function")

    if not isinstance(function_definition, dict):
        return tool

    normalized: Dict[str, Any] = {
        "type": "function",
        "name": function_definition["name"],
        "description": function_definition.get("description", ""),
        "parameters": function_definition.get(
            "parameters",
            {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
    }

    if "strict" in function_definition:
        normalized["strict"] = function_definition["strict"]
    elif "strict" in tool:
        normalized["strict"] = tool["strict"]

    return normalized


def load_function_tools(file_path: Path = TOOL_FUNCTIONS_FILE) -> List[Dict[str, Any]]:
    """
    Load function-tool definitions.

    Both the current Responses API format and the repository's previous
    Assistants-style nested format are accepted during migration.
    """
    if not file_path.exists():
        print(
            f"Warning: {file_path} was not found. "
            "The assistant will run without custom tools."
        )
        return []

    try:
        with file_path.open("r", encoding="utf-8") as file:
            tools = json.load(file)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{file_path} contains invalid JSON: {exc}"
        ) from exc

    if not isinstance(tools, list):
        raise RuntimeError(
            f"{file_path} must contain a JSON array of tool definitions."
        )

    return [_normalize_function_tool(tool) for tool in tools]


def execute_ai_engineering_tool(
    function_name: str,
    function_args: Dict[str, Any],
    custom_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Safe local tool runtime.

    The runtime intentionally performs no external provider calls and handles
    no credentials. Unknown tools return adapter_required rather than
    pretending that an external action was completed.
    """
    custom_params = custom_params or {}
    normalized_name = function_name.strip().lower()

    if normalized_name in {
        "design_rag_pipeline",
        "create_rag_architecture",
        "generate_rag_design",
        "rag_pipeline_design",
    }:
        return {
            "status": "completed",
            "tool": function_name,
            "execution_mode": "local_ai_engineering_runtime",
            "input": function_args,
            "result": {
                "architecture": {
                    "ingestion": [
                        "Normalize documents",
                        "Extract text and metadata",
                        "Deduplicate content",
                        "Version source documents",
                        "Preserve source lineage",
                    ],
                    "indexing": [
                        "Create semantic chunks",
                        "Store raw documents in object storage",
                        "Store metadata in a relational or document database",
                        "Use hybrid retrieval with keyword and vector indexes",
                        "Apply permission and metadata filters before retrieval",
                    ],
                    "retrieval": [
                        "Run hybrid search",
                        "Apply reranking",
                        "Assemble context with source IDs",
                        "Detect weak, missing, stale, or conflicting evidence",
                    ],
                    "generation": [
                        "Generate only from retrieved evidence",
                        "Require citations for factual claims",
                        "Refuse when evidence is insufficient",
                        "Avoid unsupported synthesis",
                    ],
                    "verification": [
                        "Extract answer claims",
                        "Check claims against retrieved evidence",
                        "Reject unsupported claims",
                        "Return confidence and refusal reasons",
                    ],
                    "observability": [
                        (
                            "Log query, retrieved chunks, reranker scores, "
                            "answer, citations, and verifier results"
                        ),
                        (
                            "Track retrieval recall, faithfulness, citation "
                            "accuracy, refusal quality, latency, and cost"
                        ),
                    ],
                },
                "recommended_controls": [
                    "Evidence sufficiency gate",
                    "Citation validator",
                    "Claim-level verifier",
                    "Prompt-injection filtering",
                    "PII and sensitive-data handling",
                    "Human review for high-risk workflows",
                ],
            },
        }

    if normalized_name in {
        "verify_grounded_answer",
        "check_answer_grounding",
        "validate_claim_support",
        "claim_verification",
    }:
        answer = function_args.get("answer", "")
        evidence = function_args.get("evidence", [])
        claims = function_args.get("claims", [])

        if not answer:
            return {
                "status": "completed",
                "tool": function_name,
                "execution_mode": "local_ai_engineering_runtime",
                "result": {
                    "grounded": False,
                    "confidence": "low",
                    "reason": "No answer was provided for verification.",
                },
            }

        if not evidence:
            return {
                "status": "completed",
                "tool": function_name,
                "execution_mode": "local_ai_engineering_runtime",
                "result": {
                    "grounded": False,
                    "confidence": "low",
                    "reason": "No supporting evidence was provided.",
                    "action": (
                        "Refuse, ask for more evidence, or retrieve "
                        "additional sources."
                    ),
                },
            }

        return {
            "status": "completed",
            "tool": function_name,
            "execution_mode": "local_ai_engineering_runtime",
            "result": {
                "grounding_status": (
                    "evidence_available_not_semantically_verified"
                ),
                "confidence": "medium",
                "claims_checked": (
                    len(claims) if isinstance(claims, list) else 0
                ),
                "evidence_items": (
                    len(evidence) if isinstance(evidence, list) else 1
                ),
                "note": (
                    "Supporting evidence was provided, but this local runtime "
                    "does not perform semantic entailment. A production "
                    "verifier should perform claim-level semantic verification "
                    "before marking the answer as fully grounded."
                ),
            },
        }

    if normalized_name in {
        "evaluate_ai_architecture",
        "review_ai_system_design",
        "architecture_review",
        "assess_ai_system",
    }:
        return {
            "status": "completed",
            "tool": function_name,
            "execution_mode": "local_ai_engineering_runtime",
            "input": function_args,
            "result": {
                "review_dimensions": [
                    "retrieval quality",
                    "grounding strategy",
                    "tool execution safety",
                    "authorization boundaries",
                    "observability",
                    "evaluation coverage",
                    "latency and cost",
                    "failure handling",
                    "deployment readiness",
                ],
                "recommended_improvements": [
                    "Add offline evaluation datasets with expected citations.",
                    "Add retrieval recall and reranking quality metrics.",
                    "Add claim-level answer verification.",
                    "Log tool calls and outputs for auditability.",
                    (
                        "Separate model reasoning from deterministic "
                        "business logic."
                    ),
                    (
                        "Add refusal behavior for low-confidence or "
                        "unsupported answers."
                    ),
                ],
                "risk_areas": [
                    "hallucinated answers",
                    "prompt injection",
                    "stale retrieved context",
                    "over-permissive tools",
                    "missing audit trails",
                    "unclear ownership of generated output",
                ],
            },
        }

    if normalized_name in {
        "create_agent_observability_plan",
        "agent_observability_plan",
        "design_ai_observability",
    }:
        return {
            "status": "completed",
            "tool": function_name,
            "execution_mode": "local_ai_engineering_runtime",
            "input": function_args,
            "result": {
                "logs": [
                    "user query",
                    "selected tools",
                    "tool inputs",
                    "tool outputs",
                    "retrieved evidence",
                    "model response",
                    "citations",
                    "verification result",
                    "latency",
                    "cost estimate",
                    "error and refusal reason",
                ],
                "metrics": [
                    "retrieval recall",
                    "answer faithfulness",
                    "citation accuracy",
                    "tool success rate",
                    "refusal precision",
                    "p95 latency",
                    "cost per request",
                    "failed verification rate",
                ],
                "traces": [
                    "request trace ID",
                    "retrieval span",
                    "reranking span",
                    "generation span",
                    "tool execution span",
                    "verification span",
                ],
                "alerts": [
                    "high unsupported-claim rate",
                    "tool failure spike",
                    "retrieval-empty spike",
                    "latency regression",
                    "unexpected cost increase",
                ],
            },
        }

    return {
        "status": "adapter_required",
        "tool": function_name,
        "execution_mode": "safe_local_fallback",
        "input": function_args,
        "custom_params_detected": bool(custom_params),
        "message": (
            "No local implementation exists for this tool. "
            "A real external integration should use a dedicated adapter "
            "with authentication, authorization, validation, observability, "
            "timeouts, and error handling."
        ),
        "recommended_adapter_contract": {
            "validate_input": True,
            "enforce_authorization": True,
            "execute_external_call": "service-layer responsibility",
            "sanitize_output": True,
            "log_trace": True,
            "return_structured_result": True,
        },
    }


def build_tool_output(tool_call: Any) -> Dict[str, Any]:
    """
    Execute one Responses API function call and convert its result into a
    function_call_output item.
    """
    function_name = getattr(tool_call, "name", "unknown_function")
    raw_arguments = getattr(tool_call, "arguments", "{}") or "{}"
    call_id = getattr(tool_call, "call_id", None)

    if not call_id:
        raise RuntimeError(
            f"Function call '{function_name}' did not contain a call_id."
        )

    try:
        function_args = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        result = {
            "status": "error",
            "tool": function_name,
            "error_type": "invalid_json_arguments",
            "message": "Tool arguments were not valid JSON.",
            "details": str(exc),
        }

        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(result, ensure_ascii=False),
        }

    if not isinstance(function_args, dict):
        result = {
            "status": "error",
            "tool": function_name,
            "error_type": "invalid_argument_shape",
            "message": "Tool arguments must be a JSON object.",
        }

        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": json.dumps(result, ensure_ascii=False),
        }

    custom_params = function_args.pop("_custom_params", {})

    if not isinstance(custom_params, dict):
        custom_params = {}

    result = execute_ai_engineering_tool(
        function_name=function_name,
        function_args=function_args,
        custom_params=custom_params,
    )

    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps(result, ensure_ascii=False),
    }


class AISystemsAssistant:
    """
    Small Responses API client with:

    - multi-turn conversation state
    - function calling
    - safe local tool execution
    - bounded tool-call loops
    - dependency injection for tests
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        model: str = MODEL,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ) -> None:
        self.client = client or get_client()
        self.model = model
        self.tools = tools if tools is not None else load_function_tools()
        self.max_tool_rounds = max_tool_rounds
        self.previous_response_id: Optional[str] = None

    def reset(self) -> None:
        """Start a new conversation."""
        self.previous_response_id = None

    def _create_response(
        self,
        input_data: Any,
        previous_response_id: Optional[str] = None,
    ) -> Any:
        request: Dict[str, Any] = {
            "model": self.model,
            "instructions": SYSTEM_INSTRUCTIONS,
            "input": input_data,
            "tools": self.tools,
        }

        if previous_response_id:
            request["previous_response_id"] = previous_response_id

        return self.client.responses.create(**request)

    def ask(self, user_input: str) -> str:
        """
        Send one user turn.

        The method continues automatically when the model requests function
        calls. Each local result is returned as function_call_output and the
        Responses API continues from the preceding response.
        """
        if not user_input.strip():
            raise ValueError("user_input must not be empty.")

        response = self._create_response(
            input_data=user_input,
            previous_response_id=self.previous_response_id,
        )

        for _ in range(self.max_tool_rounds):
            tool_calls = [
                item
                for item in response.output
                if getattr(item, "type", None) == "function_call"
            ]

            if not tool_calls:
                self.previous_response_id = response.id

                output_text = (response.output_text or "").strip()

                if output_text:
                    return output_text

                return (
                    "The model completed the response without returning "
                    "text output."
                )

            tool_outputs = [
                build_tool_output(tool_call)
                for tool_call in tool_calls
            ]

            response = self._create_response(
                input_data=tool_outputs,
                previous_response_id=response.id,
            )

        raise RuntimeError(
            f"Exceeded maximum tool-call depth of "
            f"{self.max_tool_rounds} rounds."
        )


def run_interactive_session() -> None:
    assistant = AISystemsAssistant()

    print("AI Systems Engineering Assistant")
    print(f"Model: {assistant.model}")
    print(f"Loaded tools: {len(assistant.tools)}")
    print()
    print("Commands:")
    print("  reset - start a new conversation")
    print("  exit  - stop")
    print()
    print("Example prompts:")
    print("- Design a grounded RAG architecture for a large document corpus.")
    print("- Review an AI agent architecture for reliability and observability.")
    print("- Create an observability plan for a production AI agent.")
    print()

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        command = user_input.lower()

        if command in {"exit", "quit", "stop"}:
            print("Exiting.")
            break

        if command == "reset":
            assistant.reset()
            print("Conversation reset.")
            continue

        try:
            response = assistant.ask(user_input)
            print(f"\nAI: {response}\n")
        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    run_interactive_session()