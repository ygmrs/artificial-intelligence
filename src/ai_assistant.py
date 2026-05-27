from typing import Any, Dict, List, Optional
import uuid
import time
import json
import os
import signal
import sys

try:
    from openai import OpenAI, AssistantEventHandler
except ImportError:  # Keeps the module importable for local linting/tests before dependencies are installed.
    OpenAI = None

    class AssistantEventHandler:  # type: ignore[no-redef]
        pass

# Files used by the local development runtime to store assistant and thread IDs.
ID_FILE = os.getenv("ASSISTANT_ID_FILE", "openai-assistants.json")
TOOL_FUNCTIONS_FILE = os.getenv("TOOL_FUNCTIONS_FILE", "tool_functions.json")

ASSISTANT_NAME = "AI Systems Engineering Assistant"
USER = None
TOOLS: List[Dict[str, Any]] = []
client = None

# Global dictionary to hold assistant parameters
ASSISTANT_PARAMS = {
    "name": ASSISTANT_NAME,
    "instructions": """
You are an AI Systems Engineering Assistant designed for production-grade AI engineering work.

Your role is to help engineers design, evaluate, and improve modern AI systems with a focus on:
- RAG pipelines
- Agentic workflows
- Tool/function orchestration
- Grounded generation
- Answer verification
- AI safety guardrails
- Evaluation frameworks
- Observability and tracing
- Latency, cost, reliability, and scalability
- Secure AI integration patterns

You should behave like a senior AI engineering advisor, not a generic chatbot.

Core operating principles:

1. Production engineering mindset
   - Give practical, implementation-oriented answers.
   - Prefer clear architecture, trade-offs, failure modes, and operational concerns.
   - Consider reliability, monitoring, testing, deployment, and maintainability.

2. Grounded and verifiable responses
   - Do not fabricate facts, tool results, logs, metrics, or external system behavior.
   - If information is missing, state the assumption clearly.
   - If the retrieved or provided context is insufficient, say so directly.
   - Prefer evidence-based reasoning over confident guessing.

3. Tool/function calling discipline
   - Use available tools only when they are relevant to the user’s request.
   - Treat tool outputs as system evidence.
   - Never claim a tool was executed unless a tool output confirms it.
   - If a tool cannot be executed by the local runtime, explain what adapter or implementation is required.
   - Never expose secrets, tokens, credentials, or private implementation details.

4. Modern AI engineering focus
   When discussing RAG systems, include:
   - ingestion
   - normalization
   - chunking
   - metadata strategy
   - embeddings
   - hybrid search
   - reranking
   - context assembly
   - citation strategy
   - generation constraints
   - claim verification
   - refusal logic
   - evaluation
   - observability

   When discussing agentic systems, include:
   - tool contracts
   - execution boundaries
   - state management
   - authorization
   - audit trails
   - safety controls
   - deterministic fallbacks
   - human-in-the-loop escalation where needed

5. Security and governance
   - Avoid unsafe tool execution.
   - Do not request or store user credentials.
   - Recommend environment variables, secret managers, least-privilege access, audit logs, and scoped permissions.
   - Call out prompt injection, data leakage, tool misuse, and unsupported generation risks.

6. Answer style
   - Be concise but technically strong.
   - Structure answers clearly.
   - Use engineering language appropriate for a U.S. software engineering portfolio.
   - Avoid hype and unsupported claims.
   - When giving code, make it runnable, organized, and production-aware.

Your goal is to help users design credible, high-quality AI engineering systems that are suitable for real-world implementation and strong enough to showcase in a professional GitHub portfolio.
""",
    "model": os.getenv("OPENAI_ASSISTANT_MODEL", "gpt-4o"),
}


def get_client():
    """Create the OpenAI client lazily so the file remains importable in tests."""
    global client

    if client is not None:
        return client

    if OpenAI is None:
        raise RuntimeError(
            "The OpenAI Python SDK is not installed. Install it with: pip install openai"
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running this script."
        )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client


class EventHandler(AssistantEventHandler):
    def on_event(self, event):
        if event.event == "thread.run.requires_action":
            run_id = event.data.id
            thread_id = getattr(event.data, "thread_id", None)
            self.handle_requires_action(event.data, run_id, thread_id)
        elif event.event.startswith("thread.message"):
            self.handle_message(event)

    def handle_requires_action(self, data, run_id, thread_id=None):
        """
        Handles tool/function calls requested by the Assistant.

        Portfolio-oriented behavior:
        - Parses each tool call safely.
        - Removes legacy provider/auth/gateway handling.
        - Supports local structured AI-engineering tool outputs.
        - Returns an explicit adapter_required response for unknown external tools.
        - Submits all tool outputs together, following the Assistants API pattern.
        """
        tool_outputs = []

        required_action = getattr(data, "required_action", None)
        submit_tool_outputs = getattr(required_action, "submit_tool_outputs", None)
        tool_calls = getattr(submit_tool_outputs, "tool_calls", None) or []

        if not tool_calls:
            print("No tool calls found in required_action.")
            return

        for tool in tool_calls:
            function = getattr(tool, "function", None)
            function_name = getattr(function, "name", "unknown_function")
            raw_arguments = getattr(function, "arguments", "{}") or "{}"

            try:
                function_args = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                tool_outputs.append({
                    "tool_call_id": tool.id,
                    "output": json.dumps({
                        "status": "error",
                        "tool": function_name,
                        "error_type": "invalid_json_arguments",
                        "message": "The tool arguments could not be parsed as valid JSON.",
                        "details": str(exc),
                    }),
                })
                continue

            if not isinstance(function_args, dict):
                tool_outputs.append({
                    "tool_call_id": tool.id,
                    "output": json.dumps({
                        "status": "error",
                        "tool": function_name,
                        "error_type": "invalid_argument_shape",
                        "message": "Tool arguments must be a JSON object.",
                    }),
                })
                continue

            custom_params = function_args.pop("_custom_params", {})

            output_payload = self.execute_ai_engineering_tool(
                function_name=function_name,
                function_args=function_args,
                custom_params=custom_params,
            )

            tool_outputs.append({
                "tool_call_id": tool.id,
                "output": json.dumps(output_payload, ensure_ascii=False),
            })

        if tool_outputs:
            self.submit_tool_outputs(tool_outputs, run_id, thread_id)

    @staticmethod
    def execute_ai_engineering_tool(
        function_name: str,
        function_args: Dict[str, Any],
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Local structured tool runtime for portfolio use.

        This replaces the old dispatcher/gateway/token flow.
        It is intentionally safe:
        - No external API calls.
        - No credential handling.
        - No OAuth/token storage.
        - No hidden provider execution.

        If tool_functions.json defines local AI-engineering tools, this method
        returns useful structured outputs. If a tool is unknown, it returns a clear
        adapter_required response instead of pretending execution happened.
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
                            "Check each claim against retrieved evidence",
                            "Reject unsupported claims",
                            "Return confidence and refusal reasons",
                        ],
                        "observability": [
                            "Log query, retrieved chunks, reranker scores, answer, citations, and verifier results",
                            "Track retrieval recall, faithfulness, citation accuracy, refusal quality, latency, and cost",
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
                        "action": "Refuse, ask for more evidence, or retrieve additional sources.",
                    },
                }

            return {
                "status": "completed",
                "tool": function_name,
                "execution_mode": "local_ai_engineering_runtime",
                "result": {
                    "grounding_status": "evidence_available_not_semantically_verified",
                    "confidence": "medium",
                    "claims_checked": len(claims) if isinstance(claims, list) else 0,
                    "evidence_items": len(evidence) if isinstance(evidence, list) else 1,
                    "note": (
                        "This local verifier confirms that supporting evidence was provided. "
                        "A production verifier should perform claim-level semantic entailment checks "
                        "before marking an answer as fully grounded."
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
                        "Log tool calls and tool outputs for auditability.",
                        "Separate model reasoning from deterministic business logic.",
                        "Add refusal logic for low-confidence or unsupported answers.",
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
                "This portfolio version removed legacy gateway, OAuth, token, and provider-dispatch logic. "
                "To execute this tool against a real external system, implement a dedicated adapter/service layer "
                "with authentication, authorization, validation, observability, and error handling."
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

    def submit_tool_outputs(self, tool_outputs, run_id, thread_id=None):
        if thread_id is None:
            current_run = getattr(self, "current_run", None)
            thread_id = getattr(current_run, "thread_id", None)

        if not thread_id:
            raise RuntimeError("Cannot submit tool outputs because thread_id is missing.")

        with get_client().beta.threads.runs.submit_tool_outputs_stream(
            thread_id=thread_id,
            run_id=run_id,
            tool_outputs=tool_outputs,
            event_handler=EventHandler(),
        ) as stream:
            for _ in stream.text_deltas:
                pass

    @staticmethod
    def handle_message(event):
        message = event.data

        if hasattr(message, "content") and message.content:
            for content_block in message.content:
                if hasattr(content_block, "text") and hasattr(content_block.text, "value"):
                    print(f"\033[91mAI: {content_block.text.value}\033[0m")


def save_assistant_details(name, assistant_id, thread_id):
    assistants = load_all_assistant_details()

    for assistant in assistants:
        if assistant["name"] == name:
            assistant["assistant_id"] = assistant_id
            assistant["thread_id"] = thread_id
            break
    else:
        assistants.append({
            "name": name,
            "assistant_id": assistant_id,
            "thread_id": thread_id,
        })

    with open(ID_FILE, "w", encoding="utf-8") as f:
        json.dump(assistants, f, indent=4)


def load_all_assistant_details():
    if not os.path.exists(ID_FILE):
        return []

    try:
        with open(ID_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: {ID_FILE} contains invalid JSON. Starting with empty assistant registry.")
        return []


def load_assistant_details(name):
    assistants = load_all_assistant_details()

    for assistant in assistants:
        if assistant["name"] == name:
            return assistant["assistant_id"], assistant["thread_id"]

    return None, None


def load_function_tools(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Assistant will run without custom function tools.")
        return []

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_assistant():
    print("Creating an assistant...")
    assistant = get_client().beta.assistants.create(**ASSISTANT_PARAMS)
    print(f"Assistant Created: {assistant.id}")
    return assistant


def update_assistant(ass_id):
    assistant = get_client().beta.assistants.update(ass_id, **ASSISTANT_PARAMS)
    print(f"Assistant Updated: {assistant.id}")
    return assistant


def get_assistant(ass_id):
    assistant = get_client().beta.assistants.retrieve(ass_id)
    return assistant


def delete_assistant(ass_id):
    assistant = get_client().beta.assistants.delete(ass_id)
    return assistant


def create_thread():
    print("Creating a thread...")
    thread = get_client().beta.threads.create()
    print(f"Thread Created: {thread.id}")
    return thread


def send_message(thread_id, content):
    print(f"Sending message to thread {thread_id}...")
    message = get_client().beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )
    return message


def run_stream(thread_id, assistant_id):
    print(f"Running and streaming on thread {thread_id}...")
    with get_client().beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()


def cancel_run(thread_id, run_id):
    try:
        run_status = get_client().beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id,
        ).status

        if run_status not in ["completed", "failed", "canceled"]:
            get_client().beta.threads.runs.cancel(thread_id=thread_id, run_id=run_id)
            print(f"Run {run_id} canceled successfully.")
        else:
            print(f"Run {run_id} is already {run_status} and cannot be canceled.")

    except Exception as e:
        print(f"Failed to cancel run {run_id}: {e}")


def signal_handler(sig, frame):
    print("Signal detected, stopping the script...")
    sys.exit(0)


def run_interactive_session():
    assistant_id, thread_id = load_assistant_details(ASSISTANT_NAME)

    if not assistant_id:
        assistant = create_assistant()
        assistant_id = assistant.id
    else:
        update_assistant(assistant_id)

    if not thread_id:
        thread = create_thread()
        thread_id = thread.id

    save_assistant_details(ASSISTANT_NAME, assistant_id, thread_id)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("Starting interactive session. Type 'exit' or 'stop' to stop.")
    print("Example prompts:")
    print("- Design a grounded RAG architecture for a large document corpus.")
    print("- Review this AI agent architecture for reliability and observability.")
    print("- Create an evaluation plan for a RAG assistant that must avoid unsupported answers.")

    while True:
        user_input = input("User: ")

        if user_input.lower() in ["exit", "stop"]:
            print("Exiting interactive session.")
            time.sleep(1)
            break

        send_message(thread_id, user_input)
        run_stream(thread_id, assistant_id)


if __name__ == "__main__":
    USER = str(uuid.uuid4())
    print(f"USERID: {USER}")

    ASSISTANT_PARAMS["tools"] = load_function_tools(TOOL_FUNCTIONS_FILE)
    run_interactive_session()
