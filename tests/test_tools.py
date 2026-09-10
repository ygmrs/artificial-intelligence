from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from ai_assistant import execute_ai_engineering_tool


def test_design_rag_pipeline_returns_completed_result():
    result = execute_ai_engineering_tool(
        function_name="design_rag_pipeline",
        function_args={
            "use_case": "enterprise document search",
        },
    )

    assert result["status"] == "completed"
    assert result["tool"] == "design_rag_pipeline"
    assert result["execution_mode"] == "local_ai_engineering_runtime"

    architecture = result["result"]["architecture"]

    assert "ingestion" in architecture
    assert "indexing" in architecture
    assert "retrieval" in architecture
    assert "generation" in architecture
    assert "verification" in architecture
    assert "observability" in architecture


def test_unknown_tool_returns_adapter_required():
    result = execute_ai_engineering_tool(
        function_name="external_provider_action",
        function_args={
            "resource": "example",
        },
    )

    assert result["status"] == "adapter_required"
    assert result["tool"] == "external_provider_action"
    assert result["execution_mode"] == "safe_local_fallback"

    assert result["recommended_adapter_contract"]["validate_input"] is True
    assert result["recommended_adapter_contract"]["enforce_authorization"] is True


def test_grounding_verification_rejects_missing_evidence():
    result = execute_ai_engineering_tool(
        function_name="verify_grounded_answer",
        function_args={
            "answer": "The system uses PostgreSQL.",
            "claims": [
                "The system uses PostgreSQL.",
            ],
            "evidence": [],
        },
    )

    assert result["status"] == "completed"

    verification = result["result"]

    assert verification["grounded"] is False
    assert verification["confidence"] == "low"
    assert verification["reason"] == "No supporting evidence was provided."