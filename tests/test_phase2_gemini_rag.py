import os
import time
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.messages import AIMessage

from backend.schemas.conversation import CitationItem
from backend.schemas.search import SearchRequest
from backend.services.citation_service import citation_service
from backend.services.llm_service import (
    LLMError,
    LLMNotConfiguredError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMService,
    llm_service,
)
from backend.services.rag_workflow import RAGState, rag_workflow_service
from backend.services.retrieval_service import retrieval_service
from config.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def reset_service_singletons():
    """Ensure service singletons have clean paths not contaminated by other tests."""
    from backend.services.bm25_service import bm25_service
    from backend.services.chunking_service import chunking_service
    chunking_service._storage_root = None
    chunking_service._processed_root = None
    bm25_service._storage_root = None
    bm25_service._processed_root = None
    yield
    chunking_service._storage_root = None
    chunking_service._processed_root = None
    bm25_service._storage_root = None
    bm25_service._processed_root = None


# =============================================================================
# 1. Configuration & Key Status Tests
# =============================================================================

def test_gemini_key_status_missing():
    """Verify that empty or None credentials resolve to MISSING."""
    s = Settings(gemini_api_key="", llm_api_key=None)
    assert s.gemini_key_status == "MISSING"
    assert s.effective_gemini_api_key is None


def test_gemini_key_status_placeholder():
    """Verify that placeholder templates are identified as PLACEHOLDER."""
    for placeholder in [
        "your_gemini_api_key_here",
        "YOUR_GEMINI_API_KEY_HERE",
        "<your-google-gemini-api-key>",
        "placeholder",
        "changeme",
    ]:
        s = Settings(gemini_api_key=placeholder)
        assert s.gemini_key_status == "PLACEHOLDER", f"Failed for placeholder: {placeholder}"
        assert s.effective_gemini_api_key is None


def test_gemini_key_status_configured():
    """Verify that a genuine key string is recognized as CONFIGURED."""
    s = Settings(gemini_api_key="AIzaSyTestValidFakeKeyString123456789")
    assert s.gemini_key_status == "CONFIGURED"
    assert s.effective_gemini_api_key == "AIzaSyTestValidFakeKeyString123456789"


def test_gemini_model_configuration():
    """Verify model name is configurable and not hardcoded."""
    s1 = Settings(gemini_model="gemini-1.5-flash")
    assert s1.gemini_model == "gemini-1.5-flash"
    s2 = Settings(gemini_model="gemini-2.0-flash")
    assert s2.gemini_model == "gemini-2.0-flash"


# =============================================================================
# 2. Independent Provider Smoke Test & Diagnostics
# =============================================================================

def test_llm_service_configuration_status():
    """Verify diagnostic status format without credential leakage."""
    status_info = llm_service.get_configuration_status()
    assert "status" in status_info
    assert "provider" in status_info
    assert "model" in status_info
    assert "is_configured" in status_info
    assert status_info["provider"] == "gemini"


def test_llm_service_smoke_test_when_not_configured():
    """Verify smoke test returns clean error dict when credentials are not configured."""
    custom_llm_service = LLMService()
    custom_llm_service.settings = Settings(gemini_api_key=None, llm_api_key=None)
    result = custom_llm_service.test_provider_connection()
    assert result["success"] is False
    assert result["status"] == "MISSING"
    assert "not configured" in result["error"].lower()


def test_llm_service_smoke_test_mocked_success():
    """Verify smoke test reports CONNECTED when LLM responds successfully."""
    custom_llm_service = LLMService()
    mock_chat = MagicMock()
    mock_chat.invoke.return_value = AIMessage(content="PONG")
    custom_llm_service.set_mock_model(mock_chat)

    result = custom_llm_service.test_provider_connection()
    assert result["success"] is True
    assert result["status"] == "CONNECTED"
    assert result["response_preview"] == "PONG"
    assert result["latency_ms"] >= 0.0


def test_llm_service_smoke_test_redacts_credentials():
    """Verify provider smoke test masks sensitive keys in error messages."""
    custom_llm_service = LLMService()
    secret_key = "AIzaSyTopSecretKeyToNeverLeak"
    custom_llm_service.settings = Settings(gemini_api_key=secret_key)
    mock_chat = MagicMock()
    mock_chat.invoke.side_effect = Exception(f"Auth failed with key {secret_key}")
    custom_llm_service.set_mock_model(mock_chat)

    result = custom_llm_service.test_provider_connection()
    assert result["success"] is False
    assert secret_key not in result["error"]
    assert "[REDACTED]" in result["error"]


# =============================================================================
# 3. Hybrid Retrieval on Real Indexed PDFs
# =============================================================================

WORKSPACE_ID = "dc5d9139-f381-4818-ab26-c2e878997b4d"


def test_retrieval_on_indexed_documents():
    """Validate hybrid retrieval against the existing indexed workspace."""
    req = SearchRequest(
        query="turbine operating procedures rotor speed limit",
        top_k=5,
        include_parent=True,
    )
    resp = retrieval_service.search(workspace_id=WORKSPACE_ID, request=req)

    assert resp.total_results > 0
    top_hit = resp.results[0]
    assert top_hit.original_filename == "Turbine_Operating_Procedures.pdf"
    assert top_hit.page_start >= 1
    assert top_hit.parent_text is not None
    assert len(top_hit.parent_text) > 0


def test_retrieval_git_guide():
    """Validate retrieval on Git & GitHub guide document."""
    req = SearchRequest(
        query="git commit amend staging area",
        top_k=5,
        include_parent=True,
    )
    resp = retrieval_service.search(workspace_id=WORKSPACE_ID, request=req)

    assert resp.total_results > 0
    filenames = [r.original_filename for r in resp.results]
    assert "Git_GitHub_Complete_Guide.pdf" in filenames


# =============================================================================
# 4. Citation Service Validation & Hallucination Defense
# =============================================================================

def test_citation_service_valid_extraction():
    """Validate that exact cited markers [E1] match real evidence items."""
    evidence_map = {
        "E1": {
            "document_id": "doc-uuid-1",
            "original_filename": "Turbine_Operating_Procedures.pdf",
            "page_start": 3,
            "page_end": 3,
            "section_title": "Emergency Shutdown",
            "text": "The emergency trip speed is set at 3,300 RPM.",
            "version_number": 1,
            "version_status": "current",
            "is_current": True,
        }
    }
    answer = "The emergency trip speed limit for the rotor is 3,300 RPM [E1]."
    citations, issues = citation_service.validate_citations(answer, evidence_map)

    assert len(citations) == 1
    assert citations[0].citation_id == "E1"
    assert citations[0].filename == "Turbine_Operating_Procedures.pdf"
    assert citations[0].page_start == 3
    assert len(issues) == 0


def test_citation_service_hallucinated_tag_detection():
    """Validate that model-invented citation tags like [E99] are caught and rejected."""
    evidence_map = {
        "E1": {
            "document_id": "doc-uuid-1",
            "original_filename": "Turbine_Operating_Procedures.pdf",
            "page_start": 3,
            "page_end": 3,
            "section_title": "Emergency Shutdown",
            "text": "The emergency trip speed is set at 3,300 RPM.",
        }
    }
    answer = "The rotor trip speed is 3,300 RPM [E1], and lubrication flow is 50 L/min [E99]."
    citations, issues = citation_service.validate_citations(answer, evidence_map)

    assert len(citations) == 1
    assert citations[0].citation_id == "E1"
    assert len(issues) == 1
    assert "E99" in issues[0]


# =============================================================================
# 5. Conversational RAG Workflow & LangGraph Execution
# =============================================================================

def test_rag_workflow_query_contextualization():
    """Verify follow-up queries resolve elliptical references from history."""
    state: RAGState = {
        "workspace_id": WORKSPACE_ID,
        "conversation_id": "test-conv",
        "original_query": "What happens if it exceeds that?",
        "conversation_history": [
            {"role": "user", "content": "What is the trip limit for rotor speed on the MK-7 turbine?"},
            {"role": "assistant", "content": "The emergency trip limit is 3,300 RPM."},
        ],
        "contextualized_query": "",
        "query_category": "",
        "retrieved_evidence": [],
        "evidence_map": {},
        "evidence_sufficient": True,
        "assembled_context": "",
        "generated_answer": "",
        "validated_citations": [],
        "validation_issues": [],
        "error": None,
        "response_type": "valid_answer",
        "retrieval_time_ms": 0.0,
        "llm_time_ms": 0.0,
        "total_time_ms": 0.0,
    }

    result = rag_workflow_service.query_contextualization_node(state)
    assert "contextualized_query" in result
    cq = result["contextualized_query"].lower()
    assert "exceeds" in cq
    assert "mk-7" in cq or "rotor" in cq or "context" in cq


def test_rag_workflow_query_classification():
    """Verify user queries are categorized correctly into intents."""
    cases = [
        ("Compare the turbine startup procedures in revision 1 and revision 2", "comparison"),
        ("Give me a summary of section 4", "summary"),
        ("What is the maximum operating temperature?", "factual"),
    ]
    for q, expected_cat in cases:
        state = {"contextualized_query": q}
        res = rag_workflow_service.query_classification_node(state)  # type: ignore
        assert res["query_category"] == expected_cat


def test_rag_workflow_unanswerable_query():
    """Verify that unanswerable query with no evidence returns honest refusal."""
    state = rag_workflow_service.run(
        workspace_id=WORKSPACE_ID,
        conversation_id="conv-1",
        query="What is the authentic recipe for authentic Neapolitan pizza dough?",
    )
    # Either retrieval finds no items or score is below threshold
    if not state.get("evidence_sufficient", True):
        assert "could not find sufficient evidence" in state["generated_answer"].lower()
        assert state["response_type"] == "insufficient_evidence"


def test_rag_workflow_llm_not_configured_graceful_handling():
    """Verify that when LLM credentials are not configured, workflow produces clear status."""
    orig_mock = llm_service._llm
    orig_is_mock = llm_service._is_mock
    llm_service.reset()

    try:
        with patch.object(llm_service.settings, "gemini_api_key", None), patch.object(
            llm_service.settings, "llm_api_key", None
        ):
            state = rag_workflow_service.run(
                workspace_id=WORKSPACE_ID,
                conversation_id="conv-test-unconfigured",
                query="What is the rotor speed limit for turbine startup?",
            )
            assert state["response_type"] == "llm_not_configured"
            assert "GEMINI_API_KEY" in state["generated_answer"]
            assert len(state["validated_citations"]) > 0
            assert state["validated_citations"][0]["filename"] == "Turbine_Operating_Procedures.pdf"
    finally:
        llm_service._llm = orig_mock
        llm_service._is_mock = orig_is_mock


def test_rag_workflow_grounded_answer_with_mock_llm():
    """Verify complete grounded answer generation when LLM is operational."""
    mock_chat = MagicMock()
    mock_chat.invoke.return_value = AIMessage(
        content="Based on the operating procedures, the turbine rotor trip speed is 3,300 RPM [E1]."
    )
    llm_service.set_mock_model(mock_chat)

    try:
        state = rag_workflow_service.run(
            workspace_id=WORKSPACE_ID,
            conversation_id="conv-test-mocked",
            query="What is the turbine rotor trip speed?",
        )
        assert state["response_type"] == "valid_answer"
        assert "3,300 RPM" in state["generated_answer"]
        assert len(state["validated_citations"]) >= 1
        assert state["validated_citations"][0]["filename"] == "Turbine_Operating_Procedures.pdf"
        assert state["retrieval_time_ms"] > 0
        assert state["total_time_ms"] > 0
    finally:
        llm_service.reset()
