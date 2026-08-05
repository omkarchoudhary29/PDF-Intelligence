import json
from pathlib import Path
import sys
import uuid
from typing import Any, List, Optional

from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.database.base import Base
from backend.database.connection import get_db, set_test_engine
from backend.database.models import Conversation, Document, Message, Workspace
from backend.main import app
from backend.schemas.conversation import CitationItem
from backend.schemas.indexing import ChildChunkRecord
from backend.services.bm25_service import bm25_service
from backend.services.citation_service import citation_service
from backend.services.chunking_service import chunking_service
from backend.services.embedding_service import embedding_service
from backend.services.llm_service import (
    LLMError,
    LLMNotConfiguredError,
    LLMRateLimitError,
    LLMTimeoutError,
    llm_service,
)
from backend.services.rag_workflow import RAGState, rag_workflow_service
from backend.services.vector_index_service import vector_index_service


class MockChatModel(BaseChatModel):
    """Deterministic mock LangChain chat model for testing conversational RAG without external API calls."""

    response_text: str = (
        "**Answer**\n\n"
        "The AX-450 gas turbine operates at 3600 RPM under nominal load [E1]. "
        "Routine maintenance requires quarterly valve inspections [E2]."
    )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.response_text))])

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"


@pytest.fixture
def test_db():
    """Isolated in-memory SQLite database for RAG tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    set_test_engine(engine)

    with session_factory() as session:
        yield session

    set_test_engine(None)
    engine.dispose()


@pytest.fixture
def api_client(test_db):
    """FastAPI TestClient wired to the isolated test database."""
    def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def test_workspace(test_db):
    """Create a sample workspace for RAG testing."""
    ws = Workspace(
        name="RAG Testing Workspace",
        slug=f"rag-test-{uuid.uuid4().hex[:8]}",
        description="Workspace for Phase 6 conversational RAG testing",
    )
    test_db.add(ws)
    test_db.commit()
    test_db.refresh(ws)
    return ws


@pytest.fixture(autouse=True)
def cleanup_llm():
    """Ensure mock LLM is reset after every test."""
    yield
    llm_service.reset()


# =============================================================================
# 1. Database Model & Workspace Boundary Tests
# =============================================================================

def test_conversation_and_message_persistence(test_db, test_workspace):
    """Test conversation creation, cascade message storage, and timestamps."""
    conv = Conversation(
        workspace_id=test_workspace.id,
        title="Turbine Maintenance Chat",
    )
    test_db.add(conv)
    test_db.commit()
    test_db.refresh(conv)

    assert conv.id is not None
    assert conv.workspace_id == test_workspace.id
    assert conv.is_active is True

    # Add user message
    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content="What is the operating speed of the AX-450?",
    )
    test_db.add(user_msg)

    # Add assistant message with citations
    citations_data = [
        {
            "citation_id": "E1",
            "document_id": "doc123",
            "filename": "manual.pdf",
            "page_start": 42,
            "page_end": 43,
            "section": "Operating Specs",
            "excerpt": "Speed is 3600 RPM",
        }
    ]
    asst_msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content="The AX-450 operates at 3600 RPM [E1].",
        citations=citations_data,
    )
    test_db.add(asst_msg)
    test_db.commit()

    # Query back via relationships
    loaded_conv = test_db.query(Conversation).filter(Conversation.id == conv.id).first()
    assert len(loaded_conv.messages) == 2
    assert loaded_conv.messages[0].role == "user"
    assert loaded_conv.messages[1].role == "assistant"
    assert loaded_conv.messages[1].citations[0]["citation_id"] == "E1"


def test_workspace_scoped_conversation_isolation(api_client, test_db, test_workspace):
    """Verify that conversations in workspace A cannot be accessed from workspace B."""
    # Create conversation in workspace A
    conv = Conversation(workspace_id=test_workspace.id, title="Secret Conv")
    test_db.add(conv)
    test_db.commit()

    # Attempt to retrieve via a different workspace UUID
    other_ws_id = str(uuid.uuid4())
    resp = api_client.get(f"/api/v1/workspaces/{other_ws_id}/conversations/{conv.id}")
    assert resp.status_code == 404

    # Verify listing conversations filters strictly by workspace
    list_resp = api_client.get(f"/api/v1/workspaces/{test_workspace.id}/conversations")
    assert list_resp.status_code == 200
    assert list_resp.json()["total"] == 1


# =============================================================================
# 2. Query Understanding & Contextualization Tests
# =============================================================================

def test_query_contextualization_followup_resolution():
    """Verify follow-up queries with pronouns resolve previous entity references."""
    history = [
        {"role": "user", "content": "Tell me about the AX-450 gas turbine specifications."},
        {"role": "assistant", "content": "The AX-450 is a heavy-duty industrial gas turbine."},
    ]

    # Follow-up with pronoun reference
    state: RAGState = {
        "workspace_id": "ws1",
        "conversation_id": "c1",
        "original_query": "What about its maintenance requirements?",
        "conversation_history": history,
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
    }

    res = rag_workflow_service.query_contextualization_node(state)
    contextualized = res["contextualized_query"]

    assert "AX-450" in contextualized
    # User's original query preserved
    assert state["original_query"] == "What about its maintenance requirements?"


def test_query_classification_intents():
    """Verify query categorization across comparison, summary, analytical, and factual."""
    test_cases = [
        ("Compare the AX-450 with the P-99 pump", "comparison"),
        ("Summarize the entire safety procedure", "summary"),
        ("Why did the pressure drop during startup?", "analytical"),
        ("What is the nominal rotor speed?", "factual"),
    ]

    for query, expected_category in test_cases:
        state: RAGState = {
            "workspace_id": "ws1",
            "conversation_id": "c1",
            "original_query": query,
            "conversation_history": [],
            "contextualized_query": query,
            "query_category": "",
            "retrieved_evidence": [],
            "evidence_map": {},
            "evidence_sufficient": True,
            "assembled_context": "",
            "generated_answer": "",
            "validated_citations": [],
            "validation_issues": [],
            "error": None,
        }
        res = rag_workflow_service.query_classification_node(state)
        assert res["query_category"] == expected_category


# =============================================================================
# 3. Citation Validation Service Tests
# =============================================================================

def test_citation_extraction_and_mapping():
    """Verify valid citation tags map to real metadata and hallucinated IDs are purged."""
    evidence_map = {
        "E1": {
            "document_id": "doc_1",
            "original_filename": "turbine_manual.pdf",
            "page_start": 42,
            "page_end": 43,
            "section_title": "Specifications",
            "text": "Nominal rotational velocity is 3600 RPM.",
        },
        "E2": {
            "document_id": "doc_2",
            "original_filename": "safety_policy.pdf",
            "page_start": 18,
            "page_end": 18,
            "section_title": "PPE Requirements",
            "text": "Safety goggles must be worn in testing zones.",
        },
    }

    # Model cited valid E1, valid E2, and hallucinated E99
    model_answer = (
        "The turbine operates at 3600 RPM [E1]. "
        "All technicians must wear goggles [E2]. "
        "Also it has laser sensors [E99]."
    )

    valid_citations, issues = citation_service.validate_citations(model_answer, evidence_map)

    assert len(valid_citations) == 2
    assert valid_citations[0].citation_id == "E1"
    assert valid_citations[0].filename == "turbine_manual.pdf"
    assert valid_citations[0].page_start == 42
    assert valid_citations[1].citation_id == "E2"
    assert valid_citations[1].filename == "safety_policy.pdf"

    # Verify hallucinated E99 recorded in issues
    assert len(issues) == 1
    assert "E99" in issues[0]


# =============================================================================
# 4. Context Assembly & Prompt-Injection Resistance
# =============================================================================

def test_context_assembly_stable_tagging():
    """Verify context assembly attaches stable [E1], [E2] tags with parent context."""
    state: RAGState = {
        "workspace_id": "ws1",
        "conversation_id": "c1",
        "original_query": "Test query",
        "conversation_history": [],
        "contextualized_query": "Test query",
        "query_category": "factual",
        "retrieved_evidence": [
            {
                "chunk_id": "c1",
                "document_id": "doc1",
                "original_filename": "manual.pdf",
                "page_start": 5,
                "page_end": 5,
                "section_title": "Overview",
                "text": "Primary system overview text.",
                "parent_text": "Broader contextual parent section.",
            }
        ],
        "evidence_map": {},
        "evidence_sufficient": True,
        "assembled_context": "",
        "generated_answer": "",
        "validated_citations": [],
        "validation_issues": [],
        "error": None,
    }

    res = rag_workflow_service.context_assembly_node(state)
    assembled = res["assembled_context"]
    ev_map = res["evidence_map"]

    assert "[E1]" in assembled
    assert "manual.pdf" in assembled
    assert "Pages: 5-5" in assembled
    assert "Primary system overview text." in assembled
    assert "Broader contextual parent section." in assembled
    assert "E1" in ev_map


def test_evidence_evaluation_insufficient_handling():
    """Verify empty evidence results trigger honest uncertainty without hallucination."""
    state: RAGState = {
        "workspace_id": "ws1",
        "conversation_id": "c1",
        "original_query": "What is the flux capacitor capacity?",
        "conversation_history": [],
        "contextualized_query": "What is the flux capacitor capacity?",
        "query_category": "factual",
        "retrieved_evidence": [],
        "evidence_map": {},
        "evidence_sufficient": True,
        "assembled_context": "",
        "generated_answer": "",
        "validated_citations": [],
        "validation_issues": [],
        "error": None,
    }

    res = rag_workflow_service.evidence_evaluation_node(state)
    assert res["evidence_sufficient"] is False
    assert "could not find sufficient evidence" in res["generated_answer"]


# =============================================================================
# 5. LLM Service Configuration & Error Handling Tests
# =============================================================================

def test_missing_llm_credentials_handled_gracefully(monkeypatch):
    """Verify missing GEMINI_API_KEY raises LLMNotConfiguredError and does not crash app."""
    monkeypatch.setattr(llm_service.settings, "gemini_api_key", None)
    monkeypatch.setattr(llm_service.settings, "llm_api_key", None)
    llm_service.reset()

    assert llm_service.is_configured is False
    with pytest.raises(LLMNotConfiguredError) as exc_info:
        llm_service.get_llm()
    assert "LLM provider is not configured" in str(exc_info.value)


def test_llm_mock_invocation():
    """Verify mock LLM injection returns expected text."""
    mock = MockChatModel(response_text="Mocked Grounded Answer [E1].")
    llm_service.set_mock_model(mock)

    answer = llm_service.invoke([])
    assert answer == "Mocked Grounded Answer [E1]."


# =============================================================================
# 6. End-to-End Conversational RAG API Tests
# =============================================================================

def test_conversational_rag_api_flow(api_client, test_workspace, test_db, tmp_path, monkeypatch):
    """Full integration test: Create conversation -> send query -> receive grounded answer with citations."""
    monkeypatch.setattr(bm25_service, "storage_root", tmp_path)
    monkeypatch.setattr(bm25_service, "processed_root", tmp_path / "processed")
    monkeypatch.setattr(chunking_service, "storage_root", tmp_path)
    monkeypatch.setattr(chunking_service, "processed_root", tmp_path / "processed")
    bm25_service.invalidate_cache()

    # Inject mock LLM
    mock_answer = (
        "**Answer**\n\n"
        "The AX-450 gas turbine operates at 3600 RPM under nominal load conditions [E1].\n\n"
        "**Sources**\n"
        "1. turbine_specs.pdf — Page 1"
    )
    llm_service.set_mock_model(MockChatModel(response_text=mock_answer))

    # Index sample child chunk into BM25
    doc_id = f"doc_{uuid.uuid4().hex[:6]}"
    chunk = ChildChunkRecord(
        chunk_id=f"{doc_id}_c1",
        parent_id=f"{doc_id}_p1",
        document_id=doc_id,
        workspace_id=test_workspace.id,
        original_filename="turbine_specs.pdf",
        page_start=1,
        page_end=1,
        section_title="Operating Specifications",
        text="The AX-450 gas turbine operates at 3600 RPM under nominal load conditions.",
        chunk_type="text",
        token_count=14,
        file_hash="hash_rag",
        embedding_model="test-model",
        document_version=1,
    )
    bm25_service.add_or_update_document_chunks(test_workspace.id, doc_id, [chunk])

    # 1. Create conversation via API
    create_resp = api_client.post(
        f"/api/v1/workspaces/{test_workspace.id}/conversations",
        json={"title": "Turbine Inquiries"},
    )
    assert create_resp.status_code == 201
    conv_data = create_resp.json()
    conv_id = conv_data["id"]
    assert conv_data["title"] == "Turbine Inquiries"

    # 2. Send question to conversation
    msg_resp = api_client.post(
        f"/api/v1/workspaces/{test_workspace.id}/conversations/{conv_id}/messages",
        json={"content": "What is the operating speed of the AX-450 turbine?"},
    )
    assert msg_resp.status_code == 200
    rag_data = msg_resp.json()

    assert rag_data["conversation_id"] == conv_id
    assert rag_data["evidence_sufficient"] is True
    assert "3600 RPM" in rag_data["answer"]
    assert len(rag_data["citations"]) >= 1

    citation = rag_data["citations"][0]
    assert citation["citation_id"] == "E1"
    assert citation["filename"] == "turbine_specs.pdf"
    assert citation["page_start"] == 1
    assert "3600 RPM" in citation["excerpt"]

    # 3. Retrieve conversation messages list
    list_resp = api_client.get(f"/api/v1/workspaces/{test_workspace.id}/conversations/{conv_id}/messages")
    assert list_resp.status_code == 200
    messages = list_resp.json()["items"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert len(messages[1]["citations"]) >= 1


def test_send_message_unconfigured_llm(api_client, test_workspace, tmp_path, monkeypatch):
    """Verify attempting to send a message when LLM is unconfigured returns honest response."""
    monkeypatch.setattr(bm25_service, "storage_root", tmp_path)
    monkeypatch.setattr(bm25_service, "processed_root", tmp_path / "processed")
    bm25_service.invalidate_cache()
    monkeypatch.setattr(llm_service.settings, "gemini_api_key", None)
    monkeypatch.setattr(llm_service.settings, "llm_api_key", None)
    llm_service.reset()

    # Index sample chunk so evidence is retrieved
    doc_id = f"doc_{uuid.uuid4().hex[:6]}"
    chunk = ChildChunkRecord(
        chunk_id=f"{doc_id}_c1",
        parent_id=f"{doc_id}_p1",
        document_id=doc_id,
        workspace_id=test_workspace.id,
        original_filename="sample.pdf",
        page_start=1,
        page_end=1,
        section_title="Sample",
        text="Sample text content for unconfigured LLM test",
        chunk_type="text",
        token_count=10,
        file_hash="hash_unc",
        embedding_model="test-model",
        document_version=1,
    )
    bm25_service.add_or_update_document_chunks(test_workspace.id, doc_id, [chunk])

    # Create conversation
    create_resp = api_client.post(
        f"/api/v1/workspaces/{test_workspace.id}/conversations",
        json={"title": "Test Unconfigured"},
    )
    conv_id = create_resp.json()["id"]

    resp = api_client.post(
        f"/api/v1/workspaces/{test_workspace.id}/conversations/{conv_id}/messages",
        json={"content": "Sample text content query"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "LLM provider is not configured" in data["answer"]


def test_send_message_empty_content_validation(api_client, test_workspace):
    """POST /messages with empty content returns 422."""
    create_resp = api_client.post(
        f"/api/v1/workspaces/{test_workspace.id}/conversations",
        json={"title": "Validation Test"},
    )
    conv_id = create_resp.json()["id"]

    resp = api_client.post(
        f"/api/v1/workspaces/{test_workspace.id}/conversations/{conv_id}/messages",
        json={"content": "   "},
    )
    assert resp.status_code == 422
