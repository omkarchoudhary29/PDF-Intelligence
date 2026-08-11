"""Unit and integration test suite for Phase 8 Knowledge Graph & Relationship Discovery.

Covers:
1. Entity normalization, type validation, and alias detection
2. Deterministic and LLM entity extraction
3. Document-entity association grounding and page references
4. Shared entity relationship discovery
5. Direct reference relationship discovery
6. Semantic topic relationship discovery via embeddings
7. Document connectivity score calculation
8. NetworkX graph assembly, node/edge attributes, and serialization
9. Document neighborhood (ego graph) querying
10. Graph visualization (Matplotlib PNG rendering and empty-state canvas)
11. Incremental graph update and deletion cascading
12. Workspace boundary isolation and FastAPI REST endpoints
"""

import io
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
import fitz
import networkx as nx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.database.base import Base
from backend.database.connection import get_db, set_test_engine
from backend.database.models import (
    Document,
    DocumentEntityAssociation,
    DocumentRelationship,
    Entity,
    Workspace,
)
from backend.main import app
from backend.services.entity_extraction_service import (
    EntityExtractionService,
    entity_extraction_service,
)
from backend.services.entity_normalization_service import (
    EntityNormalizationService,
    entity_normalization_service,
)
from backend.services.graph_visualization_service import (
    GraphVisualizationService,
    graph_visualization_service,
)
from backend.services.knowledge_graph_service import (
    KnowledgeGraphService,
    knowledge_graph_service,
)
from backend.services.relationship_discovery_service import (
    RelationshipDiscoveryService,
    relationship_discovery_service,
)


@pytest.fixture
def test_db():
    """Isolated in-memory SQLite database for graph tests."""
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
    """FastAPI TestClient wired to isolated in-memory test database."""
    def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_workspace(test_db):
    """Create test workspace."""
    ws = Workspace(
        name="Turbine Engineering Knowledge",
        slug="turbine-engineering-kg",
        description="Workspace for testing cross-document relationships",
        is_active=True,
    )
    test_db.add(ws)
    test_db.commit()
    test_db.refresh(ws)
    return ws


@pytest.fixture
def sample_documents(test_db, sample_workspace, tmp_path, monkeypatch):
    """Create two related test documents with extracted pages."""
    from backend.services.storage_service import storage_service
    storage_root = tmp_path / "storage"
    monkeypatch.setattr(storage_service, "storage_root", storage_root)

    # Document 1: Maintenance Manual
    doc1 = Document(
        workspace_id=sample_workspace.id,
        original_filename="Turbine_Manual_AX450.pdf",
        display_name="Turbine Manual AX-450",
        storage_key=f"workspaces/{sample_workspace.id}/documents/Turbine_Manual_AX450.pdf",
        file_size_bytes=1024 * 50,
        sha256_hash="hash_doc_1111",
        mime_type="application/pdf",
        processing_status="completed",
        total_pages=3,
    )
    test_db.add(doc1)

    # Document 2: Safety Policy
    doc2 = Document(
        workspace_id=sample_workspace.id,
        original_filename="Safety_Policy_2024.pdf",
        display_name="Safety Policy 2024",
        storage_key=f"workspaces/{sample_workspace.id}/documents/Safety_Policy_2024.pdf",
        file_size_bytes=1024 * 30,
        sha256_hash="hash_doc_2222",
        mime_type="application/pdf",
        processing_status="completed",
        total_pages=2,
    )
    test_db.add(doc2)
    test_db.commit()
    test_db.refresh(doc1)
    test_db.refresh(doc2)

    return doc1, doc2


# =============================================================================
# 1. Entity Normalization & Deduplication Tests
# =============================================================================

def test_entity_normalization_rules():
    """Verify case-folding, whitespace collapsing, and punctuation stripping."""
    norm = entity_normalization_service

    assert norm.normalize_name("  AX-450  ") == "ax-450"
    assert norm.normalize_name('"Gas Turbine Unit 4"') == "gas turbine unit 4"
    assert norm.normalize_name("ISO-9001:2015.") == "iso-9001:2015"
    assert norm.clean_canonical_name("  Turbine Unit 4,  ") == "Turbine Unit 4"


def test_entity_type_validation_and_aliases():
    """Verify mapping of raw entity types and aliases to standard categories."""
    norm = entity_normalization_service

    assert norm.validate_and_normalize_type("machine") == "Machine"
    assert norm.validate_and_normalize_type("device") == "Equipment"
    assert norm.validate_and_normalize_type("guideline") == "Policy"
    assert norm.validate_and_normalize_type("spec") == "Standard"
    assert norm.validate_and_normalize_type("unknown_nonexistent") == "Other"


def test_alias_detection():
    """Verify alias equivalence between hyphenated and spaced variants."""
    norm = entity_normalization_service

    assert norm.are_aliases("Unit-4", "Unit 4") is True
    assert norm.are_aliases("AX-450", "ax450") is True
    assert norm.are_aliases("Turbine", "Boiler") is False


# =============================================================================
# 2. Entity Extraction Tests
# =============================================================================

def test_deterministic_technical_regex_extraction():
    """Verify deterministic extraction of technical codes, regulations, and units."""
    page_data = {
        "page_number": 1,
        "raw_text": (
            "The AX-450 gas turbine is manufactured under ISO-9001 standards. "
            "Refer to Turbine Unit 4 maintenance procedures in Manual_AX450.pdf."
        ),
        "headings": [{"text": "1.0 Specifications"}],
    }

    entities = entity_extraction_service.extract_deterministic_entities(page_data)
    names = [e.name for e in entities]

    assert any("AX-450" in n for n in names)
    assert any("ISO-9001" in n for n in names)
    assert any("Turbine Unit 4" in n for n in names)


def test_extract_and_store_document_entities(test_db, sample_workspace, sample_documents):
    """Verify database persistence of Entity records and DocumentEntityAssociation."""
    doc1, _ = sample_documents
    pages = [
        {
            "page_number": 1,
            "raw_text": "Turbine AX-450 operating temperature must comply with ISO-9001 guidelines.",
            "headings": [{"text": "1.0 Operating Conditions"}],
        }
    ]

    assocs = entity_extraction_service.extract_and_store_document_entities(
        db=test_db,
        workspace_id=sample_workspace.id,
        document=doc1,
        pages_records=pages,
        use_llm=False,  # Test deterministic path
    )

    assert len(assocs) >= 1
    # Verify entity stored in DB
    entity = test_db.query(Entity).filter(Entity.normalized_name == "ax-450").first()
    assert entity is not None
    assert entity.workspace_id == sample_workspace.id

    # Verify association metadata
    assoc = assocs[0]
    assert assoc.document_id == doc1.id
    assert assoc.page_start == 1
    assert "AX-450" in assoc.excerpt


# =============================================================================
# 3. Relationship Discovery Tests
# =============================================================================

def test_shared_entity_relationship_discovery(test_db, sample_workspace, sample_documents):
    """Verify discovery of 'shared_entity' relationships when two documents mention same entity."""
    doc1, doc2 = sample_documents

    # Create shared entity in both documents
    shared_ent = Entity(
        workspace_id=sample_workspace.id,
        canonical_name="AX-450",
        normalized_name="ax-450",
        entity_type="Equipment",
    )
    test_db.add(shared_ent)
    test_db.flush()

    assoc1 = DocumentEntityAssociation(
        workspace_id=sample_workspace.id,
        document_id=doc1.id,
        entity_id=shared_ent.id,
        page_start=2,
        excerpt="The AX-450 turbine requires quarterly lube checks.",
    )
    assoc2 = DocumentEntityAssociation(
        workspace_id=sample_workspace.id,
        document_id=doc2.id,
        entity_id=shared_ent.id,
        page_start=1,
        excerpt="All personnel near AX-450 must wear hearing protection.",
    )
    test_db.add_all([assoc1, assoc2])
    test_db.commit()

    rels = relationship_discovery_service.discover_shared_entity_relationships(
        db=test_db,
        workspace_id=sample_workspace.id,
    )

    assert len(rels) >= 1
    rel = rels[0]
    assert rel.relationship_type == "shared_entity"
    assert rel.confidence_score == 0.90
    assert "AX-450" in rel.explanation
    assert len(rel.supporting_evidence) == 2


def test_direct_reference_relationship_discovery(test_db, sample_workspace, sample_documents):
    """Verify discovery of 'direct_reference' when doc1 explicitly names doc2."""
    doc1, doc2 = sample_documents

    doc1_pages = [
        {
            "page_number": 2,
            "raw_text": f"For fire emergency protocols, refer directly to {doc2.original_filename} on page 4.",
            "headings": [],
        }
    ]

    rels = relationship_discovery_service.discover_direct_reference_relationships(
        db=test_db,
        workspace_id=sample_workspace.id,
        source_doc=doc1,
        pages_records=doc1_pages,
        all_workspace_docs=[doc1, doc2],
    )

    assert len(rels) == 1
    rel = rels[0]
    assert rel.relationship_type == "direct_reference"
    assert rel.source_document_id == doc1.id
    assert rel.target_document_id == doc2.id
    assert rel.confidence_score == 1.0
    assert rel.source_page_start == 2


def test_semantic_topic_relationship_discovery(test_db, sample_workspace, sample_documents):
    """Verify semantic topic overlap discovery using embedding cosine similarity."""
    doc1, doc2 = sample_documents

    text1 = "Gas turbine operation, electrical generator output, turbine blades, rotating machinery."
    text2 = "Turbine rotor maintenance, generator electrical windings, rotating blade inspection."

    rels = relationship_discovery_service.discover_related_topics_via_embeddings(
        db=test_db,
        workspace_id=sample_workspace.id,
        source_doc=doc1,
        source_text_summary=text1,
        other_docs=[doc1, doc2],
        other_texts={doc2.id: text2},
        threshold=0.60,
    )

    assert len(rels) >= 1
    rel = rels[0]
    assert rel.relationship_type == "related_topic"
    assert rel.confidence_score >= 0.60
    assert "cosine similarity" in rel.explanation.lower()


def test_document_connectivity_score_calculation(test_db, sample_workspace, sample_documents):
    """Verify explainable document connectivity score calculation."""
    doc1, doc2 = sample_documents

    # Add a direct reference and a shared entity relationship
    rel1 = DocumentRelationship(
        workspace_id=sample_workspace.id,
        source_document_id=doc1.id,
        target_document_id=doc2.id,
        relationship_type="direct_reference",
        confidence_score=1.0,
        explanation="Direct citation",
    )
    rel2 = DocumentRelationship(
        workspace_id=sample_workspace.id,
        source_document_id=doc1.id,
        target_document_id=doc2.id,
        relationship_type="shared_entity",
        confidence_score=0.9,
        explanation="Common entity",
    )
    test_db.add_all([rel1, rel2])
    test_db.commit()

    score = relationship_discovery_service.calculate_document_connectivity_score(
        db=test_db,
        workspace_id=sample_workspace.id,
        document_id=doc1.id,
    )
    # 3.0 * (1 ref) + 2.0 * (1 shared) = 5.0
    assert score == 5.0


# =============================================================================
# 4. NetworkX Graph Assembly & Neighborhood Query Tests
# =============================================================================

def test_networkx_graph_construction(test_db, sample_workspace, sample_documents):
    """Verify NetworkX graph construction with correct node and edge types."""
    doc1, doc2 = sample_documents

    # Add an entity and an association
    ent = Entity(
        workspace_id=sample_workspace.id,
        canonical_name="ISO-9001",
        normalized_name="iso-9001",
        entity_type="Regulation",
    )
    test_db.add(ent)
    test_db.flush()

    assoc = DocumentEntityAssociation(
        workspace_id=sample_workspace.id,
        document_id=doc1.id,
        entity_id=ent.id,
        page_start=1,
    )
    rel = DocumentRelationship(
        workspace_id=sample_workspace.id,
        source_document_id=doc1.id,
        target_document_id=doc2.id,
        relationship_type="shared_entity",
        confidence_score=0.9,
        explanation="Both comply with ISO-9001",
    )
    test_db.add_all([assoc, rel])
    test_db.commit()

    G = knowledge_graph_service.construct_networkx_graph(
        db=test_db,
        workspace_id=sample_workspace.id,
        limit=50,
    )

    assert isinstance(G, nx.Graph)
    assert f"doc_{doc1.id}" in G
    assert f"doc_{doc2.id}" in G
    assert f"ent_{ent.id}" in G

    # Check edge between doc1 and doc2
    edge_data = G.get_edge_data(f"doc_{doc1.id}", f"doc_{doc2.id}")
    assert edge_data is not None
    assert edge_data["relationship_type"] == "shared_entity"


def test_document_neighborhood_subgraph(test_db, sample_workspace, sample_documents):
    """Verify 1-hop ego subgraph extraction around a target document."""
    doc1, doc2 = sample_documents

    ent = Entity(
        workspace_id=sample_workspace.id,
        canonical_name="Turbine Unit 4",
        normalized_name="turbine unit 4",
        entity_type="Machine",
    )
    test_db.add(ent)
    test_db.flush()

    assoc = DocumentEntityAssociation(
        workspace_id=sample_workspace.id,
        document_id=doc1.id,
        entity_id=ent.id,
        page_start=1,
    )
    test_db.add(assoc)
    test_db.commit()

    subgraph_data = knowledge_graph_service.get_graph_data(
        db=test_db,
        workspace_id=sample_workspace.id,
        focus_document_id=doc1.id,
    )

    node_ids = [n.id for n in subgraph_data.nodes]
    assert f"doc_{doc1.id}" in node_ids
    assert f"ent_{ent.id}" in node_ids
    # doc2 has no connection to doc1 here, so it should not be in the 1-hop neighborhood
    assert f"doc_{doc2.id}" not in node_ids


# =============================================================================
# 5. Graph Visualization (Matplotlib) Tests
# =============================================================================

def test_graph_visualization_renders_valid_png():
    """Verify GraphVisualizationService generates valid PNG bytes."""
    G = nx.Graph()
    G.add_node("doc_1", label="Manual.pdf", node_type="document")
    G.add_node("ent_1", label="Turbine X", node_type="entity")
    G.add_edge("doc_1", "ent_1", relationship_type="mentions")

    png_bytes = graph_visualization_service.render_graph_to_png(G, title="Test Network")
    assert isinstance(png_bytes, bytes)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_graph_visualization_empty_graph():
    """Verify empty graph returns a valid placeholder PNG without crashing."""
    G = nx.Graph()
    png_bytes = graph_visualization_service.render_graph_to_png(G, title="Empty Graph")
    assert isinstance(png_bytes, bytes)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")


# =============================================================================
# 6. Deletion Cascading & Incremental Graph Updates
# =============================================================================

def test_document_deletion_cascades_associations_and_relationships(test_db, sample_workspace, sample_documents):
    """Verify deleting a document automatically purges its associations and relationships."""
    doc1, doc2 = sample_documents

    ent = Entity(
        workspace_id=sample_workspace.id,
        canonical_name="Turbine Core",
        normalized_name="turbine core",
        entity_type="Equipment",
    )
    test_db.add(ent)
    test_db.flush()

    assoc = DocumentEntityAssociation(
        workspace_id=sample_workspace.id,
        document_id=doc1.id,
        entity_id=ent.id,
        page_start=1,
    )
    rel = DocumentRelationship(
        workspace_id=sample_workspace.id,
        source_document_id=doc1.id,
        target_document_id=doc2.id,
        relationship_type="shared_entity",
        confidence_score=0.9,
        explanation="Test",
    )
    test_db.add_all([assoc, rel])
    test_db.commit()

    # Delete doc1
    test_db.delete(doc1)
    test_db.commit()

    # Associations and relationships for doc1 must be removed
    remaining_assocs = test_db.query(DocumentEntityAssociation).filter(DocumentEntityAssociation.document_id == doc1.id).all()
    remaining_rels = test_db.query(DocumentRelationship).filter(
        (DocumentRelationship.source_document_id == doc1.id) | (DocumentRelationship.target_document_id == doc1.id)
    ).all()

    assert len(remaining_assocs) == 0
    assert len(remaining_rels) == 0
    # Entity itself is preserved
    assert test_db.query(Entity).filter(Entity.id == ent.id).first() is not None


# =============================================================================
# 7. FastAPI Knowledge Graph Endpoints Tests
# =============================================================================

def test_api_graph_summary(api_client, sample_workspace, sample_documents):
    """Verify GET /api/v1/workspaces/{workspace_id}/graph/summary endpoint."""
    resp = api_client.get(f"/api/v1/workspaces/{sample_workspace.id}/graph/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["workspace_id"] == sample_workspace.id
    assert data["total_documents"] == 2
    assert "relationship_breakdown" in data


def test_api_graph_data(api_client, sample_workspace, sample_documents):
    """Verify GET /api/v1/workspaces/{workspace_id}/graph returns valid nodes and edges."""
    resp = api_client.get(f"/api/v1/workspaces/{sample_workspace.id}/graph?limit=20")
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data
    assert data["total_nodes"] >= 2


def test_api_graph_render_image(api_client, sample_workspace, sample_documents):
    """Verify GET /api/v1/workspaces/{workspace_id}/graph/render returns PNG image stream."""
    resp = api_client.get(f"/api/v1/workspaces/{sample_workspace.id}/graph/render")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_api_graph_workspace_isolation(api_client, sample_workspace):
    """Verify 404 for requests on nonexistent workspace."""
    resp = api_client.get("/api/v1/workspaces/invalid-workspace-id/graph/summary")
    assert resp.status_code == 404
