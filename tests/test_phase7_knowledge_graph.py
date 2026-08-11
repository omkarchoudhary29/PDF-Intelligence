"""Comprehensive Test Suite for Phase 7: Enterprise Knowledge Graph, Entity Extraction, and Multi-Document Provenance.

Verifies:
1. Entity normalization, taxonomy extension, and aliases handling
2. Deterministic and LLM entity extraction
3. Direct entity-to-entity relationship discovery (PART_OF, CONNECTED_TO, REGULATED_BY, etc.)
4. Multi-document independent relationship provenance
5. Safe document deletion & graph cascade (retaining relationships if other company documents support them)
6. Orphaned entity pruning and personal document privacy isolation
7. NetworkX graph assembly, ego graphs, and shortest path traversal
8. Vector and document count reconciliation (workspace-scoped vs. collection-wide)
9. Knowledge graph FastAPI REST endpoints
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
    EntityRelationship,
    Workspace,
    WorkspaceMembership,
)
from backend.main import app
from backend.services.entity_extraction_service import entity_extraction_service
from backend.services.entity_normalization_service import entity_normalization_service
from backend.services.graph_visualization_service import graph_visualization_service
from backend.services.ingestion_service import ingestion_service
from backend.services.knowledge_graph_service import knowledge_graph_service
from backend.services.relationship_discovery_service import relationship_discovery_service
from backend.services.storage_service import storage_service


@pytest.fixture
def test_db():
    """Isolated in-memory SQLite database for Phase 7 graph tests."""
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
def test_workspace(test_db):
    """Create test workspace."""
    ws = Workspace(
        name="Aerospace Propulsion & Systems",
        slug="aerospace-propulsion-systems",
        description="Workspace for Phase 7 Knowledge Graph validation",
        is_active=True,
    )
    test_db.add(ws)
    test_db.commit()
    test_db.refresh(ws)
    return ws


@pytest.fixture
def test_documents(test_db, test_workspace, tmp_path, monkeypatch):
    """Create two related test documents with extracted pages."""
    storage_root = tmp_path / "storage"
    monkeypatch.setattr(storage_service, "storage_root", storage_root)

    # Document 1: Turbine Manual
    doc1 = Document(
        workspace_id=test_workspace.id,
        original_filename="Turbine_Manual_AX450.pdf",
        display_name="Turbine Manual AX-450",
        storage_key=f"workspaces/{test_workspace.id}/documents/Turbine_Manual_AX450.pdf",
        file_size_bytes=1024 * 50,
        sha256_hash="hash_doc_1111",
        mime_type="application/pdf",
        processing_status="completed",
        visibility_scope="company",
        total_pages=3,
    )
    test_db.add(doc1)

    # Document 2: Maintenance Guide
    doc2 = Document(
        workspace_id=test_workspace.id,
        original_filename="Maintenance_Guide_GT200.pdf",
        display_name="Maintenance Guide GT-200",
        storage_key=f"workspaces/{test_workspace.id}/documents/Maintenance_Guide_GT200.pdf",
        file_size_bytes=1024 * 40,
        sha256_hash="hash_doc_2222",
        mime_type="application/pdf",
        processing_status="completed",
        visibility_scope="company",
        total_pages=2,
    )
    test_db.add(doc2)
    test_db.commit()
    test_db.refresh(doc1)
    test_db.refresh(doc2)

    return doc1, doc2


# =============================================================================
# 1. Entity Normalization & Taxonomy
# =============================================================================

def test_01_entity_type_validation_and_taxonomy():
    """Verify expanded entity taxonomy types are recognized and normalized."""
    assert entity_normalization_service.validate_and_normalize_type("Machine") == "Machine"
    assert entity_normalization_service.validate_and_normalize_type("System") == "System"
    assert entity_normalization_service.validate_and_normalize_type("Component") == "Component"
    assert entity_normalization_service.validate_and_normalize_type("Material") == "Material"
    assert entity_normalization_service.validate_and_normalize_type("Standard") == "Standard"
    assert entity_normalization_service.validate_and_normalize_type("Safety hazard") == "Safety hazard"
    assert entity_normalization_service.validate_and_normalize_type("Maintenance procedure") == "Maintenance procedure"
    assert entity_normalization_service.validate_and_normalize_type("Measurement") == "Measurement"
    assert entity_normalization_service.validate_and_normalize_type("UnknownType") == "Other"


def test_02_entity_normalization_and_aliases():
    """Verify type aliases and canonical name normalization."""
    assert entity_normalization_service.validate_and_normalize_type("subsystem") == "System"
    assert entity_normalization_service.validate_and_normalize_type("part") == "Component"
    assert entity_normalization_service.validate_and_normalize_type("hazard") == "Safety hazard"
    assert entity_normalization_service.validate_and_normalize_type("inspection") == "Maintenance procedure"
    assert entity_normalization_service.validate_and_normalize_type("metric") == "Measurement"

    assert entity_normalization_service.normalize_entity_name("  Turbine   Unit 4  ") == "turbine unit 4"
    assert entity_normalization_service.normalize_entity_name('"Inconel 718"') == "inconel 718"


def test_03_aliases_and_attributes_storage(test_db, test_workspace):
    """Verify Entity model persists aliases_json and attributes_json."""
    ent = Entity(
        workspace_id=test_workspace.id,
        canonical_name="Turbine Unit 4",
        normalized_name="turbine unit 4",
        entity_type="Machine",
        description="Main high-pressure turbine unit",
        aliases_json=["TU-4", "Turbine-4", "Primary Turbine"],
        attributes_json={"max_rpm": 12000, "manufacturer": "AeroDynamics Inc"},
    )
    test_db.add(ent)
    test_db.commit()
    test_db.refresh(ent)

    assert ent.id is not None
    assert "TU-4" in ent.aliases_json
    assert ent.attributes_json["max_rpm"] == 12000


# =============================================================================
# 2. Deterministic & LLM Entity Extraction
# =============================================================================

def test_04_deterministic_regex_systems_components_materials():
    """Verify deterministic regex patterns extract Systems, Components, Materials, Standards."""
    text = (
        "The Hydraulic System powers the Actuator and Fuel Pump. "
        "Manufactured from Inconel 718, it complies with AS9100 standard. "
        "Technicians follow Borescope Inspection according to OSHA-1910."
    )
    extracted = entity_extraction_service.extract_deterministic_entities(text)
    names = [e.name for e in extracted]
    types = {e.name: e.entity_type for e in extracted}

    assert any("Hydraulic System" in n for n in names)
    assert any("Inconel 718" in n for n in names)
    assert any("AS9100" in n for n in names)


def test_05_extract_and_store_document_entities_with_aliases(test_db, test_workspace, test_documents):
    """Verify extract_and_store_document_entities populates associations and aliases."""
    doc1, _ = test_documents
    pages = [
        {"page_number": 1, "text": "Turbine Unit 4 is part of Hydraulic System. Inspection per AS9100."},
        {"page_number": 2, "text": "Operating manual for AX-450 Equipment."},
    ]
    assocs = entity_extraction_service.extract_and_store_document_entities(
        db=test_db,
        workspace_id=test_workspace.id,
        document=doc1,
        pages_records=pages,
        use_llm=False,
    )
    assert len(assocs) > 0
    ents = test_db.query(Entity).filter(Entity.workspace_id == test_workspace.id).all()
    assert len(ents) >= 2


# =============================================================================
# 3. Direct Entity-to-Entity Relationships & Provenance
# =============================================================================

def test_06_entity_relationship_discovery_deterministic(test_db, test_workspace, test_documents):
    """Verify discovery of PART_OF and CONNECTED_TO relationships from page text."""
    doc1, _ = test_documents
    ent1 = Entity(workspace_id=test_workspace.id, canonical_name="Fuel Pump", normalized_name="fuel pump", entity_type="Component")
    ent2 = Entity(workspace_id=test_workspace.id, canonical_name="Hydraulic System", normalized_name="hydraulic system", entity_type="System")
    test_db.add_all([ent1, ent2])
    test_db.commit()

    assoc1 = DocumentEntityAssociation(workspace_id=test_workspace.id, document_id=doc1.id, entity_id=ent1.id, page_start=1, excerpt="Fuel Pump")
    assoc2 = DocumentEntityAssociation(workspace_id=test_workspace.id, document_id=doc1.id, entity_id=ent2.id, page_start=1, excerpt="Hydraulic System")
    test_db.add_all([assoc1, assoc2])
    test_db.commit()

    pages = [{
        "page_number": 1,
        "text": "The Fuel Pump is a part of Hydraulic System and delivers high pressure fluid.",
    }]

    rels = relationship_discovery_service.discover_entity_relationships(
        db=test_db,
        workspace_id=test_workspace.id,
        document=doc1,
        pages_records=pages,
        use_llm=False,
    )

    assert len(rels) >= 1
    rel = rels[0]
    assert rel.relationship_type == "PART_OF"
    assert rel.source_entity_id == ent1.id
    assert rel.target_entity_id == ent2.id
    assert rel.source_document_id == doc1.id
    assert rel.page_number == 1
    assert "part of" in rel.evidence_excerpt.lower()


def test_07_entity_relationship_provenance_fields(test_db, test_workspace, test_documents):
    """Verify all provenance fields are stored on EntityRelationship."""
    doc1, _ = test_documents
    ent1 = Entity(workspace_id=test_workspace.id, canonical_name="Actuator", normalized_name="actuator", entity_type="Component")
    ent2 = Entity(workspace_id=test_workspace.id, canonical_name="Avionics System", normalized_name="avionics system", entity_type="System")
    test_db.add_all([ent1, ent2])
    test_db.commit()

    rel = EntityRelationship(
        workspace_id=test_workspace.id,
        source_entity_id=ent1.id,
        target_entity_id=ent2.id,
        source_document_id=doc1.id,
        relationship_type="CONNECTED_TO",
        confidence_score=0.95,
        explanation="Actuator is connected to Avionics System on page 2.",
        evidence_excerpt="The Actuator is connected to Avionics System via bus cable.",
        page_number=2,
    )
    test_db.add(rel)
    test_db.commit()
    test_db.refresh(rel)

    assert rel.workspace_id == test_workspace.id
    assert rel.source_document_id == doc1.id
    assert rel.page_number == 2
    assert rel.evidence_excerpt is not None
    assert rel.confidence_score == 0.95


def test_08_multi_document_relationship_provenance(test_db, test_workspace, test_documents):
    """Verify that multiple documents supporting the same relationship have distinct evidence records."""
    doc1, doc2 = test_documents
    ent1 = Entity(workspace_id=test_workspace.id, canonical_name="Boiler 2", normalized_name="boiler 2", entity_type="Machine")
    ent2 = Entity(workspace_id=test_workspace.id, canonical_name="OSHA-1910", normalized_name="osha-1910", entity_type="Standard")
    test_db.add_all([ent1, ent2])
    test_db.commit()

    # Document 1 establishes relationship on Page 1
    rel_doc1 = EntityRelationship(
        workspace_id=test_workspace.id,
        source_entity_id=ent1.id,
        target_entity_id=ent2.id,
        source_document_id=doc1.id,
        relationship_type="REGULATED_BY",
        confidence_score=0.95,
        explanation="Boiler 2 is regulated by OSHA-1910 in Document 1.",
        evidence_excerpt="Boiler 2 must be maintained in compliance with OSHA-1910.",
        page_number=1,
    )
    # Document 2 establishes same relationship on Page 3
    rel_doc2 = EntityRelationship(
        workspace_id=test_workspace.id,
        source_entity_id=ent1.id,
        target_entity_id=ent2.id,
        source_document_id=doc2.id,
        relationship_type="REGULATED_BY",
        confidence_score=0.95,
        explanation="Boiler 2 is regulated by OSHA-1910 in Document 2.",
        evidence_excerpt="Inspection of Boiler 2 per OSHA-1910 guidelines.",
        page_number=3,
    )
    test_db.add_all([rel_doc1, rel_doc2])
    test_db.commit()

    matching_rels = (
        test_db.query(EntityRelationship)
        .filter(
            EntityRelationship.workspace_id == test_workspace.id,
            EntityRelationship.source_entity_id == ent1.id,
            EntityRelationship.target_entity_id == ent2.id,
            EntityRelationship.relationship_type == "REGULATED_BY",
        )
        .all()
    )
    assert len(matching_rels) == 2
    doc_ids = {r.source_document_id for r in matching_rels}
    assert doc_ids == {doc1.id, doc2.id}


# =============================================================================
# 4. Safe Document Deletion & Cascade
# =============================================================================

def test_09_document_deletion_removes_only_associated_relationship_evidence(test_db, test_workspace, test_documents):
    """Deleting Doc 1 removes Doc 1's evidence but leaves Doc 2's evidence intact."""
    doc1, doc2 = test_documents
    ent1 = Entity(workspace_id=test_workspace.id, canonical_name="Compressor C-1", normalized_name="compressor c-1", entity_type="Machine")
    ent2 = Entity(workspace_id=test_workspace.id, canonical_name="Cooling System", normalized_name="cooling system", entity_type="System")
    test_db.add_all([ent1, ent2])
    test_db.commit()

    # Add associations to both documents so entities are not orphaned
    assoc1 = DocumentEntityAssociation(workspace_id=test_workspace.id, document_id=doc1.id, entity_id=ent1.id, page_start=1)
    assoc2 = DocumentEntityAssociation(workspace_id=test_workspace.id, document_id=doc2.id, entity_id=ent1.id, page_start=1)
    assoc3 = DocumentEntityAssociation(workspace_id=test_workspace.id, document_id=doc2.id, entity_id=ent2.id, page_start=1)
    test_db.add_all([assoc1, assoc2, assoc3])

    rel1 = EntityRelationship(
        workspace_id=test_workspace.id,
        source_entity_id=ent1.id,
        target_entity_id=ent2.id,
        source_document_id=doc1.id,
        relationship_type="PART_OF",
        confidence_score=0.90,
        explanation="Compressor C-1 is part of Cooling System in Doc 1",
        evidence_excerpt="Compressor C-1 is part of Cooling System.",
        page_number=1,
    )
    rel2 = EntityRelationship(
        workspace_id=test_workspace.id,
        source_entity_id=ent1.id,
        target_entity_id=ent2.id,
        source_document_id=doc2.id,
        relationship_type="PART_OF",
        confidence_score=0.90,
        explanation="Compressor C-1 is part of Cooling System in Doc 2",
        evidence_excerpt="Inspection of Compressor C-1 within Cooling System.",
        page_number=1,
    )
    test_db.add_all([rel1, rel2])
    test_db.commit()

    # Delete Document 1
    cascade_summary = knowledge_graph_service.handle_document_deletion(test_db, test_workspace.id, doc1.id)
    assert cascade_summary["deleted_entity_relationships"] == 1

    # Verify Doc 1 relationship is gone, Doc 2 relationship remains!
    remaining_rels = (
        test_db.query(EntityRelationship)
        .filter(
            EntityRelationship.workspace_id == test_workspace.id,
            EntityRelationship.source_entity_id == ent1.id,
            EntityRelationship.target_entity_id == ent2.id,
        )
        .all()
    )
    assert len(remaining_rels) == 1
    assert remaining_rels[0].source_document_id == doc2.id


def test_10_deleting_last_supporting_document_removes_relationship(test_db, test_workspace, test_documents):
    """When the last document supporting a relationship is deleted, the relationship is completely removed."""
    _, doc2 = test_documents
    ent1 = Entity(workspace_id=test_workspace.id, canonical_name="Solo Machine", normalized_name="solo machine", entity_type="Machine")
    ent2 = Entity(workspace_id=test_workspace.id, canonical_name="Solo System", normalized_name="solo system", entity_type="System")
    test_db.add_all([ent1, ent2])
    test_db.commit()

    rel = EntityRelationship(
        workspace_id=test_workspace.id,
        source_entity_id=ent1.id,
        target_entity_id=ent2.id,
        source_document_id=doc2.id,
        relationship_type="OPERATES",
        confidence_score=0.90,
        explanation="Solo Machine operates Solo System",
        evidence_excerpt="Solo Machine operates Solo System.",
        page_number=1,
    )
    test_db.add(rel)
    test_db.commit()

    # Now delete Document 2
    cascade_summary = knowledge_graph_service.handle_document_deletion(test_db, test_workspace.id, doc2.id)
    assert cascade_summary["deleted_entity_relationships"] == 1

    remaining_rels = (
        test_db.query(EntityRelationship)
        .filter(
            EntityRelationship.workspace_id == test_workspace.id,
            EntityRelationship.source_entity_id == ent1.id,
            EntityRelationship.target_entity_id == ent2.id,
        )
        .all()
    )
    assert len(remaining_rels) == 0


def test_11_orphaned_entity_cleanup(test_db, test_workspace, test_documents):
    """When a document is deleted and an entity has 0 remaining associations, the entity is pruned."""
    doc1, _ = test_documents
    lone_ent = Entity(workspace_id=test_workspace.id, canonical_name="Unique Orphan Widget", normalized_name="unique orphan widget", entity_type="Equipment")
    test_db.add(lone_ent)
    test_db.commit()

    assoc = DocumentEntityAssociation(workspace_id=test_workspace.id, document_id=doc1.id, entity_id=lone_ent.id, page_start=1)
    test_db.add(assoc)
    test_db.commit()

    # Deleting doc1 should prune lone_ent
    cascade = knowledge_graph_service.handle_document_deletion(test_db, test_workspace.id, doc1.id)
    assert cascade["pruned_entities"] >= 1

    check = test_db.query(Entity).filter(Entity.id == lone_ent.id).first()
    assert check is None


def test_12_personal_document_deletion_does_not_affect_company_graph(test_db, test_workspace):
    """Personal document deletion does not alter company graph associations or relationships."""
    company_ent = Entity(workspace_id=test_workspace.id, canonical_name="Company Core Engine", normalized_name="company core engine", entity_type="Machine")
    test_db.add(company_ent)
    test_db.commit()

    # Create personal doc
    personal_doc = Document(
        workspace_id=test_workspace.id,
        original_filename="Employee_Notes.pdf",
        display_name="Employee Notes",
        storage_key=f"workspaces/{test_workspace.id}/documents/personal_notes.pdf",
        file_size_bytes=1024,
        sha256_hash="hash_personal_999",
        mime_type="application/pdf",
        processing_status="completed",
        visibility_scope="personal",
        owner_user_id="user_emp_123",
    )
    test_db.add(personal_doc)
    test_db.commit()

    # Delete personal document
    cascade = knowledge_graph_service.handle_document_deletion(test_db, test_workspace.id, personal_doc.id)
    assert cascade["deleted_entity_relationships"] == 0

    # Company entity remains intact
    assert test_db.query(Entity).filter(Entity.id == company_ent.id).first() is not None


# =============================================================================
# 5. Company Scope & Data Isolation
# =============================================================================

def test_13_company_scope_filtering_in_graph_build(test_db, test_workspace):
    """Personal documents are excluded from build_workspace_graph."""
    p_doc = Document(
        workspace_id=test_workspace.id,
        original_filename="Secret_Salary_Notes.pdf",
        display_name="Salary Notes",
        storage_key=f"workspaces/{test_workspace.id}/documents/salary.pdf",
        file_size_bytes=1024,
        sha256_hash="hash_salary",
        mime_type="application/pdf",
        processing_status="completed",
        visibility_scope="personal",
        owner_user_id="user_emp_123",
    )
    test_db.add(p_doc)
    test_db.commit()

    with patch.object(ingestion_service, "get_extracted_pages", return_value=[{"page_number": 1, "text": "Secret Info"}]):
        res = knowledge_graph_service.build_workspace_graph(test_db, test_workspace.id)
        # p_doc should NOT have been processed
        assert res["documents_processed"] == 0


def test_14_workspace_boundary_isolation(test_db):
    """Knowledge graphs of two workspaces are strictly isolated."""
    ws_a = Workspace(name="Workspace A", slug="workspace-a", is_active=True)
    ws_b = Workspace(name="Workspace B", slug="workspace-b", is_active=True)
    test_db.add_all([ws_a, ws_b])
    test_db.commit()

    ent_a = Entity(workspace_id=ws_a.id, canonical_name="Alpha Engine", normalized_name="alpha engine", entity_type="Machine")
    ent_b = Entity(workspace_id=ws_b.id, canonical_name="Beta Turbine", normalized_name="beta turbine", entity_type="Machine")
    test_db.add_all([ent_a, ent_b])
    test_db.commit()

    G_a = knowledge_graph_service.construct_networkx_graph(test_db, ws_a.id)
    G_b = knowledge_graph_service.construct_networkx_graph(test_db, ws_b.id)

    assert f"ent_{ent_a.id}" in G_a
    assert f"ent_{ent_b.id}" not in G_a
    assert f"ent_{ent_b.id}" in G_b
    assert f"ent_{ent_a.id}" not in G_b


# =============================================================================
# 6. NetworkX Graph Assembly, Ego Graphs, & Shortest Path
# =============================================================================

def test_15_networkx_graph_multi_provenance_edges(test_db, test_workspace, test_documents):
    """Verify construct_networkx_graph builds edges with aggregated multi-document provenance."""
    doc1, doc2 = test_documents
    ent1 = Entity(workspace_id=test_workspace.id, canonical_name="System A", normalized_name="system a", entity_type="System")
    ent2 = Entity(workspace_id=test_workspace.id, canonical_name="Component B", normalized_name="component b", entity_type="Component")
    test_db.add_all([ent1, ent2])
    test_db.commit()

    r1 = EntityRelationship(
        workspace_id=test_workspace.id,
        source_entity_id=ent1.id,
        target_entity_id=ent2.id,
        source_document_id=doc1.id,
        relationship_type="HAS_COMPONENT",
        confidence_score=0.90,
        explanation="Doc 1 explanation",
        evidence_excerpt="Evidence 1",
        page_number=1,
    )
    r2 = EntityRelationship(
        workspace_id=test_workspace.id,
        source_entity_id=ent1.id,
        target_entity_id=ent2.id,
        source_document_id=doc2.id,
        relationship_type="HAS_COMPONENT",
        confidence_score=0.95,
        explanation="Doc 2 explanation",
        evidence_excerpt="Evidence 2",
        page_number=2,
    )
    test_db.add_all([r1, r2])
    test_db.commit()

    G = knowledge_graph_service.construct_networkx_graph(test_db, test_workspace.id)
    u = f"ent_{ent1.id}"
    v = f"ent_{ent2.id}"

    assert G.has_edge(u, v)
    edge_data = G[u][v]
    assert edge_data["relationship_type"] == "HAS_COMPONENT"
    assert edge_data["evidence_count"] == 2
    assert len(edge_data["evidence_records"]) == 2


def test_16_shortest_path_traversal(test_db, test_workspace):
    """Verify find_shortest_path discovers multi-hop graph path between nodes."""
    ent1 = Entity(workspace_id=test_workspace.id, canonical_name="Radar Sensor", normalized_name="radar sensor", entity_type="Equipment")
    ent2 = Entity(workspace_id=test_workspace.id, canonical_name="Telemetry Hub", normalized_name="telemetry hub", entity_type="System")
    ent3 = Entity(workspace_id=test_workspace.id, canonical_name="Ground Station", normalized_name="ground station", entity_type="Location")
    test_db.add_all([ent1, ent2, ent3])
    test_db.commit()

    # ent1 -> ent2 -> ent3
    r1 = EntityRelationship(workspace_id=test_workspace.id, source_entity_id=ent1.id, target_entity_id=ent2.id, source_document_id="doc_stub_1", relationship_type="CONNECTED_TO", explanation="conn", evidence_excerpt="ev1", page_number=1)
    r2 = EntityRelationship(workspace_id=test_workspace.id, source_entity_id=ent2.id, target_entity_id=ent3.id, source_document_id="doc_stub_2", relationship_type="LOCATED_IN", explanation="loc", evidence_excerpt="ev2", page_number=2)
    test_db.add_all([r1, r2])
    test_db.commit()

    path_res = knowledge_graph_service.find_shortest_path(test_db, test_workspace.id, ent1.id, ent3.id)
    assert path_res["found"] is True
    assert path_res["path_length"] == 2
    assert path_res["path"] == [f"ent_{ent1.id}", f"ent_{ent2.id}", f"ent_{ent3.id}"]


def test_17_ego_graph_focus_entity_and_document(test_db, test_workspace, test_documents):
    """Verify ego graph extraction for focused document and entity nodes."""
    doc1, _ = test_documents
    ent1 = Entity(workspace_id=test_workspace.id, canonical_name="Focus Target", normalized_name="focus target", entity_type="Machine")
    test_db.add(ent1)
    test_db.commit()

    assoc = DocumentEntityAssociation(workspace_id=test_workspace.id, document_id=doc1.id, entity_id=ent1.id, page_start=1)
    test_db.add(assoc)
    test_db.commit()

    G_doc = knowledge_graph_service.construct_networkx_graph(test_db, test_workspace.id, focus_document_id=doc1.id)
    assert f"doc_{doc1.id}" in G_doc
    assert f"ent_{ent1.id}" in G_doc

    G_ent = knowledge_graph_service.construct_networkx_graph(test_db, test_workspace.id, focus_entity_id=ent1.id)
    assert f"ent_{ent1.id}" in G_ent
    assert f"doc_{doc1.id}" in G_ent


# =============================================================================
# 7. Vector & Document Count Reconciliation
# =============================================================================

def test_18_vector_and_document_count_reconciliation(test_db, test_workspace, test_documents):
    """Verify get_reconciliation_status reports workspace vs collection counts accurately."""
    status = knowledge_graph_service.get_reconciliation_status(test_db, test_workspace.id)
    assert status["reconciled"] is True
    assert status["workspace_id"] == test_workspace.id
    assert status["company_documents_count"] == 2
    assert "workspace_vector_count" in status
    assert "collection_wide_vector_count" in status
    assert "reconciliation_explanation" in status


# =============================================================================
# 8. REST API Endpoints
# =============================================================================

def test_19_api_graph_summary(api_client, test_workspace, test_documents):
    """Verify GET /api/v1/workspaces/{workspace_id}/graph/summary."""
    resp = api_client.get(f"/api/v1/workspaces/{test_workspace.id}/graph/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["workspace_id"] == test_workspace.id
    assert data["total_documents"] == 2
    assert "total_entities" in data
    assert "relationship_breakdown" in data


def test_20_api_graph_entities_pagination_and_search(api_client, test_db, test_workspace):
    """Verify GET /api/v1/workspaces/{workspace_id}/graph/entities."""
    ent = Entity(
        workspace_id=test_workspace.id,
        canonical_name="Hydraulic Actuator H-1",
        normalized_name="hydraulic actuator h-1",
        entity_type="Component",
        aliases_json=["HA-1"],
    )
    test_db.add(ent)
    test_db.commit()

    # Search by query
    resp = api_client.get(f"/api/v1/workspaces/{test_workspace.id}/graph/entities?q=Actuator")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    assert any("Actuator" in item["canonical_name"] for item in data["items"])


def test_21_api_entity_relationships_endpoint(api_client, test_db, test_workspace, test_documents):
    """Verify GET /api/v1/workspaces/{workspace_id}/graph/entities/{entity_id}/relationships."""
    doc1, _ = test_documents
    ent1 = Entity(workspace_id=test_workspace.id, canonical_name="Sensor X", normalized_name="sensor x", entity_type="Equipment")
    ent2 = Entity(workspace_id=test_workspace.id, canonical_name="Controller Y", normalized_name="controller y", entity_type="Equipment")
    test_db.add_all([ent1, ent2])
    test_db.commit()

    rel = EntityRelationship(
        workspace_id=test_workspace.id,
        source_entity_id=ent1.id,
        target_entity_id=ent2.id,
        source_document_id=doc1.id,
        relationship_type="CONNECTED_TO",
        explanation="Sensor X feeds Controller Y",
        evidence_excerpt="Sensor X connects to Controller Y.",
        page_number=1,
    )
    test_db.add(rel)
    test_db.commit()

    resp = api_client.get(f"/api/v1/workspaces/{test_workspace.id}/graph/entities/{ent1.id}/relationships")
    assert resp.status_code == 200
    rels = resp.json()
    assert len(rels) >= 1
    assert rels[0]["relationship_type"] == "CONNECTED_TO"
    assert rels[0]["source_document_name"] == doc1.original_filename


def test_22_api_graph_path_endpoint(api_client, test_db, test_workspace):
    """Verify GET /api/v1/workspaces/{workspace_id}/graph/path."""
    ent1 = Entity(workspace_id=test_workspace.id, canonical_name="Node 1", normalized_name="node 1", entity_type="System")
    ent2 = Entity(workspace_id=test_workspace.id, canonical_name="Node 2", normalized_name="node 2", entity_type="Component")
    test_db.add_all([ent1, ent2])
    test_db.commit()

    rel = EntityRelationship(workspace_id=test_workspace.id, source_entity_id=ent1.id, target_entity_id=ent2.id, source_document_id="stub", relationship_type="REQUIRES", explanation="req", evidence_excerpt="ev", page_number=1)
    test_db.add(rel)
    test_db.commit()

    resp = api_client.get(f"/api/v1/workspaces/{test_workspace.id}/graph/path?source_id={ent1.id}&target_id={ent2.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["path_length"] == 1


def test_23_api_graph_reconciliation_endpoint(api_client, test_workspace):
    """Verify GET /api/v1/workspaces/{workspace_id}/graph/reconciliation."""
    resp = api_client.get(f"/api/v1/workspaces/{test_workspace.id}/graph/reconciliation")
    assert resp.status_code == 200
    data = resp.json()
    assert data["reconciled"] is True
    assert data["workspace_id"] == test_workspace.id


def test_24_api_graph_backfill_endpoint(api_client, test_workspace):
    """Verify POST /api/v1/workspaces/{workspace_id}/graph/backfill."""
    with patch.object(ingestion_service, "get_extracted_pages", return_value=[]):
        resp = api_client.post(f"/api/v1/workspaces/{test_workspace.id}/graph/backfill")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"


def test_25_api_graph_render_image_png(api_client, test_workspace):
    """Verify GET /api/v1/workspaces/{workspace_id}/graph/render returns PNG image."""
    resp = api_client.get(f"/api/v1/workspaces/{test_workspace.id}/graph/render")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(b"\x89PNG")


def test_26_api_get_relationship_details_unified(api_client, test_db, test_workspace, test_documents):
    """Verify GET /relationships/{id} resolves EntityRelationship records."""
    doc1, _ = test_documents
    ent1 = Entity(workspace_id=test_workspace.id, canonical_name="Part A", normalized_name="part a", entity_type="Component")
    ent2 = Entity(workspace_id=test_workspace.id, canonical_name="Part B", normalized_name="part b", entity_type="Component")
    test_db.add_all([ent1, ent2])
    test_db.commit()

    rel = EntityRelationship(
        workspace_id=test_workspace.id,
        source_entity_id=ent1.id,
        target_entity_id=ent2.id,
        source_document_id=doc1.id,
        relationship_type="REPLACES",
        explanation="Part A replaces Part B",
        evidence_excerpt="Part A supersedes Part B per bulletin.",
        page_number=1,
    )
    test_db.add(rel)
    test_db.commit()

    resp = api_client.get(f"/api/v1/workspaces/{test_workspace.id}/graph/relationships/{rel.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == rel.id
    assert data["relationship_type"] == "REPLACES"
    assert data["evidence_excerpt"] == "Part A supersedes Part B per bulletin."
