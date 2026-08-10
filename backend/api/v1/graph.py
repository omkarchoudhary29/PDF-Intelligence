"""FastAPI Router for Knowledge Graph and Relationship Discovery."""

import logging
from typing import List, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from backend.core.dependencies import (
    require_workspace_admin,
    require_workspace_member,
)
from backend.database.connection import get_db
from backend.database.models import (
    Document,
    DocumentEntityAssociation,
    Entity,
    EntityRelationship,
    Workspace,
    WorkspaceMembership,
)
from backend.schemas.graph import (
    DocumentRelationshipResponse,
    EntityRelationshipResponse,
    EntityResponse,
    GraphBuildRequest,
    GraphBuildResponse,
    GraphDataResponse,
    GraphReconciliationResponse,
    GraphSummaryResponse,
    PaginatedEntitiesResponse,
    ShortestPathResponse,
)
from backend.services.graph_visualization_service import graph_visualization_service
from backend.services.knowledge_graph_service import knowledge_graph_service
from backend.services.workspace_service import WorkspaceService

logger = logging.getLogger("ekn.graph_api")

router = APIRouter(
    prefix="/api/v1/workspaces/{workspace_id}/graph",
    tags=["Knowledge Graph"],
)


def _get_workspace_or_404(workspace_id: str, db: Session) -> Workspace:
    """Validate workspace existence."""
    ws = WorkspaceService.get_by_id(db, workspace_id)
    if not ws:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' was not found.",
        )
    return ws


@router.post(
    "/build",
    response_model=GraphBuildResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger knowledge graph extraction and relationship discovery",
)
def build_graph(
    workspace_id: str,
    request: Optional[GraphBuildRequest] = None,
    membership: WorkspaceMembership = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> GraphBuildResponse:
    """Trigger entity extraction and relationship discovery across workspace documents."""
    _get_workspace_or_404(workspace_id, db)
    doc_ids = request.document_ids if request else None
    force = request.force if request else False

    result = knowledge_graph_service.build_workspace_graph(
        db=db,
        workspace_id=workspace_id,
        document_ids=doc_ids,
        force=force,
    )
    return GraphBuildResponse(**result)


@router.post(
    "/backfill",
    response_model=GraphBuildResponse,
    status_code=status.HTTP_200_OK,
    summary="Backfill knowledge graph extraction for company documents without re-embedding",
)
def backfill_graph(
    workspace_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_admin),
    db: Session = Depends(get_db),
) -> GraphBuildResponse:
    """Extract entities and discover relationships on company documents lacking graph data without modifying vectors."""
    _get_workspace_or_404(workspace_id, db)
    # Identify company documents lacking entity associations
    docs = (
        db.query(Document)
        .filter(
            Document.workspace_id == workspace_id,
            Document.processing_status == "completed",
            Document.visibility_scope != "personal",
        )
        .all()
    )

    doc_ids_to_process = []
    for d in docs:
        assoc_count = (
            db.query(DocumentEntityAssociation)
            .filter(DocumentEntityAssociation.document_id == d.id)
            .count()
        )
        if assoc_count == 0:
            doc_ids_to_process.append(d.id)

    # If all already processed, backfill all company documents
    if not doc_ids_to_process:
        doc_ids_to_process = [d.id for d in docs]

    result = knowledge_graph_service.build_workspace_graph(
        db=db,
        workspace_id=workspace_id,
        document_ids=doc_ids_to_process,
        force=True,
    )
    return GraphBuildResponse(**result)


@router.get(
    "/summary",
    response_model=GraphSummaryResponse,
    summary="Get high-level knowledge graph statistics",
)
def get_graph_summary(
    workspace_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> GraphSummaryResponse:
    """Retrieve counts of documents, entities, relationships, and top connected documents."""
    _get_workspace_or_404(workspace_id, db)
    return knowledge_graph_service.get_workspace_summary(db, workspace_id)


@router.get(
    "/reconciliation",
    response_model=GraphReconciliationResponse,
    summary="Verify document and vector count reconciliation",
)
def get_reconciliation_status(
    workspace_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> GraphReconciliationResponse:
    """Retrieve verified vector counts (workspace vs. collection-wide) and document reconciliation."""
    _get_workspace_or_404(workspace_id, db)
    status_data = knowledge_graph_service.get_reconciliation_status(db, workspace_id)
    return GraphReconciliationResponse(**status_data)


@router.get(
    "/path",
    response_model=ShortestPathResponse,
    summary="Compute shortest path between two nodes in the knowledge graph",
)
def get_shortest_path(
    workspace_id: str,
    source_id: str = Query(..., description="Source entity or document ID"),
    target_id: str = Query(..., description="Target entity or document ID"),
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> ShortestPathResponse:
    """Compute and return the shortest traversal path between two entities or documents."""
    _get_workspace_or_404(workspace_id, db)
    res = knowledge_graph_service.find_shortest_path(db, workspace_id, source_id, target_id)
    return ShortestPathResponse(**res)


@router.get(
    "/entities",
    response_model=PaginatedEntitiesResponse,
    summary="Search and list knowledge graph entities",
)
def list_entities(
    workspace_id: str,
    q: Optional[str] = Query(default=None, description="Search term for entity canonical name or aliases"),
    entity_type: Optional[str] = Query(default=None, description="Filter by entity type"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=5, le=200),
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> PaginatedEntitiesResponse:
    """List entities in the workspace knowledge graph with pagination and search."""
    _get_workspace_or_404(workspace_id, db)
    query = db.query(Entity).filter(Entity.workspace_id == workspace_id)

    if entity_type:
        query = query.filter(Entity.entity_type == entity_type)

    if q and q.strip():
        term = f"%{q.strip().lower()}%"
        query = query.filter(
            (Entity.canonical_name.ilike(term)) |
            (Entity.normalized_name.ilike(term))
        )

    total = query.count()
    entities = query.order_by(Entity.canonical_name.asc()).offset((page - 1) * page_size).limit(page_size).all()

    items: List[EntityResponse] = []
    for ent in entities:
        mention_cnt = (
            db.query(DocumentEntityAssociation)
            .filter(
                DocumentEntityAssociation.workspace_id == workspace_id,
                DocumentEntityAssociation.entity_id == ent.id,
            )
            .count()
        )
        items.append(
            EntityResponse(
                id=ent.id,
                workspace_id=ent.workspace_id,
                canonical_name=ent.canonical_name,
                normalized_name=ent.normalized_name,
                entity_type=ent.entity_type,
                description=ent.description,
                aliases_json=ent.aliases_json or [],
                attributes_json=ent.attributes_json or {},
                mention_count=max(mention_cnt, 1),
                created_at=ent.created_at,
                updated_at=ent.updated_at,
            )
        )

    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    return PaginatedEntitiesResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get(
    "",
    response_model=GraphDataResponse,
    summary="Retrieve bounded knowledge graph data",
)
def get_graph_data(
    workspace_id: str,
    limit: int = Query(default=50, ge=5, le=200, description="Max nodes to return"),
    relationship_type: Optional[str] = Query(default=None, description="Filter by relationship type"),
    focus_document_id: Optional[str] = Query(default=None, description="Focus on document neighborhood"),
    focus_entity_id: Optional[str] = Query(default=None, description="Focus on entity neighborhood"),
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> GraphDataResponse:
    """Retrieve serialized nodes and edges for rendering or querying."""
    _get_workspace_or_404(workspace_id, db)
    rel_types = [relationship_type] if relationship_type else None

    return knowledge_graph_service.get_graph_data(
        db=db,
        workspace_id=workspace_id,
        limit=limit,
        relationship_types=rel_types,
        focus_document_id=focus_document_id,
        focus_entity_id=focus_entity_id,
    )


@router.get(
    "/render",
    summary="Render knowledge graph to a PNG diagram using Matplotlib",
)
def render_graph_image(
    workspace_id: str,
    limit: int = Query(default=40, ge=5, le=100),
    relationship_type: Optional[str] = Query(default=None),
    focus_document_id: Optional[str] = Query(default=None),
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> Response:
    """Render NetworkX graph to PNG bytes stream."""
    _get_workspace_or_404(workspace_id, db)
    rel_types = [relationship_type] if relationship_type else None

    # Construct graph
    G = knowledge_graph_service.construct_networkx_graph(
        db=db,
        workspace_id=workspace_id,
        limit=limit,
        relationship_types=rel_types,
        focus_document_id=focus_document_id,
    )

    title = "Enterprise Knowledge Graph"
    if focus_document_id:
        doc = db.query(Document).filter(Document.id == focus_document_id).first()
        if doc:
            title = f"Document Neighborhood: {doc.original_filename}"

    png_bytes = graph_visualization_service.render_graph_to_png(G, title=title)
    return Response(content=png_bytes, media_type="image/png")


@router.get(
    "/documents/{document_id}",
    response_model=GraphDataResponse,
    summary="Retrieve 1-hop neighborhood for a specific document",
)
def get_document_neighborhood(
    workspace_id: str,
    document_id: str,
    limit: int = Query(default=30, ge=5, le=100),
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> GraphDataResponse:
    """Retrieve graph nodes and edges directly connected to the specified document."""
    _get_workspace_or_404(workspace_id, db)
    doc = (
        db.query(Document)
        .filter(Document.id == document_id, Document.workspace_id == workspace_id)
        .first()
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' was not found in workspace '{workspace_id}'.",
        )

    return knowledge_graph_service.get_graph_data(
        db=db,
        workspace_id=workspace_id,
        limit=limit,
        focus_document_id=document_id,
    )


@router.get(
    "/entities/{entity_id}",
    response_model=GraphDataResponse,
    summary="Retrieve entity details and connected documents",
)
def get_entity_neighborhood(
    workspace_id: str,
    entity_id: str,
    limit: int = Query(default=30, ge=5, le=100),
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> GraphDataResponse:
    """Retrieve entity details and connected documents."""
    _get_workspace_or_404(workspace_id, db)
    entity = (
        db.query(Entity)
        .filter(Entity.id == entity_id, Entity.workspace_id == workspace_id)
        .first()
    )
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity '{entity_id}' was not found in workspace '{workspace_id}'.",
        )

    return knowledge_graph_service.get_graph_data(
        db=db,
        workspace_id=workspace_id,
        limit=limit,
        focus_entity_id=entity_id,
    )


@router.get(
    "/entities/{entity_id}/relationships",
    response_model=List[EntityRelationshipResponse],
    summary="Retrieve all direct entity relationships involving this entity",
)
def get_entity_relationships(
    workspace_id: str,
    entity_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> List[EntityRelationshipResponse]:
    """Retrieve all entity-to-entity relationships where this entity is either source or target."""
    _get_workspace_or_404(workspace_id, db)
    rels = (
        db.query(EntityRelationship)
        .filter(
            EntityRelationship.workspace_id == workspace_id,
            (EntityRelationship.source_entity_id == entity_id) |
            (EntityRelationship.target_entity_id == entity_id),
        )
        .all()
    )
    result: List[EntityRelationshipResponse] = []
    for r in rels:
        src_e = db.query(Entity).filter(Entity.id == r.source_entity_id).first()
        tgt_e = db.query(Entity).filter(Entity.id == r.target_entity_id).first()
        doc = db.query(Document).filter(Document.id == r.source_document_id).first()
        result.append(
            EntityRelationshipResponse(
                id=r.id,
                workspace_id=r.workspace_id,
                source_entity_id=r.source_entity_id,
                target_entity_id=r.target_entity_id,
                source_entity_name=src_e.canonical_name if src_e else "Unknown",
                target_entity_name=tgt_e.canonical_name if tgt_e else "Unknown",
                source_document_id=r.source_document_id,
                source_document_name=doc.original_filename if doc else "Unknown",
                relationship_type=r.relationship_type,
                confidence_score=r.confidence_score,
                explanation=r.explanation,
                evidence_excerpt=r.evidence_excerpt,
                page_number=r.page_number,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
        )
    return result


@router.get(
    "/relationships/{relationship_id}",
    response_model=Union[DocumentRelationshipResponse, EntityRelationshipResponse],
    summary="Retrieve full relationship details and supporting evidence",
)
def get_relationship_details(
    workspace_id: str,
    relationship_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    db: Session = Depends(get_db),
) -> Union[DocumentRelationshipResponse, EntityRelationshipResponse]:
    """Retrieve complete evidence, page references, and explanations for a relationship."""
    _get_workspace_or_404(workspace_id, db)
    details = knowledge_graph_service.get_relationship_details(
        db=db,
        workspace_id=workspace_id,
        relationship_id=relationship_id,
    )
    if details:
        return details

    ent_details = knowledge_graph_service.get_entity_relationship_details(
        db=db,
        workspace_id=workspace_id,
        relationship_id=relationship_id,
    )
    if ent_details:
        return ent_details

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Relationship '{relationship_id}' was not found in workspace '{workspace_id}'.",
    )
