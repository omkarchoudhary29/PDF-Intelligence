"""Pydantic schemas for Knowledge Graph data models, API responses, and graph querying."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Entity Schemas
# =============================================================================

class EntityBase(BaseModel):
    canonical_name: str = Field(..., min_length=1, max_length=255)
    entity_type: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = None


class EntityCreate(EntityBase):
    pass


class EntityResponse(EntityBase):
    id: str
    workspace_id: str
    normalized_name: str
    aliases_json: Optional[Any] = None
    attributes_json: Optional[Any] = None
    mention_count: int = 1
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Document-Entity Association Schemas
# =============================================================================

class DocumentEntityAssociationResponse(BaseModel):
    id: str
    workspace_id: str
    document_id: str
    entity_id: str
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None
    section_title: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    excerpt: Optional[str] = None
    mention_count: int = 1
    extraction_method: str = "llm"
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Document Relationship Schemas
# =============================================================================

class DocumentRelationshipResponse(BaseModel):
    id: str
    workspace_id: str
    source_document_id: str
    target_document_id: str
    source_document_name: Optional[str] = None
    target_document_name: Optional[str] = None
    relationship_type: str
    confidence_score: float
    explanation: str
    supporting_evidence: Optional[Any] = None
    source_page_start: Optional[int] = None
    source_page_end: Optional[int] = None
    target_page_start: Optional[int] = None
    target_page_end: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Entity Relationship Schemas
# =============================================================================

class EntityRelationshipResponse(BaseModel):
    id: str
    workspace_id: str
    source_entity_id: str
    target_entity_id: str
    source_entity_name: Optional[str] = None
    target_entity_name: Optional[str] = None
    source_document_id: str
    source_document_name: Optional[str] = None
    relationship_type: str
    confidence_score: float
    explanation: str
    evidence_excerpt: Optional[str] = None
    page_number: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# NetworkX Graph Representation Schemas
# =============================================================================

class GraphNode(BaseModel):
    id: str
    label: str
    node_type: str  # "document" or "entity"
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    relationship_type: str
    confidence: float = 1.0
    explanation: str
    supporting_evidence: Optional[Any] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphDataResponse(BaseModel):
    workspace_id: str
    total_nodes: int
    total_edges: int
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class GraphSummaryResponse(BaseModel):
    workspace_id: str
    total_documents: int
    total_entities: int
    total_relationships: int
    relationship_breakdown: Dict[str, int]
    top_connected_documents: List[Dict[str, Any]]
    top_entities: List[Dict[str, Any]]


class GraphBuildRequest(BaseModel):
    document_ids: Optional[List[str]] = None
    force: bool = False


class GraphBuildResponse(BaseModel):
    workspace_id: str
    documents_processed: int
    entities_extracted: int
    relationships_discovered: int
    entity_relationships_discovered: Optional[int] = 0
    document_relationships_discovered: Optional[int] = 0
    status: str


class PaginatedEntitiesResponse(BaseModel):
    items: List[EntityResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class GraphReconciliationResponse(BaseModel):
    workspace_id: str
    reconciled: bool
    workspace_vector_count: int
    collection_wide_vector_count: int
    company_documents_count: int
    personal_documents_count: int
    total_database_documents: int
    reconciliation_explanation: str


class ShortestPathResponse(BaseModel):
    found: bool
    source: str
    target: str
    path_length: Optional[int] = None
    path: Optional[List[str]] = None
    steps: Optional[List[Dict[str, Any]]] = None
    message: Optional[str] = None
    error: Optional[str] = None
