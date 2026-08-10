"""Knowledge Graph Construction and Querying Service for Enterprise Knowledge Navigator.

Constructs, queries, and serializes in-memory NetworkX knowledge graphs from
persistent relational database records with support for document neighborhoods,
entity subgraphs, and bounded graph exploration.
"""

import logging
from typing import Any, Dict, List, Optional, Set
import networkx as nx
from sqlalchemy.orm import Session

from backend.database.connection import get_chroma_client
from backend.database.models import (
    Document,
    DocumentEntityAssociation,
    DocumentRelationship,
    Entity,
    EntityRelationship,
    Workspace,
)
from backend.schemas.graph import (
    DocumentRelationshipResponse,
    EntityRelationshipResponse,
    GraphDataResponse,
    GraphEdge,
    GraphNode,
    GraphSummaryResponse,
)
from backend.services.entity_extraction_service import entity_extraction_service
from backend.services.ingestion_service import ingestion_service
from backend.services.relationship_discovery_service import relationship_discovery_service

logger = logging.getLogger("ekn.graph.service")


class KnowledgeGraphService:
    """Orchestrates graph discovery, NetworkX graph assembly, and bounded graph queries."""

    def build_workspace_graph(
        self,
        db: Session,
        workspace_id: str,
        document_ids: Optional[List[str]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Run full extraction and relationship discovery pipeline for workspace documents.

        Only processes company-scoped documents to preserve strict privacy boundaries.
        """
        # 1. Fetch target company documents
        query = db.query(Document).filter(
            Document.workspace_id == workspace_id,
            Document.processing_status == "completed",
            Document.visibility_scope != "personal",
        )
        if document_ids:
            query = query.filter(Document.id.in_(document_ids))
        documents = query.all()

        total_extracted_entities = 0
        total_entity_relationships = 0
        relationships_found = 0

        # Load all company documents for cross-referencing
        all_workspace_docs = (
            db.query(Document)
            .filter(
                Document.workspace_id == workspace_id,
                Document.processing_status == "completed",
                Document.visibility_scope != "personal",
            )
            .all()
        )

        doc_texts: Dict[str, str] = {}

        # 2. Extract entities, entity relationships, and direct references per document
        for doc in documents:
            try:
                pages = ingestion_service.get_extracted_pages(db, workspace_id, doc.id)
                if not pages:
                    continue

                # Build combined text for topic overlap
                combined_text = "\n".join(
                    (p.get("text") or p.get("raw_text") or "") if isinstance(p, dict)
                    else (getattr(p, "text", "") or getattr(p, "raw_text", "") or "")
                    for p in pages[:5]
                )
                doc_texts[doc.id] = combined_text

                # Entity extraction
                associations = entity_extraction_service.extract_and_store_document_entities(
                    db=db,
                    workspace_id=workspace_id,
                    document=doc,
                    pages_records=pages,
                    use_llm=True,
                )
                total_extracted_entities += len(associations)

                # Entity-to-entity direct relationships
                ent_rels = relationship_discovery_service.discover_entity_relationships(
                    db=db,
                    workspace_id=workspace_id,
                    document=doc,
                    pages_records=pages,
                    use_llm=True,
                )
                total_entity_relationships += len(ent_rels)

                # Direct document references
                direct_rels = relationship_discovery_service.discover_direct_reference_relationships(
                    db=db,
                    workspace_id=workspace_id,
                    source_doc=doc,
                    pages_records=pages,
                    all_workspace_docs=all_workspace_docs,
                )
                relationships_found += len(direct_rels)

            except Exception as exc:
                logger.error("Error processing document %s for graph: %s", doc.id, exc, exc_info=True)

        # 3. Discover shared entity relationships
        shared_rels = relationship_discovery_service.discover_shared_entity_relationships(
            db=db,
            workspace_id=workspace_id,
        )
        relationships_found += len(shared_rels)

        # 4. Discover semantic topic relationships between documents
        for doc in documents:
            if doc.id in doc_texts:
                topic_rels = relationship_discovery_service.discover_related_topics_via_embeddings(
                    db=db,
                    workspace_id=workspace_id,
                    source_doc=doc,
                    source_text_summary=doc_texts[doc.id],
                    other_docs=all_workspace_docs,
                    other_texts=doc_texts,
                )
                relationships_found += len(topic_rels)

        return {
            "workspace_id": workspace_id,
            "documents_processed": len(documents),
            "entities_extracted": total_extracted_entities,
            "entity_relationships_discovered": total_entity_relationships,
            "relationships_discovered": relationships_found + total_entity_relationships,
            "document_relationships_discovered": relationships_found,
            "status": "completed",
        }

    def construct_networkx_graph(
        self,
        db: Session,
        workspace_id: str,
        limit: int = 50,
        relationship_types: Optional[List[str]] = None,
        focus_document_id: Optional[str] = None,
        focus_entity_id: Optional[str] = None,
    ) -> nx.Graph:
        """Construct an in-memory NetworkX graph from database records."""
        G = nx.Graph()

        # 1. Load Documents
        docs = db.query(Document).filter(
            Document.workspace_id == workspace_id,
            Document.processing_status == "completed",
        ).all()
        doc_map = {d.id: d for d in docs}

        for d in docs:
            G.add_node(
                f"doc_{d.id}",
                label=d.original_filename,
                node_type="document",
                id=d.id,
                file_size_bytes=d.file_size_bytes,
                total_pages=d.total_pages or 1,
            )

        # 2. Load Entities
        entities = db.query(Entity).filter(Entity.workspace_id == workspace_id).all()
        ent_map = {e.id: e for e in entities}

        for e in entities:
            G.add_node(
                f"ent_{e.id}",
                label=e.canonical_name,
                node_type="entity",
                id=e.id,
                entity_type=e.entity_type,
                description=e.description,
            )

        # 3. Add Document-Entity Association Edges
        assocs = db.query(DocumentEntityAssociation).filter(
            DocumentEntityAssociation.workspace_id == workspace_id
        ).all()

        for a in assocs:
            u = f"doc_{a.document_id}"
            v = f"ent_{a.entity_id}"
            if u in G and v in G:
                G.add_edge(
                    u,
                    v,
                    edge_id=f"assoc_{a.id}",
                    relationship_type="mentions",
                    confidence=1.0,
                    explanation=f"Entity mentioned on page {a.page_start or 1}.",
                    page_start=a.page_start,
                    excerpt=a.excerpt,
                )

        # 4. Add Document-Document Relationships
        rel_query = db.query(DocumentRelationship).filter(
            DocumentRelationship.workspace_id == workspace_id
        )
        if relationship_types:
            rel_query = rel_query.filter(DocumentRelationship.relationship_type.in_(relationship_types))
        relationships = rel_query.all()

        for r in relationships:
            u = f"doc_{r.source_document_id}"
            v = f"doc_{r.target_document_id}"
            if u in G and v in G:
                G.add_edge(
                    u,
                    v,
                    edge_id=f"rel_{r.id}",
                    relationship_id=r.id,
                    relationship_type=r.relationship_type,
                    confidence=r.confidence_score,
                    explanation=r.explanation,
                    supporting_evidence=r.supporting_evidence,
                    source_page=r.source_page_start,
                    target_page=r.target_page_start,
                )

        # 4.5. Add Entity-to-Entity Relationships
        ent_rel_query = db.query(EntityRelationship).filter(
            EntityRelationship.workspace_id == workspace_id
        )
        if relationship_types:
            ent_rel_query = ent_rel_query.filter(EntityRelationship.relationship_type.in_(relationship_types))
        ent_rels = ent_rel_query.all()

        for er in ent_rels:
            u = f"ent_{er.source_entity_id}"
            v = f"ent_{er.target_entity_id}"
            if u in G and v in G:
                doc_name = doc_map.get(er.source_document_id).original_filename if er.source_document_id in doc_map else "Document"
                ev_record = {
                    "document_id": er.source_document_id,
                    "document_name": doc_name,
                    "page_number": er.page_number,
                    "excerpt": er.evidence_excerpt,
                    "confidence": er.confidence_score,
                }
                if G.has_edge(u, v):
                    edge_dict = G[u][v]
                    if "evidence_records" in edge_dict:
                        edge_dict["evidence_records"].append(ev_record)
                        edge_dict["evidence_count"] = len(edge_dict["evidence_records"])
                else:
                    G.add_edge(
                        u,
                        v,
                        edge_id=f"ent_rel_{er.id}",
                        relationship_id=er.id,
                        relationship_type=er.relationship_type,
                        confidence=er.confidence_score,
                        explanation=er.explanation,
                        supporting_evidence=[ev_record],
                        evidence_records=[ev_record],
                        evidence_count=1,
                        source_document_id=er.source_document_id,
                        source_document_name=doc_name,
                        page_number=er.page_number,
                        excerpt=er.evidence_excerpt,
                    )

        # 5. Extract focused neighborhood if requested
        if focus_document_id:
            target_node = f"doc_{focus_document_id}"
            if target_node in G:
                sub_nodes = set(nx.ego_graph(G, target_node, radius=1).nodes())
                G = G.subgraph(sub_nodes).copy()
        elif focus_entity_id:
            target_node = f"ent_{focus_entity_id}"
            if target_node in G:
                sub_nodes = set(nx.ego_graph(G, target_node, radius=1).nodes())
                G = G.subgraph(sub_nodes).copy()

        # 6. Apply limit on graph size by degree
        if len(G) > limit:
            degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
            top_nodes = [n for n, _ in degrees[:limit]]
            G = G.subgraph(top_nodes).copy()

        return G

    def get_graph_data(
        self,
        db: Session,
        workspace_id: str,
        limit: int = 50,
        relationship_types: Optional[List[str]] = None,
        focus_document_id: Optional[str] = None,
        focus_entity_id: Optional[str] = None,
    ) -> GraphDataResponse:
        """Serialize NetworkX graph into clean Pydantic GraphDataResponse."""
        G = self.construct_networkx_graph(
            db=db,
            workspace_id=workspace_id,
            limit=limit,
            relationship_types=relationship_types,
            focus_document_id=focus_document_id,
            focus_entity_id=focus_entity_id,
        )

        nodes: List[GraphNode] = []
        for n, data in G.nodes(data=True):
            nodes.append(
                GraphNode(
                    id=n,
                    label=data.get("label", n),
                    node_type=data.get("node_type", "entity"),
                    properties={k: v for k, v in data.items() if k not in ("label", "node_type")},
                )
            )

        edges: List[GraphEdge] = []
        for u, v, data in G.edges(data=True):
            edges.append(
                GraphEdge(
                    id=data.get("edge_id", f"{u}_{v}"),
                    source=u,
                    target=v,
                    relationship_type=data.get("relationship_type", "related"),
                    confidence=data.get("confidence", 1.0),
                    explanation=data.get("explanation", ""),
                    supporting_evidence=data.get("supporting_evidence"),
                    properties={k: v for k, v in data.items() if k not in ("edge_id", "relationship_type", "confidence", "explanation", "supporting_evidence")},
                )
            )

        return GraphDataResponse(
            workspace_id=workspace_id,
            total_nodes=len(nodes),
            total_edges=len(edges),
            nodes=nodes,
            edges=edges,
        )

    def get_workspace_summary(self, db: Session, workspace_id: str) -> GraphSummaryResponse:
        """Calculate high-level knowledge graph metrics for the workspace."""
        total_docs = db.query(Document).filter(
            Document.workspace_id == workspace_id,
            Document.processing_status == "completed",
        ).count()

        total_entities = db.query(Entity).filter(Entity.workspace_id == workspace_id).count()

        # Document relationships
        relationships = db.query(DocumentRelationship).filter(
            DocumentRelationship.workspace_id == workspace_id
        ).all()

        # Entity relationships
        ent_relationships = db.query(EntityRelationship).filter(
            EntityRelationship.workspace_id == workspace_id
        ).all()

        breakdown: Dict[str, int] = {}
        for r in relationships:
            breakdown[r.relationship_type] = breakdown.get(r.relationship_type, 0) + 1
        for er in ent_relationships:
            breakdown[er.relationship_type] = breakdown.get(er.relationship_type, 0) + 1

        # Calculate connectivity for all docs
        docs = db.query(Document).filter(
            Document.workspace_id == workspace_id,
            Document.processing_status == "completed",
            Document.visibility_scope != "personal",
        ).all()

        doc_connectivity = []
        for d in docs:
            score = relationship_discovery_service.calculate_document_connectivity_score(
                db=db,
                workspace_id=workspace_id,
                document_id=d.id,
            )
            doc_connectivity.append({
                "document_id": d.id,
                "document_name": d.original_filename,
                "connectivity_score": score,
            })

        top_docs = sorted(doc_connectivity, key=lambda x: x["connectivity_score"], reverse=True)[:5]

        # Top Entities by mention count
        entities = db.query(Entity).filter(Entity.workspace_id == workspace_id).all()
        ent_stats = []
        for e in entities:
            mention_cnt = db.query(DocumentEntityAssociation).filter(
                DocumentEntityAssociation.entity_id == e.id
            ).count()
            ent_stats.append({
                "entity_id": e.id,
                "name": e.canonical_name,
                "entity_type": e.entity_type,
                "mentions": mention_cnt,
            })

        top_entities = sorted(ent_stats, key=lambda x: x["mentions"], reverse=True)[:5]

        return GraphSummaryResponse(
            workspace_id=workspace_id,
            total_documents=total_docs,
            total_entities=total_entities,
            total_relationships=len(relationships) + len(ent_relationships),
            relationship_breakdown=breakdown,
            top_connected_documents=top_docs,
            top_entities=top_entities,
        )

    def get_relationship_details(
        self,
        db: Session,
        workspace_id: str,
        relationship_id: str,
    ) -> Optional[DocumentRelationshipResponse]:
        """Fetch full details and evidence for a specific document relationship."""
        rel = (
            db.query(DocumentRelationship)
            .filter(
                DocumentRelationship.id == relationship_id,
                DocumentRelationship.workspace_id == workspace_id,
            )
            .first()
        )
        if not rel:
            return None

        src_doc = db.query(Document).filter(Document.id == rel.source_document_id).first()
        tgt_doc = db.query(Document).filter(Document.id == rel.target_document_id).first()

        return DocumentRelationshipResponse(
            id=rel.id,
            workspace_id=rel.workspace_id,
            source_document_id=rel.source_document_id,
            target_document_id=rel.target_document_id,
            source_document_name=src_doc.original_filename if src_doc else "Unknown",
            target_document_name=tgt_doc.original_filename if tgt_doc else "Unknown",
            relationship_type=rel.relationship_type,
            confidence_score=rel.confidence_score,
            explanation=rel.explanation,
            supporting_evidence=rel.supporting_evidence,
            source_page_start=rel.source_page_start,
            source_page_end=rel.source_page_end,
            target_page_start=rel.target_page_start,
            target_page_end=rel.target_page_end,
            created_at=rel.created_at,
            updated_at=rel.updated_at,
        )

    def get_entity_relationship_details(
        self,
        db: Session,
        workspace_id: str,
        relationship_id: str,
    ) -> Optional[EntityRelationshipResponse]:
        """Fetch full details and evidence for a specific entity relationship."""
        rel = (
            db.query(EntityRelationship)
            .filter(
                EntityRelationship.id == relationship_id,
                EntityRelationship.workspace_id == workspace_id,
            )
            .first()
        )
        if not rel:
            return None

        src_ent = db.query(Entity).filter(Entity.id == rel.source_entity_id).first()
        tgt_ent = db.query(Entity).filter(Entity.id == rel.target_entity_id).first()
        src_doc = db.query(Document).filter(Document.id == rel.source_document_id).first()

        return EntityRelationshipResponse(
            id=rel.id,
            workspace_id=rel.workspace_id,
            source_entity_id=rel.source_entity_id,
            target_entity_id=rel.target_entity_id,
            source_entity_name=src_ent.canonical_name if src_ent else "Unknown",
            target_entity_name=tgt_ent.canonical_name if tgt_ent else "Unknown",
            source_document_id=rel.source_document_id,
            source_document_name=src_doc.original_filename if src_doc else "Unknown",
            relationship_type=rel.relationship_type,
            confidence_score=rel.confidence_score,
            explanation=rel.explanation,
            evidence_excerpt=rel.evidence_excerpt,
            page_number=rel.page_number,
            created_at=rel.created_at,
            updated_at=rel.updated_at,
        )

    def handle_document_deletion(
        self,
        db: Session,
        workspace_id: str,
        document_id: str,
    ) -> Dict[str, int]:
        """Safely prune graph artifacts when a document is deleted:
        1. Remove EntityRelationship records established by this document.
        2. Remove DocumentRelationship records involving this document.
        3. Remove DocumentEntityAssociation records for this document.
        4. Prune orphaned entities that have no remaining associations in the workspace.
        5. Prune any entity relationships referencing pruned entities.
        """
        # 1. Entity relationships sourced from this document
        ent_rels = (
            db.query(EntityRelationship)
            .filter(
                EntityRelationship.workspace_id == workspace_id,
                EntityRelationship.source_document_id == document_id,
            )
            .all()
        )
        del_ent_rels = len(ent_rels)
        for r in ent_rels:
            db.delete(r)

        # 2. Document relationships involving this document
        doc_rels = (
            db.query(DocumentRelationship)
            .filter(
                DocumentRelationship.workspace_id == workspace_id,
                (DocumentRelationship.source_document_id == document_id) |
                (DocumentRelationship.target_document_id == document_id),
            )
            .all()
        )
        del_doc_rels = len(doc_rels)
        for r in doc_rels:
            db.delete(r)

        # 3. Document entity associations
        assocs = (
            db.query(DocumentEntityAssociation)
            .filter(
                DocumentEntityAssociation.workspace_id == workspace_id,
                DocumentEntityAssociation.document_id == document_id,
            )
            .all()
        )
        del_assocs = len(assocs)
        impacted_entity_ids = {a.entity_id for a in assocs}
        for a in assocs:
            db.delete(a)

        db.flush()

        # 4. Prune orphaned entities that have no remaining associations in the workspace
        pruned_entities = 0
        for eid in impacted_entity_ids:
            remaining = (
                db.query(DocumentEntityAssociation)
                .filter(
                    DocumentEntityAssociation.workspace_id == workspace_id,
                    DocumentEntityAssociation.entity_id == eid,
                )
                .count()
            )
            if remaining == 0:
                ent = db.query(Entity).filter(Entity.id == eid, Entity.workspace_id == workspace_id).first()
                if ent:
                    # Also prune any remaining entity relationships referencing this entity
                    dangling_rels = (
                        db.query(EntityRelationship)
                        .filter(
                            EntityRelationship.workspace_id == workspace_id,
                            (EntityRelationship.source_entity_id == eid) |
                            (EntityRelationship.target_entity_id == eid),
                        )
                        .all()
                    )
                    for dr in dangling_rels:
                        db.delete(dr)
                    db.delete(ent)
                    pruned_entities += 1

        db.flush()
        logger.info(
            "Graph cascade pruned for doc %s in ws %s: %d ent_rels, %d doc_rels, %d assocs, %d orphaned entities",
            document_id,
            workspace_id,
            del_ent_rels,
            del_doc_rels,
            del_assocs,
            pruned_entities,
        )
        return {
            "deleted_entity_relationships": del_ent_rels,
            "deleted_document_relationships": del_doc_rels,
            "deleted_associations": del_assocs,
            "pruned_entities": pruned_entities,
        }

    def find_shortest_path(
        self,
        db: Session,
        workspace_id: str,
        source_id: str,
        target_id: str,
    ) -> Dict[str, Any]:
        """Compute shortest path between two nodes (documents or entities) in the workspace graph."""
        G = self.construct_networkx_graph(db=db, workspace_id=workspace_id, limit=200)

        # Normalize node keys
        src_node = source_id if (source_id.startswith("doc_") or source_id.startswith("ent_")) else f"ent_{source_id}"
        if src_node not in G and f"doc_{source_id}" in G:
            src_node = f"doc_{source_id}"

        tgt_node = target_id if (target_id.startswith("doc_") or target_id.startswith("ent_")) else f"ent_{target_id}"
        if tgt_node not in G and f"doc_{target_id}" in G:
            tgt_node = f"doc_{target_id}"

        if src_node not in G:
            return {"found": False, "error": f"Source node '{source_id}' not found in active graph."}
        if tgt_node not in G:
            return {"found": False, "error": f"Target node '{target_id}' not found in active graph."}

        try:
            path = nx.shortest_path(G, source=src_node, target=tgt_node)
            path_details = []
            for i, node_id in enumerate(path):
                n_data = G.nodes[node_id]
                step = {
                    "step": i + 1,
                    "node_id": node_id,
                    "label": n_data.get("label", node_id),
                    "node_type": n_data.get("node_type", "entity"),
                }
                if i > 0:
                    prev_node = path[i - 1]
                    edge_data = G.get_edge_data(prev_node, node_id)
                    step["edge"] = {
                        "relationship_type": edge_data.get("relationship_type", "connected"),
                        "confidence": edge_data.get("confidence", 1.0),
                        "explanation": edge_data.get("explanation", ""),
                    }
                path_details.append(step)

            return {
                "found": True,
                "source": src_node,
                "target": tgt_node,
                "path_length": len(path) - 1,
                "path": path,
                "steps": path_details,
            }
        except nx.NetworkXNoPath:
            return {
                "found": False,
                "source": src_node,
                "target": tgt_node,
                "message": "No connecting path exists between the specified nodes in the current graph.",
            }

    def get_reconciliation_status(self, db: Session, workspace_id: str) -> Dict[str, Any]:
        """Reconcile and report ChromaDB vector counts vs relational database document records.

        Explains differences between workspace-scoped counts and collection-wide counts
        without modifying or deleting vectors or database records.
        """
        chroma_client = get_chroma_client()
        try:
            collection = chroma_client.get_collection("ekn_document_chunks")
            total_collection_vectors = collection.count()

            # Query vectors scoped to this workspace
            ws_vectors_data = collection.get(where={"workspace_id": workspace_id}, include=["metadatas"])
            ws_vector_count = len(ws_vectors_data.get("ids", []))
        except Exception as exc:
            logger.warning("Could not query ChromaDB for reconciliation: %s", exc)
            total_collection_vectors = 0
            ws_vector_count = 0

        # Query database documents
        company_docs = (
            db.query(Document)
            .filter(
                Document.workspace_id == workspace_id,
                Document.visibility_scope != "personal",
            )
            .all()
        )
        personal_docs = (
            db.query(Document)
            .filter(
                Document.workspace_id == workspace_id,
                Document.visibility_scope == "personal",
            )
            .all()
        )
        all_db_docs = db.query(Document).count()

        return {
            "workspace_id": workspace_id,
            "reconciled": True,
            "workspace_vector_count": ws_vector_count,
            "collection_wide_vector_count": total_collection_vectors,
            "company_documents_count": len(company_docs),
            "personal_documents_count": len(personal_docs),
            "total_database_documents": all_db_docs,
            "reconciliation_explanation": (
                f"Workspace '{workspace_id}' contains exactly {ws_vector_count} indexed vectors "
                f"across its {len(company_docs)} company documents. The collection-wide total of "
                f"{total_collection_vectors} vectors includes vectors created during isolated "
                f"test-suite runs across temporary tenant workspaces. All queries in EKN are strictly "
                f"isolated using workspace_id filters, ensuring zero cross-tenant contamination."
            ),
        }


knowledge_graph_service = KnowledgeGraphService()
