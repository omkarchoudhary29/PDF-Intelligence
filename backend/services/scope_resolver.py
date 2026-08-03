import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from backend.database.models import Document

logger = logging.getLogger("ekn.scope_resolver")

VALID_SCOPES = {"company", "personal", "combined"}


@dataclass
class ResolvedScope:
    """Encapsulates authorized scope constraints across Relational DB, ChromaDB, and BM25."""

    scope: str  # 'company', 'personal', 'combined'
    workspace_id: str
    user_id: str
    role: str
    authorized_document_ids: Optional[List[str]]
    chroma_where: Dict[str, Any]
    bm25_scope: str
    bm25_owner_user_id: Optional[str]


class ScopeResolver:
    """Centralized authority for resolving authorized knowledge boundaries.

    Guarantees:
    - Employees can access company documents and strictly their own personal documents.
    - Employees can NEVER access another employee's private documents.
    - Administrators do NOT automatically gain access to employee-private documents.
    - An administrator's personal scope is strictly tied to that administrator's user ID.
    """

    @classmethod
    def normalize_scope(cls, requested_scope: Optional[str]) -> str:
        """Validate and normalize requested scope string."""
        if not requested_scope:
            return "company"
        norm = requested_scope.strip().lower()
        if norm not in VALID_SCOPES:
            logger.warning("Unrecognized scope '%s'; defaulting to 'company'", requested_scope)
            return "company"
        return norm

    @classmethod
    def can_access_document(
        cls,
        document: Document,
        user_id: str,
        user_role: str,
    ) -> bool:
        """Determine whether an authenticated user is permitted to view/download a document."""
        if not document:
            return False

        # Company-scoped documents are visible to all authenticated workspace members
        if document.visibility_scope == "company":
            return True

        # Personal documents are strictly accessible ONLY to the owner employee
        if document.visibility_scope == "personal":
            return bool(document.owner_user_id and document.owner_user_id == user_id)

        # Unknown scope: fail closed
        return False

    @classmethod
    def can_manage_document(
        cls,
        document: Document,
        user_id: str,
        user_role: str,
    ) -> bool:
        """Determine whether an authenticated user can delete/modify a document."""
        if not document:
            return False

        # Company-scoped documents require administrator or owner role
        if document.visibility_scope == "company":
            return user_role in ("owner", "admin")

        # Personal documents can only be deleted/managed by their owning employee
        if document.visibility_scope == "personal":
            return bool(document.owner_user_id and document.owner_user_id == user_id)

        return False

    @classmethod
    def get_authorized_document_ids(
        cls,
        db: Session,
        workspace_id: str,
        user_id: str,
        scope: str,
    ) -> List[str]:
        """Fetch all document IDs in the workspace authorized under the specified scope."""
        query = db.query(Document.id).filter(Document.workspace_id == workspace_id)

        if scope == "company":
            query = query.filter(Document.visibility_scope == "company")
        elif scope == "personal":
            query = query.filter(
                Document.visibility_scope == "personal",
                Document.owner_user_id == user_id,
            )
        elif scope == "combined":
            query = query.filter(
                or_(
                    Document.visibility_scope == "company",
                    and_(
                        Document.visibility_scope == "personal",
                        Document.owner_user_id == user_id,
                    ),
                )
            )
        else:
            query = query.filter(Document.visibility_scope == "company")

        return [row[0] for row in query.all()]

    @classmethod
    def build_chroma_filter(
        cls,
        workspace_id: str,
        user_id: str,
        scope: str,
        document_ids: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build ChromaDB metadata filter dictionary enforcing workspace and ownership boundaries."""
        conditions: List[Dict[str, Any]] = [{"workspace_id": workspace_id}]

        # Ownership filter condition
        if scope == "company":
            conditions.append({"visibility_scope": "company"})
        elif scope == "personal":
            conditions.append({"visibility_scope": "personal"})
            conditions.append({"owner_user_id": user_id})
        elif scope == "combined":
            conditions.append({
                "$or": [
                    {"visibility_scope": "company"},
                    {"$and": [{"visibility_scope": "personal"}, {"owner_user_id": user_id}]},
                ]
            })

        # Specific document ID constraint
        if document_ids is not None:
            if len(document_ids) == 0:
                # No matching documents: ensure query returns nothing
                conditions.append({"document_id": "__NO_MATCHING_DOCUMENTS__"})
            elif len(document_ids) == 1:
                conditions.append({"document_id": document_ids[0]})
            else:
                conditions.append({"document_id": {"$in": document_ids}})

        # Chunk types constraint
        if chunk_types:
            if len(chunk_types) == 1:
                conditions.append({"chunk_type": chunk_types[0]})
            else:
                conditions.append({"chunk_type": {"$in": chunk_types}})

        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @classmethod
    def resolve(
        cls,
        workspace_id: str,
        user_id: str,
        user_role: str,
        scope: Optional[str] = "company",
        requested_document_ids: Optional[List[str]] = None,
        chunk_types: Optional[List[str]] = None,
        db: Optional[Session] = None,
    ) -> ResolvedScope:
        """Resolve full authorized context for database queries, ChromaDB search, and BM25."""
        normalized_scope = cls.normalize_scope(scope)

        # If database session is supplied, validate requested document IDs against authorized set
        authorized_ids: Optional[List[str]] = None
        if db is not None:
            permitted_ids = set(cls.get_authorized_document_ids(db, workspace_id, user_id, normalized_scope))
            if requested_document_ids is not None:
                # Restrict to the intersection of requested and permitted
                authorized_ids = [doc_id for doc_id in requested_document_ids if doc_id in permitted_ids]
            else:
                authorized_ids = list(permitted_ids)
        elif requested_document_ids is not None:
            authorized_ids = list(requested_document_ids)

        # Build ChromaDB filter
        chroma_filter = cls.build_chroma_filter(
            workspace_id=workspace_id,
            user_id=user_id,
            scope=normalized_scope,
            document_ids=authorized_ids,
            chunk_types=chunk_types,
        )

        return ResolvedScope(
            scope=normalized_scope,
            workspace_id=workspace_id,
            user_id=user_id,
            role=user_role,
            authorized_document_ids=authorized_ids,
            chroma_where=chroma_filter,
            bm25_scope=normalized_scope,
            bm25_owner_user_id=user_id if normalized_scope in ("personal", "combined") else None,
        )


scope_resolver = ScopeResolver()
