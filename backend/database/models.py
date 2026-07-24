import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base


class Workspace(Base):
    """Represents an isolated organizational workspace for documents and knowledge graphs."""

    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    documents: Mapped[List["Document"]] = relationship(
        "Document",
        back_populates="workspace",
        cascade="all, delete-orphan",
        order_by="desc(Document.created_at)",
    )
    conversations: Mapped[List["Conversation"]] = relationship(
        "Conversation",
        back_populates="workspace",
        cascade="all, delete-orphan",
        order_by="desc(Conversation.created_at)",
    )
    entities: Mapped[List["Entity"]] = relationship(
        "Entity",
        back_populates="workspace",
        cascade="all, delete-orphan",
    )
    document_relationships: Mapped[List["DocumentRelationship"]] = relationship(
        "DocumentRelationship",
        back_populates="workspace",
        cascade="all, delete-orphan",
    )
    entity_relationships: Mapped[List["EntityRelationship"]] = relationship(
        "EntityRelationship",
        back_populates="workspace",
        cascade="all, delete-orphan",
    )
    reports: Mapped[List["Report"]] = relationship(
        "Report",
        back_populates="workspace",
        cascade="all, delete-orphan",
        order_by="desc(Report.created_at)",
    )
    memberships: Mapped[List["WorkspaceMembership"]] = relationship(
        "WorkspaceMembership",
        back_populates="workspace",
        cascade="all, delete-orphan",
    )
    document_families: Mapped[List["DocumentFamily"]] = relationship(
        "DocumentFamily",
        back_populates="workspace",
        cascade="all, delete-orphan",
        order_by="desc(DocumentFamily.created_at)",
    )
    contradiction_findings: Mapped[List["ContradictionFinding"]] = relationship(
        "ContradictionFinding",
        back_populates="workspace",
        cascade="all, delete-orphan",
        order_by="desc(ContradictionFinding.created_at)",
    )
    invitations: Mapped[List["WorkspaceInvitation"]] = relationship(
        "WorkspaceInvitation",
        back_populates="workspace",
        cascade="all, delete-orphan",
        order_by="desc(WorkspaceInvitation.created_at)",
    )

    def __repr__(self) -> str:
        return f"<Workspace(id='{self.id}', name='{self.name}', slug='{self.slug}')>"


class Document(Base):
    """Represents an uploaded original PDF document within a workspace."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(500), unique=True, nullable=False)
    sha256_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False, default="application/pdf")
    total_pages: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    extracted_pages_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary_status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # 'completed', 'partial', 'unavailable'
    uploaded_by_user_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Document Ownership and Privacy Scoping (Phase 3)
    visibility_scope: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="company",
        index=True,
    )  # 'company' or 'personal'
    owner_user_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    content_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    # Document Version Intelligence & Lifecycle
    family_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("document_families.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    previous_version_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    version_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="current",
        index=True,
    )  # 'current', 'superseded', 'archived'
    approval_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="unverified",
        index=True,
    )  # 'unverified', 'pending_review', 'approved', 'rejected'
    issued_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    effective_from: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    effective_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Lifecycle status: 'uploaded', 'queued', 'processing', 'completed', 'failed', 'deleted'
    processing_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="uploaded",
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Database-level uniqueness: identical file content cannot be duplicated within the same workspace
    __table_args__ = (
        UniqueConstraint("workspace_id", "sha256_hash", name="uq_workspace_sha256"),
        Index("ix_documents_workspace_status", "workspace_id", "processing_status"),
        Index("ix_documents_ws_scope_owner", "workspace_id", "visibility_scope", "owner_user_id"),
    )

    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="documents")
    owner_user: Mapped[Optional["User"]] = relationship("User", foreign_keys=[owner_user_id])
    conversation_attachments: Mapped[List["ConversationAttachment"]] = relationship(
        "ConversationAttachment",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    ingestion_jobs: Mapped[List["IngestionJob"]] = relationship(
        "IngestionJob",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="desc(IngestionJob.created_at)",
    )
    document_index: Mapped[Optional["DocumentIndex"]] = relationship(
        "DocumentIndex",
        back_populates="document",
        uselist=False,
        cascade="all, delete-orphan",
    )
    entity_associations: Mapped[List["DocumentEntityAssociation"]] = relationship(
        "DocumentEntityAssociation",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    source_relationships: Mapped[List["DocumentRelationship"]] = relationship(
        "DocumentRelationship",
        foreign_keys="DocumentRelationship.source_document_id",
        back_populates="source_document",
        cascade="all, delete-orphan",
    )
    target_relationships: Mapped[List["DocumentRelationship"]] = relationship(
        "DocumentRelationship",
        foreign_keys="DocumentRelationship.target_document_id",
        back_populates="target_document",
        cascade="all, delete-orphan",
    )
    family: Mapped[Optional["DocumentFamily"]] = relationship(
        "DocumentFamily",
        foreign_keys="Document.family_id",
        back_populates="versions",
    )
    previous_version: Mapped[Optional["Document"]] = relationship(
        "Document",
        remote_side="Document.id",
        foreign_keys=[previous_version_id],
    )

    def __repr__(self) -> str:
        return f"<Document(id='{self.id}', filename='{self.original_filename}', status='{self.processing_status}')>"


class IngestionJob(Base):
    """Tracks document processing jobs through the ingestion lifecycle."""

    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="queued",
        index=True,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="ingestion_jobs")

    def __repr__(self) -> str:
        return f"<IngestionJob(id='{self.id}', document_id='{self.document_id}', status='{self.status}')>"


class DocumentIndex(Base):
    """Tracks hierarchical chunking, embedding generation, and vector index state for a document."""

    __tablename__ = "document_indices"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    # Lifecycle status: 'pending', 'indexing', 'completed', 'failed'
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
        index=True,
    )
    total_parents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_children: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(150), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, default=384, nullable=False)
    parent_storage_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    indexed_file_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="document_index")

    def __repr__(self) -> str:
        return f"<DocumentIndex(document_id='{self.document_id}', status='{self.status}', children={self.total_children})>"


class Conversation(Base):
    """Represents a conversational session within a workspace."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="New Conversation")
    knowledge_scope: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="company",
        index=True,
    )  # 'company', 'personal', 'combined'
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="conversations")
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at.asc()",
    )
    attachments: Mapped[List["ConversationAttachment"]] = relationship(
        "ConversationAttachment",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationAttachment.created_at.asc()",
    )

    def __repr__(self) -> str:
        return f"<Conversation(id='{self.id}', title='{self.title}', scope='{self.knowledge_scope}', workspace_id='{self.workspace_id}')>"


class ConversationAttachment(Base):
    """Represents an association between a conversation and an authorized document."""

    __tablename__ = "conversation_attachments"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("conversation_id", "document_id", name="uq_conv_doc_attachment"),
        Index("ix_conv_attach_conv", "conversation_id"),
        Index("ix_conv_attach_doc", "document_id"),
    )

    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="attachments")
    document: Mapped["Document"] = relationship("Document", back_populates="conversation_attachments")

    def __repr__(self) -> str:
        return f"<ConversationAttachment(conversation_id='{self.conversation_id}', document_id='{self.document_id}')>"


class Message(Base):
    """Represents an individual user query or assistant response within a conversation."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # 'user' or 'assistant'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)  # List of CitationItem dicts

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    # Relationships
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message(id='{self.id}', role='{self.role}', conversation_id='{self.conversation_id}')>"


class Entity(Base):
    """Represents a named concept, equipment, organization, policy, or technical item extracted from documents."""

    __tablename__ = "entities"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    aliases_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    attributes_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("workspace_id", "normalized_name", "entity_type", name="uq_workspace_entity_type"),
        Index("ix_entities_workspace_normalized", "workspace_id", "normalized_name"),
    )

    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="entities")
    document_associations: Mapped[List["DocumentEntityAssociation"]] = relationship(
        "DocumentEntityAssociation",
        back_populates="entity",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Entity(id='{self.id}', name='{self.canonical_name}', type='{self.entity_type}')>"


class DocumentEntityAssociation(Base):
    """Grounds an entity occurrence inside a specific document page and section."""

    __tablename__ = "document_entity_associations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    entity_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    section_title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    page_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    page_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    excerpt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    mention_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    extraction_method: Mapped[str] = mapped_column(String(50), default="llm", nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_doc_entity_assoc", "workspace_id", "document_id", "entity_id"),
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="entity_associations")
    entity: Mapped["Entity"] = relationship("Entity", back_populates="document_associations")

    def __repr__(self) -> str:
        return f"<DocumentEntityAssociation(doc='{self.document_id}', entity='{self.entity_id}', page={self.page_start})>"


class DocumentRelationship(Base):
    """Represents a discovered relationship between two documents with explicit supporting evidence."""

    __tablename__ = "document_relationships"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Relationship type: 'shared_entity', 'direct_reference', 'related_topic', 'explicit_dependency'
    relationship_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    confidence_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    supporting_evidence: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    source_page_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source_page_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    target_page_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    target_page_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "source_document_id",
            "target_document_id",
            "relationship_type",
            name="uq_doc_relationship",
        ),
        Index("ix_doc_rel_ws_type", "workspace_id", "relationship_type"),
    )

    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="document_relationships")
    source_document: Mapped["Document"] = relationship(
        "Document",
        foreign_keys=[source_document_id],
        back_populates="source_relationships",
    )
    target_document: Mapped["Document"] = relationship(
        "Document",
        foreign_keys=[target_document_id],
        back_populates="target_relationships",
    )

    def __repr__(self) -> str:
        return f"<DocumentRelationship(src='{self.source_document_id}', tgt='{self.target_document_id}', type='{self.relationship_type}')>"


class EntityRelationship(Base):
    """Represents an evidence-backed directed relationship between two entities."""

    __tablename__ = "entity_relationships"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_entity_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_entity_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    relationship_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    confidence_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_excerpt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_entity_rel_src_tgt", "workspace_id", "source_entity_id", "target_entity_id"),
    )

    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="entity_relationships")
    source_entity: Mapped["Entity"] = relationship("Entity", foreign_keys=[source_entity_id])
    target_entity: Mapped["Entity"] = relationship("Entity", foreign_keys=[target_entity_id])
    document: Mapped["Document"] = relationship("Document")

    def __repr__(self) -> str:
        return f"<EntityRelationship(src='{self.source_entity_id}', tgt='{self.target_entity_id}', type='{self.relationship_type}')>"


class Report(Base):
    """Represents an automated Enterprise Knowledge Dossier PDF generation record."""

    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    generated_by_user_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    report_name: Mapped[str] = mapped_column(String(255), nullable=False)
    report_type: Mapped[str] = mapped_column(String(50), default="knowledge_dossier", nullable=False)
    status: Mapped[str] = mapped_column(
        String(50),
        default="queued",
        nullable=False,
        index=True,
    )  # 'queued', 'generating', 'completed', 'partially_completed', 'failed'
    storage_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    generation_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    generation_completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    document_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_pages_covered: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_reports_workspace_status", "workspace_id", "status"),
    )

    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="reports")

    def __repr__(self) -> str:
        return f"<Report(id='{self.id}', name='{self.report_name}', status='{self.status}')>"


class User(Base):
    """Represents an authenticated enterprise user."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    workspace_memberships: Mapped[List["WorkspaceMembership"]] = relationship(
        "WorkspaceMembership",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    refresh_sessions: Mapped[List["RefreshSession"]] = relationship(
        "RefreshSession",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<User(id='{self.id}', email='{self.email}', is_active={self.is_active})>"


class WorkspaceMembership(Base):
    """Represents an authorized user's role and membership within a workspace."""

    __tablename__ = "workspace_memberships"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # 'owner', 'admin', 'employee'
    membership_status: Mapped[str] = mapped_column(String(50), default="active", nullable=False)  # 'active', 'suspended', 'revoked'

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("user_id", "workspace_id", name="uq_user_workspace_membership"),
        Index("ix_membership_ws_role", "workspace_id", "role"),
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="workspace_memberships")
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="memberships")

    def __repr__(self) -> str:
        return f"<WorkspaceMembership(user='{self.user_id}', ws='{self.workspace_id}', role='{self.role}')>"


class WorkspaceInvitation(Base):
    """Represents a secure invitation / joining code for employee workspace onboarding."""

    __tablename__ = "workspace_invitations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    code_hash: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )
    created_by_user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    intended_role: Mapped[str] = mapped_column(
        String(50),
        default="employee",
        nullable=False,
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    max_usage_count: Mapped[int] = mapped_column(
        Integer,
        default=10,
        nullable=False,
    )
    current_usage_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    is_revoked: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_invitations_ws_active", "workspace_id", "is_revoked"),
    )

    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="invitations")
    created_by: Mapped["User"] = relationship("User")

    def __repr__(self) -> str:
        return f"<WorkspaceInvitation(id='{self.id}', ws='{self.workspace_id}', role='{self.intended_role}', revoked={self.is_revoked})>"


class RefreshSession(Base):
    """Secure database-backed storage for user refresh tokens."""

    __tablename__ = "refresh_sessions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    replaced_by_session_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="refresh_sessions")

    def __repr__(self) -> str:
        return f"<RefreshSession(id='{self.id}', user='{self.user_id}', revoked={self.revoked_at is not None})>"


class SecurityAuditLog(Base):
    """Tracks security-relevant actions for compliance and enterprise auditability."""

    __tablename__ = "security_audit_logs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    actor_user_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    workspace_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("workspaces.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    resource_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    resource_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    outcome: Mapped[str] = mapped_column(String(50), nullable=False)  # 'success', 'failure', 'denied'
    ip_address: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    details_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    __table_args__ = (
        Index("ix_audit_ws_action", "workspace_id", "action"),
    )

    def __repr__(self) -> str:
        return f"<SecurityAuditLog(action='{self.action}', outcome='{self.outcome}', actor='{self.actor_user_id}')>"


class DocumentFamily(Base):
    """Represents a logical family of iterative document revisions within a workspace."""

    __tablename__ = "document_families"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    current_document_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("workspace_id", "canonical_name", name="uq_workspace_family_canonical"),
        Index("ix_document_families_workspace_title", "workspace_id", "title"),
    )

    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="document_families")
    versions: Mapped[List["Document"]] = relationship(
        "Document",
        foreign_keys="Document.family_id",
        back_populates="family",
        cascade="all, delete-orphan",
        order_by="Document.version_number",
    )
    current_document: Mapped[Optional["Document"]] = relationship(
        "Document",
        foreign_keys=[current_document_id],
        post_update=True,
    )
    comparisons: Mapped[List["VersionComparison"]] = relationship(
        "VersionComparison",
        back_populates="family",
        cascade="all, delete-orphan",
        order_by="desc(VersionComparison.created_at)",
    )

    def __repr__(self) -> str:
        return f"<DocumentFamily(id='{self.id}', title='{self.title}', current_doc='{self.current_document_id}')>"


class VersionComparison(Base):
    """Stores deterministic page-level and text-level diff results between two document versions."""

    __tablename__ = "version_comparisons"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    family_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("document_families.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    target_version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    comparison_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    change_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ai_explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("family_id", "source_document_id", "target_document_id", name="uq_version_comparison_pair"),
    )

    # Relationships
    family: Mapped["DocumentFamily"] = relationship("DocumentFamily", back_populates="comparisons")
    source_document: Mapped["Document"] = relationship("Document", foreign_keys=[source_document_id])
    target_document: Mapped["Document"] = relationship("Document", foreign_keys=[target_document_id])

    def __repr__(self) -> str:
        return f"<VersionComparison(id='{self.id}', family='{self.family_id}', v{self.source_version_number}->v{self.target_version_number})>"


class ContradictionFinding(Base):
    """Represents a potential or confirmed factual contradiction across documents or versions."""

    __tablename__ = "contradiction_findings"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    subject: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    predicate: Mapped[str] = mapped_column(String(255), nullable=False)
    finding_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )  # 'numerical_discrepancy', 'instruction_conflict', 'policy_conflict', 'dated_revision', 'context_dependent_difference'
    explanation: Mapped[str] = mapped_column(Text, nullable=False)

    first_document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    first_page: Mapped[int] = mapped_column(Integer, nullable=False)
    first_evidence_excerpt: Mapped[str] = mapped_column(Text, nullable=False)

    second_document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    second_page: Mapped[int] = mapped_column(Integer, nullable=False)
    second_evidence_excerpt: Mapped[str] = mapped_column(Text, nullable=False)

    confidence_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    review_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="candidate",
        index=True,
    )  # 'candidate', 'confirmed', 'dismissed', 'resolved'

    reviewer_user_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    review_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    authoritative_document_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_contradiction_workspace_status", "workspace_id", "review_status"),
        Index("ix_contradiction_pair", "first_document_id", "second_document_id"),
    )

    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="contradiction_findings")
    first_document: Mapped["Document"] = relationship("Document", foreign_keys=[first_document_id])
    second_document: Mapped["Document"] = relationship("Document", foreign_keys=[second_document_id])
    authoritative_document: Mapped[Optional["Document"]] = relationship("Document", foreign_keys=[authoritative_document_id])
    reviewer: Mapped[Optional["User"]] = relationship("User", foreign_keys=[reviewer_user_id])

    def __repr__(self) -> str:
        return f"<ContradictionFinding(id='{self.id}', type='{self.finding_type}', status='{self.review_status}')>"





