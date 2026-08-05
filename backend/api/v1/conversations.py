from datetime import datetime
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.core.dependencies import get_current_user, require_workspace_member
from backend.database.connection import get_db
from backend.database.models import (
    Conversation,
    ConversationAttachment,
    Document,
    Message,
    User,
    Workspace,
    WorkspaceMembership,
)
from backend.schemas.conversation import (
    CitationItem,
    ConversationAttachmentListResponse,
    ConversationAttachmentRequest,
    ConversationAttachmentResponse,
    ConversationCreateRequest,
    ConversationListResponse,
    ConversationResponse,
    ConversationUpdateRequest,
    MessageCreateRequest,
    MessageListResponse,
    MessageResponse,
    RAGResponse,
)
from backend.services.llm_service import llm_service
from backend.services.rag_workflow import rag_workflow_service
from backend.services.scope_resolver import scope_resolver

logger = logging.getLogger("ekn.conversations_api")

router = APIRouter(
    prefix="/api/v1/workspaces/{workspace_id}/conversations",
    tags=["Conversations & RAG"],
)


def _get_workspace_or_404(workspace_id: str, db: Session) -> Workspace:
    """Verify workspace exists."""
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workspace '{workspace_id}' was not found.",
        )
    return ws


def _get_conversation_or_404(
    workspace_id: str,
    conversation_id: str,
    db: Session,
    user: Optional[User] = None,
) -> Conversation:
    """Verify conversation exists, belongs to the workspace, and is accessible to the user."""
    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.workspace_id == workspace_id,
        )
        .first()
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' was not found in workspace '{workspace_id}'.",
        )

    # Restrict private conversations to their creator
    if user and conv.user_id and conv.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' was not found in workspace '{workspace_id}'.",
        )

    return conv


def _build_conversation_response(conv: Conversation) -> ConversationResponse:
    """Build public schema for conversation including scope and attached document IDs."""
    attached_ids = [att.document_id for att in conv.attachments] if conv.attachments else []
    return ConversationResponse(
        id=conv.id,
        workspace_id=conv.workspace_id,
        user_id=conv.user_id,
        title=conv.title,
        knowledge_scope=conv.knowledge_scope or "company",
        is_active=conv.is_active,
        attached_document_ids=attached_ids,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation session",
)
def create_conversation(
    workspace_id: str,
    request: ConversationCreateRequest,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ConversationResponse:
    """Create a new conversation session isolated within a workspace and owned by the creator."""
    _get_workspace_or_404(workspace_id, db)

    title = (request.title or "").strip() or "New Conversation"
    norm_scope = scope_resolver.normalize_scope(request.knowledge_scope)
    conversation = Conversation(
        workspace_id=workspace_id,
        user_id=current_user.id,
        title=title,
        knowledge_scope=norm_scope,
        is_active=True,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    logger.info(
        "Created conversation %s in workspace %s by user %s (scope=%s)",
        conversation.id,
        workspace_id,
        current_user.id,
        norm_scope,
    )
    return _build_conversation_response(conversation)


@router.get(
    "",
    response_model=ConversationListResponse,
    summary="List workspace conversations",
)
def list_conversations(
    workspace_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ConversationListResponse:
    """Retrieve paginated list of conversations belonging to the user in this workspace."""
    _get_workspace_or_404(workspace_id, db)

    query = (
        db.query(Conversation)
        .filter(
            Conversation.workspace_id == workspace_id,
            (Conversation.user_id == current_user.id) | (Conversation.user_id.is_(None)),
        )
        .order_by(Conversation.updated_at.desc())
    )
    total = query.count()
    items = query.offset((page - 1) * page_size).limit(page_size).all()

    return ConversationListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[_build_conversation_response(c) for c in items],
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
    summary="Retrieve a conversation by ID",
)
def get_conversation(
    workspace_id: str,
    conversation_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ConversationResponse:
    """Retrieve conversation metadata, enforcing workspace boundary and user creator privacy."""
    conv = _get_conversation_or_404(workspace_id, conversation_id, db, user=current_user)
    return _build_conversation_response(conv)


@router.patch(
    "/{conversation_id}",
    response_model=ConversationResponse,
    summary="Update conversation metadata (rename title or change knowledge scope)",
)
def update_conversation(
    workspace_id: str,
    conversation_id: str,
    request: ConversationUpdateRequest,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ConversationResponse:
    """Rename a conversation or update its knowledge scope (owner only)."""
    conv = _get_conversation_or_404(workspace_id, conversation_id, db, user=current_user)

    if request.title is not None:
        clean_title = request.title.strip()
        if clean_title:
            conv.title = clean_title

    if request.knowledge_scope is not None:
        conv.knowledge_scope = scope_resolver.normalize_scope(request.knowledge_scope)

    conv.updated_at = datetime.now()
    db.commit()
    db.refresh(conv)
    return _build_conversation_response(conv)


@router.delete(
    "/{conversation_id}",
    summary="Delete a conversation session and all its messages and attachment links",
)
def delete_conversation(
    workspace_id: str,
    conversation_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Delete a conversation owned by the authenticated employee."""
    conv = _get_conversation_or_404(workspace_id, conversation_id, db, user=current_user)

    db.delete(conv)
    db.commit()

    logger.info("Deleted conversation %s in workspace %s by user %s", conversation_id, workspace_id, current_user.id)
    return {
        "message": "Conversation deleted successfully.",
        "conversation_id": conversation_id,
        "workspace_id": workspace_id,
    }


@router.get(
    "/{conversation_id}/messages",
    response_model=MessageListResponse,
    summary="Retrieve messages within a conversation",
)
def get_conversation_messages(
    workspace_id: str,
    conversation_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MessageListResponse:
    """Retrieve all messages in a conversation in chronological order."""
    _get_conversation_or_404(workspace_id, conversation_id, db, user=current_user)

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    items = []
    for m in messages:
        citations_data = None
        if m.citations:
            citations_data = [CitationItem.model_validate(c) for c in m.citations]
        items.append(
            MessageResponse(
                id=m.id,
                conversation_id=m.conversation_id,
                role=m.role,
                content=m.content,
                citations=citations_data,
                created_at=m.created_at,
            )
        )

    return MessageListResponse(total=len(items), items=items)


@router.post(
    "/{conversation_id}/messages",
    response_model=RAGResponse,
    summary="Send a user message and generate a grounded RAG response",
)
def send_message(
    workspace_id: str,
    conversation_id: str,
    request: MessageCreateRequest,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RAGResponse:
    """Execute LangGraph conversational RAG workflow for a user message."""
    conv = _get_conversation_or_404(workspace_id, conversation_id, db, user=current_user)

    user_query = request.content.strip()
    if not user_query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message content cannot be empty.",
        )

    # 1. Persist user message
    user_message = Message(
        conversation_id=conv.id,
        role="user",
        content=user_query,
    )
    db.add(user_message)
    db.commit()

    # 2. Load recent conversation history (excluding current turn)
    prior_messages = (
        db.query(Message)
        .filter(
            Message.conversation_id == conv.id,
            Message.id != user_message.id,
        )
        .order_by(Message.created_at.asc())
        .all()
    )
    formatted_history = [{"role": m.role, "content": m.content} for m in prior_messages]

    # 3. Execute LangGraph RAG workflow with active conversation knowledge scope
    attached_doc_ids = [att.document_id for att in conv.attachments] if conv.attachments else None
    try:
        workflow_result = rag_workflow_service.run(
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            query=user_query,
            conversation_history=formatted_history,
            knowledge_scope=conv.knowledge_scope or "company",
            user_id=current_user.id,
            document_ids=attached_doc_ids,
        )
    except Exception as exc:
        logger.error("RAG workflow execution failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conversational RAG execution failed: {str(exc)}",
        )

    answer = workflow_result.get("generated_answer") or "I could not generate an answer at this time."
    raw_citations = workflow_result.get("validated_citations") or []
    citations_data = [CitationItem.model_validate(c) for c in raw_citations]

    # 4. Persist assistant message with verified citations
    assistant_message = Message(
        conversation_id=conv.id,
        role="assistant",
        content=answer,
        citations=[c.model_dump() for c in citations_data],
    )
    db.add(assistant_message)

    # Update conversation title from initial user message if still default
    if conv.title == "New Conversation":
        conv.title = user_query[:50].strip()

    conv.updated_at = datetime.now()
    db.commit()
    db.refresh(assistant_message)

    llm_status = llm_service.get_configuration_status()["status"]
    return RAGResponse(
        conversation_id=conv.id,
        message_id=assistant_message.id,
        answer=answer,
        citations=citations_data,
        evidence_sufficient=workflow_result.get("evidence_sufficient", True),
        response_type=workflow_result.get("response_type", "valid_answer"),
        query_category=workflow_result.get("query_category"),
        contextualized_query=workflow_result.get("contextualized_query"),
        validation_issues=workflow_result.get("validation_issues"),
        llm_provider_status=llm_status,
        retrieval_time_ms=workflow_result.get("retrieval_time_ms"),
        llm_time_ms=workflow_result.get("llm_time_ms"),
        total_time_ms=workflow_result.get("total_time_ms"),
    )


# -----------------------------------------------------------------------------
# Conversation Document Attachments (Phase 3)
# -----------------------------------------------------------------------------


@router.post(
    "/{conversation_id}/attachments",
    response_model=ConversationAttachmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Attach an authorized document to a conversation",
)
def attach_document_to_conversation(
    workspace_id: str,
    conversation_id: str,
    request: ConversationAttachmentRequest,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ConversationAttachmentResponse:
    """Attach an authorized document (company-wide or current user's personal) to a conversation."""
    conv = _get_conversation_or_404(workspace_id, conversation_id, db, user=current_user)

    doc = (
        db.query(Document)
        .filter(
            Document.id == request.document_id,
            Document.workspace_id == workspace_id,
        )
        .first()
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{request.document_id}' was not found in workspace '{workspace_id}'.",
        )

    # Ownership validation: Reject attachment if user cannot access the document
    if not scope_resolver.can_access_document(doc, current_user.id, membership.role):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{request.document_id}' was not found in workspace '{workspace_id}'.",
        )

    # Check for existing attachment
    existing = (
        db.query(ConversationAttachment)
        .filter(
            ConversationAttachment.conversation_id == conv.id,
            ConversationAttachment.document_id == doc.id,
        )
        .first()
    )
    if existing:
        return ConversationAttachmentResponse(
            id=existing.id,
            conversation_id=existing.conversation_id,
            document_id=existing.document_id,
            original_filename=doc.original_filename,
            visibility_scope=doc.visibility_scope,
            created_at=existing.created_at,
        )

    attachment = ConversationAttachment(
        conversation_id=conv.id,
        document_id=doc.id,
    )
    db.add(attachment)
    db.commit()
    db.refresh(attachment)

    logger.info(
        "Attached document %s (%s) to conversation %s by user %s",
        doc.id,
        doc.visibility_scope,
        conv.id,
        current_user.id,
    )
    return ConversationAttachmentResponse(
        id=attachment.id,
        conversation_id=attachment.conversation_id,
        document_id=attachment.document_id,
        original_filename=doc.original_filename,
        visibility_scope=doc.visibility_scope,
        created_at=attachment.created_at,
    )


@router.get(
    "/{conversation_id}/attachments",
    response_model=ConversationAttachmentListResponse,
    summary="List documents attached to a conversation",
)
def list_conversation_attachments(
    workspace_id: str,
    conversation_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ConversationAttachmentListResponse:
    """Retrieve all documents attached to the specified conversation."""
    conv = _get_conversation_or_404(workspace_id, conversation_id, db, user=current_user)

    attachments = (
        db.query(ConversationAttachment)
        .filter(ConversationAttachment.conversation_id == conv.id)
        .order_by(ConversationAttachment.created_at.asc())
        .all()
    )

    items = []
    for att in attachments:
        doc = db.query(Document).filter(Document.id == att.document_id).first()
        items.append(
            ConversationAttachmentResponse(
                id=att.id,
                conversation_id=att.conversation_id,
                document_id=att.document_id,
                original_filename=doc.original_filename if doc else "unknown.pdf",
                visibility_scope=doc.visibility_scope if doc else "company",
                created_at=att.created_at,
            )
        )

    return ConversationAttachmentListResponse(total=len(items), items=items)


@router.delete(
    "/{conversation_id}/attachments/{document_id}",
    summary="Remove a document attachment from a conversation",
)
def remove_conversation_attachment(
    workspace_id: str,
    conversation_id: str,
    document_id: str,
    membership: WorkspaceMembership = Depends(require_workspace_member),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Detach a document from a conversation."""
    conv = _get_conversation_or_404(workspace_id, conversation_id, db, user=current_user)

    att = (
        db.query(ConversationAttachment)
        .filter(
            ConversationAttachment.conversation_id == conv.id,
            ConversationAttachment.document_id == document_id,
        )
        .first()
    )
    if not att:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Attachment for document '{document_id}' was not found on conversation '{conversation_id}'.",
        )

    db.delete(att)
    db.commit()

    logger.info("Removed attachment %s from conversation %s", document_id, conversation_id)
    return {
        "message": "Attachment removed successfully.",
        "document_id": document_id,
        "conversation_id": conversation_id,
    }
