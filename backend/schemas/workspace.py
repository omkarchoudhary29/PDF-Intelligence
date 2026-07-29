import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def slugify(text: str) -> str:
    """Generate a clean URL-friendly slug from text."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")


class WorkspaceBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Workspace display name")
    description: Optional[str] = Field(None, max_length=1000, description="Workspace description")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Workspace name cannot be empty or purely whitespace.")
        return cleaned


class WorkspaceCreate(WorkspaceBase):
    slug: Optional[str] = Field(None, max_length=100, description="Custom slug; generated from name if omitted")

    @field_validator("slug")
    @classmethod
    def validate_slug(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            cleaned = slugify(v)
            if not cleaned:
                raise ValueError("Workspace slug cannot be empty.")
            return cleaned
        return None


class WorkspaceUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="Workspace display name")
    description: Optional[str] = Field(None, max_length=1000, description="Workspace description")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            cleaned = v.strip()
            if not cleaned:
                raise ValueError("Workspace name cannot be empty or purely whitespace.")
            return cleaned
        return v


class WorkspaceResponse(BaseModel):
    id: str
    name: str
    slug: str
    description: Optional[str] = None
    is_active: bool
    document_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class WorkspaceStatsResponse(BaseModel):
    workspace_id: str
    total_documents: int = 0
    indexed_documents: int = 0
    total_pages: int = 0
    completed_extraction: int = 0
    processing_documents: int = 0
    failed_documents: int = 0
    company_documents: int = 0
    active_employees: int = 0
    pending_invitations: int = 0
    pending_reviews: int = 0

    model_config = ConfigDict(from_attributes=True)


class ActivityItemResponse(BaseModel):
    id: str
    action: str
    actor_user_id: Optional[str] = None
    actor_name: Optional[str] = None
    actor_email: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    outcome: str
    details: Optional[dict] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ActivityListResponse(BaseModel):
    total: int
    items: list[ActivityItemResponse]
