from __future__ import annotations

"""Pydantic schemas for authentication, user profiles, and workspace memberships."""

from datetime import datetime
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class LoginRequest(BaseModel):
    """User credentials for authentication."""
    email: Optional[str] = None
    username_or_email: Optional[str] = None
    username: Optional[str] = None
    password: str = Field(..., min_length=1)
    remember_me: bool = True
    portal: Optional[str] = None  # 'admin' or 'employee' for server-side role validation


class RegisterRequest(BaseModel):
    """Payload for user self-registration."""
    full_name: str = Field(..., min_length=2, max_length=255)
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=8)
    workspace_name: Optional[str] = Field(default=None, max_length=100)


class TokenResponse(BaseModel):
    """Access and refresh token pair returned upon successful authentication."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # in seconds
    must_change_password: bool = False
    user: Optional[UserResponse] = None


class RefreshRequest(BaseModel):
    """Refresh token payload for renewing access tokens."""
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    """Payload for updating user password."""
    current_password: Optional[str] = None
    new_password: str = Field(..., min_length=8)


class UserResponse(BaseModel):
    """Sanitized public profile for an authenticated user."""
    id: str
    full_name: str
    email: str
    is_active: bool
    must_change_password: bool
    last_login_at: Optional[datetime] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserWorkspaceItem(BaseModel):
    """Summary of a workspace accessible to the user."""
    workspace_id: str
    workspace_name: str
    workspace_slug: str
    role: str  # 'owner', 'admin', 'employee'
    membership_status: str


class UserMeResponse(BaseModel):
    """Full current user context including all workspace memberships."""
    id: str
    full_name: str
    email: str
    is_active: bool
    must_change_password: bool
    last_login_at: Optional[datetime] = None
    created_at: datetime
    workspaces: List[UserWorkspaceItem]


class MemberCreateRequest(BaseModel):
    """Administrative request to add a user to a workspace."""
    email: str = Field(..., min_length=3, max_length=255)
    full_name: Optional[str] = None
    role: str = Field(default="employee", pattern="^(owner|admin|employee)$")
    initial_password: Optional[str] = None


class MemberUpdateRequest(BaseModel):
    """Administrative request to modify a member's role or status."""
    role: Optional[str] = Field(default=None, pattern="^(owner|admin|employee)$")
    membership_status: Optional[str] = Field(default=None, pattern="^(active|suspended|revoked|deactivated)$")


class MemberResponse(BaseModel):
    """Workspace member record returned to administrators."""
    id: str
    user_id: str
    workspace_id: str
    full_name: str
    email: str
    role: str
    membership_status: str
    created_at: datetime
    temporary_password: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class MemberListResponse(BaseModel):
    """Paginated list of workspace members."""
    total: int
    items: List[MemberResponse]


class AdminRegisterRequest(BaseModel):
    """Payload for administrator company registration."""
    full_name: str = Field(..., min_length=2, max_length=255)
    company_name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=8)


class EmployeeRegisterRequest(BaseModel):
    """Payload for employee registration using a company joining code (optional in demo mode)."""
    full_name: str = Field(..., min_length=2, max_length=255)
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=8)
    code: Optional[str] = Field(default=None, max_length=64)


class InvitationValidateRequest(BaseModel):
    """Payload for verifying a company joining code."""
    code: str = Field(..., min_length=4, max_length=64)


class InvitationValidateResponse(BaseModel):
    """Result of joining code validation."""
    is_valid: bool
    company_name: Optional[str] = None
    workspace_id: Optional[str] = None
    intended_role: Optional[str] = None
    message: Optional[str] = None


class InvitationCreateRequest(BaseModel):
    """Payload for generating an employee joining code."""
    max_usage_count: Optional[int] = Field(default=10, ge=1, le=1000)
    expires_in_days: Optional[int] = Field(default=7, ge=1, le=90)
    intended_role: Optional[str] = Field(default="employee", pattern="^(admin|employee)$")


class InvitationCreateResponse(BaseModel):
    """Response containing the one-time raw joining code for the administrator."""
    id: str
    raw_code: str
    workspace_id: str
    intended_role: str
    expires_at: datetime
    max_usage_count: int
    created_at: datetime


class InvitationItemResponse(BaseModel):
    """Sanitized invitation record (raw code and hash are never exposed)."""
    id: str
    workspace_id: str
    intended_role: str
    expires_at: datetime
    max_usage_count: int
    current_usage_count: int
    is_revoked: bool
    is_valid: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class InvitationListResponse(BaseModel):
    """List of invitations for a workspace."""
    total: int
    items: List[InvitationItemResponse]
