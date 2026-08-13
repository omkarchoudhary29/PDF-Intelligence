import logging
from typing import Any, Dict, List, Optional, Tuple
import requests

logger = logging.getLogger("ekn.frontend.api")


class APIClientError(Exception):
    """Custom exception raised when backend communication fails."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class BackendAPIClient:
    """Reusable HTTP client interface for communicating with the EKN FastAPI backend.

    Supports JWT Bearer token authentication, refresh token sessions,
    workspace and tenant isolation, document lifecycle, hybrid search,
    conversational RAG, knowledge graphs, enterprise dossier reports,
    and member administration.
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.current_user: Optional[Dict[str, Any]] = None

    def _headers(self, custom_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Inject Authorization header when access token is present."""
        headers = dict(custom_headers) if custom_headers else {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    def _extract_error(self, resp: requests.Response) -> str:
        """Extract clean, human-readable error detail from response."""
        try:
            content_type = resp.headers.get("content-type", "")
            if "application/json" in content_type:
                data = resp.json()
                if isinstance(data, dict):
                    detail = data.get("detail", data.get("message", None))
                    if detail:
                        if isinstance(detail, list):
                            msgs = []
                            for item in detail:
                                if isinstance(item, dict):
                                    loc = " -> ".join(str(l) for l in item.get("loc", []) if l != "body")
                                    msg = item.get("msg", "")
                                    msgs.append(f"{loc}: {msg}" if loc else msg)
                                else:
                                    msgs.append(str(item))
                            return "; ".join(msgs)
                        return str(detail)
                    return str(data)
                return str(data)
            return resp.text or f"HTTP {resp.status_code}"
        except Exception:
            return resp.text or f"HTTP {resp.status_code}"

    # -------------------------------------------------------------------------
    # Authentication & Session Management
    # -------------------------------------------------------------------------

    def login(
        self,
        username_or_email: str,
        password: str,
        remember_me: bool = True,
        portal: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Authenticate user and obtain JWT access and refresh tokens with optional portal role validation."""
        payload = {
            "email": username_or_email,
            "username_or_email": username_or_email,
            "password": password,
            "remember_me": remember_me,
        }
        if portal:
            payload["portal"] = portal
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/auth/login",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data.get("access_token")
                self.refresh_token = data.get("refresh_token")
                self.current_user = data.get("user")
                return data
            raise APIClientError(self._extract_error(resp), resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Login connection error: {exc}") from exc

    def register(
        self,
        full_name: str,
        email: str,
        password: str,
        workspace_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a new enterprise user and obtain JWT tokens."""
        payload = {
            "full_name": full_name,
            "email": email,
            "password": password,
            "workspace_name": workspace_name,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/auth/register",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                self.access_token = data.get("access_token")
                self.refresh_token = data.get("refresh_token")
                self.current_user = data.get("user")
                return data
            raise APIClientError(self._extract_error(resp), resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Registration connection error: {exc}") from exc

    def admin_register(
        self,
        full_name: str,
        company_name: str,
        email: str,
        password: str,
    ) -> Dict[str, Any]:
        """Register a new company and primary administrator account."""
        payload = {
            "full_name": full_name,
            "company_name": company_name,
            "email": email,
            "password": password,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/auth/admin/register",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                self.access_token = data.get("access_token")
                self.refresh_token = data.get("refresh_token")
                self.current_user = data.get("user")
                return data
            raise APIClientError(self._extract_error(resp), resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Admin registration connection error: {exc}") from exc

    def employee_register(
        self,
        full_name: str,
        email: str,
        password: str,
        code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a new employee using an authorized company joining code (optional in demo mode)."""
        payload = {
            "full_name": full_name,
            "email": email,
            "password": password,
        }
        if code:
            payload["code"] = code
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/auth/employee/register",
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                self.access_token = data.get("access_token")
                self.refresh_token = data.get("refresh_token")
                self.current_user = data.get("user")
                return data
            raise APIClientError(self._extract_error(resp), resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Employee registration connection error: {exc}") from exc

    def validate_invitation(self, code: str) -> Dict[str, Any]:
        """Validate a company joining code before submitting registration."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/auth/invitations/validate",
                json={"code": code},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(self._extract_error(resp), resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Validation connection error: {exc}") from exc

    def create_invitation(
        self,
        workspace_id: str,
        max_usage_count: int = 10,
        expires_in_days: int = 7,
        intended_role: str = "employee",
    ) -> Dict[str, Any]:
        """Generate a new employee joining code (Administrator only)."""
        payload = {
            "max_usage_count": max_usage_count,
            "expires_in_days": expires_in_days,
            "intended_role": intended_role,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/invitations",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code in (200, 201):
                return resp.json()
            raise APIClientError(self._extract_error(resp), resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Create invitation connection error: {exc}") from exc

    def list_invitations(self, workspace_id: str) -> Dict[str, Any]:
        """List joining codes for a workspace (Administrator only)."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/invitations",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(self._extract_error(resp), resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"List invitations connection error: {exc}") from exc

    def revoke_invitation(self, workspace_id: str, invitation_id: str) -> Dict[str, Any]:
        """Revoke a joining code (Administrator only)."""
        try:
            resp = requests.delete(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/invitations/{invitation_id}",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(self._extract_error(resp), resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Revoke invitation connection error: {exc}") from exc

    def get_company_activity(self, workspace_id: str, limit: int = 50) -> Dict[str, Any]:
        """List company activity audit logs (Administrator only)."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/activity",
                params={"limit": limit},
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(self._extract_error(resp), resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Get company activity connection error: {exc}") from exc

    def refresh_session(self, token: Optional[str] = None) -> bool:
        """Rotate and refresh access token using database-backed refresh token."""
        use_token = token or self.refresh_token
        if not use_token:
            return False
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/auth/refresh",
                json={"refresh_token": use_token},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data.get("access_token")
                self.refresh_token = data.get("refresh_token")
                return True
            return False
        except requests.RequestException:
            return False

    def logout(self) -> bool:
        """Revoke current refresh session and clear local tokens."""
        try:
            if self.refresh_token:
                requests.post(
                    f"{self.base_url}/api/v1/auth/logout",
                    json={"refresh_token": self.refresh_token},
                    headers=self._headers(),
                    timeout=self.timeout,
                )
        except Exception as exc:
            logger.warning("Logout backend error: %s", exc)
        finally:
            self.access_token = None
            self.refresh_token = None
            self.current_user = None
        return True

    def get_current_user_profile(self) -> Dict[str, Any]:
        """Fetch authenticated user profile, permissions, and workspace memberships."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/auth/me",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                self.current_user = data
                return data
            raise APIClientError(self._extract_error(resp), resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"User profile connection error: {exc}") from exc

    def change_password(self, current_password: str, new_password: str) -> Dict[str, Any]:
        """Change user password enforcing enterprise password complexity."""
        payload = {
            "current_password": current_password,
            "new_password": new_password,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/auth/change-password",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(self._extract_error(resp), resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Password change connection error: {exc}") from exc

    # -------------------------------------------------------------------------
    # System Health & Readiness
    # -------------------------------------------------------------------------

    def check_health(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Liveness check against GET /health."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=2.0)
            if resp.status_code == 200:
                return True, resp.json()
            return False, None
        except requests.RequestException:
            return False, None

    def check_readiness(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Readiness check against GET /ready."""
        try:
            resp = requests.get(f"{self.base_url}/ready", timeout=2.0)
            if resp.status_code in (200, 503):
                return True, resp.json()
            return False, None
        except requests.RequestException:
            return False, None

    # -------------------------------------------------------------------------
    # Workspaces
    # -------------------------------------------------------------------------

    def list_workspaces(self) -> List[Dict[str, Any]]:
        """Fetch all active workspaces accessible to current user."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to list workspaces: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Backend connection error: {exc}") from exc

    def create_workspace(
        self,
        name: str,
        description: Optional[str] = None,
        slug: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new workspace."""
        payload = {"name": name}
        if description:
            payload["description"] = description
        if slug:
            payload["slug"] = slug

        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 201:
                return resp.json()
            error_msg = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise APIClientError(f"{error_msg}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Backend connection error: {exc}") from exc

    def get_workspace(self, workspace_id: str) -> Dict[str, Any]:
        """Get details for a single workspace."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Workspace not found: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Backend connection error: {exc}") from exc

    def update_workspace(
        self,
        workspace_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update company/workspace profile settings (Administrator/Owner only)."""
        payload: Dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        try:
            resp = requests.patch(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(self._extract_error(resp), resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Network error updating workspace: {exc}") from exc

    def get_workspace_stats(self, workspace_id: str) -> Dict[str, Any]:
        """Fetch repository statistics for a workspace."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/stats",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to fetch workspace stats: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Workspace stats connection error: {exc}") from exc

    # -------------------------------------------------------------------------
    # Workspace Member Management (RBAC)
    # -------------------------------------------------------------------------

    def list_members(self, workspace_id: str) -> List[Dict[str, Any]]:
        """List active members and their roles for a workspace."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/members",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("items", []) if isinstance(data, dict) else data
            raise APIClientError(f"Failed to list members: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"List members connection error: {exc}") from exc

    def add_member(
        self,
        workspace_id: str,
        email: str,
        role: str = "employee",
        full_name: Optional[str] = None,
        temporary_password: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add or invite a member to a workspace."""
        payload: Dict[str, Any] = {"email": email, "role": role}
        if full_name:
            payload["full_name"] = full_name
        if temporary_password:
            payload["temporary_password"] = temporary_password
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/members",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 201:
                return resp.json()
            detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise APIClientError(detail, resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Add member connection error: {exc}") from exc

    def update_member(
        self,
        workspace_id: str,
        user_id: str,
        role: Optional[str] = None,
        membership_status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update an existing member's role or status."""
        payload: Dict[str, Any] = {}
        if role:
            payload["role"] = role
        if membership_status:
            payload["membership_status"] = membership_status
        try:
            resp = requests.patch(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/members/{user_id}",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise APIClientError(detail, resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Update member connection error: {exc}") from exc

    def remove_member(self, workspace_id: str, user_id: str) -> Dict[str, Any]:
        """Remove a member from a workspace."""
        try:
            resp = requests.delete(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/members/{user_id}",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise APIClientError(detail, resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Remove member connection error: {exc}") from exc

    # -------------------------------------------------------------------------
    # Documents
    # -------------------------------------------------------------------------

    def list_documents(
        self,
        workspace_id: str,
        page: int = 1,
        page_size: int = 50,
        status: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve paginated documents for a workspace with optional scope filtering ('company', 'personal', 'all')."""
        params = {"page": page, "page_size": page_size}
        if status:
            params["status"] = status
        if scope:
            params["scope"] = scope

        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/documents",
                params=params,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to fetch documents: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Backend connection error: {exc}") from exc

    def upload_personal_documents(
        self,
        workspace_id: str,
        files: List[Tuple[str, bytes]],
    ) -> Dict[str, Any]:
        """Upload private personal PDF files for the authenticated employee."""
        multipart_files = [
            ("files", (filename, data, "application/pdf"))
            for filename, data in files
        ]

        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/documents/personal",
                files=multipart_files,
                headers=self._headers(),
                timeout=max(self.timeout, 120.0),
            )
            if resp.status_code == 200:
                return resp.json()
            error_detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise APIClientError(f"Personal upload failed: {error_detail}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Personal upload connection error: {exc}") from exc

    def upload_documents(
        self,
        workspace_id: str,
        files: List[Tuple[str, bytes]],
    ) -> Dict[str, Any]:
        """Upload multiple PDF files to the specified workspace."""
        multipart_files = [
            ("files", (filename, data, "application/pdf"))
            for filename, data in files
        ]

        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/documents",
                files=multipart_files,
                headers=self._headers(),
                timeout=max(self.timeout, 120.0),
            )
            if resp.status_code == 200:
                return resp.json()
            error_detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise APIClientError(f"Upload failed: {error_detail}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Upload connection error: {exc}") from exc

    def download_document(self, workspace_id: str, document_id: str) -> Tuple[bytes, str]:
        """Download original PDF file bytes from the backend."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/documents/{document_id}/download",
                headers=self._headers(),
                timeout=self.timeout,
                stream=True,
            )
            if resp.status_code == 200:
                cd = resp.headers.get("content-disposition", "")
                filename = "document.pdf"
                if "filename=" in cd:
                    filename = cd.split("filename=")[-1].strip('"\' ')
                return resp.content, filename
            raise APIClientError(f"Download failed: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Download connection error: {exc}") from exc

    def delete_document(self, workspace_id: str, document_id: str) -> Dict[str, Any]:
        """Delete document from the specified workspace."""
        try:
            resp = requests.delete(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/documents/{document_id}",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Delete failed: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Delete connection error: {exc}") from exc

    def retry_document(self, workspace_id: str, document_id: str) -> Dict[str, Any]:
        """Retry ingestion for a failed document."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/documents/{document_id}/retry",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Retry request failed: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Retry connection error: {exc}") from exc

    def get_extracted_pages(self, workspace_id: str, document_id: str) -> Dict[str, Any]:
        """Fetch extracted JSON Lines page records for inspection."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/documents/{document_id}/pages",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to fetch extraction details: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Extraction details connection error: {exc}") from exc

    def get_document_index(self, workspace_id: str, document_id: str) -> Dict[str, Any]:
        """Fetch vector indexing status and metadata for a document."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/documents/{document_id}/index",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to fetch index status: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Index status connection error: {exc}") from exc

    def get_document_index_status(self, workspace_id: str, document_id: str) -> Dict[str, Any]:
        """Backward-compatible alias for get_document_index."""
        return self.get_document_index(workspace_id=workspace_id, document_id=document_id)

    def reindex_document(self, workspace_id: str, document_id: str, force: bool = True) -> Dict[str, Any]:
        """Trigger re-indexing or backfilling for an extracted document."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/documents/{document_id}/reindex",
                params={"force": force},
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Re-indexing failed: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Re-index connection error: {exc}") from exc

    def preview_page(
        self,
        workspace_id: str,
        document_id: str,
        page_number: int,
        dpi: int = 150,
    ) -> bytes:
        """Fetch rendered PNG bytes for a specific page of an original PDF."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/documents/{document_id}/pages/{page_number}/preview",
                params={"dpi": dpi},
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.content
            raise APIClientError(f"Failed to render page preview: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Page preview connection error: {exc}") from exc

    # -------------------------------------------------------------------------
    # Conversations & RAG
    # -------------------------------------------------------------------------

    def list_conversations(
        self,
        workspace_id: str,
        page: int = 1,
        page_size: int = 50,
    ) -> Dict[str, Any]:
        """Fetch paginated conversations for a workspace."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/conversations",
                params={"page": page, "page_size": page_size},
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to list conversations: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Conversations connection error: {exc}") from exc

    def create_conversation(
        self,
        workspace_id: str,
        title: Optional[str] = None,
        knowledge_scope: Optional[str] = "company",
    ) -> Dict[str, Any]:
        """Create a new conversation session inside a workspace."""
        payload: Dict[str, Any] = {"knowledge_scope": knowledge_scope or "company"}
        if title:
            payload["title"] = title

        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/conversations",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 201:
                return resp.json()
            raise APIClientError(f"Failed to create conversation: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Create conversation connection error: {exc}") from exc

    def update_conversation(
        self,
        workspace_id: str,
        conversation_id: str,
        title: Optional[str] = None,
        knowledge_scope: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update conversation title or knowledge scope."""
        payload: Dict[str, Any] = {}
        if title is not None:
            payload["title"] = title
        if knowledge_scope is not None:
            payload["knowledge_scope"] = knowledge_scope

        try:
            resp = requests.patch(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/conversations/{conversation_id}",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise APIClientError(f"Failed to update conversation: {detail}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Update conversation connection error: {exc}") from exc

    def delete_conversation(self, workspace_id: str, conversation_id: str) -> Dict[str, Any]:
        """Delete a conversation owned by the employee."""
        try:
            resp = requests.delete(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/conversations/{conversation_id}",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise APIClientError(f"Failed to delete conversation: {detail}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Delete conversation connection error: {exc}") from exc

    def attach_document_to_conversation(
        self,
        workspace_id: str,
        conversation_id: str,
        document_id: str,
    ) -> Dict[str, Any]:
        """Attach an authorized document to a conversation."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/conversations/{conversation_id}/attachments",
                json={"document_id": document_id},
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 201:
                return resp.json()
            detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise APIClientError(f"Failed to attach document: {detail}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Attach document connection error: {exc}") from exc

    def list_conversation_attachments(
        self,
        workspace_id: str,
        conversation_id: str,
    ) -> Dict[str, Any]:
        """List documents attached to a conversation."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/conversations/{conversation_id}/attachments",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise APIClientError(f"Failed to list attachments: {detail}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"List attachments connection error: {exc}") from exc

    def remove_conversation_attachment(
        self,
        workspace_id: str,
        conversation_id: str,
        document_id: str,
    ) -> Dict[str, Any]:
        """Detach a document from a conversation."""
        try:
            resp = requests.delete(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/conversations/{conversation_id}/attachments/{document_id}",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise APIClientError(f"Failed to remove attachment: {detail}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Remove attachment connection error: {exc}") from exc

    def get_conversation(self, workspace_id: str, conversation_id: str) -> Dict[str, Any]:
        """Get details for a single conversation."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/conversations/{conversation_id}",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Conversation not found: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Get conversation connection error: {exc}") from exc

    def list_messages(self, workspace_id: str, conversation_id: str) -> Dict[str, Any]:
        """Fetch all messages for a conversation in chronological order."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/conversations/{conversation_id}/messages",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to fetch messages: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Messages connection error: {exc}") from exc

    def send_message(
        self,
        workspace_id: str,
        conversation_id: str,
        content: str,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """Send user message, execute LangGraph RAG workflow, and return grounded answer with citations."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/conversations/{conversation_id}/messages",
                json={"content": content},
                headers=self._headers(),
                timeout=timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise APIClientError(f"Query execution error: {detail}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Query execution connection error: {exc}") from exc

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    def search(
        self,
        workspace_id: str,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> Dict[str, Any]:
        """Execute hybrid search directly."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/search",
                json={"query": query, "top_k": top_k, "mode": mode},
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Search failed: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Search connection error: {exc}") from exc

    # -------------------------------------------------------------------------
    # Knowledge Graph
    # -------------------------------------------------------------------------

    def build_graph(
        self,
        workspace_id: str,
        document_ids: Optional[List[str]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Trigger knowledge graph entity extraction and relationship discovery."""
        payload = {"force": force}
        if document_ids:
            payload["document_ids"] = document_ids
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/graph/build",
                json=payload,
                headers=self._headers(),
                timeout=max(self.timeout, 120.0),
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Graph build failed: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Graph build connection error: {exc}") from exc

    def get_graph_summary(self, workspace_id: str) -> Dict[str, Any]:
        """Fetch summary statistics for workspace knowledge graph."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/graph/summary",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to fetch graph summary: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Graph summary connection error: {exc}") from exc

    def get_graph_data(
        self,
        workspace_id: str,
        limit: int = 50,
        relationship_type: Optional[str] = None,
        focus_document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch bounded graph nodes and edges."""
        params: Dict[str, Any] = {"limit": limit}
        if relationship_type:
            params["relationship_type"] = relationship_type
        if focus_document_id:
            params["focus_document_id"] = focus_document_id
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/graph",
                params=params,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to fetch graph data: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Graph data connection error: {exc}") from exc

    def render_graph_image(
        self,
        workspace_id: str,
        limit: int = 40,
        relationship_type: Optional[str] = None,
        focus_document_id: Optional[str] = None,
    ) -> bytes:
        """Fetch rendered Matplotlib PNG image stream for the graph."""
        params: Dict[str, Any] = {"limit": limit}
        if relationship_type:
            params["relationship_type"] = relationship_type
        if focus_document_id:
            params["focus_document_id"] = focus_document_id
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/graph/render",
                params=params,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.content
            raise APIClientError(f"Failed to render graph image: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Render graph connection error: {exc}") from exc

    def get_relationship_details(
        self,
        workspace_id: str,
        relationship_id: str,
    ) -> Dict[str, Any]:
        """Fetch full details and evidence for a specific relationship."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/graph/relationships/{relationship_id}",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to fetch relationship: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Relationship details connection error: {exc}") from exc

    def backfill_graph(self, workspace_id: str) -> Dict[str, Any]:
        """Trigger graph backfill for company documents lacking entity data without re-embedding."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/graph/backfill",
                headers=self._headers(),
                timeout=max(self.timeout, 120.0),
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Graph backfill failed: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Graph backfill connection error: {exc}") from exc

    def get_graph_entities(
        self,
        workspace_id: str,
        q: Optional[str] = None,
        entity_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Dict[str, Any]:
        """List and search entities in workspace knowledge graph."""
        params: Dict[str, Any] = {"page": page, "page_size": page_size}
        if q:
            params["q"] = q
        if entity_type:
            params["entity_type"] = entity_type
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/graph/entities",
                params=params,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to fetch entities: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Entities connection error: {exc}") from exc

    def get_entity_relationships(
        self,
        workspace_id: str,
        entity_id: str,
    ) -> List[Dict[str, Any]]:
        """Fetch all direct relationships involving a specific entity."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/graph/entities/{entity_id}/relationships",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to fetch entity relationships: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Entity relationships connection error: {exc}") from exc

    def find_graph_path(
        self,
        workspace_id: str,
        source_id: str,
        target_id: str,
    ) -> Dict[str, Any]:
        """Compute shortest path between two nodes in the knowledge graph."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/graph/path",
                params={"source_id": source_id, "target_id": target_id},
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Path query failed: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Path query connection error: {exc}") from exc

    def get_graph_reconciliation(self, workspace_id: str) -> Dict[str, Any]:
        """Fetch document and vector count reconciliation details."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/graph/reconciliation",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Reconciliation query failed: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Reconciliation connection error: {exc}") from exc

    # -------------------------------------------------------------------------
    # Reports & Knowledge Dossiers
    # -------------------------------------------------------------------------

    def create_report(
        self,
        workspace_id: str,
        report_name: Optional[str] = None,
        include_graph: bool = True,
        run_synchronous: bool = True,
    ) -> Dict[str, Any]:
        """Request generation of an Enterprise Knowledge Dossier PDF."""
        payload: Dict[str, Any] = {"include_graph": include_graph}
        if report_name:
            payload["report_name"] = report_name
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/reports",
                json=payload,
                params={"run_synchronous": run_synchronous},
                headers=self._headers(),
                timeout=max(self.timeout, 120.0),
            )
            if resp.status_code in (200, 201):
                return resp.json()
            detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type") == "application/json" else resp.text
            raise APIClientError(f"Report generation request failed: {detail}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Report creation connection error: {exc}") from exc

    def list_reports(
        self,
        workspace_id: str,
        page: int = 1,
        page_size: int = 50,
    ) -> Dict[str, Any]:
        """Fetch list of generated reports in workspace."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/reports",
                params={"page": page, "page_size": page_size},
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to list reports: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"List reports connection error: {exc}") from exc

    def get_report(self, workspace_id: str, report_id: str) -> Dict[str, Any]:
        """Fetch status and metadata for a specific report."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/reports/{report_id}",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to fetch report: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Get report connection error: {exc}") from exc

    def download_report(self, workspace_id: str, report_id: str) -> Tuple[bytes, str]:
        """Download finalized Knowledge Dossier PDF bytes."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/reports/{report_id}/download",
                headers=self._headers(),
                timeout=max(self.timeout, 60.0),
                stream=True,
            )
            if resp.status_code == 200:
                cd = resp.headers.get("content-disposition") or resp.headers.get("Content-Disposition") or ""
                filename = "knowledge_dossier.pdf"
                if "filename=" in cd.lower():
                    parts = cd.split("filename=") if "filename=" in cd else cd.split("filename*=")
                    filename = parts[-1].split(";")[0].strip('"\' ')
                return resp.content, filename
            raise APIClientError(f"Report download failed: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Download report connection error: {exc}") from exc

    def delete_report(self, workspace_id: str, report_id: str) -> Dict[str, Any]:
        """Delete report record and remove stored PDF."""
        try:
            resp = requests.delete(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/reports/{report_id}",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to delete report: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Delete report connection error: {exc}") from exc

    # -------------------------------------------------------------------------
    # Document Version Intelligence & Contradiction Analysis
    # -------------------------------------------------------------------------

    def list_document_families(self, workspace_id: str) -> List[Dict[str, Any]]:
        """List document families in a workspace."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/document-families",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("items", []) if isinstance(data, dict) else data
            raise APIClientError(f"Failed to list document families: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"List document families connection error: {exc}") from exc

    def get_document_family(self, workspace_id: str, family_id: str) -> Dict[str, Any]:
        """Get details and version history for a document family."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/document-families/{family_id}",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to get document family: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Get document family connection error: {exc}") from exc

    def create_document_family(
        self,
        workspace_id: str,
        title: str,
        canonical_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an explicit logical document family."""
        try:
            payload: Dict[str, Any] = {"title": title}
            if canonical_name:
                payload["canonical_name"] = canonical_name
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/document-families",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code in (200, 201):
                return resp.json()
            raise APIClientError(f"Failed to create document family: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Create document family connection error: {exc}") from exc

    def detect_version_candidate(
        self,
        workspace_id: str,
        filename: str,
        file_hash: Optional[str] = None,
        text_preview: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Multi-signal document version detection on prospective file."""
        try:
            payload: Dict[str, Any] = {"filename": filename}
            if file_hash:
                payload["file_hash"] = file_hash
            if text_preview:
                payload["text_preview"] = text_preview
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/detect-version-candidate",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Version candidate detection failed: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Version candidate detection connection error: {exc}") from exc

    def detect_existing_document_version(
        self,
        workspace_id: str,
        document_id: str,
    ) -> Dict[str, Any]:
        """Multi-signal document version detection on an existing uploaded document."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/documents/{document_id}/detect-version",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Document version detection failed: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Document version detection connection error: {exc}") from exc

    def attach_document_to_family(
        self,
        workspace_id: str,
        family_id: str,
        document_id: str,
        version_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Manually attach an existing document to a document family."""
        try:
            payload: Dict[str, Any] = {"document_id": document_id}
            if version_number is not None:
                payload["version_number"] = version_number
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/document-families/{family_id}/attach-document",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to attach document to family: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Attach document connection error: {exc}") from exc

    def detach_document_from_family(
        self,
        workspace_id: str,
        family_id: str,
        document_id: str,
    ) -> Dict[str, Any]:
        """Detach a document from a family into its own independent family."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/document-families/{family_id}/detach-document/{document_id}",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to detach document from family: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Detach document connection error: {exc}") from exc

    def upload_document_version(
        self,
        workspace_id: str,
        family_id: str,
        file_name: str,
        file_bytes: bytes,
    ) -> Dict[str, Any]:
        """Upload a new PDF version to an existing document family."""
        try:
            files = {"file": (file_name, file_bytes, "application/pdf")}
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/document-families/{family_id}/versions",
                files=files,
                headers=self._auth_header(),
                timeout=self.timeout * 2,
            )
            if resp.status_code in (200, 201):
                return resp.json()
            raise APIClientError(f"Upload version failed: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Upload version connection error: {exc}") from exc

    def set_current_version(
        self,
        workspace_id: str,
        family_id: str,
        document_id: str,
    ) -> Dict[str, Any]:
        """Switch the authoritative current version for a document family."""
        try:
            resp = requests.patch(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/document-families/{family_id}/current-version",
                json={"document_id": document_id},
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to set current version: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Set current version connection error: {exc}") from exc

    def update_approval_status(
        self,
        workspace_id: str,
        document_id: str,
        approval_status: str,
    ) -> Dict[str, Any]:
        """Update approval status for a document version."""
        try:
            resp = requests.patch(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/documents/{document_id}/approval-status",
                json={"approval_status": approval_status},
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to update approval status: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Update approval status connection error: {exc}") from exc

    def compare_versions(
        self,
        workspace_id: str,
        family_id: str,
        source_doc_id: str,
        target_doc_id: str,
    ) -> Dict[str, Any]:
        """Request deterministic and AI comparison between two versions."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/document-families/{family_id}/compare",
                json={
                    "source_document_id": source_doc_id,
                    "target_document_id": target_doc_id,
                },
                headers=self._headers(),
                timeout=self.timeout * 2,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Version comparison failed: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Version comparison connection error: {exc}") from exc

    def list_findings(
        self,
        workspace_id: str,
        status_filter: Optional[str] = None,
        finding_type: Optional[str] = None,
        scan: bool = False,
    ) -> List[Dict[str, Any]]:
        """List contradiction findings across documents in a workspace."""
        params: Dict[str, Any] = {}
        if status_filter and status_filter != "All":
            params["status"] = status_filter
        if finding_type and finding_type != "All":
            params["finding_type"] = finding_type
        if scan:
            params["scan"] = "true"

        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/findings",
                params=params,
                headers=self._headers(),
                timeout=self.timeout * 3,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("items", []) if isinstance(data, dict) else data
            raise APIClientError(f"Failed to list findings: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"List findings connection error: {exc}") from exc

    def review_finding(
        self,
        workspace_id: str,
        finding_id: str,
        review_status: str,
        review_reason: Optional[str] = None,
        authoritative_document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform authorized review on a contradiction finding."""
        payload: Dict[str, Any] = {"review_status": review_status}
        if review_reason:
            payload["review_reason"] = review_reason
        if authoritative_document_id:
            payload["authoritative_document_id"] = authoritative_document_id

        try:
            resp = requests.patch(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/findings/{finding_id}/review",
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            raise APIClientError(f"Failed to review finding: {resp.text}", resp.status_code)
        except requests.RequestException as exc:
            raise APIClientError(f"Review finding connection error: {exc}") from exc
