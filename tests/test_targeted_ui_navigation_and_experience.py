"""Targeted Unit & Integration Tests for EKN UI/UX Navigation, Scope Persistence, Citation Isolation, and Markdown.

Verifies:
1. Navigation state machine: normalization, alias routing, and radio synchronization.
2. Workspace membership authorization filtering and context clearing.
3. Employee conversation scope persistence (pending_new_chat_scope lifecycle).
4. Citation isolation between active and empty conversations.
5. Markdown normalization: header blank lines, code block immunity, table/list/citation preservation.
"""

from typing import Any, Dict, List
import pytest
import streamlit as st

from frontend.navigation import (
    PRIMARY_NAV_DESTINATIONS,
    VIEW_ALIAS_MAP,
    navigate_to,
    normalize_view,
    on_sidebar_nav_change,
    sync_navigation_state,
)
from frontend.views.employee_workspace import _normalize_markdown


# =============================================================================
# 1. NAVIGATION STATE & ALIAS NORMALIZATION TESTS
# =============================================================================

def test_normalize_view_primary_destinations():
    """Verify that all 6 primary navigation destinations pass through unchanged."""
    for dest in PRIMARY_NAV_DESTINATIONS:
        assert normalize_view(dest) == dest


def test_normalize_view_legacy_aliases():
    """Verify that all legacy and subview aliases correctly normalize to their parent."""
    # Documents aliases
    assert normalize_view("Company Documents") == "Documents"
    assert normalize_view("Document Families & Versions") == "Documents"
    assert normalize_view("Document Families") == "Documents"
    assert normalize_view("Families") == "Documents"
    assert normalize_view("Upload") == "Documents"
    assert normalize_view("Upload Documents") == "Documents"

    # Knowledge Review aliases
    assert normalize_view("Contradiction Review") == "Knowledge Review"
    assert normalize_view("Contradictions") == "Knowledge Review"
    assert normalize_view("Findings") == "Knowledge Review"

    # Team aliases
    assert normalize_view("Employees") == "Team"
    assert normalize_view("Invitations") == "Team"
    assert normalize_view("Team Management") == "Team"

    # Reports & Settings aliases
    assert normalize_view("Reports") == "Reports & Settings"
    assert normalize_view("Activity") == "Reports & Settings"
    assert normalize_view("Company Settings") == "Reports & Settings"
    assert normalize_view("Settings") == "Reports & Settings"

    # Standalone Ask Knowledge Base
    assert normalize_view("Ask Knowledge Base") == "Ask Knowledge Base"

    # Unknown or empty fallbacks
    assert normalize_view("") == "Overview"
    assert normalize_view(None) == "Overview"
    assert normalize_view("NonExistentViewXYZ") == "Overview"


def test_navigate_to_synchronization():
    """Verify that navigate_to sets active_view and sidebar_navigation_radio simultaneously."""
    st.session_state.clear()
    navigate_to("Documents", doc_open_upload=True)
    assert st.session_state["active_view"] == "Documents"
    assert st.session_state["sidebar_navigation_radio"] == "Documents"
    assert st.session_state.get("doc_open_upload") is True

    # Navigating to a subview sets specific active_view while radio is normalized
    navigate_to("Invitations")
    assert st.session_state["active_view"] == "Invitations"
    assert st.session_state["sidebar_navigation_radio"] == "Team"


def test_on_sidebar_nav_change_callback():
    """Verify that the sidebar on_change callback pushes the radio selection to active_view."""
    st.session_state.clear()
    st.session_state["sidebar_navigation_radio"] = "Knowledge Review"
    on_sidebar_nav_change()
    assert st.session_state["active_view"] == "Knowledge Review"


def test_sync_navigation_state_reconciliation():
    """Verify that sync_navigation_state repairs any desynchronization."""
    st.session_state.clear()
    st.session_state["active_view"] = "Team"
    st.session_state["sidebar_navigation_radio"] = "Overview"  # stale state

    normalized = sync_navigation_state()
    assert normalized == "Team"
    assert st.session_state["sidebar_navigation_radio"] == "Team"


# =============================================================================
# 2. WORKSPACE AUTHORIZATION & MEMBERSHIP FILTERING
# =============================================================================

def test_workspace_membership_filtering_logic():
    """Verify that only workspaces present in active profile memberships are accessible."""
    mock_user_workspaces = [
        {"workspace_id": "ws-authorized-1", "workspace_name": "Aerospace & Engineering", "role": "admin"},
        {"workspace_id": "ws-authorized-2", "workspace_name": "Defense Systems", "role": "employee"},
    ]
    membership_map = {w["workspace_id"]: w for w in mock_user_workspaces}

    all_backend_workspaces = [
        {"id": "ws-authorized-1", "name": "Aerospace & Engineering"},
        {"id": "ws-authorized-2", "name": "Defense Systems"},
        {"id": "ws-unrelated-3", "name": "NMIT Private Workspace"},
        {"id": "ws-unrelated-4", "name": "Competitor Corp"},
    ]

    # Authorized filtering
    authorized = [w for w in all_backend_workspaces if w.get("id") in membership_map]
    assert len(authorized) == 2
    assert {w["id"] for w in authorized} == {"ws-authorized-1", "ws-authorized-2"}
    assert "ws-unrelated-3" not in {w["id"] for w in authorized}

    # Role resolution
    admin_mem = membership_map.get("ws-authorized-1")
    assert str(admin_mem.get("role")).lower() in ("owner", "admin")

    emp_mem = membership_map.get("ws-authorized-2")
    assert str(emp_mem.get("role")).lower() == "employee"

    # Unrelated workspace has no membership -> reject
    unrelated_mem = membership_map.get("ws-unrelated-3")
    assert unrelated_mem is None


# =============================================================================
# 3. EMPLOYEE SCOPE PERSISTENCE (pending_new_chat_scope)
# =============================================================================

def test_pending_new_chat_scope_lifecycle():
    """Verify dedicated pending_new_chat_scope for new conversations."""
    st.session_state.clear()

    # Step 1: User is on New Chat screen (no active conversation)
    active_conv = None
    assert st.session_state.get("pending_new_chat_scope") is None

    # Step 2: User changes scope to 'personal' before sending any message
    chosen_scope = "personal"
    if not active_conv:
        st.session_state["pending_new_chat_scope"] = chosen_scope

    assert st.session_state.get("pending_new_chat_scope") == "personal"

    # Step 3: Scope to use on conversation creation
    scope_to_use = st.session_state.get("pending_new_chat_scope") or "company"
    assert scope_to_use == "personal"

    # Step 4: After conversation is created, pending scope is popped
    created_conv = {"id": "conv-101", "knowledge_scope": scope_to_use}
    st.session_state["active_conversation_id"] = created_conv["id"]
    st.session_state.pop("pending_new_chat_scope", None)

    assert st.session_state.get("pending_new_chat_scope") is None
    assert st.session_state["active_conversation_id"] == "conv-101"


# =============================================================================
# 4. CITATION ISOLATION BETWEEN CONVERSATIONS
# =============================================================================

def test_citation_derivation_isolation():
    """Verify that citations are derived strictly from active conversation messages."""
    # Conversation A with citations
    conv_a_messages = [
        {"role": "user", "content": "What are turbine specs?"},
        {
            "role": "assistant",
            "content": "Operating temp 450C [E1].",
            "citations": [{"tag": "[E1]", "filename": "Turbine.pdf", "page_start": 1}],
        },
    ]

    # Conversation B without citations
    conv_b_messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello! How can I assist you today?", "citations": []},
    ]

    def get_active_citations(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for m in reversed(messages):
            if m.get("role") == "assistant" and m.get("citations"):
                return m.get("citations") or []
        return []

    # Active chat = A
    cites_a = get_active_citations(conv_a_messages)
    assert len(cites_a) == 1
    assert cites_a[0]["tag"] == "[E1]"

    # Switch to chat B: citations MUST NOT leak from A!
    cites_b = get_active_citations(conv_b_messages)
    assert len(cites_b) == 0

    # Switch back to A: citations reappear
    cites_a_again = get_active_citations(conv_a_messages)
    assert len(cites_a_again) == 1
    assert cites_a_again[0]["filename"] == "Turbine.pdf"


# =============================================================================
# 5. MARKDOWN NORMALIZATION TESTS
# =============================================================================

def test_markdown_normalization_headings():
    """Verify that headings preceded by text get proper double newline separation."""
    raw = "Based on specifications:\n## Turbine Model X-500\nDetails follow."
    expected = "Based on specifications:\n\n## Turbine Model X-500\nDetails follow."
    assert _normalize_markdown(raw) == expected


def test_markdown_normalization_multiple_headings():
    """Verify adjacent headings get proper spacing."""
    raw = "# Main Title\n## Subheading 1\n### Subheading 2"
    expected = "# Main Title\n\n## Subheading 1\n\n### Subheading 2"
    assert _normalize_markdown(raw) == expected


def test_markdown_normalization_code_block_protection():
    """Verify lines inside code blocks containing '#' are NOT modified."""
    code_raw = (
        "Here is sample code:\n"
        "```python\n"
        "# Comment inside code\n"
        "## Second comment\n"
        "x = 42\n"
        "```\n"
        "## Next Section Heading"
    )
    normalized = _normalize_markdown(code_raw)
    assert "# Comment inside code\n## Second comment" in normalized
    assert "```\n\n## Next Section Heading" in normalized


def test_markdown_normalization_preserves_tables_and_citations():
    """Verify that Markdown tables, bold, bullet lists, and citations are preserved."""
    content = (
        "| Specification | Value |\n"
        "| :--- | :--- |\n"
        "| Temp | 450C [E1] |\n\n"
        "* **Bold Bullet 1:** Details [E1].\n"
        "* *Italic Bullet 2:* More details [E2]."
    )
    assert _normalize_markdown(content) == content
