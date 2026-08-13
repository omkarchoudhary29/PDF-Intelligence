"""Centralized Navigation State Manager for Enterprise Knowledge Navigator (EKN).

Ensures deterministic, synchronized routing across Streamlit session state keys:
- `active_view`: Current application view/subview
- `sidebar_navigation_radio`: Sidebar radio widget selection

Eliminates navigation loops, stale radio state overwrites, and alias fragmentation.
"""

from typing import Any, Dict, List, Optional
import streamlit as st

PRIMARY_NAV_DESTINATIONS: List[str] = [
    "Overview",
    "Documents",
    "Knowledge Review",
    "Knowledge Graph",
    "Team",
    "Reports & Settings",
]

VIEW_ALIAS_MAP: Dict[str, str] = {
    # Documents aliases & subviews
    "Company Documents": "Documents",
    "Document Families & Versions": "Documents",
    "Document Families": "Documents",
    "Families": "Documents",
    "Upload": "Documents",
    "Upload Documents": "Documents",
    # Knowledge Review aliases
    "Contradiction Review": "Knowledge Review",
    "Contradictions": "Knowledge Review",
    "Findings": "Knowledge Review",
    # Team aliases & subviews
    "Employees": "Team",
    "Invitations": "Team",
    "Team Management": "Team",
    # Reports & Settings aliases & subviews
    "Reports": "Reports & Settings",
    "Activity": "Reports & Settings",
    "Company Settings": "Reports & Settings",
    "Settings": "Reports & Settings",
}


def normalize_view(view_name: Optional[str]) -> str:
    """Normalize any view string, subview, or legacy alias to an approved primary destination.
    
    Preserves special standalone views such as 'Ask Knowledge Base'.
    Defaults to 'Overview' for unrecognized or empty inputs.
    """
    if not view_name:
        return "Overview"
    clean = view_name.strip()
    if clean in PRIMARY_NAV_DESTINATIONS:
        return clean
    if clean == "Ask Knowledge Base":
        return "Ask Knowledge Base"
    if clean in VIEW_ALIAS_MAP:
        return VIEW_ALIAS_MAP[clean]
    return "Overview"


def navigate_to(target_view: str, **kwargs: Any) -> None:
    """Programmatically transition to a view and synchronize all navigation state keys.
    
    Sets `active_view` to the specific target (allowing subview routing) and sets
    `sidebar_navigation_radio` to the normalized primary destination so that Streamlit's
    radio widget does not revert the view during the next script pass.
    """
    normalized = normalize_view(target_view)
    st.session_state["active_view"] = target_view
    st.session_state["sidebar_navigation_radio"] = normalized
    for key, val in kwargs.items():
        st.session_state[key] = val


def on_sidebar_nav_change() -> None:
    """Streamlit on_change callback for sidebar radio widget.
    
    Synchronizes user-driven sidebar selection directly to `active_view` during the callback phase.
    """
    chosen = st.session_state.get("sidebar_navigation_radio")
    if chosen:
        st.session_state["active_view"] = chosen


def sync_navigation_state() -> str:
    """Ensure `active_view` and `sidebar_navigation_radio` are mutually synchronized.
    
    Returns the normalized primary view name.
    """
    active = st.session_state.get("active_view", "Overview")
    normalized = normalize_view(active)
    radio_val = st.session_state.get("sidebar_navigation_radio")
    
    if radio_val != normalized:
        st.session_state["sidebar_navigation_radio"] = normalized
        
    return normalized
