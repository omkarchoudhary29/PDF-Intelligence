"""Company Activity and Audit Log View for Enterprise Knowledge Navigator.

Displays genuine company-level events: document uploads, deletions, employee registrations,
and invitation lifecycle events, strictly scoped to the administrator's company.
"""

from typing import Any, Dict, List
import streamlit as st

from frontend.api_client import APIClientError, BackendAPIClient
from frontend.styles import format_date


def render_activity(
    client: BackendAPIClient,
    workspace: Dict[str, Any],
) -> None:
    """Render the Company Activity log view for administrators."""
    ws_id = workspace["id"]
    ws_name = workspace["name"]

    col_title, col_refresh = st.columns([3, 1])
    with col_title:
        st.markdown(f"## Company Activity Log — {ws_name}")
        st.caption("Review genuine organizational events, document operations, employee joins, and security actions.")
    with col_refresh:
        if st.button("🔄 Refresh Activity", use_container_width=True):
            st.rerun()

    st.markdown("<hr style='margin: 0.6rem 0 1.25rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

    # Fetch genuine activity logs from backend
    try:
        activity_data = client.get_company_activity(ws_id, limit=100)
        events: List[Dict[str, Any]] = activity_data.get("items", [])
    except APIClientError as err:
        st.error(f"Failed to load company activity: {err}")
        return

    # Filter controls
    col_search, col_filter = st.columns([2.5, 1.5])
    with col_search:
        search_kw = st.text_input("Search Events", placeholder="Search by actor, filename, or keyword...", key="act_search_input").strip().lower()

    with col_filter:
        action_types = ["All Events", "Document Operations", "Employee & Member Events", "Invitations"]
        chosen_type = st.selectbox("Filter Event Type", options=action_types, index=0, key="act_type_filter")

    # Apply filters
    filtered_events = events
    if search_kw:
        filtered_events = [
            e for e in filtered_events
            if search_kw in e.get("action", "").lower()
            or search_kw in str(e.get("actor_name", "")).lower()
            or search_kw in str(e.get("actor_email", "")).lower()
            or search_kw in str(e.get("details", "")).lower()
        ]

    if chosen_type == "Document Operations":
        filtered_events = [e for e in filtered_events if "document" in e.get("action", "").lower()]
    elif chosen_type == "Employee & Member Events":
        filtered_events = [e for e in filtered_events if "member" in e.get("action", "").lower() or "employee" in e.get("action", "").lower() or "user" in e.get("action", "").lower()]
    elif chosen_type == "Invitations":
        filtered_events = [e for e in filtered_events if "invitation" in e.get("action", "").lower()]

    st.markdown(f"#### Audit Events ({len(filtered_events)})")

    if not filtered_events:
        st.markdown(
            """
            <div class="ekn-empty-state">
                <div class="ekn-empty-title">No Activity Events Recorded</div>
                <div class="ekn-empty-desc">
                    Company-level operations such as document uploads, employee joins, and invitations will appear here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Table Header
    h_cols = st.columns([2.2, 2.0, 2.8, 1.2, 1.8])
    h_cols[0].markdown("**Event Type**")
    h_cols[1].markdown("**Actor**")
    h_cols[2].markdown("**Resource / Details**")
    h_cols[3].markdown("**Outcome**")
    h_cols[4].markdown("**Timestamp**")

    st.markdown("<hr style='margin: 0.2rem 0 0.8rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

    for ev in filtered_events:
        act_raw = ev.get("action", "")
        act_label = act_raw.replace(".", " ").replace("_", " ").title()

        actor_name = ev.get("actor_name") or ev.get("actor_email") or "System"
        actor_email = ev.get("actor_email") or ""

        outcome = ev.get("outcome", "success").lower()
        outcome_badge = "<span class='ekn-badge ekn-badge-online'>Success</span>" if outcome == "success" else f"<span class='ekn-badge ekn-badge-failed'>{outcome.title()}</span>"

        details = ev.get("details") or {}
        details_str = ""
        if isinstance(details, dict):
            if "filename" in details:
                details_str = f"📄 `{details['filename']}`"
            elif "email" in details:
                details_str = f"👤 `{details['email']}`"
            elif "role" in details:
                details_str = f"Role: `{details['role']}`"
            else:
                pairs = [f"{k}: {v}" for k, v in list(details.items())[:2]]
                details_str = ", ".join(pairs)
        elif details:
            details_str = str(details)[:60]

        res_id = ev.get("resource_id") or ""
        res_caption = f"ID: `{res_id[:8]}...`" if res_id and len(res_id) > 8 else ""

        cols = st.columns([2.2, 2.0, 2.8, 1.2, 1.8])

        with cols[0]:
            st.write(f"📌 **{act_label}**")
            st.caption(f"`{act_raw}`")

        with cols[1]:
            st.write(f"**{actor_name}**")
            if actor_email and actor_email != actor_name:
                st.caption(f"`{actor_email}`")

        with cols[2]:
            st.write(details_str or "—")
            if res_caption:
                st.caption(res_caption)

        with cols[3]:
            st.markdown(outcome_badge, unsafe_allow_html=True)

        with cols[4]:
            st.write(format_date(ev.get("created_at")))

        st.markdown("<hr style='margin: 0.3rem 0; border: none; border-top: 1px solid #f1f5f9;'/>", unsafe_allow_html=True)
