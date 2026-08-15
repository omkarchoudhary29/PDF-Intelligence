"""Team & Access Management View for Enterprise Knowledge Navigator.

Provides role-based access control (RBAC) administration:
- Listing workspace members and roles.
- Inviting new team members with secure joining codes.
- Updating member roles (Owner, Admin, Employee) with final-owner protection.
- Revoking or removing workspace access.
"""

from typing import Any, Dict, List, Optional
import streamlit as st

from frontend.api_client import APIClientError, BackendAPIClient
from frontend.styles import format_date
from frontend.views.invitations import render_invitations


def render_team_management(
    client: BackendAPIClient,
    workspace: Dict[str, Any],
    current_user_role: str = "owner",
    initial_subtab: Optional[str] = None,
) -> None:
    """Render the Workspace Team & Membership Administration dashboard."""
    ws_id = workspace["id"]
    ws_name = workspace["name"]
    can_manage = current_user_role in ("owner", "admin")

    col_title, col_refresh = st.columns([3, 1])
    with col_title:
        st.markdown(f"## Team & Access Management — {ws_name}")
        st.caption("Manage employee directory, role assignments, onboarding joining codes, and access policies.")
    with col_refresh:
        if st.button("🔄 Refresh Team", use_container_width=True):
            st.rerun()

    st.markdown("<hr style='margin: 0.8rem 0 1.25rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

    tab_employees, tab_invites = st.tabs(["👥 Employee Directory", "✉️ Invitations & Joining Codes"])

    # -------------------------------------------------------------------------
    # TAB 1: Employee Directory
    # -------------------------------------------------------------------------
    with tab_employees:
        try:
            members: List[Dict[str, Any]] = client.list_members(ws_id)
        except APIClientError as err:
            st.error(f"Failed to load workspace members: {err}")
            return

        # Filtering and Search
        col_s, col_r, col_st = st.columns([2.5, 1.2, 1.3])
        with col_s:
            search_query = st.text_input("Search Team", placeholder="Search by name or email...", key="team_search_kw").strip().lower()
        with col_r:
            role_filter = st.selectbox("Role Filter", options=["All Roles", "owner", "admin", "employee"], index=0, key="team_role_filter")
        with col_st:
            status_filter = st.selectbox("Status Filter", options=["All Statuses", "active", "suspended"], index=0, key="team_status_filter")

        filtered_members = members
        if search_query:
            filtered_members = [
                m for m in filtered_members
                if search_query in str(m.get("user", {}).get("full_name") or m.get("full_name", "")).lower()
                or search_query in str(m.get("user", {}).get("email") or m.get("email", "")).lower()
            ]
        if role_filter != "All Roles":
            filtered_members = [m for m in filtered_members if m.get("role") == role_filter]
        if status_filter != "All Statuses":
            filtered_members = [m for m in filtered_members if m.get("membership_status") == status_filter]

        st.markdown(f"#### Active Workspace Members ({len(filtered_members)})")

        if not filtered_members:
            st.markdown(
                """
                <div class="ekn-empty-state">
                    <div class="ekn-empty-title">No Members Match the Selected Filters</div>
                    <div class="ekn-empty-desc">Try clearing the search query or adjusting role and status filters.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Table Header
            h_cols = st.columns([2.5, 2.5, 1.5, 1.5, 2.0])
            h_cols[0].markdown("**Name**")
            h_cols[1].markdown("**Email**")
            h_cols[2].markdown("**Role**")
            h_cols[3].markdown("**Status**")
            h_cols[4].markdown("**Actions**")

            st.markdown("<hr style='margin: 0.2rem 0 0.8rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

            for m in filtered_members:
                u_id = m.get("user_id") or m.get("id")
                user_obj = m.get("user", {})
                full_name = user_obj.get("full_name") or m.get("full_name", "User")
                email = user_obj.get("email") or m.get("email", "")
                role = m.get("role", "employee")
                status = m.get("membership_status", "active")

                cols = st.columns([2.5, 2.5, 1.5, 1.5, 2.0])

                with cols[0]:
                    st.write(f"👤 **{full_name}**")
                    st.caption(f"Joined {format_date(m.get('created_at'))}")

                with cols[1]:
                    st.write(f"`{email}`")

                with cols[2]:
                    badge_class = "ekn-badge-ready" if role == "owner" else ("ekn-badge-online" if role == "admin" else "ekn-badge-neutral")
                    st.markdown(f"<span class='ekn-badge {badge_class}'>{role.upper()}</span>", unsafe_allow_html=True)

                with cols[3]:
                    status_class = "ekn-badge-online" if status == "active" else "ekn-badge-offline"
                    st.markdown(f"<span class='ekn-badge {status_class}'>{status.capitalize()}</span>", unsafe_allow_html=True)

                with cols[4]:
                    if can_manage:
                        act_col1, act_col2 = st.columns([1, 1])
                        with act_col1:
                            if current_user_role == "owner":
                                with st.popover("Edit Role"):
                                    new_role = st.selectbox(
                                        "Role",
                                        options=["owner", "admin", "employee"],
                                        index=["owner", "admin", "employee"].index(role),
                                        key=f"team_role_sel_{u_id}",
                                    )
                                    if st.button("Save", key=f"team_save_role_{u_id}", use_container_width=True):
                                        try:
                                            client.update_member(ws_id, u_id, role=new_role)
                                            st.success(f"Role updated to {new_role}")
                                            st.rerun()
                                        except APIClientError as err:
                                            st.error(str(err))
                            else:
                                st.empty()

                        with act_col2:
                            # Suspend/Activate toggle (prevent owner from suspending self)
                            is_self = client.current_user and client.current_user.get("id") == u_id
                            if not is_self and current_user_role in ("owner", "admin"):
                                if status == "active":
                                    if st.button("Suspend", key=f"team_susp_{u_id}", use_container_width=True):
                                        try:
                                            client.update_member(ws_id, u_id, membership_status="suspended")
                                            st.warning(f"{full_name} suspended.")
                                            st.rerun()
                                        except APIClientError as err:
                                            st.error(str(err))
                                else:
                                    if st.button("Activate", key=f"team_act_{u_id}", use_container_width=True):
                                        try:
                                            client.update_member(ws_id, u_id, membership_status="active")
                                            st.success(f"{full_name} activated.")
                                            st.rerun()
                                        except APIClientError as err:
                                            st.error(str(err))
                            else:
                                st.empty()
                    else:
                        st.caption("View only")

    # -------------------------------------------------------------------------
    # TAB 2: Invitations & Joining Codes
    # -------------------------------------------------------------------------
    with tab_invites:
        render_invitations(client, workspace)
