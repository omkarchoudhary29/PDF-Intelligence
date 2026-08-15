"""Employee Access and Management View for Enterprise Knowledge Navigator.

Provides employee directory, status management (active/deactivated), role administration,
and workspace membership controls strictly scoped to the administrator's company.
"""

from typing import Any, Dict, List
import streamlit as st

from frontend.api_client import APIClientError, BackendAPIClient
from frontend.styles import format_date


def render_employees(
    client: BackendAPIClient,
    workspace: Dict[str, Any],
    current_user_role: str = "owner",
) -> None:
    """Render the Employee Management view for company administrators."""
    ws_id = workspace["id"]
    ws_name = workspace["name"]

    col_title, col_actions = st.columns([3, 1])
    with col_title:
        st.markdown(f"## Employee Management — {ws_name}")
        st.caption("Manage employee access, review team directory, and configure workspace roles.")
    with col_actions:
        if st.button("🔄 Refresh Directory", use_container_width=True):
            st.rerun()

    st.markdown("<hr style='margin: 0.6rem 0 1.25rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

    # 1. Fetch Members
    try:
        members: List[Dict[str, Any]] = client.list_members(ws_id)
    except APIClientError as err:
        st.error(f"Failed to load company members: {err}")
        return

    # 2. Filters
    col_search, col_status = st.columns([2.5, 1.5])
    with col_search:
        search_query = st.text_input(
            "Search Employees",
            placeholder="Search by employee name or email address...",
            key="emp_search_input",
        ).strip().lower()

    with col_status:
        status_filter = st.selectbox(
            "Filter by Status",
            options=["All", "Active", "Suspended / Deactivated"],
            index=0,
            key="emp_status_filter",
        )

    # Filter members list
    filtered_members = members
    if search_query:
        filtered_members = [
            m for m in filtered_members
            if search_query in str(m.get("full_name", "")).lower()
            or search_query in str(m.get("email", "")).lower()
        ]

    if status_filter == "Active":
        filtered_members = [m for m in filtered_members if m.get("membership_status") == "active"]
    elif status_filter == "Suspended / Deactivated":
        filtered_members = [m for m in filtered_members if m.get("membership_status") != "active"]

    st.markdown(f"#### Company Directory ({len(filtered_members)} members)")

    if not filtered_members:
        st.markdown(
            """
            <div class="ekn-empty-state">
                <div class="ekn-empty-title">No Employees Found</div>
                <div class="ekn-empty-desc">
                    No team members match your current search criteria.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_inv1, col_inv2, col_inv3 = st.columns([1, 2, 1])
        with col_inv2:
            if st.button("✉️ Invite New Employee via Joining Code", type="primary", use_container_width=True):
                st.session_state["active_view"] = "Invitations"
                st.rerun()
        return

    # Table Header
    h_cols = st.columns([2.5, 2.5, 1.5, 1.5, 2.0])
    h_cols[0].markdown("**Employee Name**")
    h_cols[1].markdown("**Work Email**")
    h_cols[2].markdown("**Company Role**")
    h_cols[3].markdown("**Access Status**")
    h_cols[4].markdown("**Actions**")

    st.markdown("<hr style='margin: 0.2rem 0 0.8rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

    for m in filtered_members:
        u_id = m.get("user_id") or m.get("id")
        user_obj = m.get("user", {})
        full_name = user_obj.get("full_name") or m.get("full_name", "Employee")
        email = user_obj.get("email") or m.get("email", "")
        role = m.get("role", "employee")
        status = m.get("membership_status", "active")
        is_active = (status == "active")

        cols = st.columns([2.5, 2.5, 1.5, 1.5, 2.0])

        with cols[0]:
            st.write(f"👤 **{full_name}**")
            st.caption(f"Joined: {format_date(m.get('created_at'))}")

        with cols[1]:
            st.write(f"`{email}`")

        with cols[2]:
            badge_class = "ekn-badge-ready" if role == "owner" else ("ekn-badge-online" if role == "admin" else "ekn-badge-neutral")
            st.markdown(f"<span class='ekn-badge {badge_class}'>{role.upper()}</span>", unsafe_allow_html=True)

        with cols[3]:
            status_class = "ekn-badge-online" if is_active else "ekn-badge-offline"
            status_label = "Active" if is_active else status.title()
            st.markdown(f"<span class='ekn-badge {status_class}'>{status_label}</span>", unsafe_allow_html=True)

        with cols[4]:
            act_c1, act_c2 = st.columns([1, 1])

            # Status toggle / Deactivate / Reactivate
            with act_c1:
                if role != "owner":
                    if is_active:
                        with st.popover("Suspend"):
                            st.write(f"Suspend access for **{full_name}**?")
                            if st.button("Confirm Suspend", key=f"sus_{u_id}", type="primary", use_container_width=True):
                                try:
                                    client.update_member(ws_id, u_id, membership_status="suspended")
                                    st.success("Access suspended.")
                                    st.rerun()
                                except APIClientError as err:
                                    st.error(str(err))
                    else:
                        if st.button("Reactivate", key=f"reac_{u_id}", use_container_width=True):
                            try:
                                client.update_member(ws_id, u_id, membership_status="active")
                                st.success("Access reactivated.")
                                st.rerun()
                            except APIClientError as err:
                                st.error(str(err))

            # Edit Role / Remove
            with act_c2:
                can_remove = (current_user_role == "owner" and role != "owner") or (current_user_role == "admin" and role == "employee")
                if can_remove:
                    with st.popover("Remove"):
                        st.write(f"Remove **{full_name}** from company?")
                        if st.button("Confirm Remove", key=f"rm_emp_{u_id}", type="primary", use_container_width=True):
                            try:
                                client.remove_member(ws_id, u_id)
                                st.success("Removed member.")
                                st.rerun()
                            except APIClientError as err:
                                st.error(str(err))
                elif current_user_role == "owner":
                    with st.popover("Role"):
                        new_role = st.selectbox(
                            "Select Role",
                            options=["owner", "admin", "employee"],
                            index=["owner", "admin", "employee"].index(role),
                            key=f"edit_role_{u_id}",
                        )
                        if st.button("Save", key=f"save_r_{u_id}", use_container_width=True):
                            try:
                                client.update_member(ws_id, u_id, role=new_role)
                                st.success("Role updated.")
                                st.rerun()
                            except APIClientError as err:
                                st.error(str(err))

        st.markdown("<hr style='margin: 0.3rem 0; border: none; border-top: 1px solid #f1f5f9;'/>", unsafe_allow_html=True)
