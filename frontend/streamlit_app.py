"""Enterprise Knowledge Navigator (EKN) — Streamlit Enterprise Application Shell.

Provides unified enterprise branding, pale lavender sidebar navigation, workspace isolation,
backend connectivity monitoring, user authentication, and routing across
Overview, Documents, Knowledge Review, Knowledge Graph, Team, and Reports & Settings.
"""

from pathlib import Path
import html
import sys
from typing import Any, Dict, List, Optional
import streamlit as st

# Ensure project root is on sys.path for direct imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from frontend.api_client import APIClientError, BackendAPIClient
from frontend.navigation import (
    PRIMARY_NAV_DESTINATIONS,
    VIEW_ALIAS_MAP,
    navigate_to,
    normalize_view,
    on_sidebar_nav_change,
    sync_navigation_state,
)
from frontend.styles import ENTERPRISE_CSS
from frontend.views.chat import render_ask_knowledge_base
from frontend.views.documents import render_documents
from frontend.views.employee_workspace import render_employee_workspace
from frontend.views.employees import render_employees
from frontend.views.invitations import render_invitations
from frontend.views.activity import render_activity
from frontend.views.knowledge_graph import render_knowledge_graph
from frontend.views.knowledge_review import render_knowledge_review
from frontend.views.login import render_login
from frontend.views.overview import render_overview
from frontend.views.reports import render_reports
from frontend.views.reports_settings import render_reports_settings
from frontend.views.team import render_team_management

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Enterprise Knowledge Navigator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom enterprise CSS
st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)


def main() -> None:
    settings = get_settings()
    client = BackendAPIClient(settings.backend_api_url)

    # Process query parameters for navigation, demo login, and testing
    qp = st.query_params
    if "logout" in qp and qp.get("logout") == "true":
        try:
            client.logout()
        except Exception:
            pass
        target_portal = qp.get("portal", "role_selection")
        st.session_state.clear()
        st.session_state["auth_portal"] = target_portal
        st.query_params.pop("logout", None)
        st.rerun()

    if settings.ekn_demo_mode:
        admin_demo_email = "admin.demo@example.com"
        emp_demo_email = "employee.demo@example.com"
        if qp.get("demo_login") == "admin" and not st.session_state.get("auth_token"):
            try:
                token_data = client.login(
                    admin_demo_email,
                    settings.demo_admin_password,
                    portal="admin",
                )
                st.session_state["auth_token"] = token_data["access_token"]
                st.session_state["refresh_token"] = token_data.get("refresh_token")
                client.access_token = token_data["access_token"]
                auth_user = client.get_current_user_profile()
                st.session_state["auth_user"] = auth_user
                st.session_state["auth_portal"] = "admin"
                st.session_state["user_role"] = "admin"
                if settings.demo_workspace_id:
                    st.session_state["selected_workspace_id"] = settings.demo_workspace_id
            except Exception:
                pass
        elif qp.get("demo_login") == "employee" and not st.session_state.get("auth_token"):
            try:
                token_data = client.login(
                    emp_demo_email,
                    settings.demo_employee_password,
                    portal="employee",
                )
                st.session_state["auth_token"] = token_data["access_token"]
                st.session_state["refresh_token"] = token_data.get("refresh_token")
                client.access_token = token_data["access_token"]
                auth_user = client.get_current_user_profile()
                st.session_state["auth_user"] = auth_user
                st.session_state["auth_portal"] = "employee"
                st.session_state["user_role"] = "employee"
                if settings.demo_workspace_id:
                    st.session_state["selected_workspace_id"] = settings.demo_workspace_id
            except Exception:
                pass

    if "portal" in qp:
        st.session_state["auth_portal"] = qp.get("portal")
    if "view" in qp:
        req_view = qp.get("view")
        navigate_to(req_view)
    if "conv_id" in qp:
        st.session_state["active_conversation_id"] = qp.get("conv_id")
    if "emp_right_open" in qp:
        st.session_state["emp_right_open"] = (qp.get("emp_right_open", "false").lower() == "true")
    if "emp_right_view" in qp:
        req_rv = qp.get("emp_right_view")
        st.session_state["emp_right_view"] = req_rv
        st.session_state["emp_right_view_selector"] = req_rv

    # Check for authenticated session in session_state (no sensitive tokens in URL parameters)
    token = st.session_state.get("auth_token")
    if token:
        client.access_token = token
        client.refresh_token = st.session_state.get("refresh_token")
        client.current_user = st.session_state.get("auth_user")

    # Clear any residual session tokens from query params to avoid exposure
    if "session_token" in st.query_params:
        st.query_params.pop("session_token", None)

    # If unauthenticated, render login view
    if not client.access_token:
        render_login(client)
        return

    auth_user = st.session_state.get("auth_user") or {}
    # Ensure profile has workspace memberships loaded from /api/v1/auth/me
    if client.access_token and "workspaces" not in auth_user:
        try:
            auth_user = client.get_current_user_profile()
            st.session_state["auth_user"] = auth_user
        except Exception:
            pass

    user_workspaces: List[Dict[str, Any]] = auth_user.get("workspaces", [])
    membership_map: Dict[str, Dict[str, Any]] = {
        w["workspace_id"]: w
        for w in user_workspaces
        if isinstance(w, dict) and "workspace_id" in w
    }

    # Restrict workspaces strictly to the authenticated user's active memberships
    is_online, _ = client.check_health()
    all_backend_workspaces: List[Dict[str, Any]] = []
    if is_online:
        try:
            all_backend_workspaces = client.list_workspaces()
        except APIClientError:
            pass

    # Filter to authorized memberships
    workspaces: List[Dict[str, Any]] = [
        w for w in all_backend_workspaces
        if w.get("id") in membership_map
    ]

    # Reconstruct from membership profile if backend list was empty or failed
    if not workspaces and membership_map:
        workspaces = [
            {
                "id": m["workspace_id"],
                "name": m.get("workspace_name", "Workspace"),
                "slug": m.get("workspace_slug", ""),
                "document_count": m.get("document_count", 0),
            }
            for m in user_workspaces
            if isinstance(m, dict) and "workspace_id" in m
        ]

    # Resolve active selected workspace strictly against authorized memberships
    selected_workspace = None
    if workspaces:
        saved_ws_id = st.session_state.get("selected_workspace_id")
        if saved_ws_id and saved_ws_id in membership_map:
            selected_workspace = next((w for w in workspaces if w.get("id") == saved_ws_id), workspaces[0])
        else:
            selected_workspace = workspaces[0]
            st.session_state["selected_workspace_id"] = selected_workspace.get("id")

    # Resolve active role strictly from membership association
    current_user_role = None
    is_admin = False
    if selected_workspace:
        sel_id = selected_workspace.get("id")
        if sel_id and sel_id in membership_map:
            membership = membership_map[sel_id]
            raw_role = str(membership.get("role", "")).lower()
            if raw_role in ("owner", "admin"):
                current_user_role = raw_role
                is_admin = True
            elif raw_role == "employee":
                current_user_role = "employee"
                is_admin = False
            else:
                # Invalid role string
                selected_workspace = None
                current_user_role = None
                is_admin = False
        else:
            # Unauthorized workspace selection attempt: reject immediately
            selected_workspace = None
            current_user_role = None
            is_admin = False

    # If authenticated account is an employee (not owner/admin), direct immediately into Employee Workspace
    if not is_admin:
        if not selected_workspace:
            st.info("No active company workspace available. Please ask your administrator for an invitation code.")
            if st.button("Sign Out"):
                client.logout()
                st.session_state.clear()
                st.session_state["auth_portal"] = "role_selection"
                st.query_params.clear()
                st.rerun()
            return
        render_employee_workspace(client, selected_workspace)
        return

    # Initialize session state defaults for Administrator
    if "active_view" not in st.session_state:
        st.session_state["active_view"] = "Overview"

    # --- Sidebar: Brand Header, Company Selector, Navigation, and Admin Profile ---
    with st.sidebar:
        # EKN Logo and Wordmark
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 0.65rem; padding: 0.2rem 0.2rem 0.8rem 0.2rem; border-bottom: 1px solid #e2e8f0; margin-bottom: 0.85rem;">
                <div style="width: 36px; height: 36px; border-radius: 8px; background: #4f46e5; display: flex; align-items: center; justify-content: center; color: #ffffff; font-size: 1.2rem; font-weight: 700; box-shadow: 0 2px 4px rgba(79, 70, 229, 0.25);">📄</div>
                <div>
                    <div style="font-weight: 800; color: #0f172a; font-size: 1.05rem; letter-spacing: -0.02em; line-height: 1.2;">EKN</div>
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Knowledge Navigator</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Company Selector
        st.markdown("<div style='font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em; margin-bottom: 0.25rem;'>Company Workspace</div>", unsafe_allow_html=True)

        if workspaces:
            workspace_ids = [w["id"] for w in workspaces if "id" in w]
            workspace_labels = {
                w["id"]: f"{w.get('name', 'Workspace')} ({w.get('document_count', 0)} docs)"
                for w in workspaces
                if "id" in w
            }

            saved_ws_id = st.session_state.get("selected_workspace_id")
            default_idx = workspace_ids.index(saved_ws_id) if saved_ws_id in workspace_ids else 0

            chosen_ws_id = st.selectbox(
                "Company Workspace",
                options=workspace_ids,
                index=default_idx,
                format_func=lambda wid: workspace_labels.get(wid, wid),
                label_visibility="collapsed",
                key="workspace_dropdown",
            )

            if chosen_ws_id != st.session_state.get("selected_workspace_id"):
                st.session_state["selected_workspace_id"] = chosen_ws_id
                # Clear conversation and scope context on workspace transition
                st.session_state["active_conversation_id"] = None
                st.session_state.pop("pending_query", None)
                st.session_state.pop("pending_emp_query", None)
                st.session_state.pop("pending_new_chat_scope", None)
                st.session_state.pop("latest_citations", None)
                st.session_state.pop("citations_by_conversation", None)
                st.rerun()

            selected_workspace = next((w for w in workspaces if w.get("id") == chosen_ws_id), None)
            if selected_workspace:
                sel_ws_id = selected_workspace.get("id")
                membership = membership_map.get(sel_ws_id)
                if membership:
                    raw_role = str(membership.get("role", "employee")).lower()
                    if raw_role in ("owner", "admin"):
                        current_user_role = raw_role
                        is_admin = True

                if is_admin:
                    st.write("")
                    if st.button("📤 Upload PDF Document", type="primary", use_container_width=True, key="sidebar_quick_upload_btn"):
                        navigate_to("Documents", doc_open_upload=True)
                        st.rerun()
        else:
            if is_online:
                st.info("No workspaces available. Create your first workspace below.")
            else:
                st.warning("Backend offline. Cannot load workspaces.")

        st.markdown("<hr style='margin: 0.8rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

        # 6 Core Navigation Sections matching Approved Specification
        st.markdown("<div style='font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em; margin-bottom: 0.35rem;'>Main Navigation</div>", unsafe_allow_html=True)
        nav_options = PRIMARY_NAV_DESTINATIONS
        nav_icons = {
            "Overview": "📊",
            "Documents": "📁",
            "Knowledge Review": "⚠️",
            "Knowledge Graph": "🕸️",
            "Team": "👥",
            "Reports & Settings": "⚙️",
        }

        # Synchronize active_view and sidebar_navigation_radio before widget instantiation
        normalized_current = sync_navigation_state()
        radio_index = nav_options.index(normalized_current) if normalized_current in nav_options else 0

        st.radio(
            "Navigation",
            options=nav_options,
            index=radio_index,
            format_func=lambda opt: f"{nav_icons.get(opt, '📄')} {opt}",
            label_visibility="collapsed",
            key="sidebar_navigation_radio",
            on_change=on_sidebar_nav_change,
        )

        st.markdown("<hr style='margin: 1.2rem 0 0.8rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

        # Administrator Profile at Bottom
        admin_full_name = auth_user.get("full_name", "Administrator")
        admin_email = auth_user.get("email", "")
        role_label = f"Administrator ({current_user_role.title()})" if current_user_role else "Administrator"

        st.markdown(
            f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; padding: 0.75rem 0.85rem; border-radius: 8px; margin-bottom: 0.6rem; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                    <div style="font-weight: 700; color: #0f172a; font-size: 0.92rem;">👤 {admin_full_name}</div>
                </div>
                <div style="color: #64748b; font-size: 0.78rem; margin-bottom: 0.35rem; overflow: hidden; text-overflow: ellipsis;">{admin_email}</div>
                <div><span class='ekn-badge ekn-badge-ready'>{role_label}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        logout_col1, logout_col2 = st.columns([1, 1])
        with logout_col1:
            if st.button("Sign Out", key="sidebar_logout_btn", use_container_width=True):
                client.logout()
                st.session_state.clear()
                st.session_state["auth_portal"] = "role_selection"
                st.query_params.clear()
                st.rerun()

        with logout_col2:
            with st.popover("Password"):
                with st.form("sidebar_change_pwd_form"):
                    old_p = st.text_input("Current", type="password")
                    new_p = st.text_input("New", type="password")
                    save_p = st.form_submit_button("Update", use_container_width=True)
                    if save_p:
                        try:
                            client.change_password(old_p, new_p)
                            st.success("Updated!")
                        except APIClientError as e:
                            st.error(str(e))

        # Workspace Creation (clean popover when workspaces exist, or banner if none)
        if not bool(workspaces):
            with st.expander("➕ Create First Workspace", expanded=True):
                with st.form("create_workspace_form", clear_on_submit=True):
                    new_name = st.text_input("Workspace Name*", placeholder="e.g. Engineering Manuals")
                    new_slug = st.text_input("Slug (Optional)", placeholder="e.g. engineering-manuals")
                    new_desc = st.text_area("Description", placeholder="Scope of documents in this workspace...", height=68)
                    submitted = st.form_submit_button("Create Workspace", use_container_width=True)

                    if submitted:
                        if not new_name.strip():
                            st.error("Workspace name is required.")
                        elif not is_online:
                            st.error("Cannot create workspace: Backend is offline.")
                        else:
                            try:
                                created = client.create_workspace(
                                    name=new_name.strip(),
                                    description=new_desc.strip() if new_desc else None,
                                    slug=new_slug.strip() if new_slug else None,
                                )
                                st.session_state["selected_workspace_id"] = created["id"]
                                try:
                                    refreshed_profile = client.get_current_user_profile()
                                    st.session_state["auth_user"] = refreshed_profile
                                except Exception:
                                    pass
                                st.success(f"Workspace '{created['name']}' created!")
                                st.rerun()
                            except APIClientError as err:
                                st.error(f"Creation failed: {err}")

    # --- Main Content Canvas ---
    if not is_online:
        st.error(
            f"**Backend Server Offline:** Cannot communicate with `{settings.backend_api_url}`. "
            "Please ensure the FastAPI backend is running on port 8010."
        )
        return

    if not selected_workspace:
        st.markdown(
            """
            <div class="ekn-empty-state">
                <div class="ekn-empty-title">No Workspace Selected</div>
                <div class="ekn-empty-desc">Get started by creating your first workspace using the sidebar.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    ws_id = selected_workspace["id"]
    ws_name = selected_workspace.get("name", "Workspace")
    ws_name_escaped = html.escape(ws_name)
    active_view = st.session_state.get("active_view", "Overview")
    active_view_escaped = html.escape(active_view)

    # Safe truncation of workspace name if long to prevent layout overflow
    display_ws_name = ws_name_escaped
    if len(ws_name) > 28:
        display_ws_name = f"<span title='{ws_name_escaped}'>{html.escape(ws_name[:26])}…</span>"

    # --- TOP HEADER: Breadcrumbs, Global Search, Notification Indicator ---
    hdr_col_left, hdr_col_search, hdr_col_right = st.columns([1.3, 2.5, 1.0])

    with hdr_col_left:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 0.45rem; padding: 0.35rem 0;">
                <span style="font-size: 1.15rem;">🏢</span>
                <span style="font-weight: 700; color: #1e3a8a; font-size: 0.98rem;">{display_ws_name}</span>
                <span style="color: #cbd5e1;">/</span>
                <span style="font-weight: 600; color: #4f46e5; font-size: 0.95rem;">{active_view_escaped}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with hdr_col_search:
        with st.form("top_header_global_search_form", clear_on_submit=False):
            gs_col1, gs_col2 = st.columns([3.2, 1.2])
            with gs_col1:
                global_q = st.text_input(
                    "Search company knowledge...",
                    placeholder="Search documents, entities, specifications...",
                    label_visibility="collapsed",
                    key="global_top_search_input",
                )
            with gs_col2:
                gs_submit = st.form_submit_button("Search", use_container_width=True)

        if gs_submit and global_q.strip():
            st.session_state["pending_query"] = global_q.strip()
            navigate_to("Ask Knowledge Base")
            st.rerun()

    with hdr_col_right:
        # Determine actual backend & knowledge-review status without fabricating health
        if not is_online:
            notif_text = "System Offline"
            notif_badge = "ekn-badge-failed"
            notif_icon = "🔴"
        else:
            try:
                ws_stats = client.get_workspace_stats(ws_id)
                p_rev = ws_stats.get("pending_reviews", 0)
                if p_rev > 0:
                    notif_text = f"{p_rev} Review(s) Pending"
                    notif_badge = "ekn-badge-failed"
                    notif_icon = "⚠️"
                else:
                    notif_text = "All Systems Normal"
                    notif_badge = "ekn-badge-online"
                    notif_icon = "✓"
            except Exception:
                # If stats API call fails, do NOT claim 'All Systems Normal'
                notif_text = "Stats Unavailable"
                notif_badge = "ekn-badge-neutral"
                notif_icon = "⚠️"

        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; align-items: center; gap: 0.6rem; padding: 0.35rem 0;">
                <span class="ekn-badge {notif_badge}">{notif_icon} {notif_text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin: 0.3rem 0 1.25rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

    # --- Main Content Router ---
    if active_view == "Overview":
        render_overview(client, selected_workspace)
    elif active_view == "Documents" or active_view == "Company Documents":
        render_documents(client, selected_workspace, current_user_role or "admin", initial_tab="library")
    elif active_view == "Document Families & Versions":
        render_documents(client, selected_workspace, current_user_role or "admin", initial_tab="families")
    elif active_view == "Knowledge Review" or active_view == "Contradiction Review":
        render_knowledge_review(client, selected_workspace, current_user_role or "admin")
    elif active_view == "Knowledge Graph":
        render_knowledge_graph(client, selected_workspace)
    elif active_view == "Team" or active_view == "Employees":
        render_team_management(client, selected_workspace, current_user_role or "admin")
    elif active_view == "Invitations":
        render_team_management(client, selected_workspace, current_user_role or "admin", initial_subtab="invitations")
    elif active_view == "Reports & Settings" or active_view == "Reports":
        render_reports_settings(client, selected_workspace, current_user_role or "admin")
    elif active_view == "Activity":
        render_reports_settings(client, selected_workspace, current_user_role or "admin", initial_subtab="activity")
    elif active_view == "Ask Knowledge Base":
        render_ask_knowledge_base(client, selected_workspace)
    else:
        render_overview(client, selected_workspace)


if __name__ == "__main__":
    main()
