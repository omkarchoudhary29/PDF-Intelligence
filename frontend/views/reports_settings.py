"""Reports & Settings Consolidated View for Enterprise Knowledge Navigator.

Combines Knowledge Dossier generation, Company Activity Audit Logging, and
Company Workspace Configuration into one unified administrative control center.
"""

from typing import Any, Dict, List, Optional
import streamlit as st

from config.settings import get_settings
from frontend.api_client import APIClientError, BackendAPIClient
from frontend.styles import format_date
from frontend.views.activity import render_activity
from frontend.views.reports import render_reports


def render_reports_settings(
    client: BackendAPIClient,
    workspace: Dict[str, Any],
    current_user_role: str = "owner",
    initial_subtab: Optional[str] = None,
) -> None:
    """Render the Reports, Activity, and Company Settings interface."""
    settings = get_settings()
    ws_id = workspace["id"]
    ws_name = workspace["name"]
    can_manage = current_user_role in ("owner", "admin")

    col_title, col_refresh = st.columns([3, 1])
    with col_title:
        st.markdown(f"## Reports & Company Settings — {ws_name}")
        st.caption("Generate executive knowledge dossiers, review security activity audit trails, and manage company settings.")
    with col_refresh:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    st.markdown("<hr style='margin: 0.6rem 0 1.25rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

    tab_titles = ["📋 Knowledge Dossiers & Reports", "📜 Company Activity History", "⚙️ Company Settings"]
    tabs = st.tabs(tab_titles)

    # 1. Reports Tab
    with tabs[0]:
        render_reports(client, workspace, current_user_role)

    # 2. Activity Tab
    with tabs[1]:
        render_activity(client, workspace)

    # 3. Company Settings Tab
    with tabs[2]:
        st.markdown("#### Company Profile & Workspace Settings")
        st.write("Configure organizational details, review onboarding security policies, and update administrative credentials.")

        # Workspace Profile Form
        with st.form("company_settings_form"):
            st.markdown("**Company Information**")
            new_comp_name = st.text_input(
                "Company Name*",
                value=workspace.get("name", ""),
                disabled=not can_manage,
            )
            new_comp_desc = st.text_area(
                "Company Description",
                value=workspace.get("description") or "",
                height=80,
                disabled=not can_manage,
            )

            st.markdown(
                f"""
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:0.75rem 1rem; margin:0.8rem 0; font-size:0.85rem; color:#475569;">
                    <div><strong>Workspace ID:</strong> <code>{ws_id}</code></div>
                    <div style="margin-top:0.3rem;"><strong>Slug:</strong> <code>{workspace.get('slug', '')}</code></div>
                    <div style="margin-top:0.3rem;"><strong>Created At:</strong> {format_date(workspace.get('created_at', ''))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            save_settings = st.form_submit_button("Save Company Changes", type="primary", disabled=not can_manage)
            if save_settings:
                if not new_comp_name.strip():
                    st.error("Company name cannot be empty.")
                else:
                    try:
                        updated_ws = client.update_workspace(
                            ws_id,
                            name=new_comp_name.strip(),
                            description=new_comp_desc.strip() if new_comp_desc.strip() else None,
                        )
                        st.success(f"Company settings for '{updated_ws['name']}' saved successfully!")
                        st.session_state["selected_workspace_id"] = updated_ws["id"]
                        st.rerun()
                    except APIClientError as err:
                        st.error(f"Failed to update company settings: {err}")

        st.markdown("<hr style='margin: 1.5rem 0 1rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

        # Onboarding & Security Policy Section
        st.markdown("#### Onboarding & Invitation Policy")
        is_demo = bool(settings.ekn_demo_mode)
        if is_demo:
            st.markdown(
                """
                <div style="background:#eff6ff; border:1px solid #bfdbfe; border-radius:6px; padding:0.85rem 1rem; font-size:0.88rem; color:#1e40af;">
                    <strong>Demonstration Mode Active:</strong> New employees can self-register without requiring an invitation joining code directly into this designated demo workspace. Production invitation enforcement is bypassed for interview evaluation.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:6px; padding:0.85rem 1rem; font-size:0.88rem; color:#166534;">
                    <strong>Production Mode Active:</strong> Strict company joining code policy is enforced. Public self-registration is disabled; employees must present an active, non-expired joining code generated in the Team section.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='margin: 1.5rem 0 1rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

        # Administrator Password Security
        st.markdown("#### Administrator Account Security")
        with st.expander("Change Administrator Password", expanded=False):
            with st.form("admin_password_change_form"):
                curr_pwd = st.text_input("Current Password", type="password")
                new_pwd = st.text_input("New Password (12+ characters, uppercase, lowercase, digit, symbol)", type="password")
                conf_pwd = st.text_input("Confirm New Password", type="password")
                pwd_submit = st.form_submit_button("Update Password", type="primary")

                if pwd_submit:
                    if not curr_pwd or not new_pwd:
                        st.error("All password fields are required.")
                    elif new_pwd != conf_pwd:
                        st.error("New passwords do not match.")
                    elif len(new_pwd) < 12:
                        st.error("New password must be at least 12 characters long.")
                    else:
                        try:
                            client.change_password(curr_pwd, new_pwd)
                            st.success("Administrator password updated successfully!")
                        except APIClientError as err:
                            st.error(f"Password update failed: {err}")
