"""Invitations and Joining Codes View for Enterprise Knowledge Navigator.

Provides employee invitation code generation, one-time raw code display,
active code listing, and code revocation scoped to the administrator's company.
"""

from typing import Any, Dict, List
import streamlit as st

from frontend.api_client import APIClientError, BackendAPIClient
from frontend.styles import format_date


def render_invitations(
    client: BackendAPIClient,
    workspace: Dict[str, Any],
) -> None:
    """Render the Invitations and Joining Codes view for company administrators."""
    ws_id = workspace["id"]
    ws_name = workspace["name"]

    col_title, col_refresh = st.columns([3, 1])
    with col_title:
        st.markdown(f"## Employee Invitations & Joining Codes — {ws_name}")
        st.caption("Generate secure joining codes for employee onboarding and manage active invitations.")
    with col_refresh:
        if st.button("🔄 Refresh Invitations", use_container_width=True):
            st.rerun()

    st.markdown("<hr style='margin: 0.6rem 0 1.25rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

    # 1. Generate Code Section
    with st.container(border=True):
        st.markdown("#### ➕ Generate New Employee Joining Code")
        st.caption("Create an invitation token that allows new employees to sign up and join this company workspace.")

        with st.form("generate_code_form"):
            c1, c2, c3 = st.columns([1.5, 1.5, 1.5])
            with c1:
                usage_limit = st.number_input("Maximum Usages", min_value=1, max_value=500, value=10, help="How many employees can register with this code.")
            with c2:
                expires_days = st.number_input("Valid For (Days)", min_value=1, max_value=90, value=7, help="Number of days before the code expires.")
            with c3:
                intended_role = st.selectbox("Role for New Signups", options=["employee", "admin"], index=0, help="Role automatically granted upon signup.")

            gen_submit = st.form_submit_button("Generate Joining Code", type="primary", use_container_width=True)

            if gen_submit:
                try:
                    inv_res = client.create_invitation(
                        workspace_id=ws_id,
                        max_usage_count=int(usage_limit),
                        expires_in_days=int(expires_days),
                        intended_role=intended_role,
                    )
                    raw_code = inv_res.get("raw_code")
                    st.session_state["phase6_last_code"] = raw_code
                    st.success("New joining code generated successfully!")
                    st.rerun()
                except APIClientError as err:
                    st.error(f"Failed to generate invitation code: {err}")

    # One-Time Code Display Notice
    last_code = st.session_state.get("phase6_last_code")
    if last_code:
        st.markdown(
            f"""
            <div style="background: #eff6ff; border: 1.5px solid #3b82f6; border-radius: 8px; padding: 1.2rem; margin: 1.2rem 0;">
                <div style="font-weight: 700; color: #1d4ed8; font-size: 1rem;">🔑 Newly Generated Joining Code:</div>
                <div style="font-family: monospace; font-size: 1.4rem; font-weight: 700; color: #1e3a8a; letter-spacing: 0.1em; margin: 0.5rem 0;">
                    {last_code}
                </div>
                <div style="color: #475569; font-size: 0.85rem;">
                    Share this code securely with new employees. For security, the plaintext code is never stored in the database and will not be displayed again.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Dismiss Code Display", key="dismiss_code_btn"):
            st.session_state.pop("phase6_last_code", None)
            st.rerun()

    # 2. Invitations Listing
    st.markdown("#### Company Joining Codes History")
    try:
        inv_data = client.list_invitations(ws_id)
        inv_items: List[Dict[str, Any]] = inv_data.get("items", []) if isinstance(inv_data, dict) else []

        if not inv_items:
            st.markdown(
                """
                <div class="ekn-empty-state">
                    <div class="ekn-empty-title">No Joining Codes Created</div>
                    <div class="ekn-empty-desc">
                        Use the generator above to create an onboarding code for your employees.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        # Table Header
        h_cols = st.columns([2.0, 1.5, 1.5, 2.0, 1.5, 1.5])
        h_cols[0].markdown("**Intended Role**")
        h_cols[1].markdown("**Usage**")
        h_cols[2].markdown("**Status**")
        h_cols[3].markdown("**Expiry Date**")
        h_cols[4].markdown("**Created**")
        h_cols[5].markdown("**Actions**")

        st.markdown("<hr style='margin: 0.2rem 0 0.8rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

        for inv in inv_items:
            inv_id = inv["id"]
            is_active = inv.get("is_valid", False)
            is_revoked = inv.get("is_revoked", False)
            used_cnt = inv.get("current_usage_count", 0)
            max_cnt = inv.get("max_usage_count", 1)

            if is_revoked:
                status_badge = "<span class='ekn-badge ekn-badge-failed'>Revoked</span>"
            elif not is_active:
                status_badge = "<span class='ekn-badge ekn-badge-neutral'>Expired / Exhausted</span>"
            else:
                status_badge = "<span class='ekn-badge ekn-badge-ready'>Active</span>"

            cols = st.columns([2.0, 1.5, 1.5, 2.0, 1.5, 1.5])

            with cols[0]:
                st.write(f"💼 **{inv.get('intended_role', 'employee').title()}**")
                st.caption(f"ID: `{inv_id[:8]}...`")

            with cols[1]:
                st.write(f"**{used_cnt}** / {max_cnt} used")

            with cols[2]:
                st.markdown(status_badge, unsafe_allow_html=True)

            with cols[3]:
                st.write(format_date(inv.get("expires_at")))

            with cols[4]:
                st.caption(format_date(inv.get("created_at")))

            with cols[5]:
                if not is_revoked and is_active:
                    with st.popover("Revoke"):
                        st.write("Revoke this code immediately?")
                        if st.button("Confirm Revoke", key=f"rev_btn_{inv_id}", type="primary", use_container_width=True):
                            try:
                                client.revoke_invitation(ws_id, inv_id)
                                st.success("Joining code revoked.")
                                st.rerun()
                            except APIClientError as err:
                                st.error(str(err))
                else:
                    st.caption("No action")

            st.markdown("<hr style='margin: 0.3rem 0; border: none; border-top: 1px solid #f1f5f9;'/>", unsafe_allow_html=True)

    except APIClientError as err:
        st.error(f"Failed to load invitations: {err}")
