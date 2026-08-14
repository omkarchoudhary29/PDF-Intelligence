"""Enterprise Authentication Gateway for Enterprise Knowledge Navigator (EKN).

Implements:
1. Entry Screen: Polished welcome view with two distinct cards for Administrator and Employee portals.
2. Administrator Portal: Dedicated sign-in and company registration with strict server-side role validation.
3. Employee Portal: Dedicated sign-in and employee onboarding (direct demo mode or secure invitation code).
4. Strict Role Separation: Cross-portal attempts are denied with authoritative server-side 403 feedback.
5. In-Memory Session Security: Streamlit session_state persistence without leaking tokens into URL query parameters.
"""

from typing import Any, Dict, List, Optional
import streamlit as st

from config.settings import get_settings
from frontend.api_client import APIClientError, BackendAPIClient


def _validate_password(password: str) -> Optional[str]:
    """Validate password according to enterprise security standards."""
    if len(password) < 12:
        return "Password must be at least 12 characters long."
    if not any(c.isupper() for c in password):
        return "Password must contain at least one uppercase letter (A-Z)."
    if not any(c.islower() for c in password):
        return "Password must contain at least one lowercase letter (a-z)."
    if not any(c.isdigit() for c in password):
        return "Password must contain at least one number (0-9)."
    special_chars = set("!@#$%^&*()_+-=[]{}|;:,.<>?/~`'\"\\")
    if not any(c in special_chars for c in password):
        return "Password must contain at least one special character (!@#$%^&* etc.)."
    return None


def render_login(client: BackendAPIClient) -> None:
    """Entry point for unauthenticated visitors."""
    portal = st.session_state.get("auth_portal", "role_selection")

    # Mandatory password change prompt on first login
    if st.session_state.get("pending_password_change"):
        _render_password_change_form(client)
        return

    if portal == "admin":
        _render_admin_portal(client)
    elif portal == "employee":
        _render_employee_portal(client)
    else:
        _render_role_selection(client)


def _render_role_selection(client: Optional[BackendAPIClient] = None) -> None:
    """Render the primary identity selection screen with two distinct portal cards."""
    settings = get_settings()

    st.markdown(
        """
        <div style="max-width: 720px; margin: 1.8rem auto 1.6rem auto; text-align: center;">
            <div style="display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; background: #4F46E5; border-radius: 12px; color: #FFFFFF; font-size: 1.6rem; font-weight: 800; box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25); margin-bottom: 0.85rem;">
                📄
            </div>
            <div>
                <div style="display: inline-block; padding: 0.3rem 0.85rem; background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 9999px; margin-bottom: 0.85rem;">
                    <span style="color: #4F46E5; font-weight: 700; font-size: 0.80rem; letter-spacing: 0.05em; text-transform: uppercase;">
                        Knowledge Intelligence Platform
                    </span>
                </div>
            </div>
            <div style="font-size: 2.5rem; font-weight: 800; color: #172033; letter-spacing: -0.03em; margin-bottom: 0.35rem; line-height: 1.15;">
                Enterprise Knowledge Navigator
            </div>
            <div style="font-size: 1.15rem; color: #475569; font-weight: 400; margin-bottom: 0.45rem;">
                Your organization's knowledge, in one place.
            </div>
            <div style="font-size: 1.0rem; font-weight: 600; color: #172033;">
                How would you like to continue?
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, col1, col2, _ = st.columns([0.6, 3.2, 3.2, 0.6])

    with col1:
        with st.container(border=True):
            st.markdown(
                """
                <div style="text-align: center; padding: 1.5rem 1.0rem 0.75rem 1.0rem;">
                    <div style="font-size: 2.8rem; margin-bottom: 0.75rem;">🛡️</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #172033; margin-bottom: 0.5rem;">
                        ADMINISTRATOR PORTAL
                    </div>
                    <div style="font-size: 0.92rem; color: #475569; line-height: 1.5; min-height: 48px; margin-bottom: 1.25rem;">
                        Manage company documents, employee access, and organizational knowledge intelligence.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Continue as Administrator", type="primary", use_container_width=True, key="role_btn_admin"):
                st.session_state["auth_portal"] = "admin"
                st.rerun()

            if settings.ekn_demo_mode:
                if st.button("⚡ Quick Demo Sign In (1-Click)", type="secondary", use_container_width=True, key="role_quick_demo_admin"):
                    if client:
                        try:
                            token_data = client.login(
                                "admin.demo@example.com",
                                settings.demo_admin_password,
                                remember_me=True,
                                portal="admin",
                            )
                            profile = client.get_current_user_profile()
                            user_workspaces = profile.get("workspaces", [])
                            admin_workspaces = [
                                w for w in user_workspaces
                                if w.get("role") in ("owner", "admin") and w.get("membership_status") == "active"
                            ]
                            st.session_state["auth_token"] = token_data.get("access_token")
                            st.session_state["refresh_token"] = token_data.get("refresh_token")
                            st.session_state["auth_user"] = profile
                            st.session_state["user_role"] = admin_workspaces[0]["role"] if admin_workspaces else "admin"
                            st.session_state["selected_workspace_id"] = (
                                admin_workspaces[0]["workspace_id"] if admin_workspaces else settings.demo_workspace_id
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"Demo sign-in failed: {e}")

    with col2:
        with st.container(border=True):
            st.markdown(
                """
                <div style="text-align: center; padding: 1.5rem 1.0rem 0.75rem 1.0rem;">
                    <div style="font-size: 2.8rem; margin-bottom: 0.75rem;">👤</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #172033; margin-bottom: 0.5rem;">
                        EMPLOYEE PORTAL
                    </div>
                    <div style="font-size: 0.92rem; color: #475569; line-height: 1.5; min-height: 48px; margin-bottom: 1.25rem;">
                        Ask questions, explore verified enterprise documents, and query personal files with AI.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Continue as Employee", type="secondary", use_container_width=True, key="role_btn_emp"):
                st.session_state["auth_portal"] = "employee"
                st.rerun()

            if settings.ekn_demo_mode:
                if st.button("⚡ Quick Demo Sign In (1-Click)", type="primary", use_container_width=True, key="role_quick_demo_emp"):
                    if client:
                        try:
                            token_data = client.login(
                                "employee.demo@example.com",
                                settings.demo_employee_password,
                                remember_me=True,
                                portal="employee",
                            )
                            profile = client.get_current_user_profile()
                            user_workspaces = profile.get("workspaces", [])
                            st.session_state["auth_token"] = token_data.get("access_token")
                            st.session_state["refresh_token"] = token_data.get("refresh_token")
                            st.session_state["auth_user"] = profile
                            st.session_state["user_role"] = "employee"
                            emp_ws = next((w for w in user_workspaces if w.get("role") == "employee"), user_workspaces[0] if user_workspaces else {})
                            st.session_state["selected_workspace_id"] = emp_ws.get("workspace_id") or settings.demo_workspace_id
                            st.rerun()
                        except Exception as e:
                            st.error(f"Demo sign-in failed: {e}")

    # Enterprise Trust Bar
    st.markdown(
        """
        <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 1.8rem; margin-top: 1.8rem; color: #64748B; font-size: 0.82rem; font-weight: 500; text-align: center;">
            <span>🔒 SOC2 Architecture</span>
            <span>⚡ Hybrid Vector & Keyword Search</span>
            <span>🛡️ Contradiction Detection</span>
            <span>📜 Verifiable Page Provenance</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Optional local demo accounts expander (only shown when demo mode is active)
    if settings.ekn_demo_mode:
        _, exp_col, _ = st.columns([1, 6.6, 1])
        with exp_col:
            st.markdown("<div style='margin-top: 2.0rem;'></div>", unsafe_allow_html=True)
            with st.expander("🔑 Demo Account Details (Local Interview Demonstration)", expanded=False):
                st.markdown(
                    f"""
                    <div style="font-size: 0.92rem; color: #334155; line-height: 1.6;">
                        <p style="margin-bottom: 0.8rem;">
                            <strong>Local Demo Mode is active.</strong> You can sign in using either of the pre-seeded demo accounts:
                        </p>
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.8rem; margin-bottom: 0.8rem;">
                            <strong style="color: #4F46E5;">🛡️ Administrator Account:</strong><br/>
                            • <strong>Email:</strong> <code>admin.demo@example.com</code><br/>
                            • <strong>Password:</strong> <code>{settings.demo_admin_password or 'Configured in .env'}</code><br/>
                            • <strong>Access:</strong> Administrator Portal (Company Documents, Ingestion, Knowledge Graph, Team)
                        </div>
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.8rem;">
                            <strong style="color: #0284C7;">👤 Employee Account:</strong><br/>
                            • <strong>Email:</strong> <code>employee.demo@example.com</code><br/>
                            • <strong>Password:</strong> <code>{settings.demo_employee_password or 'Configured in .env'}</code><br/>
                            • <strong>Access:</strong> Employee Portal (Conversational Assistant, Personal PDF Scoping)
                        </div>
                        <p style="font-size: 0.82rem; color: #64748B; margin-top: 0.6rem; margin-bottom: 0;">
                            <em>Cross-portal isolation is enforced: each account can only enter through its authorized portal.</em>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _render_admin_portal(client: BackendAPIClient) -> None:
    """Render the dedicated Administrator authentication screen."""
    # Top back navigation
    top_col1, _ = st.columns([1, 4])
    with top_col1:
        if st.button("← Back to identity selection", key="admin_back_btn"):
            st.session_state["auth_portal"] = "role_selection"
            st.rerun()

    st.markdown(
        """
        <div style="max-width: 580px; margin: 1.0rem auto 1.5rem auto; text-align: center;">
            <div style="display: inline-block; padding: 0.35rem 0.85rem; background: #EEF2FF; border: 1px solid #C7D2FE; border-radius: 9999px; margin-bottom: 0.75rem;">
                <span style="color: #4F46E5; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.05em; text-transform: uppercase;">
                    Administrator Portal
                </span>
            </div>
            <div style="font-size: 2.1rem; font-weight: 800; color: #172033; letter-spacing: -0.02em; margin-bottom: 0.35rem;">
                Welcome to your company workspace
            </div>
            <div style="font-size: 1.0rem; color: #475569;">
                Access your organization's knowledge management platform.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 2.6, 1])

    with center_col:
        tab_options = ["Sign In", "Create Company"]
        admin_tab_param = st.query_params.get("admin_tab", "").lower()
        if admin_tab_param in ("create_company", "signup", "register"):
            st.session_state["admin_mode_selector"] = "Create Company"
        elif admin_tab_param in ("signin", "login"):
            st.session_state["admin_mode_selector"] = "Sign In"

        default_idx = 1 if admin_tab_param in ("create_company", "signup", "register") else 0
        chosen_admin_tab = st.radio(
            "Admin Mode",
            tab_options,
            index=default_idx,
            horizontal=True,
            label_visibility="collapsed",
            key="admin_mode_selector",
        )
        st.markdown("<div style='margin-bottom: 0.6rem;'></div>", unsafe_allow_html=True)

        # --- ADMIN TAB 1: SIGN IN ---
        if chosen_admin_tab == "Sign In":
            settings = get_settings()
            if settings.ekn_demo_mode:
                if not st.session_state.get("admin_signin_email"):
                    st.session_state["admin_signin_email"] = "admin.demo@example.com"
                if not st.session_state.get("admin_signin_password"):
                    st.session_state["admin_signin_password"] = settings.demo_admin_password

            with st.container(border=True):
                if settings.ekn_demo_mode:
                    st.markdown(
                        """
                        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 0.65rem 0.85rem; margin-bottom: 0.85rem; display: flex; align-items: center; gap: 0.5rem;">
                            <span style="font-size: 1.1rem;">⚡</span>
                            <span style="font-size: 0.84rem; color: #166534; font-weight: 500;">
                                <strong>Demo Credentials Pre-filled:</strong> Click <strong>Sign In as Administrator</strong> below to enter immediately.
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with st.form("admin_signin_form"):
                    email = st.text_input(
                        "Work Email",
                        placeholder="admin@yourcompany.com",
                        key="admin_signin_email",
                    )
                    show_pwd = st.checkbox("Show password", value=False, key="admin_show_pwd")
                    password = st.text_input(
                        "Password",
                        type="default" if show_pwd else "password",
                        placeholder="Enter your password",
                        key="admin_signin_password",
                    )
                    remember_me = st.checkbox(
                        "Remember session",
                        value=True,
                        key="admin_remember_me",
                    )
                    submit = st.form_submit_button("Sign In as Administrator", type="primary", use_container_width=True)

                    if submit:
                        clean_email = email.strip()
                        clean_pw = password.strip()
                        if not clean_email or not clean_pw:
                            st.error("Please enter both work email and password.")
                        else:
                            try:
                                with st.spinner("Authenticating administrator credentials..."):
                                    # Server-side validation: portal="admin" rejects non-admin users with 403
                                    token_data = client.login(
                                        username_or_email=clean_email,
                                        password=clean_pw,
                                        remember_me=remember_me,
                                        portal="admin",
                                    )

                                # Fetch full profile to retrieve verified workspace context
                                profile = client.get_current_user_profile()
                                user_workspaces = profile.get("workspaces", [])
                                admin_workspaces = [
                                    w for w in user_workspaces
                                    if w.get("role") in ("owner", "admin") and w.get("membership_status") == "active"
                                ]

                                if not admin_workspaces:
                                    client.logout()
                                    st.session_state.clear()
                                    st.session_state["auth_portal"] = "admin"
                                    st.error("This account does not have administrator access. Please use the Employee Portal.")
                                    return

                                # Store session in session_state (no tokens in URL parameters)
                                st.session_state["auth_token"] = token_data.get("access_token")
                                st.session_state["refresh_token"] = token_data.get("refresh_token")
                                st.session_state["auth_user"] = profile
                                st.session_state["user_role"] = admin_workspaces[0]["role"]
                                st.session_state["selected_workspace_id"] = admin_workspaces[0]["workspace_id"]

                                if token_data.get("user", {}).get("must_change_password"):
                                    st.session_state["pending_password_change"] = True
                                    st.session_state["temp_password_holder"] = clean_pw
                                    st.rerun()

                                st.success(f"Welcome back, {profile.get('full_name')}!")
                                st.rerun()

                            except APIClientError as err:
                                if err.status_code == 403:
                                    # Clear any stale session state
                                    client.access_token = None
                                    client.refresh_token = None
                                    st.session_state.pop("auth_token", None)
                                    st.session_state.pop("refresh_token", None)
                                    st.error("This account does not have administrator access. Please use the Employee Portal.")
                                elif err.status_code == 423:
                                    st.error("Account Locked: Too many failed attempts. Try again in 15 minutes.")
                                else:
                                    st.error(f"Sign In Failed: {err}")

        # --- ADMIN TAB 2: CREATE COMPANY ---
        else:
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="font-weight: 700; color: #172033; font-size: 1.1rem; margin-bottom: 0.25rem;">
                        Create Company Account
                    </div>
                    <div style="font-size: 0.88rem; color: #64748B; margin-bottom: 1.0rem;">
                        Register your organizational workspace and initial primary owner account.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.form("admin_signup_form"):
                    su_name = st.text_input(
                        "Full Name",
                        placeholder="e.g. Sarah Jenkins",
                        key="admin_signup_name",
                    )
                    su_company = st.text_input(
                        "Company Name",
                        placeholder="e.g. Acme Aerospace",
                        key="admin_signup_company",
                    )
                    su_email = st.text_input(
                        "Work Email",
                        placeholder="e.g. sarah@acme.com",
                        key="admin_signup_email",
                    )
                    su_show_pwd = st.checkbox("Show passwords", value=False, key="admin_su_show_pwd")
                    su_pw = st.text_input(
                        "Password",
                        type="default" if su_show_pwd else "password",
                        placeholder="At least 12 characters (upper, lower, number, symbol)",
                        key="admin_signup_pw",
                        help="Must be at least 12 characters with uppercase, lowercase, numbers, and a special character.",
                    )
                    su_confirm = st.text_input(
                        "Confirm Password",
                        type="default" if su_show_pwd else "password",
                        placeholder="Re-enter password",
                        key="admin_signup_confirm",
                    )
                    su_submit = st.form_submit_button("Create Company Account", type="primary", use_container_width=True)

                    if su_submit:
                        clean_name = su_name.strip()
                        clean_comp = su_company.strip()
                        clean_email = su_email.strip()
                        clean_pw = su_pw.strip()
                        clean_conf = su_confirm.strip()

                        pw_err = _validate_password(clean_pw)
                        if not clean_name or not clean_comp or not clean_email or not clean_pw:
                            st.error("Please fill in all required fields.")
                        elif clean_pw != clean_conf:
                            st.error("Passwords do not match.")
                        elif pw_err:
                            st.error(pw_err)
                        else:
                            try:
                                with st.spinner("Setting up company workspace and administrator account..."):
                                    token_data = client.admin_register(
                                        full_name=clean_name,
                                        company_name=clean_comp,
                                        email=clean_email,
                                        password=clean_pw,
                                    )

                                st.session_state["auth_token"] = token_data.get("access_token")
                                st.session_state["refresh_token"] = token_data.get("refresh_token")
                                profile = client.get_current_user_profile()
                                st.session_state["auth_user"] = profile
                                st.session_state["user_role"] = "owner"
                                workspaces = profile.get("workspaces", [])
                                if workspaces:
                                    st.session_state["selected_workspace_id"] = workspaces[0]["workspace_id"]

                                st.success(f"Company '{clean_comp}' registered successfully! Welcome, {clean_name}.")
                                st.rerun()

                            except APIClientError as err:
                                st.error(f"Registration Failed: {err}")


def _render_employee_portal(client: BackendAPIClient) -> None:
    """Render the dedicated Employee authentication screen."""
    settings = get_settings()

    # Top back navigation
    top_col1, _ = st.columns([1, 4])
    with top_col1:
        if st.button("← Back to identity selection", key="emp_back_btn"):
            st.session_state["auth_portal"] = "role_selection"
            st.rerun()

    st.markdown(
        """
        <div style="max-width: 580px; margin: 1.0rem auto 1.5rem auto; text-align: center;">
            <div style="display: inline-block; padding: 0.35rem 0.85rem; background: #E0F2FE; border: 1px solid #BAE6FD; border-radius: 9999px; margin-bottom: 0.75rem;">
                <span style="color: #0284C7; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.05em; text-transform: uppercase;">
                    Employee Portal
                </span>
            </div>
            <div style="font-size: 2.1rem; font-weight: 800; color: #172033; letter-spacing: -0.02em; margin-bottom: 0.35rem;">
                Welcome back
            </div>
            <div style="font-size: 1.0rem; color: #475569;">
                Sign in to your team workspace. Access your company's knowledge and AI-powered document assistant.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 2.6, 1])

    with center_col:
        tab_title = "Create Employee Account" if settings.ekn_demo_mode else "Join Company"
        emp_options = ["Sign In", tab_title]
        emp_tab_param = st.query_params.get("emp_tab", "").lower()
        if emp_tab_param in ("create_account", "signup", "register"):
            st.session_state["emp_mode_selector"] = tab_title
        elif emp_tab_param in ("signin", "login"):
            st.session_state["emp_mode_selector"] = "Sign In"

        emp_default_idx = 1 if emp_tab_param in ("create_account", "signup", "register") else 0
        chosen_emp_tab = st.radio(
            "Employee Mode",
            emp_options,
            index=emp_default_idx,
            horizontal=True,
            label_visibility="collapsed",
            key="emp_mode_selector",
        )
        st.markdown("<div style='margin-bottom: 0.6rem;'></div>", unsafe_allow_html=True)

        # --- EMPLOYEE TAB 1: SIGN IN ---
        if chosen_emp_tab == "Sign In":
            if settings.ekn_demo_mode:
                if not st.session_state.get("emp_signin_email"):
                    st.session_state["emp_signin_email"] = "employee.demo@example.com"
                if not st.session_state.get("emp_signin_password"):
                    st.session_state["emp_signin_password"] = settings.demo_employee_password

            with st.container(border=True):
                if settings.ekn_demo_mode:
                    st.markdown(
                        """
                        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 0.65rem 0.85rem; margin-bottom: 0.85rem; display: flex; align-items: center; gap: 0.5rem;">
                            <span style="font-size: 1.1rem;">⚡</span>
                            <span style="font-size: 0.84rem; color: #166534; font-weight: 500;">
                                <strong>Demo Credentials Pre-filled:</strong> Click <strong>Sign In as Employee</strong> below to enter immediately.
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with st.form("emp_signin_form"):
                    email = st.text_input(
                        "Work Email",
                        placeholder="user@yourcompany.com",
                        key="emp_signin_email",
                    )
                    show_pwd = st.checkbox("Show password", value=False, key="emp_show_pwd")
                    password = st.text_input(
                        "Password",
                        type="default" if show_pwd else "password",
                        placeholder="Enter your password",
                        key="emp_signin_password",
                    )
                    remember_me = st.checkbox(
                        "Remember session",
                        value=True,
                        key="emp_remember_me",
                    )
                    submit = st.form_submit_button("Sign In as Employee", type="primary", use_container_width=True)

                    if submit:
                        clean_email = email.strip()
                        clean_pw = password.strip()
                        if not clean_email or not clean_pw:
                            st.error("Please enter both email and password.")
                        else:
                            try:
                                with st.spinner("Authenticating employee credentials..."):
                                    # Server-side validation: portal="employee" rejects admin-only users with 403
                                    token_data = client.login(
                                        username_or_email=clean_email,
                                        password=clean_pw,
                                        remember_me=remember_me,
                                        portal="employee",
                                    )

                                profile = client.get_current_user_profile()
                                user_workspaces = profile.get("workspaces", [])

                                if not user_workspaces:
                                    client.logout()
                                    st.session_state.clear()
                                    st.session_state["auth_portal"] = "employee"
                                    st.error("This account does not have an active company membership. Please contact your administrator.")
                                    return

                                # Store session in session_state (no tokens in URL parameters)
                                st.session_state["auth_token"] = token_data.get("access_token")
                                st.session_state["refresh_token"] = token_data.get("refresh_token")
                                st.session_state["auth_user"] = profile
                                st.session_state["user_role"] = "employee"

                                emp_ws = next((w for w in user_workspaces if w.get("role") == "employee"), user_workspaces[0])
                                st.session_state["selected_workspace_id"] = emp_ws["workspace_id"]

                                if token_data.get("user", {}).get("must_change_password"):
                                    st.session_state["pending_password_change"] = True
                                    st.session_state["temp_password_holder"] = clean_pw
                                    st.rerun()

                                st.success(f"Welcome back, {profile.get('full_name')}!")
                                st.rerun()

                            except APIClientError as err:
                                if err.status_code == 403:
                                    # Clear any stale session state
                                    client.access_token = None
                                    client.refresh_token = None
                                    st.session_state.pop("auth_token", None)
                                    st.session_state.pop("refresh_token", None)
                                    st.error("This account has administrator access. Please use the Administrator Portal.")
                                elif err.status_code == 423:
                                    st.error("Account Locked: Too many failed attempts. Try again in 15 minutes.")
                                else:
                                    st.error(f"Sign In Failed: {err}")

        # --- EMPLOYEE TAB 2: REGISTRATION ---
        else:
            with st.container(border=True):
                if settings.ekn_demo_mode:
                    st.markdown(
                        """
                        <div style="font-weight: 700; color: #172033; font-size: 1.1rem; margin-bottom: 0.25rem;">
                            Create Employee Account
                        </div>
                        <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px; padding: 0.6rem 0.8rem; margin-bottom: 1.0rem;">
                            <span style="color: #166534; font-size: 0.85rem; font-weight: 600;">
                                ✨ Demo Mode Active: No invitation code required. Account will be added directly to the Aerospace & Engineering demo company.
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div style="font-weight: 700; color: #172033; font-size: 1.1rem; margin-bottom: 0.25rem;">
                            Join Your Team
                        </div>
                        <div style="font-size: 0.88rem; color: #64748B; margin-bottom: 1.0rem;">
                            Enter your details and the company joining code provided by your administrator.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with st.form("emp_signup_form"):
                    su_name = st.text_input(
                        "Full Name",
                        placeholder="e.g. David Miller",
                        key="emp_signup_name",
                    )
                    su_email = st.text_input(
                        "Personal or Work Email" if settings.ekn_demo_mode else "Work Email",
                        placeholder="e.g. david@example.com",
                        key="emp_signup_email",
                    )
                    clean_code: Optional[str] = None
                    if not settings.ekn_demo_mode:
                        su_code = st.text_input(
                            "Company Joining Code",
                            placeholder="e.g. JOIN-XXXXXX-XXXXXX",
                            key="emp_signup_code",
                            help="Obtain this code from your workspace administrator",
                        )
                        clean_code = su_code.strip().upper() if su_code else None

                    su_show_pwd = st.checkbox("Show passwords", value=False, key="emp_su_show_pwd")
                    su_pw = st.text_input(
                        "Password",
                        type="default" if su_show_pwd else "password",
                        placeholder="At least 12 characters (upper, lower, number, symbol)",
                        key="emp_signup_pw",
                        help="Must be at least 12 characters with uppercase, lowercase, numbers, and a special character.",
                    )
                    su_confirm = st.text_input(
                        "Confirm Password",
                        type="default" if su_show_pwd else "password",
                        placeholder="Re-enter password",
                        key="emp_signup_confirm",
                    )
                    btn_label = "Create Employee Account" if settings.ekn_demo_mode else "Join Company & Create Account"
                    su_submit = st.form_submit_button(btn_label, type="primary", use_container_width=True)

                    if su_submit:
                        clean_name = su_name.strip()
                        clean_email = su_email.strip()
                        clean_pw = su_pw.strip()
                        clean_conf = su_confirm.strip()

                        pw_err = _validate_password(clean_pw)
                        if not clean_name or not clean_email or not clean_pw:
                            st.error("Please fill in all required fields.")
                        elif not settings.ekn_demo_mode and not clean_code:
                            st.error("Please provide the Company Joining Code.")
                        elif clean_pw != clean_conf:
                            st.error("Passwords do not match.")
                        elif pw_err:
                            st.error(pw_err)
                        else:
                            try:
                                with st.spinner("Setting up your employee account..."):
                                    token_data = client.employee_register(
                                        full_name=clean_name,
                                        email=clean_email,
                                        password=clean_pw,
                                        code=clean_code,
                                    )

                                st.session_state["auth_token"] = token_data.get("access_token")
                                st.session_state["refresh_token"] = token_data.get("refresh_token")
                                profile = client.get_current_user_profile()
                                st.session_state["auth_user"] = profile
                                st.session_state["user_role"] = "employee"
                                workspaces = profile.get("workspaces", [])
                                if workspaces:
                                    st.session_state["selected_workspace_id"] = workspaces[0]["workspace_id"]

                                st.success(f"Welcome to the team, {clean_name}!")
                                st.rerun()

                            except APIClientError as err:
                                st.error(f"Registration Failed: {err}")


def _render_password_change_form(client: BackendAPIClient) -> None:
    """Render mandatory password change form upon first login."""
    with st.container(border=True):
        st.markdown("#### Update Password Required")
        st.warning("Your administrator requires you to set a new password before accessing the system.")

        with st.form("mandatory_password_change_form"):
            current_pass = st.text_input(
                "Current Password",
                type="password",
                value=st.session_state.get("temp_password_holder", ""),
                key="pwd_change_current",
            )
            new_pass = st.text_input(
                "New Password",
                type="password",
                placeholder="At least 12 chars (upper, lower, digit, symbol)",
                key="pwd_change_new",
            )
            confirm_pass = st.text_input(
                "Confirm New Password",
                type="password",
                placeholder="Re-enter your new password",
                key="pwd_change_confirm",
            )

            st.caption("Password requirements: 12+ characters, uppercase letter, lowercase letter, number, and special character.")

            update_btn = st.form_submit_button("Update Password and Continue", type="primary", use_container_width=True)

            if update_btn:
                if not new_pass or not confirm_pass:
                    st.error("Please fill in all password fields.")
                elif new_pass != confirm_pass:
                    st.error("New password and confirmation do not match.")
                else:
                    try:
                        client.change_password(
                            current_password=current_pass,
                            new_password=new_pass,
                        )
                        st.session_state["pending_password_change"] = False
                        st.session_state.pop("temp_password_holder", None)
                        try:
                            refreshed = client.get_current_user_profile()
                            st.session_state["auth_user"] = refreshed
                        except Exception:
                            pass
                        st.success("Password updated successfully!")
                        st.rerun()
                    except APIClientError as err:
                        st.error(f"Failed to update password: {err}")
