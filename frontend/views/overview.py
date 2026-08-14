"""Administrator Overview Dashboard View for Enterprise Knowledge Navigator.

Provides company-level document statistics, 4 core KPI cards with icons,
recent company documents table, reliable knowledge review queue state,
and direct Quick Action workflows adhering to the approved Reference A design.
"""

import datetime
import html
from typing import Any, Dict, List, Optional
import streamlit as st

from frontend.api_client import APIClientError, BackendAPIClient
from frontend.navigation import navigate_to
from frontend.styles import format_bytes, format_date, get_status_badge_html


def render_overview(client: BackendAPIClient, workspace: Dict[str, Any]) -> None:
    """Render the clean, real-data Administrator Overview Dashboard matching Reference A."""
    ws_id = workspace["id"]
    ws_name = workspace.get("name", "Workspace")
    ws_slug = workspace.get("slug", "")

    ws_name_escaped = html.escape(ws_name)
    ws_slug_escaped = html.escape(ws_slug)

    # Resolve greeting based on time of day
    curr_hour = datetime.datetime.now().hour
    if curr_hour < 12:
        time_greeting = "Good morning"
    elif curr_hour < 18:
        time_greeting = "Good afternoon"
    else:
        time_greeting = "Good evening"

    auth_user = st.session_state.get("auth_user") or {}
    full_name = auth_user.get("full_name", "Admin")
    first_name = html.escape(full_name.split()[0] if full_name else "Admin")

    # =========================================================================
    # 1. WELCOME SECTION & WORKSPACE REFRESH
    # =========================================================================
    col_welcome, col_refresh = st.columns([4, 1.2])
    with col_welcome:
        st.markdown(
            f"""
            <div style="margin-bottom: 0.25rem;">
                <div style="font-size: 1.85rem; font-weight: 800; color: #0F172A; letter-spacing: -0.025em; line-height: 1.2;">
                    {time_greeting}, {first_name}
                </div>
                <div style="color: #64748B; font-size: 0.95rem; font-weight: 500; margin-top: 0.2rem;">
                    Your company knowledge at a glance.
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0.4rem;">
                    <span style="font-size: 0.82rem; color: #475569; font-weight: 600;">Workspace:</span>
                    <strong style="color: #1E293B; font-size: 0.88rem;">{ws_name_escaped}</strong>
                    <span class="ekn-badge ekn-badge-neutral" style="font-family: monospace;">{ws_slug_escaped}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_refresh:
        st.write("")
        if st.button("🔄 Refresh Overview", use_container_width=True, key="admin_refresh_overview_btn"):
            st.rerun()

    st.markdown("<div style='margin-bottom: 0.85rem;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 2. EXACTLY FOUR PRIMARY METRIC (KPI) CARDS (With Real API Error Handling)
    # =========================================================================
    stats_error: Optional[str] = None
    try:
        stats = client.get_workspace_stats(ws_id)
        comp_docs_display = str(stats.get("company_documents", stats.get("total_documents", 0)))
        search_ready_display = str(stats.get("indexed_documents", 0))
        pending_reviews_num = int(stats.get("pending_reviews", 0))
        pending_reviews_display = str(pending_reviews_num)
        active_emp_display = str(stats.get("active_employees", 0))
    except APIClientError as err:
        stats_error = str(err)
        comp_docs_display = "—"
        search_ready_display = "—"
        pending_reviews_num = -1
        pending_reviews_display = "—"
        active_emp_display = "—"

    if stats_error:
        st.warning(f"⚠️ Could not load workspace statistics: {html.escape(stats_error)}")

    kpi_cols = st.columns(4)

    with kpi_cols[0]:
        st.markdown(
            f"""
            <div class="ekn-stat-card">
                <div class="ekn-stat-icon-wrapper" style="background: #EFF6FF; color: #1D4ED8;">📁</div>
                <div class="ekn-stat-value" style="color: #1E3A8A;">{comp_docs_display}</div>
                <div class="ekn-stat-label">Company Documents</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_cols[1]:
        st.markdown(
            f"""
            <div class="ekn-stat-card">
                <div class="ekn-stat-icon-wrapper" style="background: #F0FDF4; color: #15803D;">⚡</div>
                <div class="ekn-stat-value" style="color: #15803D;">{search_ready_display}</div>
                <div class="ekn-stat-label">Search Ready</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_cols[2]:
        if pending_reviews_num > 0:
            rev_color = "#B91C1C"
            rev_bg = "#FEF2F2"
            rev_icon = "⚠️"
        elif pending_reviews_num == 0:
            rev_color = "#64748B"
            rev_bg = "#F8FAFC"
            rev_icon = "✓"
        else:
            rev_color = "#94A3B8"
            rev_bg = "#F1F5F9"
            rev_icon = "⚠️"

        st.markdown(
            f"""
            <div class="ekn-stat-card">
                <div class="ekn-stat-icon-wrapper" style="background: {rev_bg}; color: {rev_color};">{rev_icon}</div>
                <div class="ekn-stat-value" style="color: {rev_color};">{pending_reviews_display}</div>
                <div class="ekn-stat-label">Pending Reviews</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_cols[3]:
        st.markdown(
            f"""
            <div class="ekn-stat-card">
                <div class="ekn-stat-icon-wrapper" style="background: #EEF2FF; color: #4F46E5;">👥</div>
                <div class="ekn-stat-value" style="color: #4F46E5;">{active_emp_display}</div>
                <div class="ekn-stat-label">Active Employees</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 1.0rem;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 3. TWO-COLUMN MAIN AREA: RECENT DOCUMENTS & KNOWLEDGE REVIEW
    # =========================================================================
    col_docs, col_review = st.columns([1.5, 1.2])

    # Left Column: Recent Documents Table Card
    with col_docs:
        with st.container(border=True):
            hdr_c1, hdr_c2 = st.columns([3, 1.2])
            with hdr_c1:
                st.markdown("<h4 style='margin: 0; font-size: 1.05rem; font-weight: 700; color: #0F172A;'>Recent Documents</h4>", unsafe_allow_html=True)
            with hdr_c2:
                if st.button("View All →", key="overview_view_all_docs_link", use_container_width=True):
                    navigate_to("Documents")
                    st.rerun()

            st.markdown("<hr style='margin: 0.4rem 0 0.6rem 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)

            try:
                doc_data = client.list_documents(ws_id, scope="company", page=1, page_size=5)
                recent_docs: List[Dict[str, Any]] = doc_data.get("items", [])

                if not recent_docs:
                    st.markdown(
                        """
                        <div class="ekn-empty-state" style="padding: 1.25rem;">
                            <div class="ekn-empty-title" style="font-size: 0.95rem;">No Company Documents</div>
                            <div class="ekn-empty-desc" style="font-size: 0.82rem;">Upload PDF manuals, specifications, or policies to build your library.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    # Structured table header
                    st.markdown(
                        """
                        <div style="display: flex; justify-content: space-between; padding: 0.3rem 0.2rem; font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; border-bottom: 1px solid #E2E8F0; margin-bottom: 0.25rem;">
                            <div style="width: 55%;">Document</div>
                            <div style="width: 25%;">Updated</div>
                            <div style="width: 20%; text-align: right;">Status</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    for doc in recent_docs:
                        fname = html.escape(doc.get("original_filename", "Document"))
                        fsize = format_bytes(doc.get("file_size_bytes", 0))
                        dt = html.escape(format_date(doc.get("created_at")))
                        pstatus = doc.get("processing_status", "completed")

                        st.markdown(
                            f"""
                            <div style="background: #FFFFFF; border-bottom: 1px solid #F1F5F9; padding: 0.55rem 0.2rem; display: flex; justify-content: space-between; align-items: center;">
                                <div style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 55%;">
                                    <div style="font-weight: 600; color: #1E293B; font-size: 0.86rem; overflow: hidden; text-overflow: ellipsis;">📄 {fname}</div>
                                    <div style="color: #64748B; font-size: 0.72rem; margin-top: 0.1rem;">{fsize}</div>
                                </div>
                                <div style="color: #64748B; font-size: 0.78rem; width: 25%;">{dt}</div>
                                <div style="width: 20%; text-align: right;">
                                    {get_status_badge_html(pstatus)}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            except APIClientError as err:
                st.error(f"Failed to load recent documents: {html.escape(str(err))}")

    # Right Column: Knowledge Review Panel
    with col_review:
        with st.container(border=True):
            r_hdr1, r_hdr2 = st.columns([3, 1.4])
            with r_hdr1:
                st.markdown("<h4 style='margin: 0; font-size: 1.05rem; font-weight: 700; color: #0F172A;'>Knowledge Review</h4>", unsafe_allow_html=True)
            with r_hdr2:
                if st.button("Open Queue →", key="overview_open_review_link", use_container_width=True):
                    navigate_to("Knowledge Review")
                    st.rerun()

            st.markdown("<hr style='margin: 0.4rem 0 0.6rem 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)

            try:
                findings = client.list_findings(ws_id)
                pending_findings = [f for f in findings if f.get("review_status") in ("candidate", "under_review")]

                if not pending_findings:
                    st.markdown(
                        """
                        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.5rem 1.0rem; text-align: center; margin: 0.4rem 0;">
                            <div style="font-size: 1.8rem; margin-bottom: 0.35rem;">✨</div>
                            <div style="font-weight: 700; color: #0F172A; font-size: 0.95rem; margin-bottom: 0.25rem;">
                                Your company knowledge is up to date.
                            </div>
                            <div style="color: #64748B; font-size: 0.82rem; line-height: 1.4;">
                                All indexed documents are verified without unresolved contradictions or version conflicts.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    for f in pending_findings[:3]:
                        f_subj = html.escape(f.get("subject", "Unresolved Finding"))
                        f_exp = html.escape(f.get("explanation", ""))
                        st.markdown(
                            f"""
                            <div style="background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 6px; padding: 0.65rem 0.85rem; margin-bottom: 0.45rem;">
                                <div style="font-weight: 600; color: #92400E; font-size: 0.86rem;">⚠️ {f_subj}</div>
                                <div style="color: #78350F; font-size: 0.78rem; margin-top: 0.15rem; line-height: 1.35;">{f_exp[:110]}...</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            except Exception as err:
                # Do NOT claim knowledge is up to date when the query fails!
                st.markdown(
                    f"""
                    <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; padding: 1.25rem; text-align: center;">
                        <div style="font-size: 1.4rem; margin-bottom: 0.3rem;">⚠️</div>
                        <div style="font-weight: 700; color: #991B1B; font-size: 0.9rem;">Review Queue Unavailable</div>
                        <div style="color: #B91C1C; font-size: 0.8rem; margin-top: 0.2rem;">Could not query contradiction findings: {html.escape(str(err))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='margin-bottom: 1.0rem;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # 4. THREE PROMINENT QUICK ACTION CARDS (ABOVE THE FOLD)
    # =========================================================================
    st.markdown("<h4 style='margin: 0 0 0.55rem 0; font-size: 1.1rem; font-weight: 700; color: #0F172A;'>Quick Actions</h4>", unsafe_allow_html=True)
    action_cols = st.columns(3)

    with action_cols[0]:
        with st.container(border=True):
            st.markdown(
                """
                <div style="margin-bottom: 0.6rem;">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin-bottom: 0.2rem;">📤 Upload Document</div>
                    <div style="color: #64748B; font-size: 0.82rem; line-height: 1.4;">Add new PDF documents, specifications, or policies to the company library.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Upload PDF →", key="qa_upload_pdf_btn", use_container_width=True, type="primary"):
                navigate_to("Documents", doc_open_upload=True)
                st.rerun()

    with action_cols[1]:
        with st.container(border=True):
            st.markdown(
                """
                <div style="margin-bottom: 0.6rem;">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin-bottom: 0.2rem;">👥 Invite Employee</div>
                    <div style="color: #64748B; font-size: 0.82rem; line-height: 1.4;">Generate joining codes and onboard team members to access company knowledge.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Manage Team →", key="qa_invite_emp_btn", use_container_width=True):
                navigate_to("Team")
                st.rerun()

    with action_cols[2]:
        with st.container(border=True):
            st.markdown(
                """
                <div style="margin-bottom: 0.6rem;">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin-bottom: 0.2rem;">🕸️ Explore Knowledge Graph</div>
                    <div style="color: #64748B; font-size: 0.82rem; line-height: 1.4;">Visualize connections, entity dependencies, and provenance across company documents.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Explore Graph →", key="qa_explore_graph_btn", use_container_width=True):
                navigate_to("Knowledge Graph")
                st.rerun()
