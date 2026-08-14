"""Knowledge Review View for Enterprise Knowledge Navigator.

Provides authoritative inspection, side-by-side evidence verification, and administrative
resolution workflows for AI-detected cross-document contradictions and document version changes.
"""

from typing import Any, Dict, List, Optional
import streamlit as st

from frontend.api_client import APIClientError, BackendAPIClient
from frontend.styles import format_date


def render_knowledge_review(
    client: BackendAPIClient,
    workspace: Dict[str, Any],
    current_user_role: str = "owner",
) -> None:
    """Render the Knowledge Review and Contradiction Resolution Center."""
    ws_id = workspace["id"]
    ws_name = workspace["name"]
    can_manage = current_user_role in ("owner", "admin")

    col_title, col_scan, col_refresh = st.columns([3, 1.2, 0.8])
    with col_title:
        st.markdown(f"## Knowledge Review — {ws_name}")
        st.caption("Inspect AI-detected contradictions, revision discrepancies, and conflicting claims across company PDFs.")
    with col_scan:
        scan_clicked = st.button("🔍 Scan for Issues", type="primary", use_container_width=True, help="Scan company document pairs for factual conflicts and discrepancies")
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    st.markdown("<hr style='margin: 0.6rem 0 1.25rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

    # Review Queue Tabs
    tabs = st.tabs([
        "⚠️ All Pending",
        "⚡ Contradictions",
        "📝 Document Changes",
        "🔍 Under Review",
        "✅ Resolved",
    ])

    tab_map = {
        0: ("all_pending", "All Pending Issues"),
        1: ("contradictions", "Direct Contradictions"),
        2: ("changes", "Document Revision Changes"),
        3: ("under_review", "Under Active Investigation"),
        4: ("resolved", "Resolved & Closed Findings"),
    }

    # Fetch all findings once for the workspace
    try:
        with st.spinner("Loading knowledge review findings...") if scan_clicked else st.empty():
            findings = client.list_findings(
                workspace_id=ws_id,
                scan=scan_clicked,
            )
    except APIClientError as err:
        st.error(f"Error fetching knowledge review findings: {err}")
        return

    # Helper function to render a list of findings
    def render_finding_cards(items: List[Dict[str, Any]], empty_msg: str) -> None:
        if not items:
            st.markdown(
                f"""
                <div class="ekn-empty-state">
                    <div class="ekn-empty-title">Your company knowledge is up to date</div>
                    <div class="ekn-empty-desc">{empty_msg}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        st.caption(f"Showing **{len(items)}** finding(s).")

        status_badges = {
            "candidate": "<span class='ekn-badge' style='background:#fef3c7;color:#92400e;border:1px solid #fde68a;'>Pending Review</span>",
            "under_review": "<span class='ekn-badge' style='background:#ede9fe;color:#6d28d9;border:1px solid #ddd6fe;'>Under Review</span>",
            "confirmed": "<span class='ekn-badge' style='background:#fee2e2;color:#991b1b;border:1px solid #fecaca;'>Confirmed Issue</span>",
            "not_a_contradiction": "<span class='ekn-badge' style='background:#dbeafe;color:#1e40af;border:1px solid #bfdbfe;'>Not a Contradiction</span>",
            "dismissed": "<span class='ekn-badge' style='background:#f1f5f9;color:#64748b;border:1px solid #e2e8f0;'>Dismissed</span>",
            "resolved": "<span class='ekn-badge' style='background:#dcfce7;color:#166534;border:1px solid #bbf7d0;'>Resolved</span>",
        }

        category_labels = {
            "numerical_discrepancy": "Numerical Difference",
            "instruction_conflict": "Procedural Conflict",
            "dated_revision": "Dated Revision Discrepancy",
            "context_dependent_difference": "Context-Dependent Scope",
            "direct_factual_conflict": "Direct Factual Conflict",
            "date_validity_conflict": "Date & Validity Mismatch",
        }

        for f in items:
            f_id = f["id"]
            status = f.get("review_status", "candidate")
            badge_html = status_badges.get(status, f"<span class='ekn-badge'>{status}</span>")
            cat_name = category_labels.get(f.get("finding_type", ""), f.get("finding_type", "").replace("_", " ").title())

            with st.container():
                st.markdown(
                    f"""
                    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:1.25rem; margin-bottom:1rem; box-shadow:0 1px 3px rgba(0,0,0,0.03);">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.5rem;">
                            <div>
                                <span style="font-size:1.1rem; font-weight:700; color:#0f172a;">{f.get('subject', 'Untitled Finding')}</span>
                                <span style="color:#64748b; font-size:0.85rem; margin-left:0.5rem;">({cat_name})</span>
                            </div>
                            <div>{badge_html}</div>
                        </div>
                        <div style="color:#334155; font-size:0.92rem; margin-bottom:1rem; line-height:1.5;">
                            <strong>Analysis:</strong> {f.get('explanation', 'No explanation provided.')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Side-by-side Evidence Comparison
                e_col1, e_col2 = st.columns(2)
                doc1_id = f.get("first_document_id", "")
                doc2_id = f.get("second_document_id", "")

                with e_col1:
                    st.markdown(f"**Source Document A (Page {f.get('first_page', 1)}):**")
                    st.info(f"\"{f.get('first_evidence_excerpt', 'No excerpt available.')}\"")
                with e_col2:
                    st.markdown(f"**Source Document B (Page {f.get('second_page', 1)}):**")
                    st.info(f"\"{f.get('second_evidence_excerpt', 'No excerpt available.')}\"")

                if f.get("review_reason"):
                    st.markdown(
                        f"<div style='background:#f8fafc; border-left:3px solid #4f46e5; padding:0.5rem 0.75rem; font-size:0.85rem; color:#475569; margin-bottom:0.75rem;'><strong>Resolution Notes:</strong> {f['review_reason']}</div>",
                        unsafe_allow_html=True,
                    )

                # Administrative Decision Controls (Owner/Admin only)
                if can_manage and status in ("candidate", "under_review"):
                    with st.expander("⚖️ Administrative Decision & Resolution", expanded=False):
                        r_cols = st.columns(4)
                        with r_cols[0]:
                            if st.button("Confirm Issue", key=f"kr_conf_{f_id}", type="primary", use_container_width=True):
                                try:
                                    client.review_finding(ws_id, f_id, review_status="confirmed")
                                    st.success("Finding confirmed as an active issue.")
                                    st.rerun()
                                except APIClientError as err:
                                    st.error(str(err))
                        with r_cols[1]:
                            if st.button("Under Review", key=f"kr_rev_{f_id}", use_container_width=True):
                                try:
                                    client.review_finding(ws_id, f_id, review_status="under_review")
                                    st.info("Marked as Under Review.")
                                    st.rerun()
                                except APIClientError as err:
                                    st.error(str(err))
                        with r_cols[2]:
                            if st.button("Not Contradiction", key=f"kr_not_{f_id}", use_container_width=True):
                                try:
                                    client.review_finding(ws_id, f_id, review_status="not_a_contradiction")
                                    st.success("Dismissed as non-conflicting.")
                                    st.rerun()
                                except APIClientError as err:
                                    st.error(str(err))
                        with r_cols[3]:
                            if st.button("Dismiss", key=f"kr_dism_{f_id}", use_container_width=True):
                                try:
                                    client.review_finding(ws_id, f_id, review_status="dismissed")
                                    st.caption("Finding dismissed.")
                                    st.rerun()
                                except APIClientError as err:
                                    st.error(str(err))

                        # Authoritative Resolution Form
                        with st.form(f"resolve_form_{f_id}"):
                            st.markdown("**Authoritative Resolution**")
                            notes = st.text_input("Resolution Notes", placeholder="e.g. Document A is the newer authoritative engineering standard.", key=f"res_notes_{f_id}")
                            doc_options = [doc1_id, doc2_id] if doc1_id and doc2_id else ([doc1_id] if doc1_id else [])
                            auth_doc_id = st.selectbox(
                                "Designate Authoritative Document",
                                options=doc_options,
                                format_func=lambda did: f"Document `{did[:8]}...`",
                                key=f"res_doc_{f_id}",
                            ) if doc_options else None

                            submit_res = st.form_submit_button("Mark Resolved with Authoritative Document", type="primary")
                            if submit_res:
                                try:
                                    client.review_finding(
                                        ws_id,
                                        f_id,
                                        review_status="resolved",
                                        resolution_notes=notes.strip() if notes.strip() else None,
                                        authoritative_document_id=auth_doc_id,
                                    )
                                    st.success("Finding marked as resolved.")
                                    st.rerun()
                                except APIClientError as err:
                                    st.error(str(err))

                st.markdown("<hr style='margin:0.8rem 0; border:none; border-top:1px solid #f1f5f9;'/>", unsafe_allow_html=True)

    # Render each tab
    with tabs[0]:
        pending_items = [f for f in findings if f.get("review_status") in ("candidate", "under_review")]
        render_finding_cards(pending_items, "No pending knowledge contradictions or change discrepancies found.")

    with tabs[1]:
        contra_items = [f for f in findings if f.get("finding_type") in ("direct_factual_conflict", "numerical_discrepancy", "instruction_conflict")]
        render_finding_cards(contra_items, "No direct factual contradictions recorded.")

    with tabs[2]:
        change_items = [f for f in findings if f.get("finding_type") in ("dated_revision", "context_dependent_difference", "date_validity_conflict")]
        render_finding_cards(change_items, "No document revision differences requiring attention.")

    with tabs[3]:
        under_review_items = [f for f in findings if f.get("review_status") == "under_review"]
        render_finding_cards(under_review_items, "No findings currently marked under investigation.")

    with tabs[4]:
        resolved_items = [f for f in findings if f.get("review_status") in ("resolved", "confirmed", "not_a_contradiction", "dismissed")]
        render_finding_cards(resolved_items, "No historical resolved findings.")
