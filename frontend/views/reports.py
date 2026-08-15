"""Reports & Dossiers View for Enterprise Knowledge Navigator.

Provides automated generation, tracking, inspection, downloading, and deletion
of consolidated Enterprise Knowledge Dossier PDFs.
"""

from typing import Any, Dict, List, Optional
import streamlit as st

from frontend.api_client import APIClientError, BackendAPIClient
from frontend.styles import format_bytes, format_date, get_status_badge_html


def render_reports(
    client: BackendAPIClient,
    workspace: Dict[str, Any],
    current_user_role: str = "owner",
) -> None:
    """Render the Reports and Knowledge Dossier management interface."""
    ws_id = workspace["id"]
    ws_name = workspace["name"]
    can_generate = current_user_role in ("owner", "admin")
    can_delete = current_user_role == "owner"

    # Header and Refresh
    col_hdr, col_ref = st.columns([3, 1])
    with col_hdr:
        st.markdown(f"## Reports & Dossiers — {ws_name}")
        st.caption("Generate, download, and manage consolidated Enterprise Knowledge Dossier PDFs.")
    with col_ref:
        if st.button("Refresh Reports", use_container_width=True):
            st.rerun()

    st.markdown("<hr style='margin: 0.8rem 0 1.25rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

    # 1. Generate New Dossier Section
    with st.expander("Generate New Knowledge Dossier", expanded=can_generate):
        if not can_generate:
            st.info("🔒 Generating Knowledge Dossiers is restricted to Workspace Administrators and Owners.")
        else:
            st.write(
                "An Enterprise Knowledge Dossier consolidates all indexed PDFs in this workspace into a compact, "
                "navigable master document containing executive synthesis, document summaries, cross-document intelligence, "
                "an embedded knowledge graph, and working source hyperlinks."
            )

            with st.form("generate_report_form", clear_on_submit=False):
                report_title_input = st.text_input(
                    "Dossier Title (Optional)",
                    placeholder=f"e.g. {ws_name} — Technical Knowledge Dossier",
                )
                include_graph_cb = st.checkbox("Embed Knowledge Graph Network Visualization", value=True)
                submit_btn = st.form_submit_button("Generate Knowledge Dossier PDF", type="primary")

                if submit_btn:
                    with st.spinner("Synthesizing document summaries, assembling knowledge graph, and generating PDF dossier..."):
                        try:
                            res = client.create_report(
                                workspace_id=ws_id,
                                report_name=report_title_input.strip() if report_title_input.strip() else None,
                                include_graph=include_graph_cb,
                                run_synchronous=True,
                            )
                            st.success(f"Dossier '{res['report_name']}' generated successfully ({format_bytes(res['file_size_bytes'])})!")
                            st.rerun()
                        except APIClientError as err:
                            st.error(f"Generation failed: {err}")

    st.markdown("<hr style='margin: 1.25rem 0 1.25rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

    # 2. Existing Reports List
    st.markdown("#### Generated Knowledge Dossiers")

    try:
        reports_data = client.list_reports(ws_id, page=1, page_size=50)
        reports_list: List[Dict[str, Any]] = reports_data.get("items", [])

        if not reports_list:
            st.markdown(
                """
                <div class="ekn-empty-state">
                    <div class="ekn-empty-title">No Reports Generated Yet</div>
                    <div class="ekn-empty-desc">
                        Generate your first Enterprise Knowledge Dossier above to compile an executive overview,
                        master index, document summaries, and relational network for this workspace.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        st.caption(f"Showing **{len(reports_list)}** generated report(s).")

        # Table Header
        h_cols = st.columns([2.8, 1.3, 1.3, 1.4, 2.8])
        h_cols[0].markdown("**Report Name**")
        h_cols[1].markdown("**Status**")
        h_cols[2].markdown("**Coverage**")
        h_cols[3].markdown("**File Size**")
        h_cols[4].markdown("**Actions**")

        st.markdown("<hr style='margin: 0.2rem 0 0.8rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

        for rep in reports_list:
            r_id = rep["id"]
            r_status = rep.get("status", "completed")
            r_cols = st.columns([2.8, 1.3, 1.3, 1.4, 2.8])

            with r_cols[0]:
                st.write(f"📄 **{rep['report_name']}**")
                st.caption(f"Created: {format_date(rep.get('created_at', ''))}")
                if rep.get("error_message"):
                    st.caption(f":red[{rep['error_message'][:70]}...]")

            with r_cols[1]:
                badge_type = r_status.replace("partially_completed", "processing")
                st.markdown(get_status_badge_html(badge_type), unsafe_allow_html=True)

            with r_cols[2]:
                doc_cnt = rep.get("document_count", 0)
                page_cnt = rep.get("total_pages_covered", 0)
                st.write(f"{doc_cnt} docs · {page_cnt} pages")

            with r_cols[3]:
                st.write(format_bytes(rep.get("file_size_bytes", 0)))

            with r_cols[4]:
                act_c1, act_c2 = st.columns([1.5, 1.2])

                # Download PDF Button
                with act_c1:
                    if r_status in ("completed", "partially_completed"):
                        try:
                            pdf_bytes, dl_filename = client.download_report(ws_id, r_id)
                            st.download_button(
                                label="Download PDF",
                                data=pdf_bytes,
                                file_name=dl_filename,
                                mime="application/pdf",
                                key=f"dl_rep_{r_id}",
                                use_container_width=True,
                            )
                        except Exception:
                            st.button("Download", key=f"dl_err_{r_id}", disabled=True, use_container_width=True)
                    else:
                        st.button("Pending", key=f"pend_{r_id}", disabled=True, use_container_width=True)

                # Delete Report Button with Popover Confirmation (Owner Only)
                with act_c2:
                    if can_delete:
                        with st.popover("Delete"):
                            st.write(f"Delete report **{rep['report_name']}**?")
                            if st.button("Confirm", key=f"del_rep_btn_{r_id}", type="primary", use_container_width=True):
                                try:
                                    client.delete_report(ws_id, r_id)
                                    st.success("Report deleted.")
                                    st.rerun()
                                except APIClientError as err:
                                    st.error(f"Delete failed: {err}")
                    else:
                        st.empty()

    except APIClientError as err:
        st.error(f"Error loading reports: {err}")
