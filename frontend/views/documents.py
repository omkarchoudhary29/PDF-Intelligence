"""Document Management View for Enterprise Knowledge Navigator.

Provides multi-file PDF upload with validation, duplicate detection,
filtering, detailed extraction inspection, re-indexing, download, deletion,
document family tracking, version comparison, and contradiction review.
"""

from typing import Any, Dict, List, Optional
import streamlit as st

from config.settings import get_settings
from frontend.api_client import APIClientError, BackendAPIClient
from frontend.styles import format_bytes, format_date, get_status_badge_html


def render_documents(
    client: BackendAPIClient,
    workspace: Dict[str, Any],
    current_user_role: str = "owner",
    initial_tab: Optional[str] = None,
) -> None:
    """Render the Document Repository and Version Intelligence Manager."""
    settings = get_settings()
    ws_id = workspace["id"]
    ws_name = workspace["name"]
    can_upload = current_user_role in ("owner", "admin")
    can_delete = current_user_role in ("owner", "admin")
    can_manage_versions = current_user_role in ("owner", "admin")

    col_title, col_refresh = st.columns([3, 1])
    with col_title:
        st.markdown(f"## Documents & Knowledge Library — {ws_name}")
        st.caption("Manage enterprise PDFs, upload new documents, track revisions, and review contradictions.")
    with col_refresh:
        if st.button("🔄 Refresh View", use_container_width=True):
            st.rerun()

    st.markdown("<hr style='margin: 0.6rem 0 1.25rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

    # --- Prominent Quick Upload Section at Top ---
    if can_upload:
        with st.expander("📤 Quick Upload PDF Document", expanded=st.session_state.pop("doc_open_upload", False)):
            st.markdown("**Upload a new PDF to this workspace:**")
            q_col1, q_col2 = st.columns([3, 1])
            with q_col1:
                top_files = st.file_uploader(
                    "Select PDF files to upload",
                    type=["pdf"],
                    accept_multiple_files=True,
                    key="doc_top_quick_uploader",
                    label_visibility="collapsed",
                )
            with q_col2:
                if top_files:
                    st.caption(f"**{len(top_files)} file(s) selected**")
                    if st.button("🚀 Upload & Start Ingestion", type="primary", use_container_width=True, key="doc_top_upload_btn"):
                        files_payload = [(f.name, f.getvalue()) for f in top_files]
                        with st.spinner("Uploading and indexing documents..."):
                            try:
                                res = client.upload_documents(ws_id, files_payload)
                                st.success(f"Upload complete: {res.get('total_uploaded', len(top_files))} uploaded, {res.get('total_duplicates', 0)} duplicates.")
                                st.rerun()
                            except APIClientError as err:
                                st.error(f"Upload failed: {err}")
                else:
                    st.caption("Select or drop PDF files here")

    section_options = [
        "📁 Document Library",
        "📤 Upload Documents",
        "📚 Document Families",
        "🔄 Compare Versions",
        "⚠️ Contradiction Review",
    ]
    default_tab_idx = 0
    tab_param = st.query_params.get("doc_tab") or initial_tab
    if tab_param:
        t_low = tab_param.lower()
        if "famil" in t_low or "version" in t_low:
            default_tab_idx = 2
        elif "contradict" in t_low or "finding" in t_low:
            default_tab_idx = 4
        elif "upload" in t_low:
            default_tab_idx = 1
        elif "compare" in t_low:
            default_tab_idx = 3
        elif "library" in t_low or "doc" in t_low:
            default_tab_idx = 0

    if tab_param and st.session_state.get("_last_doc_initial_tab") != tab_param:
        st.session_state["_last_doc_initial_tab"] = tab_param
        st.session_state["active_doc_section"] = section_options[default_tab_idx]
        st.session_state["doc_section_radio"] = section_options[default_tab_idx]

    current_sec = st.session_state.get("active_doc_section", section_options[default_tab_idx])
    if current_sec not in section_options:
        current_sec = section_options[default_tab_idx]

    selected_section = st.radio(
        "Document Section",
        options=section_options,
        index=section_options.index(current_sec),
        horizontal=True,
        label_visibility="collapsed",
        key=f"doc_section_radio_{st.session_state.get('_last_doc_initial_tab', 'default')}",
    )
    st.session_state["active_doc_section"] = selected_section

    # =========================================================================
    # TAB 1: Document Library
    # =========================================================================
    if selected_section == "📁 Document Library":
        filter_col, search_col = st.columns([1, 2])
        with filter_col:
            status_filter = st.selectbox(
                "Filter by Status",
                options=["All", "completed", "processing", "queued", "failed"],
                index=0,
                key="doc_status_filter",
            )
        with search_col:
            name_filter = st.text_input(
                "Filter by Filename",
                placeholder="Search by file name...",
                key="doc_name_filter",
            )

        api_status = None if status_filter == "All" else status_filter

        try:
            doc_data = client.list_documents(ws_id, scope="company", page=1, page_size=100, status=api_status)
            documents: List[Dict[str, Any]] = doc_data.get("items", [])
            total_count = doc_data.get("total", 0)

            if name_filter.strip():
                q = name_filter.strip().lower()
                documents = [d for d in documents if q in d["original_filename"].lower()]

            if not documents:
                st.markdown(
                    """
                    <div class="ekn-empty-state">
                        <div class="ekn-empty-title">No Documents Found</div>
                        <div class="ekn-empty-desc">
                            There are no documents matching your current filter in this workspace.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if can_upload:
                    with st.container(border=True):
                        st.markdown("#### 📤 Upload Your First Document Here")
                        first_files = st.file_uploader(
                            "Choose PDF files to upload",
                            type=["pdf"],
                            accept_multiple_files=True,
                            key="doc_empty_library_uploader",
                        )
                        if first_files:
                            st.caption(f"**{len(first_files)} file(s) selected**")
                            if st.button("🚀 Upload & Start Processing", type="primary", key="doc_empty_library_btn"):
                                files_payload = [(f.name, f.getvalue()) for f in first_files]
                                with st.spinner("Streaming, hashing, and storing documents..."):
                                    try:
                                        result = client.upload_documents(ws_id, files_payload)
                                        st.success(f"Upload complete: {result['total_uploaded']} uploaded, {result['total_duplicates']} duplicates.")
                                        st.rerun()
                                    except APIClientError as err:
                                        st.error(f"Upload failed: {err}")
            else:
                st.caption(f"Showing **{len(documents)}** of **{total_count}** documents in this workspace.")

                # Table Header
                h_cols = st.columns([2.6, 1.1, 1.1, 1.2, 3.8])
                h_cols[0].markdown("**Document & Version**")
                h_cols[1].markdown("**Extraction**")
                h_cols[2].markdown("**Vector Index**")
                h_cols[3].markdown("**Search Scope**")
                h_cols[4].markdown("**Actions**")

                st.markdown("<hr style='margin: 0.2rem 0 0.8rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

                for doc in documents:
                    d_id = doc["id"]
                    status_val = doc.get("processing_status", "uploaded")
                    v_num = doc.get("version_number", 1)
                    v_status = doc.get("version_status", "current")

                    cols = st.columns([2.6, 1.1, 1.1, 1.2, 3.8])

                    with cols[0]:
                        st.markdown(f"**{doc['original_filename']}**")
                        v_badge = f"<span class='ekn-badge' style='background:#e0e7ff;color:#3730a3;'>v{v_num}</span>"
                        s_badge = "<span class='ekn-badge' style='background:#dcfce7;color:#166534;'>Current</span>" if v_status == "current" else f"<span class='ekn-badge' style='background:#fef3c7;color:#92400e;'>{v_status.title()}</span>"
                        st.markdown(f"{v_badge} {s_badge} <small style='color:#64748b;'>{format_bytes(doc['file_size_bytes'])}</small>", unsafe_allow_html=True)

                    with cols[1]:
                        # Extraction status badge
                        extract_label = "Extracted" if status_val == "completed" else ("Extracting" if status_val == "processing" else status_val.title())
                        extract_cls = "ekn-badge-completed" if status_val == "completed" else ("ekn-badge-processing" if status_val == "processing" else ("ekn-badge-failed" if status_val == "failed" else "ekn-badge-queued"))
                        pages_str = f"{doc['total_pages']} pages" if doc.get("total_pages") else "Pending"
                        st.markdown(f"<span class='ekn-badge {extract_cls}'>{extract_label}</span><br/><small style='color:#64748b;'>{pages_str}</small>", unsafe_allow_html=True)

                    with cols[2]:
                        try:
                            idx_info = client.get_document_index(ws_id, d_id)
                            idx_record = idx_info.get("index")
                            is_search_ready = idx_info.get("is_search_ready", False)

                            if idx_record and isinstance(idx_record, dict):
                                idx_status = idx_record.get("status", "pending")
                                c_count = idx_record.get("total_children", 0)
                                if idx_status == "completed" and is_search_ready:
                                    st.markdown(
                                        f"<span class='ekn-badge ekn-badge-ready'>Ready</span><br/><small style='color:#64748b;'>{c_count} chunks</small>",
                                        unsafe_allow_html=True,
                                    )
                                elif idx_status == "indexing":
                                    st.markdown(
                                        "<span class='ekn-badge ekn-badge-indexing'>Indexing</span>",
                                        unsafe_allow_html=True,
                                    )
                                elif idx_status == "failed":
                                    err_msg = idx_record.get("error_message") or "Indexing failed"
                                    st.markdown(
                                        f"<span class='ekn-badge ekn-badge-failed' title='{err_msg}'>Failed</span>",
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.markdown(
                                        f"<span class='ekn-badge ekn-badge-queued'>{idx_status.title()}</span>",
                                        unsafe_allow_html=True,
                                    )
                            else:
                                if status_val == "completed":
                                    st.markdown(
                                        "<span class='ekn-badge ekn-badge-not-ready'>Not Indexed</span>",
                                        unsafe_allow_html=True,
                                    )
                                elif status_val in ("processing", "queued"):
                                    st.markdown(
                                        "<span class='ekn-badge ekn-badge-queued'>Waiting</span>",
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.markdown(
                                        "<span class='ekn-badge ekn-badge-neutral'>Pending</span>",
                                        unsafe_allow_html=True,
                                    )
                        except APIClientError:
                            st.markdown("<small style='color:#94a3b8;'>Unavailable</small>", unsafe_allow_html=True)
                        except Exception:
                            st.markdown("<small style='color:#ef4444;'>Error</small>", unsafe_allow_html=True)

                    with cols[3]:
                        if v_status == "current":
                            st.markdown("<span style='color:#16a34a; font-size:0.85rem; font-weight:600;'>Default Search</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("<span style='color:#d97706; font-size:0.85rem;'>Historical Scope</span>", unsafe_allow_html=True)

                    with cols[4]:
                        act_c1, act_c2, act_c3 = st.columns([1.2, 1.3, 1.1])

                        with act_c1:
                            if status_val == "completed":
                                with st.popover("Inspect"):
                                    st.markdown(f"**Extracted Pages: {doc['original_filename']}**")
                                    try:
                                        p_data = client.get_extracted_pages(ws_id, d_id)
                                        pages_list = p_data.get("pages", [])
                                        st.caption(f"Total extracted pages: {len(pages_list)}")
                                        for p in pages_list[:5]:
                                            st.markdown(f"**Page {p['page_number']}** ({p.get('extraction_method', 'native')})")
                                            st.text_area(f"p_{p['page_number']}", value=p.get("text", "")[:400], height=90, disabled=True, key=f"t_{d_id}_{p['page_number']}")
                                    except APIClientError as err:
                                        st.error(f"Error: {err}")
                            else:
                                st.button("Inspect", key=f"insp_dis_{d_id}", disabled=True, use_container_width=True)

                        with act_c2:
                            try:
                                pdf_bytes, dl_filename = client.download_document(ws_id, d_id)
                                st.download_button(
                                    label="Download",
                                    data=pdf_bytes,
                                    file_name=dl_filename,
                                    mime="application/pdf",
                                    key=f"dl_btn_{d_id}",
                                    use_container_width=True,
                                    help="Download original PDF",
                                )
                            except APIClientError:
                                st.button("Download", key=f"dl_err_{d_id}", disabled=True, use_container_width=True)

                        with act_c3:
                            with st.popover("More ⋯"):
                                st.markdown(f"**Document Operations: `{doc['original_filename']}`**")
                                st.markdown("<hr style='margin: 0.3rem 0;'/>", unsafe_allow_html=True)

                                # Version Intel Section
                                st.markdown("**Version Analysis**")
                                try:
                                    v_intel = client.detect_existing_document_version(ws_id, d_id)
                                    cls_color = {
                                        "exact_duplicate": "#dc2626",
                                        "possible_new_version": "#16a34a",
                                        "uncertain_match": "#d97706",
                                        "new_document": "#2563eb",
                                    }.get(v_intel.get("classification"), "#475569")
                                    st.markdown(f"Classification: <strong style='color:{cls_color};'>{v_intel.get('classification', '').replace('_', ' ').title()}</strong> ({round(v_intel.get('confidence', 0)*100)}% conf)", unsafe_allow_html=True)
                                    st.caption(f"Analysis: {v_intel.get('explanation')}")
                                except APIClientError as err:
                                    st.caption(f"Analysis error: {err}")

                                st.markdown("<hr style='margin: 0.4rem 0;'/>", unsafe_allow_html=True)

                                # Re-index Action
                                if can_manage_versions and status_val == "completed":
                                    if st.button("⚡ Re-Index Document", key=f"reidx_{d_id}", use_container_width=True):
                                        try:
                                            with st.spinner("Re-indexing vectors and BM25..."):
                                                client.reindex_document(ws_id, d_id, force=True)
                                            st.success("Re-indexed successfully!")
                                            st.rerun()
                                        except APIClientError as err:
                                            st.error(f"Error: {err}")

                                # Delete Action
                                if can_delete:
                                    st.markdown("<hr style='margin: 0.4rem 0;'/>", unsafe_allow_html=True)
                                    st.markdown("<span style='color: #DC2626; font-size: 0.85rem; font-weight: 600;'>Danger Zone</span>", unsafe_allow_html=True)
                                    if st.button("🗑️ Delete Document", key=f"del_btn_{d_id}", type="secondary", use_container_width=True):
                                        try:
                                            client.delete_document(ws_id, d_id)
                                            st.success("Deleted!")
                                            st.rerun()
                                        except APIClientError as err:
                                            st.error(f"Delete failed: {err}")

                    st.markdown("<hr style='margin: 0.3rem 0; border: none; border-top: 1px solid #f1f5f9;'/>", unsafe_allow_html=True)

        except APIClientError as err:
            st.error(f"Failed to fetch documents: {err}")

    # =========================================================================
    # TAB 2: Upload Documents (New or New Version)
    # =========================================================================
    elif selected_section == "📤 Upload Documents":
        if not can_upload:
            st.info("🔒 Document upload is restricted to Workspace Administrators and Owners.")
        else:
            st.markdown("#### Upload Enterprise Documents")
            st.write(
                "Upload PDFs to this workspace. Choose whether to register files as **New Independent Documents** "
                "or as a **New Revision of an Existing Document Family**."
            )

            upload_mode = st.radio(
                "Upload Mode",
                options=["New Independent Document(s)", "New Version of Existing Document Family"],
                index=0,
                key="upload_mode_radio",
            )

            if upload_mode == "New Independent Document(s)":
                uploaded_files = st.file_uploader(
                    "Select PDF files to upload",
                    type=["pdf"],
                    accept_multiple_files=True,
                    help=f"Max {settings.max_files_per_upload} files per batch. Max file size: {settings.max_upload_size_bytes // (1024*1024)} MB.",
                    key="file_uploader_batch",
                )

                if uploaded_files:
                    st.markdown(f"**Selected Files ({len(uploaded_files)}):**")
                    for uf in uploaded_files:
                        st.caption(f"• `{uf.name}` ({format_bytes(uf.size)})")

                    if st.button("Upload Independent Files", type="primary", key="batch_upload_btn"):
                        files_payload = [(uf.name, uf.getvalue()) for uf in uploaded_files]
                        with st.spinner("Streaming, hashing, and storing documents..."):
                            try:
                                result = client.upload_documents(ws_id, files_payload)
                                st.success(f"Upload complete: {result['total_uploaded']} uploaded, {result['total_duplicates']} duplicates detected.")
                                st.rerun()
                            except APIClientError as err:
                                st.error(f"Upload failed: {err}")

            else:
                # New version of existing family
                try:
                    families = client.list_document_families(ws_id)
                    if not families:
                        st.warning("No existing document families available. Upload an independent document first.")
                    else:
                        fam_map = {f"{f['title']} ({len(f.get('versions', []))} versions)": f["id"] for f in families}
                        sel_fam_title = st.selectbox("Select Target Document Family", options=list(fam_map.keys()), key="ver_fam_select")
                        target_family_id = fam_map[sel_fam_title]

                        version_file = st.file_uploader(
                            "Select Revised PDF file",
                            type=["pdf"],
                            accept_multiple_files=False,
                            key="version_single_uploader",
                        )

                        if version_file:
                            st.caption(f"Selected revision: `{version_file.name}` ({format_bytes(version_file.size)})")
                            if st.button("Upload as New Version", type="primary", key="ver_upload_btn"):
                                with st.spinner("Uploading and linking sequential version..."):
                                    try:
                                        res = client.upload_document_version(
                                            workspace_id=ws_id,
                                            family_id=target_family_id,
                                            file_name=version_file.name,
                                            file_bytes=version_file.getvalue(),
                                        )
                                        if res.get("status") == "uploaded":
                                            st.success(res.get("message", "Uploaded successfully!"))
                                            st.rerun()
                                        elif res.get("status") == "duplicate":
                                            st.warning(res.get("message", "Duplicate detected."))
                                        else:
                                            st.error(res.get("message", "Upload error."))
                                    except APIClientError as err:
                                        st.error(f"Version upload failed: {err}")
                except APIClientError as err:
                    st.error(f"Error: {err}")

    # =========================================================================
    # TAB 3: Document Families & Version Lineage
    # =========================================================================
    elif selected_section == "📚 Document Families":
        col_fam_hdr, col_fam_btn = st.columns([3, 1.2])
        with col_fam_hdr:
            st.markdown("#### Document Families & Sequential Revision Lineage")
            st.write(
                "Document families group multiple sequential revisions of the same enterprise document. "
                "The designated **Current Version** is searched by default; older versions remain accessible for historical queries."
            )
        with col_fam_btn:
            if can_manage_versions:
                with st.popover("➕ Create Document Family", use_container_width=True):
                    with st.form("create_family_form"):
                        fam_title_input = st.text_input("Family Title*", placeholder="e.g. Flight Operations Manual")
                        fam_canon_input = st.text_input("Canonical Slug (Optional)", placeholder="e.g. flight_operations_manual")
                        if st.form_submit_button("Create Family", type="primary", use_container_width=True):
                            if fam_title_input.strip():
                                try:
                                    client.create_document_family(
                                        workspace_id=ws_id,
                                        title=fam_title_input.strip(),
                                        canonical_name=fam_canon_input.strip() if fam_canon_input.strip() else None,
                                    )
                                    st.success(f"Family '{fam_title_input}' created!")
                                    st.rerun()
                                except APIClientError as err:
                                    st.error(f"Error: {err}")
                            else:
                                st.warning("Please provide a family title.")

        try:
            families = client.list_document_families(ws_id)
            if not families:
                st.info("No document families recorded yet in this workspace. Upload documents to create families.")
            else:
                for fam in families:
                    f_id = fam["id"]
                    versions = fam.get("versions", [])
                    curr_doc_id = fam.get("current_document_id")

                    with st.expander(f"📁 **{fam['title']}** ({len(versions)} versions) — Canonical: `{fam['canonical_name']}`", expanded=True):
                        st.caption(f"Family ID: `{f_id}` | Registered: {format_date(fam['created_at'])}")

                        for v in versions:
                            is_curr = (v["id"] == curr_doc_id)
                            v_badge = f"<span class='ekn-badge' style='background:#dcfce7;color:#166534;'>Active Current</span>" if is_curr else f"<span class='ekn-badge' style='background:#f1f5f9;color:#475569;'>{v.get('version_status', 'superseded').title()}</span>"
                            app_badge = f"<span class='ekn-badge' style='background:#e0f2fe;color:#0369a1;'>Approval: {v.get('approval_status', 'unverified').title()}</span>"

                            v_cols = st.columns([3, 1.5, 2.5])
                            with v_cols[0]:
                                st.markdown(f"**Version {v.get('version_number', 1)}**: `{v['original_filename']}` {v_badge} {app_badge}", unsafe_allow_html=True)
                                st.caption(f"Uploaded: {format_date(v['created_at'])} | Size: {format_bytes(v['file_size_bytes'])}")

                            with v_cols[1]:
                                if can_manage_versions:
                                    app_opts = ["unverified", "pending_review", "approved", "rejected"]
                                    curr_app = v.get("approval_status", "unverified")
                                    app_idx = app_opts.index(curr_app) if curr_app in app_opts else 0
                                    new_app = st.selectbox("Status", options=app_opts, index=app_idx, key=f"app_sel_{v['id']}", label_visibility="collapsed")
                                    if new_app != curr_app:
                                        if st.button("Update", key=f"upd_app_{v['id']}"):
                                            try:
                                                client.update_approval_status(ws_id, v["id"], new_app)
                                                st.success("Approval status updated!")
                                                st.rerun()
                                            except APIClientError as err:
                                                st.error(f"Error: {err}")

                            with v_cols[2]:
                                if can_manage_versions:
                                    b_col1, b_col2 = st.columns([1.5, 1.2])
                                    with b_col1:
                                        if not is_curr:
                                            if st.button("Make Current", key=f"make_curr_{v['id']}", use_container_width=True):
                                                try:
                                                    client.set_current_version(ws_id, f_id, v["id"])
                                                    st.success(f"Version {v.get('version_number', 1)} set to Current!")
                                                    st.rerun()
                                                except APIClientError as err:
                                                    st.error(f"Error: {err}")
                                        else:
                                            st.markdown("<span style='color:#16a34a; font-weight:600;'>Primary Target</span>", unsafe_allow_html=True)
                                    with b_col2:
                                        if len(versions) > 1:
                                            if st.button("Detach", key=f"detach_{v['id']}", help="Split this version into an independent document family", use_container_width=True):
                                                try:
                                                    client.detach_document_from_family(ws_id, f_id, v["id"])
                                                    st.success("Detached into independent family.")
                                                    st.rerun()
                                                except APIClientError as err:
                                                    st.error(f"Detach failed: {err}")

                            st.markdown("<hr style='margin: 0.2rem 0; border: none; border-top: 1px solid #f8fafc;'/>", unsafe_allow_html=True)

                            st.markdown("<hr style='margin: 0.2rem 0; border: none; border-top: 1px solid #f8fafc;'/>", unsafe_allow_html=True)

        except APIClientError as err:
            st.error(f"Failed to fetch document families: {err}")

    # =========================================================================
    # TAB 4: Compare Versions
    # =========================================================================
    elif selected_section == "🔄 Compare Versions":
        st.markdown("#### Document Version Comparison & Page-Level Diffing")
        st.write(
            "Select two versions within a document family to perform deterministic page alignment, "
            "text/table diffing, and AI-assisted change explanation."
        )

        try:
            families = client.list_document_families(ws_id)
            if not families:
                st.info("No document families available for comparison.")
            else:
                fam_options = {f"{f['title']} ({len(f.get('versions', []))} versions)": f for f in families}
                selected_fam_label = st.selectbox("Select Document Family", options=list(fam_options.keys()), key="cmp_fam_select")
                selected_fam = fam_options[selected_fam_label]
                fam_versions = selected_fam.get("versions", [])

                if len(fam_versions) < 2:
                    st.info(f"Family '{selected_fam['title']}' currently has only 1 version. Upload another version to compare.")
                else:
                    v_options = {f"v{v.get('version_number', 1)}: {v['original_filename']} ({v.get('version_status', 'current')})": v["id"] for v in fam_versions}

                    c_src, c_tgt = st.columns(2)
                    with c_src:
                        src_label = st.selectbox("Source (Earlier) Version", options=list(v_options.keys()), index=0, key="cmp_src_select")
                        src_doc_id = v_options[src_label]
                    with c_tgt:
                        tgt_label = st.selectbox("Target (Later) Version", options=list(v_options.keys()), index=len(v_options) - 1, key="cmp_tgt_select")
                        tgt_doc_id = v_options[tgt_label]

                    if src_doc_id == tgt_doc_id:
                        st.warning("Please select two distinct versions to compare.")
                    else:
                        if st.button("Compare Versions", type="primary", key="btn_run_comparison"):
                            with st.spinner("Aligning pages, analyzing text diffs, and generating summary..."):
                                try:
                                    comp = client.compare_versions(
                                        workspace_id=ws_id,
                                        family_id=selected_fam["id"],
                                        source_doc_id=src_doc_id,
                                        target_doc_id=tgt_doc_id,
                                    )
                                    st.session_state["active_comparison"] = comp
                                except APIClientError as err:
                                    st.error(f"Comparison failed: {err}")

                        # Render active comparison results
                        if "active_comparison" in st.session_state:
                            comp = st.session_state["active_comparison"]
                            stats = comp.get("stats", {})

                            st.markdown("### Comparison Results")
                            s_cols = st.columns(4)
                            s_cols[0].metric("Unchanged Pages", stats.get("unchanged_pages", 0))
                            s_cols[1].metric("Modified Pages", stats.get("modified_pages", 0))
                            s_cols[2].metric("Added Pages", stats.get("added_pages", 0))
                            s_cols[3].metric("Removed Pages", stats.get("removed_pages", 0))

                            # AI Explanation
                            if comp.get("ai_explanation"):
                                st.markdown("##### 🤖 Executive Change Explanation")
                                st.info(comp["ai_explanation"])
                            elif comp.get("change_summary"):
                                st.markdown("##### 📋 Summary of Changes")
                                st.markdown(f"*{comp['change_summary']}*")

                            # Page Alignment Overview
                            alignments = comp.get("page_alignments", [])
                            if alignments:
                                with st.expander(f"📑 Page Alignment Map ({len(alignments)} pages)", expanded=False):
                                    for a in alignments:
                                        sp = f"Page {a['source_page']}" if a.get("source_page") else "None"
                                        tp = f"Page {a['target_page']}" if a.get("target_page") else "None"
                                        st.write(f"• **{sp}** ➔ **{tp}** | Status: `{a['status']}` (Similarity: {a.get('similarity_score', 1.0)})")

                            # Numerical Parameter Changes
                            num_changes = comp.get("numerical_changes", [])
                            if num_changes:
                                with st.expander(f"🔢 Numerical & Tolerance Changes ({len(num_changes)})", expanded=True):
                                    for nc in num_changes:
                                        u = f" {nc['units']}" if nc.get("units") else ""
                                        st.markdown(
                                            f"- **Page {nc.get('source_page')} ➔ {nc.get('target_page')}**: "
                                            f"Changed from `{nc['old_value']}{u}` to `{nc['new_value']}{u}` "
                                            f"<br/><small style='color:#64748b;'>{nc.get('context', '')}</small>",
                                            unsafe_allow_html=True,
                                        )

                            # Table Differences
                            tab_diffs = comp.get("table_diffs", [])
                            if tab_diffs:
                                with st.expander(f"📊 Table Differences ({len(tab_diffs)})", expanded=True):
                                    for tb in tab_diffs:
                                        st.markdown(f"- **Page {tb.get('source_page')} ➔ {tb.get('target_page')}**: Type: `{tb.get('change_type')}` — Details: {tb.get('details')}")

                            # Text Differences
                            text_diffs = comp.get("text_diffs", [])
                            if text_diffs:
                                with st.expander(f"📝 Text Modifications & Excerpts ({len(text_diffs)})", expanded=True):
                                    for td in text_diffs:
                                        cat = td.get("category", "Unclassified")
                                        cat_color = {
                                            "Safety-related": "#dc2626",
                                            "Technical specification": "#2563eb",
                                            "Regulatory": "#d97706",
                                            "Procedural": "#7c3aed",
                                            "Responsibility change": "#0891b2",
                                            "Administrative": "#475569",
                                            "Formatting-only": "#64748b",
                                        }.get(cat, "#334155")
                                        cat_badge = f"<span class='ekn-badge' style='background:{cat_color}15;color:{cat_color};border:1px solid {cat_color}40;font-weight:600;'>{cat}</span>"
                                        st.markdown(f"**Page {td.get('source_page')} ➔ {td.get('target_page')}** ({td.get('diff_type')}) {cat_badge}", unsafe_allow_html=True)
                                        d_cols = st.columns(2)
                                        with d_cols[0]:
                                            st.markdown("**Previous Version Excerpt:**")
                                            st.text_area("src", td.get("source_excerpt", "(None)"), height=80, disabled=True, key=f"td_src_{td.get('source_page')}_{td.get('target_page')}")
                                        with d_cols[1]:
                                            st.markdown("**Revised Version Excerpt:**")
                                            st.text_area("tgt", td.get("target_excerpt", "(None)"), height=80, disabled=True, key=f"td_tgt_{td.get('source_page')}_{td.get('target_page')}")

        except APIClientError as err:
            st.error(f"Comparison error: {err}")

    # =========================================================================
    # TAB 5: Contradiction Review
    # =========================================================================
    elif selected_section == "⚠️ Contradiction Review":
        st.markdown("#### Evidence-Backed Contradiction Review")
        st.write(
            "Detect and review potential factual contradictions, numerical discrepancies, and conflicting instructions "
            "across enterprise documents. Distinguishes legitimate revisions and context-dependent differences from unresolved conflicts."
        )

        col_scan, col_status, col_type = st.columns([1.5, 1.5, 1.5])
        with col_scan:
            scan_clicked = st.button("🔍 Scan for Contradictions", use_container_width=True, help="Scan document pairs connected via knowledge graph or family revisions")
        with col_status:
            stat_options = ["All", "candidate", "under_review", "confirmed", "not_a_contradiction", "dismissed", "resolved"]
            stat_labels = {
                "All": "All Statuses",
                "candidate": "Pending Review",
                "under_review": "Under Review",
                "confirmed": "Confirmed Issue",
                "not_a_contradiction": "Not a Contradiction",
                "dismissed": "Dismissed",
                "resolved": "Resolved",
            }
            review_filter = st.selectbox(
                "Status Filter",
                options=stat_options,
                format_func=lambda s: stat_labels.get(s, s),
                index=0,
                key="find_stat_filter",
            )
        with col_type:
            type_options = [
                "All",
                "numerical_discrepancy",
                "instruction_conflict",
                "dated_revision",
                "context_dependent_difference",
                "direct_factual_conflict",
                "date_validity_conflict",
            ]
            type_labels = {
                "All": "All Categories",
                "numerical_discrepancy": "Numerical difference",
                "instruction_conflict": "Procedural disagreement",
                "dated_revision": "Potentially outdated info",
                "context_dependent_difference": "Scope-dependent difference",
                "direct_factual_conflict": "Direct factual conflict",
                "date_validity_conflict": "Date/validity conflict",
            }
            type_filter = st.selectbox(
                "Category Filter",
                options=type_options,
                format_func=lambda t: type_labels.get(t, t),
                index=0,
                key="find_type_filter",
            )

        try:
            findings = client.list_findings(
                workspace_id=ws_id,
                status_filter=review_filter,
                finding_type=type_filter,
                scan=scan_clicked,
            )

            if not findings:
                st.info("No contradiction findings match the selected filters.")
            else:
                st.caption(f"Showing **{len(findings)}** contradiction finding(s).")
                for f in findings:
                    f_id = f["id"]
                    status_badge = {
                        "candidate": "<span class='ekn-badge' style='background:#fef3c7;color:#92400e;'>Pending Review</span>",
                        "under_review": "<span class='ekn-badge' style='background:#ede9fe;color:#6d28d9;'>Under Review</span>",
                        "confirmed": "<span class='ekn-badge' style='background:#fee2e2;color:#991b1b;'>Confirmed Issue</span>",
                        "not_a_contradiction": "<span class='ekn-badge' style='background:#dbeafe;color:#1e40af;'>Not a Contradiction</span>",
                        "dismissed": "<span class='ekn-badge' style='background:#f1f5f9;color:#64748b;'>Dismissed</span>",
                        "resolved": "<span class='ekn-badge' style='background:#dcfce7;color:#166534;'>Resolved</span>",
                    }.get(f["review_status"], f["review_status"])

                    with st.container():
                        st.markdown(f"### {f['subject']} — `{f['finding_type']}` {status_badge}", unsafe_allow_html=True)
                        st.write(f"**Explanation:** {f['explanation']}")

                        e_col1, e_col2 = st.columns(2)
                        with e_col1:
                            st.markdown(f"**Evidence 1 (Document: `{f['first_document_id'][:8]}...`, Page {f['first_page']}):**")
                            st.info(f"\"{f['first_evidence_excerpt']}\"")
                        with e_col2:
                            st.markdown(f"**Evidence 2 (Document: `{f['second_document_id'][:8]}...`, Page {f['second_page']}):**")
                            st.info(f"\"{f['second_evidence_excerpt']}\"")

                        if f.get("review_reason"):
                            st.caption(f"Review Notes: {f['review_reason']}")

                        # Review Action Controls (Owners/Admins only)
                        if can_manage_versions and f["review_status"] in ("candidate", "under_review"):
                            with st.expander("Review Finding Decisions", expanded=False):
                                r_cols = st.columns(4)
                                with r_cols[0]:
                                    if st.button("Confirm Issue", key=f"conf_{f_id}", type="primary", use_container_width=True):
                                        try:
                                            client.review_finding(ws_id, f_id, review_status="confirmed", review_reason="Confirmed by authorized reviewer.")
                                            st.success("Finding confirmed as issue.")
                                            st.rerun()
                                        except APIClientError as err:
                                            st.error(f"Error: {err}")
                                with r_cols[1]:
                                    if st.button("Under Review", key=f"urev_{f_id}", use_container_width=True):
                                        try:
                                            client.review_finding(ws_id, f_id, review_status="under_review", review_reason="Investigation underway.")
                                            st.success("Status set to Under Review.")
                                            st.rerun()
                                        except APIClientError as err:
                                            st.error(f"Error: {err}")
                                with r_cols[2]:
                                    if st.button("Not Contradiction", key=f"notc_{f_id}", use_container_width=True):
                                        try:
                                            client.review_finding(ws_id, f_id, review_status="not_a_contradiction", review_reason="Evaluated as compatible or context-dependent.")
                                            st.success("Marked as Not a Contradiction.")
                                            st.rerun()
                                        except APIClientError as err:
                                            st.error(f"Error: {err}")
                                with r_cols[3]:
                                    with st.popover("Resolve Authority"):
                                        auth_doc = st.selectbox(
                                            "Select Authoritative Document",
                                            options=[f['first_document_id'], f['second_document_id']],
                                            key=f"auth_sel_{f_id}",
                                        )
                                        reason = st.text_input("Resolution Explanation", placeholder="E.g. Approved standard supersedes draft procedure", key=f"res_txt_{f_id}")
                                        if st.button("Submit Resolution", key=f"res_btn_{f_id}", type="primary"):
                                            try:
                                                client.review_finding(
                                                    workspace_id=ws_id,
                                                    finding_id=f_id,
                                                    review_status="resolved",
                                                    review_reason=reason or "Resolved by selecting authoritative document.",
                                                    authoritative_document_id=auth_doc,
                                                )
                                                st.success("Finding resolved.")
                                                st.rerun()
                                            except APIClientError as err:
                                                st.error(f"Error: {err}")

                        st.markdown("<hr style='margin: 0.8rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

        except APIClientError as err:
            st.error(f"Failed to fetch contradiction findings: {err}")
