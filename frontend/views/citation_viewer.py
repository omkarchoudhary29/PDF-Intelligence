"""Citation and Source Viewer Component for Enterprise Knowledge Navigator.

Renders verified citations with provenance metadata (filename, page numbers,
section title, and supporting excerpt) and provides interactive PyMuPDF-rendered
original page previews without exposing local filesystem paths.
Supports compact citation pills with collapsible evidence to keep conversational
responses clear, readable, and non-redundant.
"""

import html
from typing import Any, Dict, List
import streamlit as st

from frontend.api_client import APIClientError, BackendAPIClient


def render_citations(
    client: BackendAPIClient,
    workspace_id: str,
    citations: List[Dict[str, Any]],
    compact: bool = True,
) -> None:
    """Render a structured list of verified citations with original PDF page preview.

    When compact=True (default for central chat), renders compact clickable citation pills
    and collapses full excerpts inside an expander so answers remain conversational,
    while full context is displayed in the Context & Sources panel.
    """
    if not citations:
        return

    if compact:
        # 1. Compact Citation Pills Row
        pills_html = []
        for idx, cite in enumerate(citations):
            tag = html.escape(cite.get("tag", f"[E{idx + 1}]"))
            filename = html.escape(cite.get("filename", "Document"))
            page_start = cite.get("page_start", 1)
            pills_html.append(
                f"<span class='ekn-citation-pill'>📄 {tag} {filename} · p. {page_start}</span>"
            )

        st.markdown(
            f"""
            <div style="margin-top: 0.65rem; margin-bottom: 0.35rem;">
                <div style="font-size: 0.78rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.3rem;">
                    Verified Citations
                </div>
                <div>{''.join(pills_html)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Collapsed Supporting Evidence Expander
        with st.expander("📑 Supporting Evidence Excerpt", expanded=False):
            for idx, cite in enumerate(citations):
                tag = html.escape(cite.get("tag", f"[E{idx + 1}]"))
                filename = html.escape(cite.get("filename", "Unknown Document"))
                doc_id = cite.get("document_id")
                page_start = cite.get("page_start", 1)
                page_end = cite.get("page_end", page_start)
                section = html.escape(cite.get("section") or "")
                excerpt = html.escape(cite.get("excerpt", ""))

                page_label = f"Page {page_start}" if page_start == page_end else f"Pages {page_start}–{page_end}"

                st.markdown(
                    f"""
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 3px solid #4F46E5; border-radius: 6px; padding: 0.65rem 0.85rem; margin-bottom: 0.5rem;">
                        <div style="font-weight: 600; color: #1E293B; font-size: 0.86rem;">
                            <span class="ekn-badge ekn-badge-uploaded">{tag}</span> {filename} &middot; <span style="color:#64748B;">{page_label}</span>
                        </div>
                        {f"<div style='font-size: 0.78rem; color: #475569; font-style: italic; margin-top: 0.2rem;'>Section: {section}</div>" if section else ""}
                        {f"<div style='font-size: 0.82rem; color: #334155; background: #FFFFFF; border: 1px solid #F1F5F9; border-radius: 4px; padding: 0.4rem 0.6rem; margin-top: 0.35rem; font-family: monospace;'>\"{excerpt[:240]}...\"</div>" if excerpt else ""}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if doc_id:
                    col_p, col_d, _ = st.columns([1.5, 1.5, 3.0])
                    with col_p:
                        with st.popover(f"Preview {page_label}"):
                            st.caption(f"Original PDF Page Render: **{filename}** ({page_label})")
                            try:
                                png_bytes = client.preview_page(
                                    workspace_id=workspace_id,
                                    document_id=doc_id,
                                    page_number=page_start,
                                    dpi=150,
                                )
                                st.image(png_bytes, caption=f"{filename} — {page_label}", use_container_width=True)
                            except APIClientError as err:
                                st.error(f"Could not render page preview: {err}")
                    with col_d:
                        try:
                            file_bytes, dl_filename = client.download_document(workspace_id, doc_id)
                            st.download_button(
                                label="Download PDF",
                                data=file_bytes,
                                file_name=dl_filename,
                                mime="application/pdf",
                                key=f"cite_compact_dl_{doc_id}_{idx}",
                            )
                        except Exception:
                            pass

            st.markdown(
                """
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.4rem;">
                    💡 Full passage evidence and continuous search scope are open in the <strong>Context & Sources</strong> panel on the right.
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    # Non-compact mode (full cards)
    st.markdown("##### Verified Citations & Sources")
    for idx, cite in enumerate(citations):
        tag = cite.get("tag", f"[E{idx + 1}]")
        filename = cite.get("filename", "Unknown Document")
        doc_id = cite.get("document_id")
        page_start = cite.get("page_start", 1)
        page_end = cite.get("page_end", page_start)
        section = cite.get("section")
        excerpt = cite.get("excerpt", "")

        page_label = f"Page {page_start}" if page_start == page_end else f"Pages {page_start}–{page_end}"

        with st.container():
            st.markdown(
                f"""
                <div class="ekn-citation-card">
                    <div class="ekn-citation-header">
                        <div>
                            <span class="ekn-citation-tag">{tag}</span>
                            <span class="ekn-citation-source"> &nbsp;{filename}</span>
                            <span class="ekn-citation-page"> &middot; {page_label}</span>
                        </div>
                    </div>
                """,
                unsafe_allow_html=True,
            )

            if section:
                st.markdown(f"<div class='ekn-citation-section'>Section: {section}</div>", unsafe_allow_html=True)

            if excerpt:
                st.markdown(f"<div class='ekn-citation-excerpt'>\"{excerpt}\"</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            if doc_id:
                col_prev, col_dl, _ = st.columns([1.5, 1.5, 4.0])
                with col_prev:
                    with st.popover(f"Preview {page_label}"):
                        st.caption(f"Original PDF Page Render: **{filename}** ({page_label})")
                        try:
                            png_bytes = client.preview_page(
                                workspace_id=workspace_id,
                                document_id=doc_id,
                                page_number=page_start,
                                dpi=150,
                            )
                            st.image(png_bytes, caption=f"{filename} — {page_label}", use_container_width=True)
                        except APIClientError as err:
                            st.error(f"Could not render page preview: {err}")

                with col_dl:
                    try:
                        file_bytes, dl_filename = client.download_document(workspace_id, doc_id)
                        st.download_button(
                            label="Download PDF",
                            data=file_bytes,
                            file_name=dl_filename,
                            mime="application/pdf",
                            key=f"cite_dl_{doc_id}_{idx}",
                        )
                    except Exception:
                        pass
