"""Employee Chat-First Workspace View for Enterprise Knowledge Navigator.

Implements:
1. Complete three-panel ChatGPT-like experience with independent content scrolling:
   - LEFT: Collapsible sidebar with branding, new chat, conversation search,
           scrollable conversation history (rename/delete), My Files/Company Files nav, and profile.
   - CENTRE: Main AI conversation with scrollable message history, scope badge, suggestion pills,
             normalized Markdown answers, verified citations, and docked composer beneath history.
   - RIGHT: Toggleable segmented panel (Context & Sources, My Files, Company Files)
            with source scope selector, workspace info, cited sources derived strictly from
            active conversation, personal PDF upload with PyMuPDF validation, download, preview, attach/detach.
2. Strict conversation-specific knowledge scopes ('company', 'personal', 'combined') and
   dedicated `pending_new_chat_scope` session state for new conversations.
3. Complete citation isolation between conversations.
4. Strict employee authorization and read-only enforcement on company documents.
"""

import html
import re
from typing import Any, Dict, List, Optional
import fitz  # PyMuPDF for client-side PDF verification
import streamlit as st

from frontend.api_client import APIClientError, BackendAPIClient
from frontend.styles import format_bytes, format_date, get_status_badge_html
from frontend.views.citation_viewer import render_citations


def _normalize_markdown(text: str) -> str:
    """Ensure Markdown headings and lists have proper spacing for Streamlit CommonMark parsing.

    Guarantees code blocks (fenced with ```) are completely protected from mutation,
    while ensuring proper blank lines before headings (#, ##, ###, etc.) outside code blocks.
    """
    if not text:
        return ""

    lines = text.split("\n")
    result: List[str] = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()

        # Track code block boundaries
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            result.append(line)
            continue

        if not in_code_block:
            # If line is a Markdown heading (# H1, ## H2, ### H3, etc.)
            if re.match(r"^#{1,6}\s", stripped):
                # Ensure at least one empty line before this heading if preceded by non-empty text
                if result and result[-1].strip() != "":
                    result.append("")

        result.append(line)

    return "\n".join(result)


def render_employee_workspace(client: BackendAPIClient, workspace: Dict[str, Any]) -> None:
    """Render the complete three-panel employee conversational workspace."""
    ws_id = workspace["id"]
    ws_name = workspace.get("name", "Workspace")
    ws_name_escaped = html.escape(ws_name)

    auth_user = st.session_state.get("auth_user") or {}
    user_name = auth_user.get("full_name", "Employee")
    user_email = auth_user.get("email", "")

    # Ensure workspace-scoped session state keys
    current_ws_key = st.session_state.get("_emp_current_ws_id")
    if current_ws_key != ws_id:
        st.session_state["_emp_current_ws_id"] = ws_id
        if "conv_id" not in st.query_params:
            st.session_state["active_conversation_id"] = None
        st.session_state["emp_left_expanded"] = True
        if "emp_right_open" not in st.query_params:
            st.session_state["emp_right_open"] = True  # Default OPEN as mandated by Reference A
        if "emp_right_view" not in st.query_params:
            st.session_state["emp_right_view"] = "Context & Sources"
        st.session_state.pop("pending_new_chat_scope", None)

    # Query param overrides
    qp = st.query_params
    if "conv_id" in qp:
        st.session_state["active_conversation_id"] = qp.get("conv_id")
    if "emp_right_open" in qp:
        st.session_state["emp_right_open"] = (qp.get("emp_right_open", "true").lower() == "true")
    if "emp_right_view" in qp:
        req_v = qp.get("emp_right_view")
        st.session_state["emp_right_view"] = req_v
        st.session_state["emp_right_view_selector"] = req_v

    # Initialize layout toggles (Context & Sources OPEN by default)
    if "emp_left_expanded" not in st.session_state:
        st.session_state["emp_left_expanded"] = True
    if "emp_right_open" not in st.session_state:
        st.session_state["emp_right_open"] = True
    if "emp_right_view" not in st.session_state:
        st.session_state["emp_right_view"] = "Context & Sources"

    # Fetch total documents count for Files toggle badge
    try:
        personal_docs_resp = client.list_documents(ws_id, scope="personal", page_size=100)
        company_docs_resp = client.list_documents(ws_id, scope="company", page_size=100)
        personal_docs = personal_docs_resp.get("items", [])
        company_docs = company_docs_resp.get("items", [])
    except Exception:
        personal_docs = []
        company_docs = []

    # =========================================================================
    # 1. LOAD CONVERSATIONS LIST
    # =========================================================================
    conversations: List[Dict[str, Any]] = []
    try:
        conv_resp = client.list_conversations(ws_id, page=1, page_size=100)
        conversations = conv_resp.get("items", [])
    except APIClientError as err:
        st.error(f"Error loading conversations: {err}")

    # Ensure valid active conversation selection
    active_conv_id = st.session_state.get("active_conversation_id")
    conv_ids = [c["id"] for c in conversations if "id" in c]
    if qp.get("conv_id") == "new":
        active_conv_id = None
        st.session_state["active_conversation_id"] = None
    elif conversations and (not active_conv_id or active_conv_id not in conv_ids):
        active_conv_id = conversations[0]["id"]
        st.session_state["active_conversation_id"] = active_conv_id

    # Retrieve active conversation details & scope
    active_conv = next((c for c in conversations if c.get("id") == active_conv_id), None)
    if active_conv:
        active_scope = active_conv.get("knowledge_scope", "company")
    else:
        # For new conversations without active conversation, read dedicated pending_new_chat_scope
        active_scope = st.session_state.get("pending_new_chat_scope", "company")

    # Fetch attachments for active conversation
    attached_docs: List[Dict[str, Any]] = []
    if active_conv_id:
        try:
            att_res = client.list_conversation_attachments(ws_id, active_conv_id)
            attached_docs = att_res.get("items", [])
        except Exception:
            attached_docs = []

    # =========================================================================
    # 2. TOP APPLICATION BAR
    # =========================================================================
    top_c1, top_c2 = st.columns([3, 2])
    with top_c1:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.35rem 0;">
                <span style="font-size: 1.25rem; font-weight: 800; color: #4F46E5; letter-spacing: -0.02em;">
                    📄 EKN
                </span>
                <span style="font-size: 1.05rem; font-weight: 600; color: #1E293B;">
                    Document Assistant
                </span>
                <span class="ekn-badge ekn-badge-neutral" style="font-size: 0.8rem;">🏢 {ws_name_escaped}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_c2:
        top_btn1, top_btn2 = st.columns([1.2, 1])
        with top_btn1:
            right_open = st.session_state.get("emp_right_open", False)
            btn_lbl = "✕ Close Panel" if right_open else "📋 Context & Sources"
            btn_kind = "primary" if right_open else "secondary"
            if st.button(btn_lbl, type=btn_kind, use_container_width=True, key="emp_top_toggle_panel_btn"):
                st.session_state["emp_right_open"] = not right_open
                if st.session_state["emp_right_open"]:
                    st.session_state["emp_right_view"] = "Context & Sources"
                st.rerun()
        with top_btn2:
            first_name = html.escape(user_name.split()[0] if user_name else "User")
            with st.popover(f"👤 {first_name}"):
                st.write(f"**{html.escape(user_name)}**")
                st.caption(f"{html.escape(user_email)}")
                st.markdown("<span class='ekn-badge ekn-badge-neutral'>Employee</span>", unsafe_allow_html=True)
                st.markdown("<hr style='margin: 0.5rem 0;'/>", unsafe_allow_html=True)
                if st.button("Sign Out", key="emp_top_signout_btn", use_container_width=True, type="secondary"):
                    client.logout()
                    st.query_params.clear()
                    st.session_state.clear()
                    st.session_state["auth_portal"] = "role_selection"
                    st.rerun()

    st.markdown("<hr style='margin: 0.3rem 0 0.8rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

    # =========================================================================
    # 3. THREE-PANEL LAYOUT ROUTER (Approx. Left: 250px, Centre: 760px, Right: 320px)
    # =========================================================================
    left_exp = st.session_state["emp_left_expanded"]
    right_open = st.session_state["emp_right_open"]

    if left_exp and right_open:
        col_left, col_centre, col_right = st.columns([1.1, 2.7, 1.4])
    elif left_exp and not right_open:
        col_left, col_centre = st.columns([1.1, 4.1])
        col_right = None
    elif not left_exp and right_open:
        col_left, col_centre, col_right = st.columns([0.18, 3.62, 1.4])
    else:
        col_left, col_centre = st.columns([0.18, 5.02])
        col_right = None

    # =========================================================================
    # PANEL A: LEFT SIDEBAR (COLLAPSIBLE CHAT HISTORY & FILE NAV)
    # =========================================================================
    with col_left:
        if not left_exp:
            if st.button("💬", key="expand_left_btn", help="Expand Chat History", use_container_width=True):
                st.session_state["emp_left_expanded"] = True
                st.rerun()
        else:
            with st.container(border=True):
                # Header row: New Chat & Collapse button
                n_col1, n_col2 = st.columns([3.5, 1.2])
                with n_col1:
                    if st.button("➕ New Chat", type="primary", use_container_width=True, key="emp_new_chat_btn"):
                        try:
                            # Read dedicated pending_new_chat_scope or active_scope
                            scope_to_use = st.session_state.get("pending_new_chat_scope") or active_scope or "company"
                            new_c = client.create_conversation(ws_id, title="New Conversation", knowledge_scope=scope_to_use)
                            st.session_state["active_conversation_id"] = new_c["id"]
                            st.session_state.pop("pending_new_chat_scope", None)
                            st.rerun()
                        except APIClientError as err:
                            st.error(f"Failed to create chat: {err}")
                with n_col2:
                    if st.button("◀", key="collapse_left_btn", help="Collapse sidebar", use_container_width=True):
                        st.session_state["emp_left_expanded"] = False
                        st.rerun()

                # Conversation search filter
                search_term = st.text_input(
                    "Search chats",
                    placeholder="Search chats...",
                    label_visibility="collapsed",
                    key="emp_chat_filter_term",
                ).strip().lower()

                filtered_convs = [
                    c for c in conversations
                    if not search_term or search_term in (c.get("title") or "").lower()
                ]

                st.markdown("<hr style='margin: 0.4rem 0 0.5rem 0; border: none; border-top: 1px solid #f1f5f9;'/>", unsafe_allow_html=True)

                # Scrollable conversation history container
                with st.container(height=380, border=False):
                    if not filtered_convs:
                        st.caption("No conversations found.")
                    else:
                        for c in filtered_convs:
                            cid = c["id"]
                            ctitle = c.get("title") or "New Conversation"
                            is_active = (cid == active_conv_id)
                            c_date = format_date(c.get("updated_at", ""))

                            c_row1, c_row2 = st.columns([3.8, 1.2])
                            with c_row1:
                                btn_style = "primary" if is_active else "secondary"
                                display_title = f"💬 {ctitle}"
                                if len(display_title) > 24:
                                    display_title = display_title[:22] + "…"
                                if st.button(display_title, key=f"sel_conv_{cid}", use_container_width=True, type=btn_style):
                                    if active_conv_id != cid:
                                        st.session_state["active_conversation_id"] = cid
                                        st.session_state.pop("pending_new_chat_scope", None)
                                        # Synchronize scope selector to target conversation's scope
                                        target_conv = next((conv for conv in conversations if conv.get("id") == cid), None)
                                        if target_conv:
                                            target_scope = target_conv.get("knowledge_scope", "company")
                                            st.session_state["scope_radio_selector"] = target_scope
                                        st.rerun()

                            with c_row2:
                                with st.popover("⋯"):
                                    st.caption(f"**{html.escape(ctitle)}**")
                                    st.caption(f"Last updated: {c_date}")
                                    st.markdown("<hr style='margin: 0.4rem 0;'/>", unsafe_allow_html=True)
                                    # Rename form
                                    with st.form(f"rename_form_{cid}"):
                                        new_t = st.text_input("Rename Title", value=ctitle, key=f"rename_input_{cid}")
                                        save_t = st.form_submit_button("Save Title", use_container_width=True)
                                        if save_t and new_t.strip():
                                            try:
                                                client.update_conversation(ws_id, cid, title=new_t.strip())
                                                st.rerun()
                                            except APIClientError as err:
                                                st.error(str(err))
                                    # Delete action
                                    if st.button("🗑️ Delete Chat", key=f"del_conv_btn_{cid}", type="secondary", use_container_width=True):
                                        try:
                                            client.delete_conversation(ws_id, cid)
                                            if st.session_state.get("active_conversation_id") == cid:
                                                st.session_state["active_conversation_id"] = None
                                                st.session_state.pop("pending_new_chat_scope", None)
                                            st.rerun()
                                        except APIClientError as err:
                                            st.error(str(err))

                st.markdown("<hr style='margin: 0.6rem 0 0.5rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

                # Navigation Links to Files
                st.caption("**Knowledge Files**")
                if st.button(f"🔒 My Files ({len(personal_docs)})", use_container_width=True, key="emp_nav_my_files"):
                    st.session_state["emp_right_open"] = True
                    st.session_state["emp_right_view"] = "My Files"
                    st.rerun()
                if st.button(f"🏢 Company Files ({len(company_docs)})", use_container_width=True, key="emp_nav_company_files"):
                    st.session_state["emp_right_open"] = True
                    st.session_state["emp_right_view"] = "Company Files"
                    st.rerun()

                # Employee Profile Section at Bottom
                st.markdown(
                    f"""
                    <div style="padding: 0.75rem 0.25rem 0.25rem 0.25rem; border-top: 1px solid #e2e8f0; margin-top: 0.8rem;">
                        <div style="font-weight: 600; font-size: 0.90rem; color: #0f172a;">👤 {html.escape(user_name)}</div>
                        <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.35rem;">{html.escape(user_email)}</div>
                        <span class="ekn-badge ekn-badge-neutral" style="font-size: 0.72rem;">Employee</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("🚪 Sign Out", key="emp_sidebar_signout_btn", use_container_width=True, type="secondary"):
                    client.logout()
                    st.query_params.clear()
                    st.session_state.clear()
                    st.session_state["auth_portal"] = "role_selection"
                    st.rerun()

    # =========================================================================
    # PANEL B: CENTRE PANEL (MAIN AI CONVERSATION WITH INDEPENDENT SCROLLING)
    # =========================================================================
    with col_centre:
        # Header Row: Title and Active Knowledge Scope Badge
        title_c1, title_c2 = st.columns([3, 2])
        with title_c1:
            chat_title = active_conv.get("title", "New Conversation") if active_conv else "New Conversation"
            st.markdown(f"### {html.escape(chat_title)}")
        with title_c2:
            scope_badge_map = {
                "company": ("🏢 Company Scope", "ekn-badge-ready"),
                "personal": ("🔒 Personal Scope", "ekn-badge-uploaded"),
                "combined": ("🌐 Combined Scope", "ekn-badge-completed"),
            }
            s_label, s_cls = scope_badge_map.get(active_scope, ("🏢 Company Scope", "ekn-badge-ready"))
            st.markdown(
                f"""
                <div style="text-align: right; padding-top: 0.4rem;">
                    <span class="ekn-badge {s_cls}" style="font-size: 0.85rem; padding: 0.35rem 0.75rem;">
                        {s_label}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Show Attached File Chips if any
        if attached_docs:
            chips_html = " ".join([
                f"<span class='ekn-badge ekn-badge-uploaded' style='margin-right: 0.4rem;'>📎 {html.escape(att.get('original_filename', 'Doc'))}</span>"
                for att in attached_docs
            ])
            st.markdown(f"<div style='margin-bottom: 0.5rem;'>{chips_html}</div>", unsafe_allow_html=True)

        st.markdown("<hr style='margin: 0.2rem 0 0.8rem 0; border: none; border-top: 1px solid #f1f5f9;'/>", unsafe_allow_html=True)

        # Load messages for the active conversation
        messages: List[Dict[str, Any]] = []
        if active_conv_id:
            try:
                msg_resp = client.list_messages(ws_id, active_conv_id)
                messages = msg_resp.get("items", [])
            except APIClientError as err:
                st.error(f"Could not load messages: {err}")

        # Independent Scrollable Container for Conversation History
        with st.container(height=520, border=False):
            # Handle Initial Empty State
            if not messages:
                st.markdown(
                    """
                    <div class="ekn-empty-chat">
                        <div class="ekn-empty-chat-title">How can I help you today?</div>
                        <div class="ekn-empty-chat-subtitle">
                            Ask questions about your company's documents or upload your own personal PDFs.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # 3 Helpful Suggestion Pills
                sug_col1, sug_col2, sug_col3 = st.columns([1, 1, 1])
                with sug_col1:
                    if st.button("🛡️ Key safety procedures", key="sug_btn_1", use_container_width=True):
                        st.session_state["pending_emp_query"] = "What are the key safety procedures described in the manual?"
                        st.rerun()
                with sug_col2:
                    if st.button("📋 Summarize company policies", key="sug_btn_2", use_container_width=True):
                        st.session_state["pending_emp_query"] = "Summarize the main requirements from company policies."
                        st.rerun()
                with sug_col3:
                    if st.button("⚙️ Maintenance requirements", key="sug_btn_3", use_container_width=True):
                        st.session_state["pending_emp_query"] = "What maintenance steps are required for engine overhaul?"
                        st.rerun()

                st.write("")
            else:
                # Render Conversation Message Stream
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    citations = msg.get("citations") or []

                    if role == "user":
                        with st.chat_message("user"):
                            st.write(content)
                    else:
                        with st.chat_message("assistant"):
                            st.markdown(_normalize_markdown(content))
                            if citations:
                                render_citations(client, ws_id, citations)

        # Docked Central Message Composer OUTSIDE the scrollable message stream
        st.markdown("<div style='margin-top: 0.85rem;'></div>", unsafe_allow_html=True)
        pending_query = st.session_state.pop("pending_emp_query", None)
        with st.container(border=True):
            with st.form("emp_central_composer_form", clear_on_submit=True):
                ci_c1, ci_c2 = st.columns([5.5, 1.2])
                with ci_c1:
                    user_text = st.text_input(
                        "Ask a question about your documents...",
                        placeholder="Ask a question about your company or personal documents...",
                        label_visibility="collapsed",
                        key="emp_text_composer_input",
                    )
                with ci_c2:
                    submitted = st.form_submit_button("Send ↗", type="primary", use_container_width=True)

        effective_query = pending_query or (user_text.strip() if submitted and user_text.strip() else None)

        if effective_query:
            clean_q = effective_query.strip()
            if clean_q:
                # Ensure conversation exists
                target_conv_id = active_conv_id
                if not target_conv_id:
                    try:
                        title_candidate = clean_q[:40] + ("..." if len(clean_q) > 40 else "")
                        scope_to_use = st.session_state.get("pending_new_chat_scope") or active_scope or "company"
                        new_c = client.create_conversation(ws_id, title=title_candidate, knowledge_scope=scope_to_use)
                        target_conv_id = new_c["id"]
                        st.session_state["active_conversation_id"] = target_conv_id
                        st.session_state.pop("pending_new_chat_scope", None)
                    except APIClientError as err:
                        st.error(f"Could not initialize conversation: {err}")
                        return

                # Optimistic render of user message
                with st.chat_message("user"):
                    st.write(clean_q)

                # Execute RAG workflow
                with st.chat_message("assistant"):
                    with st.spinner("Searching documents and synthesizing answer..."):
                        try:
                            rag_resp = client.send_message(
                                workspace_id=ws_id,
                                conversation_id=target_conv_id,
                                content=clean_q,
                                timeout=90.0,
                            )
                            resp_type = rag_resp.get("response_type", "valid_answer")
                            if resp_type == "llm_not_configured":
                                st.warning("⚠️ **Gemini API Not Configured**: Set `GEMINI_API_KEY` in `.env` to enable natural-language synthesis.")
                            st.markdown(_normalize_markdown(rag_resp.get("answer", "")))
                            result_citations = rag_resp.get("citations") or []
                            if result_citations:
                                render_citations(client, ws_id, result_citations)
                        except APIClientError as err:
                            st.error(f"Query error: {err}")

                # Rerun to cleanly update persisted message list
                st.rerun()

    # =========================================================================
    # PANEL C: RIGHT PANEL (SEGMENTED: CONTEXT & SOURCES / MY FILES / COMPANY FILES)
    # =========================================================================
    if col_right is not None:
        with col_right:
            with st.container(border=True):
                # Panel Header with Close Button
                rh_c1, rh_c2 = st.columns([3.8, 1.2])
                with rh_c1:
                    st.markdown("#### 📋 Workspace Panel")
                with rh_c2:
                    if st.button("✕", key="close_right_btn", help="Close Panel", use_container_width=True):
                        st.session_state["emp_right_open"] = False
                        st.rerun()

                # View Switcher
                view_options = ["Context & Sources", "My Files", "Company Files"]
                current_view = st.session_state.get("emp_right_view", "Context & Sources")
                if current_view not in view_options:
                    current_view = "Context & Sources"

                selected_view = st.radio(
                    "Select View",
                    options=view_options,
                    index=view_options.index(current_view),
                    horizontal=True,
                    label_visibility="collapsed",
                    key="emp_right_view_selector",
                )
                if selected_view != current_view:
                    st.session_state["emp_right_view"] = selected_view
                    st.rerun()

                st.markdown("<hr style='margin: 0.5rem 0;'/>", unsafe_allow_html=True)

                # Scrollable Content Container for Right Panel Inspector
                with st.container(height=540, border=False):
                    # =============================================================
                    # VIEW 1: CONTEXT & SOURCES
                    # =============================================================
                    if selected_view == "Context & Sources":
                        st.markdown("##### 🎯 Source Scope")
                        scope_options = ["company", "personal", "combined"]
                        scope_labels = {
                            "company": "🏢 Company Documents",
                            "personal": "🔒 My Files (Personal)",
                            "combined": "🌐 Combined (All)",
                        }
                        cur_scope_idx = scope_options.index(active_scope) if active_scope in scope_options else 0
                        chosen_scope = st.radio(
                            "Active Scope",
                            options=scope_options,
                            index=cur_scope_idx,
                            format_func=lambda s: scope_labels.get(s, s),
                            label_visibility="collapsed",
                            key="scope_radio_selector",
                        )
                        if chosen_scope != active_scope:
                            if active_conv_id:
                                try:
                                    client.update_conversation(ws_id, active_conv_id, knowledge_scope=chosen_scope)
                                    st.rerun()
                                except APIClientError as err:
                                    st.error(f"Failed to update scope: {err}")
                            else:
                                # Persist pending scope for brand new conversation
                                st.session_state["pending_new_chat_scope"] = chosen_scope
                                st.rerun()

                        st.markdown("<hr style='margin: 0.5rem 0;'/>", unsafe_allow_html=True)
                        st.markdown(f"**Workspace:** {ws_name_escaped}")

                        # Sources Used for Latest Response (DERIVED STRICTLY FROM CURRENT CONVERSATION)
                        st.markdown("<hr style='margin: 0.5rem 0;'/>", unsafe_allow_html=True)
                        st.markdown("##### 🔍 Sources Used")

                        # Derive citations exclusively from current conversation messages
                        current_chat_citations: List[Dict[str, Any]] = []
                        if messages:
                            for m in reversed(messages):
                                if m.get("role") == "assistant" and m.get("citations"):
                                    current_chat_citations = m.get("citations") or []
                                    break

                        if not current_chat_citations:
                            st.caption("No sources cited for recent response.")
                        else:
                            for c_idx, c_item in enumerate(current_chat_citations):
                                c_tag = html.escape(c_item.get("tag", f"[E{c_idx+1}]"))
                                c_name = html.escape(c_item.get("filename", "Doc"))
                                c_pstart = c_item.get("page_start", 1)
                                c_excerpt = html.escape(c_item.get("excerpt", ""))
                                c_doc_id = c_item.get("document_id")

                                with st.expander(f"{c_tag} {c_name[:18]}… (p. {c_pstart})", expanded=(c_idx == 0)):
                                    st.caption(f"**Document:** {c_name}")
                                    st.caption(f"**Page:** {c_pstart}")
                                    if c_excerpt:
                                        st.markdown(f"<div style='font-size: 0.82rem; font-style: italic; color: #475569;'>\"{c_excerpt[:200]}…\"</div>", unsafe_allow_html=True)

                                    if c_doc_id:
                                        prev_c, dl_c = st.columns([1, 1])
                                        with prev_c:
                                            with st.popover("Preview"):
                                                try:
                                                    p_bytes = client.preview_page(ws_id, c_doc_id, page_number=c_pstart, dpi=150)
                                                    st.image(p_bytes, caption=f"{c_name} — Page {c_pstart}", use_container_width=True)
                                                except Exception as p_err:
                                                    st.error(str(p_err))
                                        with dl_c:
                                            try:
                                                fb, fn = client.download_document(ws_id, c_doc_id)
                                                st.download_button("Download", data=fb, file_name=fn, mime="application/pdf", key=f"rt_dl_{c_doc_id}_{c_idx}")
                                            except Exception:
                                                pass

                        # Attached Documents to Active Conversation
                        st.markdown("<hr style='margin: 0.5rem 0;'/>", unsafe_allow_html=True)
                        st.markdown(f"##### 📎 Attached to Chat ({len(attached_docs)})")
                        if not attached_docs:
                            st.caption("No files attached to this conversation.")
                        else:
                            for att in attached_docs:
                                att_doc_id = att.get("document_id")
                                att_name = html.escape(att.get("original_filename", "Document"))
                                a_col1, a_col2 = st.columns([3.2, 1.8])
                                with a_col1:
                                    st.caption(f"📄 {att_name[:18]}…")
                                Sentiment_detach = False
                                with a_col2:
                                    if st.button("Detach", key=f"r_detach_{att_doc_id}", use_container_width=True):
                                        try:
                                            client.remove_conversation_attachment(ws_id, active_conv_id, att_doc_id)
                                            st.rerun()
                                        except APIClientError as err:
                                            st.error(str(err))

                    # =============================================================
                    # VIEW 2: MY FILES (PERSONAL DOCUMENTS)
                    # =============================================================
                    elif selected_view == "My Files":
                        st.markdown("##### 🔒 Upload Personal PDF")
                        uploaded_files = st.file_uploader(
                            "Upload personal PDF",
                            type=["pdf"],
                            accept_multiple_files=True,
                            label_visibility="collapsed",
                            key="emp_personal_uploader_widget",
                        )
                        if uploaded_files:
                            if st.button("Upload to My Files", type="primary", use_container_width=True, key="emp_submit_personal_upload"):
                                valid_payloads = []
                                has_invalid = False
                                for f in uploaded_files:
                                    data = f.getvalue()
                                    try:
                                        doc = fitz.open(stream=data, filetype="pdf")
                                        if doc.page_count < 1:
                                            st.error(f"'{f.name}' has 0 pages.")
                                            has_invalid = True
                                            continue
                                        doc.close()
                                        valid_payloads.append((f.name, data))
                                    except Exception as err:
                                        st.error(f"'{f.name}' is not a valid PDF: {err}")
                                        has_invalid = True

                                if valid_payloads:
                                    try:
                                        with st.spinner(f"Uploading and indexing {len(valid_payloads)} file(s)..."):
                                            client.upload_personal_documents(ws_id, valid_payloads)
                                        st.success("Personal file(s) uploaded successfully!")
                                        st.rerun()
                                    except APIClientError as err:
                                        st.error(f"Upload failed: {err}")

                        st.markdown("<hr style='margin: 0.5rem 0;'/>", unsafe_allow_html=True)
                        st.markdown(f"##### Your Documents ({len(personal_docs)})")

                        p_filter = st.text_input("Filter personal files", placeholder="Filter...", label_visibility="collapsed", key="emp_p_filter").strip().lower()
                        filtered_p_docs = [
                            d for d in personal_docs
                            if not p_filter or p_filter in (d.get("original_filename") or "").lower()
                        ]

                        if not filtered_p_docs:
                            st.caption("No personal documents found.")
                        else:
                            for pdoc in filtered_p_docs:
                                pid = pdoc["id"]
                                pname = html.escape(pdoc.get("original_filename", "document.pdf"))
                                psize = format_bytes(pdoc.get("file_size_bytes", 0))
                                pstatus = pdoc.get("processing_status", "ready")
                                pbadge = get_status_badge_html(pstatus)

                                with st.expander(f"📄 {pname[:20]}…", expanded=False):
                                    st.caption(f"**Filename:** {pname}")
                                    st.caption(f"**Size:** {psize} | **Status:** {pbadge}", unsafe_allow_html=True)
                                    st.caption(f"**Uploaded:** {format_date(pdoc.get('created_at'))}")

                                    act_c1, act_c2 = st.columns([1, 1])
                                    with act_c1:
                                        with st.popover("Preview"):
                                            try:
                                                p_bytes = client.preview_page(ws_id, pid, page_number=1, dpi=150)
                                                st.image(p_bytes, caption=f"{pname} (Page 1)", use_container_width=True)
                                            except Exception as p_err:
                                                st.error(str(p_err))
                                    with act_c2:
                                        try:
                                            fb, fn = client.download_document(ws_id, pid)
                                            st.download_button("Download", data=fb, file_name=fn, mime="application/pdf", key=f"dl_p_{pid}", use_container_width=True)
                                        except Exception:
                                            pass

                                    act_c3, act_c4 = st.columns([1, 1])
                                    with act_c3:
                                        if active_conv_id:
                                            is_att = any(att.get("document_id") == pid for att in attached_docs)
                                            if is_att:
                                                st.caption("Attached ✓")
                                            else:
                                                if st.button("📎 Attach", key=f"att_p_{pid}", use_container_width=True):
                                                    try:
                                                        client.attach_document_to_conversation(ws_id, active_conv_id, pid)
                                                        st.rerun()
                                                    except APIClientError as err:
                                                        st.error(str(err))
                                    with act_c4:
                                        if st.button("🗑️ Delete", key=f"del_p_{pid}", type="secondary", use_container_width=True):
                                            try:
                                                client.delete_document(ws_id, pid)
                                                st.rerun()
                                            except APIClientError as err:
                                                st.error(str(err))

                    # =============================================================
                    # VIEW 3: COMPANY FILES (SHARED READ-ONLY)
                    # =============================================================
                    elif selected_view == "Company Files":
                        st.markdown(f"##### 🏢 Shared Company Files ({len(company_docs)})")
                        st.caption("These documents are managed by your administrator and shared across your company.")

                        c_filter = st.text_input("Filter company files", placeholder="Filter...", label_visibility="collapsed", key="emp_c_filter").strip().lower()
                        filtered_c_docs = [
                            d for d in company_docs
                            if not c_filter or c_filter in (d.get("original_filename") or "").lower()
                        ]

                        if not filtered_c_docs:
                            st.caption("No company documents found.")
                        else:
                            for cdoc in filtered_c_docs:
                                cid = cdoc["id"]
                                cname = html.escape(cdoc.get("original_filename", "document.pdf"))
                                csize = format_bytes(cdoc.get("file_size_bytes", 0))
                                cstatus = cdoc.get("processing_status", "ready")
                                cbadge = get_status_badge_html(cstatus)

                                with st.expander(f"🏢 {cname[:20]}…", expanded=False):
                                    st.caption(f"**Filename:** {cname}")
                                    st.caption(f"**Size:** {csize} | **Status:** {cbadge}", unsafe_allow_html=True)
                                    st.caption(f"**Added:** {format_date(cdoc.get('created_at'))}")

                                    c_act1, c_act2 = st.columns([1, 1])
                                    with c_act1:
                                        with st.popover("Preview"):
                                            try:
                                                p_bytes = client.preview_page(ws_id, cid, page_number=1, dpi=150)
                                                st.image(p_bytes, caption=f"{cname} (Page 1)", use_container_width=True)
                                            except Exception as p_err:
                                                st.error(str(p_err))
                                    with c_act2:
                                        try:
                                            fb, fn = client.download_document(ws_id, cid)
                                            st.download_button("Download", data=fb, file_name=fn, mime="application/pdf", key=f"dl_c_{cid}", use_container_width=True)
                                        except Exception:
                                            pass

                                    if active_conv_id:
                                        is_att = any(att.get("document_id") == cid for att in attached_docs)
                                        if is_att:
                                            st.caption("Attached ✓")
                                        else:
                                            if st.button("📎 Attach to Chat", key=f"att_c_{cid}", use_container_width=True):
                                                try:
                                                    client.attach_document_to_conversation(ws_id, active_conv_id, cid)
                                                    st.rerun()
                                                except APIClientError as err:
                                                    st.error(str(err))
