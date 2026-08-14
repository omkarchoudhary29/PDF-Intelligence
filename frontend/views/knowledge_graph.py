"""Knowledge Graph View for Enterprise Knowledge Navigator.

Provides interactive cross-document relationship discovery, NetworkX visual rendering,
entity catalog with aliases and attributes, multi-document evidence provenance,
shortest-path traversal, and direct PDF page preview navigation.
"""

from typing import Any, Dict, List, Optional
import streamlit as st

from frontend.api_client import APIClientError, BackendAPIClient
from frontend.styles import format_bytes, get_status_badge_html


def render_knowledge_graph(client: BackendAPIClient, workspace: Dict[str, Any]) -> None:
    """Render the Knowledge Graph view."""
    ws_id = workspace["id"]
    ws_name = workspace["name"]

    # Header and Actions
    col_hdr, col_btn1, col_btn2 = st.columns([3, 1.2, 1.2])
    with col_hdr:
        st.markdown(f"## Knowledge Graph — {ws_name}")
        st.caption("Evidence-backed entity connections, cross-document relationships, and verifiable page provenance.")
    with col_btn1:
        if st.button("Run Full Discovery", type="primary", use_container_width=True, help="Extract entities and discover all relationships"):
            with st.spinner("Extracting entities and discovering relationships across company documents..."):
                try:
                    res = client.build_graph(ws_id, force=True)
                    st.success(
                        f"Discovery complete: {res.get('documents_processed', 0)} docs processed, "
                        f"{res.get('entities_extracted', 0)} entities extracted, "
                        f"{res.get('relationships_discovered', 0)} relationships discovered!"
                    )
                    st.rerun()
                except APIClientError as err:
                    st.error(f"Graph discovery failed: {err}")
    with col_btn2:
        if st.button("Backfill Graph", type="secondary", use_container_width=True, help="Extract graph on unindexed company documents without re-embedding"):
            with st.spinner("Backfilling entity extraction for company documents..."):
                try:
                    res = client.backfill_graph(ws_id)
                    st.success(f"Backfill complete: {res.get('entities_extracted', 0)} entities extracted!")
                    st.rerun()
                except APIClientError as err:
                    st.error(f"Backfill failed: {err}")

    st.markdown("<hr style='margin: 0.8rem 0 1.25rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

    # 1. High-Level Summary Statistics
    try:
        summary = client.get_graph_summary(ws_id)
        stat_cols = st.columns(4)

        with stat_cols[0]:
            st.markdown(
                f"""
                <div class="ekn-stat-card">
                    <div class="ekn-stat-value" style="color: #2563eb;">{summary['total_documents']}</div>
                    <div class="ekn-stat-label">Document Nodes</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with stat_cols[1]:
            st.markdown(
                f"""
                <div class="ekn-stat-card">
                    <div class="ekn-stat-value" style="color: #059669;">{summary['total_entities']}</div>
                    <div class="ekn-stat-label">Confirmed Entities</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with stat_cols[2]:
            st.markdown(
                f"""
                <div class="ekn-stat-card">
                    <div class="ekn-stat-value" style="color: #7c3aed;">{summary['total_relationships']}</div>
                    <div class="ekn-stat-label">Discovered Edges</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with stat_cols[3]:
            top_doc = summary.get("top_connected_documents", [])
            top_name = top_doc[0]["document_name"] if top_doc else "None"
            if len(top_name) > 20:
                top_name = top_name[:18] + "…"
            st.markdown(
                f"""
                <div class="ekn-stat-card">
                    <div class="ekn-stat-value" style="font-size: 1.15rem; margin-top: 0.35rem;">{top_name}</div>
                    <div class="ekn-stat-label">Most Connected Document</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    except APIClientError as err:
        st.warning(f"Could not load graph summary: {err}")
        return

    # 2. Empty State Handling
    if summary["total_documents"] == 0 or (summary["total_entities"] == 0 and summary["total_relationships"] == 0):
        st.markdown(
            """
            <div class="ekn-empty-state">
                <div class="ekn-empty-title">No Knowledge Graph Data Available</div>
                <div class="ekn-empty-desc">
                    Your workspace contains documents, but knowledge graph discovery has not been executed yet.
                    Click 'Run Full Discovery' above to identify domain entities, multi-document provenance, and topic connections.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown("<hr style='margin: 1rem 0 1.25rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

    # Fetch document list for selectors
    doc_options: Dict[str, str] = {"": "All Documents (Entire Workspace)"}
    try:
        docs_resp = client.list_documents(ws_id, page=1, page_size=100)
        for d in docs_resp.get("items", []):
            if d.get("visibility_scope") != "personal":
                doc_options[d["id"]] = d["original_filename"]
    except Exception:
        pass

    # Navigation Tabs
    tab_viz, tab_entities, tab_rels, tab_path, tab_recon = st.tabs([
        "🌐 Network Topology",
        "🏷️ Entity Catalog & Inspector",
        "🔗 Provenanced Relationships",
        "🧭 Graph Traversal & Path",
        "📊 Data Reconciliation",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Network Topology
    # -------------------------------------------------------------------------
    with tab_viz:
        st.markdown("##### Visual Relationship Network")

        c_doc, c_rel, c_limit = st.columns([2.5, 1.8, 1.2])
        with c_doc:
            selected_doc_id = st.selectbox(
                "Focus Document",
                options=list(doc_options.keys()),
                format_func=lambda k: doc_options.get(k, k),
                index=0,
                key="graph_focus_doc",
            )
        with c_rel:
            rel_type_filter = st.selectbox(
                "Relationship Filter",
                options=["All Types", "PART_OF", "CONNECTED_TO", "OPERATES", "REGULATED_BY", "shared_entity", "direct_reference", "related_topic"],
                index=0,
                key="graph_rel_type",
            )
        with c_limit:
            max_nodes = st.slider("Max Nodes", min_value=10, max_value=80, value=35, step=5, key="graph_max_nodes")

        api_rel_type = None if rel_type_filter == "All Types" else rel_type_filter
        focus_id = selected_doc_id if selected_doc_id else None

        with st.spinner("Rendering knowledge graph visualization..."):
            try:
                png_bytes = client.render_graph_image(
                    workspace_id=ws_id,
                    limit=max_nodes,
                    relationship_type=api_rel_type,
                    focus_document_id=focus_id,
                )
                st.image(png_bytes, use_container_width=True)

                st.caption(
                    "🔵 **Blue Circles**: Documents · 🟢 **Green Squares**: Entities | "
                    "**Solid Blue**: Shared Entity · **Solid Purple**: Direct Citation · **Solid Green**: Entity Relationship · **Dashed Gray**: Topical Similarity"
                )
            except APIClientError as err:
                st.error(f"Could not render visual graph: {err}")

    # -------------------------------------------------------------------------
    # TAB 2: Entity Catalog & Inspector
    # -------------------------------------------------------------------------
    with tab_entities:
        st.markdown("##### Domain Entity Catalog")
        st.caption("Normalized entities extracted from company documents with type categorization, aliases, and direct relations.")

        col_q, col_type = st.columns([2, 1.5])
        with col_q:
            search_query = st.text_input("Search Entities", placeholder="Filter by entity name or alias...", key="ent_search_q")
        with col_type:
            entity_type_opts = [
                "All Types", "Equipment", "Machine", "Component", "System", "Material",
                "Standard", "Regulation", "Process", "Maintenance procedure", "Measurement", "Safety hazard", "Other"
            ]
            selected_ent_type = st.selectbox("Entity Type Filter", options=entity_type_opts, key="ent_type_filter")

        ent_filter = None if selected_ent_type == "All Types" else selected_ent_type

        try:
            ent_resp = client.get_graph_entities(
                workspace_id=ws_id,
                q=search_query if search_query.strip() else None,
                entity_type=ent_filter,
                page=1,
                page_size=50,
            )
            entities_list = ent_resp.get("items", [])
            total_ents = ent_resp.get("total", 0)

            st.write(f"Showing **{len(entities_list)}** of **{total_ents}** entities")

            if not entities_list:
                st.info("No entities match the current query and filters.")
            else:
                for ent in entities_list:
                    ent_id = ent["id"]
                    canonical = ent["canonical_name"]
                    etype = ent["entity_type"]
                    mentions = ent.get("mention_count", 1)
                    aliases = ent.get("aliases_json") or []
                    alias_str = ", ".join(aliases) if aliases else "None"
                    desc = ent.get("description") or "Domain entity extracted from company knowledge base."

                    with st.expander(f"{canonical} ({etype}) — {mentions} mentions"):
                        c_left, c_right = st.columns([2, 1])
                        with c_left:
                            st.markdown(f"**Canonical Name:** `{canonical}`")
                            st.markdown(f"**Entity Type:** `{etype}`")
                            st.markdown(f"**Aliases:** {alias_str}")
                            st.markdown(f"**Description:** {desc}")
                        with c_right:
                            st.metric("Document Mentions", mentions)

                        # Inspect relationships for this entity
                        try:
                            ent_rels = client.get_entity_relationships(ws_id, ent_id)
                            if ent_rels:
                                st.markdown("###### Direct Entity Relationships & Provenance:")
                                for r in ent_rels:
                                    r_type = r["relationship_type"]
                                    src_name = r.get("source_entity_name", "Source")
                                    tgt_name = r.get("target_entity_name", "Target")
                                    doc_name = r.get("source_document_name", "Document")
                                    doc_id = r.get("source_document_id")
                                    p_num = r.get("page_number", 1)
                                    excerpt = r.get("evidence_excerpt") or ""
                                    conf = r.get("confidence_score", 1.0)

                                    st.markdown(
                                        f"""
                                        <div style="background: #f8fafc; border-left: 3px solid #10b981; padding: 0.5rem 0.75rem; margin: 0.35rem 0; font-size: 0.88rem;">
                                            <strong>{src_name}</strong> &rarr; <code>{r_type}</code> &rarr; <strong>{tgt_name}</strong>
                                            (Confidence: {conf:.2f})<br/>
                                            <span style="color: #64748b;">Source: {doc_name} (Page {p_num})</span><br/>
                                            <em>"{excerpt}"</em>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                                    if doc_id and p_num:
                                        with st.popover(f"Preview {doc_name} (p. {p_num})", key=f"prev_ent_{r['id']}"):
                                            try:
                                                prev_bytes = client.preview_page(ws_id, doc_id, p_num)
                                                st.image(prev_bytes, caption=f"{doc_name} — Page {p_num}", use_container_width=True)
                                            except Exception as prev_err:
                                                st.error(f"Preview unavailable: {prev_err}")
                            else:
                                st.caption("No direct entity-to-entity relationships recorded for this entity.")
                        except Exception as e_rel_err:
                            st.caption(f"Could not load entity relationships: {e_rel_err}")

        except APIClientError as err:
            st.error(f"Error loading entity catalog: {err}")

    # -------------------------------------------------------------------------
    # TAB 3: Provenanced Relationships
    # -------------------------------------------------------------------------
    with tab_rels:
        st.markdown("##### Verified Relationships & Multi-Document Evidence")
        st.caption("Verifiable relationships extracted from company PDFs. Each relationship contains page-level citations and supporting excerpts.")

        try:
            graph_data = client.get_graph_data(
                workspace_id=ws_id,
                limit=max_nodes,
                relationship_type=api_rel_type,
                focus_document_id=focus_id,
            )

            edges = graph_data.get("edges", [])
            verified_edges = [e for e in edges if e.get("relationship_type") != "mentions"]

            if not verified_edges:
                st.info("No cross-document or entity relationships match the current filters.")
            else:
                st.caption(f"Showing **{len(verified_edges)}** verified relationships.")

                for idx, edge in enumerate(verified_edges):
                    src_node = edge["source"]
                    tgt_node = edge["target"]
                    rel_type = edge["relationship_type"]
                    explanation = edge.get("explanation", "Relationship identified from document analysis.")
                    confidence = edge.get("confidence", 1.0)
                    evidence = edge.get("supporting_evidence") or []
                    ev_count = edge.get("properties", {}).get("evidence_count", len(evidence) if isinstance(evidence, list) else 1)

                    # Determine readable names
                    src_clean = src_node.replace("doc_", "").replace("ent_", "")
                    tgt_clean = tgt_node.replace("doc_", "").replace("ent_", "")

                    src_name = doc_options.get(src_clean, f"Node {src_clean[:8]}")
                    tgt_name = doc_options.get(tgt_clean, f"Node {tgt_clean[:8]}")

                    title_str = f"{src_name} ↔ {tgt_name} ({rel_type})"
                    if ev_count > 1:
                        title_str += f" — [{ev_count} independent proofs]"

                    with st.expander(title_str):
                        st.markdown(f"**Relationship Type:** `{rel_type}` | **Confidence:** `{confidence:.2f}`")
                        st.write(f"**Explanation:** {explanation}")

                        if evidence and isinstance(evidence, list):
                            st.markdown(f"**Supporting Evidence ({len(evidence)} document citations):**")
                            for ev_idx, ev in enumerate(evidence):
                                if isinstance(ev, dict):
                                    ev_doc = ev.get("document_name") or ev.get("filename") or ev.get("source_filename") or "Document"
                                    ev_page = ev.get("page_number") or ev.get("page")
                                    ev_txt = ev.get("excerpt") or ev.get("note", "")
                                    ev_doc_id = ev.get("document_id") or ev.get("source_document_id")

                                    page_str = f"Page {ev_page}" if ev_page else "General"
                                    st.markdown(
                                        f"""
                                        <div style="background: #f8fafc; border-left: 3px solid #3b82f6; padding: 0.5rem 0.75rem; margin: 0.35rem 0; font-size: 0.88rem;">
                                            <strong>{ev_doc}</strong> ({page_str}): <em>"{ev_txt}"</em>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                                    if ev_doc_id and ev_page:
                                        with st.popover(f"Preview {ev_doc} ({page_str})", key=f"prev_edge_{idx}_{ev_idx}"):
                                            try:
                                                prev_png = client.preview_page(ws_id, ev_doc_id, ev_page)
                                                st.image(prev_png, caption=f"{ev_doc} — {page_str}", use_container_width=True)
                                            except Exception as prev_err:
                                                st.error(f"Preview error: {prev_err}")

        except APIClientError as err:
            st.error(f"Error loading relationship details: {err}")

    # -------------------------------------------------------------------------
    # TAB 4: Graph Traversal & Path
    # -------------------------------------------------------------------------
    with tab_path:
        st.markdown("##### Shortest Path & Traversal Finder")
        st.caption("Traverse the company knowledge graph to discover multi-hop connections between any two technical concepts, components, or documents.")

        col_src, col_tgt, col_go = st.columns([2, 2, 1])
        with col_src:
            source_input = st.text_input("Source Concept or Document ID", placeholder="e.g. AX-450 or entity ID", key="path_source_in")
        with col_tgt:
            target_input = st.text_input("Target Concept or Document ID", placeholder="e.g. ISO-9001 or entity ID", key="path_target_in")
        with col_go:
            st.write("")
            st.write("")
            find_btn = st.button("Find Path", type="primary", use_container_width=True)

        if find_btn and source_input.strip() and target_input.strip():
            with st.spinner("Traversing knowledge graph..."):
                try:
                    path_result = client.find_graph_path(ws_id, source_input.strip(), target_input.strip())
                    if path_result.get("found"):
                        st.success(f"Path found! Length: {path_result.get('path_length', 0)} hops.")
                        steps = path_result.get("steps", [])
                        for s in steps:
                            step_num = s.get("step")
                            label = s.get("label")
                            ntype = s.get("node_type")
                            edge_info = s.get("edge")

                            if edge_info:
                                r_type = edge_info.get("relationship_type")
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;⬇️ *via* `[{r_type}]`")

                            st.markdown(
                                f"""
                                <div style="background: #f1f5f9; padding: 0.4rem 0.8rem; border-radius: 4px; font-weight: 500;">
                                    Step {step_num}: <strong>{label}</strong> <span style="color: #64748b; font-size: 0.82rem;">({ntype})</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.warning(path_result.get("message") or path_result.get("error") or "No connecting path exists between the specified nodes.")
                except APIClientError as path_err:
                    st.error(f"Traversal failed: {path_err}")

    # -------------------------------------------------------------------------
    # TAB 5: Data Reconciliation
    # -------------------------------------------------------------------------
    with tab_recon:
        st.markdown("##### Vector & Document Count Reconciliation")
        st.caption("Verifiable reconciliation between relational database documents and ChromaDB vector embeddings.")

        try:
            recon = client.get_graph_reconciliation(ws_id)
            r_cols = st.columns(3)
            with r_cols[0]:
                st.metric("Workspace Vectors", recon.get("workspace_vector_count", 0))
            with r_cols[1]:
                st.metric("Company Documents", recon.get("company_documents_count", 0))
            with r_cols[2]:
                st.metric("Collection Total Vectors", recon.get("collection_wide_vector_count", 0))

            st.info(recon.get("reconciliation_explanation", "Reconciliation verified."))
            st.markdown(
                """
                > [!NOTE]
                > **ChromaDB Count Architecture**:
                > - **Workspace Scoped**: Queries filtering by `workspace_id` retrieve exactly the vectors indexed for this workspace's company PDFs.
                > - **Collection Wide**: Total vectors in ChromaDB include vectors indexed across independent tenant workspaces and automated phase tests.
                > - **Data Safety**: No vectors are modified, re-embedded, or deleted to reconcile numbers.
                """
            )
        except APIClientError as recon_err:
            st.warning(f"Could not load reconciliation metrics: {recon_err}")
