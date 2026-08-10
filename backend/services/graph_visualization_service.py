"""Graph Visualization Service for Enterprise Knowledge Navigator.

Renders NetworkX graphs into clean, restrained, publication-quality PNG diagrams
using Matplotlib (Agg backend) with distinguishable node shapes and edge styles.
"""

import io
import logging
from typing import Optional
import matplotlib
# Use non-interactive Agg backend for safe server rendering
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

logger = logging.getLogger("ekn.graph.visualization")


class GraphVisualizationService:
    """Renders NetworkX knowledge graphs to PNG bytes."""

    def render_graph_to_png(
        self,
        G: nx.Graph,
        title: str = "Enterprise Knowledge Graph",
        figsize: tuple = (10, 6.5),
        dpi: int = 150,
    ) -> bytes:
        """Render NetworkX graph to PNG image bytes."""
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Handle empty graph
        if not G or G.number_of_nodes() == 0:
            ax.text(
                0.5,
                0.5,
                "No graph nodes or relationships match current filters.\nSelect a document or trigger graph discovery.",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=11,
                color="#64748b",
                transform=ax.transAxes,
            )
            ax.axis("off")
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            return buf.getvalue()

        # Compute layout
        try:
            pos = nx.spring_layout(G, k=0.6, iterations=50, seed=42)
        except Exception:
            pos = nx.circular_layout(G)

        # Partition nodes by type
        doc_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "document"]
        ent_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "entity"]

        # Draw Document Nodes
        if doc_nodes:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=doc_nodes,
                node_color="#2563eb",
                node_size=750,
                alpha=0.9,
                node_shape="o",
                ax=ax,
            )

        # Draw Entity Nodes
        if ent_nodes:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=ent_nodes,
                node_color="#059669",
                node_size=420,
                alpha=0.85,
                node_shape="s",
                ax=ax,
            )

        # Partition edges by relationship type
        shared_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("relationship_type") == "shared_entity"
        ]
        direct_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("relationship_type") == "direct_reference"
        ]
        topic_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("relationship_type") == "related_topic"
        ]
        mention_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("relationship_type") == "mentions"
        ]

        if shared_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=shared_edges,
                edge_color="#3b82f6",
                width=2.0,
                alpha=0.8,
                ax=ax,
            )

        if direct_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=direct_edges,
                edge_color="#8b5cf6",
                width=2.5,
                alpha=0.9,
                ax=ax,
            )

        if topic_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=topic_edges,
                edge_color="#94a3b8",
                width=1.4,
                style="dashed",
                alpha=0.7,
                ax=ax,
            )

        if mention_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=mention_edges,
                edge_color="#cbd5e1",
                width=1.0,
                style="dotted",
                alpha=0.6,
                ax=ax,
            )

        # Draw labels cleanly with truncation
        labels = {}
        for n, d in G.nodes(data=True):
            raw_lbl = d.get("label", n)
            if len(raw_lbl) > 16:
                labels[n] = raw_lbl[:14] + "…"
            else:
                labels[n] = raw_lbl

        nx.draw_networkx_labels(
            G,
            pos,
            labels=labels,
            font_size=8,
            font_family="sans-serif",
            font_color="#0f172a",
            ax=ax,
        )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=12, color="#0f172a")
        ax.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return buf.getvalue()


graph_visualization_service = GraphVisualizationService()
