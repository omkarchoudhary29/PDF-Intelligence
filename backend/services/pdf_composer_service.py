"""ReportLab PDF Composer Service for Enterprise Knowledge Navigator.

Composes high-quality, professional, navigable Enterprise Knowledge Dossier PDFs
with cover page, running headers/footers, master document index, individual summaries,
embedded knowledge graph visualization, internal bookmarks, and clickable source hyperlinks.
"""

from datetime import datetime
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import fitz  # PyMuPDF for validation

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.schemas.report import DocumentSummaryData, ExecutiveSummaryData
from config.settings import get_settings

logger = logging.getLogger("ekn.reports.composer")


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas that computes total pages dynamically and stamps running header/footer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Any] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count: int) -> None:
        # Do not draw headers and footers on cover page (page 1)
        if self._pageNumber == 1:
            return

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Running Header
        self.drawString(54, letter[1] - 36, "Enterprise Knowledge Navigator — Knowledge Dossier")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)

        # Running Footer
        self.line(54, 45, letter[0] - 54, 45)
        self.drawString(54, 32, "Confidential — For Internal Enterprise Use Only")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 32, page_text)
        self.restoreState()


class PDFComposerService:
    """Constructs publication-quality Enterprise Knowledge Dossier PDFs using ReportLab."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def compose_dossier_pdf(
        self,
        output_path: Path,
        workspace_name: str,
        workspace_id: str,
        executive_summary: ExecutiveSummaryData,
        document_summaries: List[DocumentSummaryData],
        relationships: List[Dict[str, Any]],
        graph_png_bytes: Optional[bytes] = None,
        report_title: str = "Enterprise Knowledge Dossier",
    ) -> int:
        """Compose complete multi-page PDF document and return total page count."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=54,
            bottomMargin=54,
        )

        app_url = self.settings.app_base_url.rstrip("/")

        # Styles
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "CoverTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=34,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=8,
        )
        subtitle_style = ParagraphStyle(
            "CoverSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#475569"),
            spaceAfter=20,
        )
        h1_style = ParagraphStyle(
            "Heading1_Custom",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=14,
            spaceAfter=8,
            keepWithNext=True,
        )
        h2_style = ParagraphStyle(
            "Heading2_Custom",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=10,
            spaceAfter=4,
            keepWithNext=True,
        )
        body_style = ParagraphStyle(
            "Body_Custom",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#334155"),
            spaceAfter=6,
        )
        body_bold = ParagraphStyle(
            "Body_Bold",
            parent=body_style,
            fontName="Helvetica-Bold",
        )
        table_cell = ParagraphStyle(
            "TableCell",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#1e293b"),
        )
        table_cell_bold = ParagraphStyle(
            "TableCellBold",
            parent=table_cell,
            fontName="Helvetica-Bold",
        )
        link_style = ParagraphStyle(
            "LinkStyle",
            parent=body_style,
            textColor=colors.HexColor("#2563eb"),
        )

        story: List[Any] = []

        # =====================================================================
        # 1. COVER PAGE
        # =====================================================================
        story.append(Spacer(1, 1.2 * inch))
        story.append(Paragraph("Enterprise Knowledge Navigator", ParagraphStyle("Brand", fontName="Helvetica-Bold", fontSize=12, textColor=colors.HexColor("#2563eb"), spaceAfter=6)))
        story.append(Paragraph(report_title, title_style))
        story.append(Paragraph(f"Workspace: <b>{workspace_name}</b>", subtitle_style))
        story.append(Paragraph("Consolidated Repository Intelligence &middot; Master Index &middot; Knowledge Graph &middot; Provenance", body_style))
        story.append(Spacer(1, 0.4 * inch))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=25))

        # Metadata Box Table
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        meta_data = [
            [Paragraph("Generated On", table_cell_bold), Paragraph(gen_time, table_cell)],
            [Paragraph("Workspace ID", table_cell_bold), Paragraph(f"<code>{workspace_id}</code>", table_cell)],
            [Paragraph("Total Documents", table_cell_bold), Paragraph(str(executive_summary.document_count), table_cell)],
            [Paragraph("Total Pages Indexed", table_cell_bold), Paragraph(str(executive_summary.total_pages), table_cell)],
            [Paragraph("Application Base URL", table_cell_bold), Paragraph(f'<a href="{app_url}">{app_url}</a>', table_cell)],
        ]
        meta_table = Table(meta_data, colWidths=[2.0 * inch, 4.5 * inch])
        meta_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(meta_table)
        story.append(Spacer(1, 1.5 * inch))
        story.append(Paragraph("<i>This automated dossier provides evidence-backed summaries and cross-document relational mappings. Every citation and link connects directly to original source PDF pages.</i>", ParagraphStyle("Disclaimer", fontName="Helvetica-Oblique", fontSize=8.5, textColor=colors.HexColor("#64748b"))))
        story.append(PageBreak())

        # =====================================================================
        # 2. EXECUTIVE REPOSITORY OVERVIEW
        # =====================================================================
        story.append(Paragraph("1. Executive Repository Overview", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1"), spaceAfter=10))
        story.append(Paragraph(executive_summary.overview_text, body_style))
        story.append(Spacer(1, 8))

        if executive_summary.key_themes:
            story.append(Paragraph("<b>Primary Knowledge Themes:</b>", body_style))
            for theme in executive_summary.key_themes:
                story.append(Paragraph(f"&bull; {theme}", body_style))
            story.append(Spacer(1, 8))

        if executive_summary.documentation_gaps:
            story.append(Paragraph("<b>Observed Documentation Gaps & Recommendations:</b>", body_style))
            for gap in executive_summary.documentation_gaps:
                story.append(Paragraph(f"&bull; {gap}", body_style))

        story.append(Spacer(1, 15))
        story.append(PageBreak())

        # =====================================================================
        # 3. MASTER DOCUMENT INDEX
        # =====================================================================
        story.append(Paragraph("2. Master Document Index", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1"), spaceAfter=10))
        story.append(Paragraph("Every document indexed in this workspace is recorded below with page counts and direct application navigation links.", body_style))
        story.append(Spacer(1, 8))

        index_headers = [
            Paragraph("#", table_cell_bold),
            Paragraph("Document Name", table_cell_bold),
            Paragraph("Pages", table_cell_bold),
            Paragraph("Purpose Summary", table_cell_bold),
            Paragraph("Action", table_cell_bold),
        ]
        index_rows = [index_headers]

        for i, doc_item in enumerate(document_summaries, start=1):
            doc_link = f"{app_url}/?document={doc_item.document_id}&page=1"
            name_cell = Paragraph(f'<b><a href="{doc_link}">{doc_item.original_filename}</a></b>', table_cell)
            purpose_cell = Paragraph(doc_item.purpose, table_cell)
            action_cell = Paragraph(f'<a href="{doc_link}">Open PDF &rarr;</a>', table_cell)

            index_rows.append([
                Paragraph(str(i), table_cell),
                name_cell,
                Paragraph(str(doc_item.total_pages), table_cell),
                purpose_cell,
                action_cell,
            ])

        index_table = Table(index_rows, colWidths=[0.35 * inch, 2.3 * inch, 0.6 * inch, 2.8 * inch, 0.95 * inch])
        index_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ])
        )
        story.append(index_table)
        story.append(PageBreak())

        # =====================================================================
        # 4. INDIVIDUAL DOCUMENT SUMMARIES
        # =====================================================================
        story.append(Paragraph("3. Individual Document Summaries", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1"), spaceAfter=12))

        for idx, doc_item in enumerate(document_summaries, start=1):
            doc_link = f"{app_url}/?document={doc_item.document_id}&page=1"
            summary_blocks = []

            summary_blocks.append(Paragraph(f"3.{idx} {doc_item.original_filename}", h2_style))
            summary_blocks.append(
                Paragraph(
                    f"<b>Total Pages:</b> {doc_item.total_pages} &nbsp;|&nbsp; "
                    f'<b>Direct Link:</b> <a href="{doc_link}">Open in Navigator</a>',
                    body_style,
                )
            )
            summary_blocks.append(Paragraph(f"<b>Purpose:</b> {doc_item.purpose}", body_style))
            summary_blocks.append(Paragraph(doc_item.summary_text, body_style))

            if doc_item.important_findings:
                findings_str = "; ".join(doc_item.important_findings)
                summary_blocks.append(Paragraph(f"<b>Key Findings:</b> {findings_str}", body_style))

            if doc_item.key_sections:
                sec_str = ", ".join(doc_item.key_sections)
                summary_blocks.append(Paragraph(f"<b>Key Sections:</b> {sec_str}", body_style))

            if doc_item.page_references:
                page_links = [
                    f'<a href="{app_url}/?document={doc_item.document_id}&page={p}">Page {p}</a>'
                    for p in sorted(set(doc_item.page_references))
                ]
                summary_blocks.append(Paragraph(f"<b>Key Page Coordinates:</b> {', '.join(page_links)}", body_style))

            if doc_item.related_documents:
                summary_blocks.append(Paragraph(f"<b>Related Documents:</b> {', '.join(doc_item.related_documents)}", body_style))

            summary_blocks.append(Spacer(1, 10))
            summary_blocks.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=12))
            story.append(KeepTogether(summary_blocks))

        story.append(PageBreak())

        # =====================================================================
        # 5. CROSS-DOCUMENT INTELLIGENCE & RELATIONSHIPS
        # =====================================================================
        story.append(Paragraph("4. Cross-Document Intelligence & Findings", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1"), spaceAfter=10))
        story.append(
            Paragraph(
                "Cross-document relationships discovered through shared entity mentions, direct citations, "
                "and semantic topic overlap are synthesized below with verbatim evidence citations.",
                body_style,
            )
        )
        story.append(Spacer(1, 8))

        if not relationships:
            story.append(Paragraph("<i>No cross-document relationships have been identified yet.</i>", body_style))
        else:
            rel_headers = [
                Paragraph("Type", table_cell_bold),
                Paragraph("Source & Target", table_cell_bold),
                Paragraph("Confidence", table_cell_bold),
                Paragraph("Relationship Explanation & Evidence", table_cell_bold),
            ]
            rel_rows = [rel_headers]

            for r in relationships[:25]:  # Bounded table
                rel_type = r.get("relationship_type", "shared_entity")
                src_name = r.get("source_document_name") or f"Doc {r.get('source_document_id', '')[:8]}"
                tgt_name = r.get("target_document_name") or f"Doc {r.get('target_document_id', '')[:8]}"
                conf = f"{r.get('confidence_score', 1.0):.2f}"
                expl = r.get("explanation", "")

                rel_rows.append([
                    Paragraph(f"<code>{rel_type}</code>", table_cell),
                    Paragraph(f"{src_name}<br/>&harr; {tgt_name}", table_cell),
                    Paragraph(conf, table_cell),
                    Paragraph(expl, table_cell),
                ])

            rel_table = Table(rel_rows, colWidths=[1.1 * inch, 1.8 * inch, 0.7 * inch, 3.4 * inch])
            rel_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("PADDING", (0, 0), (-1, -1), 5),
                ])
            )
            story.append(rel_table)

        story.append(Spacer(1, 15))
        story.append(PageBreak())

        # =====================================================================
        # 6. KNOWLEDGE GRAPH VISUALIZATION
        # =====================================================================
        story.append(Paragraph("5. Knowledge Graph Network", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1"), spaceAfter=10))

        if graph_png_bytes:
            graph_stream = io.BytesIO(graph_png_bytes)
            # Embed image scaled to fit letter width
            graph_img = Image(graph_stream, width=6.5 * inch, height=4.2 * inch)
            story.append(graph_img)
            story.append(Spacer(1, 6))
            story.append(
                Paragraph(
                    "<b>Figure 1:</b> Knowledge Graph Network &middot; Blue Circles: Documents &middot; Green Squares: Entities &middot; "
                    "Solid Blue: Shared Entities &middot; Solid Purple: Direct References &middot; Dashed: Semantic Similarity",
                    ParagraphStyle("FigCaption", fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#64748b")),
                )
            )
        else:
            story.append(Paragraph("<i>Knowledge graph visualization was omitted or unavailable.</i>", body_style))

        story.append(Spacer(1, 20))

        # =====================================================================
        # 7. GLOSSARY & REFERENCES
        # =====================================================================
        story.append(Paragraph("6. Technical Methodology & Provenance", h1_style))
        story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1"), spaceAfter=10))
        story.append(
            Paragraph(
                "This dossier was generated autonomously by the Enterprise Knowledge Navigator. "
                "Document summaries were extracted hierarchically with strict ground truth preservation. "
                "All hyperlinks route to authenticated application endpoints. "
                "Vector representations were computed using Sentence Transformers (<code>all-MiniLM-L6-v2</code>). "
                "Graph structures were constructed in-memory using NetworkX.",
                body_style,
            )
        )

        # Build PDF with two-pass canvas for header/footer stamping
        doc.build(story, canvasmaker=NumberedCanvas)

        # Post-generation validation with PyMuPDF
        with fitz.open(str(output_path)) as pdf_check:
            page_count = len(pdf_check)
            logger.info("PDF dossier validated with PyMuPDF: %s (%d pages)", output_path.name, page_count)

        return page_count


pdf_composer_service = PDFComposerService()
