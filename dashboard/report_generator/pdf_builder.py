"""
PDF Builder — Assembles NLG text + charts + tables into a ReportLab PDF.
"""
import os
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable
)
from reportlab.platypus.flowables import Flowable

# Colors
NAVY = HexColor("#1a1a2e")
ACCENT = HexColor("#0f3460")
HIGHLIGHT = HexColor("#e94560")
TEXT = HexColor("#212529")
MUTED = HexColor("#6c757d")
BORDER = HexColor("#dee2e6")
TABLE_HEADER = HexColor("#2c3e50")
TABLE_ALT = HexColor("#ecf0f1")
GREEN = HexColor("#27ae60")
RED = HexColor("#e74c3c")
ORANGE = HexColor("#f39c12")
BLUE = HexColor("#3498db")
WHITE = HexColor("#ffffff")
LIGHT_BG = HexColor("#f8f9fa")
VERDICT_BG = {"green": HexColor("#eafaf1"), "yellow": HexColor("#fef9e7"), "red": HexColor("#fdedec")}
VERDICT_BORDER = {"green": GREEN, "yellow": ORANGE, "red": RED}


class ColoredLine(Flowable):
    def __init__(self, width, height=2, color=HIGHLIGHT):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.color = color
    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.height)
        self.canv.line(0, 0, self.width, 0)


def _styles():
    return {
        "title": ParagraphStyle("T", fontSize=22, leading=28, textColor=NAVY, fontName="Helvetica-Bold"),
        "subtitle": ParagraphStyle("ST", fontSize=12, leading=16, textColor=ACCENT, fontName="Helvetica"),
        "h1": ParagraphStyle("H1", fontSize=16, leading=22, textColor=NAVY, fontName="Helvetica-Bold",
                             spaceBefore=16, spaceAfter=8),
        "h2": ParagraphStyle("H2", fontSize=12, leading=16, textColor=ACCENT, fontName="Helvetica-Bold",
                             spaceBefore=10, spaceAfter=6),
        "body": ParagraphStyle("B", fontSize=10, leading=15, textColor=TEXT, fontName="Helvetica",
                               spaceAfter=6, alignment=TA_JUSTIFY),
        "small": ParagraphStyle("SM", fontSize=8, leading=11, textColor=MUTED, fontName="Helvetica"),
        "verdict": ParagraphStyle("V", fontSize=10, leading=14, textColor=TEXT, fontName="Helvetica-Bold",
                                  alignment=TA_CENTER),
        "rec": ParagraphStyle("REC", fontSize=9.5, leading=14, textColor=TEXT, fontName="Helvetica",
                              leftIndent=15, bulletIndent=5, spaceAfter=8),
    }


def _make_table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("LEADING", (0,0), (-1,-1), 12),
        ("TEXTCOLOR", (0,0), (-1,-1), TEXT),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("GRID", (0,0), (-1,-1), 0.5, BORDER),
        ("BACKGROUND", (0,0), (-1,0), TABLE_HEADER),
        ("TEXTCOLOR", (0,0), (-1,0), WHITE),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ]
    for i in range(2, len(data), 2):
        style_cmds.append(("BACKGROUND", (0,i), (-1,i), TABLE_ALT))
    t.setStyle(TableStyle(style_cmds))
    return t


def build_pdf(rd, insights, nlg_sections, chart_paths):
    """Build complete PDF, return bytes."""
    from . import nlg_engine
    from . import lexicon

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    s = _styles()
    e = []
    W = 460

    verdict = nlg_engine.get_verdict(insights)

    # ══════════ COVER PAGE ══════════
    e.append(Spacer(1, 40))

    # Logo
    logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "logo.png")
    if os.path.exists(logo_path):
        try:
            e.append(Image(logo_path, width=80, height=60))
            e.append(Spacer(1, 12))
        except Exception:
            pass

    e.append(ColoredLine(W, 3, HIGHLIGHT))
    e.append(Spacer(1, 10))
    e.append(Paragraph("Post-Simulation Analysis Report", s["title"]))
    e.append(Paragraph("Seatwise Revenue Management System", s["subtitle"]))
    e.append(Spacer(1, 16))
    e.append(ColoredLine(W, 1, ACCENT))
    e.append(Spacer(1, 12))

    # Sim params
    params = [
        ["Parameter", "Value"],
        ["Routes", ", ".join(rd.routes[:5]) + (f" +{len(rd.routes)-5} more" if len(rd.routes) > 5 else "")],
        ["Cabins", ", ".join(rd.cabins)],
        ["Flights", str(rd.n_flights)],
        ["Total Capacity", f"{rd.total_capacity:,}"],
        ["Departure Date", rd.date],
        ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M")],
    ]
    e.append(_make_table(params, col_widths=[180, 280]))
    e.append(PageBreak())

    # ══════════ EXECUTIVE SUMMARY ══════════
    e.append(Paragraph("Executive Summary", s["h1"]))
    e.append(ColoredLine(W, 2, HIGHLIGHT))
    e.append(Spacer(1, 8))

    # Verdict box
    v_bg = VERDICT_BG.get(verdict, VERDICT_BG["yellow"])
    v_border = VERDICT_BORDER.get(verdict, ORANGE)
    v_label = lexicon.VERDICT_LABEL.get(verdict, "")
    verdict_table = Table([[Paragraph(v_label, s["verdict"])]], colWidths=[W])
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), v_bg),
        ("BOX", (0,0), (-1,-1), 2, v_border),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ]))
    e.append(verdict_table)
    e.append(Spacer(1, 10))

    # KPI box
    kpis = [
        ["Revenue Delta", "Avg Load Factor", "Total Sold", "Denied Boardings"],
        [f"{rd.revenue_delta_pct:+.1f}%", f"{rd.avg_lf*100:.1f}%",
         f"{rd.total_sold:,} / {rd.total_capacity:,}", str(rd.denied_boardings)],
    ]
    e.append(_make_table(kpis, col_widths=[115, 115, 115, 115]))
    e.append(Spacer(1, 10))

    # NLG paragraphs
    for para in nlg_sections.get("executive_summary", []):
        e.append(Paragraph(para, s["body"]))

    e.append(PageBreak())

    # ══════════ REVENUE ══════════
    e.append(Paragraph("1. Revenue Performance", s["h1"]))
    e.append(ColoredLine(W, 2, BLUE))
    e.append(Spacer(1, 6))

    for para in nlg_sections.get("revenue", []):
        e.append(Paragraph(para, s["body"]))

    # Revenue chart
    if "rev_compare" in chart_paths:
        e.append(Spacer(1, 6))
        e.append(Image(chart_paths["rev_compare"], width=350, height=210))

    # Region table
    if rd.region_performance:
        e.append(Spacer(1, 8))
        e.append(Paragraph("Revenue by Region", s["h2"]))
        reg_data = [["Region", "Flights", "Sold", "Revenue", "Avg LF"]]
        for r in rd.region_performance:
            reg_data.append([
                r["region"], str(r["count"]), str(r["sold"]),
                f"${r['revenue']:,.0f}", f"{r['avg_lf']*100:.1f}%"
            ])
        e.append(_make_table(reg_data, col_widths=[100, 60, 60, 120, 80]))

    e.append(PageBreak())

    # ══════════ LOAD FACTOR ══════════
    e.append(Paragraph("2. Load Factor & Demand Analysis", s["h1"]))
    e.append(ColoredLine(W, 2, GREEN))
    e.append(Spacer(1, 6))

    for para in nlg_sections.get("lf", []):
        e.append(Paragraph(para, s["body"]))

    if "lf_flights" in chart_paths:
        e.append(Spacer(1, 6))
        e.append(Image(chart_paths["lf_flights"], width=400, height=200))

    # Flight table
    if rd.flights:
        e.append(Spacer(1, 8))
        e.append(Paragraph("Per-Flight Summary", s["h2"]))
        fl_data = [["Route", "Cabin", "Sold/Cap", "LF", "Dynamic $", "Delta %"]]
        for f in sorted(rd.flights, key=lambda x: -x.get("load_factor", 0))[:12]:
            fl_data.append([
                f["route"], f["cabin"][:3], f"{f['sold']}/{f['capacity']}",
                f"{f['load_factor']*100:.0f}%", f"${f['revenue_dynamic']:,.0f}",
                f"{f['delta_pct']:+.1f}%"
            ])
        e.append(_make_table(fl_data, col_widths=[75, 40, 65, 50, 100, 60]))

    e.append(PageBreak())

    # ══════════ FARE CLASS & RISK ══════════
    e.append(Paragraph("3. Fare Class Optimization & Risk Assessment", s["h1"]))
    e.append(ColoredLine(W, 2, ORANGE))
    e.append(Spacer(1, 6))

    for para in nlg_sections.get("fareclass", []):
        e.append(Paragraph(para, s["body"]))

    if "fc_pie" in chart_paths:
        e.append(Spacer(1, 6))
        e.append(Image(chart_paths["fc_pie"], width=280, height=210))

    e.append(PageBreak())

    # ══════════ SENTIMENT ══════════
    e.append(Paragraph("4. Sentiment Intelligence", s["h1"]))
    e.append(ColoredLine(W, 2, HIGHLIGHT))
    e.append(Spacer(1, 6))

    for para in nlg_sections.get("sentiment", []):
        e.append(Paragraph(para, s["body"]))

    if rd.sentiment_alerts:
        e.append(Spacer(1, 6))
        e.append(Paragraph("Active Alerts", s["h2"]))
        alert_data = [["City", "Score", "Alert", "Articles"]]
        for a in rd.sentiment_alerts[:8]:
            alert_data.append([a["city"], f"{a['score']*100:.0f}", a["alert"].upper(), str(a["articles"])])
        e.append(_make_table(alert_data, col_widths=[150, 80, 80, 80]))

    # ══════════ RECOMMENDATIONS ══════════
    e.append(Spacer(1, 16))
    e.append(Paragraph("5. Recommendations", s["h1"]))
    e.append(ColoredLine(W, 2, ACCENT))
    e.append(Spacer(1, 6))

    recs = nlg_sections.get("recommendations", [])
    for i, rec in enumerate(recs, 1):
        e.append(Paragraph(f"<bullet>{i}.</bullet> {rec}", s["rec"]))

    # Footer
    e.append(Spacer(1, 20))
    e.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    e.append(Paragraph(
        f"Generated by Seatwise NLG Report Engine | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        s["small"]
    ))

    # Build
    doc.build(e)
    buf.seek(0)
    return buf.read()
