"""Generate PDF from markdown technical doc."""
import re
import sys
from fpdf import FPDF

FONT_DIR = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "matplotlib", "mpl-data", "fonts", "ttf")

# Allow command-line override or use defaults
INPUT = sys.argv[1] if len(sys.argv) > 1 else "SENTIMENT_TECHNICAL_DOC.md"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "Sentiment_Teknik_Dokumantasyon.pdf"


class PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("dejavu", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "Sentiment Analiz Modulu - Teknik Dokumantasyon", align="C")
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("dejavu", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Sayfa {self.page_no()}/{{nb}}", align="C")


def build_pdf():
    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Register fonts
    pdf.add_font("dejavu", "", f"{FONT_DIR}\\DejaVuSans.ttf")
    pdf.add_font("dejavu", "B", f"{FONT_DIR}\\DejaVuSans-Bold.ttf")
    pdf.add_font("dejavu", "I", f"{FONT_DIR}\\DejaVuSans-Oblique.ttf")
    pdf.add_font("dejavu", "BI", f"{FONT_DIR}\\DejaVuSans-BoldOblique.ttf")
    pdf.add_font("mono", "", f"{FONT_DIR}\\DejaVuSansMono.ttf")
    pdf.add_font("mono", "B", f"{FONT_DIR}\\DejaVuSansMono-Bold.ttf")

    # Read markdown
    with open(INPUT, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract title from first # heading
    title_match = re.match(r"^#\s+(.+?)(?:\s*-\s*(.+))?$", content, re.MULTILINE)
    if title_match:
        main_title = title_match.group(1).strip()
        subtitle = title_match.group(2).strip() if title_match.group(2) else ""
    else:
        main_title = "Teknik Dokumantasyon"
        subtitle = ""

    # Remove the top-level title (we already have a title page)
    content = re.sub(r"^#\s+.*\n", "", content, count=1)

    pdf.add_page()

    # Title page
    pdf.ln(40)
    pdf.set_font("dejavu", "B", 22)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 12, main_title, align="C", new_x="LMARGIN", new_y="NEXT")
    if subtitle:
        pdf.ln(4)
        pdf.set_font("dejavu", "", 14)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 10, subtitle, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("dejavu", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Flight Snapshot Dashboard", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "FENS 401/402 Bitirme Projesi", align="C", new_x="LMARGIN", new_y="NEXT")

    lines = content.split("\n")
    in_code_block = False
    in_table = False
    table_rows = []
    table_col_count = 0

    pdf.add_page()

    i = 0
    while i < len(lines):
        line = lines[i]

        # Code blocks
        if line.strip().startswith("```"):
            if not in_code_block:
                in_code_block = True
                # Draw code block background
                i += 1
                code_lines = []
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                # Render code block
                pdf.ln(2)
                x_start = pdf.get_x()
                y_start = pdf.get_y()
                pdf.set_font("mono", "", 8)
                line_h = 4.5
                block_h = len(code_lines) * line_h + 6
                # Check page break
                if pdf.get_y() + block_h > pdf.h - 20:
                    pdf.add_page()
                    y_start = pdf.get_y()
                pdf.set_fill_color(240, 240, 240)
                pdf.rect(pdf.l_margin, y_start, pdf.w - pdf.l_margin - pdf.r_margin, block_h, "F")
                pdf.set_text_color(40, 40, 40)
                pdf.set_y(y_start + 3)
                for cl in code_lines:
                    pdf.set_x(pdf.l_margin + 4)
                    pdf.cell(0, line_h, cl, new_x="LMARGIN", new_y="NEXT")
                pdf.set_y(y_start + block_h + 2)
                in_code_block = False
            i += 1
            continue

        # Table detection
        if "|" in line and line.strip().startswith("|"):
            if not in_table:
                in_table = True
                table_rows = []
            # Parse table row
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # Skip separator rows
            if all(re.match(r"^[-:]+$", c) for c in cells):
                i += 1
                continue
            table_rows.append(cells)
            table_col_count = max(table_col_count, len(cells))
            # Check if next line is still table
            if i + 1 >= len(lines) or "|" not in lines[i + 1]:
                # Render table
                _render_table(pdf, table_rows, table_col_count)
                in_table = False
                table_rows = []
                table_col_count = 0
            i += 1
            continue

        # Headings
        if line.startswith("## "):
            pdf.ln(8)
            text = line[3:].strip()
            # Remove numbering like "1. "
            pdf.set_font("dejavu", "B", 14)
            pdf.set_text_color(25, 25, 25)
            if pdf.get_y() > pdf.h - 40:
                pdf.add_page()
            pdf.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
            pdf.set_draw_color(70, 130, 200)
            pdf.set_line_width(0.5)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + 50, pdf.get_y())
            pdf.set_line_width(0.2)
            pdf.ln(4)
            i += 1
            continue

        if line.startswith("### "):
            pdf.ln(5)
            text = line[4:].strip()
            pdf.set_font("dejavu", "B", 11)
            pdf.set_text_color(50, 50, 50)
            if pdf.get_y() > pdf.h - 30:
                pdf.add_page()
            pdf.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            i += 1
            continue

        # Horizontal rules
        if line.strip() == "---":
            pdf.ln(4)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(4)
            i += 1
            continue

        # Empty lines
        if not line.strip():
            pdf.ln(3)
            i += 1
            continue

        # Bullet points
        if line.strip().startswith("- ") or line.strip().startswith("* "):
            indent = len(line) - len(line.lstrip())
            text = line.strip()[2:]
            pdf.set_font("dejavu", "", 9.5)
            pdf.set_text_color(40, 40, 40)
            x_offset = pdf.l_margin + 4 + (indent * 2)
            pdf.set_x(x_offset)
            pdf.cell(4, 6, chr(8226))  # bullet
            _write_rich_line(pdf, text, x_offset + 5)
            i += 1
            continue

        # Numbered lists
        m = re.match(r"^(\d+)\.\s+(.*)", line.strip())
        if m:
            pdf.set_font("dejavu", "", 9.5)
            pdf.set_text_color(40, 40, 40)
            pdf.set_x(pdf.l_margin + 4)
            pdf.cell(6, 6, f"{m.group(1)}.")
            _write_rich_line(pdf, m.group(2), pdf.l_margin + 10)
            i += 1
            continue

        # Normal paragraph
        if line.strip():
            pdf.set_font("dejavu", "", 9.5)
            pdf.set_text_color(40, 40, 40)
            _write_rich_line(pdf, line.strip(), pdf.l_margin)
            i += 1
            continue

        i += 1

    pdf.output(OUTPUT)
    print(f"PDF saved: {OUTPUT}")


def _write_rich_line(pdf, text, x_start):
    """Write a line with bold/code inline formatting."""
    pdf.set_x(x_start)
    usable_w = pdf.w - pdf.r_margin - x_start

    # Split by inline code and bold markers
    parts = re.split(r"(`[^`]+`|\*\*[^*]+\*\*)", text)
    line_h = 6

    # Check if we need a page break
    if pdf.get_y() + line_h > pdf.h - 20:
        pdf.add_page()
        pdf.set_x(x_start)

    for part in parts:
        if part.startswith("`") and part.endswith("`"):
            inner = part[1:-1]
            pdf.set_font("mono", "", 8.5)
            pdf.set_text_color(180, 50, 50)
            w = pdf.get_string_width(inner) + 2
            if pdf.get_x() + w > pdf.w - pdf.r_margin:
                pdf.ln(line_h)
                pdf.set_x(x_start)
            pdf.cell(w, line_h, inner)
            pdf.set_font("dejavu", "", 9.5)
            pdf.set_text_color(40, 40, 40)
        elif part.startswith("**") and part.endswith("**"):
            inner = part[2:-2]
            pdf.set_font("dejavu", "B", 9.5)
            w = pdf.get_string_width(inner) + 1
            if pdf.get_x() + w > pdf.w - pdf.r_margin:
                pdf.ln(line_h)
                pdf.set_x(x_start)
            pdf.cell(w, line_h, inner)
            pdf.set_font("dejavu", "", 9.5)
        else:
            # Word-wrap long text
            words = part.split(" ")
            for wi, word in enumerate(words):
                w = pdf.get_string_width(word + " ")
                if pdf.get_x() + w > pdf.w - pdf.r_margin and pdf.get_x() > x_start + 5:
                    pdf.ln(line_h)
                    pdf.set_x(x_start)
                pdf.cell(w, line_h, word + " " if wi < len(words) - 1 else word)

    pdf.ln(line_h)


def _render_table(pdf, rows, col_count):
    """Render a table."""
    if not rows:
        return

    pdf.ln(3)
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    col_w = usable / col_count

    # Try to auto-size columns based on content
    col_widths = []
    for c in range(col_count):
        max_w = 15
        for row in rows:
            if c < len(row):
                w = pdf.get_string_width(row[c]) + 6
                max_w = max(max_w, w)
        col_widths.append(max_w)
    # Normalize to fit page
    total = sum(col_widths)
    if total > usable:
        col_widths = [w * usable / total for w in col_widths]

    for ri, row in enumerate(rows):
        # Check page break
        if pdf.get_y() + 8 > pdf.h - 20:
            pdf.add_page()

        if ri == 0:
            # Header row
            pdf.set_font("dejavu", "B", 8.5)
            pdf.set_fill_color(55, 65, 81)
            pdf.set_text_color(255, 255, 255)
        else:
            pdf.set_font("dejavu", "", 8.5)
            if ri % 2 == 0:
                pdf.set_fill_color(245, 245, 245)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(40, 40, 40)

        for c in range(col_count):
            cell_text = row[c] if c < len(row) else ""
            w = col_widths[c] if c < len(col_widths) else col_w
            pdf.cell(w, 7, cell_text, border=0, fill=True)
        pdf.ln()

    pdf.set_text_color(40, 40, 40)
    pdf.ln(3)


if __name__ == "__main__":
    build_pdf()
