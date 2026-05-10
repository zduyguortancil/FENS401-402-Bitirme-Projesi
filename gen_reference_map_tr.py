"""
Reference Insertion Map for repo_fincal.docx
Generates: <Desktop>/Referans_Yerlestirme_Haritasi.pdf

Bu doküman, repo_fincal.docx içindeki konum-konum eklenecek IEEE-stili
akademik kaynaklarin tam haritasidir. Mevcut [1]-[15] referanslarinin
numaralari korunur; yeni eklemeler [16]-[25] araliginda numaralandirilmistir.
"""
import os
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

WIN_FONTS = "C:/Windows/Fonts"
pdfmetrics.registerFont(TTFont("TR-Roman",      f"{WIN_FONTS}/times.ttf"))
pdfmetrics.registerFont(TTFont("TR-Bold",       f"{WIN_FONTS}/timesbd.ttf"))
pdfmetrics.registerFont(TTFont("TR-Italic",     f"{WIN_FONTS}/timesi.ttf"))
pdfmetrics.registerFont(TTFont("TR-BoldItalic", f"{WIN_FONTS}/timesbi.ttf"))
registerFontFamily("TR-Roman", normal="TR-Roman", bold="TR-Bold",
                    italic="TR-Italic", boldItalic="TR-BoldItalic")

F_NORMAL, F_BOLD, F_ITALIC = "TR-Roman", "TR-Bold", "TR-Italic"
DESKTOP = Path(os.path.expanduser("~")) / "OneDrive" / "Desktop"
OUT = DESKTOP / "Referans_Yerlestirme_Haritasi.pdf"

styles = getSampleStyleSheet()
ACCENT = HexColor("#0b3d91")
GREEN = HexColor("#0a6b3d")
GREY = HexColor("#444444")
LIGHT = HexColor("#e8eaf0")
SOFT = HexColor("#f6f7fa")
ORANGE = HexColor("#a04a00")

S = {}
S["Title"] = ParagraphStyle("Title", parent=styles["Title"], fontName=F_BOLD,
                             fontSize=18, leading=22, alignment=TA_CENTER,
                             textColor=ACCENT, spaceAfter=4)
S["Subtitle"] = ParagraphStyle("Subtitle", parent=styles["Normal"], fontName=F_ITALIC,
                                fontSize=11, leading=14, alignment=TA_CENTER,
                                textColor=GREY, spaceAfter=4)
S["Author"] = ParagraphStyle("Author", parent=styles["Normal"], fontName=F_NORMAL,
                              fontSize=10, leading=13, alignment=TA_CENTER,
                              spaceAfter=2)
S["Affiliation"] = ParagraphStyle("Aff", parent=styles["Normal"], fontName=F_ITALIC,
                                   fontSize=9.5, leading=12, alignment=TA_CENTER,
                                   textColor=GREY, spaceAfter=18)
S["AbstractHead"] = ParagraphStyle("AbstractHead", parent=styles["Normal"],
                                    fontName=F_BOLD, fontSize=10, leading=12,
                                    alignment=TA_CENTER, textColor=GREY, spaceAfter=4)
S["Abstract"] = ParagraphStyle("Abstract", parent=styles["Normal"], fontName=F_NORMAL,
                                fontSize=9.5, leading=13.5, alignment=TA_JUSTIFY,
                                leftIndent=18, rightIndent=18, spaceAfter=14)
S["H1"] = ParagraphStyle("H1", parent=styles["Heading1"], fontName=F_BOLD,
                          fontSize=14, leading=18, textColor=ACCENT,
                          spaceBefore=14, spaceAfter=8, keepWithNext=1)
S["H2"] = ParagraphStyle("H2", parent=styles["Heading2"], fontName=F_BOLD,
                          fontSize=12, leading=15, textColor=ACCENT,
                          spaceBefore=12, spaceAfter=5, keepWithNext=1)
S["H3"] = ParagraphStyle("H3", parent=styles["Heading3"], fontName=F_BOLD,
                          fontSize=10.5, leading=13, textColor=black,
                          spaceBefore=8, spaceAfter=3, keepWithNext=1)
S["Body"] = ParagraphStyle("Body", parent=styles["Normal"], fontName=F_NORMAL,
                            fontSize=10.5, leading=14, alignment=TA_JUSTIFY,
                            spaceAfter=6, firstLineIndent=0)
S["Ref"] = ParagraphStyle("Ref", parent=styles["Normal"], fontName=F_NORMAL,
                           fontSize=9.5, leading=12.5, alignment=TA_LEFT,
                           leftIndent=24, firstLineIndent=-24, spaceAfter=5)
S["RefNew"] = ParagraphStyle("RefNew", parent=S["Ref"], textColor=ACCENT,
                              fontName=F_BOLD)
S["LocationBox"] = ParagraphStyle("LocBox", parent=styles["Normal"],
                                   fontName=F_NORMAL, fontSize=9.5, leading=13,
                                   alignment=TA_LEFT, leftIndent=12,
                                   rightIndent=12, spaceBefore=4, spaceAfter=4,
                                   borderWidth=0.6, borderColor=HexColor("#bbbbbb"),
                                   borderPadding=8, backColor=SOFT)
S["Quote"] = ParagraphStyle("Quote", parent=styles["Normal"], fontName=F_ITALIC,
                             fontSize=9.5, leading=12.5, alignment=TA_JUSTIFY,
                             leftIndent=14, rightIndent=14,
                             textColor=GREY, spaceAfter=4)
S["Warning"] = ParagraphStyle("Warning", parent=styles["Normal"], fontName=F_NORMAL,
                               fontSize=10, leading=13, alignment=TA_JUSTIFY,
                               leftIndent=10, rightIndent=10,
                               borderWidth=0.6, borderColor=ORANGE,
                               borderPadding=8, backColor=HexColor("#fff7e6"),
                               spaceBefore=6, spaceAfter=8)


def P(t, st="Body"):
    return Paragraph(t, S[st])


def on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFont(F_ITALIC, 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2*cm, h - 1.2*cm, "Referans Yerlestirme Haritasi")
    canvas.drawRightString(w - 2*cm, h - 1.2*cm, "Seatwise / FENS 402 Group 16")
    canvas.line(2*cm, h - 1.3*cm, w - 2*cm, h - 1.3*cm)
    canvas.setFont(F_NORMAL, 8.5)
    canvas.drawCentredString(w / 2.0, 1.2*cm, f"— {doc.page} —")
    canvas.restoreState()


def loc_block(section, paragraph_idx, sentence_end_quote, citation,
              rationale):
    """Yerleştirme kutusu: bölüm + paragraf id + cümle sonu + ekleme."""
    txt = (
        f"<b>Bölüm:</b> {section}<br/>"
        f"<b>Paragraf:</b> {paragraph_idx}<br/>"
        f"<b>Cümle sonu:</b> &ldquo;&hellip;{sentence_end_quote}&rdquo;<br/>"
        f"<b>Eklenecek alıntı:</b> <font color='#0b3d91'><b>{citation}</b></font><br/>"
        f"<b>Gerekçe:</b> <i>{rationale}</i>"
    )
    return P(txt, "LocationBox")


# ─────────────────────────────────────────────────────────────────
flow = []
flow.append(Spacer(1, 8))
flow.append(P("Akademik Kaynak Yerlestirme Haritasi", "Title"))
flow.append(P("repo_fincal.docx i&ccedil;in IEEE-stili Referans "
              "Geni&scedil;letme Raporu", "Subtitle"))
flow.append(Spacer(1, 4))
flow.append(P("Group 16 &mdash; FENS 402 Engineering Design Project II", "Author"))
flow.append(P("End&uuml;stri M&uuml;hendisli&#287;i B&ouml;l&uuml;m&uuml; "
              "&mdash; Kadir Has &Uuml;niversitesi &mdash; May&#305;s 2026",
              "Affiliation"))

# ─── ÖZET ─────────────────────────────────────────────────────
flow.append(P("&Ouml;zet", "AbstractHead"))
flow.append(P(
    "Bu rapor, <i>repo_fincal.docx</i> i&ccedil;ine eklenecek 10 yeni "
    "IEEE-stili akademik kaynak i&ccedil;in tam yerle&scedil;tirme haritas&#305;n&#305; "
    "i&ccedil;erir. Mevcut [1]&ndash;[15] numaralar&#305; korunmu&scedil;; yeni "
    "eklemeler s&#305;ras&#305;yla [16]&ndash;[25] olarak numaraland&#305;r&#305;lm&#305;&scedil;t&#305;r. "
    "Her giri&scedil; i&ccedil;in: (i) ait oldu&#287;u ana b&ouml;l&uuml;m ba&scedil;l&#305;&#287;&#305;, "
    "(ii) docx i&ccedil;indeki paragraf indeksi (1&ndash;1897 aral&#305;&#287;&#305;), "
    "(iii) c&uuml;mle-sonu metin parmak izi, (iv) eklenecek alt &ldquo;[N]&rdquo; "
    "kodu ve (v) k&#305;sa akademik gerek&ccedil;e verilmi&scedil;tir. Raporun sonunda "
    "g&uuml;ncellenmi&scedil; tam IEEE referans listesi yer al&#305;r.",
    "Abstract"))

# ═════════════════════════════════════════════════════════════════
# A. Yöntem
# ═════════════════════════════════════════════════════════════════
flow.append(P("A. Y&ouml;ntem", "H1"))
flow.append(P(
    "Eklenen referans setinin se&ccedil;iminde &uuml;&ccedil; ilke g&ouml;zetilmi&scedil;tir: "
    "<b>(i) kalite &ouml;ncelikli</b> &mdash; her referans, c&uuml;mlenin tam "
    "kar&scedil;&#305;l&#305;&#287;&#305;n&#305; veren kanonik veya en yayg&#305;n at&#305;f yap&#305;lan kaynakt&#305;r; "
    "<b>(ii) bo&scedil;luk-kapatma</b> &mdash; tezde s&#305;k&ccedil;a kullan&#305;lan ancak "
    "&scedil;u ana kadar referans verilmemi&scedil; teknik unsurlar (TFT, XGBoost, "
    "DuckDB, K-Means, overbooking modellemesi gibi) hedeflenmi&scedil;tir; "
    "<b>(iii) numaralama tutarl&#305;l&#305;&#287;&#305;</b> &mdash; mevcut [1]&ndash;[15] alanlar&#305; "
    "hi&ccedil; de&#287;i&scedil;tirilmemi&scedil;, yeni at&#305;flar [16]&rsquo;dan ba&scedil;lat&#305;lm&#305;&scedil;t&#305;r.",
    "Body"))
flow.append(P(
    "Mevcut [1]&ndash;[15] referanslar&#305;ndan hi&ccedil;biri silinmemi&scedil;, ta&scedil;&#305;nmam&#305;&scedil; "
    "veya yeniden numaraland&#305;r&#305;lmam&#305;&scedil;t&#305;r. Bu sayede docx i&ccedil;indeki "
    "halihaz&#305;rdaki t&uuml;m in-text [N] alt-alt&#305;flar&#305; (&ouml;rn. 2.7&rsquo;de [1], "
    "[2], [3] vb., 2.8.1.10&rsquo;da [4], [13]) hi&ccedil; m&uuml;dahale gerektirmeden "
    "yerli yerinde kal&#305;r.",
    "Body"))

# ═════════════════════════════════════════════════════════════════
# B. Eklenecek Yeni Kaynaklar — Özet
# ═════════════════════════════════════════════════════════════════
flow.append(P("B. Eklenecek Yeni Kaynaklar &mdash; &Ouml;zet", "H1"))
flow.append(P(
    "A&scedil;a&#287;&#305;daki on referans, raporun farkl&#305; b&ouml;l&uuml;mlerine "
    "yerle&scedil;tirilecektir. T&uuml;m&uuml; hakemli akademik dergi/konferans "
    "yay&#305;n&#305;d&#305;r ve ilgili teknik unsurun kanonik kayna&#287;&#305;n&#305; temsil eder.",
    "Body"))
flow.append(Table([
    [P("<b>No</b>", "Body"), P("<b>Yazar &amp; Y&#305;l</b>", "Body"),
     P("<b>Konu</b>", "Body"), P("<b>Hedef B&ouml;l&uuml;m</b>", "Body")],
    [P("[16]", "Body"), P("Lim et al. 2021", "Body"),
     P("Temporal Fusion Transformer", "Body"), P("2.4 Macro Forecasting", "Body")],
    [P("[17]", "Body"), P("Chen &amp; Guestrin 2016", "Body"),
     P("XGBoost", "Body"), P("2.4 Two-Stage XGBoost", "Body")],
    [P("[18]", "Body"), P("Raasveldt &amp; M&uuml;hleisen 2019", "Body"),
     P("DuckDB embedded analytics", "Body"), P("2.2 Data Storage", "Body")],
    [P("[19]", "Body"), P("McGill &amp; van Ryzin 1999", "Body"),
     P("RM Research Survey", "Body"), P("1.1 Background", "Body")],
    [P("[20]", "Body"), P("Subramanian et al. 1999", "Body"),
     P("Overbooking &amp; no-shows", "Body"), P("2.6.3 No-show Modelling", "Body")],
    [P("[21]", "Body"), P("Brons et al. 2002", "Body"),
     P("Air-travel price elasticity", "Body"), P("2.5.2 Demand Functions", "Body")],
    [P("[22]", "Body"), P("Smith et al. 1992", "Body"),
     P("Yield management at AA", "Body"), P("4.4 Traditional RM", "Body")],
    [P("[23]", "Body"), P("Hartigan &amp; Wong 1979", "Body"),
     P("K-Means algorithm", "Body"), P("2.5.1 Segmentation", "Body")],
    [P("[24]", "Body"), P("Koenker &amp; Bassett 1978", "Body"),
     P("Quantile regression / pinball", "Body"), P("3.5.1 TFT Evaluation", "Body")],
    [P("[25]", "Body"), P("Stonebraker et al. 2005", "Body"),
     P("Columnar (C-Store) DB", "Body"), P("2.2 Data Storage", "Body")],
], colWidths=[1.0*cm, 4.5*cm, 5.5*cm, 5.0*cm], hAlign="CENTER", style=TableStyle([
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LINEABOVE", (0, 0), (-1, 0), 0.7, black),
    ("LINEBELOW", (0, 0), (-1, 0), 0.4, black),
    ("LINEBELOW", (0, -1), (-1, -1), 0.7, black),
    ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
])))

# ═════════════════════════════════════════════════════════════════
# C. Yerleştirme Konumları (sırayla, paragraf indeksine göre)
# ═════════════════════════════════════════════════════════════════
flow.append(P("C. Yerle&scedil;tirme Konumlar&#305; (Tam Konum)", "H1"))
flow.append(P(
    "Yerle&scedil;tirme s&#305;ras&#305; docx&rsquo;teki paragraf indeksine g&ouml;re "
    "(yukar&#305;dan a&scedil;a&#287;&#305;) verilmi&scedil;tir; b&ouml;ylece d&uuml;zenlemeyi tek bir "
    "ge&ccedil;i&scedil;te yapabilirsin. Her kutuda alt&#305; ge&ccedil;ilen c&uuml;mlenin son "
    "birka&ccedil; kelimesi <i>parmak izi</i> olarak verilmi&scedil;tir &mdash; Word&rsquo;de "
    "<b>Ctrl+F</b> ile bu metni aratmak konumu hemen bulur.",
    "Body"))

# ── 1. INTRODUCTION → 1.1 Background and Motivation
flow.append(P("1. INTRODUCTION &rarr; Background and Motivation", "H2"))
flow.append(loc_block(
    "INTRODUCTION &rarr; <i>Background and Motivation</i>",
    "Paragraf 307 (giri&scedil; b&ouml;l&uuml;m&uuml; ilk &ouml;ze paragraf&#305;)",
    "various customer groups and time horizons.",
    "[19]",
    "McGill &amp; van Ryzin (1999) <i>Transportation Science</i>: havayolu RM "
    "literat&uuml;r&uuml;n&uuml;n kanonik tarama makalesi. Giri&scedil; paragraf&#305;ndaki RM "
    "tan&#305;m&#305;n&#305; literat&uuml;re ba&#287;lar."
))

# ── 2.2 Data Engineering → Data Storage and Processing Optimization
flow.append(P("2.2 Data Engineering &rarr; Data Storage and Processing "
              "Optimization", "H2"))
flow.append(loc_block(
    "2.2 Data Engineering and Data Pipeline &rarr; "
    "<i>Data Storage and Processing Optimization</i>",
    "Paragraf 543 (Parquet aciklamasi)",
    "than traditional columnar data formats like CSV.",
    "[25]",
    "Stonebraker et al. (2005) C-Store, VLDB: column-tabanl&#305; depolaman&#305;n "
    "kanonik makalesi. Parquet&rsquo;in mimari &ouml;nc&uuml;l&uuml;n&uuml; akademik olarak "
    "ger&ccedil;ekler."
))
flow.append(loc_block(
    "2.2 Data Engineering and Data Pipeline &rarr; "
    "<i>Data Storage and Processing Optimization</i>",
    "Paragraf 545 (DuckDB tan&#305;t&#305;m&#305;)",
    "system uses DuckDB as an in-process analytical database",
    "[18]",
    "Raasveldt &amp; M&uuml;hleisen (2019) SIGMOD demo: DuckDB&rsquo;nin orijinal "
    "akademik makalesi. G&ouml;m&uuml;l&uuml; analitik veritaban&#305; se&ccedil;iminin sebebini "
    "akademik olarak destekler."
))

# ── 2.4 Forecasting Layer → Temporal Fusion Transformer Architecture
flow.append(P("2.4 Forecasting Layer &rarr; Temporal Fusion Transformer "
              "Architecture", "H2"))
flow.append(loc_block(
    "2.4 Forecasting Layer and Demand Modeling &rarr; "
    "<i>Temporal Fusion Transformer (TFT) Architecture</i>",
    "Paragraf 585 (TFT mimarisinin ilk a&ccedil;&#305;klamas&#305;)",
    "non-temporal time series with multiple horizons.",
    "[16]",
    "Lim, Ar&#305;k, Loeff, Pfister (2021) <i>International Journal of "
    "Forecasting</i>: TFT mimarisinin orijinal makalesi. T&uuml;m raporun "
    "macro forecasting omurgas&#305; bu kayna&#287;a dayan&#305;r."
))

# ── 2.4 Forecasting Layer → Two-Stage XGBoost
flow.append(P("2.4 Forecasting Layer &rarr; Two-Stage XGBoost (Micro)", "H2"))
flow.append(loc_block(
    "2.4 Forecasting Layer &rarr; <i>Micro Demand Forecasting "
    "(Two-Stage XGBoost)</i>",
    "Paragraf 641 (XGBoost'un faydas&#305;n&#305;n a&ccedil;&#305;kland&#305;&#287;&#305; ozet paragrafi)",
    "XGBoost is able to model non-linear",
    "[17]",
    "Chen &amp; Guestrin (2016) KDD: XGBoost&rsquo;un orijinal makalesi. "
    "&Uuml;&ccedil; ayr&#305; XGBoost modeli kullan&#305;lmas&#305;na ra&#287;men hen&uuml;z y&ouml;ntem "
    "kayna&#287;&#305; verilmemi&scedil;tir."
))

# ── 2.5.1 Passenger Segmentation
flow.append(P("2.5.1 Passenger Segmentation", "H2"))
flow.append(loc_block(
    "2.5 Passenger Behavior and Market Modeling &rarr; "
    "<i>2.5.1 Passenger Segmentation</i>",
    "Paragraf 846 (K-Means&rsquo;in se&ccedil;ildi&#287;i ifade)",
    "passenger groups are determined by the K-Means clustering algorithm.",
    "[23]",
    "Hartigan &amp; Wong (1979) <i>Applied Statistics</i>: AS-136 K-Means "
    "algoritmas&#305;n&#305;n kanonik referans&#305;. Y&ouml;ntem tercihinin akademik "
    "gerek&ccedil;esini tamamlar."
))

# ── 2.5.2 Demand Function Modeling
flow.append(P("2.5.2 Demand Function Modeling", "H2"))
flow.append(loc_block(
    "2.5 Passenger Behavior and Market Modeling &rarr; "
    "<i>2.5.2 Demand Function Modeling</i>",
    "Paragraf 923 (segment-bazl&#305; elasticity sav&#305;)",
    "students and leisure passengers are highly price-sensitive.",
    "[21]",
    "Brons, Pels, Nijkamp, Rietveld (2002) <i>Journal of Air Transport "
    "Management</i>: havayolu yolcu fiyat elastikiyetinin meta-analizi. "
    "Segment-bazl&#305; elastikiyet de&#287;erleri i&ccedil;in en uygun ampirik kaynakt&#305;r."
))

# ── 2.6.3 Overbooking & No-Show
flow.append(P("2.6.3 Overbooking and No-Show Modeling", "H2"))
flow.append(loc_block(
    "2.6 Simulation Environment and Decision Framework &rarr; "
    "<i>2.6.3 Overbooking and No-Show Modeling</i>",
    "Paragraf 999 (iptal+no-show modellemesinin a&ccedil;&#305;klamas&#305;)",
    "Both phenomena are modeled explicitly.",
    "[20]",
    "Subramanian, Stidham &amp; Lautenbacher (1999) <i>Transportation "
    "Science</i>: airline overbooking + cancellation + no-show "
    "modellemesinin ana makalesi. Bu alt-b&ouml;l&uuml;m&uuml;n teorik temelidir."
))

# ── 3.5.1 TFT Evaluation
flow.append(P("3.5.1 Evaluation Approach for TFT", "H2"))
flow.append(loc_block(
    "3 Results &amp; Performance Evaluation &rarr; "
    "<i>3.5.1 Evaluation Approach for TFT</i>",
    "Paragraf 1715 (quantile loss / pinball loss tan&#305;t&#305;m&#305;)",
    "quantile loss (pinball loss) or continuous ranked probability score",
    "[24]",
    "Koenker &amp; Bassett (1978) <i>Econometrica</i>: quantile regression "
    "kayb&#305;n&#305;n (pinball loss) orijinal teorik &ccedil;er&ccedil;evesi. TFT&rsquo;nin "
    "olas&#305;l&#305;ksal &ccedil;&#305;kt&#305;s&#305;n&#305;n de&#287;erlendirme metri&#287;i bu kayna&#287;a dayan&#305;r."
))

# ── 4.4 Comparison with Traditional RM
flow.append(P("4.4 Comparison with Traditional RM Systems", "H2"))
flow.append(loc_block(
    "4 Discussion &rarr; <i>4.4 Comparison with Traditional RM Systems</i>",
    "Paragraf 1820 (geleneksel RM&rsquo;in tarihsel tan&#305;t&#305;m&#305;)",
    "it has worked well for decades, but ...",
    "[22]",
    "Smith, Leimkuhler &amp; Darrow (1992) <i>Interfaces</i>: <b>Yield "
    "Management at American Airlines</b> &mdash; geleneksel RM&rsquo;in "
    "endeks &ouml;rne&#287;i ve as&#305;l referans makalesi. Geleneksel-AI "
    "kar&scedil;&#305;la&scedil;t&#305;rmas&#305;n&#305; akademik olarak temellendirir."
))

# ═════════════════════════════════════════════════════════════════
# D. Mevcut Numaralama Uyarısı
# ═════════════════════════════════════════════════════════════════
flow.append(P("D. Mevcut Numaraland&#305;rmada Tespit Edilen Tutars&#305;zl&#305;k",
              "H1"))
flow.append(P(
    "<b>Bilgi:</b> docx&rsquo;in <i>2.7.6.1 Model Selection</i> b&ouml;l&uuml;m&uuml;ndeki "
    "<b>paragraf 1091</b>&rsquo;de &ldquo;Hugging Face Transformers <b>[12]</b>&rdquo; "
    "ifadesi ge&ccedil;mektedir; ancak mevcut kaynak listesinde [12] = "
    "<i>Banks et al., Discrete-Event System Simulation</i> olarak "
    "tan&#305;ml&#305;d&#305;r. Bu, orijinal yaz&#305;mdan kalan bir <i>numara &ccedil;ak&#305;&scedil;mas&#305;d&#305;r</i>; "
    "Hugging Face referans&#305; (Wolf et al. 2020) listede yoktur. "
    "Bu raporda numaralama bozulmas&#305;n diye herhangi bir d&uuml;zenleme "
    "&ouml;nerilmemi&scedil;tir, ancak fark&#305;nda olman&#305;z i&ccedil;in not edilmi&scedil;tir. "
    "&#350;ayet d&uuml;zeltmek isterseniz, Wolf vd. (2020) <i>EMNLP System "
    "Demonstrations</i> bildirisini [26] olarak listeye ekleyip "
    "paragraf 1091&rsquo;deki [12]&rsquo;yi [26] yapabilirsiniz.",
    "Warning"))

# ═════════════════════════════════════════════════════════════════
# E. Tam Güncellenmiş IEEE Kaynakça
# ═════════════════════════════════════════════════════════════════
flow.append(P("E. G&uuml;ncellenmi&scedil; Tam Kaynak&ccedil;a (IEEE)", "H1"))
flow.append(P(
    "Mevcut [1]&ndash;[15] kalemleri (siyah) ile yeni eklenen [16]&ndash;[25] "
    "kalemleri (mavi-kal&#305;n) birle&scedil;tirilmi&scedil; tam liste a&scedil;a&#287;&#305;dad&#305;r. Bu "
    "listenin tamam&#305;, raporun en sonundaki <i>References</i> b&ouml;l&uuml;m&uuml;n&uuml;n "
    "yerini al&#305;r.",
    "Body"))

existing_refs = [
    ("[1] K. Leetaru and P. A. Schrodt, &ldquo;GDELT: Global data on "
     "events, location, and tone, 1979&ndash;2012,&rdquo; in <i>ISA Annual "
     "Convention</i>, vol. 2, no. 4, 2013."),
    ("[2] R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, "
     "A. Y. Ng, and C. Potts, &ldquo;Recursive deep models for semantic "
     "compositionality over a sentiment treebank,&rdquo; in <i>Proc. Conf. "
     "Empirical Methods in Natural Language Processing (EMNLP)</i>, 2013, "
     "pp. 1631&ndash;1642."),
    ("[3] P. He, J. Gao, and W. Chen, &ldquo;DeBERTaV3: Improving DeBERTa "
     "using ELECTRA-style pre-training with gradient-disentangled "
     "embedding sharing,&rdquo; in <i>Proc. Int. Conf. Learn. Represent. "
     "(ICLR)</i>, 2023."),
    ("[4] K. T. Talluri and G. J. van Ryzin, <i>The Theory and Practice "
     "of Revenue Management</i>. New York, NY, USA: Springer, 2004."),
    ("[5] A. Vaswani et al., &ldquo;Attention is all you need,&rdquo; in "
     "<i>Adv. Neural Inf. Process. Syst. (NeurIPS)</i>, vol. 30, 2017."),
    ("[6] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, &ldquo;BERT: "
     "Pre-training of deep bidirectional transformers for language "
     "understanding,&rdquo; in <i>Proc. NAACL-HLT</i>, 2019, "
     "pp. 4171&ndash;4186."),
    ("[7] P. He, X. Liu, J. Gao, and W. Chen, &ldquo;DeBERTa: "
     "Decoding-enhanced BERT with disentangled attention,&rdquo; in "
     "<i>Proc. Int. Conf. Learn. Represent. (ICLR)</i>, 2021."),
    ("[8] B. Pang and L. Lee, &ldquo;Opinion mining and sentiment "
     "analysis,&rdquo; <i>Foundations and Trends in Information "
     "Retrieval</i>, vol. 2, no. 1&ndash;2, pp. 1&ndash;135, 2008."),
    ("[9] B. Liu, <i>Sentiment Analysis: Mining Opinions, Sentiments, "
     "and Emotions</i>. Cambridge, U.K.: Cambridge Univ. Press, 2015."),
    ("[10] A. McCallum and K. Nigam, &ldquo;A comparison of event "
     "models for naive Bayes text classification,&rdquo; in <i>AAAI "
     "Workshop on Learning for Text Categorization</i>, 1998, "
     "pp. 41&ndash;48."),
    ("[11] S. Sun, Y. Wei, K.-L. Tsui, and S. Wang, &ldquo;Forecasting "
     "tourist arrivals with machine learning and internet search "
     "index,&rdquo; <i>Tourism Management</i>, vol. 70, "
     "pp. 1&ndash;10, 2019."),
    ("[12] J. Banks, J. S. Carson II, B. L. Nelson, and D. M. Nicol, "
     "<i>Discrete-Event System Simulation</i>, 5th ed. Upper Saddle "
     "River, NJ, USA: Prentice Hall, 2010."),
    ("[13] P. P. Belobaba, &ldquo;Application of a probabilistic "
     "decision model to airline seat inventory control,&rdquo; "
     "<i>Operations Research</i>, vol. 37, no. 2, "
     "pp. 183&ndash;197, 1989."),
    ("[14] A. C. Cameron and P. K. Trivedi, <i>Regression Analysis of "
     "Count Data</i>, 2nd ed. Cambridge, U.K.: Cambridge Univ. "
     "Press, 2013."),
    ("[15] P. Glasserman, <i>Monte Carlo Methods in Financial "
     "Engineering</i>. New York, NY, USA: Springer, 2004."),
]

new_refs = [
    ("[16] B. Lim, S. &Ouml;. Ar&#305;k, N. Loeff, and T. Pfister, "
     "&ldquo;Temporal fusion transformers for interpretable "
     "multi-horizon time series forecasting,&rdquo; <i>International "
     "Journal of Forecasting</i>, vol. 37, no. 4, "
     "pp. 1748&ndash;1764, 2021."),
    ("[17] T. Chen and C. Guestrin, &ldquo;XGBoost: A scalable tree "
     "boosting system,&rdquo; in <i>Proc. 22nd ACM SIGKDD Int. Conf. "
     "Knowledge Discovery and Data Mining (KDD)</i>, 2016, "
     "pp. 785&ndash;794."),
    ("[18] M. Raasveldt and H. M&uuml;hleisen, &ldquo;DuckDB: An "
     "embeddable analytical database,&rdquo; in <i>Proc. ACM "
     "SIGMOD Int. Conf. on Management of Data</i>, 2019, "
     "pp. 1981&ndash;1984."),
    ("[19] J. I. McGill and G. J. van Ryzin, &ldquo;Revenue "
     "management: Research overview and prospects,&rdquo; "
     "<i>Transportation Science</i>, vol. 33, no. 2, "
     "pp. 233&ndash;256, 1999."),
    ("[20] J. Subramanian, S. Stidham Jr., and C. J. Lautenbacher, "
     "&ldquo;Airline yield management with overbooking, "
     "cancellations, and no-shows,&rdquo; <i>Transportation "
     "Science</i>, vol. 33, no. 2, pp. 147&ndash;167, 1999."),
    ("[21] M. Brons, E. Pels, P. Nijkamp, and P. Rietveld, "
     "&ldquo;Price elasticities of demand for passenger air "
     "travel: A meta-analysis,&rdquo; <i>Journal of Air Transport "
     "Management</i>, vol. 8, no. 3, pp. 165&ndash;175, 2002."),
    ("[22] B. C. Smith, J. F. Leimkuhler, and R. M. Darrow, "
     "&ldquo;Yield management at American Airlines,&rdquo; "
     "<i>Interfaces</i>, vol. 22, no. 1, pp. 8&ndash;31, 1992."),
    ("[23] J. A. Hartigan and M. A. Wong, &ldquo;Algorithm AS 136: A "
     "K-Means clustering algorithm,&rdquo; <i>Journal of the Royal "
     "Statistical Society. Series C (Applied Statistics)</i>, "
     "vol. 28, no. 1, pp. 100&ndash;108, 1979."),
    ("[24] R. Koenker and G. Bassett Jr., &ldquo;Regression "
     "quantiles,&rdquo; <i>Econometrica</i>, vol. 46, no. 1, "
     "pp. 33&ndash;50, 1978."),
    ("[25] M. Stonebraker et al., &ldquo;C-Store: A column-oriented "
     "DBMS,&rdquo; in <i>Proc. 31st Int. Conf. on Very Large Data "
     "Bases (VLDB)</i>, 2005, pp. 553&ndash;564."),
]

for r in existing_refs:
    flow.append(P(r, "Ref"))
for r in new_refs:
    flow.append(P(r, "RefNew"))

flow.append(Spacer(1, 12))
flow.append(P(
    "<i>Toplam:</i> 15 mevcut + 10 yeni = <b>25 ge&ccedil;erli IEEE referans&#305;</b>.",
    "Body"))

# ── Build ──
DESKTOP.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=2.0*cm, rightMargin=2.0*cm,
    topMargin=2.2*cm, bottomMargin=2.0*cm,
    title="Reference Insertion Map",
    author="Group 16 - Seatwise",
)
doc.build(flow, onFirstPage=on_page, onLaterPages=on_page)
print(f"OK -> {OUT}")
print(f"Size: {OUT.stat().st_size/1024:.1f} KB")
