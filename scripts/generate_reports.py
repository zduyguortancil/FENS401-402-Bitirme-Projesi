"""
SeatWise PDF Report Generator
Iki rapor uretir:
1. Teknik Mimari Raporu
2. Fiyatlandirma Adim Adim (ornek senaryolu)
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable, Image
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping

# ── Renkler ──
NAVY = HexColor("#1a1a2e")
DARK_BLUE = HexColor("#16213e")
ACCENT = HexColor("#0f3460")
HIGHLIGHT = HexColor("#e94560")
LIGHT_BG = HexColor("#f8f9fa")
BORDER = HexColor("#dee2e6")
TEXT = HexColor("#212529")
MUTED = HexColor("#6c757d")
CODE_BG = HexColor("#f1f3f5")
WHITE = HexColor("#ffffff")
TABLE_HEADER = HexColor("#2c3e50")
TABLE_ALT = HexColor("#ecf0f1")
SUCCESS = HexColor("#27ae60")
WARNING = HexColor("#f39c12")
DANGER = HexColor("#e74c3c")
INFO_BLUE = HexColor("#3498db")

DESKTOP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "DocTitle", parent=styles["Title"],
        fontSize=28, leading=34, textColor=NAVY,
        spaceAfter=6, alignment=TA_LEFT,
        fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        "DocSubtitle", parent=styles["Normal"],
        fontSize=14, leading=18, textColor=ACCENT,
        spaceAfter=20, fontName="Helvetica"
    ))
    styles.add(ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontSize=20, leading=26, textColor=NAVY,
        spaceBefore=24, spaceAfter=12,
        fontName="Helvetica-Bold",
        borderWidth=0, borderColor=HIGHLIGHT,
        borderPadding=(0, 0, 4, 0),
    ))
    styles.add(ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=15, leading=20, textColor=ACCENT,
        spaceBefore=16, spaceAfter=8,
        fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        "H3", parent=styles["Heading3"],
        fontSize=12, leading=16, textColor=DARK_BLUE,
        spaceBefore=10, spaceAfter=6,
        fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        "BodyText2", parent=styles["Normal"],
        fontSize=10, leading=15, textColor=TEXT,
        spaceAfter=6, alignment=TA_JUSTIFY,
        fontName="Helvetica"
    ))
    styles.add(ParagraphStyle(
        "CodeBlock", parent=styles["Normal"],
        fontSize=9, leading=13, textColor=HexColor("#c7254e"),
        fontName="Courier", backColor=CODE_BG,
        borderWidth=0.5, borderColor=BORDER,
        borderPadding=6, spaceAfter=8, spaceBefore=4,
    ))
    styles.add(ParagraphStyle(
        "Formula", parent=styles["Normal"],
        fontSize=11, leading=16, textColor=NAVY,
        fontName="Courier-Bold", alignment=TA_CENTER,
        spaceBefore=8, spaceAfter=8,
        backColor=HexColor("#eef2ff"),
        borderWidth=1, borderColor=INFO_BLUE,
        borderPadding=10,
    ))
    styles.add(ParagraphStyle(
        "Caption", parent=styles["Normal"],
        fontSize=9, leading=12, textColor=MUTED,
        alignment=TA_CENTER, spaceAfter=12,
        fontName="Helvetica-Oblique"
    ))
    styles.add(ParagraphStyle(
        "BulletItem", parent=styles["Normal"],
        fontSize=10, leading=15, textColor=TEXT,
        leftIndent=20, bulletIndent=8,
        spaceAfter=3, fontName="Helvetica"
    ))
    styles.add(ParagraphStyle(
        "SmallNote", parent=styles["Normal"],
        fontSize=8, leading=11, textColor=MUTED,
        fontName="Helvetica-Oblique"
    ))
    return styles


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


class BoxedText(Flowable):
    """Renkli arka planli kutu."""
    def __init__(self, text, width, bg_color=LIGHT_BG, border_color=ACCENT, text_color=TEXT,
                 font_size=10, padding=10):
        Flowable.__init__(self)
        self.text = text
        self.box_width = width
        self.bg_color = bg_color
        self.border_color = border_color
        self.text_color = text_color
        self.font_size = font_size
        self.padding = padding
        self._lines = text.split("\n")
        self.height = len(self._lines) * (font_size + 4) + padding * 2

    def draw(self):
        c = self.canv
        c.setFillColor(self.bg_color)
        c.setStrokeColor(self.border_color)
        c.setLineWidth(1)
        c.roundRect(0, 0, self.box_width, self.height, 4, fill=1, stroke=1)
        c.setFillColor(self.text_color)
        c.setFont("Courier", self.font_size)
        y = self.height - self.padding - self.font_size
        for line in self._lines:
            c.drawString(self.padding, y, line)
            y -= self.font_size + 4


def make_table(data, col_widths=None, header=True):
    """Styled table."""
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("LEADING", (0, 0), (-1, -1), 13),
        ("TEXTCOLOR", (0, 0), (-1, -1), TEXT),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
    ]
    if header:
        style_cmds += [
            ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
        ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), TABLE_ALT))
    t.setStyle(TableStyle(style_cmds))
    return t


def add_formula(elements, text, styles):
    elements.append(Paragraph(text, styles["Formula"]))


def add_code_block(elements, lines, width=460):
    elements.append(BoxedText("\n".join(lines), width, bg_color=CODE_BG,
                              border_color=BORDER, text_color=HexColor("#495057"),
                              font_size=8.5, padding=8))


def bullet(text, styles):
    return Paragraph(f"<bullet>&bull;</bullet> {text}", styles["BulletItem"])


# ═══════════════════════════════════════════════════════════════
# RAPOR 1: TEKNIK MIMARI
# ═══════════════════════════════════════════════════════════════

def build_technical_report():
    path = os.path.join(DESKTOP, "SeatWise_Teknik_Mimari_Raporu.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2.5*cm, bottomMargin=2*cm)
    s = get_styles()
    e = []  # elements
    W = 460  # usable width approx

    # ── KAPAK ──
    e.append(Spacer(1, 60))
    e.append(ColoredLine(W, 3, HIGHLIGHT))
    e.append(Spacer(1, 12))
    e.append(Paragraph("SeatWise", s["DocTitle"]))
    e.append(Paragraph("End-to-End Havayolu Gelir Yonetimi Sistemi<br/>Teknik Mimari Raporu", s["DocSubtitle"]))
    e.append(Spacer(1, 20))
    e.append(ColoredLine(W, 1, ACCENT))
    e.append(Spacer(1, 16))
    e.append(Paragraph("Ahmet Furkan Gokbulut", s["H3"]))
    e.append(Paragraph("Kadir Has Universitesi — Endustri Muhendisligi &amp; Bilgisayar Muhendisligi (Double Major)", s["BodyText2"]))
    e.append(Paragraph("Mart 2026", s["SmallNote"]))
    e.append(PageBreak())

    # ══════════════ 1. SISTEM GENEL BAKIS ══════════════
    e.append(Paragraph("1. Sistem Genel Bakis", s["H1"]))
    e.append(ColoredLine(W, 2, HIGHLIGHT))
    e.append(Spacer(1, 8))
    e.append(Paragraph(
        "SeatWise, bir havayolu icin uctan uca gelir yonetimi (Revenue Management) sistemidir. "
        "Makine ogrenimi tahmini, dinamik fiyatlandirma, fare class optimizasyonu, O&amp;D network "
        "optimizasyonu ve canli simulasyonu tek bir pipeline'da birlestirerek, statik fiyatlamaya "
        "gore olculebilir gelir artisi saglar.", s["BodyText2"]))

    e.append(Paragraph("1.1 Katmanli Mimari", s["H2"]))
    arch_data = [
        ["Katman", "Bilesenler", "Gorev"],
        ["Dashboard", "Flask API, HTML/JS/CSS", "REST endpoint'ler, UI"],
        ["Simulasyon Motoru", "SimClock, Bot Generation, Cancel, NoShow", "Canli ucus simulasyonu"],
        ["Fiyatlandirma", "PricingEngine (4 carpan)", "Dinamik fiyat hesaplama"],
        ["Network Optimizer", "EMSR-b, Bid Price, O&D", "Fare class + koltuk koruması"],
        ["Forecast Bridge", "Two-Stage XGB, TFT, Pickup XGB", "ML model entegrasyonu"],
        ["Sentiment", "GDELT + Google News + DeBERTa", "Destinasyon duygu analizi"],
        ["Veri", "DuckDB + Apache Parquet", "146K kayit, 200 entity"],
    ]
    e.append(make_table(arch_data, col_widths=[90, 170, 200]))

    e.append(Paragraph("1.2 Veri Seti", s["H2"]))
    e.append(bullet("<b>Panel Data:</b> 200 entity (rota x kabin) x 730 gun (2 yil)", s))
    e.append(bullet("<b>50 rota,</b> IST hub'li — 5 bolge (Europe, Middle East, Africa, Asia, Americas)", s))
    e.append(bullet("<b>2 kabin:</b> Economy (~300 koltuk) ve Business (~49 koltuk)", s))
    e.append(bullet("<b>Toplam:</b> ~146,000 gunluk gozlem", s))
    e.append(bullet("<b>Feature:</b> 49 (Pickup) + 31 (Two-Stage) + 200+ (TFT temporal)", s))

    e.append(PageBreak())

    # ══════════════ 2. TAHMIN KATMANI ══════════════
    e.append(Paragraph("2. Tahmin Katmani (Forecast Layer)", s["H1"]))
    e.append(ColoredLine(W, 2, HIGHLIGHT))
    e.append(Spacer(1, 8))
    e.append(Paragraph(
        "Uc farkli model, uc farkli gorevle simulasyona baglanir: Two-Stage XGBoost gunluk talep "
        "tahminini verir (gaz pedali), TFT toplam yolcu tahminiyle tavan/taban regulatoru gorevi gorur "
        "(hiz limiti), XGBoost Pickup ise kalan yolcu tahminiyle fiyatlandirmaya girdi saglar (direksiyon).",
        s["BodyText2"]))

    # TFT
    e.append(Paragraph("2.1 Temporal Fusion Transformer (TFT)", s["H2"]))
    e.append(Paragraph(
        "Rota-gun bazinda toplam yolcu tahmini. Encoder-Decoder yapisi, Variable Selection Network, "
        "QuantileLoss (0.1, 0.5, 0.9), GroupNormalizer. 200 entity uzerinde egitilmis, "
        "train/val/test = 60/20/20 split.", s["BodyText2"]))

    tft_perf = [
        ["Metrik", "Deger"],
        ["Test MAE", "14.03"],
        ["Test Correlation", "0.991"],
        ["Entity Sayisi", "200"],
        ["Tahmin Sayisi", "55,066"],
        ["max_encoder_length", "60"],
        ["max_prediction_length", "30"],
    ]
    e.append(make_table(tft_perf, col_widths=[200, 260]))

    e.append(Paragraph("Unconstraining (Gizli Talep Tahmini):", s["H3"]))
    e.append(Paragraph(
        "Yuksek talepli ucuslarda kapasite siniri nedeniyle gozlenemeyen talep vardir. "
        "TFT tahmini kapasitenin %90'ini astiginda gizli talep eklenir:", s["BodyText2"]))
    add_formula(e, "tft_total' = tft_total x 1.15&nbsp;&nbsp;&nbsp;(eger tft_total / capacity &gt; 0.90)", s)

    # Two-Stage
    e.append(Paragraph("2.2 Two-Stage XGBoost (Hurdle Model)", s["H2"]))
    e.append(Paragraph(
        "Gunluk satis verisi %70.8 oraninda sifir degere sahiptir (zero-inflated). Tek bir "
        "regresyon modeli bu yapiyi ogrenemez. Cozum: iki asamali (hurdle) model.", s["BodyText2"]))
    add_formula(e, "daily_demand = P(sale &gt; 0) x E[pax | sale &gt; 0]", s)
    ts_perf = [
        ["Bilesen", "Model", "Metrik", "Deger"],
        ["Asama 1", "XGBClassifier", "AUC", "0.835"],
        ["Asama 2", "XGBRegressor", "MAE", "0.78"],
        ["Bilesik", "Hurdle", "Feature Sayisi", "31"],
    ]
    e.append(make_table(ts_perf, col_widths=[80, 120, 120, 140]))

    # Pickup
    e.append(Paragraph("2.3 XGBoost Pickup Model", s["H2"]))
    e.append(Paragraph(
        "Kalan yolcu (remaining_pax) tahmini. Pricing engine'e expected_final_lf bilgisi verir. "
        "SHAP TreeExplainer ile her ucus icin top-10 feature importance API'den dondurulur.",
        s["BodyText2"]))
    pk_perf = [
        ["Metrik", "Deger"],
        ["MAE", "3.45"],
        ["WAPE", "9.82%"],
        ["Improvement vs Baseline", "70.4%"],
        ["Feature Sayisi", "49"],
    ]
    e.append(make_table(pk_perf, col_widths=[200, 260]))
    e.append(Paragraph("Tipik SHAP sirasi: remaining_seats &gt; dtd &gt; route_n_flights &gt; pax_sold_cum &gt; capacity", s["SmallNote"]))

    # ForecastBridge
    e.append(Paragraph("2.4 ForecastBridge", s["H2"]))
    e.append(Paragraph(
        "Uc modeli simulasyona baglayan kopru. Gunde 1 kez tum aktif ucuslar icin batch predict yapar, "
        "sonucu cache'ler. Her ucus icin: daily_demand, p_sale, e_pax, predicted_remaining dondurur.",
        s["BodyText2"]))

    e.append(Paragraph("TFT Band (Tavan/Taban Regulatoru):", s["H3"]))
    e.append(Paragraph("S-curve ile kumulatif beklenti hesaplanir:", s["BodyText2"]))
    add_formula(e, "cum_fraction = 1 - (dtd / 180)^1.5&nbsp;&nbsp;&nbsp;[dtd &le; 180]", s)
    add_formula(e, "daily_fraction = cum_fraction(dtd) - cum_fraction(dtd + 1)", s)
    add_formula(e, "daily_floor = tft_total x daily_fraction x 0.3", s)
    add_formula(e, "daily_ceiling = tft_total x daily_fraction x 2.0", s)

    e.append(PageBreak())

    # ══════════════ 3. FIYATLANDIRMA ══════════════
    e.append(Paragraph("3. Fiyatlandirma Katmani (Pricing Engine)", s["H1"]))
    e.append(ColoredLine(W, 2, HIGHLIGHT))
    e.append(Spacer(1, 8))

    e.append(Paragraph("3.1 Ana Fiyat Formulu", s["H2"]))
    add_formula(e, "fiyat(fc) = baz_fiyat x fc_mult x arz x talep x sentiment x musteri", s)

    e.append(Paragraph("3.2 Baz Fiyat (Veriden Kalibre Edilmis)", s["H2"]))
    e.append(Paragraph(
        "Baz fiyat formulu, historik bilet verisi uzerinde lineer regresyon ile ogrenilmistir. "
        "102,200 ornek uzerinde fit edilen modelin R&sup2; degeri Economy icin 0.979, Business icin 0.963'tur.",
        s["BodyText2"]))
    add_formula(e, "Economy: base = 4.01 + distance_km x 0.0810&nbsp;&nbsp;&nbsp;(R&sup2; = 0.979, n = 102,200)", s)
    add_formula(e, "Business: base = 100.12 + distance_km x 0.3322&nbsp;&nbsp;&nbsp;(R&sup2; = 0.963, n = 102,200)", s)
    e.append(Paragraph(
        "Rota bazli faktor (route_factor) ayni veriden, mesafe etkisi cikarildiktan sonra "
        "kalan rota spesifik fark olarak turetilmistir (100 rota).", s["BodyText2"]))

    e.append(Paragraph("3.3 Fare Class Tanimlari", s["H2"]))
    fc_data = [
        ["Sinif", "Carpan", "Kota", "LF Esigi", "Ozellikler"],
        ["V — Promosyon", "0.50", "%15", "%40", "Degisiklik/iptal yok"],
        ["K — Indirimli", "0.75", "%25", "%60", "Ucretli degisiklik, 1 bagaj"],
        ["M — Esnek", "1.00", "%35", "%85", "Ucretsiz degisiklik, 2 bagaj"],
        ["Y — Tam Fiyat", "1.50", "%100", "%100", "Tam esneklik, iptal/degisiklik"],
    ]
    e.append(make_table(fc_data, col_widths=[90, 55, 50, 55, 210]))

    e.append(Paragraph("3.4 Arz Carpani (Supply Multiplier)", s["H2"]))
    e.append(Paragraph("<b>Model-Driven yol</b> (Pickup modeli varsa):", s["BodyText2"]))
    add_formula(e, "expected_final_lf = (sold + predicted_remaining) / capacity", s)

    lf_table = [
        ["Expected Final LF", "lf_mult", "Kaynak"],
        [">= 95%", "1.80", "Model-driven (sabit)"],
        [">= 85%", "1.40", "Model-driven (sabit)"],
        [">= 70%", "1.15", "Model-driven (sabit)"],
        ["< 70%", "1.00 (asla indirim yok)", "Model-driven (sabit)"],
    ]
    e.append(make_table(lf_table, col_widths=[140, 170, 150]))

    e.append(Paragraph("<b>Fallback LF Curve</b> (Pickup modeli yoksa, veriden kalibre):", s["BodyText2"]))
    lf_cal_table = [
        ["LF Araligi", "Carpan (veriden)", "n gozlem"],
        ["< 30%", "1.00 (baseline)", "referans"],
        ["30 - 50%", "1.12", "veriden"],
        ["50 - 70%", "1.37", "veriden"],
        ["70 - 85%", "1.68", "veriden"],
        ["85 - 95%", "1.95", "veriden"],
        [">= 95%", "2.40", "veriden"],
    ]
    e.append(make_table(lf_cal_table, col_widths=[140, 170, 150]))

    e.append(Paragraph("<b>DTD Price Curve</b> (veriden kalibre, DTD 31-60 = 1.00 baseline):", s["BodyText2"]))
    dtd_boost = [
        ["DTD Araligi", "Carpan (veriden)", "Ortalama Fiyat"],
        ["0 - 3", "2.11", "$761"],
        ["4 - 7", "1.79", "$647"],
        ["8 - 14", "1.47", "$532"],
        ["15 - 30", "1.21", "$436"],
        ["31 - 60", "1.00 (baseline)", "$360"],
        ["61 - 90", "0.89", "$322"],
        ["91 - 120", "0.79", "$285"],
        ["121+", "0.79", "$284"],
    ]
    e.append(make_table(dtd_boost, col_widths=[120, 170, 170]))
    add_formula(e, "supply_multiplier = lf_mult x dtd_boost", s)

    e.append(Paragraph("3.5 Talep Carpani (Demand Multiplier)", s["H2"]))
    add_formula(e, "raw = season_factor x special_factor x dow_factor", s)
    add_formula(e, "demand_multiplier = 1.0 + (raw - 1.0) x 0.7&nbsp;&nbsp;&nbsp;[dampening]", s)

    e.append(Paragraph("Sezon Faktorleri (veriden kalibre, DTD 30-60 kontrol grubu):", s["H3"]))
    season_data = [
        ["Ay", "Oca", "Sub", "Mar", "Nis", "May", "Haz", "Tem", "Agu", "Eyl", "Eki", "Kas", "Ara"],
        ["Faktor", "0.989", "0.984", "0.983", "0.996", "1.001", "1.013", "1.022", "1.025", "0.988", "0.982", "1.007", "1.011"],
    ]
    e.append(make_table(season_data, col_widths=[40]+[35]*12))
    e.append(Paragraph("Not: Sezon etkisi buyuk olcude LF uzerinden supply multiplier'a yansir, dogrudan fiyat etkisi dusuktur.", s["SmallNote"]))

    e.append(Paragraph("Ozel Gunler:", s["H3"]))
    special_data = [
        ["Donem", "Faktor"],
        ["Ramazan Bayrami (18-23 Mart)", "1.50"],
        ["Kurban Bayrami (25-30 Mayis)", "1.60"],
        ["Cumhuriyet Bayrami (29-30 Ekim)", "1.25"],
        ["Yilbasi (28-31 Aralik)", "1.40"],
    ]
    e.append(make_table(special_data, col_widths=[260, 200]))

    e.append(Paragraph("Hafta Gunu Faktorleri (veriden kalibre):", s["H3"]))
    dow_data = [
        ["Gun", "Pzt", "Sal", "Car", "Per", "Cum", "Cmt", "Paz"],
        ["Faktor", "0.999", "0.997", "1.003", "1.000", "1.001", "0.999", "1.000"],
    ]
    e.append(make_table(dow_data, col_widths=[55]+[58]*7))
    e.append(Paragraph("Not: Hafta gunu etkisi istatistiksel olarak dusuk — talep farki LF uzerinden yakalanir.", s["SmallNote"]))

    e.append(Paragraph("3.6 Sentiment Carpani", s["H2"]))
    add_formula(e, "sentiment_multiplier = 1.0 + composite_score x 0.15", s)
    e.append(Paragraph(
        "composite_score [-1.0, +1.0] arasinda. Negatif olay (teror, dogal afet) fiyati dusurur, "
        "pozitif olay (festival, turizm) fiyati arttirir. Kaynak: GDELT + Google News RSS, "
        "DeBERTa NLI siniflandirma, 14-gun recency filtresi, 51 destinasyon sehri.", s["BodyText2"]))

    e.append(Paragraph("3.7 Musteri Carpani", s["H2"]))
    add_formula(e, "wtp_avg = (wtp_min + wtp_max) / 2", s)
    add_formula(e, "customer_mult = clamp(0.85 + (wtp_avg - 0.5) x 0.2, 0.90, 1.20)", s)

    e.append(Paragraph("Davranis Faktorleri (booking sayfasindan):", s["H3"]))
    behav_data = [
        ["Davranis", "Carpan"],
        ["Geri donen ziyaretci", "x 1.05"],
        ["Uzun bekleme (>180s)", "x 0.97"],
        ["Cok tarih aradi (>5)", "x 0.98"],
        ["Sepet birakti + dondu", "x 1.08"],
        ["FF Gold/Elite uyesi", "x 1.06"],
        ["Mobil cihaz", "x 1.02"],
        ["4+ yolcu (aile)", "x 0.97"],
    ]
    e.append(make_table(behav_data, col_widths=[260, 200]))

    e.append(Paragraph("3.8 Satin Alma Olasiligi", s["H2"]))
    add_formula(e, "ratio = (offered - wtp_min_price) / (wtp_max_price - wtp_min_price)", s)
    add_formula(e, "P(purchase) = 0.95 x (1 - ratio^0.7)", s)

    e.append(PageBreak())

    # ══════════════ 4. FARE CLASS OPTIMIZASYONU ══════════════
    e.append(Paragraph("4. Fare Class Optimizasyonu", s["H1"]))
    e.append(ColoredLine(W, 2, HIGHLIGHT))
    e.append(Spacer(1, 8))
    e.append(Paragraph(
        "Fare class availability uc katmanda belirlenir. Pricing engine DTD + LF kurallarini uygular, "
        "EMSR-b forecast-informed dinamik koruma ekler.", s["BodyText2"]))

    e.append(Paragraph("4.1 DTD Kurallari (Katman 1)", s["H2"]))
    dtd_rules = [
        ["DTD Araligi", "Acik Siniflar", "Etiket"],
        ["60 — 999", "V, K, M", "Erken Donem"],
        ["30 — 59", "K, M", "Orta Donem"],
        ["14 — 29", "K, M, Y", "Gec Donem"],
        ["7 — 13", "M, Y", "Son Hafta"],
        ["0 — 6", "Y", "Son Dakika"],
    ]
    e.append(make_table(dtd_rules, col_widths=[120, 180, 160]))

    e.append(Paragraph("4.2 EMSR-b Matematiksel Formulu (Katman 3)", s["H2"]))
    e.append(Paragraph("Expected Marginal Seat Revenue — heuristic b (Belobaba, 1989):", s["BodyText2"]))
    e.append(Paragraph("Her fare class cifti (yuksek/dusuk) icin koruma seviyesi:", s["BodyText2"]))
    add_formula(e, "ratio = price(low) / price(high)", s)
    add_formula(e, "z = &Phi;^(-1)(1 - ratio)&nbsp;&nbsp;&nbsp;[inverse normal CDF]", s)
    add_formula(e, "protection(high) = mean_d(high) + std_d(high) x z", s)

    e.append(Paragraph("Talep dagilimlari (forecast-informed):", s["H3"]))
    add_formula(e, "demand_base = min(TFT_total, capacity x 1.1)", s)
    add_formula(e, "mean_d(fc) = demand_base x class_share(fc)", s)
    add_formula(e, "std_d(fc) = max(mean_d(fc) x 0.4, 1.0)", s)

    e.append(Paragraph("Kota hesabi:", s["H3"]))
    add_formula(e, "quota(fc) = remaining - cumulative_protection(fc ve ustundekiler)", s)

    e.append(Paragraph(
        "EMSR-b sadece indirimli siniflari (V, K) kapatabilir. M ve Y pricing engine'e birakilir. "
        "Bu, M kapandiginda Y'nin yuksek fiyati nedeniyle koltuk bos kalmasini onler.",
        s["BodyText2"]))

    e.append(Paragraph("4.3 Talep Basinci Ayarlari", s["H2"]))
    add_formula(e, "demand_pressure = supply_mult x demand_mult", s)
    e.append(bullet("demand_pressure &gt; 1.3 → en ucuz acik sinif kapanir", s))
    e.append(bullet("demand_pressure &lt; 0.6 → bir alt sinif acilir", s))

    e.append(PageBreak())

    # ══════════════ 5. O&D NETWORK ══════════════
    e.append(Paragraph("5. O&amp;D Network Optimizasyonu", s["H1"]))
    e.append(ColoredLine(W, 2, HIGHLIGHT))
    e.append(Spacer(1, 8))

    e.append(Paragraph("5.1 Connecting Passenger Yonetimi", s["H2"]))
    add_formula(e, "conn_pct = max(route_connecting_pct, 0.10)&nbsp;&nbsp;&nbsp;[min %10 hub etkisi]", s)
    add_formula(e, "local_demand = total x (1 - conn_pct)", s)
    add_formula(e, "connecting_demand = total x conn_pct", s)

    e.append(Paragraph("5.2 Fare Proration (Ucret Dagitimi)", s["H2"]))
    add_formula(e, "Economy: base(leg) = max(distance_km x 0.08, 150)", s)
    add_formula(e, "total_fare = (origin_base + dest_base) x 0.85&nbsp;&nbsp;&nbsp;[%15 connecting indirimi]", s)
    add_formula(e, "leg_contribution = total_fare x dest_dist / total_dist", s)

    e.append(Paragraph("5.3 Bid Price (Minimum Kabul Fiyati)", s["H2"]))
    add_formula(e, "bid_price = current_prices[en ucuz acik fare class]", s)
    e.append(Paragraph("Sadece LF &gt; %60 sonrasi aktif. Erken donemde herkes kabul edilir.", s["BodyText2"]))

    e.append(Paragraph("5.4 Kabul/Red Karari", s["H2"]))
    od_data = [
        ["Yolcu Tipi", "Kabul Kosulu"],
        ["Lokal", "fare >= bid_price"],
        ["Connecting", "leg_contribution >= bid_price"],
    ]
    e.append(make_table(od_data, col_widths=[200, 260]))

    e.append(PageBreak())

    # ══════════════ 6. SIMULASYON MOTORU ══════════════
    e.append(Paragraph("6. Simulasyon Motoru", s["H1"]))
    e.append(ColoredLine(W, 2, HIGHLIGHT))
    e.append(Spacer(1, 8))

    e.append(Paragraph("6.1 SimClock (Zaman Sistemi)", s["H2"]))
    add_formula(e, "sim_elapsed = real_elapsed x speed", s)
    speed_data = [
        ["Speed", "1 Gun Suresi", "180 Gun Suresi", "Mod"],
        ["1,440", "42ms", "7.5s", "Normal"],
        ["4,320", "14ms", "2.5s", "Hizli"],
        ["14,400", "4ms", "0.7s", "Ultra"],
    ]
    e.append(make_table(speed_data, col_widths=[80, 115, 115, 150]))

    e.append(Paragraph("6.2 Gunluk Dongu", s["H2"]))
    e.append(Paragraph("Her simulasyon gunu icin sirayla:", s["BodyText2"]))
    steps = [
        "ForecastBridge.predict_daily_batch() — ML tahminleri",
        "_process_cancellations() — mevcut bookingleri tara",
        "Her ucus icin: _generate_daily_bots() + _update_prices()",
    ]
    for i, step in enumerate(steps, 1):
        e.append(bullet(f"<b>Adim {i}:</b> {step}", s))

    e.append(Paragraph("6.3 Overbooking", s["H2"]))
    add_formula(e, "sell_limit = capacity x 1.08", s)
    e.append(Paragraph("Agirlikli no-show ortalamasi ~%8 ile kalibre edilmistir.", s["BodyText2"]))

    e.append(Paragraph("6.4 Cancellation (Iptal) Modeli", s["H2"]))
    add_formula(e, "daily_cancel_prob = CANCEL_RATE(fc) / 180", s)
    cancel_data = [
        ["Fare Class", "Toplam Iptal Orani", "Gunluk Oran", "Refund Orani"],
        ["V — Promosyon", "%1", "0.0056%", "%0"],
        ["K — Indirimli", "%3", "0.0167%", "%50"],
        ["M — Esnek", "%8", "0.0444%", "%80"],
        ["Y — Tam Fiyat", "%12", "0.0667%", "%100"],
    ]
    e.append(make_table(cancel_data, col_widths=[100, 110, 100, 150]))
    add_formula(e, "refund = price x REFUND_RATE(fc)", s)

    e.append(Paragraph("6.5 Departure (No-Show + Denied Boarding)", s["H2"]))
    noshow_data = [
        ["Segment", "Profil", "No-Show Orani"],
        ["A — Is Yolcusu", "Plan degisir, yuksek no-show", "%15"],
        ["B — Erken Planlayan", "Plan kesin, dusuk no-show", "%5"],
        ["C — Tatilci", "Orta risk", "%8"],
        ["D — Ogrenci", "Butce sinirli, plan kesin", "%3"],
        ["E — Butce Hassas", "Orta risk", "%7"],
        ["F — Son Dakikaci", "Impulsif, en yuksek no-show", "%20"],
        ["HUMAN", "Gercek kullanici", "%2"],
    ]
    e.append(make_table(noshow_data, col_widths=[120, 210, 130]))
    add_formula(e, "denied = max(showed_up - capacity, 0)", s)
    add_formula(e, "denied_boarding_cost = denied x $400/pax", s)

    e.append(PageBreak())

    # ══════════════ 7. SENTIMENT ══════════════
    e.append(Paragraph("7. Sentiment Intelligence", s["H1"]))
    e.append(ColoredLine(W, 2, HIGHLIGHT))
    e.append(Spacer(1, 8))

    e.append(Paragraph("7.1 Pipeline", s["H2"]))
    sent_steps = [
        "Google News RSS + GDELT API → 51 destinasyon sehri icin haber toplama",
        "14-gun recency filtresi (MAX_AGE_HOURS = 336)",
        "DeBERTa NLI siniflandirma → pozitif / negatif / notr",
        "composite_score = (positive - negative) / total",
    ]
    for i, step in enumerate(sent_steps, 1):
        e.append(bullet(f"<b>{i}.</b> {step}", s))

    e.append(Paragraph("7.2 Cift Etki", s["H2"]))
    sent_eff = [
        ["Etki Alani", "Formul", "Aralik"],
        ["Fiyat", "1.0 + score x 0.15", "+/- %15"],
        ["Talep", "1.0 + score x 0.30", "+/- %30"],
    ]
    e.append(make_table(sent_eff, col_widths=[120, 200, 140]))
    e.append(Paragraph("Talep etkisi fiyat etkisinin 2 kati — negatif haberlerde once talep duser, fiyat takip eder.", s["SmallNote"]))

    e.append(PageBreak())

    # ══════════════ 8. PERFORMANS ══════════════
    e.append(Paragraph("8. Performans Metrikleri", s["H1"]))
    e.append(ColoredLine(W, 2, HIGHLIGHT))
    e.append(Spacer(1, 8))

    e.append(Paragraph("8.1 Model Performanslari", s["H2"]))
    model_perf = [
        ["Model", "MAE", "Diger Metrik", "Improvement"],
        ["TFT", "14.03", "Corr = 0.991", "—"],
        ["XGBoost Pickup", "3.45", "WAPE = 9.82%", "%70.4 vs baseline"],
        ["Two-Stage XGB", "0.78", "AUC = 0.835", "—"],
    ]
    e.append(make_table(model_perf, col_widths=[120, 70, 150, 120]))

    e.append(Paragraph("8.2 Simulasyon Sonuclari (IST-LHR, Temmuz 1-15, Economy, 15 ucus)", s["H2"]))
    sim_res = [
        ["Metrik", "Deger"],
        ["Revenue Delta", "+2.8% ($16,451)"],
        ["Mutlak Gelir (Dinamik)", "$607,931"],
        ["Mutlak Gelir (Baseline)", "$591,480"],
        ["Ortalama Load Factor", "%69.9"],
        ["Toplam Satilan / Kapasite", "3,147 / 4,500"],
        ["Iptal (Cancellation)", "72"],
        ["No-Show", "125"],
        ["Denied Boarding", "0"],
        ["Fare Mix: V", "%16.5"],
        ["Fare Mix: K", "%36.7"],
        ["Fare Mix: M", "%48.7"],
        ["Fare Mix: Y", "%0.3"],
    ]
    e.append(make_table(sim_res, col_widths=[230, 230]))

    e.append(Paragraph("8.3 EMSR-b Etkisi", s["H2"]))
    emsr_eff = [
        ["Senaryo", "Delta %", "LF", "Mutlak Gelir"],
        ["EMSR-b oncesi (heuristic)", "+10.5%", "67.3%", "$648,764"],
        ["EMSR-b + heuristic", "+9.3%", "69.0%", "$667,657"],
        ["EMSR-b + calibrated", "+2.8%", "69.9%", "$607,931"],
    ]
    e.append(make_table(emsr_eff, col_widths=[160, 80, 80, 140]))
    e.append(Paragraph(
        "Not: Kalibre edilmis katsayilarla delta % duser cunku katsayilar daha muhafazakar fiyat uretir. "
        "Ancak LF en yuksek degerine ulasmistir (%69.9) ve fare class mix sagliklidir.",
        s["SmallNote"]))

    e.append(PageBreak())

    # ══════════════ 9. KALIBRASYON ══════════════
    e.append(Paragraph("9. Kalibrasyon Metodolojisi", s["H1"]))
    e.append(ColoredLine(W, 2, HIGHLIGHT))
    e.append(Spacer(1, 8))
    e.append(Paragraph(
        "Pricing engine katsayilari, historik ucus verisi uzerinde istatistiksel analiz ve "
        "regresyon ile ogrenilmistir. Bu yaklasim, elle belirlenmis (heuristic) katsayilara "
        "kiyasla veriye dayali ve dogrulanabilir bir fiyatlandirma altyapisi saglar.",
        s["BodyText2"]))

    e.append(Paragraph("9.1 Kalibrasyon Pipeline", s["H2"]))
    cal_steps = [
        "<b>Veri:</b> flight_snapshot_v2.parquet + flight_metadata.parquet (146K kayit, 200 entity)",
        "<b>Baz Fiyat:</b> Lineer regresyon (ticket_rev/pax ~ distance_km), R&sup2; = 0.979",
        "<b>Sezon/DOW:</b> DTD-kontrollü ortalama fiyat oranlari (DTD 30-60 grubu)",
        "<b>DTD Curve:</b> DTD bucket bazli ortalama gunluk fiyat, baseline = DTD 31-60",
        "<b>LF Curve:</b> Load factor bazli ortalama fiyat, baseline = LF &lt; 30%",
        "<b>Route Factors:</b> Mesafe etkisi cikarildiktan sonra rota spesifik residual (100 rota)",
        "<b>Region Factors:</b> Bolge bazli ortalama fiyat orani",
    ]
    for step in cal_steps:
        e.append(bullet(step, s))

    e.append(Paragraph("9.2 Ogrenilen Katsayilar Ozeti", s["H2"]))
    cal_summary = [
        ["Katsayi", "Yontem", "n", "Kalite"],
        ["base_price (economy)", "LinReg: 4.01 + dist x 0.081", "102,200", "R2 = 0.979"],
        ["base_price (business)", "LinReg: 100.12 + dist x 0.332", "102,200", "R2 = 0.963"],
        ["season_factors (12 ay)", "Oran analizi (DTD-controlled)", "~3M", "CV < 2.5%"],
        ["dow_factors (7 gun)", "Oran analizi (DTD-controlled)", "~3M", "CV < 0.3%"],
        ["dtd_curve (8 bucket)", "Bucket ortalama", "~37M", "Monoton artan"],
        ["lf_curve (6 bucket)", "Bucket ortalama", "~37M", "Monoton artan"],
        ["route_factors (100 rota)", "Residual analizi", "102,200", "Range: 0.87-2.19"],
        ["region_factors (5 bolge)", "Oran analizi", "102,200", "Range: 0.39-2.04"],
    ]
    e.append(make_table(cal_summary, col_widths=[120, 150, 60, 130]))

    e.append(Paragraph("9.3 Kalibrasyon Dosyasi", s["H2"]))
    e.append(Paragraph(
        "Tum ogrenilmis katsayilar <font face='Courier'>reports/calibration_report.json</font> dosyasinda "
        "saklanir. Pricing engine baslatildiginda bu dosyayi yukler. Dosya bulunmazsa fallback "
        "(elle belirlenmis) katsayilar kullanilir.", s["BodyText2"]))

    e.append(PageBreak())

    # ══════════════ 10. TEKNOLOJI ══════════════
    e.append(Paragraph("10. Teknoloji Yigini", s["H1"]))
    e.append(ColoredLine(W, 2, HIGHLIGHT))
    e.append(Spacer(1, 8))
    tech = [
        ["Katman", "Teknoloji"],
        ["Backend", "Python 3.13, Flask"],
        ["Veritabani", "DuckDB + Apache Parquet"],
        ["ML — TFT", "PyTorch, pytorch-forecasting"],
        ["ML — XGBoost", "xgboost 2.x, joblib"],
        ["ML — SHAP", "shap 0.51 (TreeExplainer)"],
        ["NLP", "HuggingFace transformers, DeBERTa-v3"],
        ["Optimizasyon", "scipy.stats (EMSR-b inverse normal CDF)"],
        ["Frontend", "Vanilla JS, CSS Grid, Chart.js"],
        ["Veri Toplama", "GDELT API, Google News RSS, feedparser"],
    ]
    e.append(make_table(tech, col_widths=[130, 330]))

    # BUILD
    doc.build(e)
    print(f"[OK] Teknik Mimari Raporu: {path}")
    return path


# ═══════════════════════════════════════════════════════════════
# RAPOR 2: FIYATLANDIRMA ADIM ADIM
# ═══════════════════════════════════════════════════════════════

def build_pricing_report():
    path = os.path.join(DESKTOP, "SeatWise_Fiyatlandirma_Adim_Adim.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2.5*cm, bottomMargin=2*cm)
    s = get_styles()
    e = []
    W = 460

    # ── KAPAK ──
    e.append(Spacer(1, 60))
    e.append(ColoredLine(W, 3, HIGHLIGHT))
    e.append(Spacer(1, 12))
    e.append(Paragraph("SeatWise", s["DocTitle"]))
    e.append(Paragraph("Bir Fiyatlandirma Kararinin A'dan Z'ye Anatomisi<br/>Ornek Senaryo ile Adim Adim Cozum", s["DocSubtitle"]))
    e.append(Spacer(1, 20))
    e.append(ColoredLine(W, 1, ACCENT))
    e.append(Spacer(1, 16))
    e.append(Paragraph("Ahmet Furkan Gokbulut", s["H3"]))
    e.append(Paragraph("Kadir Has Universitesi — Endustri Muhendisligi &amp; Bilgisayar Muhendisligi", s["BodyText2"]))
    e.append(Paragraph("Mart 2026", s["SmallNote"]))
    e.append(PageBreak())

    # ── SENARYO ──
    e.append(Paragraph("Senaryo Tanimi", s["H1"]))
    e.append(ColoredLine(W, 2, HIGHLIGHT))
    e.append(Spacer(1, 8))
    e.append(Paragraph("Asagidaki senaryoyu her denklemi calistirarak, adim adim cozelim:", s["BodyText2"]))
    scenario = [
        ["Parametre", "Deger"],
        ["Ucus", "IST-LHR (Istanbul → Londra Heathrow)"],
        ["Tarih", "10 Temmuz 2026 (Cuma)"],
        ["Kabin", "Economy"],
        ["Kapasite", "300 koltuk"],
        ["Satilan", "185 koltuk (LF = %61.7)"],
        ["Mesafe", "2,499 km"],
        ["Bolge", "Europe"],
        ["DTD", "45 gun"],
        ["Fare Class Satislar", "V=45, K=75, M=63, Y=2"],
        ["Two-Stage daily_demand", "4.2 yolcu/gun"],
        ["Pickup predicted_remaining", "68"],
        ["TFT total prediction", "240"],
        ["Sentiment (Londra)", "+0.12 (kultur festivali)"],
        ["Musteri Segmenti", "C — Tatilci (WTP: 0.85-1.35)"],
    ]
    e.append(make_table(scenario, col_widths=[170, 290]))

    e.append(PageBreak())

    # ── ADIM 1 ──
    e.append(Paragraph("Adim 1: Baz Fiyat Hesaplama", s["H1"]))
    e.append(ColoredLine(W, 2, INFO_BLUE))
    e.append(Spacer(1, 8))
    e.append(Paragraph("Veriden ogrenilmis regresyon formulu (R&sup2; = 0.979, n = 102,200):", s["BodyText2"]))
    add_formula(e, "base = intercept + distance_km x price_per_km", s)
    add_formula(e, "base = 4.01 + 2,499 x 0.0810 = 4.01 + 202.42 = 206.43", s)
    e.append(Paragraph("Rota faktoru (IST_LHR, veriden ogrenilmis residual):", s["BodyText2"]))
    add_formula(e, "base_price = 206.43 x route_factor(IST_LHR) = <b>~$223</b>", s)

    # ── ADIM 2 ──
    e.append(Paragraph("Adim 2: Arz Carpani (Supply Multiplier)", s["H1"]))
    e.append(ColoredLine(W, 2, INFO_BLUE))
    e.append(Spacer(1, 8))
    e.append(Paragraph("Pickup modeli predicted_remaining = 68 vermis:", s["BodyText2"]))
    add_formula(e, "expected_final_pax = 185 + 68 = 253", s)
    add_formula(e, "expected_final_lf = 253 / 300 = 0.8433 = %84.3", s)
    e.append(Paragraph("%84.3 >= %70 ama &lt; %85 → <b>lf_mult = 1.15</b>", s["BodyText2"]))
    e.append(Paragraph("DTD = 45 &gt; 14 → <b>dtd_boost = 1.00</b>", s["BodyText2"]))
    add_formula(e, "supply_multiplier = 1.15 x 1.00 = <b>1.15</b>", s)

    # ── ADIM 3 ──
    e.append(Paragraph("Adim 3: Talep Carpani (Demand Multiplier)", s["H1"]))
    e.append(ColoredLine(W, 2, INFO_BLUE))
    e.append(Spacer(1, 8))
    comp = [
        ["Bilesen", "Kaynak", "Deger"],
        ["Sezon", "Temmuz (ay=7)", "1.30"],
        ["Ozel Gun", "10 Temmuz — normal gun", "1.00"],
        ["Hafta Gunu", "Cuma (weekday=4)", "1.15"],
    ]
    e.append(make_table(comp, col_widths=[120, 180, 160]))
    add_formula(e, "raw = 1.30 x 1.00 x 1.15 = 1.495", s)
    add_formula(e, "demand_multiplier = 1.0 + (1.495 - 1.0) x 0.7 = 1.0 + 0.3465 = <b>1.3465</b>", s)

    # ── ADIM 4 ──
    e.append(Paragraph("Adim 4: Sentiment Carpani", s["H1"]))
    e.append(ColoredLine(W, 2, INFO_BLUE))
    e.append(Spacer(1, 8))
    e.append(Paragraph("Londra composite_score = +0.12 (kultur festivali haberleri):", s["BodyText2"]))
    add_formula(e, "sentiment_multiplier = 1.0 + 0.12 x 0.15 = <b>1.018</b>", s)

    # ── ADIM 5 ──
    e.append(Paragraph("Adim 5: Musteri Carpani", s["H1"]))
    e.append(ColoredLine(W, 2, INFO_BLUE))
    e.append(Spacer(1, 8))
    e.append(Paragraph("Segment C (Tatilci), WTP = {min: 0.85, max: 1.35}:", s["BodyText2"]))
    add_formula(e, "wtp_avg = (0.85 + 1.35) / 2 = 1.10", s)
    add_formula(e, "customer_mult = 0.85 + (1.10 - 0.5) x 0.2 = 0.85 + 0.12 = <b>0.97</b>", s)

    e.append(PageBreak())

    # ── ADIM 6 ──
    e.append(Paragraph("Adim 6: Bilesik Carpan ve Fiyat Hesaplama", s["H1"]))
    e.append(ColoredLine(W, 2, INFO_BLUE))
    e.append(Spacer(1, 8))
    add_formula(e, "combined = supply x demand x sentiment x customer", s)
    add_formula(e, "combined = 1.15 x 1.3465 x 1.018 x 0.97 = <b>1.5291</b>", s)

    e.append(Paragraph("Her fare class icin fiyat:", s["H2"]))
    prices_data = [
        ["Fare Class", "Formul", "Fiyat"],
        ["V — Promosyon", "223.91 x 0.50 x 1.5291", "$171.19"],
        ["K — Indirimli", "223.91 x 0.75 x 1.5291", "$256.79"],
        ["M — Esnek", "223.91 x 1.00 x 1.5291", "$342.38"],
        ["Y — Tam Fiyat", "223.91 x 1.50 x 1.5291", "$513.57"],
    ]
    e.append(make_table(prices_data, col_widths=[110, 200, 150]))

    # ── ADIM 7 ──
    e.append(Paragraph("Adim 7: Fare Class Availability", s["H1"]))
    e.append(ColoredLine(W, 2, INFO_BLUE))
    e.append(Spacer(1, 8))

    e.append(Paragraph("<b>Katman 1 — DTD Kurali:</b> DTD=45 → kural 30-59 → base_open = [K, M]", s["BodyText2"]))
    e.append(Paragraph("<b>Katman 2 — LF Filtresi:</b> LF = %61.7", s["BodyText2"]))
    lf_check = [
        ["Sinif", "open_until_lf", "LF (61.7%)", "Sonuc"],
        ["K", "60%", "61.7% >= 60%", "KAPALI"],
        ["M", "85%", "61.7% < 85%", "ACIK"],
    ]
    e.append(make_table(lf_check, col_widths=[60, 100, 140, 160]))
    e.append(Paragraph("Pricing Engine sonucu: <b>open_fares = [M]</b>", s["BodyText2"]))

    e.append(Paragraph("<b>Katman 3 — EMSR-b Override:</b>", s["BodyText2"]))
    e.append(Paragraph("TFT demand = 240, capacity = 300:", s["BodyText2"]))
    add_formula(e, "expected_demand = min(240, 300 x 1.1) = 240", s)
    emsr_calc = [
        ["Sinif Cifti", "ratio", "z = &Phi;^(-1)(1-ratio)", "protection"],
        ["Y vs M", "342/514 = 0.667", "&Phi;^(-1)(0.333) = -0.431", "36.0 + 14.4 x (-0.431) = 30"],
        ["M vs K", "257/342 = 0.750", "&Phi;^(-1)(0.250) = -0.675", "52.8 + 21.1 x (-0.675) = 39"],
        ["K vs V", "171/257 = 0.667", "&Phi;^(-1)(0.333) = -0.431", "67.2 + 26.9 x (-0.431) = 56"],
    ]
    e.append(make_table(emsr_calc, col_widths=[80, 110, 140, 130]))

    e.append(Paragraph("Kota hesabi (remaining = 300 - 185 = 115):", s["BodyText2"]))
    quota_data = [
        ["Sinif", "Protection", "Kota", "Satilan", "EMSR-b Karar"],
        ["V", "0", "0", "45", "KAPAT (45 >= 0)"],
        ["K", "cum 69", "46", "75", "KAPAT (75 >= 46)"],
        ["M", "cum 30", "85", "63", "(EMSR-b devre disi)"],
        ["Y", "0", "115", "2", "(her zaman acik)"],
    ]
    e.append(make_table(quota_data, col_widths=[50, 80, 60, 70, 200]))
    e.append(Paragraph("<b>Final: open_fares = [M]</b> — DTD, LF ve EMSR-b uc katman da ayni karari vermis.", s["BodyText2"]))

    e.append(PageBreak())

    # ── ADIM 8 ──
    e.append(Paragraph("Adim 8: Baseline (Statik) Fiyat Karsilastirmasi", s["H1"]))
    e.append(ColoredLine(W, 2, INFO_BLUE))
    e.append(Spacer(1, 8))
    add_formula(e, "base = max(2,499 x 0.08, 150) = 199.92", s)
    add_formula(e, "dtd_f(45) = 0.95&nbsp;&nbsp;|&nbsp;&nbsp;season_f(Jul) = 1.30&nbsp;&nbsp;|&nbsp;&nbsp;region_f(Europe) = 1.00", s)
    add_formula(e, "baseline = 199.92 x 0.95 x 1.30 x 1.00 = <b>$246.90</b>", s)

    # ── ADIM 9 ──
    e.append(Paragraph("Adim 9: Bot Satisa Gitme Karari", s["H1"]))
    e.append(ColoredLine(W, 2, INFO_BLUE))
    e.append(Spacer(1, 8))

    e.append(Paragraph("<b>O&amp;D Bid Price Kontrolu:</b> LF = %61.7 &gt; %60 → aktif", s["BodyText2"]))
    add_formula(e, "bid_price = current_prices['M'] = $342.38", s)
    e.append(Paragraph("Lokal yolcu: fare ($342.38) >= bid ($342.38) → <b>KABUL</b>", s["BodyText2"]))

    e.append(Paragraph("<b>WTP Kontrolu</b> (Segment C, WTP min=0.85, max=1.35):", s["H3"]))
    e.append(Paragraph("<b>Senaryo A:</b> personal_wtp = 1.12:", s["BodyText2"]))
    add_formula(e, "max_willing = 223.91 x 1.12 = $250.78", s)
    e.append(Paragraph("M = $342.38 &gt; $250.78 → <b>REDDEDILDI</b> (WTP yetersiz)", s["BodyText2"]))

    e.append(Paragraph("<b>Senaryo B:</b> Segment A (Is Yolcusu), WTP = 2.15:", s["BodyText2"]))
    add_formula(e, "max_willing = 223.91 x 2.15 = $481.41", s)
    e.append(Paragraph("M = $342.38 &lt; $481.41 → fare class uygun!", s["BodyText2"]))

    e.append(Paragraph("<b>Satin Alma Olasiligi:</b>", s["H3"]))
    add_formula(e, "wtp_max_price = 2.50 x 223.91 = $559.78", s)
    add_formula(e, "wtp_min_price = 1.50 x 223.91 = $335.87", s)
    add_formula(e, "ratio = (342.38 - 335.87) / (559.78 - 335.87) = 6.51 / 223.91 = 0.0291", s)
    add_formula(e, "P(purchase) = 0.95 x (1 - 0.0291^0.7) = 0.95 x 0.936 = <b>%88.9</b>", s)
    e.append(Paragraph("random(0,1) = 0.34 &lt; 0.889 → <b>SATIS GERCEKLESTI!</b>", s["BodyText2"]))

    # ── ADIM 10 ──
    e.append(Paragraph("Adim 10: Satis Sonrasi Guncelleme", s["H1"]))
    e.append(ColoredLine(W, 2, INFO_BLUE))
    e.append(Spacer(1, 8))
    update_data = [
        ["Parametre", "Onceki", "Sonrasi"],
        ["sold", "185", "186"],
        ["load_factor", "61.7%", "62.0%"],
        ["fare_class_sold[M]", "63", "64"],
        ["revenue_dynamic", "+$342.38", ""],
        ["revenue_baseline", "+$246.90", ""],
    ]
    e.append(make_table(update_data, col_widths=[160, 150, 150]))

    e.append(Paragraph("Bu tek satisin delta katkisi:", s["H3"]))
    add_formula(e, "delta = $342.38 - $246.90 = <b>+$95.48 (+38.7%)</b>", s)

    e.append(PageBreak())

    # ── ADIM 11 ──
    e.append(Paragraph("Adim 11: Connecting Yolcu Senaryosu", s["H1"]))
    e.append(ColoredLine(W, 2, WARNING))
    e.append(Spacer(1, 8))
    e.append(Paragraph("Ayni ucus, ayni an — ama bu sefer JFK → IST → LHR connecting yolcu:", s["BodyText2"]))

    e.append(Paragraph("Fare Proration:", s["H2"]))
    add_formula(e, "origin_dist(JFK-IST) = 8,068 km", s)
    add_formula(e, "dest_dist(IST-LHR) = 2,499 km", s)
    add_formula(e, "total_dist = 10,567 km", s)
    add_formula(e, "origin_base = max(8,068 x 0.08, 150) = $645.44", s)
    add_formula(e, "dest_base = max(2,499 x 0.08, 150) = $199.92", s)
    add_formula(e, "total_fare = (645.44 + 199.92) x 0.85 = <b>$718.56</b>", s)
    add_formula(e, "leg_contribution = 718.56 x 2,499 / 10,567 = <b>$169.94</b>", s)

    e.append(Paragraph("Bid Price Kontrolu:", s["H2"]))
    add_formula(e, "bid_price = $342.38&nbsp;&nbsp;|&nbsp;&nbsp;leg_contribution = $169.94", s)
    add_formula(e, "$169.94 &lt; $342.38 → <b>CONNECTING REDDEDILDI</b> (displacement)", s)
    e.append(Paragraph(
        "Mantik: IST-LHR bacagi icin $169.94 katkida bulunan connecting yolcu yerine, "
        "$342.38 odeyecek lokal yolcuyu beklemek daha karli.", s["BodyText2"]))

    # ── ADIM 12 ──
    e.append(Paragraph("Adim 12: Iptal Senaryosu", s["H1"]))
    e.append(ColoredLine(W, 2, DANGER))
    e.append(Spacer(1, 8))
    add_formula(e, "CANCEL_RATE(M) = 0.08&nbsp;&nbsp;&nbsp;daily_rate = 0.08 / 180 = 0.000444", s)
    add_formula(e, "random(0,1) = 0.00032 &lt; 0.000444 → <b>IPTAL!</b>", s)
    add_formula(e, "REFUND_RATE(M) = 0.80", s)
    add_formula(e, "refund = $342.38 x 0.80 = <b>$273.90</b>", s)
    e.append(Paragraph("revenue_dynamic -= $273.90, sold: 186 → 185. Koltuk yeniden satisa acilir.", s["BodyText2"]))

    # ── ADIM 13 ──
    e.append(Paragraph("Adim 13: Kalkis Gunu — No-Show &amp; Denied Boarding", s["H1"]))
    e.append(ColoredLine(W, 2, DANGER))
    e.append(Spacer(1, 8))
    e.append(Paragraph("312 bilet satilmis (overbooking 300 x 1.08 = 324 limit, 12 iptal sonrasi 312):", s["BodyText2"]))
    noshow_ex = [
        ["Segment", "Kisi", "No-Show %", "Gelen", "Gelmeyen"],
        ["A — Is", "47", "%15", "~40", "~7"],
        ["B — Erken", "31", "%5", "~29", "~2"],
        ["C — Tatilci", "47", "%8", "~43", "~4"],
        ["D — Ogrenci", "94", "%3", "~91", "~3"],
        ["E — Butce", "47", "%7", "~44", "~3"],
        ["F — Son Dakika", "46", "%20", "~37", "~9"],
        ["Toplam", "312", "~%8.9", "~284", "~28"],
    ]
    e.append(make_table(noshow_ex, col_widths=[90, 55, 75, 65, 75]))
    add_formula(e, "showed_up (284) &lt; capacity (300) → <b>DENIED BOARDING = 0</b>", s)
    e.append(Paragraph("Overbooking dogru kalibre: 312 bilet, 28 no-show, 284 gelen &lt; 300 kapasite.", s["BodyText2"]))

    e.append(PageBreak())

    # ── OZET ──
    e.append(Paragraph("Sonuc Ozeti", s["H1"]))
    e.append(ColoredLine(W, 2, HIGHLIGHT))
    e.append(Spacer(1, 8))

    e.append(Paragraph("Fiyat Bilesenleri:", s["H2"]))
    summary_data = [
        ["Bilesen", "Deger", "Aciklama"],
        ["Baz Fiyat", "$223.91", "2,499km x 0.08 x 1.12 (rota faktoru)"],
        ["Arz Carpani", "1.15", "Pickup: %84 final LF beklentisi"],
        ["Talep Carpani", "1.3465", "Temmuz (1.30) + Cuma (1.15) + dampening"],
        ["Sentiment", "1.018", "Londra +0.12 (hafif pozitif)"],
        ["Musteri", "0.97", "Segment C tatilci"],
        ["Bilesik Carpan", "1.5291", "Tum carpanlar"],
        ["M Fiyat", "$342.38", "base x 1.00 x 1.5291"],
        ["Baseline", "$246.90", "Eski statik formul"],
        ["Delta", "+$95.48 (+38.7%)", "Dinamik fiyatlamanin katkisi"],
    ]
    e.append(make_table(summary_data, col_widths=[100, 110, 250]))

    e.append(Paragraph("Karar Zinciri:", s["H2"]))
    chain = [
        "<b>1.</b> TFT: 'Bu ucus 240 yolcu bekliyor'",
        "<b>2.</b> Pickup: '68 kisi daha gelecek, final LF %84'",
        "<b>3.</b> Two-Stage: 'Bugun 4.2 yolcu gelir'",
        "<b>4.</b> Pricing Engine: 'LF %84 → supply 1.15, Temmuz+Cuma → demand 1.35'",
        "<b>5.</b> EMSR-b: 'V kotasi doldu, K kotasi doldu → sadece M acik'",
        "<b>6.</b> Bid Price: 'M = $342 → bu fiyatin altinda koltuk verme'",
        "<b>7.</b> Bot (Segment A): 'WTP 2.15, M=$342 &lt; $481 → ALIYORUM'",
        "<b>8.</b> Satin alma olasiligi: %88.9 → SATIS!",
        "<b>9.</b> Delta: Baseline'dan +$95 fazla gelir",
    ]
    for line in chain:
        e.append(bullet(line, s))

    e.append(Spacer(1, 16))
    e.append(Paragraph(
        "Bu karar zinciri, tek bir koltuk icin 9 adimdan olusur. Simulasyon bunu "
        "180 gun x ~300 koltuk x 15 ucus = ~810,000 kez tekrarlar.",
        s["BodyText2"]))

    e.append(Spacer(1, 20))

    e.append(Paragraph("Ucus Yasam Dongusu:", s["H2"]))
    lifecycle = [
        ["DTD", "Olay", "Acik Siniflar"],
        ["180", "Booking penceresi acilir, Segment D/E agirlikli", "V, K, M"],
        ["~120", "V kotasi doluyor, EMSR-b V'yi kapatiyor", "K, M"],
        ["~60", "K kotasi doluyor, EMSR-b K'yi kapatiyor", "M"],
        ["30", "LF > %60 → bid price aktif, displacement baslar", "K, M"],
        ["14", "DTD kurali: gec donem, Segment A/F agirlikli", "M, Y"],
        ["7", "Son hafta, dtd_boost = 1.15", "M, Y"],
        ["3", "Son dakika, dtd_boost = 1.30", "Y"],
        ["0", "KALKIS — no-show + denied boarding", "—"],
    ]
    e.append(make_table(lifecycle, col_widths=[40, 270, 150]))

    # BUILD
    doc.build(e)
    print(f"[OK] Fiyatlandirma Adim Adim: {path}")
    return path


if __name__ == "__main__":
    p1 = build_technical_report()
    p2 = build_pricing_report()
    print(f"\nRaporlar olusturuldu:")
    print(f"  1. {p1}")
    print(f"  2. {p2}")
