"""
Bölüm 2.8.1.10 — Simulation Environment (Türkçe akademik teknik rapor PDF).
Çıktı: <Masaüstü>/Bolum_2_6_Simulation_Modulu_Tr.pdf

Tüm akademik iddialar, projedeki gerçek `dashboard/simulation_engine.py`
implementasyonuyla birebir uyumludur. IEEE-stili in-text [#] cite'lar ve
sondaki kaynakça birbirini kapsar; numaralar metinde ilk göründükleri
sıraya göre verilmiştir.
"""
import os
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

# ── Türkçe-uyumlu Times font ailesi ─────────────────────────────
WIN_FONTS = "C:/Windows/Fonts"
pdfmetrics.registerFont(TTFont("TR-Roman",      f"{WIN_FONTS}/times.ttf"))
pdfmetrics.registerFont(TTFont("TR-Bold",       f"{WIN_FONTS}/timesbd.ttf"))
pdfmetrics.registerFont(TTFont("TR-Italic",     f"{WIN_FONTS}/timesi.ttf"))
pdfmetrics.registerFont(TTFont("TR-BoldItalic", f"{WIN_FONTS}/timesbi.ttf"))
registerFontFamily("TR-Roman", normal="TR-Roman", bold="TR-Bold",
                    italic="TR-Italic", boldItalic="TR-BoldItalic")

F_NORMAL, F_BOLD, F_ITALIC, F_BOLDITALIC = "TR-Roman", "TR-Bold", "TR-Italic", "TR-BoldItalic"

DESKTOP = Path(os.path.expanduser("~")) / "OneDrive" / "Desktop"
OUT = DESKTOP / "Bolum_2_6_Simulation_Modulu_Tr.pdf"

# ── Stiller ─────────────────────────────────────────────────────
styles = getSampleStyleSheet()
ACCENT = HexColor("#0b3d91")
GREY = HexColor("#444444")

S = {}
S["Title"] = ParagraphStyle("Title", parent=styles["Title"], fontName=F_BOLD,
                             fontSize=18, leading=22, alignment=TA_CENTER,
                             textColor=ACCENT, spaceAfter=4)
S["Subtitle"] = ParagraphStyle("Subtitle", parent=styles["Normal"], fontName=F_ITALIC,
                                fontSize=11, leading=14, alignment=TA_CENTER,
                                textColor=GREY, spaceAfter=8)
S["Author"] = ParagraphStyle("Author", parent=styles["Normal"], fontName=F_NORMAL,
                              fontSize=10.5, leading=13, alignment=TA_CENTER, spaceAfter=2)
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
                          fontSize=14, leading=17, textColor=ACCENT,
                          spaceBefore=14, spaceAfter=8, keepWithNext=1)
S["H2"] = ParagraphStyle("H2", parent=styles["Heading2"], fontName=F_BOLD,
                          fontSize=12, leading=15, textColor=ACCENT,
                          spaceBefore=12, spaceAfter=5, keepWithNext=1)
S["H3"] = ParagraphStyle("H3", parent=styles["Heading3"], fontName=F_BOLD,
                          fontSize=10.5, leading=13, textColor=black,
                          spaceBefore=8, spaceAfter=3, keepWithNext=1)
S["Body"] = ParagraphStyle("Body", parent=styles["Normal"], fontName=F_NORMAL,
                            fontSize=10.5, leading=14, alignment=TA_JUSTIFY,
                            spaceAfter=6, firstLineIndent=14)
S["Equation"] = ParagraphStyle("Eq", parent=styles["Normal"], fontName=F_ITALIC,
                                fontSize=11, leading=14, alignment=TA_CENTER,
                                spaceBefore=4, spaceAfter=8, textColor=black)
S["Caption"] = ParagraphStyle("Caption", parent=styles["Normal"], fontName=F_ITALIC,
                               fontSize=9, leading=11, alignment=TA_CENTER,
                               textColor=GREY, spaceBefore=2, spaceAfter=10)
S["Ref"] = ParagraphStyle("Ref", parent=styles["Normal"], fontName=F_NORMAL,
                           fontSize=9, leading=11.5, alignment=TA_LEFT,
                           leftIndent=24, firstLineIndent=-24, spaceAfter=4)
S["FigBox"] = ParagraphStyle("FigBox", parent=styles["Normal"], fontName=F_NORMAL,
                              fontSize=9.5, leading=13, alignment=TA_CENTER,
                              spaceBefore=6, spaceAfter=2, leftIndent=18,
                              rightIndent=18, textColor=black)


def P(t, st="Body"):
    return Paragraph(t, S[st])


def EQ(t):
    return Paragraph(t, S["Equation"])


def TABLE(data, col_widths=None, header=True):
    cell_style = ParagraphStyle("TblCell", parent=styles["Normal"],
                                 fontName=F_NORMAL, fontSize=9.5, leading=12,
                                 alignment=TA_LEFT, firstLineIndent=0, spaceAfter=0)
    head_style = ParagraphStyle("TblHead", parent=cell_style, fontName=F_BOLD)
    pdata = []
    for i, row in enumerate(data):
        st = head_style if (header and i == 0) else cell_style
        prow = [Paragraph(c, st) if isinstance(c, str) else c for c in row]
        pdata.append(prow)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, black),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    return Table(pdata, colWidths=col_widths, hAlign="CENTER",
                 repeatRows=1 if header else 0, style=TableStyle(style))


def on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFont(F_ITALIC, 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, h - 1.2 * cm, "2.8.1.10 Simulation and Competition Panels")
    canvas.drawRightString(w - 2 * cm, h - 1.2 * cm, "Seatwise / FENS 402 Group 16")
    canvas.line(2 * cm, h - 1.3 * cm, w - 2 * cm, h - 1.3 * cm)
    canvas.setFont(F_NORMAL, 8.5)
    canvas.drawCentredString(w / 2.0, 1.2 * cm, f"— {doc.page} —")
    canvas.restoreState()


# ── Screenshot placeholder yardımcısı ──────────────────────────
S["Shot"] = ParagraphStyle("Shot", parent=styles["Normal"], fontName=F_ITALIC,
                            fontSize=9.5, leading=13, alignment=TA_CENTER,
                            textColor=GREY, leftIndent=18, rightIndent=18,
                            spaceBefore=10, spaceAfter=2,
                            borderWidth=0.7, borderColor=HexColor("#bbbbbb"),
                            borderPadding=14, backColor=HexColor("#f6f7fa"))


def SHOT(num, title):
    return Paragraph(
        f"&#x1F4F7; <b>Şekil&nbsp;{num}</b> &mdash; {title}<br/>"
        f"<font size='8' color='#888888'>(Buraya ekran görüntüsü yerleştirilecektir)</font>",
        S["Shot"])


# ── İçerik ──────────────────────────────────────────────────────
flow = []

flow.append(Spacer(1, 8))
flow.append(P("2.8.1.10 Simulation and Competition Panels", "Title"))
flow.append(P("Dashboard Üzerinde Dinamik Fiyatlandırmanın Canlı "
              "Değerlendirmesi", "Subtitle"))
flow.append(Spacer(1, 4))
flow.append(P("Group 16 &mdash; FENS 402 Engineering Design Project II", "Author"))
flow.append(P("Endüstri Mühendisliği Bölümü &mdash; Kadir Has Üniversitesi &mdash; Mayıs 2026",
              "Affiliation"))

# ─── ÖZET ─────────────────────────────────────────────────────
flow.append(P("Özet", "AbstractHead"))
flow.append(P(
    "Bu bölüm, Seatwise dashboard&rsquo;ında bulunan <b>Simulation</b> ve "
    "<b>Competition</b> panellerinin tasarımını, kullanıcı arayüzünü ve "
    "çalışma prensibini açıklamaktadır. Simulation paneli, tezde geliştirilen "
    "tüm modülleri (talep tahmini, fiyatlandırma motoru, sentiment, rakip "
    "analizi, ağ optimizasyonu) tek bir interaktif ortamda bir araya "
    "getirir; operatör tek bir ekrandan dinamik fiyatlandırma motorunun "
    "180 günlük booking yaşam döngüsündeki davranışını canlı izler ve "
    "statik yield management baseline&rsquo;ı [13] ile karşılaştırır. Hız "
    "kontrolü sayesinde aynı senaryo birkaç saniyede tamamlanabilir veya "
    "ayrıntılı incelenebilir; her koltuk tıklanabilir, fiyat oluşumu "
    "&ldquo;Pricing Decision Analysis&rdquo; modal&rsquo;inde adım adım "
    "görselleştirilir. Competition paneli ise rakip havayollarının fiyat "
    "hareketlerini eş zamanlı gösterir. Bu rapor, panellerin matematiksel "
    "iç işleyişini değil, <i>kullanıcının deneyimlediği akışı</i> ve "
    "panelin <i>hangi modülleri nasıl bir araya getirdiğini</i> belgeler.",
    "Abstract"))

# ═════════════════════════════════════════════════════════════════
# 2.8.1.10 SIMULATION AND COMPETITION PANELS
# ═════════════════════════════════════════════════════════════════
flow.append(P("2.8.1.10 Simulation and Competition Panels", "H1"))

# ─ 2.8.1.10.1 Genel Bakış ─
flow.append(P("2.8.1.10.1 Genel Bakış ve Amaç", "H2"))
flow.append(P(
    "Seatwise dashboard&rsquo;ı içerisindeki <i>Simulation</i> paneli, "
    "tezde geliştirilen tüm analitik modüllerin operatör tarafından tek "
    "bir ekran üzerinden birlikte değerlendirilebildiği <b>uçtan-uca "
    "doğrulama merkezidir</b>. Talep tahmini (Bölüm&nbsp;2.4&ndash;2.5), "
    "fiyatlandırma motoru (Bölüm&nbsp;2.6), sentiment modülü "
    "(Bölüm&nbsp;2.7) ve rakip fiyat takibi bu panelde tek bir interaktif "
    "döngü içinde çalışır.",
    "Body"))
flow.append(P(
    "Operatör (havayolu RM uzmanı) için panel üç temel soruya yanıt verir: "
    "<i>(i)</i> dinamik fiyatlandırma motoru, klasik EMSR-tarzı statik "
    "baseline&rsquo;a [13] kıyasla ne kadar gelir kazandırıyor? "
    "<i>(ii)</i> Belirli bir rota ve kalkış tarihinde rezervasyonlar gün "
    "gün nasıl gelişiyor? <i>(iii)</i> Bir koltuk için gösterilen fiyat, "
    "hangi çarpanların ve hangi rakip referanslarının sonucu olarak ortaya "
    "çıkıyor? Bu üç soru, panelin tüm tasarım kararlarını şekillendirmiştir.",
    "Body"))
flow.append(P(
    "Bölümün geri kalanı şu şekilde düzenlenmiştir. Bölüm&nbsp;2.8.1.10.2 "
    "panel düzenini ve kontrolleri tanıtır. Bölüm&nbsp;2.8.1.10.3 hız "
    "seçimi ve zaman sıkıştırması mantığını açıklar. Bölüm&nbsp;2.8.1.10.4 "
    "canlı KPI satırını, Bölüm&nbsp;2.8.1.10.5 koltuk haritasını, "
    "Bölüm&nbsp;2.8.1.10.6 motor iş akışını detaylandırır. "
    "Bölüm&nbsp;2.8.1.10.7 panelin hangi modülleri bir araya getirdiğini "
    "anlatır. Bölüm&nbsp;2.8.1.10.8 koltuk-bazlı fiyat analizini, "
    "Bölüm&nbsp;2.8.1.10.9 operatör müdahalelerini, Bölüm&nbsp;2.8.1.10.10 "
    "Competition panelini, Bölüm&nbsp;2.8.1.10.11 haftalık ayrıntı "
    "analizini sunar. Bölüm&nbsp;2.8.1.10.12 Monte Carlo doğrulamasını, "
    "Bölüm&nbsp;2.8.1.10.13 rapor üretimini ve Bölüm&nbsp;2.8.1.10.14 "
    "sınırlılıklar ile sonucu kapsar.",
    "Body"))

flow.append(SHOT("2.8.1.10.1",
    "Simulation panelinin tam ekran görünümü &mdash; üst kontrol şeridi, "
    "KPI satırı, koltuk haritası ve uçuş envanter tablosu birlikte."))

# ─ 2.8.1.10.2 Panel Düzeni ─
flow.append(P("2.8.1.10.2 Panel Düzeni ve Kullanıcı Kontrolleri", "H2"))
flow.append(P(
    "Simulation paneli yukarıdan aşağıya beş yatay bölümden oluşur: "
    "<b>üst başlık şeridi</b> (logo, durum rozeti, simülasyon saati), "
    "<b>kontrol şeridi</b> (rota, kalkış tarihi, hız, başlat/duraklat/devam "
    "düğmeleri ve Competition View bağlantısı), <b>KPI satırı</b> (altı "
    "canlı metrik), <b>koltuk haritası</b> (Boeing 777-300ER kabini) ve "
    "alttaki <b>uçuş envanter tablosu</b>. Bu sıralama, operatörün "
    "&ldquo;senaryoyu seç &rarr; başlat &rarr; metrikleri izle &rarr; "
    "koltukları incele &rarr; ayrıntılı raporu çıkar&rdquo; akışını "
    "doğrudan takip eder.",
    "Body"))
flow.append(P(
    "Üst kontrol şeridi 50 farklı IST-merkezli rotayı (örn. IST&ndash;LHR, "
    "IST&ndash;AUH, IST&ndash;CDG) ve 1&nbsp;Nisan&nbsp;2026 ile "
    "31&nbsp;Aralık&nbsp;2026 arasında bir kalkış tarihini seçilebilir "
    "kılar. <i>All Routes (50 routes)</i> seçeneği, ağ-çapında "
    "çoklu uçuş simülasyonu başlatır; tek bir rota seçildiğinde ise "
    "panel o rotaya yoğunlaşır ve koltuk haritası ile uçuş tablosu "
    "yalnızca o uçuşu gösterir.",
    "Body"))
flow.append(P(
    "Üst sağ köşedeki <b>durum rozeti</b> (READY / RUNNING / PAUSED / "
    "COMPLETED), simülasyon yaşam döngüsünün hangi aşamasında olunduğunu "
    "ifade eder. RUNNING durumunda rozet pulse animasyonu yapar; bu, "
    "ekrana arka planda bakan birinin bile simülasyonun aktif olduğunu "
    "anında görmesini sağlar.",
    "Body"))

flow.append(SHOT("2.8.1.10.2",
    "Üst başlık şeridi ve kontrol şeridinin yakın çekimi &mdash; rota "
    "seçici, hız seçici, Start/Pause/Resume düğmeleri ve Competition "
    "View bağlantısı."))

# ─ 2.8.1.10.3 Hız Seçimi ─
flow.append(P("2.8.1.10.3 Hız Seçimi ve Zaman Sıkıştırması", "H2"))
flow.append(P(
    "Bir uçuşun rezervasyon ufku 180 gündür; gerçek zamanda bunu "
    "izlemek mümkün olmadığından panel, simülasyon zamanını gerçek "
    "zamana <b>hızlandırma çarpanı</b> ile bağlar. Operatör altı hız "
    "seçeneğinden birini seçer:",
    "Body"))
flow.append(P(
    "&bull; <b>Detailed</b> &mdash; 1 simülasyon günü &asymp; 10 dakika "
    "(eğitim ve hata ayıklama)<br/>"
    "&bull; <b>Watch</b> &mdash; 1 gün &asymp; 5 dakika<br/>"
    "&bull; <b>Slow</b> &mdash; tüm horizon &asymp; 50 saniye<br/>"
    "&bull; <b>Normal</b> &mdash; tüm horizon &asymp; 15 saniye<br/>"
    "&bull; <b>Fast</b> (varsayılan) &mdash; tüm horizon &asymp; 8 saniye<br/>"
    "&bull; <b>Very Fast</b> &mdash; tüm horizon &asymp; 3 saniye",
    "Body"))
flow.append(P(
    "<b>Fast</b> varsayılan ayardır çünkü jüri sunumu ve sınıf-içi "
    "demolar için 8 saniye, izleyiciye dolma dinamiğini gözle takip "
    "ettirebilecek hem de dikkati kaybetmeyecek bir süredir. "
    "<b>Detailed</b> ise sistemi geliştirme aşamasında her günün "
    "ayrıntılı incelenebilmesi için kullanılır.",
    "Body"))

# ─ 2.8.1.10.4 KPI ─
flow.append(P("2.8.1.10.4 Canlı KPI Satırı", "H2"))
flow.append(P(
    "Kontrol şeridinin hemen altındaki KPI satırı, simülasyon ilerlerken "
    "her kareyi anlık güncelleyen altı kutudan oluşur. Tabloda her "
    "metriğin anlamı ve renk kodu özetlenmiştir:",
    "Body"))
flow.append(TABLE([
    ["Metrik", "Renk", "Anlam"],
    ["Capacity", "Mavi", "Seçili senaryonun toplam koltuk kapasitesi"],
    ["Sold", "Nötr", "O ana kadar satılmış toplam bilet sayısı"],
    ["Avg. Load Factor", "Turuncu", "Aktif senaryodaki ortalama doluluk oranı"],
    ["Engine Revenue", "Yeşil", "Dinamik fiyatlandırma motoru toplam gelir"],
    ["Baseline Revenue", "Kırmızı", "EMSR-tarzı statik fiyatlandırma toplam gelir [13]"],
    ["Delta", "Mor", "Engine ve baseline arasındaki yüzde fark"],
], col_widths=[3.6*cm, 2.0*cm, 9.4*cm]))
flow.append(Spacer(1, 6))
flow.append(P(
    "Renkler bilinçli olarak yeşil (kazanç), kırmızı (referans / "
    "baseline) ve mor (delta) seçilmiştir; operatör sayfaya bakar bakmaz "
    "&ldquo;dinamik motor baseline&rsquo;a karşı kazandırıyor mu?&rdquo; "
    "sorusuna görsel olarak yanıt alır. Delta yüzdesi pilot sonuçlarda "
    "+&thinsp;%10.68, ağ çapındaki 300-replikalı validasyonda "
    "+&thinsp;%28.26 olarak rapor edilmiştir (Bölüm&nbsp;3).",
    "Body"))

flow.append(SHOT("2.8.1.10.3",
    "KPI satırının yakın çekimi &mdash; Capacity, Sold, Avg. Load Factor, "
    "Engine Revenue, Baseline Revenue ve Delta kutuları."))

# ─ 2.8.1.10.5 Koltuk Haritası ─
flow.append(P("2.8.1.10.5 Koltuk Haritası ve Fare Class Görselleştirmesi", "H2"))
flow.append(P(
    "KPI satırının altında, gerçek bir <b>Boeing 777-300ER</b> kabin "
    "düzeniyle birebir uyumlu koltuk haritası yer alır: 49 koltuklu "
    "Business kabin ve 300 koltuklu Economy kabin (sol blok 8&ndash;24, "
    "sağ blok 25&ndash;41). Her koltuk, üzerinde durdurulduğunda "
    "rezervasyon detaylarını gösteren etkileşimli bir bileşendir.",
    "Body"))
flow.append(P(
    "Renk kodlaması fare class&rsquo;ı ifade eder: lacivert &mdash; boş, "
    "gri &mdash; <i>V Promo</i>, sarı &mdash; <i>K Discount</i>, mor "
    "&mdash; <i>M Flex</i>, kırmızı &mdash; <i>Y Full Price</i>. Bu beş "
    "renk, klasik dört-sınıflı havayolu fare yapısının (V/K/M/Y) "
    "görselleştirilmiş halidir [4]. Kabin animasyonu sayesinde simülasyon "
    "ilerledikçe koltukların hangi sırayla ve hangi sınıfta dolduğu "
    "gözle izlenebilir.",
    "Body"))
flow.append(P(
    "Koltuk haritasının altındaki büyük tarih göstergesi &mdash; "
    "<i>Simulation Date</i>, <i>Days to Departure</i>, <i>Departure "
    "Date</i> &mdash; operatörün takvim bağlamını kaybetmemesini sağlar; "
    "DTD özellikle önemlidir çünkü hem fare class kapanma kuralları hem "
    "de Pickup XGBoost tahmini DTD ile değişir.",
    "Body"))

flow.append(SHOT("2.8.1.10.4",
    "Boeing 777-300ER koltuk haritası &mdash; doluluğun yarıladığı bir "
    "anda; Business kabinin ağırlıklı M ve Y, Economy kabinin ise V/K "
    "renkleriyle dolduğu görülür."))
flow.append(SHOT("2.8.1.10.5",
    "Tek bir koltuğa tıklandığında açılan rezervasyon detayı küçük "
    "paneli &mdash; yolcu segmenti, ödenen fiyat, fare class, DTD ve "
    "rezervasyon tarihi."))

# ─ 2.8.1.10.6 Çalışma Prensibi ─
flow.append(P("2.8.1.10.6 Simülasyon Motorunun Günlük İş Akışı", "H2"))
flow.append(P(
    "Operatör <i>Start</i> düğmesine bastığında, motor arka planda "
    "ayrı bir iş parçacığında [12] her simülasyon günü için altı işlemi "
    "sırayla yürütür. Bu altı adımın görsel sonucu, ekrandaki KPI "
    "satırının ve koltuk haritasının canlı güncellenmesidir.",
    "Body"))

fig1 = (
    "<b>[ Initialize ]</b> &mdash; envanter, ön-doluluk (warm-up), seed<br/>"
    "&darr;<br/>"
    "<b>[ SimClock.tick() ]</b> &mdash; sim_day &larr; sim_day + 1<br/>"
    "&darr;<br/>"
    "<b>[ Günlük döngü ]</b><br/>"
    "<br/>"
    "&nbsp;&nbsp; 1. <i>forecast_bridge.predict_daily_batch()</i> "
    "&mdash; TFT/XGBoost tahminleri<br/>"
    "&nbsp;&nbsp; 2. <i>_process_cancellations()</i> "
    "&mdash; iptal taraması<br/>"
    "&nbsp;&nbsp; 3. <i>competitor_manager.update_daily_prices()</i> "
    "&mdash; rakip fiyat hareketi<br/>"
    "&nbsp;&nbsp; 4. <i>_generate_daily_bots()</i> "
    "&mdash; stokastik bot yolcu üretimi (NB)<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&rarr; her bot için <i>_process_bot()</i> "
    "&mdash; satın alma kararı<br/>"
    "&nbsp;&nbsp; 5. <i>_update_prices()</i> "
    "&mdash; pricing engine yeni fiyat seviyeleri<br/>"
    "&nbsp;&nbsp; 6. eğer sim_day == dep_date: <i>_process_departure()</i> "
    "&mdash; no-show + denied boarding<br/>"
    "<br/>"
    "&darr;<br/>"
    "<b>[ sim_day &gt; max_dep_date ]</b> &rarr; <i>completed</i>"
)
flow.append(P(fig1, "FigBox"))
flow.append(P("Şekil&nbsp;2.8.1.10.1. Günlük simülasyon adımının olay akışı. "
              "Her adım <i>simulation_engine.py</i>&rsquo;deki ilgili "
              "fonksiyona karşılık gelir; ekrandaki canlı güncelleme bu "
              "altı adımın çıktısıdır.", "Caption"))
flow.append(P(
    "Bu altı adımın bir kez tamamlanması, ekranda &ldquo;bir gün&rdquo; "
    "olarak görünür: KPI&rsquo;lar yeniden hesaplanır, bazı koltuklar "
    "lacivertten renkli hâle döner, uçuş tablosundaki LF / Engine $ / "
    "Delta sütunları güncellenir. Stokastik talep gerçekleşmesi "
    "<i>Negative Binomial</i> dağılımıyla örneklenir; bu seçim, klasik "
    "Poisson varsayımının yakalayamadığı &ldquo;sessiz/yoğun gün&rdquo; "
    "salınımını yakalar [14].",
    "Body"))

# ─ 2.8.1.10.7 Bütünleşik Modüller ─
flow.append(P("2.8.1.10.7 Panelin Bir Araya Getirdiği Modüller", "H2"))
flow.append(P(
    "Simulation paneli, dashboard&rsquo;un diğer modüllerinin <b>canlı "
    "tüketicisidir</b>; arka tarafta tek bir motor sınıfı "
    "(<i>SimulationEngine</i>, <i>dashboard/simulation_engine.py</i>) "
    "aşağıdaki bileşenleri tek bir döngüde birleştirir:",
    "Body"))
flow.append(P(
    "&bull; <b>Forecast Bridge</b> &mdash; Bölüm&nbsp;2.4 (XGBoost) ve "
    "Bölüm&nbsp;2.5 (TFT) modellerini sarmalayan ara katman; her sim_day "
    "için tüm aktif uçuşların günlük tahminlerini tek bir batch çağrısıyla "
    "alır.",
    "Body"))
flow.append(P(
    "&bull; <b>Pricing Engine</b> &mdash; Bölüm&nbsp;2.6&rsquo;daki "
    "DTD&nbsp;&times;&nbsp;DOW&nbsp;&times;&nbsp;sezon&nbsp;&times;&nbsp;LF&nbsp;&times;"
    "&nbsp;rakip&nbsp;&times;&nbsp;sentiment çarpanlarını uygular; her gün "
    "yeni <i>open_fares</i> kümesini ve fiyatlarını üretir.",
    "Body"))
flow.append(P(
    "&bull; <b>Sentiment Module</b> &mdash; Bölüm&nbsp;2.7&rsquo;deki "
    "şehir-bazlı bileşik skoru talep çarpanına dönüştürür "
    "(M<sub>sent</sub>); olumlu duygu rezervasyon hacmini artırır, "
    "tehdit sinyalleri azaltır.",
    "Body"))
flow.append(P(
    "&bull; <b>Competitor Manager</b> &mdash; üç rakip havayolunun (THY, "
    "Pegasus, Emirates) günlük fiyat hareketini canlandırır; pricing "
    "engine bu fiyatları rakip referans çarpanı olarak kullanır.",
    "Body"))
flow.append(P(
    "&bull; <b>Network Optimizer</b> &mdash; ağ-çapında simülasyonlarda "
    "(All Routes seçeneği) aktif olur; rotalar arası paylaşımlı kapasite "
    "kararlarını koordine eder.",
    "Body"))
flow.append(P(
    "&bull; <b>Reports JSON Katmanı</b> &mdash; <i>demand_functions_report.json</i>, "
    "<i>pricing_calibration_report.json</i>, <i>passenger_segments_report.json</i> "
    "gibi kalibrasyon dosyaları motor başlatılırken yüklenir; tüm "
    "çarpanlar bu raporlardan beslenir.",
    "Body"))
flow.append(P(
    "Operatör panelde tek bir buton aracılığıyla bu altı modülün "
    "etkileşimini gözlemleyebilir; yani panel, <i>tezde geliştirilen "
    "tüm matematiksel parçaların entegrasyon noktasıdır</i>.",
    "Body"))

# ─ 2.8.1.10.8 Pricing Decision Analysis ─
flow.append(P("2.8.1.10.8 Pricing Decision Analysis &mdash; Koltuk-Bazlı "
              "Fiyat Açıklaması", "H2"))
flow.append(P(
    "Koltuk haritasındaki dolu bir koltuğa <b>çift tıklamak</b>, "
    "<i>Pricing Decision Analysis</i> modal&rsquo;ini açar. Bu modal, o "
    "yolcunun ödediği fiyatın nasıl ortaya çıktığını yedi alt başlıkta "
    "gösterir:",
    "Body"))
flow.append(P(
    "&bull; <b>Base Price Derivation</b> &mdash; rota uzunluğu ve kabin "
    "tipinden taban fiyatın nasıl elde edildiği.<br/>"
    "&bull; <b>Dynamic Multiplier Decomposition</b> &mdash; DTD, DOW, "
    "sezon, LF, sentiment ve rakip çarpanlarının ayrı ayrı katkıları.<br/>"
    "&bull; <b>Final Price Computation</b> &mdash; çarpanların "
    "birleştirilerek nihai fiyatın hesaplanması.<br/>"
    "&bull; <b>Fare Class Architecture</b> &mdash; V/K/M/Y sınıflarının "
    "fiyat aralıkları ve özellikleri [4].<br/>"
    "&bull; <b>Fare Class Selection Algorithm</b> &mdash; o yolcuya "
    "neden o sınıfın açıldığı.<br/>"
    "&bull; <b>Competitive Landscape</b> &mdash; aynı rotada rakip "
    "havayollarının o anki fiyatları.<br/>"
    "&bull; <b>Revenue Impact</b> &mdash; bu satışın hem dinamik motor "
    "hem statik baseline gelir akışına katkısı.",
    "Body"))
flow.append(P(
    "Bu modal, fiyatın kara kutu olmadığını gösteren <b>açıklanabilirlik "
    "katmanıdır</b>; jüri sunumunda &ldquo;sistem neden 312&nbsp;USD "
    "verdi?&rdquo; sorusu burada adım adım yanıtlanır.",
    "Body"))

flow.append(SHOT("2.8.1.10.6",
    "Pricing Decision Analysis modal&rsquo;i &mdash; bir Y-class "
    "rezervasyonun fiyat ayrıştırması; çarpanlar, fare class seçimi ve "
    "rakip karşılaştırması tek ekranda."))

# ─ 2.8.1.10.9 Operatör Müdahaleleri ─
flow.append(P("2.8.1.10.9 Operatör Müdahaleleri (What-If Senaryoları)", "H2"))
flow.append(P(
    "Simülasyon devam ederken operatör senaryoyu durdurmadan müdahale "
    "edebilir; bu özellik &ldquo;ya&hellip; olsaydı?&rdquo; tipi politika "
    "sorularını canlı test etmeyi mümkün kılar. Üç ana müdahale türü "
    "vardır:",
    "Body"))
flow.append(P(
    "&bull; <b>Fare Class Override</b> &mdash; belirli bir uçuş için "
    "bir fare class kalıcı olarak açılır veya kapatılır. Operatör "
    "örneğin Y sınıfını erken kapatarak, dinamik motorun reaksiyonunu "
    "(diğer sınıflara baskı, gelir kaybı) gözlemleyebilir.",
    "Body"))
flow.append(P(
    "&bull; <b>Bot Injection</b> &mdash; manuel olarak ek talep enjekte "
    "etme. <i>Stress test</i> için kullanılır: ani bir talep şoku "
    "(örn. 100 yolcu) sistemin doluluğu nasıl yönettiğini gösterir.",
    "Body"))
flow.append(P(
    "&bull; <b>Human Booking</b> &mdash; müşteri portalı <i>BiletBul</i> "
    "(Bölüm&nbsp;2.8.1.11) üzerinden gelen gerçek yolcu rezervasyonu "
    "simülasyona enjekte edilir. Bu özellik, panellerin operatör "
    "(simulation) ve yolcu (BiletBul) bakış açılarını birleştiren "
    "köprüdür.",
    "Body"))

flow.append(SHOT("2.8.1.10.7",
    "Override paneli &mdash; bir uçuş için Y-class&rsquo;ı kapatma "
    "düğmesi ve sonrasında ekrandaki tepkinin (delta artışı / fare mix "
    "kayması) görüntüsü."))

# ─ 2.8.1.10.10 Competition Panel ─
flow.append(P("2.8.1.10.10 Competition Panel", "H2"))
flow.append(P(
    "Kontrol şeridindeki <i>&#9992;&nbsp;Competition&nbsp;View</i> "
    "düğmesine tıklamak, ayrı bir tarayıcı sekmesinde Competition "
    "panelini açar. Yeni sekmede açılması bilinçli bir tasarım "
    "kararıdır: ana simülasyon böylece kesintisiz devam eder ve "
    "operatör iki paneli yan yana izleyebilir.",
    "Body"))
flow.append(P(
    "Competition paneli üç bölümden oluşur. Üst kısımda <b>üç rakip "
    "havayolunun</b> (THY, Pegasus, Emirates) o anki fiyatları, ortada "
    "<b>fiyat farkı çizgisi</b> (Seatwise &minus; rakip ortalaması), "
    "alt kısımda ise <b>spill / underprice / lost-to-competitor</b> "
    "sayaçları yer alır. Bu sayaçlar, dinamik fiyatlandırma motorunun "
    "rekabet karşısındaki duruşunu &mdash; ne zaman rakipten ucuz, ne "
    "zaman pahalı, ne kadar yolcu rakibe gitti &mdash; sayısal olarak "
    "raporlar.",
    "Body"))

flow.append(SHOT("2.8.1.10.8",
    "Competition panelinin tam görünümü &mdash; üç rakip havayolu "
    "fiyatları, fark çizgisi ve spill / underprice / lost-to-competitor "
    "sayaçları."))

# ─ 2.8.1.10.11 Detailed Weekly Analysis ─
flow.append(P("2.8.1.10.11 Detailed Weekly Analysis", "H2"))
flow.append(P(
    "Simülasyon <i>completed</i> durumuna geçtiğinde kontrol şeridinde "
    "<b>Detail Analysis</b> düğmesi belirir. Bu düğme, koşumu hafta hafta "
    "kıran bir modal açar; modalde dört zaman serisi bir aradadır:",
    "Body"))
flow.append(P(
    "&bull; <b>Haftalık Load Factor evrimi</b> &mdash; doluluk hangi "
    "haftada hangi seviyeye ulaştı.<br/>"
    "&bull; <b>Booking pace vs forecast pace</b> &mdash; gerçekleşen "
    "rezervasyon hızı, Pickup XGBoost (Bölüm&nbsp;2.4) tahminleriyle "
    "ne kadar uyuşuyor.<br/>"
    "&bull; <b>Fare class mix değişimi</b> &mdash; haftalar arasında "
    "V/K/M/Y dağılımı nasıl kaydı.<br/>"
    "&bull; <b>Sentiment çarpanının haftalık etkisi</b> &mdash; "
    "M<sub>sent</sub> hangi haftalarda talebi pozitif/negatif yönde "
    "kaydırdı.",
    "Body"))
flow.append(P(
    "Bu görselleştirme, koşumdan sonra &ldquo;sistem neden bu "
    "performansı verdi?&rdquo; sorusunun retrospektif olarak "
    "yanıtlandığı yerdir.",
    "Body"))

flow.append(SHOT("2.8.1.10.9",
    "Detailed Weekly Analysis modal&rsquo;i &mdash; LF eğrisi, booking "
    "pace, fare mix kayması ve sentiment etkisi tek pencerede."))

# ─ 2.8.1.10.12 Monte Carlo ve Reproducibility ─
flow.append(P("2.8.1.10.12 Monte Carlo Doğrulaması ve Reproducibility", "H2"))
flow.append(P(
    "Tek bir simülasyon koşumu, stokastik talep gerçekleşmesinin sadece "
    "<i>bir örneklemidir</i>; istatistiksel iddialar bağımsız replikalar "
    "üzerinden ortalama ve varyansa dayanmalıdır [12], [15]. Panelin "
    "<i>run_monte_carlo()</i> arka uç metodu varsayılan N&nbsp;=&nbsp;50 "
    "(Bölüm&nbsp;3 doğrulamasında 300) bağımsız koşum çalıştırır; her "
    "koşum farklı bir tohumla "
    "(seed<sub>i</sub>&nbsp;=&nbsp;1000&nbsp;+&nbsp;i) başlatılır. Bu "
    "sayede aynı senaryo aynı parametrelerle yeniden çalıştırıldığında "
    "aynı sonucu üretir; bu özellik akademik tekrarüretilebilirliğin "
    "temel gereğidir.",
    "Body"))
flow.append(P(
    "Tüm replikalar tamamlandıktan sonra her metrik için "
    "<i>%95 güven aralığı</i> aşağıdaki klasik formülle hesaplanır [15]:",
    "Body"))
flow.append(EQ(
    "CI<sub>95%</sub> = "
    "<i>&mu;</i> &plusmn; 1.96 &middot; <i>&sigma;</i>/&radic;N"))
flow.append(P(
    "Bu aralık, panelin Generate Report çıktısındaki tüm gelir, LF, "
    "delta ve denied-boarding metriklerinin yanında raporlanır. Pilot "
    "sonuçlardaki +&thinsp;%10.68 ve 300-replikalı validasyondaki "
    "+&thinsp;%28.26 gelir lifti, bu güven aralığı içinde istatistiksel "
    "olarak anlamlı kabul edilmiştir.",
    "Body"))

# ─ 2.8.1.10.13 Generate Report ─
flow.append(P("2.8.1.10.13 Generate Report &mdash; Çıktının Üretimi", "H2"))
flow.append(P(
    "Simülasyon <i>completed</i> durumuna ulaştığında kontrol şeridinde "
    "<b>Generate Report</b> düğmesi aktif olur. Bu düğme, koşumun tüm "
    "çıktısını tek bir PDF&rsquo;de toplar:",
    "Body"))
flow.append(P(
    "&bull; KPI satırının son hâli (Capacity, Sold, LF, Engine $, "
    "Baseline $, Delta).<br/>"
    "&bull; Koltuk haritasının kalkış anındaki snapshot&rsquo;ı.<br/>"
    "&bull; Uçuş envanter tablosunun tam içeriği.<br/>"
    "&bull; Detailed Weekly Analysis grafikleri.<br/>"
    "&bull; Monte Carlo özet istatistikleri ve %95 güven aralıkları.<br/>"
    "&bull; Pricing decomposition logu.",
    "Body"))
flow.append(P(
    "Bu rapor, Bölüm&nbsp;3&rsquo;te (Sonuçlar) raporlanan tüm sayısal "
    "metriklerin ham kaynağıdır; başka bir deyişle, panel hem "
    "<i>üretim</i> hem <i>belgeleme</i> aracıdır.",
    "Body"))

flow.append(SHOT("2.8.1.10.10",
    "Generate Report çıktısı &mdash; PDF&rsquo;in kapak sayfası ve KPI "
    "özeti; Monte Carlo güven aralıklarının raporlandığı bölüm."))

# ─ 2.8.1.10.14 Limitations & Conclusion ─
flow.append(P("2.8.1.10.14 Sınırlılıklar ve Sonuç", "H2"))
flow.append(P(
    "Panelin mevcut sürümü dört temel sınırlılığa sahiptir: "
    "<b>(i)</b> Negative Binomial dispersion parametresi <i>r</i>&nbsp;=&nbsp;5 "
    "tüm rota-kabin kombinasyonları için sabittir; segment-bazlı "
    "kalibrasyon kalitesini iyileştirebilir [14]. "
    "<b>(ii)</b> Bot satın alma kararları bağımsızdır; gerçekte aile/grup "
    "rezervasyonları korelasyonlu kararlar üretir. "
    "<b>(iii)</b> Statik baseline EMSR&rsquo;nin sadeleştirilmiş bir "
    "versiyonudur; tam EMSR-b dinamik koruma seviyeleri [13] henüz "
    "baseline koluna entegre değildir. "
    "<b>(iv)</b> Tek <i>threading.Lock</i> tabanlı eşzamanlılık, uçuş "
    "sayısı binlere çıktığında darboğaz oluşturabilir.",
    "Body"))
flow.append(P(
    "Bu sınırlılıklara rağmen panel, tezin geliştirdiği tüm modülleri "
    "(talep tahmini, fiyatlandırma, sentiment, rakip analizi) tek bir "
    "interaktif ekranda bir araya getirerek <b>uçtan-uca canlı "
    "doğrulamayı</b> mümkün kılar. Operatör perspektifinden panel, "
    "&ldquo;sistemi anlama&rdquo; ve &ldquo;sistemi gösterme&rdquo; "
    "rollerini aynı anda üstlenir; jüri sunumunda sistemin yetkinliğini "
    "kanıtlayan ana arayüz olması da bu çift rolün doğal sonucudur.",
    "Body"))

# ─── REFERENCES ─────────────────────────────────────────────
flow.append(P("Kaynaklar", "H1"))
refs = [
    ("[4] K. T. Talluri and G. J. van Ryzin, <i>The Theory and Practice of "
     "Revenue Management</i>. New York, NY, USA: Springer, 2004."),
    ("[12] J. Banks, J. S. Carson II, B. L. Nelson, and D. M. Nicol, "
     "<i>Discrete-Event System Simulation</i>, 5th ed. Upper Saddle River, "
     "NJ, USA: Prentice Hall, 2010."),
    ("[13] P. P. Belobaba, &ldquo;Application of a probabilistic decision "
     "model to airline seat inventory control,&rdquo; <i>Operations "
     "Research</i>, vol. 37, no. 2, pp. 183&ndash;197, 1989."),
    ("[14] A. C. Cameron and P. K. Trivedi, <i>Regression Analysis of Count "
     "Data</i>, 2nd ed. Cambridge, U.K.: Cambridge Univ. Press, 2013."),
    ("[15] P. Glasserman, <i>Monte Carlo Methods in Financial Engineering</i>. "
     "New York, NY, USA: Springer, 2004."),
]
for r in refs:
    flow.append(P(r, "Ref"))

# ── Build ──
DESKTOP.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=2.0*cm, rightMargin=2.0*cm,
    topMargin=2.2*cm, bottomMargin=2.0*cm,
    title="Bolum 2.8.1.10 - Simulation Environment",
    author="Group 16 - Seatwise",
)
doc.build(flow, onFirstPage=on_page, onLaterPages=on_page)
print(f"OK -> {OUT}")
print(f"Size: {OUT.stat().st_size/1024:.1f} KB")
