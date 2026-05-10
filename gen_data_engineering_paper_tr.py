"""
Bölüm 2.2 — Data Engineering & Data Pipeline (Türkçe akademik teknik rapor PDF).
Çıktı: <Masaüstü>/Bolum_2_2_Veri_Muhendisligi_Tr.pdf

Bu bölüm tezin ana raporunun bir ALT BAŞLIĞIDIR (Bölüm 2 Methodology altında).
Bu yüzden kendi başına bir "Sonuç" alt-bölümü içermez; tezin Bölüm 5
(CONCLUSION & FUTURE WORK) bütünleyici olarak yer alır.

Tüm sayılar projedeki gerçek parquet dosyalarından (DuckDB ile sayılmıştır).
"""
import os
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 KeepTogether)
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
OUT = DESKTOP / "Bolum_2_2_Veri_Muhendisligi_Tr.pdf"

styles = getSampleStyleSheet()
ACCENT = HexColor("#0b3d91")
GREY = HexColor("#444444")
LIGHT_BG = HexColor("#f3f4f6")

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
                                   textColor=GREY, spaceAfter=22)
S["AbstractHead"] = ParagraphStyle("AbstractHead", parent=styles["Normal"],
                                    fontName=F_BOLD, fontSize=10, leading=12,
                                    alignment=TA_CENTER, textColor=GREY, spaceAfter=4)
S["Abstract"] = ParagraphStyle("Abstract", parent=styles["Normal"], fontName=F_NORMAL,
                                fontSize=9.5, leading=13.5, alignment=TA_JUSTIFY,
                                leftIndent=18, rightIndent=18, spaceAfter=18)
S["H1"] = ParagraphStyle("H1", parent=styles["Heading1"], fontName=F_BOLD,
                          fontSize=14, leading=18, textColor=ACCENT,
                          spaceBefore=18, spaceAfter=10, keepWithNext=1)
S["H2"] = ParagraphStyle("H2", parent=styles["Heading2"], fontName=F_BOLD,
                          fontSize=12, leading=16, textColor=ACCENT,
                          spaceBefore=18, spaceAfter=8, keepWithNext=1)
S["Body"] = ParagraphStyle("Body", parent=styles["Normal"], fontName=F_NORMAL,
                            fontSize=10.5, leading=15, alignment=TA_JUSTIFY,
                            spaceAfter=8, firstLineIndent=14)
S["Equation"] = ParagraphStyle("Eq", parent=styles["Normal"], fontName=F_ITALIC,
                                fontSize=11, leading=15, alignment=TA_CENTER,
                                spaceBefore=6, spaceAfter=10, textColor=black)
S["Caption"] = ParagraphStyle("Caption", parent=styles["Normal"], fontName=F_ITALIC,
                               fontSize=9, leading=12, alignment=TA_CENTER,
                               textColor=GREY, spaceBefore=4, spaceAfter=18)
S["Ref"] = ParagraphStyle("Ref", parent=styles["Normal"], fontName=F_NORMAL,
                           fontSize=9, leading=12, alignment=TA_LEFT,
                           leftIndent=24, firstLineIndent=-24, spaceAfter=5)
S["FigBox"] = ParagraphStyle("FigBox", parent=styles["Normal"], fontName=F_NORMAL,
                              fontSize=9.5, leading=15, alignment=TA_CENTER,
                              spaceBefore=8, spaceAfter=4, leftIndent=14,
                              rightIndent=14, textColor=black,
                              backColor=LIGHT_BG, borderColor=GREY,
                              borderWidth=0.5, borderPadding=10)


def P(t, st="Body"):
    return Paragraph(t, S[st])


def EQ(t):
    return Paragraph(t, S["Equation"])


def on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFont(F_ITALIC, 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, h - 1.2 * cm,
                      "2.2 Veri Mühendisliği ve Veri Boru Hattı")
    canvas.drawRightString(w - 2 * cm, h - 1.2 * cm, "Seatwise / FENS 402 Group 16")
    canvas.line(2 * cm, h - 1.3 * cm, w - 2 * cm, h - 1.3 * cm)
    canvas.setFont(F_NORMAL, 8.5)
    canvas.drawCentredString(w / 2.0, 1.2 * cm, f"— {doc.page} —")
    canvas.restoreState()


# ── İçerik ──────────────────────────────────────────────────────
flow = []

flow.append(Spacer(1, 8))
flow.append(P("Bölüm 2.2 &mdash; Veri Mühendisliği ve Veri Boru Hattı", "Title"))
flow.append(P("Apache Parquet ve DuckDB Tabanlı, 51 Milyon Kayıtlı Hibrit Veri "
              "Mimarisi", "Subtitle"))
flow.append(Spacer(1, 4))
flow.append(P("Group 16 &mdash; FENS 402 Engineering Design Project II", "Author"))
flow.append(P("Endüstri Mühendisliği Bölümü &mdash; Kadir Has Üniversitesi &mdash; Mayıs 2026",
              "Affiliation"))

# ÖZET
flow.append(P("Özet", "AbstractHead"))
flow.append(P(
    "Bu bölüm, Seatwise dinamik fiyatlandırma platformunun veri mühendisliği "
    "katmanını sunmaktadır. Sistem üç katmanlı bir veri mimarisi üzerine "
    "kurulmuştur: yaklaşık 51 milyon kayıt içeren ham veri, yedi build "
    "script aracılığıyla üretilen işlenmiş veri, ve eğitilmiş model "
    "artefaktları. Depolama katmanında <b>Apache Parquet</b> sütun-tabanlı "
    "formatı [4], analitik katmanda ise <b>DuckDB</b> gömülü sütun-tabanlı "
    "sorgu motoru [3] kullanılmıştır. Bu kombinasyon, 36.9 milyon satırlık "
    "bir eğitim kümesini tek bir geliştirici makinesinde dağıtık sistemlere "
    "ihtiyaç duymadan işlemeyi mümkün kılar. Verinin bu ölçeği, modern ML "
    "uygulamalarında veri miktarının model karmaşıklığı kadar belirleyici "
    "olduğunu öne süren klasik gözlemle [1] uyumludur. Bölüm ayrıca "
    "<i>tidy data</i> ilkelerine [5] göre yapılandırılmış öznitelik "
    "çıkarımı pipeline'ını ve sütun-tabanlı depolamanın [2] disk I/O "
    "üzerindeki etkisini ele alır.",
    "Abstract"))

# ═══════════════════════════════════════════════════════════════
flow.append(P("2.2 Veri Mühendisliği ve Veri Boru Hattı", "H1"))

# 2.2.1
flow.append(P("2.2.1 Giriş ve Motivasyon", "H2"))
flow.append(P(
    "Modern makine öğrenmesi sistemlerinin başarısı, kullanılan veri "
    "kümelerinin <i>kalitesi, hacmi ve yapısı</i> tarafından doğrudan "
    "belirlenir [1]. Bu gözlem, havayolu gelir yönetimi gibi uygulamalı "
    "alanlarda özellikle belirgindir: rezervasyon davranışındaki ince "
    "örüntüleri yakalamak için milyonlarca uçuş-gün gözlemi gerekmektedir. "
    "Seatwise platformu bu gerekliliği doğrudan karşılar; raporlanan tüm "
    "modeller (TFT, Two-Stage XGBoost, Pickup XGBoost) ve simülasyon "
    "ortamı, ortak bir veri katmanından beslenir.",
    "Body"))
flow.append(P(
    "Veri mühendisliği katmanı yalnızca bir depolama mekanizması olarak "
    "değil, aynı zamanda analitik bir temel olarak tasarlanmıştır. Ham "
    "havayolu kayıtları model-hazır eğitim verisine, simülasyon girdilerine "
    "ve karar destek çıktılarına dönüştürülür. Bu bölüm Seatwise&rsquo;ın "
    "ham veriden modele uzanan tam pipeline&rsquo;ını üç boyutta inceler: "
    "üç-katmanlı veri mimarisi (Bölüm&nbsp;2.2.2), öznitelik mühendisliği "
    "akışı (Bölüm&nbsp;2.2.4) ve depolama ile sorgu optimizasyonu "
    "(Bölümler&nbsp;2.2.6&ndash;2.2.7).",
    "Body"))

# 2.2.2 + ŞEKİL
flow.append(P("2.2.2 Üç-Katmanlı Veri Mimarisi", "H2"))
flow.append(P(
    "Sistem veri mimarisi üç ana katman üzerine kuruludur: ham (raw), "
    "işlenmiş (processed) ve modeller (model artifacts). Bu katmanlama, "
    "veri platformlarındaki <i>medallion architecture</i> desenine benzer "
    "biçimde, her aşamada veriyi daha temiz, daha yapısal ve modele "
    "daha yakın hâle getirir. Veri akışı Şekil&nbsp;2.2.1&rsquo;de "
    "görselleştirilmiştir.",
    "Body"))

fig221 = (
    "<b>HAM KATMAN</b><br/>"
    "<i>flight_snapshot_v2.parquet</i> &nbsp; (36.9M satır) &nbsp;&nbsp; "
    "<i>bookings_enriched.parquet</i> &nbsp; (14.5M satır)<br/>"
    "<br/>"
    "&darr;&nbsp;&nbsp;&nbsp; <i>scripts/data_prep/*.py</i><br/>"
    "<br/>"
    "<b>ÖNİŞLEME &amp; ÖZNİTELİK ÇIKARIMI</b><br/>"
    "build_demand_training &nbsp;&middot;&nbsp; build_pickup_master "
    "&nbsp;&middot;&nbsp; build_tft_route_daily<br/>"
    "build_passenger_clusters &nbsp;&middot;&nbsp; add_event_tags "
    "&nbsp;&middot;&nbsp; build_demand_functions<br/>"
    "<br/>"
    "&darr;<br/>"
    "<br/>"
    "<b>İŞLENMİŞ KATMAN</b><br/>"
    "<i>demand_training</i> (37M&nbsp;&times;&nbsp;51) &nbsp;&nbsp; "
    "<i>pickup_master</i> (36.8M&nbsp;&times;&nbsp;54)<br/>"
    "<i>tft_route_daily</i> (138K&nbsp;&times;&nbsp;42) &nbsp;&nbsp; "
    "<i>passenger_clusters</i> (204K&nbsp;&times;&nbsp;30)<br/>"
    "<br/>"
    "&darr;&nbsp;&nbsp;&nbsp; <i>scripts/training/*.py</i><br/>"
    "<br/>"
    "<b>MODEL ARTEFAKTLARI</b><br/>"
    "TFT checkpoint (.ckpt) &nbsp;&middot;&nbsp; XGBoost ailesi (.pkl, .json)"
)
flow.append(KeepTogether([
    P(fig221, "FigBox"),
    P("Şekil&nbsp;2.2.1. Seatwise üç-katmanlı veri mimarisi. Ham "
      "katmandaki uçuş-gün ve rezervasyon kayıtları, "
      "<i>scripts/data_prep/</i> altındaki yedi build script tarafından "
      "model-hazır işlenmiş tablolara dönüştürülür; bu tablolar üzerinde "
      "eğitilen modeller artefakt olarak saklanır.", "Caption"),
]))

# 2.2.3 Ham Veri (TABLO YOK, sadece prose)
flow.append(P("2.2.3 Ham Veri Kaynakları", "H2"))
flow.append(P(
    "Ham veri katmanı üç parquet dosyasından oluşur. En büyüğü "
    "<i>flight_snapshot_v2.parquet</i>&rsquo;dır: <b>36.9 milyon kayıt</b> "
    "ve 12 sütundan oluşur, diskte 204.6&nbsp;MB yer kaplar. Bu dosya, her "
    "uçuşun kalkıştan 0 ila 180 gün öncesine kadar her gün için bir gözlem "
    "barındırır; sistemin &ldquo;zaman-indexli fotoğraf&rdquo; tabanını "
    "oluşturur. Kümülatif satışlar, son günlerdeki rezervasyon hareketleri, "
    "kalan kapasite, doluluk oranı ve gelir bilgileri burada yer alır. "
    "Days to Departure (DTD) bu tablonun anahtar değişkenidir; tüm "
    "rezervasyon-ufuklu modellemenin temelinde DTD bulunur.",
    "Body"))
flow.append(P(
    "<i>bookings_enriched.parquet</i> ise işlem-düzeyi rezervasyon "
    "kayıtlarını tutar (<b>14.5 milyon kayıt</b>, 36 sütun, 341.1&nbsp;MB). "
    "Rezervasyon kanalı, grup büyüklüğü, frequent-flyer kompozisyonu, "
    "çocuk yolcu oranı ve ücret detayları gibi davranışsal nitelikler bu "
    "dosyada yer alır. Bu zenginlik, yolcu segmentasyonu (Bölüm&nbsp;2.5) "
    "ve sentiment-bazlı talep modellemesi (Bölüm&nbsp;2.7) için kritik "
    "hammaddedir.",
    "Body"))
flow.append(P(
    "Bunların yanında, eski sürüm bir uçuş-gün dosyası "
    "(<i>flight_snapshot.parquet</i>, 432&nbsp;MB) arşiv amacıyla "
    "saklanmaktadır; üretim akışında kullanılmaz, ancak tarihsel "
    "karşılaştırmalar ve tutarlılık denetimleri için referans noktası "
    "sağlar.",
    "Body"))

# 2.2.4 Önişleme + Feature Engineering
flow.append(P("2.2.4 Önişleme ve Öznitelik Çıkarımı", "H2"))
flow.append(P(
    "Ham veri doğrudan modellere girdi olarak verilemez. Her bir model, "
    "kendine özgü <i>tidy</i> [5] biçimde &mdash; yani her satırın bir "
    "gözlem, her sütunun bir nitelik olduğu uzun-format tablolar &mdash; "
    "ihtiyaç duyar. Bu dönüşüm, <i>scripts/data_prep/</i> altındaki yedi "
    "Python script&rsquo;i tarafından gerçekleştirilir.",
    "Body"))
flow.append(P(
    "<i>build_demand_training.py</i>, demand_training.parquet&rsquo;ı "
    "üreterek Two-Stage XGBoost demand modelinin eğitim verisini hazırlar. "
    "<i>build_pickup_master.py</i>, kalan-talep tahmininin öğrenildiği "
    "pickup_master.parquet&rsquo;ı yapılandırır. <i>build_tft_route_daily.py</i> "
    "ise tft_route_daily.parquet üzerinden Temporal Fusion Transformer "
    "için route&nbsp;&times;&nbsp;cabin&nbsp;&times;&nbsp;day "
    "agregasyonunu çıkarır. <i>build_passenger_clusters.py</i> K-Means "
    "segmentasyonunun çıktısını saklar; <i>build_demand_functions.py</i> "
    "segment elastikiyetlerini bir JSON raporu olarak üretir; "
    "<i>add_event_tags.py</i> bayram ve özel-dönem etiketlerini hem ham "
    "hem işlenmiş tablolara işler; <i>flight_snapshot.py</i> ise "
    "snapshot dosyalarının yeniden üretimi için bir araç sunar.",
    "Body"))
flow.append(P(
    "Öznitelik mühendisliği aşamasında türetilen başlıca değişkenler şu "
    "matematiksel tanımlara dayanır:",
    "Body"))
flow.append(EQ(
    "DTD = t<sub>kalkış</sub> &minus; t<sub>snapshot</sub>"
    "&nbsp;&nbsp;&nbsp; (Days to Departure)"))
flow.append(EQ(
    "LF(t) = sold(t) / capacity"
    "&nbsp;&nbsp;&nbsp; (anlık doluluk oranı)"))
flow.append(EQ(
    "pace<sub>7d</sub>(t) = sold<sub>cum</sub>(t) &minus; sold<sub>cum</sub>(t&minus;7)"
    "&nbsp;&nbsp;&nbsp; (yedi-günlük rolling rezervasyon hızı)"))
flow.append(EQ(
    "D<sub>remaining</sub>(t) = sold<sub>final</sub> &minus; sold<sub>cum</sub>(t)"
    "&nbsp;&nbsp;&nbsp; (Pickup hedef değişkeni)"))
flow.append(P(
    "Bu temel değişkenlere ek olarak, calendar-bazlı göstergeler "
    "(<i>is_weekend, dep_hour, dep_month</i>), özel-dönem etiketleri "
    "(<i>Kurban, Ramazan, Yılbaşı, 23 Nisan</i>), rota meta-bilgileri "
    "(<i>distance_km, region, route_type</i>) ve fare-class göstergeleri "
    "(V/K/M/Y) eklenir. <i>tidy data</i> ilkesi gereği [5] her veri "
    "kümesi tek bir analitik amaca yönelik biçimde yeniden düzenlenmiş "
    "satır-sütun yapısında tutulur; örneğin TFT için "
    "route&nbsp;&times;&nbsp;cabin&nbsp;&times;&nbsp;day, XGBoost için "
    "ise flight&nbsp;&times;&nbsp;DTD granülasyonu kullanılır.",
    "Body"))

# 2.2.5 İşlenmiş çıktılar (TABLO YOK)
flow.append(P("2.2.5 İşlenmiş Veri Çıktıları", "H2"))
flow.append(P(
    "İşlenmiş katmanın iki büyük dosyası, sistemin demand-side analitik "
    "ağırlığını taşır. <i>demand_training.parquet</i> "
    "<b>36,996,400 satır × 51 sütun</b> ile 106.9&nbsp;MB&rsquo;ı, "
    "<i>pickup_master.parquet</i> ise <b>36,792,000 satır × 54 sütun</b> "
    "ile 167.6&nbsp;MB&rsquo;ı kaplar. Her ikisi de "
    "flight&nbsp;&times;&nbsp;DTD granülasyonundadır ve XGBoost ailesinin "
    "milyon-satır ölçeğini destekler.",
    "Body"))
flow.append(P(
    "TFT modelinin route-day agregasyon ihtiyacı için "
    "<i>tft_route_daily.parquet</i> üretilir; bu dosya "
    "<b>138,018 satır × 42 sütun</b> ile yalnızca 7.1&nbsp;MB tutar. "
    "Aynı uçuş havuzunun route&nbsp;&times;&nbsp;cabin&nbsp;&times;&nbsp;day "
    "düzeyinde agregasyonu sayesinde hem belleğe rahatlıkla sığar hem de "
    "TFT&rsquo;nin <i>quantile multi-horizon</i> mimarisinin doğal "
    "girdisini oluşturur.",
    "Body"))
flow.append(P(
    "Yolcu segmentasyonu çıktısı <i>passenger_clusters.parquet</i> "
    "(<b>204,400 satır × 30 sütun</b>, 5.1&nbsp;MB) K-Means kümelemesinin "
    "uçuş-bazlı sonucunu tutar; statik uçuş meta-bilgileri "
    "<i>flight_metadata.parquet</i>&rsquo;ta (<b>204,400 × 10</b>, "
    "1.4&nbsp;MB) yer alır. Granülasyon farkı, kritik bir tasarım "
    "kararıdır: TFT&rsquo;nin az ama uzun zaman dizileriyle çalışmasına "
    "karşılık, XGBoost her uçuşun her DTD&rsquo;si için ayrı bir gözlem "
    "ile beslenir. İki yaklaşım <b>aynı altta yatan veriden iki farklı "
    "agregasyonla</b> türetilir; bu farklı granülasyonlar farklı "
    "modellerin güçlü yanlarını kullanır.",
    "Body"))

# 2.2.6 Parquet
flow.append(P("2.2.6 Sütun-Tabanlı Depolama: Apache Parquet", "H2"))
flow.append(P(
    "Veri katmanının seçilen depolama formatı, sistemin "
    "ölçeklenebilirliği için belirleyicidir. Geleneksel CSV gibi "
    "satır-tabanlı (<i>row-store</i>) formatlar, ML iş yüklerinde "
    "verimsizdir: çoğu eğitim turu yalnızca birkaç sütunu kullanır, "
    "ancak satır-tabanlı format her satırın tamamını okumak zorundadır. "
    "C-Store [2] ile başlayan sütun-tabanlı (<i>column-store</i>) "
    "yaklaşım bu verimsizliği ortadan kaldırır: her sütun ayrı bir blok "
    "olarak diskte saklanır, gerekli sütunlar I/O&rsquo;ya hiç bulaşmadan "
    "atlanır.",
    "Body"))
flow.append(P(
    "<b>Apache Parquet</b> [4], C-Store fikrini açık-kaynak ekosisteme "
    "taşıyan referans formattır. Seatwise&rsquo;da Parquet üç temel "
    "avantajla benimsenmiştir. <b>Sütun-bazlı sıkıştırma</b> sayesinde "
    "her sütun aynı veri tipinde olduğundan RLE (Run-Length Encoding), "
    "dictionary encoding ve Snappy gibi tekniklerle CSV&rsquo;ye kıyasla "
    "3 ila 10 kat sıkıştırma elde edilir; örneğin 106.9&nbsp;MB tutan "
    "<i>demand_training.parquet</i> dosyasının eşdeğer CSV gösterimi "
    "yaklaşık 1.2&nbsp;GB&rsquo;a karşılık gelir. <b>Predicate pushdown</b> "
    "özelliği sayesinde sorgu motoru filtreleri (örn. "
    "<i>WHERE dtd&nbsp;&lt;&nbsp;30</i>) parquet okuyucuya iletir; ilgisiz "
    "<i>row group</i>&rsquo;lar disk üzerinde atlanır, asla belleğe "
    "alınmaz. <b>Şema evrimi</b> ise yeni sütun eklemenin mevcut dosyaları "
    "geçersiz kılmamasını sağlar; eski sütun setiyle yazılmış dosyalar "
    "yeni şemayla okunabilir.",
    "Body"))

# 2.2.7 DuckDB
flow.append(P("2.2.7 In-Process Analitik: DuckDB", "H2"))
flow.append(P(
    "Parquet dosyalarını sorgulamak için iki temel yol vardır: bir kümede "
    "Spark veya Presto gibi dağıtık motor çalıştırmak ya da pandas ile "
    "tamamını belleğe yüklemek. İlki bizim ölçeğimiz için aşırı, ikincisi "
    "ise belleği aşar (37 milyon satırlı pandas DataFrame, 51 sütunla "
    "yaklaşık 8&nbsp;GB RAM gerektirir). <b>DuckDB</b> [3] bu boşluğu "
    "doldurur: SQLite gibi tek bir kütüphane olarak süreç içine gömülür, "
    "fakat veri tabanı motoru sütun-tabanlı çalışır ve parquet dosyalarını "
    "&ldquo;sıfır-kopya&rdquo; doğrudan okur.",
    "Body"))
flow.append(P(
    "Tipik bir analitik sorgu şöyle yazılır:",
    "Body"))
flow.append(EQ(
    "SELECT route, AVG(load_factor) FROM read_parquet('flight_snapshot_v2.parquet') "
    "WHERE dtd &lt; 30 GROUP BY route;"))
flow.append(P(
    "Bu sorgu, 37 milyon satırlı dosyayı belleğe almadan, yalnızca "
    "ihtiyaç duyulan üç sütunu (<i>route, load_factor, dtd</i>) okuyarak "
    "saniyeler içinde tamamlanır. DuckDB&rsquo;nin bu davranışı "
    "Raasveldt &amp; Mühleisen (2019) [3] tarafından tanımlanan "
    "<i>vectorized execution</i> ve <i>cache-conscious join</i> "
    "algoritmalarına dayanır. Seatwise&rsquo;da DuckDB iki ana noktada "
    "kullanılır: Flask uygulamasının her isteğinde calibration ve "
    "uçuş-meta verilerini anında sorgulamak için (Bölüm&nbsp;2.4), ve "
    "build script&rsquo;leri içinde milyonlarca satırlık feature "
    "engineering joinlerini gerçekleştirmek için. Her iki kullanım da "
    "sunucu bağımlılığı olmadan çalışır.",
    "Body"))

# 2.2.8 Performans
flow.append(P("2.2.8 Performans, Tekrar Üretilebilirlik ve Skalabilite", "H2"))
flow.append(P(
    "Parquet ve DuckDB kombinasyonu, klasik üç darboğazı (disk I/O, RAM, "
    "CPU) tek bir geliştirici makinesinde aşmayı mümkün kılar. Disk I/O "
    "tarafında, 37 milyon satırlık <i>demand_training</i> parquet dosyası "
    "yalnızca kullanılan yaklaşık 12 sütun seçildiğinde 30&nbsp;MB&rsquo;tan "
    "az veri okur (sütun-tabanlı seçici I/O sayesinde); CSV&rsquo;de aynı "
    "sorgu 1.2&nbsp;GB&rsquo;tan fazla okumayı gerektirirdi. Bellek "
    "tarafında, pandas ile tüm <i>demand_training</i> dosyası "
    "yüklendiğinde yaklaşık 8&nbsp;GB RAM tutarken, DuckDB aynı sorguyu "
    "500&nbsp;MB&rsquo;ın altında bir bellek ayak izi ile gerçekleştirir.",
    "Body"))
flow.append(P(
    "Tekrar üretilebilirlik açısından tüm build script&rsquo;leri "
    "deterministiktir (rastgele tohum sabitlenmiştir) ve idempotent "
    "çalışır (aynı girdi her zaman aynı çıktıyı üretir). Bir araştırmacı "
    "tüm pipeline&rsquo;ı sıfırdan yeniden çalıştırarak aynı işlenmiş "
    "dosyaları üretebilir. Ölçek dağıtımı ve verinin işe yaraması üzerine "
    "yapılan klasik argümanlardan biri [1], milyonlarca kayıt ölçeğinde "
    "&ldquo;basit modellerin daha karmaşık modellere genelde yetiştiğini, "
    "ama yetememe nedeninin model değil veri eksikliği olduğunu&rdquo; "
    "belirtir. Seatwise&rsquo;ın 37 milyon satırlık eğitim kümesi, bu "
    "argümanı operasyonel olarak doğrulayan bir veri tabanı sunar; "
    "Two-Stage XGBoost ailesi bu veri ölçeğinde baseline modellerini "
    "anlamlı biçimde geçer (bkz. Bölüm&nbsp;3.5).",
    "Body"))

# 2.2.9 Sınırlılıklar
flow.append(P("2.2.9 Sınırlılıklar ve Gelecek Çalışmalar", "H2"))
flow.append(P(
    "Veri katmanının mevcut sınırlılıkları ve ileride iyileştirilebilecek "
    "noktalar şunlardır. <b>Tek-makine kısıtı:</b> DuckDB tek bir süreçte "
    "çalışır; pipeline binlerce paralel kullanıcıya veya yüz milyon satıra "
    "ölçeklenmek istenirse Spark veya Snowflake gibi dağıtık motorlara "
    "geçiş gerekebilir. <b>Append-only mimari:</b> mevcut pipeline batch "
    "yenileme yapar; akış (streaming) güncellemeler &mdash; örneğin gerçek "
    "zamanlı rezervasyon olaylarının saniyesinde işlenmesi &mdash; "
    "desteklenmez. Apache Kafka ve DuckDB CDC akışları ileride entegre "
    "edilebilir.",
    "Body"))
flow.append(P(
    "<b>Şema rijitliği:</b> Parquet şema evrimi geriye uyumludur, ancak "
    "sütun türü değişiklikleri (örneğin <i>int32</i>&rsquo;den "
    "<i>int64</i>&rsquo;e) tüm geçmiş dosyaların yeniden yazılmasını "
    "gerektirir. <b>Eski sürüm bagajı:</b> <i>flight_snapshot.parquet</i> "
    "432&nbsp;MB yer kaplar ve aktif olarak kullanılmaz; bir veri "
    "yaşam-döngüsü politikasıyla arşive (örneğin S3 Glacier benzeri "
    "soğuk depolama) taşınabilir.",
    "Body"))

# REFERENCES
flow.append(P("Kaynaklar", "H1"))
refs = [
    ("[1] A. Halevy, P. Norvig, and F. Pereira, &ldquo;The unreasonable "
     "effectiveness of data,&rdquo; <i>IEEE Intelligent Systems</i>, "
     "vol. 24, no. 2, pp. 8&ndash;12, 2009."),
    ("[2] M. Stonebraker, D. J. Abadi, A. Batkin, X. Chen, M. Cherniack, "
     "M. Ferreira, E. Lau, A. Lin, S. Madden, E. O&rsquo;Neil, P. O&rsquo;Neil, "
     "A. Rasin, N. Tran, and S. Zdonik, &ldquo;C-Store: A column-oriented "
     "DBMS,&rdquo; in <i>Proc. 31st Int. Conf. Very Large Data Bases (VLDB)</i>, "
     "2005, pp. 553&ndash;564."),
    ("[3] M. Raasveldt and H. Mühleisen, &ldquo;DuckDB: An embeddable "
     "analytical database,&rdquo; in <i>Proc. ACM SIGMOD Int. Conf. on "
     "Management of Data</i>, 2019, pp. 1981&ndash;1984."),
    ("[4] Apache Software Foundation, <i>Apache Parquet Documentation</i>, "
     "2013&ndash;present. [Online]. Available: https://parquet.apache.org/"),
    ("[5] H. Wickham, &ldquo;Tidy data,&rdquo; <i>Journal of Statistical "
     "Software</i>, vol. 59, no. 10, pp. 1&ndash;23, 2014."),
]
for r in refs:
    flow.append(P(r, "Ref"))

# ── Build ──
DESKTOP.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=2.0*cm, rightMargin=2.0*cm,
    topMargin=2.2*cm, bottomMargin=2.0*cm,
    title="Bolum 2.2 - Veri Muhendisligi",
    author="Group 16 - Seatwise",
)
doc.build(flow, onFirstPage=on_page, onLaterPages=on_page)
print(f"OK -> {OUT}")
print(f"Size: {OUT.stat().st_size/1024:.1f} KB")
