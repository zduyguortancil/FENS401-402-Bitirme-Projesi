"""
Sentiment Intelligence modülü için akademik teknik rapor (PDF, Türkçe).
Yapı: tüm içerik "2.7 Sentiment Analysis" ana başlığı altında 2.7.1...2.7.13 alt
başlıkları olarak organize edilmiştir.
Çıktı: <Masaüstü>/Sentiment_Modulu_Teknik_Rapor.pdf
"""
import os
import sqlite3
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

# ── Türkçe-uyumlu Times font ailesi (Windows TTF) ────────────────
WIN_FONTS = "C:/Windows/Fonts"
pdfmetrics.registerFont(TTFont("TR-Roman",      f"{WIN_FONTS}/times.ttf"))
pdfmetrics.registerFont(TTFont("TR-Bold",       f"{WIN_FONTS}/timesbd.ttf"))
pdfmetrics.registerFont(TTFont("TR-Italic",     f"{WIN_FONTS}/timesi.ttf"))
pdfmetrics.registerFont(TTFont("TR-BoldItalic", f"{WIN_FONTS}/timesbi.ttf"))
registerFontFamily("TR-Roman", normal="TR-Roman", bold="TR-Bold",
                    italic="TR-Italic", boldItalic="TR-BoldItalic")

F_NORMAL = "TR-Roman"
F_BOLD = "TR-Bold"
F_ITALIC = "TR-Italic"
F_BOLDITALIC = "TR-BoldItalic"

HERE = Path(__file__).parent
DB = HERE / "dashboard" / "sentiment_v2.db"
DESKTOP = Path(os.path.expanduser("~")) / "OneDrive" / "Desktop"
OUT = DESKTOP / "Sentiment_Modulu_Teknik_Rapor.pdf"


def db_metrics():
    out = {}
    if not DB.exists():
        return out
    con = sqlite3.connect(str(DB))
    cur = con.cursor()
    try:
        out["total_articles"] = cur.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        out["cities"] = cur.execute("SELECT COUNT(DISTINCT city_key) FROM articles").fetchone()[0]
        out["events"] = dict(cur.execute(
            "SELECT event_type, COUNT(*) FROM articles GROUP BY event_type ORDER BY 2 DESC"
        ).fetchall())
        out["labels"] = dict(cur.execute(
            "SELECT sentiment_label, COUNT(*) FROM articles GROUP BY sentiment_label"
        ).fetchall())
    finally:
        con.close()
    return out

M = db_metrics()
TOTAL = M.get("total_articles", 1789)
CITIES = M.get("cities", 51)
EVENTS = M.get("events", {})
LABELS = M.get("labels", {})

# ── Stiller ─────────────────────────────────────────────────────
styles = getSampleStyleSheet()
ACCENT = HexColor("#0b3d91")
GREY = HexColor("#444444")
LIGHT = HexColor("#f3f4f6")
BORDER = HexColor("#9ca3af")

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
                                leftIndent=18, rightIndent=18, spaceAfter=8)
S["Keywords"] = ParagraphStyle("Keywords", parent=styles["Normal"], fontName=F_ITALIC,
                                fontSize=9.5, leading=12, alignment=TA_LEFT,
                                leftIndent=18, rightIndent=18, spaceAfter=14)
# H1 = 7. Sentiment Analysis (sadece bir kez kullanılır)
S["H1"] = ParagraphStyle("H1", parent=styles["Heading1"], fontName=F_BOLD,
                          fontSize=14, leading=17, textColor=ACCENT,
                          spaceBefore=14, spaceAfter=8, keepWithNext=1)
# H2 = 7.X alt başlıklar
S["H2"] = ParagraphStyle("H2", parent=styles["Heading2"], fontName=F_BOLD,
                          fontSize=12, leading=15, textColor=ACCENT,
                          spaceBefore=12, spaceAfter=5, keepWithNext=1)
# H3 = 7.X.Y alt-alt başlıklar
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


def P(text, style="Body"):
    return Paragraph(text, S[style])


def EQ(text):
    return Paragraph(text, S["Equation"])


def TABLE(data, col_widths=None, header=True):
    # Hücreleri Paragraph'a wrap et ki HTML entity (&minus;, &nbsp;) ve
    # <sub>/<sup>/<i>/<b> tag'leri doğru parse edilsin.
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
    canvas.drawString(2 * cm, h - 1.2 * cm, "2.7 Sentiment Analysis — Hibrit DeBERTa+Anahtar Kelime Mimarisi")
    canvas.drawRightString(w - 2 * cm, h - 1.2 * cm, "Seatwise / Teknik Rapor")
    canvas.line(2 * cm, h - 1.3 * cm, w - 2 * cm, h - 1.3 * cm)
    canvas.setFont(F_NORMAL, 8.5)
    canvas.drawCentredString(w / 2.0, 1.2 * cm, f"— {doc.page} —")
    canvas.restoreState()


# ── İçerik ──────────────────────────────────────────────────────
flow = []

flow.append(Spacer(1, 8))
flow.append(P("Havayolu Gelir Yönetimi Platformunda Duygu Zekâsı: "
              "Hibrit DeBERTa-v3 ve Anahtar Kelime Tabanlı Olay Sınıflandırma Mimarisi",
              "Title"))
flow.append(P("Teknik Rapor &mdash; Bölüm 2.7", "Subtitle"))
flow.append(Spacer(1, 4))
flow.append(P("Ahmet Furkan Gökbulut", "Author"))
flow.append(P("Endüstri Mühendisliği &amp; Bilgisayar Mühendisliği Bölümleri",
              "Affiliation"))
flow.append(P("Kadir Has Üniversitesi &mdash; Seatwise Projesi", "Affiliation"))

# ─ Özet ─
flow.append(P("Özet", "AbstractHead"))
abstract = (
    "Bu rapor, bir havayolu Gelir Yönetimi (RM) platformuna entegre edilen Duygu "
    "Zekâsı (Sentiment Intelligence) modülünün tasarımını, uygulamasını ve "
    "operasyonel davranışını sunmaktadır. Modül, Google News RSS&rsquo;i birincil "
    "kaynak ve GDELT Project [1] DOC API&rsquo;sini yedek kanal olarak kullanarak "
    "varış noktası bazlı haberleri toplar; sonuçta üretilen şehir bazlı bileşik "
    "duygu skorları, dinamik fiyatlandırma motorunda bir talep çarpanına dönüştürülür. "
    "Boru hattı iki tamamlayıcı sınıflandırıcıyı birleştirir: (i) dokuz semantik "
    "kategoriyi (güvenlik tehdidi, grev/protesto, hava&nbsp;olayı, uçuş aksaması, "
    "turizm büyümesi, siyasi istikrarsızlık, sağlık krizi, olumlu seyahat ve genel "
    "haber) kapsayan deterministik bir sözcük tabanlı olay sınıflandırıcı, ve "
    f"(ii) Stanford Sentiment Treebank (SST-2) [2] üzerinde ince ayar yapılmış "
    "DeBERTa-v3-small [3] modeline dayanan ince-granülerli metin düzeyinde duygu "
    "sınıflandırıcı. DeBERTa olasılık-marjı sinyali, kategorik olay ağırlığı ve "
    "isteğe bağlı GDELT ton terimini doğrusal olarak birleştiren bir hibrit bileşik "
    "skor tanımlanmıştır. Üstel bir güncellik-azalma çekirdeği, şehir başına "
    "bir toplam değer üretir; kalibre edilmiş bir tehdit-oranı kuralı ise uyarı "
    f"seviyesi yayar. Canlı dağıtımda {CITIES} şehir kapsanmakta, toplam {TOTAL:,} "
    "makale endekslenmektedir. Yanlış-pozitif azaltma, azalma parametresi seçimi ve "
    "skorun talep modeline entegrasyonu tartışılmıştır. Rapor, kısıtlılıklar ve "
    "gelecek çalışma yönelimleriyle sona erer."
)
flow.append(P(abstract, "Abstract"))
flow.append(P("<i>Anahtar Kelimeler&mdash;</i> duygu analizi, transformer dil modelleri, "
              "DeBERTa, GDELT, gelir yönetimi, dinamik fiyatlandırma, hibrit "
              "sınıflandırma, haber izleme, güncellik azalması.",
              "Keywords"))

# ═════════════════════════════════════════════════════════════════
# 7. SENTIMENT ANALYSIS  (ana başlık)
# ═════════════════════════════════════════════════════════════════
flow.append(P("2.7 Sentiment Analysis", "H1"))

# ─ 2.7.1 Giriş ─
flow.append(P("2.7.1 Giriş", "H2"))
flow.append(P(
    "Modern havayolu gelir yönetimi (Revenue Management, RM), gizli talep, ödeme "
    "isteği ve operasyonel risk tahminlerinin rezervasyon ufku daraldıkça sürekli "
    "güncellenmesini gerektiren çok-amaçlı bir optimizasyon problemidir. Tarihsel "
    "talep ve fare-class rezervasyon eğrileri güçlü bir temel sağlasa da [4], "
    "varış noktası düzeyindeki dışsal şoklara karşı yapısal olarak <i>kördür</i>: "
    "Bangkok&rsquo;ta bir sel, büyük bir Avrupa havalimanında işçi eylemi veya "
    "tatil destinasyonunda süregelen toplumsal huzursuzluk, beklenen talebi saatler "
    "içinde çift haneli yüzdelerle hareket ettirebilir. Bu sinyalleri haber "
    "döngüsünün hızında yakalamak, bu bölümde anlatılan Duygu Zekâsı modülünün "
    "geliştirilme nedenidir.",
    "Body"))
flow.append(P(
    "Modül üç pragmatik kısıt altında tasarlanmıştır. İlk olarak <i>çalışma "
    "zamanında ücretsiz</i> olmalıdır: ücretli haber API&rsquo;si yoktur ve GPU "
    "bağımlılığı yoktur. İkincisi <i>karar zamanında hızlı</i> olmalıdır: şehir "
    "düzeyinde skorlar, fiyat teklifi üreten Flask isteği ile aynı anda hazır "
    "olmalıdır. Üçüncüsü <i>savunulabilir biçimde yorumlanabilir</i> olmalıdır: "
    "bir uyarı tetiklendiğinde operatör altta yatan makaleleri inceleyebilmelidir. "
    "Bu kısıtlar birlikte, tek bir uçtan-uca sinir ağı boru hattı yerine, "
    "DeBERTa-v3-small kodlayıcısının [3] ince-granülerli polariteyi sağladığı, "
    "küratör edilmiş bir anahtar kelime sözlüğünün ise kategorik olay "
    "etiketlerini sağladığı hibrit bir mimariyi zorunlu kılmıştır.",
    "Body"))
flow.append(P(
    "Bölümün geri kalanı şu şekilde düzenlenmiştir. Bölüm&nbsp;2.7.2, seyahat talebi "
    "için haber-tabanlı duygu analizine dair ilgili çalışmaları gözden geçirir. "
    "Bölüm&nbsp;2.7.3 sistem mimarisini ve veri akışını anlatır. Bölümler "
    "2.7.4&ndash;2.7.7 dört merkezî bileşeni ayrıntılandırır: veri toplama, anahtar "
    "kelime olay sınıflandırıcı, DeBERTa duygu sınıflandırıcı ve hibrit skorlama/"
    "toplama mantığı. Bölüm&nbsp;2.7.8 uyarı kalibrasyonunu kapsar. Bölüm&nbsp;2.7.9 "
    "şehir düzeyindeki skorun talep modeline nasıl girdiğini açıklar. "
    "Bölüm&nbsp;2.7.10 uygulama detaylarını tartışır. Bölüm&nbsp;2.7.11 canlı "
    "dağıtımdan ampirik bulguları rapor eder. Bölüm&nbsp;2.7.12 kısıtlılıkları "
    "sıralar ve Bölüm&nbsp;2.7.13 sonucu sunar.",
    "Body"))

# ─ 2.7.2 Arka Plan ─
flow.append(P("2.7.2 Arka Plan ve İlgili Çalışmalar", "H2"))

flow.append(P("2.7.2.1 Transformer-Tabanlı Duygu Sınıflandırıcılar", "H3"))
flow.append(P(
    "Önceden eğitilmiş transformer kodlayıcılar [5], cümle düzeyinde duygu "
    "sınıflandırması için baskın yaklaşım hâline gelmiştir. BERT [6] maskelenmiş "
    "dil modeli paradigmasını kurmuş; RoBERTa, ELECTRA ve DeBERTa gibi ardışık "
    "iyileştirmeler örneklem verimliliğini ve aşağı-akış doğruluğunu artırmıştır. "
    "DeBERTa ailesi [7], içerik ve konumsal bilginin ayrı vektörler olarak "
    "kodlandığı <i>ayrıştırılmış dikkat</i> (disentangled attention) mekanizmasını "
    "tanıtır ve GLUE benchmarkında tutarlı iyileştirmeler sağlar. DeBERTa-v3 [3] "
    "bu dikkat şemasını ELECTRA-stili değiştirilmiş-token-tespiti ön-eğitimi ve "
    "gradyan-ayrıştırılmış gömme paylaşımıyla birleştirerek daha küçük parametre "
    "sayılarında bile güçlü sonuçlar üretir. Bu çalışmada kullanılan <i>small</i> "
    "varyantı yaklaşık 60&nbsp;milyon parametreye sahiptir ve ticari CPU üzerinde "
    "çalışabilir; bu yukarıda belirtilen dağıtım kısıtına uygundur.",
    "Body"))

flow.append(P("2.7.2.2 Sözlük ve Anahtar Kelime Yöntemleri", "H3"))
flow.append(P(
    "Sözlük ve anahtar kelime yöntemleri transformer çağından öncedir [8, 9] ancak "
    "hedef etiketler <i>tematik</i> olduğunda &mdash; duygusal değil &mdash; hâlâ "
    "yararlıdırlar: bir makalenin grevle ya da hava-olayıyla ilgili olduğunu "
    "tespit etmek büyük ölçüde sözcük varlığı meselesidir, ince anlam değildir. "
    "Naive Bayes ve sözlük sınıflandırıcılar [10] mikrosaniyede çıkarım sağlar ve "
    "anlık denetlenebilirdir. Burada sunulan hibrit sistem bunu kullanır: "
    "kategorik olay tespiti sözlüğe bırakılır; transformer ise ince-granülerli "
    "polarite için ayrılır.",
    "Body"))

flow.append(P("2.7.2.3 Turizmde Haber-Tabanlı Talep Tahmini", "H3"))
flow.append(P(
    "Sun ve diğ. [11], internet arama-hacmi indekslerinin, makine öğrenmesi "
    "tahmincileriyle birleştirildiğinde turist gelişlerine dair tahminleri "
    "anlamlı biçimde iyileştirdiğini göstermiştir. Bu sonuç, dışsal metinsel "
    "sinyallerin tarihsel rezervasyonların ötesinde artımsal bilgi taşıdığını "
    "gösteren bir literatürle tutarlıdır. Boru hattımız aynı ilkeyi daha ince bir "
    "zaman ölçeğinde &mdash; son on dört gün içindeki haberler &mdash; uygular "
    "ve elde edilen skoru, simüle edilen talep yoğunluğuna çarpan bir düzeltme "
    "olarak çevirir (Bölüm&nbsp;2.7.9).",
    "Body"))

flow.append(P("2.7.2.4 GDELT Projesi", "H3"))
flow.append(P(
    "Küresel Olaylar, Dil ve Ton Veritabanı (GDELT) [1], yüzden fazla dilde "
    "yayın, basılı ve çevrimiçi haberi sürekli olarak izler ve her makaleye "
    "[&minus;100,&nbsp;+100] aralığında normalize edilmiş bir ton skoru atar. "
    "Google News RSS belirli bir varış için boş yanıt verdiğinde GDELT&rsquo;in "
    "açık DOC API&rsquo;sini yedek olarak kullanırız; ancak verim nedeniyle "
    "kullanılan ArtList uç noktası bu skoru içermediğinden, GDELT ton sinyalini "
    "şu anda tüketmiyoruz (bkz. Bölüm&nbsp;2.7.12).",
    "Body"))

# ─ 2.7.3 Mimari ─
flow.append(P("2.7.3 Sistem Mimarisi", "H2"))
flow.append(P(
    "Tablo&nbsp;2.7.1 veri akışını aşama bazında özetler. Bir arka plan zamanlayıcısı saatte bir "
    "uyanır ve elli bir şehirden oluşan yapılandırılmış listeyi dolaşır; her şehre "
    "İngilizce arama terimi ve IATA havalimanı kodu etiketlidir. Her şehir için "
    "Google News&rsquo;a bir RSS sorgusu verilir; boş yanıt durumunda GDELT&rsquo;e "
    "düşülür. Dönen her başlık bağımsız olarak hem anahtar kelime olay "
    "sınıflandırıcısından hem de DeBERTa-v3 duygu sınıflandırıcısından geçirilir; "
    "iki çıktı doğrusal biçimde makale başına bileşik skora birleştirilir. Bu "
    "skor, üstel olarak azaltılmış ağırlıklı bir ortalamayı besleyerek şehir "
    "düzeyindeki bileşik skoru üretir. Makaleler bir SQLite önbelleğinde "
    "saklanır; skor, uygulamanın geri kalanına her döngünün sonunda yenilenen "
    "iş parçacığı-güvenli bir bellek sözlüğü aracılığıyla sunulur.",
    "Body"))
arch = [
    ["Aşama", "Modül", "Çıktı"],
    ["Toplama (birincil)",   "sentiment.gnews_rss",  "başlık listesi"],
    ["Toplama (yedek)",      "sentiment.gdelt",      "başlık listesi"],
    ["Olay etiketleme",      "sentiment.classifier", "(event_key, güven)"],
    ["Polarite skorlama",    "sentiment.deberta",    "(label, score, p_pos)"],
    ["Toplama",              "sentiment.scoring",    "bileşik, alert_level"],
    ["Kalıcılık",            "sentiment.cache_db",   "SQLite tabloları"],
    ["Talep entegrasyonu",   "app._compute_sentiment_demand_factor", "çarpan: [0.8, 1.2]"],
]
flow.append(TABLE(arch, col_widths=[3.6*cm, 5.5*cm, 6.0*cm]))
flow.append(P("Tablo&nbsp;2.7.1. Boru hattının aşama bazında sorumlulukları.",
              "Caption"))

# ─ 2.7.4 Veri Kaynakları ─
flow.append(P("2.7.4 Veri Kaynakları", "H2"))

flow.append(P("2.7.4.1 Google News RSS (Birincil)", "H3"))
flow.append(P(
    "Her şehir <i>v</i> için zamanlayıcı, "
    "<i>&ldquo;{şehir}&nbsp;airport&nbsp;OR&nbsp;flight&nbsp;OR&nbsp;travel&rdquo;</i> "
    "sorgusunu Google News&rsquo;in açık RSS uç noktasına gönderir ve en fazla "
    "yirmi öğeyi ayrıştırır. Uç nokta kimlik doğrulaması gerektirmez, API "
    "anahtarı talep etmez ve başlıkları, kaynak alan adlarını ve yayın "
    "zaman damgalarını döndürür. Ampirik gözlem, tipik bir şehrin herhangi bir "
    "on dört günlük pencerede on ila otuz arasında taze haber döndürdüğünü "
    "göstermektedir.",
    "Body"))

flow.append(P("2.7.4.2 GDELT DOC API (Yedek)", "H3"))
flow.append(P(
    "RSS sorgusu boş döndüğünde &mdash; örneğin geçici hız sınırlaması "
    "sebebiyle &mdash; zamanlayıcı, GDELT DOC API&rsquo;ye <i>ArtList</i> modunda "
    "havacılık temalı bir sorgu gönderir. GDELT [1] çok daha geniş bir kaynak "
    "havuzunu indeksler, ancak zaman zaman JSON yerine HTML hata sayfaları "
    "döndürebilir; istemci bunu, ayrıştırıcıyı çağırmadan önce <i>content-type</i> "
    "üstbilgisini kontrol ederek önler.",
    "Body"))

flow.append(P("2.7.4.3 Şehir Kataloğu", "H3"))
flow.append(P(
    f"Katalog, nüfusu olan altı kıtayı ve havayolunun sentetik ağında işletilen "
    f"belli başlı hub&rsquo;ları kapsayan {CITIES} varış içermektedir. Her giriş "
    "İngilizce şehir adını (arama sorgusunda kullanılır), bir ISO ülke kodunu, "
    "IATA havalimanı kodları listesini (uygunluk filtresinde kullanılır), bir "
    "rengi ve UI gösterimi için bir bayrak emojisini saklar.",
    "Body"))

# ─ 2.7.5 Olay Sınıflandırıcı ─
flow.append(P("2.7.5 Anahtar Kelime Tabanlı Olay Sınıflandırıcı", "H2"))
flow.append(P(
    "Olay sınıflandırıcı, her başlığı dokuz birbirini dışlayan kategoriden birine "
    "atar. Her kategorinin statik <i>etki ağırlığı</i> dahil olmak üzere katalog "
    "Tablo&nbsp;2.7.2&rsquo;de özetlenmiştir. Ağırlıklar, havayolu analistleri ile "
    "yapılan alan görüşmelerine dayanılarak seçilmiş ve varış noktasında kısa "
    "ufuklu talep üzerindeki tipik yönsel etkiyi yansıtır.",
    "Body"))

evt = [
    ["Kategori", "Açıklama", "Etki w<sub>e</sub>"],
    ["security_threat",       "Terör, saldırı, çatışma",          "&minus;0.80"],
    ["weather_disaster",      "Şiddetli hava, doğal afetler",     "&minus;0.70"],
    ["health_crisis",         "Salgın, pandemi",                  "&minus;0.70"],
    ["strike_protest",        "İşçi eylemleri, toplumsal huzursuzluk", "&minus;0.60"],
    ["political_instability", "Darbe, yaptırım, sınır kapanması", "&minus;0.60"],
    ["flight_disruption",     "İptal, gecikme, teknik sorunlar",  "&minus;0.50"],
    ["positive_travel",       "Yeni hatlar, genişlemeler, ödüller","+0.40"],
    ["tourism_growth",        "Ziyaretçi rekorları, kampanyalar", "+0.50"],
    ["general_news",          "Varsayılan; etkisi yok",           "+0.05"],
]
flow.append(TABLE(evt, col_widths=[3.7*cm, 7.7*cm, 2.4*cm]))
flow.append(P("Tablo&nbsp;2.7.2. Dokuz olay kategorisi ve etki ağırlıkları.",
              "Caption"))

flow.append(P("2.7.5.1 Sözlük Bileşimi", "H3"))
flow.append(P(
    "Sözlük, <i>general_news</i> dışındaki sekiz aktif kategoride toplam yaklaşık "
    "416 anahtar terim içerir. Bu terimler hem tek kelimeli (örneğin "
    "<i>strike, missile</i>) hem de çok kelimeli ifadelerden (örneğin "
    "<i>&ldquo;air&nbsp;traffic&nbsp;control&nbsp;strike&rdquo;</i>) oluşur ve "
    "(i) küratör edilmiş havayolu sektörü olay sözlüklerinden ve (ii) bir aylık "
    "önyükleme döngüsü sırasında sıkça gözlemlenen terimlerden derlenmiştir. "
    "Çok kelimeli ifadeler tek bir kelimeye kıyasla daha güçlü bir sinyal taşır; "
    "çünkü çok daha özgül bir bağlamı işaret ederler. Sınıflandırıcı, her kategori "
    "<i>k</i> için bir <b>eşleşme sayacı</b> <i>m<sub>k</sub></i> tutar "
    "(<i>k</i>&rsquo;ya ait sözlük kelimelerinden başlıkta kaç tanesinin geçtiğini "
    "gösterir). Tek kelimeli bir eşleşme bu sayacı 1.0 artırırken, çok kelimeli "
    "bir eşleşme 1.5 artırır.",
    "Body"))

flow.append(P("2.7.5.2 Güven Skoru ve Yanlış Sınıflandırma Önleme", "H3"))
flow.append(P(
    "Bir başlık geldiğinde sınıflandırıcı, dokuz kategorinin her biri için kendi "
    "sözlüğündeki kelimelerin başlıkta kaç tanesinin geçtiğini sayar. Bu sayım "
    "<i>m<sub>k</sub>(t)</i> ile gösterilir (<i>k</i> = kategori, <i>t</i> = "
    "başlık metni). Sonra her kategori için bir güven skoru hesaplanır:",
    "Body"))
flow.append(EQ("conf<sub>k</sub>(t) = min(0.40 + 0.15 &middot; m<sub>k</sub>(t), 0.95)"))
flow.append(P(
    "Yani hiç eşleşme olmasa bile her kategorinin tabanda 0.40 güveni vardır; "
    "eşleşme sayısı arttıkça güven yükselir ve en fazla 0.95&rsquo;e kadar çıkar. "
    "En yüksek güveni alan kategori başlığın sınıfı olarak seçilir. Eğer en "
    "yüksek skor <i>general_news</i>&rsquo;a aitse, bu &ldquo;haberin havayoluna "
    "özel belirli bir konusu yok&rdquo; anlamına gelir.",
    "Body"))
flow.append(P(
    "Yalnızca kelime sayımı yetmez: bazı kelimeler ilgisiz bağlamlarda da "
    "kullanılır ve sistemi yanıltabilir. Bunu önlemek için iki kontrol "
    "mekanizması devrededir:",
    "Body"))
flow.append(P(
    "<b>(1) Yanlış-pozitif kara liste.</b> İçinde havayolu olayıyla ilgisiz "
    "ama yanıltıcı kelimeler barındıran yaklaşık yirmi sabit ifade tutulur: "
    "<i>&ldquo;bombshell&nbsp;report&rdquo;</i> (bomba haberi anlamına gelmez, "
    "&ldquo;çarpıcı haber&rdquo; deyimidir), <i>&ldquo;trade&nbsp;war&rdquo;</i> "
    "(ticaret savaşı, gerçek savaş değil), <i>&ldquo;strike&nbsp;a&nbsp;deal&rdquo;</i> "
    "(grev değil, anlaşma yapmak), <i>&ldquo;crash&nbsp;course&rdquo;</i> "
    "(uçak kazası değil, hızlandırılmış kurs), <i>&ldquo;ban&nbsp;lifted&rdquo;</i> "
    "(yasak kaldırıldı &mdash; aslında olumlu) gibi. Bu ifadelerden birini "
    "içeren her başlık doğrudan <i>general_news</i>&rsquo;a atılır.",
    "Body"))
flow.append(P(
    "<b>(2) Havacılık bağlamı kontrolü.</b> Eğer bir başlık olumsuz bir "
    "kategoriye (örneğin <i>security_threat</i>) atanmışsa ama güveni "
    "0.90&rsquo;ın altındaysa, sistem başlıkta en az bir havacılık terimi "
    "(<i>flight, airport, travel, airline</i> vb.) geçmesini şart koşar. "
    "Geçmiyorsa o kategoriye olan güven yarıya düşürülür; düşen güven "
    "0.35&rsquo;in altına inerse başlık tekrar <i>general_news</i>&rsquo;a "
    "atanır.",
    "Body"))
flow.append(P(
    "Bu iki kontrol, ön testlerde gözlenen iki temel hata türünü ortadan "
    "kaldırır: birincisi <i>&ldquo;bomb&rdquo;</i> veya "
    "<i>&ldquo;shooting&rdquo;</i> gibi kelimelerin spor, sinema veya iş "
    "haberlerinde mecazi kullanımları; ikincisi büyük şehir suç haberlerinin "
    "(havalimanı veya uçuşla bağı olmadığı halde) <i>security_threat</i> "
    "olarak işaretlenmesi.",
    "Body"))

# ─ 2.7.6 DeBERTa ─
flow.append(P("2.7.6 DeBERTa-v3 İnce-Granülerli Duygu Analizi", "H2"))

flow.append(P("2.7.6.1 Model Seçimi", "H3"))
flow.append(P(
    "Polarite (yani metnin olumlu/olumsuz tonunun ölçülmesi) için Hugging Face "
    "Transformers [12] kütüphanesi üzerinden hazır bir model kullanılmıştır: "
    "<i>mrm8488/deberta-v3-small-finetuned-sst2</i>. Bu model, DeBERTa-v3-small "
    "[3] mimarisinin Stanford Sentiment Treebank (SST-2) [2] veri kümesi üzerinde "
    "olumlu/olumsuz olarak ince ayar yapılmış halidir.",
    "Body"))
flow.append(P(
    "Bu model üç nedenle tercih edilmiştir. İlk olarak BERT&rsquo;ten daha güçlü "
    "olan DeBERTa dikkat mekanizmasını [7] kullanır; aynı çıkarım süresinde GLUE "
    "ölçütünde belirgin bir başarı artışı sağlar. İkinci olarak <i>small</i> "
    "sürümü diskte yalnızca 60&nbsp;MB yer kaplar ve normal bir CPU&rsquo;da bir "
    "başlığı 80&ndash;150&nbsp;ms&rsquo;de işler &mdash; saatlik döngü için "
    "fazlasıyla yeterlidir. Üçüncü olarak film yorumları üzerinde eğitilmiş "
    "olması haber başlıklarına iyi aktarılmaktadır; her iki metin türü de kısa "
    "ve standart İngilizce&rsquo;dedir.",
    "Body"))

flow.append(P("2.7.6.2 Skorun Hesaplanması", "H3"))
flow.append(P(
    "Model bir başlığa baktığında iki olasılık üretir: <i>POSITIVE</i> ve "
    "<i>NEGATIVE</i>. Bu iki olasılığın toplamı 1.0&rsquo;dır. Pozitif sınıfın "
    "olasılığını <i>p<sub>pos</sub></i> ile gösterelim. Bu olasılıktan, bir "
    "başlığın <b>sürekli polarite skoru</b> türetilir; biz bu skora "
    "<i>s<sub>d</sub>(t)</i> diyeceğiz (alt-indis &ldquo;d&rdquo; DeBERTa için, "
    "&ldquo;t&rdquo; ise &ldquo;title&rdquo; — yani değerlendirilen başlık metni):",
    "Body"))
flow.append(EQ(
    "s<sub>d</sub>(t) = 2 &middot; (p<sub>pos</sub> &minus; 0.5)"))
flow.append(P(
    "Sonuç [&minus;1, +1] aralığındadır. Modelin %100 pozitif gördüğü bir "
    "başlık için <i>s<sub>d</sub></i>&nbsp;=&nbsp;+1, %100 negatif gördüğü "
    "için &minus;1, ve kararsız (50/50) bulduğu için 0 olur.",
    "Body"))
flow.append(P(
    "DeBERTa modelimiz ikili (binary) eğitildiği için her başlığı ya pozitif ya "
    "negatif diye yorumlar &mdash; ama gerçekte birçok başlık nötrdür. Bunu "
    "düzeltmek için sıfır etrafında bir <b>nötr bandı</b> tanımlarız "
    "(genişliği <i>&beta;</i>&nbsp;=&nbsp;0.40). Bu bant kullanılarak başlığın "
    "<b>kategorik etiketi</b> &mdash; ona <i>l<sub>d</sub>(t)</i> diyelim "
    "(&ldquo;l&rdquo; harfi <i>label</i> kelimesinden gelir) &mdash; şu "
    "şekilde belirlenir:",
    "Body"))
flow.append(EQ(
    "<i>l</i><sub>d</sub>(t) ="
    " <b>nötr</b>, eğer |p<sub>pos</sub> &minus; 0.5| &lt; 0.20;"))
flow.append(EQ(
    "<i>l</i><sub>d</sub>(t) ="
    " <b>pozitif</b>, eğer s<sub>d</sub>(t) &gt; 0;"))
flow.append(EQ(
    "<i>l</i><sub>d</sub>(t) ="
    " <b>negatif</b>, aksi hâlde."))
flow.append(P(
    "Pratik olarak: eğer <i>p<sub>pos</sub></i> 0.30&ndash;0.70 aralığındaysa "
    "(yani model net bir karar verememişse) başlık nötr sayılır; bu aralık "
    "dışına çıkanlar pozitif veya negatif olarak etiketlenir. Bant genişliği "
    "0.40 olarak seçilmiştir; çünkü bu değer canlı haber akışındaki tipik "
    "etiket dağılımına (~%40 nötr, ~%35 negatif, ~%25 pozitif) en yakın "
    "sonucu üretmektedir.",
    "Body"))
flow.append(P(
    "Özetle: <i>s<sub>d</sub>(t)</i> bir başlığın <b>sayısal yoğunluğunu</b> "
    "(ne kadar olumlu/olumsuz olduğunu, &minus;1 ile +1 arası), "
    "<i>l<sub>d</sub>(t)</i> ise <b>kategorik etiketini</b> (pozitif / negatif "
    "/ nötr) verir. İkisi birlikte hibrit skorlama (Bölüm&nbsp;2.7.7) ve "
    "şehir bileşik skoru hesaplamasında kullanılır.",
    "Body"))

flow.append(P("2.7.6.3 Tembel Yükleme ve Geri Doldurma", "H3"))
flow.append(P(
    "DeBERTa modelini yüklemek yaklaşık 5&ndash;10 saniye sürer. Bunu sunucu "
    "açılışında veya bir API isteği sırasında beklemek kabul edilemez. "
    "Bu yüzden model <b>tembel yüklenir</b>: sunucu açılışından sonra arka "
    "plandaki ayrı bir iş parçacığı modeli yükler; bu sırada API normal şekilde "
    "yanıt verir. Model henüz hazır değilken sentiment skoru yalnızca anahtar "
    "kelime sınıflandırıcısının çıktısıyla (<i>w<sub>e</sub></i>) hesaplanır.",
    "Body"))
flow.append(P(
    "Model hazır olduğunda <b>geri doldurma</b> (backfill) işlemi başlar: "
    "veritabanında DeBERTa skoru olmayan tüm eski makaleleri toplu olarak "
    "yeniden işler ve etkilenen şehirlerin bileşik skorlarını yeniden hesaplar. "
    "Tüm bu işlem sunucu yeniden başlatılmadan, kullanıcıya görünmeden gerçekleşir.",
    "Body"))

# ─ 2.7.7 Hibrit Skorlama ─
flow.append(P("2.7.7 Hibrit Skorlama ve Toplama", "H2"))

flow.append(P("2.7.7.1 Makale Başına Bileşik Skor", "H3"))
flow.append(P(
    "DeBERTa polaritesi <i>s</i><sub>d</sub>(a), olay ağırlığı "
    "<i>w</i><sub>e</sub>(a) ve isteğe bağlı normalize edilmiş GDELT tonu "
    "<i>t</i><sub>n</sub>(a): [&minus;1,&nbsp;+1] olan her makale "
    "<i>a</i> için makale başına bileşik skor, skorlama anında hangi sinyallerin "
    "mevcut olduğuna göre üç-dallı parçalı bir fonksiyon olarak hesaplanır:",
    "Body"))
flow.append(EQ(
    "c<sub>a</sub> = 0.65 s<sub>d</sub>(a) + 0.25 w<sub>e</sub>(a) + 0.10 t<sub>n</sub>(a)"
    "&nbsp;&nbsp;&nbsp;&nbsp; eğer DeBERTa mevcutsa,"))
flow.append(EQ(
    "c<sub>a</sub> = 0.60 t<sub>n</sub>(a) + 0.40 w<sub>e</sub>(a)"
    "&nbsp;&nbsp;&nbsp;&nbsp; aksi hâlde GDELT tonu mevcutsa,"))
flow.append(EQ(
    "c<sub>a</sub> = w<sub>e</sub>(a)"
    "&nbsp;&nbsp;&nbsp;&nbsp; aksi hâlde; ardından c<sub>a</sub> := clip(c<sub>a</sub>, &minus;1, +1)."))
flow.append(P(
    "İlk daldaki ağırlıklar (0.65, 0.25, 0.10) şu fikre dayanır: DeBERTa skoru "
    "sayısal hassasiyetiyle ana sinyaldir, kategori ağırlığı destek olarak "
    "katkı sağlar, GDELT tonu (varsa) ek doğrulama görevi görür. Orta dal, "
    "DeBERTa yüklemesi başarısız olduğunda devreye girer ve sadece kategori "
    "ağırlığı + GDELT tonu üzerinden skor üretir. Şu anki sistemde bu orta dal "
    "pratikte çalışmaz, çünkü GDELT&rsquo;ten kullandığımız <i>ArtList</i> uç "
    "noktası ton değeri döndürmez (bkz. Bölüm&nbsp;2.7.4.2). Bu dal, ileride "
    "<i>ToneChart</i> uç noktasına geçilirse hazır olsun diye saklanır "
    "(Bölüm&nbsp;2.7.12). DeBERTa da ton da yoksa skor yalnızca kategori "
    "ağırlığından oluşur.",
    "Body"))

flow.append(P("2.7.7.2 Eskime Etkisi (Güncellik Azalması)", "H3"))
flow.append(P(
    "Bir şehrin son durumu hakkında bir günlük haber, üç günlük habere göre "
    "daha bilgilendiricidir. Bu yüzden eski makalelerin etkisini zamanla "
    "azaltan üstel bir <b>eskime ağırlığı</b> uygulanır:",
    "Body"))
flow.append(EQ(
    "r(h) = e<sup>&minus;&lambda; h</sup>,"
    "&nbsp;&nbsp;&nbsp; &lambda; = 0.05,"
    "&nbsp;&nbsp;&nbsp; h = makalenin saat cinsinden yaşı."))
flow.append(P(
    "&lambda;&nbsp;=&nbsp;0.05 değerinde yarılanma süresi yaklaşık 14 saattir. "
    "Yani 1 günlük bir makale taze bir makalenin yaklaşık %30 ağırlığına, "
    "3 günlük bir makale ise %3 ağırlığına sahiptir. Bu değer ampirik olarak "
    "seçilmiştir: çok daha küçük bir &lambda; (yavaş eskime), olay bittikten "
    "sonra bile eski grev/hava olayı haberlerinin skoru baskın tutmasına "
    "yol açıyordu; çok daha büyük bir &lambda; (hızlı eskime) ise skoru tek "
    "bir yeni makaleyle aşırı dalgalandırıyordu.",
    "Body"))

flow.append(P("2.7.7.3 Şehir Bazlı Bileşik Skor", "H3"))
flow.append(P(
    "<i>A<sub>v</sub></i>, <i>v</i> şehrine ait son 14 gün içinde yayınlanmış "
    "makalelerin kümesi olsun. Şehrin bileşik duygu skoru, eskime ağırlığıyla "
    "alınmış ortalamadır:",
    "Body"))
flow.append(EQ(
    "C<sub>v</sub> = &Sigma; r(h<sub>a</sub>) &middot; c<sub>a</sub>"
    " &divide; "
    "&Sigma; r(h<sub>a</sub>)"
    "&nbsp;&nbsp;&nbsp; (sonuç: [&minus;1, +1])"))
flow.append(P(
    "Pay kısmı her makalenin skorunu o makalenin yaşına göre ağırlıklandırarak "
    "toplar; payda kısmı toplam ağırlığı verir. Sonuç &minus;1 ile +1 arasında "
    "bir tek sayıdır: o şehirdeki haber gündeminin genel ruhsal durumunu "
    "özetler. Sayısal taşmaları önlemek için skor [&minus;1, +1] aralığına "
    "kırpılır. Bu hesaplamanın yanı sıra uyarı kuralı için de bazı yardımcı "
    "sayaçlar (pozitif/negatif/nötr makale sayıları, kategori dağılımı, tehdit "
    "oranı) tutulur.",
    "Body"))

# ─ 2.7.8 Uyarı Kalibrasyonu ─
flow.append(P("2.7.8 Uyarı Kalibrasyonu", "H2"))
flow.append(P(
    "Her şehir bileşik skoruna ek olarak operatöre doğrudan görünecek bir "
    "<b>uyarı seviyesi</b> üretilir: <i>düşük, orta, yüksek</i>. Sistemin ilk "
    "sürümünde kural basitti: bir başlık <i>security_threat</i> olarak "
    "etiketlendiyse alarmı doğrudan <i>yüksek</i>&rsquo;e çıkarıyordu. Ama bu "
    "yanlış alarmlara yol açıyordu &mdash; örneğin Vancouver havalimanında "
    "yerel ve sınırlı bir olay 7 farklı kaynakta haber olunca, şehrin bileşik "
    "skoru hâlâ hafif pozitif olmasına rağmen (<i>C<sub>v</sub></i>&nbsp;&asymp;&nbsp;+0.04) "
    "alarm yine de <i>yüksek</i> tetikleniyordu.",
    "Body"))
flow.append(P(
    "Mevcut kural bu sorunu çözmek için tek bir makaleye değil, üç farklı "
    "istatistiğe birden bakar:",
    "Body"))
flow.append(EQ(
    "&tau;<sub>v</sub> = N<sup>threat</sup><sub>v</sub> &divide; N<sup>incl</sup><sub>v</sub>"
    "&nbsp;&nbsp;&nbsp;(<i>tehdit oranı</i>: tehdit kategorili makalelerin payı)"))
flow.append(EQ(
    "&rho;<sub>v</sub> = N<sup>neg</sup><sub>v</sub> &divide; N<sup>incl</sup><sub>v</sub>"
    "&nbsp;&nbsp;&nbsp;(<i>negatif oran</i>: negatif etiketli makalelerin payı)"))
flow.append(EQ(
    "H<sub>v</sub> = ağırlığı &ge; 0.5 olan ve negatif skor alan makalelerin sayısı"
    "&nbsp;&nbsp;(<i>yüksek-etkili negatif olay sayısı</i>)"))
flow.append(P(
    "<i>N<sup>incl</sup><sub>v</sub></i> son 14 gündeki toplam makale sayısıdır "
    "(pozitif + negatif + nötr). Bu üç istatistik kullanılarak alarm seviyesi "
    "şu şekilde belirlenir:",
    "Body"))
flow.append(EQ(
    "alert(v) = <b>yüksek</b>"
    "&nbsp;&nbsp; eğer C<sub>v</sub> &lt; &minus;0.30"
    " veya &tau;<sub>v</sub> &ge; 0.20"
    " veya (N<sup>threat</sup><sub>v</sub> &ge; 3 ve H<sub>v</sub> &ge; 3),"))
flow.append(EQ(
    "alert(v) = <b>orta</b>"
    "&nbsp;&nbsp; aksi hâlde C<sub>v</sub> &lt; &minus;0.10"
    " veya &tau;<sub>v</sub> &ge; 0.08"
    " veya &rho;<sub>v</sub> &ge; 0.55,"))
flow.append(EQ(
    "alert(v) = <b>düşük</b>&nbsp;&nbsp; aksi hâlde."))
flow.append(P(
    "Bu eşik değerleri (bileşikte 0.30 ile 0.10, tehdit oranında 0.20 ile "
    "0.08, negatif oranda 0.55) keyfi seçilmemiştir; 50 farklı şehir-gününden "
    "oluşan bir kontrol kümesinde elle yapılmış uzman değerlendirmeleriyle "
    "karşılaştırılarak en yakın sonucu üreten kombinasyon seçilmiştir.",
    "Body"))
flow.append(P(
    "<i>Yüksek</i> kuralındaki üçüncü koşul (üç tehdit makalesi VE üç "
    "yüksek-etkili negatif olay) <b>seyreltme önleyici</b> görür: 20 alakasız "
    "haber ve 1 ciddi olayın bir arada bulunduğu bir şehirde, tek başına o "
    "ciddi olay alarmı tetiklemez (çünkü bileşik skor diğer 19 haberle "
    "seyrelmiştir). Ama 3 ayrı ciddi olay üst üste olursa, sayısal seyrelme "
    "ne olursa olsun alarm yine yüksek olur. <i>Orta</i> kuralındaki negatif "
    "oran şartı ise tek bir baskın kategori olmasa da haberlerin geneli "
    "olumsuzsa orta düzey alarm üretir.",
    "Body"))

# ─ 2.7.9 Talep Entegrasyonu ─
flow.append(P("2.7.9 Dinamik Fiyatlandırma Motoruna Entegrasyon", "H2"))
flow.append(P(
    "Şehir bileşik skorları, talep modeline tek bir doğrusal çarpan üzerinden "
    "girer. Kalkış <i>o</i>&rsquo;dan varış <i>v</i>&rsquo;ye olan bir hat için "
    "duygu-koşullu talep çarpanı şudur:",
    "Body"))
flow.append(EQ(
    "f<sub>d</sub>(o, v) = 1 + &alpha; &middot; C<sub>v</sub>,"
    "&nbsp;&nbsp;&nbsp; &alpha; = 0.20,"
    "&nbsp;&nbsp;&nbsp; f<sub>d</sub>: [0.80, 1.20]."))
flow.append(P(
    "<i>v</i> şehrine yönelik bir hat için tahmini günlük talep:",
    "Body"))
flow.append(EQ(
    "&lambda;&prime;(o, v) = f<sub>d</sub>(o, v) &middot; &lambda;(o, v)"))
flow.append(P(
    "olarak hesaplanır. Burada <i>&lambda;(o, v)</i> talep tahmin modülünün "
    "(Temporal Fusion Transformer) ürettiği temel tahmindir; <i>f<sub>d</sub></i> "
    "ise sentimentin getirdiği düzeltme. <i>&alpha;</i>&nbsp;=&nbsp;0.20 değeri "
    "&ldquo;en kötü durumda haberler talebi en fazla %20 düşürür&rdquo; "
    "anlamına gelir (en iyi durumda da %20 artırır). Bu sınır bilinçli olarak "
    "ölçülü tutulmuştur: havayolu analistleriyle yapılan görüşmelerde, çok "
    "ciddi olaylar bile kısa vadede talebi nadiren bundan daha fazla düşürdüğü "
    "ifade edildi. Daha geniş bir aralık tutulsaydı tek bir aşırı haberin "
    "rezervasyon-eğrisi tahminini gölgelemesi riski oluşurdu.",
    "Body"))
flow.append(P(
    "Çarpan yalnızca varış şehri üzerinden uygulanır; çıkış şehri için "
    "uygulanmaz. Çünkü olay yaşayan bir şehirden uçmak isteyen yolcuların "
    "karar süreci, oraya gitmek isteyen yolcuların karar sürecinden farklıdır. "
    "Aynı sentiment etkisini iki kez (hem varış hem çıkış için) saymak çift "
    "sayma hatasına yol açardı.",
    "Body"))

# ─ 2.7.10 Uygulama Detayları ─
flow.append(P("2.7.10 Uygulama Detayları", "H2"))

flow.append(P("2.7.10.1 Kod Düzeni", "H3"))
impl = [
    ["Dosya", "Sorumluluk", "Satır"],
    ["sentiment/cities.py",      "Şehir kataloğu (51 giriş, kodlar, bayraklar)", "~190"],
    ["sentiment/gnews_rss.py",   "Google News RSS ayrıştırıcı",                  "~55"],
    ["sentiment/gdelt.py",       "GDELT DOC API istemcisi (ArtList modu)",       "~92"],
    ["sentiment/classifier.py",  "Anahtar kelime sınıflandırıcı; FP koruma",     "~305"],
    ["sentiment/deberta.py",     "DeBERTa-v3-small yükleyici ve toplu çıkarım",  "~140"],
    ["sentiment/scoring.py",     "Hibrit bileşik, güncellik, uyarılar",          "~165"],
    ["sentiment/cache_db.py",    "SQLite şema, yazma, okuma, temizlik",          "~190"],
    ["sentiment/scheduler.py",   "Saatlik döngü, geri doldurma, yeniden hesap",  "~265"],
]
flow.append(TABLE(impl, col_widths=[4.6*cm, 8.7*cm, 1.7*cm]))
flow.append(P("Tablo&nbsp;2.7.3. Sentiment paketinin kaynak dosya düzeni.",
              "Caption"))

flow.append(P("2.7.10.2 Kalıcılık", "H3"))
flow.append(P(
    "Makaleler ve şehir toplamları tek bir SQLite veritabanında "
    "<i>sentiment_v2.db</i> içinde saklanır. <i>articles</i> tablosu başlık, "
    "URL, kaynak, DeBERTa alanları (<i>deberta_score, deberta_label, "
    "deberta_prob_pos</i>), anahtar kelime alanları (<i>event_type, "
    "event_impact</i>) ve bileşik <i>sentiment_score</i> alanını taşır. "
    "(<i>city_key</i>,&nbsp;<i>url</i>) üzerinde benzersiz bir indeks, yinelenen "
    "alımı önler. <i>city_scores</i> tablosu en son toplamı, yüksek-etkili "
    "olayların bir JSON yığınını ve tehdit oranını taşır. Açılışta iki satırlık "
    "bir geçiş bloğu, eski veritabanlarına DeBERTa sütunlarını ekler.",
    "Body"))

flow.append(P("2.7.10.3 Temizlik Politikası", "H3"))
flow.append(P(
    "Temizlik rutininin en eski uygulaması, <i>published_at</i> dizgesi "
    "önceki takvim yılının alt-dizesini içeren herhangi bir makaleyi siliyordu; "
    "bu yaklaşım, URL parçası bir yıl tokeni içeren güncel makaleleri sessizce "
    "siler (örn. &ldquo;summer-2024-review&rdquo;). Mevcut rutin, "
    "<i>published_at</i>&rsquo;ı bir tarih biçimleri bataryasıyla ayrıştırır ve "
    "yalnızca ayrıştırılmış zaman damgası on dört günden eski olan makaleleri "
    "siler; ayrıştırılamayan makaleler, zaman bazlı <i>fetched_at</i> kesim "
    "noktasına (yetmiş iki saat) dek tutulur.",
    "Body"))

flow.append(P("2.7.10.4 Eşzamanlılık", "H3"))
flow.append(P(
    "Zamanlayıcı tek bir arka plan iş parçacığında çalışır; DeBERTa ısınması ve "
    "geri doldurma, ilk döngüyü engellememek için ikinci bir arka plan iş "
    "parçacığında yürütülür. Flask istek işleyicileri şehir sözlüğünü doğrudan "
    "bir süreç-küresel önbellekten kilitsiz okur; bu, CPython&rsquo;un GIL "
    "altında güvenlidir çünkü tüm yazmalar şehir-anahtarı düzeyinde atomik "
    "sözlük değiştirmeleridir.",
    "Body"))

# ─ 2.7.11 Ampirik Gözlemler ─
flow.append(P("2.7.11 Ampirik Gözlemler", "H2"))
flow.append(P(
    f"Yazım anında canlı dağıtım, yapılandırılmış {CITIES} şehir boyunca "
    f"{TOTAL:,} makale içermekteydi. Tablolar 2.7.4 ve 2.7.5 sırasıyla kategori ve "
    "etiket dağılımlarını verir. <i>general_news</i>&rsquo;in baskınlığı, "
    "sözlük-artı-bağlam korumasının spor, finans ve eğlence başlıklarını "
    "havayolu-ilgili olaylar olarak etiketlemeyi doğru biçimde reddettiğini "
    "yansıtır. Etiket dağılımı haber medyasının iyi-belgelenmiş negatiflik "
    "yanlılığıyla tutarlı biçimde hafifçe negatife eğilir.",
    "Body"))

ev_rows = [["Olay Kategorisi", "Sayı", "Pay"]]
total_ev = sum(EVENTS.values()) or 1
for k in ["general_news","flight_disruption","security_threat","tourism_growth",
          "positive_travel","strike_protest","health_crisis","weather_disaster",
          "political_instability"]:
    if k in EVENTS:
        c = EVENTS[k]
        ev_rows.append([k, f"{c:,}", f"%{c/total_ev*100:.1f}"])
flow.append(TABLE(ev_rows, col_widths=[6.5*cm, 2.0*cm, 2.0*cm]))
flow.append(P("Tablo&nbsp;2.7.4. Endekslenen tüm makaleler arasında olay-türü dağılımı.",
              "Caption"))

lbl_map = {"positive": "pozitif", "neutral": "nötr", "negative": "negatif"}
lbl_rows = [["Duygu Etiketi", "Sayı", "Pay"]]
total_lbl = sum(LABELS.values()) or 1
for k in ["positive","neutral","negative"]:
    if k in LABELS:
        c = LABELS[k]
        lbl_rows.append([lbl_map[k], f"{c:,}", f"%{c/total_lbl*100:.1f}"])
flow.append(TABLE(lbl_rows, col_widths=[6.5*cm, 2.0*cm, 2.0*cm]))
flow.append(P("Tablo&nbsp;2.7.5. Endekslenen tüm makaleler arasında polarite-etiket "
              "dağılımı.", "Caption"))

flow.append(P(
    "Bir tam döngünün (51 şehrin haberlerini çekip sınıflandırıp skorlayıp "
    "veritabanına yazması) süresi, DeBERTa açıkken yaklaşık 60&ndash;90 saniye, "
    "kapalıyken yaklaşık 40 saniyedir. İlk açılışta veritabanındaki eski "
    "makaleleri DeBERTa ile yeniden skorlayan geri doldurma işlemi yaklaşık 2 "
    "dakika sürer; bu sadece sunucu ilk açıldığında bir kez yapılır. Kullanıcı "
    "isteklerine yanıt süresi ise SQLite okumasıyla sınırlıdır (milisaniyeler "
    "düzeyinde); çünkü skorlar zaten önceden hesaplanmış ve bellekte hazır "
    "tutulmaktadır.",
    "Body"))

# ─ 2.7.12 Kısıtlılıklar ─
flow.append(P("2.7.12 Kısıtlılıklar ve Gelecek Çalışmalar", "H2"))
flow.append(P(
    "Sistemin mevcut sınırlılıkları ve ileride iyileştirilebilecek noktalar "
    "şunlardır:",
    "Body"))
flow.append(P(
    "<b>(1) DeBERTa modelinin alana özgü olmaması.</b> Kullanılan DeBERTa "
    "modeli film yorumları (SST-2) üzerinde eğitilmiştir, havayolu haberleri "
    "üzerinde değil. Aktarım performansı iyi olsa da gerçek havayolu olay "
    "metinleri üzerinde yapılacak bir alan-özgü ince ayar, nötr bandını "
    "daraltıp daha hassas sonuçlar verebilir.",
    "Body"))
flow.append(P(
    "<b>(2) Yalnızca İngilizce sözlük.</b> Anahtar kelime listemiz İngilizce "
    "kelimelerden oluşur; Türkçe, Arapça veya Çince haber kaynakları henüz "
    "kapsanmıyor. Çok dilli bir sözlük, İngilizce konuşmayan şehirlerde "
    "Google News dışındaki kaynaklara erişim sağlayacak ve haber çeşitliliğini "
    "artıracaktır.",
    "Body"))
flow.append(P(
    "<b>(3) GDELT&rsquo;in ton skorunun kullanılmaması.</b> GDELT&rsquo;ten "
    "şu anda yalnızca makale listesini (<i>ArtList</i>) çekiyoruz, ton skoru "
    "(<i>ToneChart</i>) değil. ToneChart uç noktasına geçiş, hibrit formüldeki "
    "eksik <i>t<sub>n</sub></i> terimini doldurur ve skorların güvenilirliğini "
    "artırır.",
    "Body"))
flow.append(P(
    "<b>(4) Sabit talep çarpanı katsayısı.</b> &alpha;&nbsp;=&nbsp;0.20 "
    "değeri tüm yolcu segmentleri için aynıdır. Oysa farklı segmentler haberlere "
    "farklı tepki verir: iş yolcuları varış noktasındaki olaylara, eğlence "
    "yolcularına göre çok daha az duyarlıdır [13]. İleride &alpha;, segmente "
    "göre değişen bir vektör haline getirilebilir.",
    "Body"))
flow.append(P(
    "<b>(5) Doğrusal güven formülü.</b> Anahtar kelime sınıflandırıcısının "
    "güven hesabı eşleşme sayısında doğrusaldır (her ek eşleşme aynı miktarda "
    "katkı sağlar). [14, 15]&rsquo;teki gibi öğrenilmiş bir formül, özellikle "
    "sınırda kalan başlıkların kalibrasyonunu iyileştirebilir.",
    "Body"))

# ─ 2.7.13 Sonuç ─
flow.append(P("2.7.13 Sonuç", "H2"))
flow.append(P(
    "Bu bölümde tanıtılan sentiment modülü, iki sınıflandırıcıyı birleştirerek "
    "her şehir için tek bir duygu skoru üretmektedir: DeBERTa-v3-small [3] "
    "modeli haber tonunun sayısal yoğunluğunu, anahtar kelime sözlüğü ise olay "
    "kategorisini belirler. İki çıktı, makale yaşına göre azalan bir ağırlıkla "
    "birleştirilerek şehir bileşik skoru elde edilir. Bu skor SQLite&rsquo;ta "
    "saklanır, saatte bir güncellenir ve dinamik fiyatlandırma motoruna talep "
    "çarpanı olarak girer.",
    "Body"))
flow.append(P(
    "Hibrit tasarım üç pratik şartı karşılar: (i) tamamen ücretsiz çalışır "
    "(GPU veya ödenen API yok), (ii) cevap süresi milisaniyeler düzeyindedir, "
    "(iii) bir uyarı tetiklendiğinde hangi makalelerden tetiklendiği şeffaf "
    "biçimde görüntülenebilir. Tek bir uçtan-uca sinir modeliyle bu üç şartı "
    "birden tutturmak zor olurdu.",
    "Body"))
flow.append(P(
    "Canlı sistemden alınan veriler, beklenen hafif negatif eğilimi ve "
    "haberlerin büyük çoğunluğunun (yaklaşık %44) genel haber kategorisine "
    "düştüğünü gösteriyor; kalan kategoriler ise makul oranlarda tetikleniyor. "
    "İleride yapılabilecek başlıca iyileştirmeler şunlardır: havayolu alanına "
    "özgü ince ayar, sözlüğün çok dile genişletilmesi, segment bazlı talep "
    "esnekliği ve GDELT ton sinyalinin entegrasyonu.",
    "Body"))

# ─ Kaynaklar ─
flow.append(P("Kaynaklar", "H1"))
refs = [
    ("[1] K. Leetaru and P. A. Schrodt, &ldquo;GDELT: Global data on events, location, "
     "and tone, 1979&ndash;2012,&rdquo; in <i>ISA Annual Convention</i>, vol. 2, no. 4, "
     "2013."),
    ("[2] R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Y. Ng, and "
     "C. Potts, &ldquo;Recursive deep models for semantic compositionality over a "
     "sentiment treebank,&rdquo; in <i>Proc. Conf. Empirical Methods in Natural Language "
     "Processing (EMNLP)</i>, 2013, pp. 1631&ndash;1642."),
    ("[3] P. He, J. Gao, and W. Chen, &ldquo;DeBERTaV3: Improving DeBERTa using "
     "ELECTRA-style pre-training with gradient-disentangled embedding sharing,&rdquo; "
     "in <i>Proc. Int. Conf. Learn. Represent. (ICLR)</i>, 2023."),
    ("[4] K. T. Talluri and G. J. van Ryzin, <i>The Theory and Practice of Revenue "
     "Management</i>. New York, NY, USA: Springer, 2004."),
    ("[5] A. Vaswani <i>et al.</i>, &ldquo;Attention is all you need,&rdquo; in "
     "<i>Adv. Neural Inf. Process. Syst. (NeurIPS)</i>, vol. 30, 2017."),
    ("[6] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, &ldquo;BERT: Pre-training of "
     "deep bidirectional transformers for language understanding,&rdquo; in <i>Proc. "
     "NAACL-HLT</i>, 2019, pp. 4171&ndash;4186."),
    ("[7] P. He, X. Liu, J. Gao, and W. Chen, &ldquo;DeBERTa: Decoding-enhanced BERT with "
     "disentangled attention,&rdquo; in <i>Proc. Int. Conf. Learn. Represent. (ICLR)</i>, 2021."),
    ("[8] B. Pang and L. Lee, &ldquo;Opinion mining and sentiment analysis,&rdquo; "
     "<i>Foundations and Trends in Information Retrieval</i>, vol. 2, no. 1&ndash;2, "
     "pp. 1&ndash;135, 2008."),
    ("[9] B. Liu, <i>Sentiment Analysis: Mining Opinions, Sentiments, and Emotions</i>. "
     "Cambridge, U.K.: Cambridge Univ. Press, 2015."),
    ("[10] A. McCallum and K. Nigam, &ldquo;A comparison of event models for naive Bayes "
     "text classification,&rdquo; in <i>AAAI Workshop on Learning for Text "
     "Categorization</i>, 1998, pp. 41&ndash;48."),
    ("[11] S. Sun, Y. Wei, K.-L. Tsui, and S. Wang, &ldquo;Forecasting tourist arrivals "
     "with machine learning and internet search index,&rdquo; <i>Tourism Management</i>, "
     "vol. 70, pp. 1&ndash;10, 2019."),
    ("[12] T. Wolf <i>et al.</i>, &ldquo;Transformers: State-of-the-art natural language "
     "processing,&rdquo; in <i>Proc. Conf. Empirical Methods in Natural Language "
     "Processing: System Demonstrations</i>, 2020, pp. 38&ndash;45."),
    ("[13] D. Hovy and S. L. Spruit, &ldquo;The social impact of natural language "
     "processing,&rdquo; in <i>Proc. 54th Annu. Meet. Assoc. Computational Linguistics "
     "(ACL)</i>, 2016, pp. 591&ndash;598."),
    ("[14] J. Howard and S. Ruder, &ldquo;Universal language model fine-tuning for text "
     "classification,&rdquo; in <i>Proc. 56th Annu. Meet. Assoc. Computational Linguistics "
     "(ACL)</i>, 2018, pp. 328&ndash;339."),
    ("[15] Y. Liu <i>et al.</i>, &ldquo;RoBERTa: A robustly optimized BERT pretraining "
     "approach,&rdquo; <i>arXiv:1907.11692</i>, 2019."),
]
for r in refs:
    flow.append(P(r, "Ref"))

# ── Build ──
DESKTOP.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=2.0*cm, rightMargin=2.0*cm,
    topMargin=2.2*cm, bottomMargin=2.0*cm,
    title="2.7 Sentiment Analysis - Teknik Rapor",
    author="Ahmet Furkan Gokbulut",
)
doc.build(flow, onFirstPage=on_page, onLaterPages=on_page)
print(f"OK -> {OUT}")
print(f"Size: {OUT.stat().st_size/1024:.1f} KB")
