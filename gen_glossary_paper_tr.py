"""
Seatwise Tezi — Terimler ve Kısaltmalar Sözlüğü (Türkçe akademik glossary).
Çıktı: <Masaüstü>/Terimler_ve_Kisaltmalar_Sozlugu_Tr.pdf

Bu doküman tezde geçen tüm akademik/teknik terim, kısaltma ve domain
ifadelerinin tanımlarını içerir; raporun "List of Abbreviations" ve
"Glossary" bölümlerinin Türkçe akademik karşılığıdır.
"""
import os
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
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
OUT = DESKTOP / "Terimler_ve_Kisaltmalar_Sozlugu_Tr.pdf"

styles = getSampleStyleSheet()
ACCENT = HexColor("#0b3d91")
GREY = HexColor("#444444")
LIGHT = HexColor("#e8eaf0")

S = {}
S["Title"] = ParagraphStyle("Title", parent=styles["Title"], fontName=F_BOLD,
                             fontSize=20, leading=24, alignment=TA_CENTER,
                             textColor=ACCENT, spaceAfter=4)
S["Subtitle"] = ParagraphStyle("Subtitle", parent=styles["Normal"], fontName=F_ITALIC,
                                fontSize=11, leading=14, alignment=TA_CENTER,
                                textColor=GREY, spaceAfter=22)
S["Intro"] = ParagraphStyle("Intro", parent=styles["Normal"], fontName=F_NORMAL,
                             fontSize=10.5, leading=15, alignment=TA_JUSTIFY,
                             leftIndent=18, rightIndent=18, spaceAfter=18,
                             textColor=GREY, fontName_=F_ITALIC)
S["H1"] = ParagraphStyle("H1", parent=styles["Heading1"], fontName=F_BOLD,
                          fontSize=15, leading=19, textColor=ACCENT,
                          spaceBefore=18, spaceAfter=10, keepWithNext=1,
                          alignment=TA_CENTER)
S["LetterHead"] = ParagraphStyle("LH", parent=styles["Normal"], fontName=F_BOLD,
                                  fontSize=14, leading=18, textColor=black,
                                  spaceBefore=12, spaceAfter=6, keepWithNext=1,
                                  borderPadding=4, backColor=LIGHT,
                                  leftIndent=4, alignment=TA_LEFT)
S["Entry"] = ParagraphStyle("Entry", parent=styles["Normal"], fontName=F_NORMAL,
                             fontSize=10.5, leading=14.5, alignment=TA_JUSTIFY,
                             leftIndent=20, firstLineIndent=-20, spaceAfter=8,
                             rightIndent=4)


def P(t, st):
    return Paragraph(t, S[st])


def E(term, body):
    """Glossary entry: TERM — body"""
    return P(f"<b>{term}</b> &mdash; {body}", "Entry")


def on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFont(F_ITALIC, 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, h - 1.2 * cm,
                      "Terimler ve Kısaltmalar Sözlüğü")
    canvas.drawRightString(w - 2 * cm, h - 1.2 * cm, "Seatwise / FENS 402")
    canvas.line(2 * cm, h - 1.3 * cm, w - 2 * cm, h - 1.3 * cm)
    canvas.setFont(F_NORMAL, 8.5)
    canvas.drawCentredString(w / 2.0, 1.2 * cm, f"— {doc.page} —")
    canvas.restoreState()


flow = []
flow.append(Spacer(1, 8))
flow.append(P("Terimler ve Kısaltmalar Sözlüğü", "Title"))
flow.append(P("Seatwise &mdash; Dynamic Pricing &amp; Decision Support System "
              "for Airline Revenue Management", "Subtitle"))

flow.append(P(
    "Bu sözlük, tezde geçen tüm akademik, teknik ve uygulamaya özgü "
    "kısaltma ve terimlerin Türkçe tanımlarını içerir. Liste iki ana "
    "bölümden oluşur: harf-harf alfabetik düzenlenmiş <b>Kısaltmalar</b> "
    "bölümü (acronyms) ve aynı düzendeki <b>Terimler Sözlüğü</b> "
    "(glossary). Her giriş, terimin tezde ilk geçtiği bölüme atıfta "
    "bulunabilir; tanımlar bağlamdan bağımsız okunabilecek biçimde, fakat "
    "Seatwise platformuna özgü kullanım örneklerine de yer verecek "
    "şekilde verilmiştir.",
    "Intro"))

# ═════════════════════════════════════════════════════════════════
# BÖLÜM I — KISALTMALAR
# ═════════════════════════════════════════════════════════════════
flow.append(P("BÖLÜM I &mdash; KISALTMALAR", "H1"))

# A
flow.append(P("A", "LetterHead"))
flow.append(E("AAAI", "Association for the Advancement of Artificial Intelligence. Yapay zeka alanındaki en eski uluslararası akademik dernek; tezde sınıflandırma yöntemleri için referans verilen <i>AAAI</i> konferans bildirilerinin sahibidir."))
flow.append(E("AI", "Artificial Intelligence (Yapay Zekâ). Genel anlamda, makinelerin algılama, akıl yürütme ve karar verme gibi insana özgü becerileri taklit etmesini sağlayan bilim alanı."))
flow.append(E("API", "Application Programming Interface (Uygulama Programlama Arayüzü). İki yazılım bileşeni arasında veri ve komut alışverişini sağlayan tanımlı sözleşme; Seatwise&rsquo;da Flask uygulaması tüm dış erişimleri REST API üzerinden sağlar."))
flow.append(E("ARM", "Airline Revenue Management. Havayolu sektöründe sınırlı kapasite (koltuk) ve değişken talep koşulları altında geliri maksimize etmeyi hedefleyen disiplin."))
flow.append(E("AUC", "Area Under the (ROC) Curve. İkili sınıflandırıcının performansını ölçen, 0 ile 1 arasında değer alan metrik. 0.5 rastgele tahmin, 1.0 mükemmel ayrımdır. Seatwise&rsquo;ın booking-classifier modeli AUC = 0.835&rsquo;tir."))
flow.append(E("AUH", "Abu Dhabi Uluslararası Havalimanı (IATA havalimanı kodu). Seatwise simülasyonunda test rotalarından biridir."))

# B
flow.append(P("B", "LetterHead"))
flow.append(E("BERT", "Bidirectional Encoder Representations from Transformers. Devlin ve diğ. (2019) tarafından önerilen, doğal dil anlayışında baskın hâline gelen transformer-tabanlı dil modeli. DeBERTa ailesinin atasıdır."))

# C
flow.append(P("C", "LetterHead"))
flow.append(E("CDC", "Change Data Capture. Bir veri kaynağındaki değişikliklerin (insert, update, delete) gerçek zamanlı olarak izlenip aşağı-akış sistemlere iletilmesi yöntemi."))
flow.append(E("CDG", "Paris Charles de Gaulle Havalimanı (IATA). Seatwise simülasyonunda test rotalarından biridir."))
flow.append(E("CI<sub>95</sub>", "%95 Güven Aralığı (Confidence Interval). Bir istatistiksel kestirimin gerçek değerinin %95 olasılıkla içinde bulunduğu aralık. Monte Carlo simülasyonlarda &mu; &plusmn; 1.96&middot;&sigma;/&radic;N formülüyle hesaplanır."))
flow.append(E("CPU", "Central Processing Unit (Merkezi İşlem Birimi). Seatwise tüm modeller GPU bağımlılığı olmadan CPU üzerinde çalışacak şekilde tasarlanmıştır."))
flow.append(E("CRPS", "Continuous Ranked Probability Score. Olasılıksal (quantile-tabanlı) tahminlerin doğruluğunu ölçen sürekli puanlama kuralı; TFT gibi quantile-tabanlı modellerin değerlendirmesinde MAE/RMSE&rsquo;nin alternatifidir."))
flow.append(E("CSV", "Comma-Separated Values. Basit metin tabanlı satır-tabanlı tablo formatı; Parquet&rsquo;e kıyasla ML iş yüklerinde verimsiz kalır (3&times;&ndash;10&times; daha fazla disk kullanımı)."))

# D
flow.append(P("D", "LetterHead"))
flow.append(E("DeBERTa-v3", "Decoding-enhanced BERT with Disentangled Attention, sürüm 3. He ve diğ. (2023) tarafından önerilen, BERT&rsquo;in geliştirilmiş halidir; Seatwise sentiment modülünde metin polaritesini ölçmek için kullanılır."))
flow.append(E("DES", "Discrete-Event Simulation (Ayrık-Olay Simülasyonu). Sistemin durumunun yalnızca belirli zaman noktalarında (booking, iptal, kalkış vb.) değiştiği simülasyon paradigması. Seatwise simülasyon ortamının metodolojik temelidir."))
flow.append(E("DOC", "DOC API. GDELT projesinin makale-arama uç noktası; Seatwise sentiment modülünde Google News RSS&rsquo;in yedeği olarak kullanılır."))
flow.append(E("DOW", "Day-of-week (Haftanın Günü). Pricing engine çarpanlarından biri; Seatwise calibration analizine göre etkisi yaklaşık ±%0.3&rsquo;tür."))
flow.append(E("DSS", "Decision Support System (Karar Destek Sistemi). Seatwise hem havayolu operatörü hem müşteri için iki paralel DSS arayüzü sunar."))
flow.append(E("DTD", "Days to Departure (Kalkışa Kalan Gün Sayısı). Bir uçuşun kalkış tarihinden mevcut tarihe olan farkı; tüm rezervasyon-ufuklu modellemenin temel değişkeni. DTD = 180 erken booking, DTD = 0 kalkış günüdür."))
flow.append(E("DuckDB", "Gömülü, sütun-tabanlı analitik veri tabanı motoru. Raasveldt &amp; Mühleisen (2019) tarafından geliştirilmiştir; Seatwise&rsquo;da Parquet dosyalarını sıfır-kopya doğrudan sorgulamak için kullanılır."))

# E
flow.append(P("E", "LetterHead"))
flow.append(E("ELECTRA", "Efficiently Learning an Encoder that Classifies Token Replacements Accurately. BERT ailesinin daha verimli ön-eğitim yöntemini kullanan transformer varyantı; DeBERTa-v3&rsquo;ün ön-eğitim şeması ELECTRA-stilidir."))
flow.append(E("EMSR", "Expected Marginal Seat Revenue. Belobaba (1989) tarafından önerilen klasik yield management heuristic&rsquo;i; fare class koruma seviyelerini koşullu beklenti hesaplarından türetir. Seatwise statik karşılaştırma tabanı olarak EMSR-tarzı bir baseline kullanır."))
flow.append(E("EMNLP", "Conference on Empirical Methods in Natural Language Processing. NLP alanında önde gelen akademik konferans; tezdeki bazı sentiment referansları (Wolf 2020, Socher 2013) bu konferansta yayınlanmıştır."))

# F
flow.append(P("F", "LetterHead"))
flow.append(E("FF", "Frequent Flyer (Sık Uçan Yolcu). Havayolu sadakat programı üyeliği; bookings_enriched veri kümesinde frequent-flyer kompozisyonu segmentasyon için kullanılır."))
flow.append(E("Fare Class (V/K/M/Y)", "Havayolu rezervasyon sınıfları. V = promosyon (en düşük), K = indirimli, M = standart/esnek, Y = tam fiyat (en yüksek). Sınıflar DTD&rsquo;ye göre dinamik olarak açılır/kapanır."))

# G
flow.append(P("G", "LetterHead"))
flow.append(E("GDELT", "Global Database of Events, Language, and Tone. Leetaru &amp; Schrodt (2013) tarafından kurulan, dünya genelindeki haberleri otomatik tarayan ve tonlama skoru atayan açık veri tabanı. Seatwise sentiment modülünde Google News&rsquo;ün yedek kaynağıdır."))
flow.append(E("GLUE", "General Language Understanding Evaluation. Doğal dil anlayışı modellerinin değerlendirildiği standart benchmark; DeBERTa, BERT&rsquo;e kıyasla GLUE&rsquo;da tutarlı ilerleme sağlar."))
flow.append(E("GPU", "Graphics Processing Unit. Derin öğrenme eğitiminde paralel matris işlemleri için tercih edilen donanım; Seatwise GPU bağımlılığı olmayacak şekilde tasarlandığından inference CPU üzerinde gerçekleşir."))
flow.append(E("GRN", "Gated Residual Network. Lim ve diğ. (2021) TFT mimarisinin temel yapı taşı; non-linear dönüşümleri kontrollü biçimde uygulayarak gradient akışını korur."))

# H
flow.append(P("H", "LetterHead"))
flow.append(E("HTML", "HyperText Markup Language. Web sayfası işaretleme dili; Seatwise dashboard&rsquo;u Flask + HTML5 ile sunulmaktadır."))
flow.append(E("HTTP", "HyperText Transfer Protocol. Web tarayıcı ile sunucu arasındaki standart iletişim protokolü; Seatwise REST API&rsquo;leri HTTP üzerinde çalışır."))

# I
flow.append(P("I", "LetterHead"))
flow.append(E("IATA", "International Air Transport Association. Havayolları uluslararası ticaret birliği; üç-harfli havalimanı kodları (IST, LHR, AUH) bu kuruluşun standardıdır."))
flow.append(E("ICLR", "International Conference on Learning Representations. Derin öğrenme alanında en önde gelen akademik konferanslardan biri; DeBERTa-v3 (He 2023) ve DeBERTa (He 2021) makaleleri burada yayınlanmıştır."))
flow.append(E("I/O", "Input/Output. Disk veya bellekten veri okuma/yazma işlemleri; sütun-tabanlı depolama ile I/O hacmi azaltılır."))
flow.append(E("IST", "İstanbul Havalimanı (IATA). Seatwise simülasyonlarında merkezi hub olarak kullanılır."))

# J
flow.append(P("J", "LetterHead"))
flow.append(E("JSON", "JavaScript Object Notation. Hafif, insan-okunabilir veri değişim formatı; Seatwise API yanıtları ve calibration raporları JSON biçimindedir."))

# K
flow.append(P("K", "LetterHead"))
flow.append(E("K-Means", "Bilinen sayıda kümeye ayırma kümeleme algoritması; Seatwise yolcu segmentasyonunda 6 davranışsal segmentin (Business, VFR, Congress/Medical, Early Leisure, Student, Last-Minute) çıkarımı için kullanılır."))
flow.append(E("KPI", "Key Performance Indicator. Sistem performansını özetleyen anahtar metrik; örn. revenue lift, load factor, yolcu memnuniyeti."))

# L
flow.append(P("L", "LetterHead"))
flow.append(E("LF", "Load Factor (Doluluk Oranı). Bir uçuşta satılan koltukların toplam kapasiteye oranı; Seatwise&rsquo;da dinamik fiyatlandırma sonrası ekonomi LF&rsquo;si 0.21&rsquo;den 0.42&rsquo;ye yükselmiştir."))
flow.append(E("LHR", "London Heathrow Havalimanı (IATA). Seatwise simülasyonlarındaki test rotalarından biridir."))

# M
flow.append(P("M", "LetterHead"))
flow.append(E("MAE", "Mean Absolute Error (Ortalama Mutlak Hata). Tahmin ile gerçek değer arasındaki mutlak farkın ortalaması; aykırı değerlere RMSE&rsquo;den daha az duyarlıdır."))
flow.append(E("MAPE", "Mean Absolute Percentage Error (Ortalama Mutlak Yüzde Hata). Hatanın gerçek değere oranının ortalaması, ölçek-bağımsızdır. Pickup XGBoost modelinin MAPE&rsquo;si %9.82&rsquo;dir."))
flow.append(E("ML", "Machine Learning (Makine Öğrenmesi). Verilerden öğrenerek tahmin/karar üreten algoritmalar disiplini."))
flow.append(E("MNLI", "Multi-Genre Natural Language Inference. Doğal dil çıkarımı için standart eğitim/değerlendirme veri kümesi; bazı zero-shot modellerin temelidir."))

# N
flow.append(P("N", "LetterHead"))
flow.append(E("NAACL", "North American Chapter of the ACL. Doğal dil işleme alanında önemli akademik konferans; BERT (Devlin 2019) bu konferansta sunulmuştur."))
flow.append(E("NB", "Negative Binomial (Negatif Binom Dağılımı). Sayma verisinde varyansı ortalamadan büyük olan (overdispersed) durumların modellenmesinde kullanılan istatistiksel dağılım. Seatwise simülasyonu r=5 ile NB kullanır."))
flow.append(E("NLP", "Natural Language Processing (Doğal Dil İşleme). İnsan dilinin bilgisayarlar tarafından anlaşılması ve üretilmesiyle ilgilenen alan; Seatwise sentiment modülünün temelidir."))
flow.append(E("No-show", "Rezervasyonu olduğu hâlde kalkışa gelmeyen yolcu. Seatwise simülasyonu segment-bazlı no-show oranlarıyla denied boarding hesaplaması yapar."))

# O
flow.append(P("O", "LetterHead"))
flow.append(E("O&amp;D", "Origin &amp; Destination (Kalkış &amp; Varış). Bir uçuşun kalkış-varış havalimanı çifti; rota analizinde temel birim."))
flow.append(E("OAT", "One-At-a-Time (Birer-Birer). Hassasiyet analizinde kullanılan klasik yöntem; tüm parametreleri sabit tutup yalnızca birini değiştirerek çıktıdaki değişimi ölçer."))

# P
flow.append(P("P", "LetterHead"))
flow.append(E("Parquet", "Apache Parquet. Sütun-tabanlı, sıkıştırılmış açık-kaynak veri formatı; Seatwise tüm raw ve processed veri bu formatta tutar."))
flow.append(E("PNR", "Passenger Name Record (Yolcu Adına Rezervasyon Kaydı). Bir uçuşa yapılan tek bir rezervasyon kaydı; aynı PNR&rsquo;de 1 ile 5 yolcu yer alabilir (grup booking)."))

# Q
flow.append(P("Q", "LetterHead"))
flow.append(E("Quantile Loss", "Bkz. Pinball Loss (Terimler Sözlüğü)."))

# R
flow.append(P("R", "LetterHead"))
flow.append(E("RAM", "Random Access Memory. Bilgisayar belleği; pandas ile 37M satırlık veri ~8 GB RAM gerektirirken DuckDB aynı işi <500 MB&rsquo;da yapar."))
flow.append(E("RM", "Revenue Management (Gelir Yönetimi). Sınırlı kapasite ve heterojen talep altında gelir maksimizasyonu disiplini; Talluri &amp; van Ryzin (2004) standart referansıdır."))
flow.append(E("RMSE", "Root Mean Squared Error. Hata karelerinin ortalamasının karekökü; aykırı değerlere MAE&rsquo;den daha duyarlıdır."))
flow.append(E("RNG", "Random Number Generator (Rasgele Sayı Üreteci). Stokastik simülasyonların reproducibility&rsquo;si için tohum (seed) ile başlatılır; Seatwise her Monte Carlo run&rsquo;ı için seed = 1000 + i kullanır."))
flow.append(E("RSS", "Really Simple Syndication. Web sitelerinin yayın akışı için kullandığı XML-tabanlı format; Seatwise sentiment modülü Google News RSS&rsquo;ten haber çeker."))

# S
flow.append(P("S", "LetterHead"))
flow.append(E("SQL", "Structured Query Language. İlişkisel veri tabanlarının standart sorgulama dili; DuckDB içinde Parquet dosyaları SQL ile sorgulanır."))
flow.append(E("SST-2", "Stanford Sentiment Treebank, sürüm 2 (binary). Socher ve diğ. (2013) tarafından oluşturulan, pozitif/negatif olarak etiketlenmiş film yorumu veri kümesi; Seatwise&rsquo;ın DeBERTa modelinin ince ayar veri kümesidir."))
flow.append(E("SIGMOD", "ACM Special Interest Group on Management of Data. Veri tabanı ve veri yönetimi alanındaki en önde gelen akademik konferans; DuckDB makalesi (Raasveldt 2019) burada yayınlandı."))

# T
flow.append(P("T", "LetterHead"))
flow.append(E("TFT", "Temporal Fusion Transformer. Lim ve diğ. (2021) tarafından önerilen, çok-ufuklu zaman serisi tahmini için tasarlanmış yorumlanabilir transformer mimarisi; Seatwise&rsquo;da makro talep tahmini için kullanılır."))
flow.append(E("THY", "Türk Hava Yolları (Turkish Airlines, IATA: TK). Seatwise customer interface&rsquo;da rakip karşılaştırması için kullanılan üç havayolundan birincisi (diğerleri Pegasus ve Emirates)."))

# U
flow.append(P("U", "LetterHead"))
flow.append(E("UI", "User Interface (Kullanıcı Arayüzü). Yazılımın insan kullanıcı ile etkileşim katmanı; Seatwise iki paralel UI sunar: airline RM dashboard (5005) ve customer portal BiletBul (5006)."))
flow.append(E("UX", "User Experience (Kullanıcı Deneyimi). Bir ürünün kullanıcı tarafından nasıl algılandığı; UI&rsquo;dan farklı olarak duygusal ve davranışsal boyutları kapsar."))

# V
flow.append(P("V", "LetterHead"))
flow.append(E("VFR", "Visiting Friends and Relatives (Akraba/Arkadaş Ziyareti). Yolcu segmentasyonunda davranışsal bir grup; Seatwise&rsquo;da %20&rsquo;lik paya sahiptir, orta fiyat-elastikiyetli bir segmenttir."))
flow.append(E("VLDB", "Very Large Data Bases. Veri tabanı sistemleri alanında köklü akademik konferans; C-Store makalesi (Stonebraker 2005) burada sunulmuştur."))
flow.append(E("VSN", "Variable Selection Network. TFT mimarisinin bileşenlerinden biri; her zaman adımında hangi girdi değişkenlerinin önemli olduğunu öğrenir, modelin yorumlanabilirliğine katkıda bulunur."))

# W
flow.append(P("W", "LetterHead"))
flow.append(E("WTP", "Willingness to Pay (Ödeme İsteği). Bir yolcunun bir bilet için ödemeye razı olduğu maksimum fiyat; segment bazında değişir, dinamik fiyatlandırmanın temel girdilerinden biridir."))

# X
flow.append(P("X", "LetterHead"))
flow.append(E("XGBoost", "eXtreme Gradient Boosting. Chen &amp; Guestrin (2016) tarafından geliştirilen ölçeklenebilir gradient-boosted trees algoritması; Seatwise&rsquo;ta üç farklı XGBoost modeli kullanılır (two-stage demand, enhanced demand, pickup)."))

# ═════════════════════════════════════════════════════════════════
# BÖLÜM II — TERİMLER SÖZLÜĞÜ
# ═════════════════════════════════════════════════════════════════
flow.append(P("BÖLÜM II &mdash; TERİMLER SÖZLÜĞÜ", "H1"))

# A
flow.append(P("A", "LetterHead"))
flow.append(E("Apache Parquet", "Açık-kaynak sütun-tabanlı veri depolama formatı; sıkıştırma, predicate pushdown ve şema evrimini destekler. Seatwise&rsquo;ın depolama altyapısının temelidir."))
flow.append(E("Attention Mekanizması", "Bir nöral ağın, girdideki farklı parçalara değişen ağırlıklar verebilmesini sağlayan yapı taşı; Vaswani ve diğ. (2017) tarafından popülerleştirilmiştir."))

# B
flow.append(P("B", "LetterHead"))
flow.append(E("Booking Curve (Rezervasyon Eğrisi)", "Bir uçuşun kalkışa kadar olan rezervasyon birikiminin zaman içinde nasıl arttığını gösteren eğri; tipik olarak S şeklindedir."))
flow.append(E("Bootstrap Cycle (Önyükleme Döngüsü)", "Sistemin ilk çalışmaya başladığı süreçte, sözlük veya parametre kalibrasyonu için kullanılan kısa süreli ön-eğitim aşaması."))

# C
flow.append(P("C", "LetterHead"))
flow.append(E("Calibration (Kalibrasyon)", "Modelin parametrelerini gerçek veriden istatistiksel olarak öğrenmesi süreci; Seatwise pricing engine&rsquo;in region, DTD, season çarpanları 18M booking üzerinden regresyonla kalibre edilmiştir."))
flow.append(E("Class Imbalance (Sınıf Dengesizliği)", "Sınıflandırma probleminde bir sınıfın diğerlerine kıyasla aşırı fazla/az olması; Seatwise&rsquo;da hedef değişkenin %70.8 zero-rate&rsquo;i nedeniyle two-stage XGBoost yapısı kullanılmıştır."))
flow.append(E("Columnar Storage (Sütun-Tabanlı Depolama)", "Verinin diskte sütun-sütun yazılması; sadece gerekli sütunların okunmasını sağlar, ML iş yükleri için satır-tabanlı CSV&rsquo;ye göre çok daha verimlidir."))
flow.append(E("Composite Score (Bileşik Skor)", "Birden fazla kaynaktan gelen sinyalin doğrusal kombinasyonuyla üretilen tek bir özet skor; Seatwise sentiment composite skoru DeBERTa, keyword event ve GDELT tone&rsquo;u birleştirir."))

# D
flow.append(P("D", "LetterHead"))
flow.append(E("DeBERTa-v3-small", "DeBERTa-v3 mimarisinin küçük (~60M parametre) varyantı; Seatwise sentiment modülünde <i>mrm8488/deberta-v3-small-finetuned-sst2</i> kontrol noktası kullanılır."))
flow.append(E("Denied Boarding", "Kabul edilen booking sayısının kapasiteyi aşması durumunda yolcunun uçağa alınamaması; overbooking stratejisinin doğal bir riskidir, simülasyonda Monte Carlo metriği olarak raporlanır."))
flow.append(E("Discrete-Event Simulation (Ayrık-Olay Simülasyonu)", "Sistemin durumunun yalnızca belirli zaman noktalarında (event&rsquo;lerde) değiştiği simülasyon paradigması; Banks ve diğ. (2010) standart referansıdır."))
flow.append(E("Disentangled Attention", "DeBERTa&rsquo;nın temel yeniliği; içerik ve konum bilgilerini ayrı vektörlerle kodlayarak GLUE benchmark&rsquo;ında BERT&rsquo;e karşı tutarlı kazanım sağlar."))
flow.append(E("Dynamic Pricing (Dinamik Fiyatlandırma)", "Talep, kapasite, zaman ve dış sinyallere göre fiyatın gerçek zamanlı olarak güncellenmesi; static pricing&rsquo;in tersidir."))

# E
flow.append(P("E", "LetterHead"))
flow.append(E("Embedded Database (Gömülü Veri Tabanı)", "Sunucu olarak değil, uygulamanın içinde kütüphane olarak çalışan veri tabanı; SQLite ve DuckDB tipik örneklerdir."))

# F
flow.append(P("F", "LetterHead"))
flow.append(E("Fare Class Optimization", "Gelir yönetiminde, koltukları farklı fiyat sınıflarına bölerek toplam geliri maksimize etme süreci; EMSR-b standart algoritmadır."))
flow.append(E("Feature Engineering (Öznitelik Mühendisliği)", "Ham veriden modele anlamlı girdi üretme sanatı; Seatwise&rsquo;da DTD bucketing, rolling pace, calendar göstergeleri vb. üretilir."))
flow.append(E("Fine-tuning (İnce Ayar)", "Önceden eğitilmiş bir modelin, daha küçük bir veri kümesi üzerinde göreve özel olarak yeniden eğitilmesi; DeBERTa-v3-small&rsquo;ın SST-2 üzerinde fine-tune edilmiş hali bunun örneğidir."))

# G
flow.append(P("G", "LetterHead"))
flow.append(E("Gradient Boosting", "Zayıf öğrenicilerin (genelde karar ağaçlarının) ardışık olarak hata azaltmaya yönelik birleştirildiği makine öğrenmesi yöntemi; XGBoost bunun ölçeklenebilir bir uygulamasıdır."))

# H
flow.append(P("H", "LetterHead"))
flow.append(E("Hold-out Test Set", "Eğitime sokulmamış, modelin gerçek genelleme performansını ölçmek için ayrılan veri kısmı; Seatwise XGBoost ailesi 18M+ satırlık hold-out test setlerinde değerlendirilmiştir."))
flow.append(E("Hurdle Model", "Sınıflandırıcı + regresör birleşiminden oluşan iki-aşamalı yapı; Cragg (1971) double-hurdle modelinin ML adaptasyonu Seatwise&rsquo;ın <i>two-stage XGBoost</i>&rsquo;udur."))
flow.append(E("Hyperparameter (Hiperparametre)", "Modelin eğitim sırasında öğrenmediği, baştan sabitlenen parametre (örn. learning rate, ağaç derinliği). Hyperparameter tuning bu değerlerin sistematik aranmasıdır."))

# I
flow.append(P("I", "LetterHead"))
flow.append(E("Idempotent (Tekrarlanabilir)", "Aynı işlemin birden fazla kez uygulanması ile bir kez uygulanmasının aynı sonucu vermesi özelliği; Seatwise build script&rsquo;leri idempotent çalışır."))
flow.append(E("In-Process Database", "Bağımsız bir sunucu süreci olmadan, uygulamanın belleği içinde çalışan veri tabanı; DuckDB bu kategorinin lider örneğidir."))

# K
flow.append(P("K", "LetterHead"))
flow.append(E("K-Means Clustering (K-Ortalamalar Kümelemesi)", "Veriyi K adet kümeye, küme içi varyansı minimize edecek şekilde ayıran gözetimsiz öğrenme algoritması."))

# L
flow.append(P("L", "LetterHead"))
flow.append(E("Load Factor Management", "Doluluk oranını hedeflenen seviyede tutma stratejisi; Seatwise&rsquo;da fare class kapama/açma kuralları bu hedefe hizmet eder."))

# M
flow.append(P("M", "LetterHead"))
flow.append(E("Monte Carlo Simulation", "Stokastik bir süreçten çok sayıda bağımsız örneklem alınarak istatistiksel özet (ortalama, varyans, güven aralığı) çıkarılması yöntemi; Glasserman (2004) standart kitabıdır."))
flow.append(E("Multi-horizon Forecasting", "Birden fazla geleceğe dair zaman noktasının tek bir modelle tahmin edilmesi; TFT bu paradigmayı destekler."))

# N
flow.append(P("N", "LetterHead"))
flow.append(E("Negative Binomial Distribution", "Sayma verisinde overdispersion (varyans &gt; ortalama) olan durumlarda Poisson dağılımının yerine kullanılan dağılım; Cameron &amp; Trivedi (2013) referans çalışmasıdır."))

# O
flow.append(P("O", "LetterHead"))
flow.append(E("Overbooking", "Kapasitenin üstünde rezervasyon kabul etme stratejisi; no-show kayıplarını telafi eder ancak denied boarding riski getirir. Seatwise&rsquo;da %5 toleranslı uygulanır."))
flow.append(E("Overdispersion", "Bir sayma değişkeninin varyansının ortalamadan büyük olması; Poisson modelinin yetersiz kaldığı durum, Negative Binomial alternatif çözümüdür."))

# P
flow.append(P("P", "LetterHead"))
flow.append(E("Pickup Forecasting", "Belirli bir andan kalkışa kadar gerçekleşmesi beklenen kalan talebin tahmini; havayolu RM literatüründe (Wickham 1995) klasik problemdir."))
flow.append(E("Pinball Loss / Quantile Loss", "Quantile-tabanlı tahminleri ölçen kayıp fonksiyonu: L<sub>q</sub>(y, ŷ) = max(q(y&minus;ŷ), (q&minus;1)(y&minus;ŷ)). TFT&rsquo;nin eğitim hedefidir."))
flow.append(E("Polarity (Polarite)", "Bir metnin olumlu veya olumsuz tonunun derecesi; Seatwise&rsquo;da DeBERTa [&minus;1, +1] aralığında sürekli polarite skoru üretir."))
flow.append(E("Predicate Pushdown", "Sorgu motorunun WHERE filtresini doğrudan dosya okuyucusuna iletip ilgisiz veri bloklarını disk düzeyinde atlama tekniği; Parquet&rsquo;in temel performans özelliklerinden biridir."))
flow.append(E("Price Elasticity (Fiyat Elastikiyeti)", "Talebin fiyat değişimine duyarlılığı; &epsilon; = %&Delta;Q / %&Delta;P. Seatwise&rsquo;ta 6 yolcu segmentinin elastikiyetleri &minus;0.1 (Emergency) ile &minus;2.2 (Budget flexible) arasında değişir."))

# R
flow.append(P("R", "LetterHead"))
flow.append(E("Recency Decay (Güncellik Azalması)", "Eski verinin etkisini zamanla üstel olarak azaltan ağırlıklandırma; Seatwise sentiment modülü r(h) = e<sup>&minus;0.05h</sup> kullanır."))
flow.append(E("Reproducibility (Tekrar Üretilebilirlik)", "Aynı kod, veri ve tohumla aynı sonucun yeniden üretilebilmesi; akademik çalışmaların temel kalite kriteridir."))
flow.append(E("Revenue Lift", "Dinamik fiyatlandırma sisteminin statik baseline&rsquo;a kıyasla sağladığı yüzdelik gelir artışı; Seatwise pilot testte +%10.68, 300-senaryo validasyonda +%28.26&rsquo;dır."))

# S
flow.append(P("S", "LetterHead"))
flow.append(E("Seed (Tohum)", "Rasgele sayı üretecini başlatan sabit değer; aynı tohum aynı rastgele sayı dizisini üretir. Reproducibility için kritiktir."))
flow.append(E("Sentiment Analysis (Duygu Analizi)", "Metnin duygusal tonunu (olumlu/olumsuz/nötr) otomatik olarak çıkaran NLP alt-alanı; Seatwise&rsquo;da DeBERTa + keyword classifier hibrit yapısıyla uygulanır."))
flow.append(E("Sequence-to-sequence Learning", "Bir girdi dizisinden çıktı dizisine eşleme öğrenen mimari; çeviri, özet ve TFT gibi çoklu-ufuk tahmininde kullanılır."))
flow.append(E("Spill", "Yield management terimi: yüksek WTP&rsquo;li yolcuların düşük fiyatlı sınıflara satılması (gelirin &ldquo;dökülmesi&rdquo;). Fare class kapama kurallarının amacı bunu engellemektir."))
flow.append(E("Static Pricing", "Booking horizon boyunca fiyatın değişmediği klasik strateji; Seatwise&rsquo;ın baseline karşılaştırma noktasıdır."))
flow.append(E("Stochastic Process (Stokastik Süreç)", "Zaman içinde rastgele evrilen değişkenler topluluğu; havayolu rezervasyon süreci doğası gereği stokastiktir."))

# T
flow.append(P("T", "LetterHead"))
flow.append(E("Threat Ratio", "Sentiment alarm kalibrasyonunda kullanılan oran: bir şehre ait makaleler içindeki güvenlik tehdidi sınıfındakilerin payı. Seatwise alarm kuralı &tau; &ge; 0.20&rsquo;da <i>yüksek</i> seviye tetikler."))
flow.append(E("Tidy Data", "Wickham (2014) tarafından tanımlanan veri organizasyonu prensibi: her satır bir gözlem, her sütun bir nitelik, her tablo tek bir analitik amaca yönelik. Seatwise işlenmiş katmanın temel ilkesidir."))
flow.append(E("Tornado Diagram", "Hassasiyet analizinde her parametrenin etki büyüklüğünü sıralı bar chart olarak gösteren görselleştirme; Bölüm 3.6&rsquo;daki çarpan etkilerinin sıralanması bu yaklaşımı izler."))
flow.append(E("Transformer", "Vaswani ve diğ. (2017) tarafından tanıtılan, attention mekanizmasına dayanan derin öğrenme mimarisi; modern NLP&rsquo;nin temel taşıdır."))
flow.append(E("Two-stage XGBoost", "Sınıflandırıcı (booking olur mu?) ve regresör (kaç bilet?) olmak üzere iki ardışık aşamadan oluşan model; sparse demand verisinde tek-aşamalı modelden üstündür."))

# V
flow.append(P("V", "LetterHead"))
flow.append(E("Vectorized Execution", "Sorgu motorunun veriyi tek tek satırlar yerine küçük bloklar (vector) hâlinde işlemesi; modern CPU&rsquo;nun SIMD birimlerini kullanarak büyük performans kazancı sağlar. DuckDB bu mimariye sahiptir."))

# W
flow.append(P("W", "LetterHead"))
flow.append(E("Warm-up Period", "Simülasyonun ilk geçici (geçiş) döneminin istatistiksel hesaplamadan dışlanması; Seatwise alternatif olarak ön-yükleme aşamasıyla bu sorunu çözer."))

# Y
flow.append(P("Y", "LetterHead"))
flow.append(E("Yield Management", "Sabit kapasiteli ürünleri (uçak koltuğu, otel odası) farklı fiyat sınıflarıyla heterojen müşteri segmentlerine satarak gelir maksimize etme disiplini; Belobaba (1989) ile başlar."))

# Z
flow.append(P("Z", "LetterHead"))
flow.append(E("Zero-inflation", "Hedef değişkende beklenenden fazla sıfır gözlem bulunması durumu; Seatwise demand_training verisinde zero-rate %70.8&rsquo;dir, two-stage modelin gerekçesidir."))

# ── Build ──
DESKTOP.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=2.0*cm, rightMargin=2.0*cm,
    topMargin=2.2*cm, bottomMargin=2.0*cm,
    title="Terimler ve Kisaltmalar Sozlugu",
    author="Group 16 - Seatwise",
)
doc.build(flow, onFirstPage=on_page, onLaterPages=on_page)
print(f"OK -> {OUT}")
print(f"Size: {OUT.stat().st_size/1024:.1f} KB")
