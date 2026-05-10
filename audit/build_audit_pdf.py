"""Seatwise — Sadece sorun listesi PDF."""
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)

# Türkçe karakter desteği
_BASE_FONT = "Helvetica"
_BOLD_FONT = "Helvetica-Bold"
for cand_reg, cand_bold in [
    (r"C:\Windows\Fonts\calibri.ttf", r"C:\Windows\Fonts\calibrib.ttf"),
    (r"C:\Windows\Fonts\arial.ttf",   r"C:\Windows\Fonts\arialbd.ttf"),
    (r"C:\Windows\Fonts\DejaVuSans.ttf", r"C:\Windows\Fonts\DejaVuSans-Bold.ttf"),
]:
    if os.path.exists(cand_reg):
        try:
            pdfmetrics.registerFont(TTFont("Body", cand_reg))
            _BASE_FONT = "Body"
            if os.path.exists(cand_bold):
                pdfmetrics.registerFont(TTFont("BodyBold", cand_bold))
                _BOLD_FONT = "BodyBold"
            else:
                _BOLD_FONT = "Body"
            break
        except Exception:
            pass

INK    = HexColor("#1f2937")
BODY   = HexColor("#374151")
MUTED  = HexColor("#6b7280")
BORDER = HexColor("#e5e7eb")

CRIT   = HexColor("#b91c1c")
HIGH   = HexColor("#c2410c")
MED    = HexColor("#b45309")
LOW    = HexColor("#166534")

S = {
    "title":  ParagraphStyle("T", fontName=_BOLD_FONT, fontSize=20, leading=24, textColor=INK),
    "sub":    ParagraphStyle("ST", fontName=_BASE_FONT, fontSize=10, leading=14, textColor=MUTED),
    "num":    ParagraphStyle("N", fontName=_BOLD_FONT, fontSize=11, leading=14, textColor=INK),
    "issue_title": ParagraphStyle("IT", fontName=_BOLD_FONT, fontSize=10.5, leading=14, textColor=INK),
    "body":   ParagraphStyle("B", fontName=_BASE_FONT, fontSize=10, leading=14, textColor=BODY,
                              alignment=TA_LEFT),
    "small":  ParagraphStyle("SM", fontName=_BASE_FONT, fontSize=8, leading=10, textColor=MUTED),
}


def issue_block(num, sev, title, problem, effect):
    """Sade bulgu kutusu. Severity şerit rengiyle ima edilir."""
    color = {"KRITIK": CRIT, "ONEMLI": HIGH, "ORTA": MED, "HAFIF": LOW}[sev]
    label = {"KRITIK": "KRİTİK", "ONEMLI": "ÖNEMLİ", "ORTA": "ORTA", "HAFİF": "HAFİF",
             "HAFIF": "HAFİF"}[sev]
    hx = "#" + color.hexval()[2:]

    head = Paragraph(
        f"<b>{num}.</b> &nbsp; {title} &nbsp; "
        f"<font color='{hx}' size='8'>· {label}</font>",
        S["issue_title"])
    rows = [
        [head],
        [Paragraph(f"<b>Sorun:</b> {problem}", S["body"])],
        [Paragraph(f"<b>Etki:</b> {effect}", S["body"])],
    ]
    t = Table(rows, colWidths=[17 * cm])
    t.setStyle(TableStyle([
        ("LINEBEFORE", (0, 0), (0, -1), 2, color),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return KeepTogether(t)


# ═══════════════════════════════════════════════════════
out_path = r"C:\Users\ahmet\OneDrive\Desktop\Seatwise_Sorun_Listesi.pdf"
doc = SimpleDocTemplate(out_path, pagesize=A4,
                        leftMargin=2 * cm, rightMargin=2 * cm,
                        topMargin=2 * cm, bottomMargin=2 * cm,
                        title="Seatwise Sorun Listesi")
e = []
W = 17 * cm

# Sade kapak
e.append(Paragraph("Seatwise — Sorun Listesi", S["title"]))
e.append(Spacer(1, 4))
e.append(Paragraph(
    f"33 bulgu · en önemliden en az önemliye sıralı · "
    f"{datetime.now().strftime('%d.%m.%Y')}",
    S["sub"]))
e.append(Spacer(1, 12))
e.append(HRFlowable(width=W, thickness=0.5, color=INK))
e.append(Spacer(1, 14))


issues = [
    # ── KRİTİK (5) ─────────────────────────────────────
    ("KRITIK",
     "Arama barında havalimanı kodu seçildiğinde sayfa kilitleniyor",
     "Kullanıcı arama barına 'IST' veya başka bir havalimanı kodu yazıp dropdown'dan "
     "'All Flights' seçtiğinde sistem o havalimanına ait <b>tüm tarihsel uçuşları</b> "
     "(IST için yaklaşık 102.000 satır, ~8 MB) tek seferde tarayıcıya gönderiyor. "
     "Frontend bu listeyi tek tek karta çevirmeye çalışırken tarayıcı kilitleniyor. "
     "Yükleniyor çubuğu görünür kalır, hiçbir şey yüklenmez.",
     "Hoca veya yönetici demonun ilk 30 saniyesinde bu senaryoyu denerse sistem donar. "
     "Sayfa yenileme gerekir; demo akışı ilk adımda kırılır. Uçuş numarası ile arama "
     "(TK60937 gibi) sorunsuz çalışır, sadece havalimanı kodu seçimi tehlikelidir."),

    ("KRITIK",
     "Manager Analysis what-if aracı her zaman %0 etki raporluyor",
     "Bir uçuş seçilip fiyat slider'ı hareket ettirildiğinde gelir delta, kalan talep ve "
     "demand değişimi her zaman sıfır olarak görünür. Slider -%30 ile +%50 arasında "
     "oynatılsa bile sonuç değişmez.",
     "Yönetici aracın çalışmadığını düşünür. Demoda fiyat ayarı yapıldığında ekranda hiçbir "
     "reaksiyon görülmez; stratejik karar üretilemez."),

    ("KRITIK",
     "Manager Analysis paneli geçmişte uçmuş uçuşları gösteriyor",
     "Panelde listelenen tüm uçuşların kalkışına 0 gün yazıyor. Sistem her uçuşun son "
     "(terminal) durumunu çekiyor; oysa amaç henüz uçmamış uçuşlarda karar vermek.",
     "Yönetici 'bu uçuş için fiyat ayarlayım' der ancak uçuş zaten kalkmıştır. Aracın "
     "varlık amacı işlevsiz hale gelir."),

    ("KRITIK",
     "Manager Analysis KPI özeti gerçek tabloyla uyuşmuyor",
     "Panelin üst kutucuklarında görünen toplam uçuş, ortalama doluluk, ortalama fiyat ve "
     "toplam gelir değerleri binlerce uçuşun ortalamasıdır. Tablo aşağıda yalnızca 100 uçuş "
     "gösterir. KPI'larda 'toplam gelir 15 milyar dolar' gibi gerçekçi olmayan rakamlar belirir.",
     "Yönetici özet ile detayın aynı veriyi anlattığını sanır; gerçekte iki farklı küme "
     "kıyaslanmıştır. Demoda 'bu rakam doğru mu?' sorusu gelirse açıklama zorlaşır."),

    ("KRITIK",
     "BiletBul müşteri uygulamasında rezervasyon butonu sistem hatası verebilir",
     "BiletBul'da uçuş seçildikten sonra 'Rezerve Et' butonuna tıklanınca müşteri sistemi "
     "ile yönetim sistemi arasındaki uyumsuzluk yüzünden teknik hata çıkar; ödeme akışı "
     "tamamlanamaz.",
     "Müşteri uçuşu seçer, kart bilgilerini girer, butona basar ve hata ekranı görür. Demo'da "
     "müşteri portalı tanıtılırken son adımda kırılma yaşanır."),

    # ── ÖNEMLİ (13) ────────────────────────────────────
    ("ONEMLI",
     "TFT 'trend' bilgisi yanıltıcı yazıyor",
     "Manager Analysis'te TFT tahmini bölümünde 'trend: rising' (yükselişte) yazısı görünür. "
     "Aslında bu yazı trend değil; tahmin edilen kalan talebin kalan koltuğa oranını söyler.",
     "Yönetici 'talep yükselişte, fiyatı artırayım' diye yanlış yorum yapabilir. Aslında "
     "uçuşa olan talep kapasiteden fazladır; yorum farklı olmalıdır."),

    ("ONEMLI",
     "Network optimizer 'protection levels' tablosunda tüm değerler 0 görünüyor",
     "Manager Analysis'in ağ önerileri bölümünde fare class koruma tablosu gösterilir; tüm "
     "kontenjan değerleri 0 olarak listelenir. Algoritma çağrılırken bir parametre eksik "
     "geçildiği için EMSR-b koruma anlamsız çıktı üretir.",
     "Yönetici 'koruma seviyesi 0 demek hiçbir koltuk korunmuyor' diye yorum yapar; "
     "gerçekte hesap eksik girdiyle yapılmıştır."),

    ("ONEMLI",
     "BiletBul'da geçmiş fiyat eğrisi uydurmadır",
     "Müşteri uçuş detayına girdiğinde 'son 180 günün fiyat trendi' grafiği çizilir. Gerçek "
     "tarihsel veri yoksa sistem matematiksel formülle düzgün bir eğri uydurur ve müşteriye "
     "tarihsel veri olarak gösterir.",
     "Müşteri 'fiyat şuana kadar düşmüş, daha düşecek' diye yanlış değerlendirir. Hocalara "
     "demoda gösterilirse 'bu veri nereden geliyor?' sorusu cevapsız kalabilir."),

    ("ONEMLI",
     "BiletBul'da rakip fiyatları sabit oranla üretiliyor",
     "Uçuş kartında Pegasus ve Emirates fiyatları gösterilir. Rakip simülasyon verisi yoksa "
     "Pegasus her zaman bizim fiyatın %25 altı, Emirates her zaman %20 üstü olarak yazılır. "
     "Gerçek rakip fiyat dinamiği yoktur.",
     "Müşteri rakip karşılaştırması yaparken yanılır. Demoda bu soru sorulursa açıklama "
     "zorlaşır; pazar gerçeği yansımaz."),

    ("ONEMLI",
     "Sentiment ekranında alert seviyesi skorla çelişebiliyor",
     "Sentiment haritasında bir şehir HIGH ALERT rozeti taşır ama composite skor +0.02 "
     "(neredeyse nötr veya hafif pozitif) görünür. Mantıksız bir kombinasyon oluşur.",
     "Kullanıcı 'pozitif skor neden HIGH ALERT?' der; ekran çelişkili görünür. Demoda soru "
     "gelirse tutarlı açıklama yapmak zorlaşır."),

    ("ONEMLI",
     "Sentiment modülü 'DeBERTa' diyor ama gerçekte basit kelime tarama yapıyor",
     "Sistem başlatılırken konsola '[Sentiment] v2 ready (GDELT + DeBERTa)' yazısı düşer. "
     "DeBERTa modern bir derin öğrenme modelidir, ancak modülde kullanılmıyor; aslında "
     "yaklaşık 416 kelimelik sabit liste ile kategorize ediliyor.",
     "Akademik raporlamada 'DeBERTa kullanıldı' iddiası varsa savunma sırasında tutarsızlık "
     "ortaya çıkabilir."),

    ("ONEMLI",
     "DTD simülasyonu doluluk oranını da değiştirmiyor",
     "Manager Analysis'te 'Bu uçuşu kalkışa 90 gün varken simüle et' seçeneği vardır. Sistem "
     "fiyatı yeni DTD'ye göre hesaplar ama doluluk oranı (örn. %93) sabit kalır. DTD 90 "
     "günde %93 doluluk gerçekçi değildir.",
     "Yönetici simülasyonda kendi kendiyle çelişen bir senaryo görür: kalkışa 90 gün var "
     "ama uçak %93 dolu. Sonuçlara güven azalır."),

    ("ONEMLI",
     "Sentiment skorunun ekrandan ekrana etki büyüklüğü farklı",
     "Aynı sentiment skoru üç farklı yerde farklı büyüklükte uygulanır: fiyatta %15, Manager "
     "Analysis'te talepte %20, simülasyonda talepte %30. Yönetici aynı haberi üç araçta "
     "test ederse üç farklı sonuç görür.",
     "Stratejik tutarsızlık. Manager Analysis'ten 'talep %20 düşer' beklenen senaryo, "
     "tam simülasyon koşturulduğunda %30 düşüş üretir; aynı veriyle iki farklı sonuç."),

    ("ONEMLI",
     "Form alanlarına geçersiz değer girilince sistem hatası çıkıyor",
     "Geçersiz parametre eklendiğinde (tarih alanına 'invalid', sayı alanına 'abc') sistem "
     "'500 Internal Error' sayfası gösterir. Kullanıcıya anlamlı hata mesajı verilmez.",
     "Hoca veya yönetici test sırasında deneme yaparken anlaşılmaz teknik hata ekranıyla "
     "karşılaşır."),

    ("ONEMLI",
     "Slider'a manuel %99999 girilince fiyat 1000 katı hesaplanıyor",
     "Manager Analysis fiyat slider'ı normalde -%30 ile +%50 arasında hareket eder. Manuel "
     "kutuya 99999 yazılırsa sistem üst sınır kontrolü yapmadan hesaplar; $429 olan fiyat "
     "$429.664'e çıkar.",
     "Yönetici yanlışlıkla yüksek değer girerse aracın anlamsız sayılar üretmesi güvenilirliği "
     "zedeler."),

    ("ONEMLI",
     "Frontend grafiklerinde 'Erken Dönem', 'Son Hafta' gibi Türkçe etiketler var",
     "Backend tarafında DTD etiketleri İngilizceye çevrilmiş ('Early Period', 'Final Week'). "
     "Ancak Manager Analysis bilgi kutusunda hâlâ Türkçe metinler görünür: 'Açık sınıflar: "
     "Erken Dönem' gibi.",
     "Arayüzün geri kalanı İngilizce iken bu kısım Türkçe görünür; profesyonel olmayan dil "
     "karışımı izlenimi verir."),

    ("ONEMLI",
     "Manager Analysis paneli 5-15 saniye yükleniyor",
     "Panel açılırken 90 günlük 25.000+ uçuşu işler. Her uçuş için fiyat motoru ayrı ayrı "
     "çalışır. Açılış süresi 5-15 saniyeye yayılır; bu sürede beyaz ekran görünür.",
     "Hoca veya yönetici 'sistem donmuş' düşünür; sayfa yenileme yapabilir. Demo akıcılığı "
     "bozulur."),

    ("ONEMLI",
     "Sistem yeniden başlatıldıktan sonra sentiment ekranı 60 saniye boş kalıyor",
     "Uygulama kapatılıp tekrar açıldığında sentiment cache 2 saatten eski sayılır ve "
     "yüklenmez. Yeni scheduler 51 şehri tek tek tarar (~60 saniye). O sürede sentiment "
     "haritası boştur.",
     "Demo başlatıldıktan hemen sonra sentiment ekranı tıklanırsa veri görünmez. Hoca "
     "'sentiment çalışmıyor' yorumu yapabilir."),

    ("ONEMLI",
     "Tahmin modellerinin başarı rakamları gerçek üretim performansını yansıtmıyor",
     "Pickup modeli %9.82 hata oranı, TFT MAE 14 olarak raporlanır. Bu rakamlar model "
     "eğitiminde kullanılan verilerle hesaplandı; ancak modelin yararlandığı bazı bilgiler "
     "üretim ortamında mevcut değildir.",
     "Akademik raporlamada 'modelimiz %90 doğrulukta' iddiası yapılırsa gerçek hayatta bu "
     "performans yakalanamayabilir. Hocaya dürüst sunum: 'validasyon koşullarındaki üst "
     "sınır'."),

    # ── ORTA (10) ──────────────────────────────────────
    ("ORTA",
     "Login ekranında 'SeatWise' büyük W yazıyor — diğer sayfalarda 'Seatwise'",
     "Login sayfası başlığı 'SeatWise — Login' olarak çıkar. Daha önce tercih edilen yazım "
     "'Seatwise' (küçük w) idi.",
     "Marka tutarsızlığı; ilk izlenim profesyonelliği zedeler."),

    ("ORTA",
     "Müşteri uygulaması ile yönetim uygulaması ayrı hesap gerektiriyor",
     "BiletBul ve Seatwise yönetim paneli ayrı kullanıcı veritabanlarını kullanır. Aynı kişi "
     "her iki uygulamada ayrı kayıt olmalı, ayrı şifreyi yönetmelidir.",
     "Yöneticinin hem yönetim sistemi hem müşteri arayüzünü test etmesi için iki ayrı "
     "kayıt yapması gerekir; akıcı değil."),

    ("ORTA",
     "BiletBul rakip fiyat fallback listesi 50 rota ile sınırlı",
     "BiletBul ana sistem ile bağlantı kuramazsa hardcoded 50 rotalık bir liste kullanır. "
     "Yeni rota eklendiğinde bu liste manuel güncellenmedikçe yeni rotalar görünmez.",
     "İlerleyen aylarda yeni rota eklendiğinde müşteri portalı 'rota yok' veya eski rotaları "
     "gösterir; bakım ihmalinde hata gizli kalabilir."),

    ("ORTA",
     "DTD override fare class doğru ama doluluk yanlış aynı senaryoda görünür",
     "DTD override yapılırken sistem fare class kurallarını yeni DTD'ye göre günceller (V/K/M/Y "
     "açık/kapalı durumu doğru çıkar). Ancak doluluk oranı sabit kalır.",
     "Sonuç: DTD 90 göstergesi kısmen mantıklı (fare class doğru), kısmen mantıksız (LF "
     "değişmiyor). Yönetici tek bir senaryoda iki farklı zaman algısı görür."),

    ("ORTA",
     "İptal ve no-show oranları kabin ayrımı yapmıyor",
     "Sistem iptal ve no-show oranlarını fare class'a göre (V/K/M/Y) farklılaştırır, ancak "
     "kabin (economy/business) ayrımı yapmaz. Gerçek havayolu pratiğinde business no-show "
     "%12-15, leisure %3-5 farklıdır.",
     "Business ağırlıklı bir uçuş için no-show beklentisi gerçeği yansıtmaz; kapasite "
     "planlaması saptar."),

    ("ORTA",
     "Sentiment skorunda 'security threat' kategorisi otomatik HIGH ALERT yapıyor",
     "Bir şehirde tek bir 'security_threat' kategorisinde haber varsa, composite skor +0.5 "
     "bile olsa alert seviyesi otomatik HIGH'a çekilir.",
     "Mantıklı olabilir ama görsel olarak çelişkili durur. Belge eksikliği nedeniyle "
     "kullanıcı şaşırır."),

    ("ORTA",
     "Sentiment skorlama sistemi kelime kök eşleşmesi yapmıyor",
     "Sistem 'smuggle' kelimesini 'smuggling' anahtar kelimesiyle eşleştirmez (tam eşleşme "
     "yapar). 'Drug mule arrested for trying to smuggle...' gibi haberler 'general_news' "
     "kategorisine düşer; aslında 'security_threat' olmalıdır.",
     "Bazı önemli haberler kategorize edilmez; sentiment skoru gerçekteki olumsuzluğu "
     "yakalamaz."),

    ("ORTA",
     "Sentiment 'tourism_growth' kategorisi çok geniş",
     "Anahtar kelime listesinde 'food', 'restaurant', 'beach', 'museum' gibi normal turizm "
     "terimleri var. Şehir sentiment'i için her seyahat yazısı otomatik +0.5 puan alır.",
     "Sentiment skoru pozitife yapay yatkın olabilir; kullanıcı 'her şehir iyi' izlenimi "
     "yaşar."),

    ("ORTA",
     "Manager Analysis duyarlılık grafikleri tarayıcı tarafında ayrı hesaplanıyor",
     "Slider hareket ettikçe çizilen 'fiyat-talep' ve 'fiyat-gelir' grafikleri tarayıcıda "
     "ayrı bir formülle hesaplanır. Sunucu tarafındaki düzeltmeler grafikte yansımıyor olabilir.",
     "Grafiklerin gösterdiği eğri sayısal olarak gerçek hesaplamadan farklı çıkabilir; iki "
     "çıktı arasında tutarsızlık doğar."),

    ("ORTA",
     "Sentiment haberlerinde tarih formatı modern değil",
     "Sentiment detay kartlarında 'Fri, 01 May 2026 00:00:12 GMT' gibi tarihler görünür. "
     "'01 Mayıs 2026' gibi modern formata çevrilmez.",
     "Kullanıcı tarihi okuyabilir ama profesyonel UI tercihiyle uyuşmaz."),

    # ── HAFİF (5) ──────────────────────────────────────
    ("HAFIF",
     "Uçuş kimliklerinde boşluk var, URL'lerde çirkin görünüyor",
     "Sistem uçuş kimliklerini 'TK60937_2026-05-05 00:12:00' (boşluklu) tutar. URL'lerde bu "
     "boşluk %20 olarak kodlanır.",
     "Son kullanıcı için sorun yok; geliştiricinin manuel testlerinde okunması zorlaşır."),

    ("HAFIF",
     "Login sonrası loading sayfasında animasyon geçişi yok",
     "Kullanıcı giriş yaptıktan sonra 'Loading...' sayfası 1-2 saniye görünür, sonra "
     "dashboard'a geçer. Yumuşak bir animasyon geçişi yok.",
     "İlk izlenim akıcılığı kısmen düşürür."),

    ("HAFIF",
     "Konsol log seviyesi tutarsız",
     "Bazı modüller bilgi mesajını basit print ile yazar, bazıları hiç loglamaz. Sistem "
     "yöneticisi debug yapmak istediğinde tutarlı log akışı yoktur.",
     "Son kullanıcıyı etkilemez; bakım pürüzü."),

    ("HAFIF",
     "Bazı yardımcı fonksiyonlarda dead code var",
     "İç kodda kullanılmayan veya iki kez hesaplanan değerler bulunur.",
     "Son kullanıcıya yansımaz; sadece kod bakımı açısından."),

    ("HAFIF",
     "Sentiment veritabanında bazı kayıtlarda 'tone' alanı boş",
     "Şema tasarımında haberler için tone (duygu yoğunluğu) alanı tanımlı, ancak haber "
     "kaynağı bu alanı doldurmadığından her kayıtta boş.",
     "Kullanıcıya yansımaz; sadece teknik bakımda göze batar."),
]


# Sıralama: severity önceliği zaten yukarıdaki sırada
order = {"KRITIK": 0, "ONEMLI": 1, "ORTA": 2, "HAFIF": 3}
issues_sorted = sorted(enumerate(issues, 1), key=lambda x: order[x[1][0]])

# Bulguları yazdır — kategori başlığı YOK
for idx, (orig_num, (sev, title, problem, effect)) in enumerate(issues_sorted, 1):
    e.append(issue_block(idx, sev, title, problem, effect))
    e.append(Spacer(1, 6))

# Alt bilgi
e.append(Spacer(1, 12))
e.append(HRFlowable(width=W, thickness=0.3, color=BORDER))
e.append(Spacer(1, 4))
e.append(Paragraph(
    f"33 sorun · {datetime.now().strftime('%d.%m.%Y %H:%M')}",
    S["small"]))

doc.build(e)
print(f"OK: {out_path}")
print(f"Boyut: {os.path.getsize(out_path):,} bayt")
