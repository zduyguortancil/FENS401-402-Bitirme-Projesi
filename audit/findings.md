# Seatwise — Bulgular (audit/findings.md)

**Sürüm:** seatwise_3 / FENS401-402-Bitirme-Projesi-Last Version_duygu(V.2)
**Tarih:** 2026-04-19
**Faz:** 1 (statik inceleme) + 2 (davranışsal test, kısmi)

---

## [CRITICAL] C-001 — Tüm API endpoint'leri auth'suz (yetkilendirme bypass)

**Lokasyon:** `dashboard/app.py` — tüm `/api/*` rotaları
**Kategori:** Security / Authorization
**Açıklama:** `/dashboard` sayfa rotası `@login_required` ile korunmakta, ancak `/api/flights`, `/api/snapshot/<id>`, `/api/forecast/<id>`, `/api/manager-analysis`, `/api/manager-override`, `/api/manager-sensitivity`, `/api/sentiment/*`, `/api/demand/*`, `/api/pickup/*`, `/api/clusters`, `/api/trends`, `/api/top-routes`, `/api/events`, `/api/risk-index` ve diğer 30+ endpoint'in hiçbiri `@login_required` taşımıyor. Login sayfası tamamen kozmetik bir engel — saldırgan login olmadan tüm gelir yönetimi verilerini, fiyat ayarlama API'sini, simülasyon kontrolünü çekebilir.

**Repro Adımları:**
1. Tarayıcıda `http://localhost:5005/dashboard` aç → 302, login'e atılırsın.
2. Doğrudan `http://localhost:5005/api/manager-analysis` aç → HTTP 200, tam JSON döner.
3. POST `http://localhost:5005/api/manager-override` ile fiyat ayarla → HTTP 200, hesaplama yapılır.
4. Beklenen: 401/403 ya da login'e yönlendirme. Gerçek: tam erişim.

**Etki:** Tüm RM kararları, müşteri segment dağılımları, sentiment skorları, pricing detayları herkese açık. BiletBul (customer) `/api/book` POST'u da `@login_required` taşımıyor — kimliksiz rezervasyon yapılabilir.

**Öneri:** API endpoint'lerine `@login_required` decorator'ı ekle. Veya bir `@app.before_request` filter ile `/api/*` yollarını otomatik korumalı hale getir:
```python
@app.before_request
def require_auth_for_api():
    if request.path.startswith("/api/") and not session.get("sw_user"):
        return jsonify({"error": "unauthorized"}), 401
```

---

## [CRITICAL] C-002 — Hardcoded session secret fallback

**Lokasyon:** `dashboard/app.py:38`, `dashboard/customer/customer_app.py:9`
**Kategori:** Security / Session forge
**Açıklama:** İki uygulama da `os.environ.get("SEATWISE_SECRET", "sw-desktop-secret-2026")` ve `os.environ.get("BILETBUL_SECRET", "bb-desktop-secret-2026")` şeklinde **tahmin edilebilir hardcoded secret** kullanıyor. Saldırgan bu secret ile geçerli session cookie üretebilir, herhangi bir kullanıcı kimliğini ele geçirebilir.

**Repro Adımları:**
1. Flask session API'sini kullanarak `{"sw_user": {"id": "1", "username": "admin"}}` payload'ını `"sw-desktop-secret-2026"` ile imzala.
2. `Set-Cookie: session=...` ile request gönder.
3. `/dashboard` admin yetkisiyle erişilir.

**Etki:** Tam yetki ele geçirme. Saldırgan admin olarak login olabilir, register/logout/data export yapabilir.

**Öneri:**
- Default değeri kaldır, env yoksa `RuntimeError` fırlat.
- `secrets.token_urlsafe(32)` ile ilk çalıştırmada üret ve diske yaz.
```python
app.secret_key = os.environ["SEATWISE_SECRET"]  # crash if missing
```

---

## [CRITICAL] C-003 — Şifre saltsız SHA256 hash

**Lokasyon:** `dashboard/app.py:56`, `dashboard/customer/customer_app.py:27`
**Kategori:** Security / Password storage
**Açıklama:** `hashlib.sha256(pw.encode()).hexdigest()` — salt yok, iterasyon yok. Saldırgan `users_db.json` dosyasını ele geçirirse rainbow table veya GPU brute-force ile saniyeler içinde çözer. NIST 2017'den beri bu yaklaşım önerilmiyor; 2026'da kabul edilemez.

**Repro Adımları:**
1. `users_db.json` veya backup'ı sızar.
2. Admin hash `5ce41ada...` rainbow table'da varsa anında çözülür; yoksa GPU ile 24 saatte 8-10 karakter şifre tahmin edilir.

**Etki:** Tüm kullanıcı şifreleri kırılır, yanal hareket riski (kullanıcılar aynı şifreyi başka servislerde kullanmış olabilir).

**Öneri:** `bcrypt`, `argon2-cffi` veya stdlib `hashlib.scrypt` ile değiştir:
```python
from werkzeug.security import generate_password_hash, check_password_hash
pw_hash = generate_password_hash(password, method="scrypt")
```

---

## [HIGH] H-001 — Brute force / rate limit / hesap kilitleme yok

**Lokasyon:** `dashboard/app.py:305-318` (login_post), customer_app.py:281-294
**Kategori:** Security
**Açıklama:** Login endpoint'i sınırsız deneme kabul ediyor. CAPTCHA, IP bazlı rate limit, başarısız deneme sayacı, hesap kilitleme — hiçbiri yok. Şifre regex'i 8 karakter zorunlu (büyük/küçük + rakam + özel), zayıf değil ama saldırgan zaman ile her şeyi dener.

**Etki:** Online brute force, credential stuffing.

**Öneri:** `flask-limiter` ile dakikada 5 deneme limiti, başarısız 5 deneme sonrası 15 dk hesap kilidi.

---

## [HIGH] H-002 — Manager Analysis hâlâ DTD=0 verisi getiriyor (K1 yarım çözüm)

**Lokasyon:** `dashboard/app.py:2622-2654` (api_manager_analysis SQL)
**Kategori:** Logic / Domain
**Açıklama:** Tarih filtresi eklenmiş (`dep_date >= today`, doğru) ancak `WITH latest AS (... MIN(dtd) AS min_dtd ...)` her uçuş için **terminal state**'i (DTD=0, kalkış günü verisi) çekiyor. Bugünden 16 gün sonra kalkacak bir uçuş için manager DTD=0 görmemeli — bugünün gerçek DTD'si (yani `dep_date - today`) görmeli.

**Repro Adımları:**
1. `curl 'http://localhost:5005/api/manager-analysis?per_page=5'`
2. Sample uçuş: `dep_date=2026-05-05`, `dtd=0` → Bugün 2026-04-19. Gerçek DTD = 16 olmalı.
3. Bu yüzden Manager Analysis'in fiyat hesabı kalkış günü kuralları (Y-only) ile yapılıyor.

**Etki:** Manager fiyat ayarlaması "%15 artır" derken, Y-only fare class kuralıyla işlem yapan bir uçuşta etki hesaplanıyor — yanlış strateji. **What-if sonuçları her zaman 0 demand impact gösteriyor** (aşağıdaki H-003'e bağlı).

**Öneri:** SQL'de `MIN(dtd)` yerine `dep_date - CURRENT_DATE` ile gerçek DTD hesabı, ardından bu DTD'ye en yakın snapshot'ı seçen `argmin(ABS(dtd - real_dtd))` mantığı.

---

## [HIGH] H-003 — Manager Override sonuçları DTD=0'da kollapse oluyor

**Lokasyon:** `dashboard/app.py:2892-2919` (`_estimate_cancellation_noshow`) + 3019-3036 demand calc
**Kategori:** Logic
**Açıklama:** Test çağrısı `TK60937 economy +15%`:
- `baseline_remaining: 0.0`
- `effective_remaining: 0.0`
- `revenue_delta_pct: 0.0`
- `cancellation_noshow.adjusted_bookable: 0`

Sebebi: DTD=0 → cancel_noshow'un `dtd_factor = 0.2` × ortalama cancel rate `0.06` = 0.012 → net_demand çok küçük. `_compute_expected_demand` `pickup_tft_blend` 0.6×pickup + 0.4×TFT band üretiyor; pickup model DTD=0'da düşük döndürüyor + TFT band cum_fraction=1.0'da neredeyse 0. Tüm pipeline çöküyor.

**Etki:** Manager hangi fiyat ayarlamasını yaparsa yapsın, "0% revenue impact" görüyor. Araç fonksiyonel olarak işe yaramaz.

**Öneri:**
1. H-002 düzeltildiğinde DTD=0 dışı pickup değerleri oluşur.
2. `_estimate_cancellation_noshow` formülünde DTD<7 için `dtd_factor=0.2` çok düşük; `0.5-0.8` daha gerçekçi (son hafta cancellation oranı düşer ama demand 0 olmaz).

---

## [HIGH] H-004 — Frontend DTD label'ları hâlâ Türkçe

**Lokasyon:** `dashboard/templates/index.html` — onMgrDtdChange() fonksiyonu (önceki audit Ö3 hâlâ açık)
**Kategori:** Consistency / UX
**Açıklama:** Backend'de DTD rules İngilizce'ye çevrildi (`Early Period`, `Mid Period` vb.) ama UI hâlâ "Erken Dönem", "Orta Dönem", "Son Hafta", "Son Dakika" gösteriyor; "Açık sınıflar:" prefix Türkçe.

**Etki:** UI tutarsızlığı; uluslararası bir bitirme projesinde dil karışımı.

**Öneri:** `index.html` içindeki DTD_RULES JS objesini İngilizceye çevir; "Açık sınıflar:" → "Open classes:".

---

## [HIGH] H-005 — Network optimizer protection quota'sı 0 dönüyor

**Lokasyon:** `dashboard/app.py:2868-2889` (`_compute_network_recommendations`) → `network_optimizer.compute_protection_levels`
**Kategori:** Logic
**Açıklama:** Test çıktısı:
```
"protection_levels": {"K": {"protected": 64, "quota": 0}, "M": {"protected": 64, "quota": 0}, "V": {"protected": 0, "quota": 0}, "Y": {"protected": 64, "quota": 0}}
```
Tüm fare class'ların `quota: 0`. EMSR-b'nin amacı yüksek-değerli class için koltuk koruyup düşük-değerli class'ı kapatmak; quota=0 → koruma yok ya da hesap çalışmıyor.

**Etki:** Network optimizer çıktısı yanıltıcı; manager bunlara güvenemez.

**Öneri:** `network_optimizer.compute_protection_levels` fonksiyonunu tek başına test et — protection vs quota semantiği nedir, neden hep aynı 64 dönüyor?

---

## [HIGH] H-006 — TFT forecast `trend` semantiği yanlış

**Lokasyon:** `dashboard/app.py:3083-3088`
**Kategori:** Logic
**Açıklama:**
```python
"trend": "rising" if tft_demand > remaining else "falling" if tft_demand < remaining * 0.8 else "stable"
```
Bu **trend** değil, **demand-supply gap**. Trend zaman içindeki değişimi anlatır; burada anlık tahmin (180) ile kalan koltuk (64) karşılaştırılıyor → yanlış semantic.

**Etki:** Manager "rising" trendi görüp "talep yükselişte, fiyat artırayım" yorumu yapabilir; aslında talep > arz (overdemand) durumu.

**Öneri:** Field adını `demand_supply_ratio` olarak değiştir veya gerçek trend için son 7 gündeki TFT tahmin serisini kullan.

---

## [HIGH] H-007 — Sentiment modülü "DeBERTa" diyor, keyword classifier kullanıyor (önceki audit kapanmamış)

**Lokasyon:** `dashboard/app.py:170` log, `sentiment/classifier.py`
**Kategori:** Yanıltıcı bilgi
**Açıklama:** Log "v2 ready (GDELT + DeBERTa)" yazıyor, gerçekte sadece keyword classifier (~416 kelime) çalışıyor. DB'deki 1507 makalenin tone alanı NULL. ML modeli yok.

**Etki:** Akademik raporlama / sunum için yanıltıcı. Performance metric'leri keyword classifier için geçerli, DeBERTa'ya dair iddialar desteksiz.

**Öneri:** Log mesajını "v2 ready (GDELT + RSS + Keyword Classifier)" olarak düzelt. README ve dokümantasyonu da uyumla.

---

## [HIGH] H-008 — Customer app `/api/book` doğrulamasız proxy

**Lokasyon:** `dashboard/customer/customer_app.py:511-517`
**Kategori:** Security / Validation
**Açıklama:**
```python
@app.route("/api/book", methods=["POST"])
def api_book():
    response = _dashboard_post("/api/pricing/book", request.json, timeout=4)
```
- `@login_required` yok
- `request.json` hiçbir validation'dan geçmeden Seatwise'a iletiliyor
- Manipüle edilmiş payload (negatif fiyat, başkasının kullanıcı ID'si, geçersiz fare class) doğrudan kabul edilebilir.

**Etki:** Sahte rezervasyon, düşük fiyat zorlama, başka kullanıcı adına işlem.

**Öneri:**
- `@login_required` ekle.
- Schema validation: pydantic veya marshmallow ile `flight_id`, `cabin`, `pax`, `total_price` alanlarını tip + aralık kontrolü.
- Backend tarafında fiyatı yeniden hesapla, müşterinin gönderdiğine güvenme.

---

## [HIGH] H-009 — `users_db.json` thread-safe değil (race condition)

**Lokasyon:** `_load_users` / `_save_users` fonksiyonları (her iki app'de)
**Kategori:** Concurrency
**Açıklama:** Register iki kullanıcı aynı anda denenirse: read → modify → write sırası ile birinin kaydı kaybolabilir. Ayrıca dosya kilitleme yok.

**Etki:** Veri kaybı; aynı username'in iki kez kaydedilebilme ihtimali (UNIQUE check race window).

**Öneri:** `fcntl.flock` (Unix) / `msvcrt.locking` (Windows) ile dosya kilidi, ya da SQLite'a geçiş.

---

## [HIGH] H-010 — `force=True` JSON parse + Content-Type bypass

**Lokasyon:** `app.py` ve `customer_app.py` login/register endpoint'leri
**Kategori:** Security / Robustness
**Açıklama:** `request.get_json(force=True)` Content-Type kontrolünü atlatır. Saldırgan `multipart/form-data` veya boş body ile crash deneyebilir. Boş body durumunda return `None`, sonra `or {}` ile boş dict olur, kontrol geçer ama beklenmeyen davranış.

**Etki:** Düşük; CSRF ile birleştirilirse anlamlı.

**Öneri:** `force=False` (default), Content-Type `application/json` kontrolü.

---

## [HIGH] H-011 — CSRF protection yok

**Lokasyon:** Tüm POST endpoint'leri
**Kategori:** Security
**Açıklama:** Flask-WTF kullanılmıyor, CSRF token yok. Saldırgan başka bir site üzerinden POST `/api/manager-override` veya `/api/book` tetikleyebilir.

**Etki:** Cross-site request forgery → istemsiz fiyat değişikliği, rezervasyon.

**Öneri:** `Flask-WTF` veya `flask-seasurf` kur, tüm POST'lara `csrf_token` zorunlu yap. AJAX için `X-CSRFToken` header.

---

## [MEDIUM] M-001 — `users_db.json`'da admin hesabı için zayıf veya bilinmeyen şifre

**Lokasyon:** `dashboard/users_db.json:7`
**Kategori:** Security / Operasyon
**Açıklama:** Admin hash `5ce41ada64f1...`. 30 yaygın aday (admin, password, Admin123! vb.) ile crack denedim, eşleşmedi — ama bu hash zaten production secret olmamalı.

**Etki:** Eğer bu repo public repoya pushlandıysa hash sızıntısı; saltsız SHA256 ile rainbow table aynı hash'i sorgulayabilir.

**Öneri:** Production'da admin user'ı manuel oluştur, default seed admin kaldır.

---

## [MEDIUM] M-002 — `_compute_expected_demand` `divide_by_zero` korumasında 1e-6

**Lokasyon:** `dashboard/app.py:3043`
**Kategori:** Logic
**Açıklama:** `(effective_remaining / max(baseline_bookable, 1e-6) - 1) * 100` — `baseline_bookable=0` olduğunda sonuç -100 olarak rapor ediliyor. Bu "%100 talep düştü" olarak gösteriliyor; gerçekte "baseline tanımsız" olmalıydı.

**Etki:** Yanıltıcı KPI; H-003 ile birleşirse her override "-%100 demand impact" gösterir.

**Öneri:** `if baseline_bookable < 1: return None` veya UI'da "n/a" göster.

---

## [MEDIUM] M-003 — `dashboard/.env` ve `<project>/.env` aynı NEWSAPI_KEY

**Lokasyon:** İki dosyada aynı key (`d206a062...`)
**Kategori:** Hygiene
**Açıklama:** İkili dosya gereksiz; hangisinin okunacağı `_load_dotenv` sırasına bağlı. Ayrıca:
- Bu key git'e push olmuş olabilir (kontrol edilmeli)
- `.gitignore`'da olmalı

**Öneri:** Bir dosya bırak, diğerini sil; key'i revoke et ve yenile.

---

## [MEDIUM] M-004 — Sentiment modülünde önceki audit'teki 23 sorun hâlâ açık

**Lokasyon:** `dashboard/sentiment/*.py`
**Kategori:** Logic / Consistency
**Açıklama:** Önceki sentiment audit'inde bulunan sorunlar (substring keyword matching, aviation context dead zone, alert/score çelişkisi, `cleanup_old` yıl bazlı silme, vb.) bu sürümde de açık. Sentiment modülü değiştirilmemiş.

**Etki:** Sentiment skoru güvenilirliği düşük; pricing/simulation'a girdi olarak kullanılırken yanlılık tetikler.

**Öneri:** Önceki sentiment raporunun (önceki sohbette üretilmiş) maddelerinin bu sürüme uygulanması.

---

## [MEDIUM] M-005 — Manager Analysis: tek SQL query 25k satır işliyor

**Lokasyon:** `app.py:2621-2654`
**Kategori:** Performance
**Açıklama:** Default 90 günlük tarih aralığında 25.476 uçuş dönüyor. Her uçuş için Python tarafında pricing engine `compute_price()` çağrılıyor → tek request başına 25.476 fiyat hesabı. Sayfa yüklemesi yavaş.

**Etki:** Manager Analysis paneli açılışı 5–15 saniye.

**Öneri:**
- DB seviyesinde pagination (`LIMIT/OFFSET` ile sadece sayfa başına 100 hesapla).
- Veya SQL ile pre-aggregate, Python loop sadece görünen 100 uçuş için pricing yapsın.
- Veya asynchronous loading: tablo önce gelir, fiyat sütunu sonradan AJAX ile.

---

## [MEDIUM] M-006 — `_dashboard_get` zaman aşımı 4 saniye, hata yutmacı

**Lokasyon:** `customer_app.py:66-83`
**Kategori:** Reliability
**Açıklama:** `_dashboard_get` ve `_dashboard_post` Seatwise'a HTTP çağrı yapıyor; timeout=4. Eğer Seatwise yavaşsa müşteri "search failed" görüyor. Hata logu sadece `print(...)`, akış sessizce fallback'e geçiyor.

**Etki:** Müşteri intermittent hatalarla karşılaşıyor; debugging zor.

**Öneri:** Retry (exponential backoff), structured logging, müşteriye anlamlı hata mesajı.

---

## [MEDIUM] M-007 — Customer app `_dashboard_post` body geri dönüş tutarsızlığı

**Lokasyon:** `customer_app.py:512-517`
**Kategori:** Bug
**Açıklama:**
```python
response = _dashboard_post("/api/pricing/book", request.json, timeout=4)
if response is None:
    return jsonify({"error": "..."}), 500
return jsonify(response.json()), response.status_code
```
`_dashboard_post` `requests.Response` obje döndürür mü, yoksa parse edilmiş dict mi? Tanımına bakılmalı; eğer parse edilmiş dict döndürüyorsa `.json()` ve `.status_code` AttributeError verir.

**Repro:** Test edilmemiş — `_dashboard_post` body'sini henüz inceledim.

**Öneri:** Sözleşmeyi netleştir; ya `Response` döndür ya parse et, karıştırma.

---

## [MEDIUM] M-008 — `api_search` exception'da generic 500

**Lokasyon:** `customer_app.py:350-352`
**Kategori:** UX
**Açıklama:** Herhangi bir hatada (TypeError, ValueError, IndexError, network) tek mesaj: "Customer search failed. Please retry." Kullanıcı sebebini anlayamıyor; geliştirici stacktrace görmek için sadece konsol log'una güveniyor.

**Öneri:** Hata sınıfına göre 400 vs 500 ayır; kullanıcıya validation hatalarını göster.

---

## [MEDIUM] M-009 — `api_flight_detail` synthetic fallback müşteriye uydurma fiyat eğrisi gösteriyor

**Lokasyon:** `customer_app.py:483-497`
**Kategori:** Logic / UX
**Açıklama:** Gerçek veri yoksa synthetic curve üretiliyor:
```python
thy_p = base_p + (180-i)*0.8 + (10 if i<30 else 0)
pc_p = thy_p * 0.75
ek_p = thy_p * 1.2
```
Müşteri bu eğriyi gerçek tarihsel fiyat olarak görüyor — aslında düz formül.

**Etki:** Müşteri hatalı fiyat trend bilgisi alıp karar veriyor (`bekleyim, fiyat düşecek mi?`).

**Öneri:** "Fiyat geçmişi mevcut değil" mesajı göster, synthetic eğri çizme. Veya UI'da "tahmini" rozeti.

---

## [MEDIUM] M-010 — `competitor estimate` hardcoded carpan

**Lokasyon:** `customer_app.py:464-465`
**Kategori:** Logic
**Açıklama:** Rakip fiyatları yoksa `pc = thy * 0.75`, `ek = thy * 1.2` — sabit oran. Pegasus her zaman %25 ucuz, Emirates %20 pahalı varsayımı; gerçek pazar dinamikleri yansıtılmıyor.

**Öneri:** Rakip verisi yoksa rakip kolonu boş bırak veya "n/a" göster.

---

## [MEDIUM] M-011 — Empty-DB / first-run davranışı

**Lokasyon:** `users_db.json` yoksa `_load_users` boş dict döner; ilk register başarılı olur. Ancak `app.secret_key` env'siz default → tahmin edilebilir secret + ilk admin oluşturma.
**Kategori:** Logic / Operasyon
**Açıklama:** İlk kurulumda kim ilk register olursa "manager" rolü alıyor. "admin" rolünü sadece seed admin alıyor. Ama register endpoint'inde role injection yok ✓ — bu güvenli.
**Öneri:** Yine de ilk register'ın admin olmasını engelleyen bir lock mekanizması.

---

## [LOW] L-001 — Dead code: `demand_change_pct` iki kez hesaplanıyor

**Lokasyon:** `app.py:3017` ve `app.py:3043`
**Açıklama:** İlk hesaplama (`(adjusted_demand_factor - 1) * 100`) kullanılmıyor — sat 3043'te override ediliyor.
**Öneri:** İlk hesabı sil veya iki ayrı isim ver (`raw_demand_change_pct` vs `net_demand_change_pct`).

---

## [LOW] L-002 — Pricing engine `_sentiment_multiplier` (×0.15) ile simulation `sentiment_factor` (×0.30) tutarsız

**Lokasyon:** `pricing_engine.py:321`, `simulation_engine.py:631`
**Açıklama:** Aynı sentiment skoru fiyatta `1.0 + score*0.15`, talepte `1.0 + score*0.30`. Belgesiz, gerekçesiz.
**Öneri:** Konstantları tek dosyada toplayıp dökümante et.

---

## [LOW] L-003 — `try/except: pass` defansif yutuma örnekleri

**Lokasyon:** `app.py:2694-2695`, `:2785-2828` (3 try/except), `:3005-3006`, `:3073-3079`, `customer_app.py:469-477`, `customer_app.py:480-481`
**Açıklama:** En az 7 yerde `try: ... except: pass` veya `except Exception: ...` sessizce yutum var. Ne hata loglanıyor ne kullanıcıya iletiliyor; sebep bilinmeden fallback'e dönülüyor.
**Öneri:** Spesifik exception type'larını yakala, en azından `print(...)` veya `logging.warning(...)` yaz.

---

## [LOW] L-004 — `flight_id` formatında boşluk URL encoding gerektiriyor

**Lokasyon:** `app.py` GET endpoint'leri (örn `/api/snapshot/<flight_id>`)
**Açıklama:** `flight_id = "TK60937_2026-05-05 00:12:00"` — boşluk URL'de `%20` veya `+` olarak encode edilmeli. Çoğu HTTP client otomatik yapar ama manuel test yaparken kafa karıştırıcı.
**Öneri:** `flight_id` formatını boşluksuz hale getir (örn `_` ile değiştir): `TK60937_2026-05-05T00-12-00`.

---

## [LOW] L-005 — `_estimate_cancellation_noshow` sabit oranlar

**Lokasyon:** `app.py:2895-2908`
**Açıklama:** Cancel rate `{V:0.01, K:0.03, M:0.08, Y:0.12}` ve no-show `{A:0.15, B:0.05, ...}` hardcoded. Veriden kalibrasyon yok; demand_functions_report.json'da no-show değerleri zaten var, neden onlar kullanılmıyor?
**Öneri:** `demand_functions_report.json`'dan oku.

---

## [LOW] L-006 — Manager Analysis frontend hâlâ önceki sürümün charts.js'ini kullanıyor olabilir

**Lokasyon:** `index.html` buildMgrCharts() — sensitivity hesabı frontend'de
**Açıklama:** Yeni `/api/manager-sensitivity` endpoint'i var (server-side curve), ancak frontend kodu hâlâ JS'de hesaplıyor olabilir. Frontend güncellemesi yapıldı mı doğrulanmadı.
**Öneri:** index.html'i `fetch('/api/manager-sensitivity')` çağıracak şekilde güncelle.

---

## [LOW] L-007 — Sentiment cache `load_cached_scores` 2 saatlik eşik

**Lokasyon:** `sentiment/cache_db.py` (önceki audit'te de var)
**Açıklama:** Sistem kapatılıp 2+ saat sonra açılırsa cache boş başlar, sentiment scheduler 60 sn'de tekrar oluşturuyor — ama o sürede UI veri görmüyor.
**Öneri:** `max_age_hours=24` veya `unlimited`.

---

## [LOW] L-008 — Logging seviyesi tutarsız

**Lokasyon:** Tüm modüller
**Açıklama:** Bazı yerlerde `print(...)`, bazı yerlerde hiç log yok, structured logging hiç yok. Production'da log toplama zorlaşır.
**Öneri:** `logging` modülü, JSON formatter, level'ları config'le.

---

## [HIGH] H-012 — Input validation yok: 500 crash kaynakları

**Lokasyon:** `app.py:2598-2599` (`int(request.args.get(...))`), `:2931` (`float(...)`)
**Kategori:** Robustness / DOS
**Açıklama:** Davranışsal testlerde aşağıdaki istekler **HTTP 500** üretiyor (validation yok):

| İstek | Sebep |
|---|---|
| `GET /api/manager-analysis?per_page=abc` | `int("abc")` crash |
| `GET /api/manager-analysis?date_from=invalid-date` | DuckDB `CAST('invalid-date' AS DATE)` crash |
| `POST /api/manager-override {"adjustment_pct": -100}` | `(1 + (-100)/100)^elasticity = 0^negatif` → OverflowError |
| `POST /api/manager-override {"adjustment_pct": null}` | `float(None)` TypeError |
| `POST /api/manager-override {"adjustment_pct": "abc"}` | `float("abc")` ValueError |

**Risk:** 500 stack trace üretmiyor (debug=False) ama saldırgan tetiklediği endpoint'e DoS yapabilir. Worker thread'leri tüketilir.

**Repro Adımları:**
1. `curl -X POST http://localhost:5005/api/manager-override -H "Content-Type: application/json" -d '{"flight_id":"TK60937_...","adjustment_pct":-100}'` → 500
2. `curl 'http://localhost:5005/api/manager-analysis?per_page=abc'` → 500

**Öneri:** Validation katmanı:
```python
try:
    page = max(1, int(request.args.get("page", 1)))
    per_page = max(1, min(500, int(request.args.get("per_page", 100))))
except ValueError:
    return jsonify({"error": "page/per_page must be integers"}), 400
```
ve `adjustment_pct` için `-99 < x < 1000` aralığı.

---

## [HIGH] H-013 — `adjustment_pct=99999` kabul ediliyor: $429 → $429.664

**Lokasyon:** `app.py:3009`
**Kategori:** Logic / UX
**Açıklama:** Manager Override'da `adjustment_pct` üst sınırı yok. `99999` girildiğinde:
- `original_price: $429.24`
- `adjusted_price: $429,664.95` (1000 kat)
- Frontend slider zaten -30..+50 arası clamp ediyordur, ama API başına direkt POST atılınca koruma yok.

**Etki:** Manager kazara veya kötü amaçlı saçma değer girerse sistem hesaplar; UI'a güveniyoruz. Defense-in-depth ihlali.

**Öneri:** Sunucu tarafı:
```python
adjustment_pct = max(-50, min(100, float(data.get("adjustment_pct", 0))))
```

---

## [MEDIUM] M-012 — `dtd_override` negatif değer kabul ediliyor: NaN davranışı

**Lokasyon:** `app.py:2982-2984`
**Kategori:** Logic
**Açıklama:** `dtd_override=-5` POST'lanınca:
- S-curve `1 - (-5/180)^1.5` → Python kompleks sayı veya nan
- `max(0, min(nan, 0.98))` → bazı versiyonlarda 0.98, bazılarında nan
- Sonuç: `lf=98%`, `pax_cum=294`, `dtd=-5` (mantıksız geçmiş)

**Risk:** Negatif DTD anlamsız (gelecekteki uçuş "geçmişe" simüle ediliyor). NaN değerler nadiren downstream'e sızar.

**Öneri:** `dtd_override = max(0, min(365, int(...)))`.

---

## [MEDIUM] M-013 — `_load_dotenv` env loading sırası tutarsız

**Lokasyon:** `app.py:19-33`
**Açıklama:** `dashboard/.env` ÖNCE okunuyor, sonra `<project>/.env`. İkisi farklı değer içerirse `dashboard/.env` kazanır. Belge yok.

**Öneri:** Tek dosya, ya da öncelik açıkça belgele.

---

# ════════════════════════════════════════════════
# FAZ 3 — Domain Logic Doğrulaması
# ════════════════════════════════════════════════

## [CRITICAL] C-004 — Pickup XGBoost'ta `route_total_pax` target leakage

**Lokasyon:** `scripts/data_prep/build_pickup_master.py:124`, `scripts/training/train_pickup_xgb.py`
**Kategori:** ML / Data Leakage
**Açıklama:** Pickup model'in feature listesine `route_total_pax`, `route_n_bookings`, `avg_fare`, `corporate_pct`, `agency_pct`, `connecting_pct`, `early_booking_pct`, `late_booking_pct`, `child_pct`, `halal_pct` dahil ediliyor. Bunlar `tft_route_daily.parquet`'tan **uçuşun kendi günündeki, kendi rotasındaki aggregate metrikler**:

```sql
LEFT JOIN rdaily r ON m.route = r.route
    AND LOWER(d.cabin_class) = LOWER(r.cabin_class)
    AND CAST(...flight_id..._dep_date AS DATE) = r.dep_date
```

`route_total_pax` = o gün, o rotada uçan tüm uçakların toplam yolcusu — yani **kendi uçuşumuzun final_pax'ı bu toplamın içinde**. Model `route_total_pax`'ı kullanarak hedefin (final_pax - pax_sold_cum) çoğunu doğrudan görüyor. Bu klasik **target leakage**.

**Sonuç:**
1. `pickup_xgb_metrics.json`'daki MAE=3.45/WAPE=9.82% rakamları **abartılı** — production'da bu değerler ulaşılamaz.
2. Inference'ta `forecast_bridge._build_features` bu feature'ları üretmiyor, default 0 değer veriyor → train-inference distribution shift (C-005).
3. EMSR-b ve manager analysis pickup tahminine güveniyor → güvenilmez girdi.

**Repro:** `pickup_master.parquet`'i incele: `SELECT flight_id, dep_year, route_total_pax, final_pax FROM pickup_master LIMIT 10` → her satırda `route_total_pax >= final_pax` görmen yeterli.

**Öneri:**
- `build_pickup_master.py` ROUTE-DAILY join'ini **lag(7)** veya **historical aggregate** ile yeniden yaz: `r.dep_date <= d.dep_date - INTERVAL 7 DAY`.
- Ya da bu feature grubunu tamamen düşür.
- Sonra modeli **yeniden eğit**, metrikleri raporla; mevcut metrikler gerçeği yansıtmıyor.

---

## [CRITICAL] C-005 — Train/Inference Feature Distribution Shift

**Lokasyon:** `dashboard/forecast_bridge.py:219-298`, `data/models/pickup_feature_list.json`
**Kategori:** ML / Production-Ready
**Açıklama:** Pickup model 49 feature ile train edildi (`pickup_feature_list.json`). Inference fonksiyonu `_build_features` yalnızca ~30 feature üretiyor. Eksikler — özellikle `route_total_pax`, `tag_yaz_tatili`, `tag_bayram`, `corporate_pct` vb. — `np.array([[feat.get(f, 0.0) ...]])` ile **default 0** alıyor.

Train'de model "yaz tatili: 1 → talep yüksek" pattern'i öğrendi; inference'ta her uçuş "yaz tatili: 0" görüyor → **sezonluk pattern devre dışı**.

**Etki:** Production'daki tahminler train metric'lerinden çok farklı. `Manager Override`'da `demand_source: pickup_tft_blend` dönüyor ama bu blend hatalı bir pickup ile yapılıyor.

**Repro Adımları:**
1. `python -c "import json; print(len(json.load(open('data/models/pickup_feature_list.json'))['features']))"` → 49
2. `forecast_bridge._build_features()` çağırıp dict key sayısını say → ~31

**Öneri:** Inference feature'larını train ile birebir hizala. Event tag'leri için tarih-bazlı `derive_tags(dep_date)` fonksiyonu yaz (yaz tatili tarih aralıkları sabitlenebilir). Route-aggregate feature'lar için **historical aggregation** yap.

---

## [HIGH] H-014 — TFT validation set rastgele group bazlı (time-series leakage riski)

**Lokasyon:** `scripts/training/train_tft_model.py:60-67`
**Kategori:** ML / Validation methodology
**Açıklama:**
```python
train_groups = train_df["group_id"].unique()
val_groups = np.random.choice(train_groups, size=int(len(train_groups) * 0.2), replace=False)
val_mask = train_df["group_id"].isin(val_groups)
val_df = train_df[val_mask].copy()
```
Validation için 2025 yılındaki **rastgele rota-kabin grupları** seçilmiş. Train/test split yıl bazlı ✓ ancak val random group split → val'da "unseen route generalization" ölçülüyor. Production senaryosu farklı: model train'de gördüğü rotaları yine tahmin ediyor. Hyperparameter tuning val MAE'sine göre yapıldıysa bu yanlış metric optimize edildi.

**Etki:** Erken durdurma (early stopping) val_loss'a göre. Eğer val grubunda outlier rotalar varsa erken durur, modeli yetersiz öğrenmiş bırakır.

**Öneri:** Time-based val split — 2025'in son 2 ayı val, geri kalanı train. Hyperparameter tuning bu setup üzerinde yapılmalı.

---

## [HIGH] H-015 — TFT inference unconstraining hack

**Lokasyon:** `dashboard/forecast_bridge.py:82-83`
**Kategori:** Logic / Calibration
**Açıklama:**
```python
if tft_total / cap_approx > 0.90:
    tft_total = tft_total * 1.15  # %15 gizli talep
```
TFT toplam tahmini kapasitenin %90'ını aşıyorsa toplam **%15 yapay olarak şişiriliyor** (gizli/unconstrained talep varsayımı). Bu kalibrasyonsuz, sezgisel bir hack. Demand-constrained pattern unconstraining gerçek RM'de Wickham/Botimer/Brumelle yöntemleriyle yapılır; %15 sabit oran teorik dayanak yok.

**Etki:** Tahmin kalitesi sapıyor. Manager `tft_forecast.predicted_remaining: 180.2` gördüğünde bu rakam %15 zaten şişirilmiş.

**Öneri:** Unconstraining'i ya tamamen kaldır (TFT zaten gerçek satışı modelliyor) ya da PD/EM gibi proper algoritma uygula.

---

## [HIGH] H-016 — Network Optimizer expected_demand parametresi geçilmiyor

**Lokasyon:** `dashboard/app.py:2879-2881`, `network_optimizer.py:84-90`
**Kategori:** Logic
**Açıklama:** Manager Analysis network recommendation'ı çağırırken:
```python
protection = _network_optimizer.compute_protection_levels(
    base_price, capacity, current_sold=pax_cum, dtd=dtd
)
```
`expected_total_demand` parametresi atlanıyor → `compute_protection_levels` default `capacity` değerini talep olarak kullanıyor. Sonuç: tüm class'lar maksimum koruma alıyor, V quota=0, K/M/Y quota=0 (önceki H-005 ile aynı kök neden).

EMSR-b'nin amacı talep tahmininden hareketle koruma vermek; talep yerine kapasite girilirse talep her zaman kapasiteye eşitlenir → pazar dinamiği yansımaz.

**Öneri:** `_compute_expected_demand` çıktısındaki `pax_cum + expected_demand`'ı `expected_total_demand` olarak ilet:
```python
protection = _network_optimizer.compute_protection_levels(
    base_price, capacity, current_sold=pax_cum, dtd=dtd,
    expected_total_demand=pax_cum + expected_demand_value
)
```

---

## [HIGH] H-017 — S-curve hardcoded üs (1.5) tüm rotalar/segmentler için aynı

**Lokasyon:** `forecast_bridge.py:86`, `simulation_engine.py:691`, `app.py:2845, 2983, 3175`
**Kategori:** Calibration
**Açıklama:** Beş farklı yerde `cum_fraction = 1 - (dtd/180)^1.5` formülü kopyala-yapıştır. Üs değeri 1.5 nereden? Veriden kalibrasyon yok. Gerçekte:
- Business yolcuları DTD<14'te hızlanır (üs ~3)
- Leisure DTD>60'ta daha yoğun (üs ~1)
- Last-minute segment DTD<3'te keskin spike (üs ~5)

Tek formül her segment için kullanılıyor → segment-spesifik booking pattern kaybolmuş.

Ek olarak: aynı hesaplama beş farklı dosyada tekrarlanıyor → DRY ihlali, değişiklik 5 yerde yapılmalı.

**Öneri:**
1. `pricing_engine.compute_cum_fraction(dtd, segment_id=None)` shared helper.
2. `demand_functions_report.json`'dan segment-spesifik DTD eğrileri (Gaussian param'ları zaten orada).

---

## [HIGH] H-018 — Pickup model'de hyperparameter tuning + early stopping yok

**Lokasyon:** `scripts/training/train_pickup_xgb.py:103-115`
**Kategori:** ML / Training methodology
**Açıklama:**
```python
params = {'max_depth': 7, 'learning_rate': 0.05, 'subsample': 0.8, ...}
bst = xgb.train(params, dtrain, num_boost_round=500, verbose_eval=50)
```
- Validation set yok
- Early stopping yok (sabit 500 round)
- Hyperparameter grid search yok

Eğer hyperparameter'lar test set'te denenmişse meta-leakage (`max_depth=7` neden seçilmiş, hangi süreçte?). Ek olarak baseline:
```python
baseline_pred = final_pax_test * (dtd_test / 180)
```
Baseline `final_pax`'ı kullanıyor — yani **gerçek değer**! XGBoost'a karşı haksız bir karşılaştırma değil, çünkü XGBoost `final_pax`'a erişemiyor. Ama baseline kendisi target leakage taşıyor → "iyileştirme" rakamı (`improvement_mae_pct`) baseline'ın leakage'ı yüzünden olduğundan daha iyi gözükmüş olabilir.

**Öneri:**
- Train'den %20 val set ayır (group bazlı, time bazlı tercih).
- `early_stopping_rounds=50` ekle, `xgb.train(...evals=[(dval, 'val')]...)`.
- Baseline'ı `final_pax` kullanmadan üret (ör. tarihsel ortalama).

---

## [HIGH] H-019 — Sentiment skoruna göre carpan tutarsızlıkları

**Lokasyon:**
- `pricing_engine.py:321` → `1.0 + score * 0.15` (fiyat)
- `simulation_engine.py:677` → `1.0 + score * 0.30` (talep, simülasyon)
- `app.py:2862` → `1.0 + score * 0.20` (talep, manager override)

**Kategori:** Consistency
**Açıklama:** Aynı sentiment skoru fiyatı ±%15, simülasyondaki talebi ±%30, manager override'daki talebi ±%20 etkiliyor. Üç farklı carpan, üç farklı dosya, hiçbir gerekçe yok.

**Etki:** Manager Override'da +%20 talep etkisi varken, simülasyon ay boyunca ±%30 etki üretiyor → manager kararını test etmek için simülasyon koşturduğunda farklı sonuç alıyor.

**Öneri:** `pricing_engine`'de tek constant: `SENTIMENT_PRICE_COEFF = 0.15`, `SENTIMENT_DEMAND_COEFF = 0.20`, üç dosyadan da import.

---

## [MEDIUM] M-014 — DOM XSS riski: 107 innerHTML kullanımı

**Lokasyon:** `dashboard/templates/index.html` (107 yer), `customer/templates/index.html` (14 yer)
**Kategori:** Security
**Açıklama:** Ana dashboard'da template literal interpolasyon ile innerHTML yazımı yaygın. Çoğu DB'den gelen veriyle (route, flight_number) — bunlar kontrollü. Ama:
- `searchInput` üzerinden gelen `q` parametresi `/api/flights` endpoint'ine, oradan response'a, oradan dropdown.innerHTML'e gidiyor.
- Eğer DB içeriği bir gün user input ile dolarsa (örn. yorum/not alanı) XSS açılır.

**Öneri:** `textContent` veya DOMPurify kullan; template literal yerine DOM API.

---

## [MEDIUM] M-015 — `compute_protection_levels` quota="V" hardcoded 0

**Lokasyon:** `network_optimizer.py:121, 134-135`
**Kategori:** Logic
**Açıklama:** `protections["V"] = 0` ve sonra `result["V"]["protected"] = 0`. V class'ı her zaman korumasız. Bu intentional EMSR-b kuralı (en düşük class korumaz) ✓. Ama `quota` hesabı `cum` üzerinden değil ayrı satırda `max(remaining - total_prot, 0)` yapılıyor — round-trip tutarsızlığı.

**Öneri:** Tek formülle `V.quota = remaining - sum(V'den yukarı tüm protectprotected)`.

---

## [MEDIUM] M-016 — `_calc_overbooking_limit` dolaylı `math.ceil` davranışı

**Lokasyon:** `simulation_engine.py:253`
**Kategori:** Logic
**Açıklama:** `math.ceil(capacity * (1 + ob_pct))` — `ob_pct=0.04, capacity=300` → `math.ceil(312.0) = 312` OK. Ama `capacity=49, ob_pct=0.015` → `math.ceil(49.735) = 50` (tek koltuk overbooking). Mantıklı ama küçük kabinlerde hassas.

**Öneri:** Belge ekle.

---

## [MEDIUM] M-017 — `_estimate_cancellation_noshow` parametre olarak `cabin` kullanmıyor

**Lokasyon:** `app.py:2892-2919`
**Kategori:** Dead parameter
**Açıklama:** `def _estimate_cancellation_noshow(remaining_demand, dtd, cabin)` — `cabin` argümanı imzada var ama gövdede hiç kullanılmıyor. Business cabin için cancellation/no-show oranları farklıdır gerçekte (business %12-15 no-show, leisure %3-5).

**Öneri:** Cabin'e göre `no_show_rates` ve `cancel_rates` ayır.

---

## [LOW] L-009 — `forecast_bridge.predict_remaining_demand` exception sessizce yutuyor

**Lokasyon:** `forecast_bridge.py:144-146`
**Açıklama:** `except Exception: return None` — model hata verirse hangi hata olduğu görünmez. `app.py:2802-2803` aynı pattern.
**Öneri:** `logging.exception(...)`.

---

## [LOW] L-010 — Network optimizer `pick_random_origin` weight'i ters mantık

**Lokasyon:** `network_optimizer.py:24-30`
**Açıklama:** `origin_weights[rk] = dist / total_dist` — uzak rota daha yüksek olasılık. Hub'ta gerçekte yakın rotadan daha çok connecting yolcu gelir (kısa-iniş, uzun-kalkış). Ters mantık.
**Öneri:** Inverse weight: `1/dist` veya gerçek connecting traffic'den kalibre et.

---

## [LOW] L-011 — `prorate_fare` magic 0.85 ve magic 3000

**Lokasyon:** `network_optimizer.py:173, 220`
**Açıklama:** `total_fare = (o_base + d_base) * 0.85` — neden %15 indirim? `dist=3000` default — neden 3000?
**Öneri:** Constant'lara isim ver: `CONNECTING_DISCOUNT = 0.85`, belge ekle.

---

# ════════════════════════════════════════════════
# FAZ 4 — Entegrasyon ve Frontend
# ════════════════════════════════════════════════

## [HIGH] H-020 — Customer app /api/routes Seatwise erişiminde fallback list 50 hardcoded rota

**Lokasyon:** `customer/customer_app.py:42-49` (`SIM_ROUTE_FALLBACK`), `:354-371`
**Kategori:** Logic / Maintenance
**Açıklama:** Seatwise erişilemezse 50 rotalık fallback liste kullanılıyor (`IST-LHR, IST-MAD, ...`). Bu liste manuel — yeni rota eklendiğinde güncellemek unutulur.
**Öneri:** Fallback'i `flight_metadata.parquet`'tan tek seferlik üret, JSON dosyası olarak kaydet.

---

## [HIGH] H-021 — `customer_app._dashboard_post` return type belirsiz, `/api/book` patlayabilir

**Lokasyon:** `customer/customer_app.py:77-83`, `:511-517`
**Kategori:** Bug
**Açıklama:**
```python
def _dashboard_post(path, payload, timeout=REQUEST_TIMEOUT):
    try:
        r = requests.post(DASHBOARD_URL + path, json=payload, timeout=timeout)
        if r.status_code == 200:
            return r.json()  # ← parsed dict
    except:
        return None
```
`api_book` return değerini `Response` objesi gibi kullanıyor:
```python
return jsonify(response.json()), response.status_code
```
Eğer `_dashboard_post` 200 dönerse parse edilmiş dict döner, dict'in `.json()` ve `.status_code` attribute'leri yoktur → **AttributeError 500**.

Bu kod hiç test edilmediği aşikar — booking flow patlar.

**Öneri:** `_dashboard_post` `Response` objesini direkt döndürsün veya `api_book` doğrudan dict bekleyen şekilde güncellensin.

---

## [MEDIUM] M-018 — Frontend Manager Analysis sensitivity hâlâ JS'de hesaplıyor olabilir

**Lokasyon:** `templates/index.html:6269+ (buildMgrCharts)`
**Kategori:** Consistency
**Açıklama:** Önceki audit'te tespit edilen frontend sensitivity matematik kopyası hâlâ index.html içinde. Backend `/api/manager-sensitivity` eklendi (Fix #10) ama frontend buna geçirilmedi (bekleyişe alındı).
**Öneri:** `buildMgrCharts`'ı `/api/manager-sensitivity` çağrısı ile değiştir, JS'deki linear/elasticity hesabını sil.

---

## [MEDIUM] M-019 — `customer/users_db.json` ile `dashboard/users_db.json` ayrı

**Lokasyon:** İki ayrı dosya
**Kategori:** Architecture
**Açıklama:** Customer ve Company kullanıcıları ayrı veritabanlarında. Eğer aynı kişi her iki uygulamayı kullanacaksa iki kez kayıt olmalı, iki şifre yönetmeli. Tek SSO/JWT mantığı yok.
**Öneri:** Tek user DB, role bazlı erişim (manager/customer).

---

## [LOW] L-012 — Login HTML'inde "SeatWise" büyük W (önceki feedback'le çelişki)

**Lokasyon:** `templates/login.html:6` (title), brand-name
**Açıklama:** Daha önce kullanıcı tarafından "Seatwise" (küçük w) tercih edildi. Login sayfası "SeatWise" diyor.
**Öneri:** Tutarlılığa "Seatwise" yap.

---

## [LOW] L-013 — `login.html` ve `loading.html` arasında animasyon transition yok

**Lokasyon:** `templates/loading.html`
**Açıklama:** Login → loading → dashboard akışında loading sayfası 1-2 sn görünüp dashboard'a geçer. Animasyon transition yok.
**Öneri:** UX iyileştirme; gerekli değil.

---

# ════════════════════════════════════════════════
# FAZ 5 — Diğer Kapsam
# ════════════════════════════════════════════════

## [INFO] I-002 — `report_generator/` modülü incelenmedi

Manager Analysis kapsamı dolu olduğu için PDF rapor üretim akışı (NLG → PDF) bu turda denetlenmedi. Önceki audit'lerde lexicon/collector/analyzer/nlg_engine/pdf_builder yapısı incelenmişti; bu sürümde dosyalar değişmemiş gözüküyor (size aynı, görsel diff alınmadı).

## [INFO] I-003 — TFT model attention/VSN extraction kontrol edilmedi

`scripts/extract_tft_attention.py` ve `reports/tft_interpretation.json` üretim süreci doğrulanmadı. `/api/tft/interpretation` endpoint'i sadece JSON dosyasını okuyor, hesaplama yok.

## [INFO] I-004 — Concurrent kullanıcı testi yapılmadı

50 eş zamanlı `/api/manager-analysis` isteği DuckDB connection pool'u zorlar mı? `get_con()` her çağrıda yeni connection açıyor, kapatılıyor — leak yok ama yük yok.

## [INFO] I-005 — Git history sızıntı denetimi yapılmadı

`.env` ve secret dosyalar git history'sine push edildi mi? `git log --all -p | grep NEWSAPI_KEY` komutu çalıştırılmadı. NEWSAPI_KEY'in revoke edilmesi tavsiye edilir (M-003).

---

## [INFO] I-001 — Manager Analysis önceki audit'teki K1–K4 + M1–M4 + m1–m2 kapatılmaya çalışılmış

**Lokasyon:** `app.py:2588-3225`
**Kategori:** İyileştirme
**Açıklama:** Yeni sürümde:
- K1 → tarih filtresi eklendi (yarım çözüm; H-002)
- K2 → `_compute_weighted_elasticity` ile segment-weighted ε (✓ doğrulandı: -1.1541)
- K3 → isoelastic formula `(1+p)^ε` (✓ doğrulandı: `demand_formula: isoelastic`)
- K4 → `_compute_expected_demand` pickup+TFT blend (✓ doğrulandı: `demand_source: pickup_tft_blend`)
- Ö1 → SQL parametrize edildi (✓)
- Ö2 → DTD override LF S-curve ile (✓)
- Ö3 → Backend İngilizce, frontend hâlâ Türkçe (H-004 açık)
- Ö4 → KPI scope tablo dilimine hizalandı (✓)
- m1 → max(0) clamp kaldırıldı (✓)
- m2 → `/api/manager-sensitivity` server-side endpoint (✓; frontend entegrasyonu L-006)

Plus: TFT info (`tft_forecast`), network optimizer (`network_recommendation`), cancellation/noshow (`cancellation_noshow`), sentiment-demand coupling (`sentiment_demand_factor`).

Bu ciddi bir kalite atlayışı. Yine de davranışsal sorunlar (H-002, H-003, H-005) kalmış.
