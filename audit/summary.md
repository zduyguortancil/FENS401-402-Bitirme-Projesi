# Seatwise Audit — Yönetici Özeti (audit/summary.md)

**Sürüm:** seatwise_3 / FENS401-402-Bitirme-Projesi-Last Version_duygu(V.2)
**Audit tarihi:** 2026-04-19
**Kapsam:** Faz 0 (✓), Faz 1 (✓), Faz 2 (✓), Faz 3 (✓), Faz 4 (✓), Faz 5 (kısmi: report_generator + concurrent test + git history dışında)
**Toplam dosya:** 41 Python + 7 HTML + 18 JSON/MD
**Toplam kod (yaklaşık):** 14.500 satır

---

## 1. Bulgu Özeti

| Severity | Adet | Notlar |
|---|---|---|
| **CRITICAL** | 5 | Auth bypass, hardcoded secret, saltsız password, ML target leakage, train-inference shift |
| **HIGH** | 21 | Manager Analysis kalan davranışsal sorunları + ML methodology + Customer app boşluklar |
| **MEDIUM** | 19 | Validation, race condition, synthetic data, sentiment, performance |
| **LOW** | 13 | Naming, dead code, magic number, UX |
| **INFO** | 5 | Kapsam dışı bırakılanlar + iyileştirme önerileri |
| **TOPLAM** | **63** | |

---

## 2. En Kritik 10 Bulgu

| # | ID | Başlık | Severity | Kategori |
|---|---|---|---|---|
| 1 | **C-001** | Tüm `/api/*` endpoint'leri auth'suz — login kozmetik | CRITICAL | Security |
| 2 | **C-002** | Hardcoded session secret `"sw-desktop-secret-2026"` | CRITICAL | Security |
| 3 | **C-003** | Şifreler saltsız SHA256 ile saklanıyor | CRITICAL | Security |
| 4 | **C-004** | Pickup XGBoost'ta `route_total_pax` target leakage | CRITICAL | ML / Data |
| 5 | **C-005** | Train/Inference feature distribution shift (49 → ~31) | CRITICAL | ML / Production |
| 6 | **H-002** | Manager Analysis hâlâ DTD=0 (terminal) verisi | HIGH | Logic |
| 7 | **H-003** | Manager Override DTD=0'da kollapse — her şey 0% | HIGH | Logic |
| 8 | **H-014** | TFT validation random group split (time-series leakage riski) | HIGH | ML |
| 9 | **H-016** | Network optimizer expected_demand parametresi geçilmiyor | HIGH | Logic |
| 10 | **H-021** | Customer app `/api/book` Response/dict tutarsızlığı — patlar | HIGH | Bug |

---

## 3. Modül Bazında Risk Haritası

| Modül | LOC | Bulgu | Risk | Notlar |
|---|---|---|---|---|
| **`app.py` (auth)** | 285 (sat 38-349) | C-001, C-002, C-003, H-001, H-009, H-010, H-011, H-012, H-013, M-011 | 🔴 KRİTİK | Auth katmanı kozmetik |
| **`app.py` (Manager Analysis)** | ~640 (sat 2588-3225) | H-002, H-003, H-004, H-005, H-006, H-016, H-019, M-002, M-005, M-017, L-001, L-005 | 🟠 YÜKSEK | K1–K4 düzeltilmiş ama davranışsal kollapse |
| **`app.py` (diğer endpoint'ler)** | ~2840 | C-001 (auth yok), H-012 (validation) | 🟠 YÜKSEK | Tüm endpoint'ler aynı sorun |
| **`customer_app.py`** | 517 | C-001, H-008, H-009, H-020, H-021, M-006, M-007, M-008, M-009, M-010, M-019 | 🔴 KRİTİK | Auth yok, validation yok, /api/book patlar |
| **`pricing_engine.py`** | 553 | H-019, L-002 | 🟢 DÜŞÜK | Stabil, sentiment carpan dışında temiz |
| **`simulation_engine.py`** | 1454 | H-017, H-019, M-016, L-005 | 🟡 ORTA | S-curve hardcoded, Tweedie+TFT pipeline OK |
| **`forecast_bridge.py`** | 298 | C-005, H-015, H-017, L-009 | 🔴 KRİTİK | Inference feature shift, unconstraining hack |
| **`network_optimizer.py`** | 220 | H-005, H-016, M-015, L-010, L-011 | 🟠 YÜKSEK | EMSR-b kavramsal doğru ama parametre eksik |
| **`competitor_engine.py`** | 257 | (incelenmedi) | ❓ | Kapsam dışı |
| **`sentiment/`** | 9 dosya | H-007, M-004 (önceki 23 sorun) | 🟡 ORTA | Modül değişmemiş, sorunlar açık |
| **`report_generator/`** | 6 dosya | (incelenmedi) | ❓ | Kapsam dışı |
| **`scripts/training/`** | 2 dosya | C-004, H-014, H-018 | 🔴 KRİTİK | TFT random val split, pickup leakage |
| **`scripts/data_prep/`** | 7 dosya | C-004 (build_pickup_master) | 🔴 KRİTİK | Target leakage source |
| **`templates/`** | 7 HTML | H-004, H-011, M-014, L-012, L-013 | 🟡 ORTA | XSS düşük ama 107 innerHTML, Türkçe TR-EN karışım |

---

## 4. Önceki Audit Karşılaştırması (12 düzeltme girişimi)

| Önceki bulgu | Düzeltme | Durum |
|---|---|---|
| K1 (date filter yok) | 90 gün filter eklendi | ⚠️ Yarım — DTD=0 sorunu (H-002) |
| K2 (hardcoded elasticity) | `_compute_weighted_elasticity` | ✅ Doğrulandı (-1.1541) |
| K3 (linear formula) | `(1+p)^ε` isoelastic | ✅ Doğrulandı |
| K4 (effective_remaining naive) | pickup+TFT blend | ✅ Doğrulandı (`pickup_tft_blend`) |
| Ö1 (SQL injection) | Parametrize | ✅ Düzeltildi |
| Ö2 (DTD override LF) | S-curve estimated_lf | ✅ Düzeltildi |
| Ö3 (frontend Türkçe) | Backend EN | ❌ Frontend hâlâ TR (H-004) |
| Ö4 (KPI scope) | Pagination + scope | ✅ Düzeltildi |
| m1 (zero clamp) | Kaldırıldı | ✅ Düzeltildi |
| m2 (frontend chart math) | API endpoint var | ⚠️ Frontend entegrasyon eksik (M-018) |
| Mimari 11 (TFT) | `tft_forecast` field | ⚠️ Trend semantiği yanlış (H-006) |
| Mimari 12 (Network optimizer) | `network_recommendation` | ⚠️ Parametre eksik (H-016) |
| Mimari 13 (Cancellation/no-show) | `_estimate_cancellation_noshow` | ⚠️ DTD=0'da kollapse (H-003), cabin yok (M-017) |
| Mimari 14 (Sentiment → talep) | `_compute_sentiment_demand_factor` | ✅ Eklendi (carpan tutarsız H-019) |

**Skor: 9/14 tam çözüm, 4/14 kısmi, 1/14 açık.**

---

## 5. Yeni Sürümle Gelen Yeni Sorunlar

| # | Açıklama | Severity |
|---|---|---|
| 1 | Auth katmanı eklendi ama API'ler hâlâ açık | C-001 |
| 2 | Customer portal eklendi ama auth/validation yok | C-001/H-008 |
| 3 | Hardcoded session secret iki uygulamada da | C-002 |
| 4 | Saltsız SHA256 password | C-003 |
| 5 | users_db.json thread-unsafe | H-009 |
| 6 | Synthetic fiyat eğrisi müşteriye gerçek gibi | M-009 |
| 7 | Hardcoded competitor estimate (PC×0.75, EK×1.2) | M-010 |
| 8 | DTD=0'da Manager Override pipeline'ı çöküyor | H-003 |
| 9 | Network optimizer protection quota=0 (parametre eksik) | H-016 |
| 10 | TFT trend semantic yanlış | H-006 |
| 11 | Customer `/api/book` Response/dict bug — booking patlar | H-021 |

---

## 6. ML Boyutunda Kritik Bulgular

Bu sürümde derin ML denetimi yapıldı; üç yapısal sorun ortaya çıktı:

### 6.1 Pickup XGBoost Target Leakage (C-004)
`build_pickup_master.py`'de route-daily aggregate features (`route_total_pax`, `route_n_bookings`, vb.) **uçuşun kendi günündeki, kendi rotasındaki** toplamları içeriyor. Yani `route_total_pax` zaten `final_pax`'ı içerir → model hedefi feature olarak görüyor. Bunun sonucunda:
- `pickup_xgb_metrics.json`'daki MAE=3.45 / WAPE=9.82% **abartılı**.
- Inference'ta bu feature'lar mevcut değil → train-inference distribution shift (C-005) → model gerçek hayatta train metric'lerinden çok kötü çalışır.

### 6.2 TFT Validation Methodology (H-014)
Train/test yıl bazlı kronolojik split ✓ ancak validation rastgele group seçimi → time-series senaryosuyla uyumsuz. Hyperparameter tuning val metric'ine göre optimize edildiyse yanlış hedef minimize edilmiş.

### 6.3 Pickup Hyperparameter Sabit (H-018)
`max_depth=7, lr=0.05, num_boost_round=500` sabit, validation set yok, early stopping yok. Hyperparameter'lar nasıl seçildi? Eğer test set'te denenmişse meta-leakage. Bu metric'ler bilim raporu için sunulurken dikkatli ifade edilmeli.

**Sonuç:** Mevcut `MAE=3.45`, `WAPE=9.82%`, `MAE_TFT=14.03` rakamları bilimsel raporlamada **iyimser üst sınır** olarak değerlendirilmeli. Production performansı bu rakamların **2-5 katı** olabilir.

---

## 7. Güvenlik Boyutu (Acil)

| Açık | Risk | Çözüm |
|---|---|---|
| API'ler auth'suz (C-001) | Tüm RM verisi public | `@login_required` veya `before_request` filter |
| Hardcoded secret (C-002) | Session forge | env zorunlu, default kaldır |
| Saltsız SHA256 (C-003) | Rainbow table / GPU brute force | `bcrypt` veya `werkzeug.generate_password_hash` |
| CSRF protection yok (H-011) | Cross-site forgery | Flask-WTF |
| Rate limit yok (H-001) | Brute force | flask-limiter |
| `force=True` JSON (H-010) | Content-Type bypass | `force=False` |
| Validation yok (H-012, H-013) | DOS / sandboxed crash | Pydantic / WTForms |
| `users_db.json` race (H-009) | Veri kaybı | flock / SQLite |

---

## 8. Genel Mimari Yorumu

**Olumlu:**
- Önceki audit'in 14 ciddi sorunundan 9'u kod düzeyinde profesyonelce kapatılmış. Manager Analysis modülü artık veri-kalibreli segment elasticity, isoelastic formül, pickup+TFT blend, EMSR-b protection level, cancellation/no-show, sentiment-demand coupling gibi gerçek RM bilim mimarisine dayanıyor. `Fix #N` etiketleri izlenebilir.
- Customer portal, desktop paketleme, auth sistemi gibi yeni katmanlar mimari olgunlaşmayı gösteriyor.
- Network optimizer, forecast bridge, sentiment scheduler katmanlı yapıda iyi ayrılmış.

**Olumsuz:**
1. **Güvenlik temeli zayıf** — yeni eklenen auth katmanı sadece sayfa rotalarını koruyor; API'ler tamamen açık. Saltsız hash, hardcoded secret, CSRF eksikliği. Bu üç sorun production'a çıkmadan kapatılmalı.
2. **ML metric'leri abartılı** — pickup model'de target leakage tespit edildi. TFT validation methodology bozuk. Mevcut başarı rakamları akademik raporlamada iyimser üst sınır olarak değerlendirilmeli.
3. **Davranışsal regresyon (Manager Analysis)** — kod doğru ama pipeline DTD=0 terminal verisi yüzünden çıktıyı sıfıra çeviriyor. K1'in yarım çözümü tüm what-if'leri işlevsiz kılıyor.
4. **Customer portal denetlenmemiş** — yepyeni 517 satır, hiçbir test yok. `/api/book` rezervasyon endpoint'i tip uyumsuzluğu nedeniyle patlar.
5. **Sentiment modülü değişmemiş** — önceki audit'teki 23 sorun açık.
6. **Train-inference distribution shift** — pickup model üretim ortamında train zamanından farklı pattern görüyor; sonuçlar güvenilmez.

---

## 9. Öncelikli Aksiyon Planı

### P0 — Production Blocker (5)
1. **C-001:** API'lere `@login_required` ekle.
2. **C-002:** `app.secret_key` env zorunlu.
3. **C-003:** `bcrypt` ile şifre re-hash.
4. **C-004:** `build_pickup_master.py`'da route-daily JOIN'i `dep_date - 7 day` ile değiştir, modeli yeniden eğit.
5. **C-005:** `forecast_bridge._build_features` train feature listesiyle birebir hizala.

### P1 — Yüksek Öncelik (15)
6. **H-002:** SQL'de `dep_date - CURRENT_DATE` ile gerçek DTD.
7. **H-003:** Cancellation/no-show formülünde DTD<7 minimum threshold.
8. **H-005, H-016:** Network optimizer'a `expected_total_demand` parametresi geç.
9. **H-006:** TFT trend → `demand_supply_ratio`.
10. **H-014:** Time-based val split.
11. **H-018:** Pickup eğitiminde val set + early stopping.
12. **H-021:** `_dashboard_post` return type'ını netleştir.
13. **H-008, H-011:** Customer `/api/book` auth + CSRF + validation.
14. **H-001:** Rate limit.
15. **H-009, H-010, H-012, H-013:** Validation, locking, force=False.
16. **H-017:** S-curve helper centralized.
17. **H-019:** Sentiment carpan constant'ları tek yerde.

### P2 — Orta Öncelik (19)
18. M-001…M-019 ve H-004, H-007, H-015, H-020.

### P3 — Refactor (13)
19. L-001…L-013 ve I-001…I-005 takip.

---

## 10. Test Edilmeyenler / Sonraki Audit Notları

| Alan | Durum | Önerilen yöntem |
|---|---|---|
| `report_generator/` PDF üretimi | İncelenmedi | NLG çıktı üret, PDF aç, içerik manuel doğrula |
| `competitor_engine.py` | İncelenmedi | Rakip fiyat üretim mantığı |
| TFT attention/VSN extraction | İncelenmedi | `extract_tft_attention.py` çalıştır |
| Concurrent yük testi | Yapılmadı | `locust`/`hey` ile 50 paralel istek |
| Git history sızıntı (NEWSAPI_KEY) | Yapılmadı | `git log --all -p -- .env` |
| TFT train script lokal çalıştırma | Yapılmadı | Kaggle'dan diff alınmalı |
| `simulation_engine._process_bot` | Kısmen | Bot karar mantığı, WTP testi |
| `simulation_engine._process_cancellations` | İncelenmedi | İptal akışı, refund formülü |
| Frontend XSS (107 innerHTML) | Manuel sample | DOMPurify veya CSP eklenmesi denenmedi |
| Booking idempotency | Yapılmadı | Aynı request iki kez yollanırsa? |
| DuckDB connection leak | Yapılmadı | 1000 ardışık request memory profil |

---

## 11. Sonuç

Seatwise sürümlerinden bu zamana kadar **en kapsamlı sürüm** budur. Önceki kritik tespitlerin çoğu kod düzeyinde profesyonelce ele alınmış. Ancak iki büyük yapısal eksiklik var:

1. **Güvenlik temeli production-grade değil** — auth, password, secret yönetimi öğrenci projesi seviyesinde.
2. **ML metric güvenilirliği problemli** — target leakage ve train-inference shift, raporlanan başarı rakamlarını sorgulanır kılıyor.

**Bitirme projesi savunması için:** Mevcut araç fonksiyonel olarak demo edilebilir (login → dashboard → manager analysis paneli açılır). Ancak savunma sırasında "production'a çıkar mıyız?" sorusu gelirse, **net cevap: hayır, yukarıdaki P0/P1 maddeleri kapatılmadan asla**. Hocaya iletilecek dürüst değerlendirme: "MVP/prototype seviyesi, akademik kalibrasyonu var ama production deployment için 2-3 hafta daha çalışma gerek."

---

## 12. Audit Çıktıları

| Dosya | İçerik |
|---|---|
| `audit/00_architecture_map.md` | Mimari haritası, modül listesi, endpoint tablosu, veri akışı |
| `audit/findings.md` | 63 bulgu, severity'ye göre (C-001 → I-005) |
| `audit/summary.md` | (bu dosya) Yönetici özeti, risk haritası, aksiyon planı |
