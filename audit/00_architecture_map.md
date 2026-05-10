# Seatwise — Mimari Haritası (Faz 0)

**Sürüm:** seatwise_3 / FENS401-402-Bitirme-Projesi-Last Version_duygu(V.2)
**Tarih:** 2026-04-19
**Toplam Python LOC (yaklaşık):** ~14.500 satır

---

## 1. İki Bağımsız Flask Uygulaması

Bu sürüm artık **tek bir uygulama değil, iki ayrı portal**:

| Uygulama | Dosya | Hedef Port | Hedef Kullanıcı |
|---|---|---|---|
| **Seatwise (Company Portal)** | `dashboard/app.py` (3764 satır) | 5005 | Havayolu yöneticisi / RM analist |
| **BiletBul (Customer Portal)** | `dashboard/customer/customer_app.py` (517 satır) | farklı port (run_customer.py'da set) | Son kullanıcı (yolcu) |

Her ikisi de Flask, ayrı `users_db.json` dosyası, ayrı session secret. **BiletBul → Seatwise'a HTTP üzerinden çağrı yapıyor** (`DASHBOARD_URL = "http://localhost:5005"`).

### Desktop paketleme
- `run_company.py` ve `run_customer.py` — `pywebview` ile Flask'ı sarıp masaüstü uygulamasına çeviriyor.
- `run_desktop.bat` (Windows) ve `run_desktop.command` (Mac) — başlatma kabukları.
- `requirements-desktop.txt` sadece `pywebview` içeriyor.

---

## 2. Modül Sorumlulukları

### 2.1 dashboard/ (Company)

| Modül | LOC | Sorumluluk |
|---|---|---|
| `app.py` | 3764 | Tüm endpoint'ler, auth, sentiment cache yönetimi, manager analysis |
| `simulation_engine.py` | 1454 | Booking bot simülatörü, fare class progression, overbooking |
| `pricing_engine.py` | 553 | 4-çarpan fiyat formülü, fare class kuralları, DTD rules |
| `forecast_bridge.py` | ? | TFT + XGBoost pickup + Two-Stage modellerinin yükleme & inference köprüsü |
| `network_optimizer.py` | ? | EMSR-b, bid-price, fare proration |
| `competitor_engine.py` | ? | Rakip uçuş analizi |
| `report_generator/` | 6 dosya | NLG-tabanlı PDF rapor (lexicon, collector, analyzer, nlg_engine, pdf_builder, charts) |
| `sentiment/` | 9 dosya | 51-şehir GDELT + RSS + keyword classifier sentiment motoru |

### 2.2 dashboard/customer/ (BiletBul)

| Modül | LOC | Sorumluluk |
|---|---|---|
| `customer_app.py` | 517 | Müşteri arama, fiyat takvimi, rezervasyon |
| `users_db.json` | — | Müşteri user db (ayrı dosya!) |
| `templates/` | 3 HTML | login, loading, index |

---

## 3. Tüm Endpoint'ler

### 3.1 Seatwise (Company) — `dashboard/app.py`

**Sayfa rotaları:**
- `/` → landing (auth değilse → /login, evet ise → /dashboard)
- `/dashboard` → ana panel (login_required)
- `/login` GET/POST
- `/register` POST
- `/logout`
- `/sentiment` → sentiment sayfası
- (presumed) `/simulation`, `/booking`, `/competition` — templates'da var

**API endpoint'leri (37 adet):**

| Endpoint | Yöntem | Satır | Amaç |
|---|---|---|---|
| /api/flights | GET | 353 | Tüm uçuşların listesi |
| /api/airport/<code> | GET | 401 | Havalimanı bazlı uçuşlar |
| /api/flight/<flight_number> | GET | 418 | Uçuşun tarih listesi |
| /api/flights/date | GET | 442 | Tarih bazlı uçuş listesi |
| /api/snapshot/<flight_id> | GET | 473 | Uçuşun anlık durumu (DTD bazlı timeline) |
| /api/forecast/<flight_id> | GET | 622 | TFT + Pickup forecast |
| /api/demand/<flight_id> | GET | 812 | Demand analizi (segment kırılımı) |
| /api/pickup/<flight_id> | GET | 956 | Pickup tahmini |
| /api/daily-brief | GET | 1120 | Ana sayfa özeti |
| /api/tft/interpretation | GET | 1302 | TFT VSN + attention |
| /api/clusters | GET | 1319 | Yolcu kümeleri |
| /api/cluster/<id> | GET | 1337 | Tek küme detayı |
| /api/trends | GET | 1372 | Trend analizi |
| /api/top-routes | GET | 1531 | En yoğun rotalar |
| /api/events | GET | 1728 | Event/sentiment tag analizi (training data'dan) |
| /api/demand-functions | GET | 2107 | Talep fonksiyonu raporu |
| /api/demand-curves | GET | 2118 | Talep eğrileri |
| /api/fare-classes | GET | 2257 | V/K/M/Y kuralları |
| /api/simulation | GET | 2362 | Simülasyon başlat/durum |
| /api/risk-index | GET | 2387 | Risk göstergesi |
| /api/manager-analysis | GET | 2588 | Manager Analysis: tüm uçuşlar tablosu |
| /api/manager-override | POST | 2922 | Manager Analysis: tek uçuş what-if |
| /api/manager-sensitivity | POST | 3128 | Manager Analysis: sensitivity curve (YENİ) |
| /api/sentiment/status | GET | 3234 | Sentiment scheduler durumu |
| /api/sentiment/all | GET | 3247 | Tüm şehir sentiment'i |
| /api/sentiment/<city> | GET | 3266 | Tek şehir sentiment'i |

**Yardımcı fonksiyonlar (Manager Analysis için yeni eklenmiş):**
- `_compute_weighted_elasticity(cabin)` (sat 2764) → demand_functions_report.json'dan segment-weighted ε hesabı
- `_compute_expected_demand(...)` (sat 2782) → pickup model + elasticity entegrasyonu
- `_compute_sentiment_demand_factor(route)` (sat 2852)
- `_compute_network_recommendations(...)` (sat 2868)
- `_estimate_cancellation_noshow(...)` (sat 2892)

> Bu fonksiyonların eklenmesi, önceki audit'te tespit edilen 4 kritik sorunu (K1–K4) **düzeltme girişimi** gibi görünüyor. Faz 1'de bunların gerçekten düzeltilip düzeltilmediği doğrulanacak.

### 3.2 BiletBul (Customer) — `customer/customer_app.py`

**Sayfa rotaları:**
- `/` → landing
- `/search` → arama sayfası
- `/login` GET/POST
- `/register` POST
- `/logout`

**API endpoint'leri (6 adet):**

| Endpoint | Yöntem | Satır | Amaç |
|---|---|---|---|
| /api/search | GET | 326 | Uçuş arama |
| /api/routes | GET | 354 | Mevcut rotalar |
| /api/price-calendar | GET | 373 | 30-günlük fiyat takvimi |
| /api/flight-detail | GET | 442 | Uçuş detayı (canlı veri) |
| /api/sim-status | GET | 499 | Simülasyon durumu (Seatwise'a proxy) |
| /api/book | POST | 511 | Rezervasyon |

---

## 4. Veri Akışı

```
┌─────────────────────────────────────────────────────────────┐
│  KAYNAK VERİ                                                 │
│  ─ data/raw/ (parquet) ─ flight_snapshot, bookings           │
│  ─ data/processed/ (parquet) ─ tft_dataset, pickup_master    │
│  ─ data/models/ ─ tft.ckpt, xgb.pkl, feature lists           │
│  ─ reports/ ─ JSON config (calibration, demand_functions)    │
│  ─ .env ─ NEWSAPI_KEY                                        │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  KÖPRÜ KATMANI                                               │
│  forecast_bridge.py → TFT + XGBoost + Two-Stage              │
│  pricing_engine.py → calibration + segment + DTD rules       │
│  sentiment/ → GDELT + RSS + classifier (her saat)            │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  İŞ KATMANI                                                  │
│  simulation_engine.py → bot bazlı simülasyon                 │
│  network_optimizer.py → EMSR-b, bid price                    │
│  report_generator/ → NLG PDF                                 │
└─────────────────────────────────────────────────────────────┘
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
       ┌─────────────────┐  ┌─────────────────┐
       │  app.py (5005)  │  │ customer_app.py │
       │  Company API    │◄─┤  BiletBul API   │
       └─────────────────┘  └─────────────────┘
                │                   │
                ▼                   ▼
       ┌─────────────────┐  ┌─────────────────┐
       │ templates/      │  │ customer/       │
       │ index.html      │  │ templates/      │
       │ simulation.html │  │ index.html      │
       │ sentiment.html  │  │ login.html      │
       │ booking.html    │  │ loading.html    │
       │ login.html      │  │                 │
       │ ...             │  │                 │
       └─────────────────┘  └─────────────────┘
```

---

## 5. Dış Bağımlılıklar

| Servis | Modül | Auth | Hata Politikası |
|---|---|---|---|
| Google News RSS | `sentiment/gnews_rss.py` | yok | sessizce boş döner |
| GDELT DOC API | `sentiment/gdelt.py` | yok | rate limit yok, fallback'e geçer |
| NewsAPI | `.env` (kullanılmıyor görünüyor) | `NEWSAPI_KEY` | — |
| HuggingFace | (DeBERTa kodu var, runtime'da çağrılmıyor) | — | log "DeBERTa" diyor ama keyword classifier kullanılıyor |
| TFT model | `data/models/tft_*.pt` | yerel dosya | startup'ta yüklenir |
| XGBoost models | `data/models/*.pkl` | yerel | startup'ta yüklenir |

---

## 6. Konfigürasyon

| Yer | Ne | Risk |
|---|---|---|
| `.env` | NEWSAPI_KEY (`d206a062...`) | Eğer git'e push olduysa **sızıntı**; .gitignore kontrol edilmeli |
| `dashboard/.env` | Aynı NEWSAPI_KEY (duplicate) | İkili dosya, hangisi okunur? |
| `app.py` `app.secret_key` | os.environ veya hardcoded "seatwise-secret-2026" | Hardcoded fallback varsa **session forge** riski |
| `customer_app.py:9` | `app.secret_key = os.environ.get("BILETBUL_SECRET", "bb-desktop-secret-2026")` | Hardcoded fallback **var** — onaylanmış risk |
| `pricing_engine.py` | FARE_CLASSES, DTD_RULES, REGION_FACTORS | Hardcoded — values değişimi commit gerektirir |
| `sentiment/scheduler.py` | INTERVAL=3600 (1 saat) | Hardcoded |
| `pricing_engine.py:321` | `1.0 + score * 0.15` | Sentiment fiyat çarpanı sabit |
| `simulation_engine.py:631` | `1.0 + score * 0.30` | Sentiment talep çarpanı sabit (pricing'den 2x) |

---

## 7. Veri Modeli Notları

- **flight_id formatı:** `TK100000_2025-11-07 16:13:00` (flight_number + departure_datetime). String. Boşluk içeriyor — URL encode dikkat.
- **DTD (Days To Departure):** 0–999 arası, 0 = kalkış günü, 180+ = uzak gelecek.
- **Fare classes:** V (Promo, multiplier 0.50), K (Discount, 0.75), M (Flex, 1.00), Y (Full Fare, 1.50).
- **Cabin:** "economy" / "business" (lowercase string).
- **Segment ID:** A-F (Business, VFR, Congress, Early Leisure, Student, Last-Minute).

---

## 8. Yeni Sürümle Gelen Önemli Yapısal Değişiklikler

1. **Auth sistemi eklendi** — login/register/logout, SHA256 password (saltsız), JSON dosya tabanlı user DB.
2. **Customer portal ayrıldı** — BiletBul ayrı Flask uygulaması.
3. **Manager Analysis tamamen elden geçti** — `_compute_*` yardımcıları + yeni `/api/manager-sensitivity`.
4. **Desktop paketleme** — pywebview ile Mac/Windows yerel uygulaması.
5. **Iki users_db.json** — `dashboard/users_db.json` ve `dashboard/customer/users_db.json`.

---

## 9. Faz 1 İçin Sıralama (Risk Önceliği)

1. **Auth & Session yönetimi** (app.py + customer_app.py'da yeni eklenmiş)
2. **Manager Analysis yeni helper'ları** (önceki K1–K4 kritik sorunlarının kapanıp kapanmadığı)
3. **simulation_engine.py** (en hassas domain logic, +63 satır)
4. **pricing_engine.py + forecast_bridge.py + network_optimizer.py**
5. **sentiment/** (önceki bulgular hâlâ geçerli mi?)
6. **customer_app.py** (tamamı yeni, hiç incelenmedi)
7. **Templates** (index, login, simulation — XSS, CSRF kontrolleri)
8. **Davranışsal test** (Faz 2)
9. **Domain logic doğrulaması** (Faz 3)
10. **Entegrasyon** (Faz 4)
11. **Rapor** (Faz 5)
