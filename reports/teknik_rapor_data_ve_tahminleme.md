# SeatWise Teknik Rapor: Data Pipeline + Tahminleme Katmanlari

## 1. Proje Ozeti

SeatWise, bir havayolu icin uctan uca gelir yonetimi (Revenue Management) sistemidir. Istanbul hub merkezli, 50 rota, 51 destinasyon, 2 kabin sinifi (economy/business) ve 2 yillik (2025-2026) sentetik veri uzerinde calisir. Toplam ~37 milyon gozlem iceren panel veri yapisina sahiptir.

Sistem su katmanlardan olusur:
- **Data Pipeline**: Ham veriden ML-ready veri uretimi
- **Tahminleme**: 3 farkli ML modeli (TFT, Pickup XGBoost, 2-Stage XGBoost)
- **Fiyatlandirma**: 4 carpanli dinamik fiyat motoru + 4 fare class yonetimi
- **Network Optimizer**: EMSR-b + O&D bid price + fare proration
- **Sentiment**: 51 sehir icin haber bazli talep etkisi
- **Simulasyon**: Gercek zamanli bot yolcu simulasyonu

Bu rapor simulasyon oncesindeki tum katmanlari kapsar.

---

## 2. Veri Yapisi: Panel Data

### 2.1 Ham Veri

| Dosya | Boyut | Icerik |
|-------|-------|--------|
| `data/raw/flight_snapshot_v2.parquet` | 205 MB | Her ucus x DTD noktasi icin ticket + ancillary revenue, ~37M satir |
| `data/raw/bookings_enriched.parquet` | 341 MB | Booking-level detay veri |
| `data/raw/flight_snapshot.parquet` | 453 MB | v1 format (kullanilmiyor) |

### 2.2 Panel Yapisinin Aritmetigi

```
100 rota x 2 kabin = 200 entity
Her entity ~511 ucus (2 yil)
Her ucus 181 DTD noktasi (0-180 gun)
Toplam: 200 x 511 x 181 = ~18.5M gozlem/yil = ~37M toplam
```

Bu yapi cross-sectional ogrenim saglar: her rota, diger 199 rotanin booking pattern'indan ogrenebilir.

### 2.3 Sentetik Veri Uretimi

`scripts/data_prep/flight_snapshot.py` sentetik veriyi uretir:
- Baz fiyat: `economy = max(mesafe_km * 0.08, 150)`, `business = max(mesafe_km * 0.35, 800)`
- Yolcu uretimi: DTD bazli S-curve ile dagilim
- Sezonsellik, hafta gunu, ozel gun faktorleri veride mevcuttur

---

## 3. Data Pipeline (scripts/data_prep/)

### 3.1 build_demand_training.py

**Girdiler:** flight_snapshot_v2.parquet + flight_metadata.parquet
**Cikti:** `data/processed/demand_training.parquet` (36.9M satir, 51 kolon)
**Teknoloji:** DuckDB (pandas OOM engellemek icin)

Temel SQL islemi:
```sql
SELECT flight_id, cabin_class, dtd,
  CASE WHEN dtd<=3 THEN 0 WHEN dtd<=7 THEN 1 ... END AS dtd_bucket,
  pax_sold_today AS y_target,
  pax_sold_cum, pax_last_7d, capacity, remaining_seats, load_factor,
  region, distance_km, flight_time_min,
  dep_year, dep_month, dep_dow, dep_hour,
  ff_gold_pct, ff_elite_pct
FROM flight_snapshot JOIN metadata
```

**Onemli istatistikler:**
- y_mean: 0.7884 pax/gun (cok dusuk — zero-inflated)
- y_zero_rate: %70.8 (gunlerin cogunlugunda satis yok)
- Bu zero-inflation, 2-Stage modelin varlik sebebidir

### 3.2 build_tft_route_daily.py

**Girdiler:** flight_snapshot_v2 (37M satir)
**Cikti:** `data/processed/tft_route_daily.parquet` (138K satir, 42 kolon)

Gruplama: `GROUP BY entity_id (= route_cabin), dep_date`

Bu script, 37M satiri 200 entity x 730 gun = 146K satira indirger. TFT'nin girdi boyutudur.

Uretilen ozellikler:
- `y_demand` (hedef): Gunluk toplam yolcu sayisi
- `n_flights`: O gun kac ucus var
- `is_special_period`, `special_period`: Bayram/ozel gun flag'leri
- Takvim ozellikleri: yil, ay, gun, hafta gunu
- Lag ozellikleri: 1-7 gunluk gecikmeli talepler
- Rolling istatistikler: 7 ve 14 gunluk hareketli ortalama/std

### 3.3 build_pickup_master.py

**Girdiler:** demand_training.parquet + flight_metadata + tft_route_daily
**Cikti:** `data/processed/pickup_master.parquet` (36.8M satir, 54 kolon)

Kritik fark: **y_target = remaining_pax = final_pax - pax_sold_cum**

Bu, "bu DTD noktasindan itibaren kac yolcu daha gelecek?" sorusunu yanitlar. Pickup XGBoost'un hedefidir.

Eklenen ozellikler:
- Booking pace: pax_sold_cum, pax_last_7d, pax_last_14d
- Kapasite: capacity, remaining_seats, load_factor
- Rota/zaman: distance_km, flight_time_min, dep_year/month/dow
- TFT tahmini: tft_prediction (gunluk talep tavani)
- Fiyat lag'leri: mean_price_last_7d, std_price_last_7d

**Train/Test split:** 2025 = train (18.4M), 2026 = test (18.4M) — temporal split

### 3.4 add_event_tags.py

demand_training'e 15 boolean event tag'i ekler:
- `is_summer_vacation`, `is_ramadan_period`, `is_eid_period`
- `is_year_end_period`, `is_spring_break`, `is_holiday_period`
- `is_weekend`, `is_high_frequency_day`, `is_low_frequency_day`
- `is_business_peak_dow` (Car-Per-Cum), `is_weekend_flight`
- `is_long_haul` (>2000km), `is_short_haul`, `is_premium_cabin`
- `is_high_load_factor_period`

### 3.5 build_demand_functions.py

**Cikti:** `reports/demand_functions_report.json`

6 yolcu segmenti tanimlar (A-F). Her segment icin:
- Booking window (min_dtd, max_dtd, peak_dtd)
- WTP (Willingness to Pay) carpani (min, max)
- Fiyat elastikiyeti
- Sezonsal boost faktorleri
- DTD bazli talep egrisi

| Segment | Ad | Pay | Booking Penceresi | WTP | No-Show |
|---------|----|-----|-------------------|-----|---------|
| A | Is Yolcusu | %15 | 0-14 DTD (peak 5) | 1.8-2.5x | %15 |
| B | VFR (akraba ziyaret) | %20 | 14-60 DTD (peak 45) | 1.3-1.8x | %5 |
| C | Kongre | %12 | 7-30 DTD (peak 21) | 1.2-1.6x | %8 |
| D | Erken Tatilci | %28 | 60-180 DTD (peak 120) | 0.8-1.2x | %3 |
| E | Son-Dk Tatilci | %10 | 0-30 DTD (peak 75) | 0.6-1.0x | %7 |
| F | Premium | %15 | 14-60 DTD (peak 2) | 2.5-3.5x | %20 |

### 3.6 build_passenger_clusters.py

K-means ile yolcu kumeleme. Cikti: `data/processed/passenger_clusters.parquet` (5.1 MB)
Dashboard'da gorsellestirilir.

### 3.7 calibrate_from_data.py

**Cikti:** `reports/calibration_report.json`

102K booking uzerinden:
- Load factor curve faktorleri (LF95+: 2.40, LF85-95: 1.95, LF70-85: 1.68, LF50-70: 1.37, LF30-50: 1.12)
- Baz fiyat regresyonu: `base = intercept + distance_km * price_per_km` (R^2 = 0.979)
- Rota bazli fiyat faktorleri

---

## 4. Tahminleme Modelleri (scripts/training/)

### 4.1 Temporal Fusion Transformer (TFT) — Stratejik Tahmin

**Dosya:** `scripts/training/train_tft_model.py` (lokal referans), `scripts/kaggle/kaggle_tft_route.py` (gercek egitim)
**Girdi:** tft_route_daily.parquet (200 entity, 730 gun, 42 ozellik)
**Cikti:** `data/models/tft_route_daily_model.pt` (2 MB)

**Mimari:**
- Encoder: 20 gunluk gecmis penceresi
- Decoder: 30 gunluk tahmin ufku
- Kayip: Quantile Loss (p10, p50, p90)
- Epoch: 50
- Yapi: Variable Selection Network -> Self-Attention -> LSTM

**Performans:**
- Test MAE: 14.03 pax
- Test Korelasyon: 0.991
- MAPE: %5.9

**Rolu:** Rota-gun bazinda toplam talep tahmini (tavan/taban regulatoru). Simulasyonda S-curve ile gunluk paylara ayrilir:
```python
cum_fraction = 1.0 - (dtd / 180.0) ** 1.5  # kumulatif oran
daily_fraction = cum_fraction_today - cum_fraction_tomorrow
tft_daily = tft_total * daily_fraction
daily_floor = tft_daily * 0.3   # alt sinir
daily_ceiling = tft_daily * 2.0  # ust sinir
```

Unconstraining: LF > %90 ise gizli talep %15 eklenir (kapasite kisitlamasindan dolayi gorunmeyen talep).

**Not:** TFT Kaggle ortaminda egitilmistir (GPU gereksinimi). Lokal script referans amaclidir.

### 4.2 Pickup XGBoost — Taktik Tahmin (Aktif Model)

**Dosya:** `scripts/training/train_pickup_xgb.py`
**Girdi:** pickup_master.parquet (2025 train: 18.4M, 2026 test: 18.4M)
**Cikti:** `data/models/pickup_xgb.json` (7 MB), `data/models/pickup_feature_list.json` (49 ozellik)

**Hedef:** `remaining_pax = final_pax - pax_sold_cum` (bu noktadan sonra kac yolcu daha gelecek?)

**Hiperparametreler:**
```python
params = {
    max_depth: 7,
    learning_rate: 0.05,
    subsample: 0.8,
    colsample_bytree: 0.8,
    min_child_weight: 10,
    eval_metric: 'mae',
    tree_method: 'hist',
    seed: 42,
    rounds: 500
}
```

**49 Ozellik (Gruplar):**
1. Booking pace (5): pax_sold_cum, pax_last_7d, pax_sold_today, ticket_rev_cum, anc_rev_cum
2. Kapasite (4): capacity, remaining_seats, load_factor, is_weekend
3. Rota/zaman (4): distance_km, flight_time_min, ff_gold_pct, ff_elite_pct
4. DTD (1): dtd
5. Takvim (4): dep_year, dep_month, dep_dow, dep_hour
6. Kabin one-hot (3): cabin_class_business, economy, nan
7. Bolge one-hot (6): Africa, Americas, Asia, Europe, Middle East, nan
8. DTD bucket one-hot (8): dtd_bucket_0.0 ... 6.0, nan

**Performans:**
| Metrik | Pickup XGB | Baseline (linear) | Iyilesme |
|--------|-----------|-------------------|----------|
| MAE | 3.45 pax | 11.65 pax | %70.4 |
| RMSE | 6.02 pax | 18.27 pax | %67 |
| WAPE | %9.82 | — | — |

**DTD bazli performans:**
- DTD 1-7: MAE 4-5 (%28-35 hata — son dakika belirsizligi)
- DTD 8-30: MAE 3-4 (%12-15 hata — en iyi bolge)
- DTD 31-60: MAE 3.5-4 (%9-10 hata)
- DTD 61+: MAE 4-5 (%20-25 hata — erken donem belirsizligi)

**Simulasyondaki rolu:** Supply multiplier (arz carpani) hesaplama:
```python
expected_final_lf = (sold + predicted_remaining) / capacity
if expected_final_lf >= 0.95: supply_mult = 1.80
elif expected_final_lf >= 0.85: supply_mult = 1.40
elif expected_final_lf >= 0.70: supply_mult = 1.15
else: supply_mult = 1.00
```

### 4.3 Two-Stage XGBoost — Baseline (Legacy)

**Dosya:** `scripts/training/train_demand_model.py`
**Cikti:** `data/models/xgb_demand_classifier.pkl` + `xgb_demand_regressor.pkl`
**Ozellikler:** 31 (pickup'in alt kumesi)

**Yapi:**
1. **Classifier (XGBClassifier):** P(satis_var) — AUC 0.835
2. **Regressor (XGBRegressor):** E[pax|satis_var] — MAE 0.78

**Tahmin:** `daily_demand = P(satis_var) * E[pax|satis_var]`

**Neden legacy?** Zero-inflation problemini cozuyor ama genel performansi basit ortalamaya yakin. Simulasyonda Two-Stage "modulasyon" rolu gorur: p_sale dusukse botlar azalir.

### 4.4 Basarisiz Denemeler

- `train_xgb_enhanced.py`: Ek ozelliklerle gelismis XGB denemesi — basarisiz
- `kaggle_tft_notebook.py`: Entity ID problemi nedeniyle basarisiz TFT denemesi

---

## 5. Forecast Bridge — Modelleri Birlestiren Katman

**Dosya:** `dashboard/forecast_bridge.py` (255 satir)

Uc modeli "gaz pedali / hiz limiti / direksiyon" metaforuyla birlestirir:

| Model | Rol | Metafor |
|-------|-----|---------|
| Two-Stage XGB | Gunluk bot sayisi | Gaz pedali |
| TFT | Tavan/taban regulatoru | Hiz limiti |
| Pickup XGB | Pricing supply multiplier | Direksiyon |

**Batch Predict Akisi (gunde 1 kez):**
1. Tum aktif ucuslar icin `_build_features()` ile ozellik vektoru olustur
2. Two-Stage: P(sale) ve E[pax] hesapla -> daily_demand
3. Pickup: remaining_pax tahmin et
4. Sonuclari cache'le (ayni gun tekrar hesaplanmaz)

**Feature Mapping (onemli detaylar):**
- `pax_last_7d`: Minimum 3 verilir (cold-start onleme — yoksa model "satis olmaz" diyor, kisir dongu)
- `dep_hour`: Sabit 12.0 (simulasyonda saat bilgisi yok)
- `ff_gold_pct`: Sabit 0.15, `ff_elite_pct`: Sabit 0.05
- `ticket_rev_cum`: revenue_dynamic * 0.85
- `anc_rev_cum`: revenue_dynamic * 0.15

---

## 6. Fiyatlandirma Motoru

**Dosya:** `dashboard/pricing_engine.py` (558 satir)

### 6.1 Ana Formul

```
fiyat = baz_fiyat x arz_carpani x talep_carpani x sentiment_carpani x musteri_carpani
```

### 6.2 Baz Fiyat

Kalibrasyon regresyonundan (R^2 = 0.979):
```
base = intercept + distance_km * price_per_km
```
Fallback: `economy = max(km * 0.08, 150)`, `business = max(km * 0.35, 800)`
Rota bazli faktor carpilir (calibration_report.json'dan).

### 6.3 Arz Carpani (Supply Multiplier)

**Model-driven yol (Pickup XGB varsa):**
```
expected_final_lf = (sold + predicted_remaining) / capacity
LF >= 95%: 1.80x | LF >= 85%: 1.40x | LF >= 70%: 1.15x | else: 1.00x
```
DTD boost: <=3 gun: 1.3x, 4-7: 1.15x, 8-14: 1.05x

**Fallback (kalibrasyon):**
LF95+: 2.40x, LF85-95: 1.95x, LF70-85: 1.68x, LF50-70: 1.37x, LF30-50: 1.12x

Sold out: 2.5x (maximum)

### 6.4 Talep Carpani (Demand Multiplier)

- **Sezon (ay):** Ocak 0.85 ... Temmuz 1.30 ... Aralik 1.20
- **Ozel gun:** Ramazan Bayrami 1.5x, Kurban Bayrami 1.6x, Cumhuriyet 1.25x, Yilbasi 1.4x
- **Hafta gunu:** Paz 1.05, Sal 1.00, Per 1.10, Cum 1.15
- **Dampening:** `damped = 1.0 + (raw - 1.0) * 0.7` (talep etkisi arz'dan dusuk kalir)

### 6.5 Sentiment Carpani

- 51 destinasyon sehri icin GDELT + Google News RSS ile haber toplama
- Keyword-based siniflandirma (9 kategori, 416 anahtar kelime):
  - Security Threat (-0.8), Strike (-0.6), Weather (-0.7)
  - Flight Disruption (-0.5), Tourism (+0.5), Political (-0.6)
  - Health Crisis (-0.7), Positive Travel (+0.4), Local Events (+0.3)
- 14 gunluk recency filtresi
- Fiyata etki: `1.0 + score * 0.15` (yani +-15%)

### 6.6 Musteri Carpani

Segment WTP'den: `mult = 0.85 + (wtp_avg - 0.5) * 0.2` (sinirlar: 0.80 - 1.40)

Session bazli modulasyonlar (booking.html'den):
- Geri donen musteri: +5%
- Sayfada >180sn: -3% (tereddut)
- 60-180sn: +2% (ilgili)
- >5 arama: -2% (fiyat avcisi)
- Tek arama: +3% (kararli)
- Terk edilmis sepet: +8%
- Gold/Elite FF: +6%, Silver: +2%
- Mobil: +2% (aciliyet)
- 4+ yolcu: -3% (aile)

Toplam sinirlar: 0.90 - 1.20

### 6.7 Fare Class Yonetimi

**4 Fare Class:**

| Class | Ad | Carpan | LF Limiti | Kota |
|-------|-----|--------|-----------|------|
| V | Promosyon | 0.50x | LF < %40 | %15 kapasite |
| K | Indirimli | 0.75x | LF < %60 | %25 kapasite |
| M | Esnek | 1.00x | LF < %85 | %35 kapasite |
| Y | Tam Fiyat | 1.50x | Her zaman | %100 |

**DTD Kurallari:**

| DTD | Acik Siniflar |
|-----|---------------|
| 60+ gun | V, K, M |
| 30-59 gun | K, M |
| 14-29 gun | K, M, Y |
| 7-13 gun | M, Y |
| 0-6 gun | Sadece Y |

Ek kurallar:
- Talep basinci > 1.3: en ucuz sinif kapanir
- Talep basinci < 0.6 ve DTD > 7: bir alt sinif acilir

### 6.8 Satin Alma Olasiligi (Bot karari)

```python
if offered_price > wtp_max_price: return 0.0
elif offered_price <= wtp_min_price: return 0.95
else: return 0.95 * (1 - ratio^0.7)  # sigmoid benzeri
```

---

## 7. Network Optimizer (EMSR-b + O&D)

**Dosya:** `dashboard/network_optimizer.py` (189 satir)

### 7.1 EMSR-b (Expected Marginal Seat Revenue)

Her fare class icin koruma seviyesi hesaplar:
```python
ratio = price_low / price_high
z = norm.ppf(1 - ratio)  # inverse normal CDF
protection = demand_mean + demand_std * z
```

Talep tahmini TFT'den beslenir (varsa). EMSR-b, V ve K siniflarinin kotalari dolunca kapatir. Booking pace kontrolu ile geri acabilir:
- pace_ratio < 0.70: V ve K tekrar acilir
- pace_ratio < 0.85: V tekrar acilir

### 7.2 Bid Price

En dusuk acik fare class'in fiyati = minimum kabul edilebilir fiyat.

### 7.3 Fare Proration (Connecting Yolcular)

```
total_fare = (origin_base + dest_base) * 0.85  (connecting indirimi)
leg_contribution = total_fare * (dest_distance / total_distance)
```

IST hub'inda minimum %10 connecting yolcu orani.

### 7.4 O&D Kabul/Red

LF > %70 sonrasi aktif:
- `leg_contribution >= bid_price` -> kabul
- Aksi halde displacement (red)

---

## 8. Sentiment Modulu

**Dosyalar:** `dashboard/sentiment/` (10 dosya)

| Dosya | Islem |
|-------|-------|
| `cities.py` | 51 sehir, havaalani kodlari, bolge mapping |
| `gdelt.py` | GDELT API wrapper (uluslararasi haber akisi) |
| `gnews_rss.py` | Google News RSS parser |
| `fetcher.py` | Veri toplama orkestrasyonu |
| `classifier.py` | Keyword-based olay siniflandirma (0 ML, 416 keyword, 9 kategori) |
| `scorer.py` + `scoring.py` | Skor hesaplama ve aggregation |
| `scheduler.py` | Arka plan zamanlayici |
| `cache_db.py` | SQLite cache |

**Veri akisi:**
1. GDELT/Google News'ten haber cek
2. Destinasyon sehrini tespit et
3. Keyword matching ile kategori belirle
4. Impact factor uygula (-0.8 ile +0.5 arasi)
5. 14 gunluk ortalamayi hesapla
6. Talebe +-30%, fiyata +-15% etki

---

## 9. Kalibrasyon ve Raporlar

### 9.1 calibration_report.json
- LF curve faktorleri (veriden ogrenilmis)
- Baz fiyat regresyonu (intercept + price_per_km, R^2=0.979)
- Rota bazli fiyat faktorleri

### 9.2 demand_functions_report.json
- 6 segment tanimlari (A-F)
- Booking window, WTP, elastikiyet, sezon boost

### 9.3 pickup_xgb_metrics.json
- MAE: 3.45, RMSE: 6.02, MAPE: %9.82

### 9.4 demand_metrics.json
- 2-Stage XGB: AUC 0.835, MAE 0.78

---

## 10. Veri Akis Semasi

```
HAM VERI (flight_snapshot_v2.parquet, 37M satir)
  |
  +--[build_demand_training.py]--> demand_training.parquet (36.9M, 51 kolon)
  |    |                              |
  |    +--[add_event_tags.py]         |
  |    |                              |
  |    +--[build_demand_functions.py]--> demand_functions_report.json (6 segment)
  |    |
  |    +--[build_tft_route_daily.py]--> tft_route_daily.parquet (138K, 42 kolon)
  |    |                                  |
  |    |                                  +--[train_tft_model.py / Kaggle]
  |    |                                       |
  |    |                                       +--> tft_route_daily_model.pt
  |    |                                       +--> tft_predictions.parquet
  |    |
  |    +--[build_pickup_master.py]--> pickup_master.parquet (36.8M, 54 kolon)
  |                                     |
  |                                     +--[train_pickup_xgb.py]
  |                                          |
  |                                          +--> pickup_xgb.json (MAE 3.45)
  |
  +--[calibrate_from_data.py]--> calibration_report.json (R^2=0.979)
  |
  +--[train_demand_model.py]--> xgb_demand_classifier.pkl + regressor.pkl

                      RUNTIME (Dashboard)
                      ===================
  tft_predictions --> ForecastBridge --> TFT band (tavan/taban)
  pickup_xgb ------> ForecastBridge --> predicted_remaining
  2-stage xgb -----> ForecastBridge --> daily_demand, p_sale
                          |
                          v
                    PricingEngine <-- calibration_report
                          |           sentiment scores
                          |           segment defs
                          v
                    NetworkOptimizer (EMSR-b, bid price)
                          |
                          v
                    SimulationEngine (bot yolcular)
```

---

## 11. Model Ensemble Stratejisi

Uc model birbirini tamamlar:

| Boyut | TFT | Pickup XGB | 2-Stage XGB |
|-------|-----|-----------|-------------|
| **Granularity** | Rota-gun | Ucus-DTD | Ucus-DTD |
| **Hedef** | Toplam gunluk talep | Kalan yolcu | P(satis) x E[pax] |
| **Rol** | Stratejik (uzun vade) | Taktik (arz sinyali) | Modulasyon (varlama) |
| **Guclu yan** | Attention, trend | Booking pace | Zero-inflation |
| **Zayif yan** | Dusuk granularity | Cold-start | Basit ortalamaya yakin |

ForecastBridge bunlari soyle birlestirir:
1. TFT toplam talebi verir
2. S-curve ile gunluk paya ayrilir
3. Catch-up mekanizmasi: satislar beklentinin gerisindeyse hizlandirilir
4. Two-Stage p_sale ile modulasyon: `daily_bots *= (0.5 + 0.5 * p_sale)`
5. Pickup, pricing engine'e supply sinyali verir
