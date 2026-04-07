# dashboard/

Flask tabanli web dashboard. Ucus arama, talep tahmini, rota analizi ve sentiment.

## Calistirma
```bash
cd dashboard
python app.py
# http://localhost:5005
```

## Dosyalar

| Dosya | Aciklama |
|-------|----------|
| app.py | Ana Flask uygulamasi — tum API endpoint'leri |
| templates/index.html | Ana dashboard HTML (~190 KB, tek sayfa uygulama) |
| templates/sentiment.html | Sentiment sayfasi HTML |
| sentiment/ | Sentiment modulu (fetcher.py + scorer.py) |
| sentiment_app.py | Sentiment analizi Flask uygulamasi (ayri) |
| sentiment_cache.db | Sentiment sonuclari cache DB |
| static/logo.png | Logo |

## Kullanici Arayuzu

### Ucus Arama
Ucus numarasi veya havaalani kodu ile arama. Tarih secimi.

### Sekmeler (ucus secildikten sonra)
**2025 ucuslari:** Sadece "Rezervasyon Verileri" sekmesi gorunur (gerceklesen veri)
**2026 ucuslari:** Uc sekme gorunur:

1. **Rezervasyon Verileri** — Gercek booking verisi
   - KPI: toplam yolcu, doluluk, gelir, ort. bilet fiyati
   - Grafikler: koltuk doluluk, gunluk satis pace
   - DTD bazli detay tablosu

2. **Talep Tahmini** (XGBoost Pickup) — Ucus bazli kalan talep tahmini
   - KPI: toplam yolcu, doluluk, kapasite, model dogruluk (MAE/MAPE)
   - Grafik: stacked bar (mevcut rez. + tahmini kalan talep) + gerceklesen toplam cizgisi
   - DTD bazli detay tablosu (tahmin vs gercek)

3. **Rota Analizi** (TFT) — Rota bazli gunluk talep trendi
   - Kalkis tarihine kadar son 60 gunluk rota talebi
   - Gecen yil ile karsilastirma (sezonsal baseline)

### Ust Menu Sekmeleri
Segmentler, Trend, Rotalar, Olaylar, Fare Class, Simulasyon, Talep Fonk., Risk, Sentiment

## API Endpoint'leri

| Endpoint | Aciklama |
|----------|----------|
| GET /api/flights?q=TK | Ucus/havaalani arama |
| GET /api/flight/{fn} | Ucus tarih listesi |
| GET /api/flights/date?date=2026-03-27 | Tarihe gore ucuslar |
| GET /api/snapshot/{flight_id}?cabin=economy | DTD bazli booking verisi |
| GET /api/pickup/{flight_id}?cabin=economy | XGBoost pickup tahminleri |
| GET /api/forecast/{flight_id}?window=60&cabin=economy | TFT rota tahmini |
| GET /api/clusters | Yolcu kumeleme raporu |
| GET /api/trends?year=2026&cabin=economy | Talep trendleri |
| GET /api/events?year=2026 | Olay bazli analiz |
| GET /api/sentiment/{city} | Sehir sentiment analizi |

## Bagimliliklar
```
flask
duckdb
pandas
numpy
xgboost
```
