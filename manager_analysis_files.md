# 📊 Manager Analysis Paneli — Kullanılan Dosyalar

Manager Analysis paneli **3 ana dosya** üzerinde oluşturulmuş + 1 destek dosyası kullanıyor:

---

## 1. Backend API — `app.py`

**Dosya:** [app.py](file:///Users/yagmurguzeler/Desktop/FENS401-402-Bitirme-Projesi/dashboard/app.py)

Bu dosyada Manager Analysis ile ilgili **2 API endpoint** var:

### `/api/manager-analysis` (GET) — Satır 2084–2212
- Tüm uçuşları pricing verileriyle birlikte döndürür
- Cabin filtresi destekler
- DuckDB ile parquet dosyalarından veri çeker
- `PricingEngine.compute_price()` ile dinamik fiyat hesaplar
- KPI özeti (toplam uçuş, ortalama LF, fiyat, gelir)

### `/api/manager-override` (POST) — Satır 2215–2356
- Manuel fiyat ayarlama (what-if analizi)
- Talep elastikiyeti etkisi hesaplar (Economy: ε=-1.2, Business: ε=-0.8)
- DTD override desteği (simüle edilmiş kalkış günü)
- Fare class bazlı fiyat kırılımı
- Gelir delta ve tahmini final LF hesaplaması
- `pricing_engine.DTD_RULES` kurallarını kullanarak fare class açık/kapalı durumunu belirler

---

## 2. Frontend — `index.html`

**Dosya:** [index.html](file:///Users/yagmurguzeler/Desktop/FENS401-402-Bitirme-Projesi/dashboard/templates/index.html)

Bu dosyada Manager Analysis ile ilgili **3 bölüm** var:

### CSS Stilleri — Satır 2333–2419
- `.mgr-panel`, `.mgr-header`, `.mgr-badge` — Panel düzeni
- `.mgr-kpi-row`, `.mgr-kpi` — KPI kartları
- `.mgr-filter-row`, `.mgr-select` — Filtre kontrolleri
- `.mgr-table-wrap`, `.mgr-table` — Uçuş tablosu
- `.mgr-override-section` — Override paneli
- `.mgr-slider-row`, `.mgr-slider`, `.mgr-pct-display` — Fiyat ayar slider'ı
- `.mgr-results-row`, `.mgr-result-card` — Sonuç kartları
- `.mgr-fare-grid`, `.mgr-fare-card` — Fare class kartları
- `.mgr-charts-row` — Grafikler düzeni
- `.mgr-info-box` — Bilgi kutusu

### HTML Yapısı — Satır 2866–3001
- Panel container (`#mgrPanel`)
- KPI satırı (`#mgrKpis`): Total Flights, Avg LF, Avg Price, Total Revenue
- Cabin filtresi ve arama (`#mgrCabinFilter`, `#mgrSearch`)
- Uçuş tablosu (`#mgrTable`, `#mgrTableBody`)
- Override bölümü (`#mgrOverrideSection`):
  - DTD Simulator dropdown (`#mgrDtdOverride`)
  - Fiyat ayar slider'ı (-30% ↔ +50%)
  - Hızlı ayar butonları (-20%, -10%, -5%, Reset, +5%, +10%, +20%)
  - Sonuç kartları: Adjusted Price, Revenue Δ, Demand Impact
  - Fare Class Breakdown grid (`#mgrFareGrid`)
  - Info Box (elastikiyet açıklaması)
  - 2 grafik: Price vs Demand Impact, Revenue Impact Curve

### JavaScript — Satır 6024–6332
- `toggleManagerAnalysis()` — Panel aç/kapa
- `loadManagerData()` — API'den veri çek, KPI ve tablo güncelle
- `renderMgrTable()` — Uçuş tablosu render
- `filterMgrTable()` — Tablo arama/filtre
- `selectMgrFlight()` — Uçuş seçimi, override panelini aç
- `setMgrPct()`, `onMgrSliderChange()`, `onMgrPctInput()` — Slider kontrolü
- `onMgrDtdChange()` — DTD simulator değişimi, fare class bilgisi
- `computeMgrOverride()` — API'ye override isteği gönder
- `renderMgrResults()` — Sonuçları render (fiyat, gelir, talep, fare class kartları)
- `buildMgrCharts()` — Chart.js ile 2 sensitivity grafik oluştur

### Navigasyon Butonu — Satır 2463
- Analytics dropdown menüsünde "📊 Manager Analysis" butonu

---

## 3. Pricing Engine — `pricing_engine.py`

**Dosya:** [pricing_engine.py](file:///Users/yagmurguzeler/Desktop/FENS401-402-Bitirme-Projesi/dashboard/pricing_engine.py)

- `PricingEngine.compute_price()` — Dinamik fiyat hesaplama
- `DTD_RULES` — Fare class açılma kuralları (DTD aralıklarına göre)
- Manager override API'si bu engine'i kullanarak temel fiyatı hesaplar

---

## 4. Veri Dosyaları (Dolaylı Kullanım)

| Dosya | Konum | Kullanım |
|-------|-------|----------|
| Snapshot Parquet | `data/processed/snapshot_enriched.parquet` | Uçuş verileri (LF, pax, gelir) |
| Metadata Parquet | `data/processed/flight_metadata.parquet` | Rota bilgileri (havalimanı, mesafe, kapasite) |
| Demand Functions Report | `reports/demand_functions_report.json` | Segment bilgileri, pricing engine'e segment parametreleri sağlar |

---

## Özet Tablo

| Dosya | İlgili Satırlar | İçerik |
|-------|-----------------|--------|
| `app.py` | 2084–2356 | Backend API (2 endpoint) |
| `index.html` | 2333–2419 | CSS stilleri |
| `index.html` | 2463 | Navigasyon butonu |
| `index.html` | 2866–3001 | HTML yapısı |
| `index.html` | 6024–6332 | JavaScript mantığı |
| `pricing_engine.py` | Tüm dosya | Fiyat hesaplama motoru |
