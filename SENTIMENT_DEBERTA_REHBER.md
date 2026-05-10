# Sentiment Modülü — DeBERTa Entegrasyonu Rehberi

**Tarih:** 2026-05-07  
**Modül:** `dashboard/sentiment/`  
**Etki alanı:** Sentiment scoring → demand multiplier → dynamic pricing engine

Bu doküman, sentiment modülünün **anahtar kelime tabanlı** halinden **hibrit (DeBERTa-v3 + anahtar kelime)** mimariye geçişinde yapılan tüm değişiklikleri içerir. Ekipteki herkesin yerel ortamında çalışması için izlemesi gereken adımlar dahildir.

---

## 1. Genel Bakış

### Ne değişti?
- **Önceden:** Sadece anahtar kelime sözlüğü (~416 kelime, 9 kategori). Score, sabit kategori ağırlıklarından geliyordu (örn. `security_threat = -0.8`). Text-level polarite yoktu.
- **Şimdi:** DeBERTa-v3-small fine-tuned sentiment modeli (~60MB) text-level polariteyi üretiyor. Anahtar kelime classifier event_type için korunuyor. İkisi hibrit composite formülünde birleşiyor.

### Hibrit composite formülü (per-article)
```
DeBERTa varsa:
    c_a = 0.65 · s_d  +  0.25 · w_e  +  0.10 · t_n

DeBERTa yoksa GDELT tone varsa:
    c_a = 0.60 · t_n  +  0.40 · w_e

Hiçbiri yoksa:
    c_a = w_e
```
- `s_d` ∈ [-1, +1] : DeBERTa polarite skoru (text-level)
- `w_e` ∈ {-0.8 .. +0.5} : kategori sabit ağırlığı (keyword classifier'dan)
- `t_n` ∈ [-1, +1] : GDELT tone (şu an kullanılmıyor; ToneChart endpoint için yer tutucu)

### Akış (özet)
1. Saatlik scheduler 51 şehir için Google News RSS'ten haber çeker
2. Her başlık için **iki paralel sınıflandırma**: keyword event + DeBERTa polarite
3. Hibrit `c_a` hesaplanır → şehir bazlı `C_v` recency-weighted average
4. `C_v` → SQLite cache → API → `_compute_sentiment_demand_factor()` → pricing engine

---

## 2. Önkoşullar (her ekip üyesi)

```bash
pip install sentencepiece
```

**Açıklama:** DeBERTa-v3 SentencePiece tokenizer kullanır. Bu paket olmadan model yüklenemez (`Converting from SentencePiece... failed` hatası verir, sistem keyword-only fallback'e düşer).

`torch` ve `transformers` zaten yüklü olmalı (önceki bağımlılıklar). Kontrol:
```bash
python -c "import torch, transformers, sentencepiece; print('OK')"
```

**Model indirmesi:** İlk `python app.py` başlatmasında `mrm8488/deberta-v3-small-finetuned-sst2` (~60MB) otomatik olarak `~/.cache/huggingface/hub/` altına iner. Tek seferlik. İnternet gerekir.

---

## 3. Yeni dosya

### `dashboard/sentiment/deberta.py` (YENİ, 140 satır)

Tek sorumluluğu: DeBERTa-v3-small modelini lazy-load eder, batch sınıflandırma sağlar.

**API:**
```python
from sentiment import deberta_clf

deberta_clf.warmup()           # opsiyonel: ön-yükle
deberta_clf.is_ready()         # bool
deberta_clf.classify(text)     # tek başlık → {deberta_score, deberta_label, deberta_prob_pos}
deberta_clf.classify_batch(texts, batch_size=16, neutral_band=0.40)
```

**Önemli detaylar:**
- Model lazy yüklenir (ilk classify çağrısına kadar bellek almaz). Server boot uzamaz.
- Yükleme başarısız olursa sessizce `is_ready()=False` döner; sistem keyword-only çalışmaya devam eder.
- Threading güvenli: `_LOAD_LOCK` ile.
- SST-2 binary modelinden 3-class (positive/negative/neutral) türetimi: `s_d = 2(p_pos - 0.5)`, `|s_d| < β/2` ise neutral (β = 0.40).

---

## 4. Değiştirilen dosyalar

### 4.1 `dashboard/sentiment/__init__.py`
DeBERTa modülü export'a eklendi.

```python
from . import deberta as deberta_clf
```

### 4.2 `dashboard/sentiment/scoring.py`

**`score_article()` fonksiyonu yeniden yazıldı.** Artık 3-dallı piecewise:

```python
deberta_score = article.get("deberta_score")
tone = article.get("tone")

if deberta_score is not None:
    combined = 0.65 * deberta_score + 0.25 * event_weight + 0.10 * tone_norm
elif tone is not None:
    combined = 0.6 * tone_norm + 0.4 * event_weight
else:
    combined = event_weight
```

**`compute_city_score()` fonksiyonunda iki kalibrasyon değişikliği:**

1. **Recency decay** (yumuşatıldı):
   ```python
   DECAY_LAMBDA = 0.05    # eski: 0.10 (24h → %9 idi, %30 oldu)
   ```

2. **Alert level** kuralı (false-positive temizlendi):
   ```python
   threat_count = event_counts.get("security_threat", 0)
   threat_ratio = threat_count / included_count
   high_impact_neg = sum(1 for e in high_impact if e["sentiment_score"] < 0)
   neg_ratio = neg / included_count

   if composite < -0.30 or threat_ratio >= 0.20 or (threat_count >= 3 and high_impact_neg >= 3):
       alert = "high"
   elif composite < -0.10 or threat_ratio >= 0.08 or neg_ratio >= 0.55:
       alert = "medium"
   else:
       alert = "low"
   ```
   **Önceden:** Tek bir security_threat makalesi alert'i `high` yapıyordu (Vancouver false-positive sorunu). **Şimdi:** dilution-aware, 3 koşullu disjunction.

### 4.3 `dashboard/sentiment/cache_db.py`

**Schema'ya 4 yeni kolon eklendi.** Otomatik migrasyon var (eski DB'lerde bile sessizce çalışır):

```sql
ALTER TABLE articles ADD COLUMN deberta_score REAL;
ALTER TABLE articles ADD COLUMN deberta_label TEXT;
ALTER TABLE articles ADD COLUMN deberta_prob_pos REAL;
ALTER TABLE city_scores ADD COLUMN threat_ratio REAL DEFAULT 0;
```

`init_db()` artık bu migrasyonu çağırıyor; ekipteki kimsenin DB'yi silmesi gerekmez.

**`store_articles()` ve `_load_cached_articles()`** yeni kolonları okur/yazar.

**`cleanup_old()` bug fix'i:**
- **Önceden:** `WHERE published_at LIKE '%2024%'` → URL slug'ında "2024" geçen güncel makaleleri (örn. "summer-2024-review") yanlışlıkla siliyordu.
- **Şimdi:** datetime parse + 14 gün karşılaştırması.

### 4.4 `dashboard/sentiment/scheduler.py`

**3 ana ekleme:**

1. **`_classify_articles_with_deberta()`** — her cycle'da aktif article'lara DeBERTa skoru ekler (sadece `deberta_clf.is_ready()` True ise).

2. **`_backfill_existing()`** — server boot'ta ayrı thread'de mevcut DB'deki `deberta_score IS NULL` olan tüm article'ları yeniden skorlar (~1788 makale ~2 dakikada CPU'da). Tek seferlik, restart'ta tekrar çalışır sadece yeni NULL'lar için.

3. **`_recompute_all_city_scores()`** — backfill bitince tüm city_scores'u yeniden hesaplar (yeni hibrit formül kullanılarak).

**Cycle interval:** `INTERVAL = 3600s` (saatte bir, değişmedi).
**Cleanup window:** `cleanup_old(hours=72)` (eski 48 idi; cycle gecikmelerinde veri kaybı riskini azalttık).

### 4.5 `dashboard/app.py` (üç ufak değişiklik)

1. **Yanıltıcı string'ler düzeltildi:**
   - Eski yorum: `# (GDELT + DeBERTa)` (DeBERTa hiç yokken)
   - Yeni: `# (Google News RSS + DeBERTa-v3 + Keyword Events)`
   - Eski log: `print("v2 ready (GDELT + DeBERTa)")`
   - Yeni: `print("v2 ready (RSS + DeBERTa-v3-small + keyword events)")`

2. **`/api/sentiment/status` endpoint güncellendi** — `deberta_active`, `deberta_model` alanları eklendi:
   ```json
   {
     "ready": true,
     "deberta_active": true,
     "deberta_model": "mrm8488/deberta-v3-small-finetuned-sst2",
     "source": "Google News RSS + DeBERTa-v3-small + keyword events",
     ...
   }
   ```

3. **`load_cached_scores(max_age_hours=72)`** — cache penceresi 2h'tan 72h'a çıkarıldı (server restart'larda boş ekran görmemek için).

**`_compute_sentiment_demand_factor()`** değişmedi (zaten doğruydu); ama şimdi gerçek DeBERTa skoruyla beslendiği için sonuçları daha hassas.

---

## 5. Migrasyon — eski makineler için

Hiç manuel adım yok. Sadece:

```bash
git pull
pip install sentencepiece
python dashboard/app.py
```

**İlk başlatmada görülecek log:**
```
[Sentiment] Loaded 51 cities from cache
[Scheduler] Started (interval=3600s, cities=51)
[Sentiment] v2 ready (RSS + DeBERTa-v3-small + keyword events)
[DeBERTa] Loading mrm8488/deberta-v3-small-finetuned-sst2 (first run downloads ~60MB)...
[DeBERTa] Ready. Labels: {0: 'negative', 1: 'positive'}
[DeBERTa] Backfill starting — existing articles being re-scored...
[DeBERTa] Backfill: scoring 1788 articles...
[DeBERTa] Backfill done — 1788 articles updated in 124s
```

İlk indirme + backfill toplamda ~3 dakika. Sonraki başlatmalar saniyeler içinde.

---

## 6. Doğrulama

```bash
# Server ayağa kalktıktan ~2 dakika sonra:
curl http://localhost:5005/api/sentiment/status
```

Beklenen yanıt:
```json
{
  "ready": true,
  "deberta_active": true,
  "deberta_model": "mrm8488/deberta-v3-small-finetuned-sst2",
  "cities_count": 51,
  "loading": false
}
```

**`deberta_active: false` ise:**
- `pip install sentencepiece` çalıştırılmamış olabilir
- Server log'larında `[DeBERTa] Load failed: ...` aranır
- Fallback olarak system yine çalışır (keyword-only), pricing kararları sentiment'tan etkilenmeye devam eder ama hassasiyeti düşer.

DB doğrulama:
```bash
sqlite3 dashboard/sentiment_v2.db "SELECT COUNT(*) FROM articles WHERE deberta_score IS NOT NULL;"
# 1700+ olmalı (cache'teki article sayısına yakın)
```

---

## 7. Etkilendiği yerler (sentiment skorunu tüketen kod)

Bu modülün ürettiği `composite_score` ve `alert_level` aşağıdaki noktalarda kullanılır. Hiçbir API kontratı değişmedi — sadece skorlar artık daha hassas:

| Dosya | Satır | Kullanım |
|---|---:|---|
| `dashboard/app.py` | 3213 | `_compute_sentiment_demand_factor(route)` → demand × (1 ± 0.20) |
| `dashboard/app.py` | 3409, 3494, 3594 | API response'larında sentiment_score alanı |
| `dashboard/pricing_engine.py` | 98, 317 | Pricing engine'in opsiyonel sentiment cache okuması |
| `dashboard/report_generator/collector.py` | 139 | PDF raporlarında composite_score gösterimi |
| `dashboard/templates/sentiment.html` | 395 | Frontend `/api/sentiment/all` polling |

---

## 8. Rollback (gerekirse)

DeBERTa'yı tamamen devre dışı bırakmak için:

```python
# dashboard/sentiment/scheduler.py içinde:
def _classify_articles_with_deberta(articles):
    return  # no-op
```

DB schema değişikliği geriye dönük uyumlu — eski koda DB ile sorun çıkmaz, sadece yeni kolonlar boş kalır.

---

## 9. Bilinen sınırlılıklar

- **DeBERTa fine-tune SST-2 üzerinde** (film yorumları). Havayolu haber alanına aktarım iyi ama domain-specific fine-tune ile daha da iyi olur.
- **GDELT tone şu an kullanılmıyor** (ArtList endpoint'i tone yaymıyor). Hibrit formülde 3. dal şu an "ölü kod" — ToneChart entegrasyonuyla canlanacak.
- **Multilingual yok**: anahtar kelime sözlüğü yalnızca İngilizce. Türkçe/Arapça destek için sözlük genişletmesi gerek.
- **CPU-only inference**: ~80-150ms/başlık. 51 şehir × ~20 makale = ~1000 başlık → cycle başına +20-40 saniye ek süre.

---

## 10. İletişim

Bu rehbere ait sorular için: ahmetfgokbulut@gmail.com / Slack #seatwise

Detaylı akademik açıklama: `Sentiment_Modulu_Teknik_Rapor.pdf` (Bölüm 2.7)
