# Sentiment Intelligence Dashboard
## Kurulum & Çalıştırma

### 1. Bağımlılıkları kur
```bash
pip install flask transformers torch requests
```

GPU yoksa (CPU için daha hafif):
```bash
pip install flask transformers[cpu] requests
```

### 2. NewsAPI key'ini al
https://newsapi.org/register → ücretsiz hesap → API key kopyala

### 3. Çalıştır
```bash
# Linux/Mac:
export NEWSAPI_KEY="buraya_key_yaz"
python sentiment_app.py

# Windows:
set NEWSAPI_KEY=buraya_key_yaz
python sentiment_app.py
```

### 4. Tarayıcıda aç
http://localhost:5002

---

## Dosya Yapısı
```
sentiment_app.py          ← Ana Flask uygulaması (port 5002)
sentiment/
  __init__.py
  fetcher.py              ← NewsAPI entegrasyonu + SQLite cache
  scorer.py               ← HuggingFace sentiment + zero-shot
templates/
  sentiment.html          ← Dashboard UI
sentiment_cache.db        ← Otomatik oluşur (6 saat TTL)
```

## Modeller (ilk çalıştırmada indirilir ~1GB)
- cardiffnlp/twitter-roberta-base-sentiment-latest  → sentiment scoring
- facebook/bart-large-mnli                           → event type classification

## Notes
- Free NewsAPI: 100 istek/gün, son 1 ay haber
- Cache: 6 saatte bir yenilenir
- Model yükleme: ilk açılışta ~30-60 sn (arka planda)
