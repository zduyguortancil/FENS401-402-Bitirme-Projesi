"""
News Fetcher — NewsAPI entegrasyonu
5 şehir için haber çeker: İstanbul, Dubai, Londra, Tel Aviv, Beirut
"""
import os
import json
import time
import hashlib
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

def _get_newsapi_key():
    return os.environ.get("NEWSAPI_KEY", "")

NEWSAPI_URL = "https://newsapi.org/v2/everything"

# Cache DB (aynı dizinde)
CACHE_DB = Path(__file__).parent.parent / "sentiment_cache.db"

# ── Şehir konfigürasyonları ──────────────────────────────────
CITIES = {
    "istanbul": {
        "label": "İstanbul",
        "flag": "🇹🇷",
        "queries": [
            "Istanbul flight travel",
            "Turkish Airlines istanbul",
            "Istanbul tourism 2025",
            "Istanbul airport strike OR weather OR disruption",
        ],
        "airline_query": "Turkish Airlines THY",
        "airport_codes": ["IST", "SAW"],
        "country": "Turkey",
        "city_name_en": "Istanbul",
        "color": "#e11d48",
    },
    "dubai": {
        "label": "Dubai",
        "flag": "🇦🇪",
        "queries": [
            "Dubai flight travel",
            "Emirates airline Dubai",
            "Dubai tourism 2025",
            "Dubai airport disruption OR strike OR weather",
        ],
        "airline_query": "Emirates airline Dubai",
        "airport_codes": ["DXB", "DWC"],
        "country": "United Arab Emirates",
        "city_name_en": "Dubai",
        "color": "#f59e0b",
    },
    "london": {
        "label": "Londra",
        "flag": "🇬🇧",
        "queries": [
            "London Heathrow flight travel",
            "London tourism travel 2025",
            "Heathrow airport disruption OR strike OR weather",
            "London flight demand",
        ],
        "airline_query": "British Airways London Heathrow",
        "airport_codes": ["LHR", "LGW", "STN"],
        "country": "United Kingdom",
        "city_name_en": "London",
        "color": "#3b82f6",
    },
    "telaviv": {
        "label": "Tel Aviv",
        "flag": "🇮🇱",
        "queries": [
            "Tel Aviv flight travel",
            "Ben Gurion airport OR TLV disruption OR strike OR weather",
            "Israel tourism Tel Aviv 2025",
            "El Al airline Tel Aviv OR TLV",
        ],
        "airline_query": "El Al Tel Aviv",
        "airport_codes": ["TLV"],
        "country": "Israel",
        "city_name_en": "Tel Aviv",
        "color": "#22c55e",
    },
    "beirut": {
        "label": "Beirut",
        "flag": "🇱🇧",
        "queries": [
            "Beirut flight travel",
            "Beirut airport OR BEY disruption OR strike OR weather",
            "Lebanon tourism Beirut 2025",
            "Middle East Airlines Beirut OR BEY",
        ],
        "airline_query": "Middle East Airlines Beirut",
        "airport_codes": ["BEY"],
        "country": "Lebanon",
        "city_name_en": "Beirut",
        "color": "#ef4444",
    },
}


def init_cache():
    con = sqlite3.connect(CACHE_DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS news_cache (
            cache_key   TEXT PRIMARY KEY,
            data        TEXT NOT NULL,
            fetched_at  TEXT NOT NULL,
            expires_at  TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()


def cache_get(key: str):
    try:
        con = sqlite3.connect(CACHE_DB)
        row = con.execute(
            "SELECT data, expires_at FROM news_cache WHERE cache_key = ?", [key]
        ).fetchone()
        con.close()
        if row:
            expires = datetime.fromisoformat(row[1])
            if datetime.utcnow() < expires:
                return json.loads(row[0])
    except Exception:
        pass
    return None


def cache_set(key: str, data, ttl_hours: int = 6):
    try:
        con = sqlite3.connect(CACHE_DB)
        now = datetime.utcnow()
        expires = now + timedelta(hours=ttl_hours)
        con.execute("""
            INSERT OR REPLACE INTO news_cache (cache_key, data, fetched_at, expires_at)
            VALUES (?, ?, ?, ?)
        """, [key, json.dumps(data, ensure_ascii=False), now.isoformat(), expires.isoformat()])
        con.commit()
        con.close()
    except Exception:
        pass


def _make_key(*parts):
    raw = "_".join(str(p) for p in parts)
    return hashlib.md5(raw.encode()).hexdigest()


def _fetch_newsapi(query: str, page_size: int = 10, days_back: int = 7) -> list[dict]:
    """NewsAPI'den makaleler çeker. Cache'i kontrol eder."""
    if not _get_newsapi_key() or not REQUESTS_OK:
        return []

    cache_key = _make_key("newsapi", query, page_size, days_back)
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": page_size,
        "apiKey": _get_newsapi_key(),
    }

    try:
        resp = requests.get(NEWSAPI_URL, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            articles = data.get("articles", [])
            clean = [
                {
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "published_at": a.get("publishedAt", ""),
                    "url": a.get("url", ""),
                }
                for a in articles
                if a.get("title") and "[Removed]" not in a.get("title", "")
            ]
            cache_set(cache_key, clean, ttl_hours=6)
            return clean
        if resp.status_code == 429:
            print("[NewsAPI] Rate limit hit, returning empty")
            return []
        print(f"[NewsAPI] Error {resp.status_code}: {resp.text[:200]}")
        return []
    except Exception as e:
        print(f"[NewsAPI] Request failed: {e}")
        return []


def fetch_city_news(city_key: str, max_articles: int = 15) -> list[dict]:
    """Bir şehir için tüm sorguları çalıştırır, sonuçları birleştirir ve tekilleştirir."""
    city = CITIES.get(city_key)
    if not city:
        return []

    all_articles = []
    seen_titles = set()

    for query in city["queries"]:
        articles = _fetch_newsapi(query, page_size=5)
        for article in articles:
            title = article["title"].strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                article["query_source"] = query
                article["city"] = city_key
                article["city_label"] = city["label"]
                article["airport_codes"] = city["airport_codes"]
                all_articles.append(article)
        time.sleep(0.1)

    all_articles.sort(key=lambda x: x.get("published_at", ""), reverse=True)
    return all_articles[:max_articles]


def fetch_all_cities() -> dict:
    """Tüm şehirlerin haberlerini toplu çeker."""
    init_cache()
    result = {}
    for city_key in CITIES:
        print(f"[Fetcher] Fetching news for {city_key}...")
        articles = fetch_city_news(city_key)
        result[city_key] = articles
        print(f"[Fetcher] {city_key}: {len(articles)} articles")
    return result


def get_api_status() -> dict:
    """API key ve bağlantı durumunu kontrol eder."""
    if not _get_newsapi_key():
        return {"status": "no_key", "message": "NEWSAPI_KEY environment variable not set"}
    if not REQUESTS_OK:
        return {"status": "no_requests", "message": "requests library not installed"}

    try:
        resp = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={"country": "us", "pageSize": 1, "apiKey": _get_newsapi_key()},
            timeout=5,
        )
        if resp.status_code == 200:
            return {"status": "ok", "message": "API key valid"}
        if resp.status_code == 401:
            return {"status": "invalid_key", "message": "Invalid API key"}
        return {"status": "error", "message": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"status": "connection_error", "message": str(e)}
