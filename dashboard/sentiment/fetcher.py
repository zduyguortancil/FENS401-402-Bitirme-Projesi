"""
News Fetcher -- NewsAPI entegrasyonu
51 destinasyon sehri icin haber ceker ve sentiment analizine gonderir.
Her sehir icin tek optimized sorgu -- API limitini asmadan maksimum kapsam.
Scorer (RoBERTa + keyword) zaten haberleri dogru kategorize ediyor.
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

# Cache DB (ayni dizinde)
CACHE_DB = Path(__file__).parent.parent / "sentiment_cache.db"

# ── Renk paleti ───────────────────────────────────────────────
_PALETTE = [
    "#e11d48", "#f59e0b", "#3b82f6", "#22c55e", "#ef4444",
    "#8b5cf6", "#06b6d4", "#ec4899", "#14b8a6", "#f97316",
    "#6366f1", "#84cc16", "#d946ef", "#0ea5e9", "#a855f7",
    "#10b981", "#e879f9", "#facc15", "#38bdf8", "#fb923c",
    "#4ade80", "#c084fc", "#fbbf24", "#2dd4bf", "#f472b6",
    "#818cf8", "#34d399", "#fb7185", "#a3e635", "#67e8f9",
    "#c4b5fd", "#fca5a5", "#86efac", "#fcd34d", "#7dd3fc",
    "#d8b4fe", "#6ee7b7", "#fda4af", "#bef264", "#a5f3fc",
    "#e9d5ff", "#fecaca", "#bbf7d0", "#fde68a", "#bae6fd",
    "#f0abfc", "#a7f3d0", "#fecdd3", "#d9f99d", "#99f6e4",
    "#c7d2fe",
]

# ── Tum sehirler ─────────────────────────────────────────────
# (key, label_tr, city_en, flag, country, airport_codes, airline)
_CITY_DEFS = [
    ("istanbul",      "Istanbul",       "Istanbul",       "\U0001f1f9\U0001f1f7", "Turkey",              ["IST","SAW"], "Turkish Airlines"),
    ("london",        "Londra",         "London",         "\U0001f1ec\U0001f1e7", "United Kingdom",      ["LHR","LGW","STN"], "British Airways"),
    ("paris",         "Paris",          "Paris",          "\U0001f1eb\U0001f1f7", "France",              ["CDG","ORY"], "Air France"),
    ("dubai",         "Dubai",          "Dubai",          "\U0001f1e6\U0001f1ea", "United Arab Emirates", ["DXB","DWC"], "Emirates"),
    ("newyork",       "New York",       "New York",       "\U0001f1fa\U0001f1f8", "United States",       ["JFK"],  "Delta JFK"),
    ("frankfurt",     "Frankfurt",      "Frankfurt",      "\U0001f1e9\U0001f1ea", "Germany",             ["FRA"],  "Lufthansa"),
    ("barcelona",     "Barselona",      "Barcelona",      "\U0001f1ea\U0001f1f8", "Spain",               ["BCN"],  "Vueling"),
    ("rome",          "Roma",           "Rome",           "\U0001f1ee\U0001f1f9", "Italy",               ["FCO"],  "ITA Airways"),
    ("milan",         "Milano",         "Milan",          "\U0001f1ee\U0001f1f9", "Italy",               ["MXP"],  "ITA Airways"),
    ("madrid",        "Madrid",         "Madrid",         "\U0001f1ea\U0001f1f8", "Spain",               ["MAD"],  "Iberia"),
    ("munich",        "Munih",          "Munich",         "\U0001f1e9\U0001f1ea", "Germany",             ["MUC"],  "Lufthansa"),
    ("nice",          "Nice",           "Nice",           "\U0001f1eb\U0001f1f7", "France",              ["NCE"],  "Air France"),
    ("manchester",    "Manchester",     "Manchester",     "\U0001f1ec\U0001f1e7", "United Kingdom",      ["MAN"],  "Manchester airport"),
    ("telaviv",       "Tel Aviv",       "Tel Aviv",       "\U0001f1ee\U0001f1f1", "Israel",              ["TLV"],  "El Al"),
    ("beirut",        "Beyrut",         "Beirut",         "\U0001f1f1\U0001f1e7", "Lebanon",             ["BEY"],  "Middle East Airlines"),
    ("amman",         "Amman",          "Amman",          "\U0001f1ef\U0001f1f4", "Jordan",              ["AMM"],  "Royal Jordanian"),
    ("jeddah",        "Cidde",          "Jeddah",         "\U0001f1f8\U0001f1e6", "Saudi Arabia",        ["JED"],  "Saudia"),
    ("riyadh",        "Riyad",          "Riyadh",         "\U0001f1f8\U0001f1e6", "Saudi Arabia",        ["RUH"],  "Saudia"),
    ("doha",          "Doha",           "Doha",           "\U0001f1f6\U0001f1e6", "Qatar",               ["DOH"],  "Qatar Airways"),
    ("bahrain",       "Bahreyn",        "Bahrain",        "\U0001f1e7\U0001f1ed", "Bahrain",             ["BAH"],  "Gulf Air"),
    ("kuwait",        "Kuveyt",         "Kuwait",         "\U0001f1f0\U0001f1fc", "Kuwait",              ["KWI"],  "Kuwait Airways"),
    ("abudhabi",      "Abu Dhabi",      "Abu Dhabi",      "\U0001f1e6\U0001f1ea", "United Arab Emirates", ["AUH"], "Etihad"),
    ("cairo",         "Kahire",         "Cairo",          "\U0001f1ea\U0001f1ec", "Egypt",               ["CAI"],  "EgyptAir"),
    ("hurghada",      "Hurghada",       "Hurghada",       "\U0001f1ea\U0001f1ec", "Egypt",               ["HRG"],  "EgyptAir"),
    ("casablanca",    "Kazablanka",     "Casablanca",     "\U0001f1f2\U0001f1e6", "Morocco",             ["CMN"],  "Royal Air Maroc"),
    ("marrakech",     "Marakes",        "Marrakech",      "\U0001f1f2\U0001f1e6", "Morocco",             ["RAK"],  "Royal Air Maroc"),
    ("nairobi",       "Nairobi",        "Nairobi",        "\U0001f1f0\U0001f1ea", "Kenya",               ["NBO"],  "Kenya Airways"),
    ("mombasa",       "Mombasa",        "Mombasa",        "\U0001f1f0\U0001f1ea", "Kenya",               ["MBA"],  "Kenya Airways"),
    ("lagos",         "Lagos",          "Lagos",          "\U0001f1f3\U0001f1ec", "Nigeria",             ["LOS"],  "Air Peace"),
    ("abuja",         "Abuja",          "Abuja",          "\U0001f1f3\U0001f1ec", "Nigeria",             ["ABV"],  "Air Peace"),
    ("johannesburg",  "Johannesburg",   "Johannesburg",   "\U0001f1ff\U0001f1e6", "South Africa",        ["JNB"],  "South African Airways"),
    ("capetown",      "Cape Town",      "Cape Town",      "\U0001f1ff\U0001f1e6", "South Africa",        ["CPT"],  "South African Airways"),
    ("tokyo",         "Tokyo",          "Tokyo",          "\U0001f1ef\U0001f1f5", "Japan",               ["NRT","HND"], "ANA"),
    ("osaka",         "Osaka",          "Osaka",          "\U0001f1ef\U0001f1f5", "Japan",               ["KIX"],  "ANA"),
    ("seoul",         "Seul",           "Seoul",          "\U0001f1f0\U0001f1f7", "South Korea",         ["ICN"],  "Korean Air"),
    ("beijing",       "Pekin",          "Beijing",        "\U0001f1e8\U0001f1f3", "China",               ["PEK"],  "Air China"),
    ("shanghai",      "Sanghay",        "Shanghai",       "\U0001f1e8\U0001f1f3", "China",               ["PVG"],  "China Eastern"),
    ("singapore",     "Singapur",       "Singapore",      "\U0001f1f8\U0001f1ec", "Singapore",           ["SIN"],  "Singapore Airlines"),
    ("bangkok",       "Bangkok",        "Bangkok",        "\U0001f1f9\U0001f1ed", "Thailand",            ["BKK"],  "Thai Airways"),
    ("phuket",        "Phuket",         "Phuket",         "\U0001f1f9\U0001f1ed", "Thailand",            ["HKT"],  "Thai Airways"),
    ("delhi",         "Delhi",          "Delhi",          "\U0001f1ee\U0001f1f3", "India",               ["DEL"],  "Air India"),
    ("mumbai",        "Mumbai",         "Mumbai",         "\U0001f1ee\U0001f1f3", "India",               ["BOM"],  "Air India"),
    ("losangeles",    "Los Angeles",    "Los Angeles",    "\U0001f1fa\U0001f1f8", "United States",       ["LAX"],  "United Airlines"),
    ("chicago",       "Chicago",        "Chicago",        "\U0001f1fa\U0001f1f8", "United States",       ["ORD"],  "United Airlines"),
    ("miami",         "Miami",          "Miami",          "\U0001f1fa\U0001f1f8", "United States",       ["MIA"],  "American Airlines"),
    ("toronto",       "Toronto",        "Toronto",        "\U0001f1e8\U0001f1e6", "Canada",              ["YYZ"],  "Air Canada"),
    ("vancouver",     "Vancouver",      "Vancouver",      "\U0001f1e8\U0001f1e6", "Canada",              ["YVR"],  "Air Canada"),
    ("mexicocity",    "Mexico City",    "Mexico City",    "\U0001f1f2\U0001f1fd", "Mexico",              ["MEX"],  "Aeromexico"),
    ("saopaulo",      "Sao Paulo",      "Sao Paulo",      "\U0001f1e7\U0001f1f7", "Brazil",              ["GRU"],  "LATAM"),
    ("buenosaires",   "Buenos Aires",   "Buenos Aires",   "\U0001f1e6\U0001f1f7", "Argentina",           ["EZE"],  "Aerolineas Argentinas"),
    ("riodejaneiro",  "Rio de Janeiro", "Rio de Janeiro", "\U0001f1e7\U0001f1f7", "Brazil",              ["GIG"],  "LATAM"),
]


def _build_query(city_en, codes):
    """Tek bir optimized sorgu uretir.
    Scorer keyword'leri (cancel, delay, strike, missile, tourism, vb.)
    zaten haberleri dogru kategorize ediyor.
    Bu sorgu sadece o sehirle ilgili haberleri getirmeli — genis kapsam.
    """
    # Parantezlerle operator precedence garantisi
    code_part = " OR ".join(f'"{c}"' for c in codes)
    return f'("{city_en}" OR {code_part}) AND (flight OR airport OR travel OR tourism OR airline OR security OR crisis)'


def _build_cities():
    cities = {}
    for i, (key, label_tr, city_en, flag, country, codes, airline) in enumerate(_CITY_DEFS):
        cities[key] = {
            "label": label_tr,
            "flag": flag,
            "queries": [_build_query(city_en, codes)],
            "airline_query": airline,
            "airport_codes": codes,
            "country": country,
            "city_name_en": city_en,
            "color": _PALETTE[i % len(_PALETTE)],
        }
    return cities


CITIES = _build_cities()

# Havaalani kodu -> sehir key'i lookup tablosu
AIRPORT_TO_CITY = {}
for _key, _cfg in CITIES.items():
    for _code in _cfg["airport_codes"]:
        AIRPORT_TO_CITY[_code] = _key


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


def cache_set(key: str, data, ttl_hours: int = 12):
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


def _fetch_newsapi(query: str, page_size: int = 50, days_back: int = 7) -> list[dict]:
    """NewsAPI'den makaleler ceker. Cache'i kontrol eder."""
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
        "sortBy": "relevancy",
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


def fetch_city_news(city_key: str, max_articles: int = 30) -> list[dict]:
    """Bir sehir icin sorguyu calistirir, sonuclari filtreler."""
    city = CITIES.get(city_key)
    if not city:
        return []

    all_articles = []
    seen_titles = set()

    # Relevance keywords — genis: sehir, ulke, havaalani, havayolu
    relevance_terms = set()
    relevance_terms.add(city["city_name_en"].lower())
    relevance_terms.add(city["country"].lower())
    relevance_terms.update(c.lower() for c in city["airport_codes"])
    if city.get("airline_query"):
        relevance_terms.update(w.lower() for w in city["airline_query"].split() if len(w) > 2)
    # Genel havacilik terimleri de kabul et (sorgu zaten sehir bazli)
    travel_terms = {"flight", "airport", "airline", "travel", "tourism", "passenger", "aviation"}
    relevance_terms.update(travel_terms)

    for query in city["queries"]:
        articles = _fetch_newsapi(query, page_size=50)
        for article in articles:
            title = article["title"].strip()
            if title and title not in seen_titles:
                text_lower = (title + " " + (article.get("description") or "")).lower()
                if not any(term in text_lower for term in relevance_terms):
                    continue
                seen_titles.add(title)
                article["query_source"] = query
                article["city"] = city_key
                article["city_label"] = city["label"]
                article["airport_codes"] = city["airport_codes"]
                all_articles.append(article)
        time.sleep(0.05)

    all_articles.sort(key=lambda x: x.get("published_at", ""), reverse=True)
    return all_articles[:max_articles]


def fetch_all_cities() -> dict:
    """Tum sehirlerin haberlerini toplu ceker."""
    init_cache()
    result = {}
    for city_key in CITIES:
        articles = fetch_city_news(city_key)
        result[city_key] = articles
    return result


def get_api_status() -> dict:
    """API key ve baglanti durumunu kontrol eder."""
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
