"""
GDELT DOC API client — birincil haber kaynagi.
Ucretsiz, API key gerektirmez, dahili tone skoru var.
"""
import requests
from datetime import datetime

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_gdelt(city_en, codes, country="", max_articles=30, timespan="24h"):
    """
    GDELT DOC API'den sehir haberlerini ceker.
    Dahili tone skoru (-100 / +100) ile birlikte doner.
    """
    # Sorgu: sehir adi + havacilik terimleri (basit tutmak GDELT icin onemli)
    query = f'"{city_en}" (airport OR flight OR travel OR tourism OR airline)'

    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": max_articles,
        "timespan": timespan,
        "format": "json",
        "sort": "DateDesc",
    }

    try:
        resp = requests.get(GDELT_URL, params=params, timeout=15)
        if resp.status_code != 200:
            print(f"[GDELT] HTTP {resp.status_code} for {city_en}")
            return []

        # GDELT bazen HTML dondurur (rate limit sayfasi)
        ct = resp.headers.get("content-type", "")
        if "json" not in ct:
            print(f"[GDELT] Non-JSON response for {city_en}: {ct}")
            return []

        try:
            data = resp.json()
        except Exception:
            print(f"[GDELT] JSON parse error for {city_en}")
            return []
        raw_articles = data.get("articles", [])
        if not raw_articles:
            return []

        articles = []
        city_lower = city_en.lower()
        country_lower = country.lower() if country else ""
        codes_lower = {c.lower() for c in codes}

        for art in raw_articles:
            title = (art.get("title") or "").strip()
            if not title:
                continue

            # Basit relevance filtre
            check_text = (title + " " + (art.get("url") or "")).lower()
            if not any(t in check_text for t in [city_lower, country_lower] + list(codes_lower)):
                # Havacilik terimleri de kabul et (sorgu zaten sehir bazli)
                if not any(t in check_text for t in ["flight", "airport", "airline", "travel", "tourism"]):
                    continue

            # GDELT tarih formati: YYYYMMDDTHHMMSSZ
            raw_date = art.get("seendate", "")
            try:
                pub_dt = datetime.strptime(raw_date, "%Y%m%dT%H%M%SZ")
                pub_iso = pub_dt.isoformat() + "Z"
            except Exception:
                pub_iso = raw_date

            articles.append({
                "title": title,
                "url": art.get("url", ""),
                "source": art.get("domain", ""),
                "tone": None,  # ArtList modunda tone yok, classifier skorlayacak
                "language": art.get("language", ""),
                "published_at": pub_iso,
                "data_source": "gdelt",
            })

        return articles[:max_articles]

    except requests.exceptions.Timeout:
        print(f"[GDELT] Timeout for {city_en}")
        return []
    except Exception as e:
        print(f"[GDELT] Error for {city_en}: {e}")
        return []
