"""
Google News RSS fallback — GDELT cokerse devreye girer.
Ucretsiz, API key gerektirmez, dahili tone YOKTUR.
"""
import xml.etree.ElementTree as ET
from urllib.parse import quote

import requests


def fetch_gnews_rss(city_en, max_articles=20):
    """
    Google News RSS'ten sehir haberlerini ceker.
    tone=None doner (DeBERTa/keyword ile skorlanacak).
    """
    query = f"{city_en} airport OR flight OR travel"
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en&gl=US&ceid=US:en"

    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (compatible; FlightSnapshot/1.0)"
        })
        if resp.status_code != 200:
            print(f"[GNewsRSS] HTTP {resp.status_code} for {city_en}")
            return []

        root = ET.fromstring(resp.content)
        articles = []

        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            if not title:
                continue

            source_el = item.find("source")
            source = source_el.text if source_el is not None else ""

            articles.append({
                "title": title,
                "url": item.findtext("link", ""),
                "source": source,
                "tone": None,  # RSS'te tone yok
                "published_at": item.findtext("pubDate", ""),
                "data_source": "gnews_rss",
            })

            if len(articles) >= max_articles:
                break

        return articles

    except Exception as e:
        print(f"[GNewsRSS] Error for {city_en}: {e}")
        return []
