"""
Arka plan zamanlayici — her INTERVAL saniyede bir tum sehirleri fetch + score eder.
Google News RSS (birincil) + GDELT (yedek).
Keyword classifier (ML model yok, aninda).
"""
import time
import threading
from datetime import datetime

from .cities import CITIES
from .gdelt import fetch_gdelt
from .gnews_rss import fetch_gnews_rss
from .classifier import classify_batch, EVENT_META
from .scoring import score_article, compute_city_score
from .cache_db import store_articles, store_city_score, cleanup_old

INTERVAL = 3600  # 1 saat
_running = False


def start_scheduler(cache_ref):
    """App startup'ta cagirilir. Arka plan thread baslatir."""
    def _loop():
        global _running
        _running = True
        while _running:
            try:
                _run_cycle(cache_ref)
            except Exception as e:
                print(f"[Scheduler] Cycle failed: {e}", flush=True)
                import traceback; traceback.print_exc()
            for _ in range(INTERVAL):
                if not _running:
                    break
                time.sleep(1)

    t = threading.Thread(target=_loop, daemon=True, name="sentiment-scheduler")
    t.start()
    print(f"[Scheduler] Started (interval={INTERVAL}s, cities={len(CITIES)})", flush=True)


def stop_scheduler():
    global _running
    _running = False


def _run_cycle(cache_ref):
    """Bir tam fetch-score dongusu. ~51 sehir, ~60sn."""
    start = time.time()
    cache_ref["loading"] = True
    print(f"[Scheduler] Cycle starting...", flush=True)

    cleanup_old(hours=48)

    result = {}
    rss_ok = 0
    gdelt_fallback = 0
    empty_cities = 0
    total_articles = 0

    for city_key, cfg in CITIES.items():
        try:
            # 1) Google News RSS (birincil — rate limit yok)
            articles = fetch_gnews_rss(cfg["city_en"], max_articles=20)

            if articles:
                rss_ok += 1
            else:
                # 2) GDELT fallback
                articles = fetch_gdelt(
                    city_en=cfg["city_en"],
                    codes=cfg["codes"],
                    country=cfg["country"],
                    max_articles=20,
                )
                if articles:
                    gdelt_fallback += 1

            if not articles:
                empty_cities += 1
                result[city_key] = _empty_city(city_key, cfg)
                continue

            # 3) Keyword classification (aninda, ML yok)
            titles = [a["title"] for a in articles]
            classifications = classify_batch(titles)

            for i, (event_key, confidence) in enumerate(classifications):
                articles[i]["event_type"] = event_key
                meta = EVENT_META.get(event_key, EVENT_META["general_news"])
                articles[i]["event_tr"] = meta["tr"]
                articles[i]["event_icon"] = meta["icon"]
                articles[i]["event_impact"] = meta["impact"]

            # 4) Score
            scored = [score_article(a) for a in articles]

            # 5) Eski makaleleri filtrele (14 gun)
            from datetime import datetime, timedelta
            cutoff = datetime.utcnow() - timedelta(days=14)
            from .scoring import _parse_date
            recent_scored = []
            for a in scored:
                pub = _parse_date(a.get("published_at", ""))
                if pub and pub < cutoff:
                    continue  # 14+ gun eski, atla
                recent_scored.append(a)

            # 6) Aggregate (sadece recent makalelerle)
            aggregate = compute_city_score(recent_scored)

            # 7) Persist
            store_articles(city_key, recent_scored)
            store_city_score(city_key, aggregate)

            total_articles += len(recent_scored)
            result[city_key] = {
                "city": city_key,
                "label": cfg["label"],
                "flag": cfg["flag"],
                "color": cfg["color"],
                "country": cfg["country"],
                "aggregate": aggregate,
                "articles": recent_scored[:10],
            }

        except Exception as e:
            print(f"[Scheduler] Error {city_key}: {e}", flush=True)
            result[city_key] = _empty_city(city_key, cfg)

        # RSS icin kisa bekleme (Google'i kizdirmamak icin)
        time.sleep(0.5)

    # Tum sehirler tamamlandi — cache guncelle
    cache_ref["data"] = result
    cache_ref["last_update"] = datetime.utcnow().isoformat()
    cache_ref["loading"] = False

    elapsed = time.time() - start
    print(
        f"[Scheduler] Done in {elapsed:.0f}s — "
        f"{total_articles} articles, {rss_ok} RSS, "
        f"{gdelt_fallback} GDELT fallback, {empty_cities} empty",
        flush=True,
    )


def _empty_city(city_key, cfg):
    return {
        "city": city_key,
        "label": cfg["label"],
        "flag": cfg["flag"],
        "color": cfg["color"],
        "country": cfg["country"],
        "aggregate": {
            "composite_score": 0.0, "alert_level": "low",
            "article_count": 0, "positive_count": 0,
            "negative_count": 0, "neutral_count": 0,
            "dominant_event": "general_news",
            "dominant_event_tr": "Veri Yok",
            "high_impact_events": [], "event_distribution": {},
        },
        "articles": [],
    }
