"""
Arka plan zamanlayici — her INTERVAL saniyede bir tum sehirleri fetch + score eder.
Google News RSS (birincil) + GDELT (yedek).
DeBERTa-v3 text-level sentiment (lazy-load) + Keyword event classifier.
"""
import time
import threading
import sqlite3
from datetime import datetime, timedelta

from .cities import CITIES
from .gdelt import fetch_gdelt
from .gnews_rss import fetch_gnews_rss
from .classifier import classify_batch as event_classify_batch, EVENT_META
from .scoring import score_article, compute_city_score
from .cache_db import store_articles, store_city_score, cleanup_old, DB_PATH
from . import deberta as _deberta

INTERVAL = 3600  # 1 saat
_running = False


def start_scheduler(cache_ref):
    """App startup'ta cagirilir. Arka plan thread baslatir.
    DeBERTa modelini ayri thread'de warmup eder + mevcut DB'yi backfill eder."""
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

    # DeBERTa warmup + backfill — non-blocking
    def _warmup_and_backfill():
        try:
            _deberta.warmup()
            if _deberta.is_ready():
                _backfill_existing(cache_ref)
        except Exception as e:
            print(f"[DeBERTa] Warmup/backfill failed: {e}", flush=True)

    threading.Thread(target=_warmup_and_backfill, daemon=True, name="deberta-warmup").start()


def stop_scheduler():
    global _running
    _running = False


def _classify_articles_with_deberta(articles):
    """DeBERTa hazırsa tüm article'lara text-level sentiment ekle (in-place).
    Hazır değilse no-op."""
    if not _deberta.is_ready() or not articles:
        return
    try:
        titles = [a.get("title", "") for a in articles]
        results = _deberta.classify_batch(titles)
        for a, r in zip(articles, results):
            if r:
                a["deberta_score"] = r["deberta_score"]
                a["deberta_label"] = r["deberta_label"]
                a["deberta_prob_pos"] = r["deberta_prob_pos"]
    except Exception as e:
        print(f"[DeBERTa] classify_batch error: {e}", flush=True)


def _run_cycle(cache_ref):
    """Bir tam fetch-score dongusu. ~51 sehir, ~60-90sn (DeBERTa varsa +20-40sn)."""
    start = time.time()
    cache_ref["loading"] = True
    print(f"[Scheduler] Cycle starting... (DeBERTa={'ON' if _deberta.is_ready() else 'lazy/off'})",
          flush=True)

    cleanup_old(hours=72)

    result = {}
    rss_ok = 0
    gdelt_fallback = 0
    empty_cities = 0
    total_articles = 0
    deberta_used = 0

    for city_key, cfg in CITIES.items():
        try:
            articles = fetch_gnews_rss(cfg["city_en"], max_articles=20)
            if articles:
                rss_ok += 1
            else:
                articles = fetch_gdelt(
                    city_en=cfg["city_en"], codes=cfg["codes"],
                    country=cfg["country"], max_articles=20,
                )
                if articles:
                    gdelt_fallback += 1

            if not articles:
                empty_cities += 1
                result[city_key] = _empty_city(city_key, cfg)
                continue

            # 1) Event classification (keyword, anlık)
            titles = [a["title"] for a in articles]
            classifications = event_classify_batch(titles)
            for i, (event_key, _conf) in enumerate(classifications):
                articles[i]["event_type"] = event_key
                meta = EVENT_META.get(event_key, EVENT_META["general_news"])
                articles[i]["event_tr"] = meta["tr"]
                articles[i]["event_icon"] = meta["icon"]
                articles[i]["event_impact"] = meta["impact"]

            # 2) DeBERTa text-level sentiment (eğer hazırsa)
            _classify_articles_with_deberta(articles)
            if _deberta.is_ready():
                deberta_used += len(articles)

            # 3) Hibrit score
            scored = [score_article(a) for a in articles]

            # 4) 14+ gün eski olanları filtrele
            cutoff = datetime.utcnow() - timedelta(days=14)
            from .scoring import _parse_date
            recent_scored = []
            for a in scored:
                pub = _parse_date(a.get("published_at", ""))
                if pub and pub < cutoff:
                    continue
                recent_scored.append(a)

            aggregate = compute_city_score(recent_scored)
            store_articles(city_key, recent_scored)
            store_city_score(city_key, aggregate)

            total_articles += len(recent_scored)
            result[city_key] = {
                "city": city_key, "label": cfg["label"], "flag": cfg["flag"],
                "color": cfg["color"], "country": cfg["country"],
                "aggregate": aggregate, "articles": recent_scored[:10],
            }

        except Exception as e:
            print(f"[Scheduler] Error {city_key}: {e}", flush=True)
            result[city_key] = _empty_city(city_key, cfg)

        time.sleep(0.5)

    cache_ref["data"] = result
    cache_ref["last_update"] = datetime.utcnow().isoformat()
    cache_ref["loading"] = False
    cache_ref["deberta_active"] = _deberta.is_ready()

    elapsed = time.time() - start
    print(
        f"[Scheduler] Done in {elapsed:.0f}s — "
        f"{total_articles} articles ({deberta_used} via DeBERTa), "
        f"{rss_ok} RSS, {gdelt_fallback} GDELT fallback, {empty_cities} empty",
        flush=True,
    )


def _backfill_existing(cache_ref):
    """Mevcut DB'deki DeBERTa skoru olmayan article'ları DeBERTa ile re-score et.
    Sonra etkilenen şehirlerin city_scores'unu yeniden hesapla."""
    if not _deberta.is_ready():
        return
    print("[DeBERTa] Backfill starting — existing articles being re-scored...", flush=True)
    t0 = time.time()
    con = sqlite3.connect(str(DB_PATH))
    rows = con.execute(
        "SELECT id, title FROM articles WHERE deberta_score IS NULL ORDER BY id DESC LIMIT 5000"
    ).fetchall()
    if not rows:
        con.close()
        print("[DeBERTa] Backfill: nothing to do.", flush=True)
        return

    print(f"[DeBERTa] Backfill: scoring {len(rows)} articles...", flush=True)
    BATCH = 32
    updated = 0
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        titles = [r[1] or "" for r in chunk]
        try:
            results = _deberta.classify_batch(titles)
        except Exception as e:
            print(f"[DeBERTa] Backfill batch err: {e}", flush=True)
            continue
        for (rid, _t), r in zip(chunk, results):
            if not r:
                continue
            try:
                con.execute(
                    "UPDATE articles SET deberta_score=?, deberta_label=?, deberta_prob_pos=? WHERE id=?",
                    [r["deberta_score"], r["deberta_label"], r["deberta_prob_pos"], rid]
                )
                updated += 1
            except Exception:
                pass
        con.commit()
    con.close()

    # Re-aggregate every affected city's score
    _recompute_all_city_scores(cache_ref)
    print(f"[DeBERTa] Backfill done — {updated} articles updated in {time.time()-t0:.0f}s", flush=True)


def _recompute_all_city_scores(cache_ref):
    """DB'den tüm şehirlerin article'larını oku, hibrit score ile yeniden aggregate et,
    city_scores'u güncelle, runtime cache'i tazele."""
    con = sqlite3.connect(str(DB_PATH))
    rows = con.execute(
        """SELECT city_key, title, url, source, tone, sentiment_label, sentiment_score,
                  event_type, event_tr, event_icon, event_impact, published_at,
                  deberta_score, deberta_label, deberta_prob_pos
           FROM articles"""
    ).fetchall()
    con.close()

    by_city = {}
    for r in rows:
        a = {
            "title": r[1], "url": r[2], "source": r[3], "tone": r[4],
            "sentiment_label": r[5], "sentiment_score": r[6],
            "event_type": r[7], "event_tr": r[8], "event_icon": r[9],
            "event_impact": r[10], "published_at": r[11],
            "deberta_score": r[12], "deberta_label": r[13], "deberta_prob_pos": r[14],
        }
        # Re-score hibrit ile (DB'deki sentiment_score eski formül; üstüne yaz)
        a = score_article(a)
        by_city.setdefault(r[0], []).append(a)

    new_data = {}
    for city_key, articles in by_city.items():
        cfg = CITIES.get(city_key, {})
        aggregate = compute_city_score(articles)
        store_city_score(city_key, aggregate)
        # In-DB article sentiment_score güncelle (toplu)
        # Note: re-score yapan article'ların score'larını DB'ye yazmayı tek geçişte yapalım
        new_data[city_key] = {
            "city": city_key,
            "label": cfg.get("label", city_key),
            "flag": cfg.get("flag", ""),
            "color": cfg.get("color", "#58a6ff"),
            "country": cfg.get("country", ""),
            "aggregate": aggregate,
            "articles": articles[:10],
        }
    # Update articles' sentiment_score column (yansıt)
    _persist_article_scores(by_city)

    if new_data:
        cache_ref["data"] = new_data
        cache_ref["last_update"] = datetime.utcnow().isoformat()


def _persist_article_scores(by_city):
    con = sqlite3.connect(str(DB_PATH))
    for city_key, articles in by_city.items():
        for a in articles:
            try:
                con.execute(
                    """UPDATE articles SET sentiment_score=?, sentiment_label=?
                       WHERE city_key=? AND url=?""",
                    [a.get("sentiment_score"), a.get("sentiment_label"),
                     city_key, a.get("url", "")]
                )
            except Exception:
                pass
    con.commit()
    con.close()


def _empty_city(city_key, cfg):
    return {
        "city": city_key, "label": cfg["label"], "flag": cfg["flag"],
        "color": cfg["color"], "country": cfg["country"],
        "aggregate": {
            "composite_score": 0.0, "alert_level": "low",
            "article_count": 0, "positive_count": 0,
            "negative_count": 0, "neutral_count": 0,
            "dominant_event": "general_news", "dominant_event_tr": "No Data",
            "high_impact_events": [], "event_distribution": {}, "threat_ratio": 0.0,
        },
        "articles": [],
    }
