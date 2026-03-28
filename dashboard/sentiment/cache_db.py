"""
SQLite cache katmani — haberler + sehir skorlari.
"""
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "sentiment_v2.db"


def _con():
    return sqlite3.connect(str(DB_PATH), timeout=5)


def init_db():
    con = _con()
    con.executescript("""
        CREATE TABLE IF NOT EXISTS articles (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            city_key    TEXT NOT NULL,
            title       TEXT NOT NULL,
            url         TEXT,
            source      TEXT,
            tone        REAL,
            sentiment_label TEXT,
            sentiment_score REAL,
            event_type  TEXT,
            event_tr    TEXT,
            event_icon  TEXT,
            event_impact REAL,
            published_at TEXT,
            fetched_at  TEXT NOT NULL,
            UNIQUE(city_key, url)
        );
        CREATE TABLE IF NOT EXISTS city_scores (
            city_key            TEXT PRIMARY KEY,
            composite_score     REAL NOT NULL,
            alert_level         TEXT NOT NULL,
            article_count       INTEGER,
            positive_count      INTEGER,
            negative_count      INTEGER,
            neutral_count       INTEGER,
            dominant_event      TEXT,
            dominant_event_tr   TEXT,
            event_distribution  TEXT,
            high_impact_events  TEXT,
            updated_at          TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_art_city ON articles(city_key);
        CREATE INDEX IF NOT EXISTS idx_art_fetched ON articles(fetched_at);
    """)
    con.commit()
    con.close()


def cleanup_old(hours=48):
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    # 14 gunden eski published_at olan makaleleri de temizle
    pub_cutoff_14d = (datetime.utcnow() - timedelta(days=14)).strftime("%Y")  # yil bazli kaba filtre
    con = _con()
    con.execute("DELETE FROM articles WHERE fetched_at < ?", [cutoff])
    # published_at icinde eski yil gecen makaleleri de sil (orn: 2025)
    current_year = datetime.utcnow().year
    for old_year in range(current_year - 1, current_year - 3, -1):
        con.execute("DELETE FROM articles WHERE published_at LIKE ?", [f"%{old_year}%"])
    con.commit()
    con.close()


def store_articles(city_key, articles):
    if not articles:
        return
    con = _con()
    now = datetime.utcnow().isoformat()
    for a in articles:
        try:
            con.execute("""
                INSERT OR REPLACE INTO articles
                (city_key, title, url, source, tone, sentiment_label, sentiment_score,
                 event_type, event_tr, event_icon, event_impact, published_at, fetched_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, [
                city_key,
                a.get("title", ""),
                a.get("url", ""),
                a.get("source", ""),
                a.get("tone"),
                a.get("sentiment_label"),
                a.get("sentiment_score"),
                a.get("event_type"),
                a.get("event_tr"),
                a.get("event_icon"),
                a.get("event_impact"),
                a.get("published_at", ""),
                now,
            ])
        except Exception:
            pass
    con.commit()
    con.close()


def store_city_score(city_key, aggregate):
    con = _con()
    now = datetime.utcnow().isoformat()
    con.execute("""
        INSERT OR REPLACE INTO city_scores
        (city_key, composite_score, alert_level, article_count,
         positive_count, negative_count, neutral_count,
         dominant_event, dominant_event_tr, event_distribution,
         high_impact_events, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        city_key,
        aggregate.get("composite_score", 0.0),
        aggregate.get("alert_level", "low"),
        aggregate.get("article_count", 0),
        aggregate.get("positive_count", 0),
        aggregate.get("negative_count", 0),
        aggregate.get("neutral_count", 0),
        aggregate.get("dominant_event", ""),
        aggregate.get("dominant_event_tr", ""),
        json.dumps(aggregate.get("event_distribution", {})),
        json.dumps(aggregate.get("high_impact_events", [])),
        now,
    ])
    con.commit()
    con.close()


def load_cached_scores(max_age_hours=2):
    """Startup'ta SQLite'tan son skorlari yukle."""
    cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()
    con = _con()
    rows = con.execute(
        "SELECT * FROM city_scores WHERE updated_at > ?", [cutoff]
    ).fetchall()
    con.close()

    if not rows:
        return None

    from .cities import CITIES
    result = {}
    for row in rows:
        city_key = row[0]
        cfg = CITIES.get(city_key, {})
        result[city_key] = {
            "city": city_key,
            "label": cfg.get("label", city_key),
            "flag": cfg.get("flag", ""),
            "color": cfg.get("color", "#58a6ff"),
            "country": cfg.get("country", ""),
            "aggregate": {
                "composite_score": row[1],
                "alert_level": row[2],
                "article_count": row[3],
                "positive_count": row[4],
                "negative_count": row[5],
                "neutral_count": row[6],
                "dominant_event": row[7],
                "dominant_event_tr": row[8],
                "event_distribution": json.loads(row[9] or "{}"),
                "high_impact_events": json.loads(row[10] or "[]"),
            },
            "articles": _load_cached_articles(city_key),
        }
    return result


def _load_cached_articles(city_key, limit=10):
    con = _con()
    rows = con.execute("""
        SELECT title, url, source, tone, sentiment_label, sentiment_score,
               event_type, event_tr, event_icon, event_impact, published_at
        FROM articles WHERE city_key = ?
        ORDER BY published_at DESC LIMIT ?
    """, [city_key, limit]).fetchall()
    con.close()
    return [{
        "title": r[0], "url": r[1], "source": r[2], "tone": r[3],
        "sentiment_label": r[4], "sentiment_score": r[5],
        "event_type": r[6], "event_tr": r[7], "event_icon": r[8],
        "event_impact": r[9], "published_at": r[10],
    } for r in rows]
