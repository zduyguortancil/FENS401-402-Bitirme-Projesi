"""
Sentiment Analysis Dashboard — Standalone Flask App
Port: 5002 (ana uygulamadan bağımsız)

Çalıştırmak için:
    export NEWSAPI_KEY="your_key_here"
    python sentiment_app.py

Gereksinimler:
    pip install flask transformers torch requests
    (torch yerine: pip install transformers[cpu] requests flask)
"""

import os
import sys
import json
import threading
from pathlib import Path
from flask import Flask, jsonify, render_template_string, request

# sentiment modülü aynı dizinde
sys.path.insert(0, str(Path(__file__).parent))
from sentiment.fetcher import fetch_all_cities, fetch_city_news, get_api_status, CITIES, init_cache
from sentiment.scorer import score_articles, aggregate_city_sentiment, load_models, MODELS_LOADED, LOAD_ERROR

app = Flask(__name__)

# Uygulama başladığında modelleri arka planda yükle
def _preload_models():
    print("[App] Pre-loading HuggingFace models in background...", flush=True)
    load_models()

threading.Thread(target=_preload_models, daemon=True).start()
init_cache()


# ── HTML Template ────────────────────────────────────────────
@app.route("/")
def index():
    html = open(Path(__file__).parent / "templates" / "sentiment.html", encoding="utf-8").read()
    return render_template_string(html)


# ── API: Durum ───────────────────────────────────────────────
@app.route("/api/sentiment/status")
def api_status():
    return jsonify({
        "models_loaded": MODELS_LOADED,
        "load_error": LOAD_ERROR,
        "api_status": get_api_status(),
        "cities": list(CITIES.keys()),
    })


# ── API: Tüm şehirler ────────────────────────────────────────
@app.route("/api/sentiment/all")
def api_sentiment_all():
    """3 şehrin sentiment özetini döner."""
    force_refresh = request.args.get("refresh", "").lower() == "true"
    result = {}

    for city_key, city_cfg in CITIES.items():
        articles = fetch_city_news(city_key, max_articles=15)
        if not articles:
            result[city_key] = {
                "city": city_key,
                "label": city_cfg["label"],
                "flag": city_cfg["flag"],
                "color": city_cfg["color"],
                "country": city_cfg.get("country", ""),
                "aggregate": {
                    "composite_score": 0.0,
                    "alert_level": "low",
                    "article_count": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": 0,
                    "dominant_event_tr": "Veri Yok",
                    "high_impact_events": [],
                    "event_distribution": {},
                },
                "articles": [],
                "error": "no_articles",
            }
            continue

        scored = score_articles(articles)
        aggregate = aggregate_city_sentiment(scored)

        result[city_key] = {
            "city": city_key,
            "label": city_cfg["label"],
            "flag": city_cfg["flag"],
            "color": city_cfg["color"],
            "country": city_cfg.get("country", ""),
            "aggregate": aggregate,
            "articles": scored[:10],
        }

    return jsonify(result)


# ── API: Tek şehir ───────────────────────────────────────────
@app.route("/api/sentiment/<city_key>")
def api_sentiment_city(city_key):
    if city_key not in CITIES:
        return jsonify({"error": f"Unknown city: {city_key}"}), 404

    city_cfg = CITIES[city_key]
    articles = fetch_city_news(city_key, max_articles=20)

    if not articles:
        return jsonify({
            "city": city_key,
            "label": city_cfg["label"],
            "aggregate": {"composite_score": 0, "alert_level": "low", "article_count": 0},
            "articles": [],
            "error": "no_articles_or_no_api_key",
        })

    scored = score_articles(articles)
    aggregate = aggregate_city_sentiment(scored)

    return jsonify({
        "city": city_key,
        "label": city_cfg["label"],
        "flag": city_cfg["flag"],
        "color": city_cfg["color"],
        "aggregate": aggregate,
        "articles": scored,
    })


if __name__ == "__main__":
    key_status = "✓ SET" if os.environ.get("NEWSAPI_KEY") else "✗ NOT SET (set NEWSAPI_KEY env var)"
    print(f"\n{'='*55}")
    print("  Sentiment Analysis Dashboard")
    print(f"{'='*55}")
    print(f"  URL:        http://localhost:5002")
    print(f"  NEWSAPI_KEY: {key_status}")
    print(f"  Models:     Loading in background...")
    print(f"{'='*55}\n")
    app.run(debug=True, port=5002, use_reloader=False)
