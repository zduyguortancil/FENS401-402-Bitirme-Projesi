"""
Skor hesaplama — GDELT tone + event weight + recency decay.
"""
import math
from datetime import datetime

from .classifier import EVENT_META


def _parse_date(date_str):
    """Cesitli tarih formatlarini parse et."""
    if not date_str:
        return None
    for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ",
                "%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"]:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def score_article(article):
    """
    Tek makale skorlama.
    GDELT tone varsa: tone * 0.6 + event_weight * 0.4
    RSS (tone yok): sadece event_weight
    """
    tone = article.get("tone")
    event_key = article.get("event_type", "general_news")
    meta = EVENT_META.get(event_key, EVENT_META["general_news"])
    event_weight = meta["impact"]

    # Tone normalization: GDELT -100/+100 → -1/+1
    if tone is not None:
        tone_norm = max(-1.0, min(1.0, tone / 100.0))
        combined = 0.6 * tone_norm + 0.4 * event_weight
    else:
        # RSS fallback: sadece event classification
        combined = event_weight
        tone_norm = 0.0

    # Sentiment label
    if combined > 0.05:
        label = "positive"
    elif combined < -0.05:
        label = "negative"
    else:
        label = "neutral"

    article["sentiment_label"] = label
    article["sentiment_score"] = round(combined, 4)
    article["event_tr"] = meta["tr"]
    article["event_icon"] = meta["icon"]
    article["event_impact"] = event_weight
    return article


def compute_city_score(scored_articles):
    """
    Sehir bazinda agirlikli skor.
    Recency decay: e^(-0.1 * hours_old) — yeni haberler agir basar.
    """
    if not scored_articles:
        return _empty_aggregate()

    now = datetime.utcnow()
    weighted_sum = 0.0
    weight_total = 0.0
    pos = neg = neu = 0
    event_counts = {}
    high_impact = []

    MAX_AGE_HOURS = 14 * 24  # 14 gun — daha eski makaleler score'a dahil edilmez

    for a in scored_articles:
        score = a.get("sentiment_score", 0.0)
        label = a.get("sentiment_label", "neutral")
        event_key = a.get("event_type", "general_news")
        impact = a.get("event_impact", 0)

        # Recency decay
        pub_dt = _parse_date(a.get("published_at", ""))
        if pub_dt:
            hours_old = max(0, (now - pub_dt).total_seconds() / 3600)
        else:
            hours_old = 168  # bilinmiyorsa 1 hafta varsay (eski: 12 saat — cok iyimser)

        # 14 gunden eski makaleleri atla
        if hours_old > MAX_AGE_HOURS:
            continue

        recency = math.exp(-0.1 * hours_old)

        weighted_sum += score * recency
        weight_total += recency

        if label == "positive": pos += 1
        elif label == "negative": neg += 1
        else: neu += 1

        event_counts[event_key] = event_counts.get(event_key, 0) + 1

        if abs(impact) >= 0.5:
            high_impact.append({
                "title": a.get("title", ""),
                "event_type": event_key,
                "event_tr": a.get("event_tr", ""),
                "event_icon": a.get("event_icon", ""),
                "sentiment_label": label,
                "sentiment_score": score,
                "published_at": a.get("published_at", ""),
                "source": a.get("source", ""),
            })

    composite = round(weighted_sum / weight_total, 4) if weight_total > 0 else 0.0
    composite = max(-1.0, min(1.0, composite))

    dominant = max(event_counts, key=event_counts.get) if event_counts else "general_news"
    dominant_meta = EVENT_META.get(dominant, EVENT_META["general_news"])

    # Article count = sadece filtreyi gecenler
    included_count = pos + neg + neu

    # Alert level (sadece dahil edilen makalelerden)
    has_threat = any(k == "security_threat" for k in event_counts)
    if composite < -0.3 or has_threat:
        alert = "high"
    elif composite < -0.1:
        alert = "medium"
    else:
        alert = "low"

    return {
        "composite_score": composite,
        "alert_level": alert,
        "dominant_event": dominant,
        "dominant_event_tr": dominant_meta["tr"],
        "article_count": included_count,
        "positive_count": pos,
        "negative_count": neg,
        "neutral_count": neu,
        "high_impact_events": sorted(high_impact, key=lambda x: abs(x.get("sentiment_score", 0)), reverse=True)[:5],
        "event_distribution": event_counts,
    }


def _empty_aggregate():
    return {
        "composite_score": 0.0,
        "alert_level": "low",
        "dominant_event": "general_news",
        "dominant_event_tr": "Genel Haber",
        "article_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "neutral_count": 0,
        "high_impact_events": [],
        "event_distribution": {},
    }
