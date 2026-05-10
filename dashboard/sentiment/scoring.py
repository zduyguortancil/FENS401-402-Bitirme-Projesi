"""
Score computation — gerçek DeBERTa text-level sentiment + keyword event weight + recency decay.

Hibrit skorlama:
  combined = 0.65 × deberta_score   (text-level, -1..+1)
           + 0.25 × event_weight     (keyword kategori ağırlığı)
           + 0.10 × tone_norm        (varsa GDELT tone, yoksa 0)

  Bu üçü olmadan: combined = event_weight (eski davranış).
"""
import math
from datetime import datetime

from .classifier import EVENT_META


def _parse_date(date_str):
    """Parse various date formats."""
    if not date_str:
        return None
    s = date_str.strip()
    for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S.%f", "%a, %d %b %Y %H:%M:%S %Z",
                "%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
        try:
            return datetime.strptime(s, fmt)
        except (ValueError, AttributeError):
            continue
    # Try ISO without Z suffix variation
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def score_article(article):
    """
    Single article scoring.

    Priority:
      1) DeBERTa varsa text-level → büyük ağırlık (0.65)
      2) Keyword event_type ağırlığı (0.25)
      3) GDELT tone varsa (0.10)
    """
    deberta_score = article.get("deberta_score")  # -1..+1 or None
    deberta_label = article.get("deberta_label")  # "positive"/"negative"/"neutral"/None
    tone = article.get("tone")
    event_key = article.get("event_type", "general_news")
    meta = EVENT_META.get(event_key, EVENT_META["general_news"])
    event_weight = meta["impact"]

    if deberta_score is not None:
        tone_norm = max(-1.0, min(1.0, (tone or 0) / 100.0)) if tone is not None else 0.0
        combined = 0.65 * deberta_score + 0.25 * event_weight + 0.10 * tone_norm
        # Label: DeBERTa primary
        label = deberta_label or _label_from_score(combined)
    elif tone is not None:
        # Eski davranış: GDELT tone + event
        tone_norm = max(-1.0, min(1.0, tone / 100.0))
        combined = 0.6 * tone_norm + 0.4 * event_weight
        label = _label_from_score(combined)
    else:
        # Fallback: sadece keyword event
        combined = event_weight
        label = _label_from_score(combined)

    combined = max(-1.0, min(1.0, combined))

    article["sentiment_label"] = label
    article["sentiment_score"] = round(combined, 4)
    article["event_tr"] = meta["tr"]
    article["event_icon"] = meta["icon"]
    article["event_impact"] = event_weight
    return article


def _label_from_score(score):
    if score > 0.05:
        return "positive"
    if score < -0.05:
        return "negative"
    return "neutral"


def compute_city_score(scored_articles):
    """
    City-level weighted score.
    Recency decay: e^(-0.05 * hours_old) — yumuşatılmış (24h → %30, 72h → %5).
    """
    if not scored_articles:
        return _empty_aggregate()

    now = datetime.utcnow()
    weighted_sum = 0.0
    weight_total = 0.0
    pos = neg = neu = 0
    event_counts = {}
    high_impact = []

    MAX_AGE_HOURS = 14 * 24  # 14 days
    DECAY_LAMBDA = 0.05      # 24h → ~0.30, 48h → ~0.09, 72h → ~0.027

    for a in scored_articles:
        score = a.get("sentiment_score", 0.0)
        label = a.get("sentiment_label", "neutral")
        event_key = a.get("event_type", "general_news")
        impact = a.get("event_impact", 0)

        pub_dt = _parse_date(a.get("published_at", ""))
        if pub_dt:
            hours_old = max(0.0, (now - pub_dt).total_seconds() / 3600.0)
        else:
            hours_old = 168.0

        if hours_old > MAX_AGE_HOURS:
            continue

        recency = math.exp(-DECAY_LAMBDA * hours_old)

        weighted_sum += score * recency
        weight_total += recency

        if label == "positive":
            pos += 1
        elif label == "negative":
            neg += 1
        else:
            neu += 1

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

    included_count = pos + neg + neu

    # ─── Alert kalibrasyonu (kalite-bazlı, false-positive azaltıldı) ───
    threat_count = event_counts.get("security_threat", 0)
    threat_ratio = threat_count / included_count if included_count else 0.0
    high_impact_neg = sum(1 for e in high_impact if e.get("sentiment_score", 0) < 0)
    neg_ratio = neg / included_count if included_count else 0.0

    if composite < -0.30 or threat_ratio >= 0.20 or (threat_count >= 3 and high_impact_neg >= 3):
        alert = "high"
    elif composite < -0.10 or threat_ratio >= 0.08 or neg_ratio >= 0.55:
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
        "threat_ratio": round(threat_ratio, 3),
    }


def _empty_aggregate():
    return {
        "composite_score": 0.0,
        "alert_level": "low",
        "dominant_event": "general_news",
        "dominant_event_tr": "General News",
        "article_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "neutral_count": 0,
        "high_impact_events": [],
        "event_distribution": {},
        "threat_ratio": 0.0,
    }
