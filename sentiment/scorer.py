"""
Sentiment Scorer — HuggingFace Transformers
Model: cardiffnlp/twitter-roberta-base-sentiment-latest
  → negative / neutral / positive + confidence score

Event Classifier: zero-shot ile haber türünü tespit eder
"""
import re
from typing import Optional

# ── Model yükleme (lazy, ilk kullanımda) ─────────────────────
_sentiment_pipe = None
_zero_shot_pipe = None
MODELS_LOADED = False
LOAD_ERROR = None

EVENT_LABELS = [
    "flight disruption",
    "strike or labor dispute",
    "weather disruption",
    "security threat",
    "tourism growth",
    "new route or airline expansion",
    "airport congestion",
    "general travel news",
]

# Event label → Türkçe etiket + etki yönü
EVENT_META = {
    "flight disruption":        {"tr": "Uçuş Aksaklığı",      "impact": -1,   "icon": "✈️⚠️"},
    "strike or labor dispute":  {"tr": "Grev / İş Uyuşmazlığı","impact": -1,  "icon": "✊"},
    "weather disruption":       {"tr": "Hava Krizi",            "impact": -1,  "icon": "🌩️"},
    "security threat":          {"tr": "Güvenlik Tehdidi",      "impact": -1,  "icon": "🚨"},
    "tourism growth":           {"tr": "Turizm Büyümesi",       "impact": +1,  "icon": "📈"},
    "new route or airline expansion": {"tr": "Yeni Rota/Genişleme","impact": +1,"icon": "🚀"},
    "airport congestion":       {"tr": "Havalimanı Kalabalığı", "impact": -0.5,"icon": "🔴"},
    "general travel news":      {"tr": "Genel Seyahat Haberi",  "impact": 0,   "icon": "📰"},
}


def load_models():
    global _sentiment_pipe, MODELS_LOADED, LOAD_ERROR
    if MODELS_LOADED:
        return True
    try:
        import torch
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

        print("[Scorer] Loading sentiment model...", flush=True)
        sent_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = AutoTokenizer.from_pretrained(sent_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            sent_model_name,
            low_cpu_mem_usage=False,
        )
        _sentiment_pipe = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            top_k=None,
            truncation=True,
            max_length=512,
            device=-1,  # CPU
        )
        print("[Scorer] Sentiment model loaded OK", flush=True)

        # Event classification uses fast keyword matching (no heavy model needed)

        MODELS_LOADED = True
        print("[Scorer] All models ready OK", flush=True)
        return True
    except Exception as e:
        LOAD_ERROR = str(e)
        print(f"[Scorer] Model load failed: {e}", flush=True)
        return False


_EVENT_KEYWORDS = {
    "flight disruption": ["cancel", "delay", "disrupt", "ground", "divert", "suspend"],
    "strike or labor dispute": ["strike", "labor", "union", "walkout", "protest", "worker"],
    "weather disruption": ["storm", "snow", "fog", "hurricane", "typhoon", "weather", "flood"],
    "security threat": ["security", "threat", "terror", "bomb", "attack", "war", "conflict", "missile"],
    "tourism growth": ["tourism", "tourist", "visitor", "growth", "record", "boom", "surge"],
    "new route or airline expansion": ["new route", "expansion", "launch", "inaugural", "fleet", "order"],
    "airport congestion": ["congestion", "overcrowd", "queue", "capacity", "busy", "chaos"],
}


def _classify_event_keywords(text: str) -> str:
    text_lower = text.lower()
    best_label = "general travel news"
    best_count = 0
    for label, keywords in _EVENT_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > best_count:
            best_count = count
            best_label = label
    return best_label


def _clean_text(text: str) -> str:
    """Başlık + açıklama birleştir, temizle."""
    if not text:
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:512]


def score_article(title: str, description: str = "") -> dict:
    """
    Tek bir makaleyi skorlar.
    Döner:
        sentiment_label: "positive" | "neutral" | "negative"
        sentiment_score: -1.0 → +1.0
        confidence: 0.0 → 1.0
        event_type: EVENT_LABELS'dan biri
        event_meta: Türkçe etiket + etki
    """
    text = _clean_text(f"{title}. {description}" if description else title)

    if not text:
        return _empty_result()

    # ── Sentiment ────────────────────────────────────────────
    sentiment_label = "neutral"
    sentiment_score = 0.0
    confidence = 0.5

    if _sentiment_pipe:
        try:
            results = _sentiment_pipe(text)[0]  # top_k=None → liste
            # Model çıktısı: [{"label": "positive", "score": 0.8}, ...]
            scores = {r["label"].lower(): r["score"] for r in results}
            pos = scores.get("positive", 0)
            neg = scores.get("negative", 0)
            neu = scores.get("neutral", 0)

            # -1 → +1 aralığına normalize
            sentiment_score = round(pos - neg, 4)

            if pos > neg and pos > neu:
                sentiment_label = "positive"
                confidence = round(pos, 4)
            elif neg > pos and neg > neu:
                sentiment_label = "negative"
                confidence = round(neg, 4)
            else:
                sentiment_label = "neutral"
                confidence = round(neu, 4)
        except Exception as e:
            print(f"[Scorer] Sentiment error: {e}")

    # ── Event Classification (keyword-based, fast) ─────────────
    event_type = _classify_event_keywords(text)

    meta = EVENT_META.get(event_type, EVENT_META["general travel news"])

    return {
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "confidence": confidence,
        "event_type": event_type,
        "event_tr": meta["tr"],
        "event_icon": meta["icon"],
        "event_impact": meta["impact"],
    }


def score_articles(articles: list[dict]) -> list[dict]:
    """Makale listesini toplu skorlar."""
    if not MODELS_LOADED:
        load_models()

    scored = []
    for a in articles:
        scores = score_article(
            title=a.get("title", ""),
            description=a.get("description", ""),
        )
        scored.append({**a, **scores})
    return scored


def aggregate_city_sentiment(scored_articles: list[dict]) -> dict:
    """
    Şehir bazında özet skor üretir.
    Döner:
        composite_score:  -1.0 → +1.0  (ağırlıklı ortalama)
        alert_level:      "low" | "medium" | "high"
        dominant_event:   en sık görülen event türü
        article_count:    analiz edilen makale sayısı
        positive_count / negative_count / neutral_count
    """
    if not scored_articles:
        return _empty_aggregate()

    # Güven ağırlıklı ortalama skor
    total_weight = 0.0
    weighted_score = 0.0
    pos_count = neg_count = neu_count = 0
    event_counts: dict[str, int] = {}
    high_impact_events = []

    for a in scored_articles:
        conf = a.get("confidence", 0.5)
        score = a.get("sentiment_score", 0.0)
        label = a.get("sentiment_label", "neutral")
        event = a.get("event_type", "general travel news")
        impact = a.get("event_impact", 0)

        weighted_score += score * conf
        total_weight += conf

        if label == "positive": pos_count += 1
        elif label == "negative": neg_count += 1
        else: neu_count += 1

        event_counts[event] = event_counts.get(event, 0) + 1

        if abs(impact) >= 1:
            high_impact_events.append({
                "title": a.get("title", ""),
                "event_type": event,
                "event_tr": a.get("event_tr", ""),
                "event_icon": a.get("event_icon", ""),
                "sentiment_label": label,
                "sentiment_score": score,
                "published_at": a.get("published_at", ""),
                "source": a.get("source", ""),
            })

    composite = round(weighted_score / total_weight, 4) if total_weight > 0 else 0.0
    dominant_event = max(event_counts, key=event_counts.get) if event_counts else "general travel news"

    # Alert level
    if composite < -0.3 or any(a.get("event_impact", 0) <= -1 for a in scored_articles):
        alert_level = "high"
    elif composite < -0.1:
        alert_level = "medium"
    else:
        alert_level = "low"

    return {
        "composite_score": composite,
        "alert_level": alert_level,
        "dominant_event": dominant_event,
        "dominant_event_tr": EVENT_META.get(dominant_event, {}).get("tr", dominant_event),
        "article_count": len(scored_articles),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": neu_count,
        "high_impact_events": high_impact_events[:5],
        "event_distribution": event_counts,
    }


def _empty_result() -> dict:
    return {
        "sentiment_label": "neutral",
        "sentiment_score": 0.0,
        "confidence": 0.0,
        "event_type": "general travel news",
        "event_tr": "Genel Seyahat Haberi",
        "event_icon": "📰",
        "event_impact": 0,
    }


def _empty_aggregate() -> dict:
    return {
        "composite_score": 0.0,
        "alert_level": "low",
        "dominant_event": "general travel news",
        "dominant_event_tr": "Genel Seyahat Haberi",
        "article_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "neutral_count": 0,
        "high_impact_events": [],
        "event_distribution": {},
    }
