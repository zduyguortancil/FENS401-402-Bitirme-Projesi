"""Debug Abu Dhabi scoring step by step."""
import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dashboard'))
from sentiment.gnews_rss import fetch_gnews_rss
from sentiment.classifier import classify_event, EVENT_META
from sentiment.scoring import score_article, compute_city_score
from datetime import datetime

rss = fetch_gnews_rss("Abu Dhabi", max_articles=15)
print(f"RSS articles: {len(rss)}")

for a in rss:
    ek, conf = classify_event(a["title"])
    a["event_type"] = ek
    a["confidence"] = conf

scored = [score_article(a) for a in rss]

# Now manually run compute_city_score logic
now = datetime.utcnow()
MAX_AGE_HOURS = 14 * 24
weighted_sum = 0.0
weight_total = 0.0
pos = neg = neu = 0
included = 0

for i, a in enumerate(scored):
    score = a.get("sentiment_score", 0.0)
    label = a.get("sentiment_label", "neutral")
    pub = a.get("published_at", "")

    # Parse date
    hours_old = 168
    from sentiment.scoring import _parse_date
    pub_dt = _parse_date(pub)
    if pub_dt:
        hours_old = max(0, (now - pub_dt).total_seconds() / 3600)

    if hours_old > MAX_AGE_HOURS:
        print(f"  {i+1:2d}. SKIPPED (age={hours_old:.0f}h > {MAX_AGE_HOURS}h)")
        continue

    recency = math.exp(-0.1 * hours_old)
    ws = score * recency
    weighted_sum += ws
    weight_total += recency
    included += 1

    if label == "positive": pos += 1
    elif label == "negative": neg += 1
    else: neu += 1

    print(f"  {i+1:2d}. [{label:8s}] score={score:+.4f} age={hours_old:6.1f}h w={recency:.6f} w*s={ws:+.8f} | {a['event_type']:20s} | {a['title'][:55]}")

print()
print(f"Included: {included} (pos={pos}, neg={neg}, neu={neu})")
print(f"Weighted sum: {weighted_sum:.10f}")
print(f"Weight total: {weight_total:.10f}")
composite = weighted_sum / weight_total if weight_total > 0 else 0
print(f"Composite: {composite:.4f}")
print()

# Compare with actual function
actual = compute_city_score(scored)
print(f"compute_city_score result: {actual['composite_score']}")
print(f"Match: {abs(composite - actual['composite_score']) < 0.001}")
