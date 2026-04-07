"""Fetch real Abu Dhabi sentiment data for paper example."""
import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dashboard'))
from sentiment.gdelt import fetch_gdelt
from sentiment.gnews_rss import fetch_gnews_rss
from sentiment.classifier import classify_event, EVENT_META
from sentiment.scoring import score_article, compute_city_score
from datetime import datetime

print("=" * 60)
print("ABU DHABI REAL-TIME SENTIMENT ANALYSIS")
print("=" * 60)

gdelt = fetch_gdelt("Abu Dhabi", ["AUH"], country="UAE", max_articles=25, timespan="14d")
rss = fetch_gnews_rss("Abu Dhabi", max_articles=15)
all_articles = gdelt + rss
print(f"\nSources: GDELT={len(gdelt)}, RSS={len(rss)}, Total={len(all_articles)}\n")

for a in all_articles:
    ek, conf = classify_event(a["title"])
    a["event_type"] = ek
    a["confidence"] = conf

scored = [score_article(a) for a in all_articles]

now = datetime.utcnow()
print("=" * 60)
print("ARTICLE-BY-ARTICLE SCORING")
print("=" * 60)
for i, a in enumerate(scored):
    title = a["title"][:70]
    label = a.get("sentiment_label", "?")
    score = a.get("sentiment_score", 0)
    etype = a.get("event_type", "?")
    impact = EVENT_META.get(etype, {}).get("impact", 0)
    pub = a.get("published_at", "")
    hours_old = 168
    try:
        for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"]:
            try:
                pub_dt = datetime.strptime(pub.strip(), fmt)
                hours_old = max(0, (now - pub_dt).total_seconds() / 3600)
                break
            except:
                pass
    except:
        pass

    if hours_old > 336:
        continue

    recency = math.exp(-0.1 * hours_old)
    weighted = score * recency

    print(f"{i+1:2d}. [{label:8s}] score={score:+.3f} impact={impact:+.1f} age={hours_old:5.0f}h w={recency:.4f} ws={weighted:+.6f}")
    print(f"    {etype:25s} | {title}")
    print()

city = compute_city_score(scored)
print("=" * 60)
print("CITY COMPOSITE SCORE")
print("=" * 60)
print(f"  composite_score: {city['composite_score']}")
print(f"  alert_level: {city['alert_level']}")
print(f"  dominant_event: {city['dominant_event']} ({city['dominant_event_tr']})")
print(f"  articles: {city['article_count']} (pos={city['positive_count']}, neg={city['negative_count']}, neu={city['neutral_count']})")
print(f"  event_distribution: {city['event_distribution']}")

s = city["composite_score"]
m_sent = 1.0 + s * 0.15
m_demand = 1.0 + s * 0.30

print()
print("=" * 60)
print("PRICING IMPACT")
print("=" * 60)
print(f"  S_city = {s:.4f}")
print(f"  m_sent (price) = {m_sent:.4f} ({(m_sent-1)*100:+.2f}%)")
print(f"  m_demand (sim) = {m_demand:.4f} ({(m_demand-1)*100:+.2f}%)")

base = 4.01 + 3420 * 0.081
supply = 1.15
demand = 1.20
normal = base * supply * demand * 1.0
crisis = base * supply * demand * m_sent
print()
print("IST-AUH Economy, DTD=30, M class:")
print(f"  base = 4.01 + 3420 x 0.081 = ${base:.0f}")
print(f"  Normal  (S=0):     M = ${normal:.0f}")
print(f"  Current (S={s:.3f}): M = ${crisis:.0f}")
print(f"  Difference: ${crisis - normal:.0f} ({(crisis/normal - 1)*100:+.1f}%)")
