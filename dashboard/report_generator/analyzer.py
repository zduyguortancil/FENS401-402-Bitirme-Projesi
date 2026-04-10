"""
Pattern Analyzer — Detects actionable insights from simulation data.
"""
from dataclasses import dataclass


@dataclass
class Insight:
    category: str    # revenue, lf, fare_mix, risk, sentiment
    severity: str    # positive, neutral, warning, critical
    key: str         # specific pattern name
    data: dict       # context for text generation


def analyze(rd):
    """Analyze ReportData, return list of Insights."""
    insights = []

    # ── Revenue ──
    dp = rd.revenue_delta_pct
    if dp >= 10:
        sev, key = "positive", "strong_positive"
    elif dp >= 3:
        sev, key = "positive", "moderate_positive"
    elif dp >= -2:
        sev, key = "neutral", "marginal"
    else:
        sev, key = "critical", "negative"

    # Determine revenue driver
    fc = rd.fare_class_totals
    total_fc = sum(fc.values()) or 1
    y_pct = fc.get("Y", 0) / total_fc * 100
    driver = "premium" if y_pct > 25 else "volume" if rd.avg_lf > 0.8 else "mix"

    insights.append(Insight("revenue", sev, key, {
        "delta_pct": round(dp, 1),
        "revenue_dynamic": rd.revenue_dynamic,
        "revenue_baseline": rd.revenue_baseline,
        "driver": driver,
        "y_rev_pct": round(y_pct, 1),
        "sold": rd.total_sold,
        "n_flights": rd.n_flights,
    }))

    # ── Load Factor ──
    lf = rd.avg_lf * 100
    if lf >= 85:
        lf_key = "excellent"
    elif lf >= 75:
        lf_key = "good"
    elif lf >= 60:
        lf_key = "moderate"
    else:
        lf_key = "low"

    # Best/worst routes
    sorted_flights = sorted(rd.flights, key=lambda f: f.get("load_factor", 0))
    best = sorted_flights[-1] if sorted_flights else {}
    worst = sorted_flights[0] if sorted_flights else {}
    best_lf = round(best.get("load_factor", 0) * 100, 1)
    worst_lf = round(worst.get("load_factor", 0) * 100, 1)
    disparity = best_lf - worst_lf

    insights.append(Insight("lf", "positive" if lf >= 75 else "warning", lf_key, {
        "avg_lf": round(lf, 1),
        "best_route": best.get("route", "N/A"),
        "best_lf": best_lf,
        "best_cabin": best.get("cabin", ""),
        "worst_route": worst.get("route", "N/A"),
        "worst_lf": worst_lf,
        "worst_cabin": worst.get("cabin", ""),
        "disparity": "high" if disparity > 30 else "moderate" if disparity > 15 else "uniform",
    }))

    # ── Fare Class Mix ──
    v_pct = fc.get("V", 0) / total_fc * 100
    k_pct = fc.get("K", 0) / total_fc * 100
    m_pct = fc.get("M", 0) / total_fc * 100
    y_pct_fc = fc.get("Y", 0) / total_fc * 100
    vk_pct = v_pct + k_pct
    my_pct = m_pct + y_pct_fc

    if my_pct > 60:
        mix_key = "premium_heavy"
    elif vk_pct > 60:
        mix_key = "discount_heavy"
    else:
        mix_key = "balanced"

    insights.append(Insight("fare_mix", "positive" if mix_key != "discount_heavy" else "warning", mix_key, {
        "v_pct": round(v_pct, 1), "k_pct": round(k_pct, 1),
        "m_pct": round(m_pct, 1), "y_pct": round(y_pct_fc, 1),
        "vk_pct": round(vk_pct, 1), "my_pct": round(my_pct, 1),
        "fc_totals": fc,
    }))

    # ── Risk (Overbooking / Cancellation / No-Show) ──
    denied_pct = rd.denied_boardings / max(rd.total_sales, 1) * 100
    if rd.denied_boardings == 0:
        risk_key = "clean"
    elif denied_pct < 1:
        risk_key = "minor"
    else:
        risk_key = "concerning"

    cancel_rate = rd.cancellations / max(rd.total_sales, 1) * 100
    noshow_rate = rd.no_shows / max(rd.total_sold, 1) * 100

    insights.append(Insight("risk", "positive" if risk_key == "clean" else "warning", risk_key, {
        "denied": rd.denied_boardings,
        "denied_pct": round(denied_pct, 2),
        "denied_cost": rd.denied_boarding_cost,
        "cancellations": rd.cancellations,
        "cancel_rate": round(cancel_rate, 1),
        "refunds": rd.cancellation_refunds,
        "no_shows": rd.no_shows,
        "noshow_rate": round(noshow_rate, 1),
    }))

    # ── Sentiment ──
    n_high = sum(1 for a in rd.sentiment_alerts if a["alert"] == "high")
    n_med = sum(1 for a in rd.sentiment_alerts if a["alert"] == "medium")
    worst_city = rd.sentiment_alerts[0] if rd.sentiment_alerts else None

    if n_high >= 3:
        sent_key = "severe"
    elif n_high + n_med >= 2:
        sent_key = "moderate"
    else:
        sent_key = "stable"

    insights.append(Insight("sentiment", "critical" if sent_key == "severe" else "neutral", sent_key, {
        "n_high": n_high, "n_med": n_med,
        "n_alerts": n_high + n_med,
        "worst_city": worst_city["city"] if worst_city else "N/A",
        "worst_score": round(worst_city["score"] * 100, 0) if worst_city else 0,
        "avg_score": round(rd.sentiment_avg * 100, 0),
        "max_impact": round(abs(rd.sentiment_avg) * 15, 1),
        "min_impact": round(abs(rd.sentiment_avg) * 5, 1),
        "alerts": rd.sentiment_alerts[:5],
    }))

    # ── Overall Verdict ──
    positives = sum(1 for i in insights if i.severity == "positive")
    criticals = sum(1 for i in insights if i.severity == "critical")
    if criticals >= 2 or dp < -5:
        verdict = "red"
    elif positives >= 3 and dp > 0:
        verdict = "green"
    else:
        verdict = "yellow"

    insights.append(Insight("verdict", verdict, verdict, {
        "positives": positives, "criticals": criticals,
    }))

    return insights
