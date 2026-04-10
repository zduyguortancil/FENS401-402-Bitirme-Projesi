"""
Lexicon — Phrase banks, synonym selection, connectors.

Provides linguistic variation so generated text reads naturally.
Selection is deterministic (hash-based), not random — same data = same report.
"""
import hashlib


def _pick(options, seed_str):
    """Deterministic selection from options based on seed string."""
    h = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    return options[h % len(options)]


# ═══════════════════════════════════════════
# REVENUE
# ═══════════════════════════════════════════

REVENUE_VERDICT = {
    "strong_positive": [
        "The dynamic pricing engine delivered a strong revenue uplift of {delta_pct}% over the static baseline.",
        "Dynamic pricing significantly outperformed the baseline, generating {delta_pct}% additional revenue.",
        "Revenue performance was notably positive, with the pricing engine achieving a {delta_pct}% improvement.",
    ],
    "moderate_positive": [
        "The system produced a moderate {delta_pct}% revenue improvement over the baseline.",
        "Dynamic pricing showed a positive but modest uplift of {delta_pct}% compared to static fares.",
        "Revenue exceeded the baseline by {delta_pct}%, indicating effective but conservative pricing.",
    ],
    "marginal": [
        "Revenue was broadly in line with the baseline, showing a {delta_pct}% difference.",
        "The pricing engine produced near-baseline results with a marginal {delta_pct}% delta.",
        "Dynamic and static pricing yielded comparable results, with only {delta_pct}% divergence.",
    ],
    "negative": [
        "Dynamic pricing underperformed the baseline by {delta_pct}%, indicating potential calibration issues.",
        "The system generated {delta_pct}% less revenue than the static baseline, warranting investigation.",
        "Revenue fell below expectations, with dynamic pricing trailing the baseline by {delta_pct}%.",
    ],
}

REVENUE_DRIVER = {
    "volume": "This was primarily driven by higher booking volumes, with {sold} seats filled across {n_flights} flights.",
    "premium": "Premium fare class uptake was the key driver, with Y-class contributing {y_rev_pct}% of total revenue.",
    "mix": "A favorable fare class mix shifted revenue toward higher-yield classes, reducing discount fare dependency.",
    "sentiment": "Sentiment-driven demand adjustments on affected routes contributed to the revenue differential.",
}

# ═══════════════════════════════════════════
# LOAD FACTOR
# ═══════════════════════════════════════════

LF_VERDICT = {
    "excellent": [
        "Average load factor reached {avg_lf}%, reflecting robust demand across the network.",
        "Load factors were strong at {avg_lf}%, indicating healthy seat utilization.",
    ],
    "good": [
        "The network achieved a {avg_lf}% average load factor, within acceptable operational range.",
        "Load factors averaged {avg_lf}%, suggesting adequate demand capture.",
    ],
    "moderate": [
        "Average load factor of {avg_lf}% suggests room for improvement in demand stimulation.",
        "At {avg_lf}% average occupancy, certain routes may benefit from promotional activity.",
    ],
    "low": [
        "Load factors averaged only {avg_lf}%, indicating significant underutilization of capacity.",
        "With an average load factor of {avg_lf}%, demand capture requires immediate attention.",
    ],
}

LF_DISPARITY = {
    "high": "Notable disparity exists between routes: the best-performing route achieved {best_lf}% while the weakest reached only {worst_lf}%.",
    "moderate": "Route performance was moderately varied, ranging from {worst_lf}% to {best_lf}% load factor.",
    "uniform": "Load factors were relatively uniform across routes, ranging from {worst_lf}% to {best_lf}%.",
}

# ═══════════════════════════════════════════
# FARE CLASS
# ═══════════════════════════════════════════

FARE_MIX_VERDICT = {
    "premium_heavy": "The fare class distribution skewed toward premium classes, with M and Y accounting for {my_pct}% of bookings. This indicates strong willingness-to-pay among the passenger mix.",
    "balanced": "Fare class distribution was well-balanced: V ({v_pct}%), K ({k_pct}%), M ({m_pct}%), Y ({y_pct}%). This suggests effective fare class management by the EMSR-b optimization.",
    "discount_heavy": "Discount fares (V and K) dominated at {vk_pct}% of total bookings. While this supports load factor targets, it limits per-passenger yield.",
}

# ═══════════════════════════════════════════
# RISK / OVERBOOKING
# ═══════════════════════════════════════════

RISK_VERDICT = {
    "clean": [
        "Overbooking management performed optimally with zero denied boardings across all flights.",
        "No passengers were denied boarding, confirming that overbooking limits are well-calibrated against no-show rates.",
    ],
    "minor": [
        "{denied} passengers were denied boarding ({denied_pct}% of total sales), within acceptable industry thresholds.",
        "Denied boarding incidents were minimal at {denied} cases, representing {denied_pct}% of bookings.",
    ],
    "concerning": [
        "{denied} denied boardings ({denied_pct}% of sales) exceeded target levels and require overbooking policy review.",
        "The denied boarding rate of {denied_pct}% signals overly aggressive overbooking on affected routes.",
    ],
}

CANCEL_INSIGHT = "Cancellation rate was {cancel_rate}% ({cancellations} bookings), with ${refunds:,.0f} in refunds processed. {fare_detail}"

NOSHOW_INSIGHT = "{no_shows} passengers did not show for their flights ({noshow_rate}% no-show rate), {comparison}."

# ═══════════════════════════════════════════
# SENTIMENT
# ═══════════════════════════════════════════

SENTIMENT_VERDICT = {
    "severe": "Sentiment monitoring detected {n_high} high-alert destinations, with {worst_city} recording the lowest composite score of {worst_score}. This triggered price reductions of up to {max_impact}% on affected routes.",
    "moderate": "{n_alerts} destinations showed elevated risk levels. The sentiment module applied pricing adjustments ranging from {min_impact}% to {max_impact}% across affected routes.",
    "stable": "Sentiment conditions were broadly stable across the monitored network. No significant pricing interventions were triggered by the sentiment module.",
}

# ═══════════════════════════════════════════
# RECOMMENDATIONS
# ═══════════════════════════════════════════

RECOMMENDATIONS = {
    "low_lf_routes": "Consider expanding V and K class availability on underperforming routes ({routes}) during the early booking window (DTD 60-180) to stimulate demand before fare class restrictions tighten.",
    "high_lf_premium": "Routes with consistently high load factors ({routes}) present opportunities for earlier M-to-Y class transition. Shifting the EMSR-b threshold 5-10 DTD earlier could capture additional premium revenue.",
    "discount_heavy_mix": "The high proportion of discount fares suggests price-sensitive demand. Evaluate whether base prices on key routes are competitive, and consider tightening V quotas by 10-15% to push demand into K/M classes.",
    "denied_boarding": "Review overbooking limits on routes with denied boardings. Consider reducing economy overbooking by 1-2 percentage points on {routes} while monitoring no-show rate stability.",
    "sentiment_alert": "Destinations with persistent negative sentiment ({cities}) may require temporary promotional pricing to maintain market share during the disruption period.",
    "cancellation_high": "Y-class cancellation rates of {y_cancel}% are above target. Consider adjusting refund policies or introducing cancellation penalties for flexible fare bookings.",
    "regional_imbalance": "Regional performance disparity (highest: {best_region} at {best_lf}%, lowest: {worst_region} at {worst_lf}%) suggests demand-side imbalances. Cross-regional promotional campaigns may help equalize load factors.",
    "general_positive": "The current pricing configuration is performing well. Continue monitoring booking pace against TFT forecasts and maintain EMSR-b protection levels at current settings.",
}

# ═══════════════════════════════════════════
# CONNECTORS & TRANSITIONS
# ═══════════════════════════════════════════

CONNECTORS = {
    "cause": ["This is primarily driven by", "The key factor is", "This stems from"],
    "contrast": ["However,", "On the other hand,", "In contrast,"],
    "addition": ["Furthermore,", "Additionally,", "Moreover,"],
    "consequence": ["As a result,", "Consequently,", "This implies that"],
    "summary": ["Overall,", "In summary,", "Taken together,"],
}

EXEC_OPENING = [
    "This report presents the post-simulation analysis for the {route_desc} simulation conducted on {date}.",
    "The following analysis summarizes the booking simulation results for {route_desc}, completed on {date}.",
]

EXEC_CLOSING = [
    "The simulation processed {total_bots:,} demand events over a {days}-day booking window, resulting in {total_sales:,} confirmed bookings.",
    "Across the {days}-day simulation window, {total_bots:,} potential passengers were evaluated, yielding {total_sales:,} confirmed sales.",
]

VERDICT_LABEL = {
    "green": "POSITIVE — System performing above expectations",
    "yellow": "ADEQUATE — Performance within acceptable range with improvement opportunities",
    "red": "ATTENTION REQUIRED — Below-target performance identified",
}


def pick(category, key, seed, **kwargs):
    """Pick a phrase from a category and format it with kwargs."""
    bank = category.get(key, category.get("marginal", ["No data available."]))
    if isinstance(bank, list):
        text = _pick(bank, seed)
    else:
        text = bank
    try:
        return text.format(**kwargs)
    except (KeyError, IndexError):
        return text


def connector(ctype, seed):
    """Pick a connector phrase."""
    options = CONNECTORS.get(ctype, [""])
    return _pick(options, seed)
