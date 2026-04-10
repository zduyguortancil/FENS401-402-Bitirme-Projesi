"""
NLG Engine — Converts Insights into natural language paragraphs.

Uses phrase banks + sentence fusion + deterministic variation.
"""
from . import lexicon


def _get_insight(insights, category):
    for i in insights:
        if i.category == category:
            return i
    return None


def generate_executive_summary(rd, insights):
    """Generate 3-4 paragraph executive summary."""
    seed = f"{rd.date}_{rd.n_flights}_{rd.revenue_delta_pct}"
    paras = []

    # Opening
    route_desc = ", ".join(rd.routes[:3])
    if len(rd.routes) > 3:
        route_desc += f" and {len(rd.routes) - 3} others"
    opening = lexicon.pick(
        {"o": lexicon.EXEC_OPENING}, "o", seed,
        route_desc=route_desc, date=rd.date
    )
    closing = lexicon.pick(
        {"o": lexicon.EXEC_CLOSING}, "o", seed + "_close",
        total_bots=rd.total_bots, total_sales=rd.total_sales, days=rd.days
    )
    paras.append(f"{opening} {closing}")

    # Revenue verdict
    rev = _get_insight(insights, "revenue")
    if rev:
        text = lexicon.pick(lexicon.REVENUE_VERDICT, rev.key, seed + "_rev", **rev.data)
        driver_text = lexicon.REVENUE_DRIVER.get(rev.data.get("driver", "mix"), "")
        try:
            driver_text = driver_text.format(**rev.data)
        except (KeyError, IndexError):
            pass
        paras.append(f"{text} {driver_text}")

    # LF + Fare mix combined
    lf = _get_insight(insights, "lf")
    fm = _get_insight(insights, "fare_mix")
    if lf and fm:
        lf_text = lexicon.pick(lexicon.LF_VERDICT, lf.key, seed + "_lf", **lf.data)
        disp_text = lexicon.LF_DISPARITY.get(lf.data.get("disparity", "moderate"), "")
        try:
            disp_text = disp_text.format(**lf.data)
        except (KeyError, IndexError):
            pass
        fm_text = lexicon.FARE_MIX_VERDICT.get(fm.key, "")
        try:
            fm_text = fm_text.format(**fm.data)
        except (KeyError, IndexError):
            pass
        conn = lexicon.connector("addition", seed + "_conn1")
        paras.append(f"{lf_text} {disp_text} {conn} {fm_text.lower() if fm_text else ''}")

    # Risk + closing
    risk = _get_insight(insights, "risk")
    if risk:
        risk_text = lexicon.pick(lexicon.RISK_VERDICT, risk.key, seed + "_risk", **risk.data)
        paras.append(risk_text)

    return paras


def generate_revenue_section(rd, insights):
    """Generate revenue analysis paragraphs."""
    seed = f"rev_{rd.revenue_delta_pct}"
    paras = []
    rev = _get_insight(insights, "revenue")
    if not rev:
        return ["Revenue data unavailable."]

    # Main verdict
    text = lexicon.pick(lexicon.REVENUE_VERDICT, rev.key, seed, **rev.data)
    paras.append(text)

    # Absolute numbers
    paras.append(
        f"Dynamic pricing generated ${rd.revenue_dynamic:,.0f} in total revenue, "
        f"compared to ${rd.revenue_baseline:,.0f} under the static baseline model. "
        f"The net revenue uplift was ${rd.revenue_delta:,.0f}."
    )

    # Per-region breakdown intro
    if rd.region_performance:
        best_reg = rd.region_performance[0]
        paras.append(
            f"At the regional level, {best_reg['region']} led with "
            f"${best_reg['revenue']:,.0f} in dynamic revenue across {best_reg['count']} flights."
        )

    return paras


def generate_lf_section(rd, insights):
    """Generate load factor analysis paragraphs."""
    seed = f"lf_{rd.avg_lf}"
    paras = []
    lf = _get_insight(insights, "lf")
    if not lf:
        return ["Load factor data unavailable."]

    text = lexicon.pick(lexicon.LF_VERDICT, lf.key, seed, **lf.data)
    paras.append(text)

    # Top/bottom flights
    sorted_fl = sorted(rd.flights, key=lambda f: f.get("load_factor", 0), reverse=True)
    if len(sorted_fl) >= 2:
        top = sorted_fl[0]
        bot = sorted_fl[-1]
        paras.append(
            f"The highest load factor was recorded on {top['route']} ({top['cabin']}) at "
            f"{top['load_factor']*100:.0f}% ({top['sold']}/{top['capacity']} seats), "
            f"while {bot['route']} ({bot['cabin']}) recorded the lowest at "
            f"{bot['load_factor']*100:.0f}% ({bot['sold']}/{bot['capacity']} seats)."
        )

    # Connecting vs local
    if rd.connecting_sales > 0:
        conn_pct = rd.connecting_sales / max(rd.total_sales, 1) * 100
        paras.append(
            f"Connecting passengers accounted for {conn_pct:.1f}% of total sales "
            f"({rd.connecting_sales:,} bookings), with the remainder being local point-to-point traffic."
        )

    return paras


def generate_fareclass_section(rd, insights):
    """Generate fare class and risk analysis paragraphs."""
    seed = f"fc_{rd.total_sold}"
    paras = []
    fm = _get_insight(insights, "fare_mix")
    risk = _get_insight(insights, "risk")

    if fm:
        text = lexicon.FARE_MIX_VERDICT.get(fm.key, "")
        try:
            text = text.format(**fm.data)
        except (KeyError, IndexError):
            pass
        if text:
            paras.append(text)

    # Absolute numbers
    fc = rd.fare_class_totals
    total = sum(fc.values()) or 1
    paras.append(
        f"Fare class breakdown: V (Promo) {fc.get('V',0)} tickets ({fc.get('V',0)/total*100:.1f}%), "
        f"K (Discount) {fc.get('K',0)} ({fc.get('K',0)/total*100:.1f}%), "
        f"M (Flex) {fc.get('M',0)} ({fc.get('M',0)/total*100:.1f}%), "
        f"Y (Full) {fc.get('Y',0)} ({fc.get('Y',0)/total*100:.1f}%)."
    )

    # Risk
    if risk:
        risk_text = lexicon.pick(lexicon.RISK_VERDICT, risk.key, seed + "_risk", **risk.data)
        paras.append(risk_text)

        # Cancellation detail
        d = risk.data
        fare_detail = "Refund exposure was concentrated in M and Y classes due to higher refund rates."
        cancel_text = lexicon.CANCEL_INSIGHT.format(
            cancel_rate=d["cancel_rate"], cancellations=d["cancellations"],
            refunds=d["refunds"], fare_detail=fare_detail
        )
        paras.append(cancel_text)

        # No-show
        comparison = "consistent with the route group no-show calibration" if d["noshow_rate"] < 12 else "above expected levels"
        noshow_text = lexicon.NOSHOW_INSIGHT.format(
            no_shows=d["no_shows"], noshow_rate=d["noshow_rate"], comparison=comparison
        )
        paras.append(noshow_text)

    return paras


def generate_sentiment_section(rd, insights):
    """Generate sentiment analysis paragraphs."""
    sent = _get_insight(insights, "sentiment")
    if not sent:
        return ["Sentiment data unavailable."]

    paras = []
    d = sent.data
    text = lexicon.SENTIMENT_VERDICT.get(sent.key, "")
    try:
        text = text.format(**d)
    except (KeyError, IndexError):
        pass
    paras.append(text)

    # Alert details
    alerts = d.get("alerts", [])
    if alerts:
        lines = []
        for a in alerts[:4]:
            lines.append(f"{a['city']} (score: {a['score']*100:.0f}, {a.get('articles',0)} articles)")
        paras.append("Destinations with elevated risk: " + "; ".join(lines) + ".")

    return paras


def generate_recommendations(rd, insights):
    """Generate prioritized recommendation list."""
    recs = []
    lf = _get_insight(insights, "lf")
    fm = _get_insight(insights, "fare_mix")
    risk = _get_insight(insights, "risk")
    sent = _get_insight(insights, "sentiment")

    # Low LF routes
    if lf and lf.data.get("worst_lf", 100) < 70:
        recs.append(lexicon.RECOMMENDATIONS["low_lf_routes"].format(
            routes=f"{lf.data['worst_route']} ({lf.data['worst_cabin']})"
        ))

    # Premium opportunity
    if lf and lf.data.get("best_lf", 0) > 85:
        recs.append(lexicon.RECOMMENDATIONS["high_lf_premium"].format(
            routes=f"{lf.data['best_route']} ({lf.data['best_cabin']})"
        ))

    # Discount heavy
    if fm and fm.key == "discount_heavy":
        recs.append(lexicon.RECOMMENDATIONS["discount_heavy_mix"])

    # Denied boardings
    if risk and risk.data.get("denied", 0) > 0:
        recs.append(lexicon.RECOMMENDATIONS["denied_boarding"].format(routes="affected routes"))

    # Sentiment
    if sent and sent.data.get("n_high", 0) > 0:
        cities = ", ".join(a["city"] for a in sent.data.get("alerts", [])[:3])
        recs.append(lexicon.RECOMMENDATIONS["sentiment_alert"].format(cities=cities))

    # Regional imbalance
    if rd.region_performance and len(rd.region_performance) >= 2:
        best_r = rd.region_performance[0]
        worst_r = rd.region_performance[-1]
        if best_r["avg_lf"] - worst_r["avg_lf"] > 0.15:
            recs.append(lexicon.RECOMMENDATIONS["regional_imbalance"].format(
                best_region=best_r["region"], best_lf=f"{best_r['avg_lf']*100:.0f}",
                worst_region=worst_r["region"], worst_lf=f"{worst_r['avg_lf']*100:.0f}"
            ))

    # General positive
    if not recs:
        recs.append(lexicon.RECOMMENDATIONS["general_positive"])

    return recs


def get_verdict(insights):
    """Get overall verdict."""
    v = _get_insight(insights, "verdict")
    return v.key if v else "yellow"
