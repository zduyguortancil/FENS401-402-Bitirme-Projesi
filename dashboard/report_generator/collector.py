"""
Data Collector — Gathers simulation results into a structured report dataset.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReportData:
    """All data needed for report generation."""
    # Simulation config
    date: str = ""
    routes: list = field(default_factory=list)
    cabins: list = field(default_factory=list)
    speed: float = 0
    days: int = 180

    # Global stats
    total_bots: int = 0
    total_sales: int = 0
    total_rejected: int = 0
    total_capacity: int = 0
    total_sold: int = 0
    avg_lf: float = 0.0
    revenue_dynamic: float = 0.0
    revenue_baseline: float = 0.0
    revenue_delta: float = 0.0
    revenue_delta_pct: float = 0.0
    cancellations: int = 0
    cancellation_refunds: float = 0.0
    no_shows: int = 0
    denied_boardings: int = 0
    denied_boarding_cost: float = 0.0
    displacement_count: int = 0
    displacement_revenue_saved: float = 0.0
    local_sales: int = 0
    connecting_sales: int = 0

    # Per-flight data
    flights: list = field(default_factory=list)       # summary per flight
    flight_details: dict = field(default_factory=dict) # key -> detail

    # Aggregates
    fare_class_totals: dict = field(default_factory=dict)  # {V:n, K:n, M:n, Y:n}
    region_performance: list = field(default_factory=list)

    # Sentiment
    sentiment_alerts: list = field(default_factory=list)
    sentiment_avg: float = 0.0

    # Verdict
    n_flights: int = 0


def collect(sim_engine, sent_cache=None):
    """Collect all data from a completed simulation."""
    rd = ReportData()

    # Status & stats
    status = sim_engine.get_status()
    stats = status.get("stats", {})
    summary = status.get("summary", {})

    rd.total_bots = stats.get("total_bots", 0)
    rd.total_sales = stats.get("total_sales", 0)
    rd.total_rejected = stats.get("total_rejected", 0)
    rd.total_capacity = summary.get("total_capacity", 0)
    rd.total_sold = summary.get("total_sold", 0)
    rd.avg_lf = summary.get("avg_load_factor", 0)
    rd.revenue_dynamic = summary.get("revenue_dynamic", 0)
    rd.revenue_baseline = summary.get("revenue_baseline", 0)
    rd.revenue_delta = summary.get("revenue_delta", 0)
    rd.revenue_delta_pct = summary.get("revenue_delta_pct", 0)
    rd.n_flights = summary.get("total_flights", 0)
    rd.cancellations = stats.get("cancellations", 0)
    rd.cancellation_refunds = stats.get("cancellation_refunds", 0)
    rd.no_shows = stats.get("no_shows", 0)
    rd.denied_boardings = stats.get("denied_boardings", 0)
    rd.denied_boarding_cost = stats.get("denied_boarding_cost", 0)
    rd.displacement_count = stats.get("displacement_count", 0)
    rd.displacement_revenue_saved = stats.get("displacement_revenue_saved", 0)
    rd.local_sales = stats.get("local_sales", 0)
    rd.connecting_sales = stats.get("connecting_sales", 0)

    # Flights list
    rd.flights = sim_engine.get_flights_list()

    # Route descriptions
    seen_routes = set()
    for f in rd.flights:
        seen_routes.add(f["route"])
        if f["cabin"] not in rd.cabins:
            rd.cabins.append(f["cabin"])
    rd.routes = sorted(seen_routes)

    # Date from first flight
    if rd.flights:
        rd.date = rd.flights[0].get("dep_date", "")

    # Per-flight details + fare class aggregation
    fc_totals = {"V": 0, "K": 0, "M": 0, "Y": 0}
    region_map = {}  # region -> {revenue, sold, capacity, count}

    for f in rd.flights:
        key = f["key"]
        try:
            detail = sim_engine.get_flight_detail(key)
            rd.flight_details[key] = detail

            # Aggregate fare classes
            fc_sold = detail.get("fare_class_sold", {})
            for fc_id in ["V", "K", "M", "Y"]:
                fc_totals[fc_id] += fc_sold.get(fc_id, 0)
        except Exception:
            pass

        # Region aggregation
        region = f.get("region", "Unknown")
        if region not in region_map:
            region_map[region] = {"revenue": 0, "sold": 0, "capacity": 0, "count": 0}
        region_map[region]["revenue"] += f.get("revenue_dynamic", 0)
        region_map[region]["sold"] += f.get("sold", 0)
        region_map[region]["capacity"] += f.get("capacity", 0)
        region_map[region]["count"] += 1

    rd.fare_class_totals = fc_totals
    rd.region_performance = [
        {"region": r, "revenue": d["revenue"], "sold": d["sold"],
         "capacity": d["capacity"], "count": d["count"],
         "avg_lf": d["sold"] / max(d["capacity"], 1)}
        for r, d in sorted(region_map.items(), key=lambda x: -x[1]["revenue"])
    ]

    # Sentiment
    if sent_cache and sent_cache.get("data"):
        scores = []
        for city_key, city_data in sent_cache["data"].items():
            agg = city_data.get("aggregate", {})
            score = agg.get("composite_score", 0)
            alert = agg.get("alert_level", "low")
            scores.append(score)
            if alert in ("high", "medium"):
                rd.sentiment_alerts.append({
                    "city": city_data.get("label", city_key),
                    "score": score, "alert": alert,
                    "articles": agg.get("article_count", 0),
                })
        rd.sentiment_alerts.sort(key=lambda x: x["score"])
        if scores:
            rd.sentiment_avg = sum(scores) / len(scores)

    return rd
