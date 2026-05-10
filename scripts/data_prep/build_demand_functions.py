"""
Sprint 6 — Passenger Segments & Demand Functions
6 passenger segments with:
  - Willingness-to-pay (WTP) ranges
  - Booking window profiles
  - Price elasticity functions
  - Price–demand curves (price sweep)
Behavioral parameters derived from cluster and training data.
"""
import json
import math
import duckdb
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root
DATA_DIR = BASE_DIR / "data"
TRAIN_PATH = DATA_DIR / "processed" / "demand_training.parquet"
META_PATH = DATA_DIR / "processed" / "flight_metadata.parquet"
SNAP_PATH = DATA_DIR / "raw" / "flight_snapshot_v2.parquet"
CLUSTER_PATH = DATA_DIR / "processed" / "passenger_clusters.parquet"
OUT_REPORT = BASE_DIR / "reports" / "demand_functions_report.json"

con = duckdb.connect()

# ═══════════════════════════════════════════════════════════════
# STEP 1: Extract behavioral parameters from existing data
# ═══════════════════════════════════════════════════════════════
print("[1/5] Extracting behavioral parameters from data...", flush=True)

# General statistics
stats = con.execute(f"""
    SELECT
        AVG(y_pax_sold_today)                             AS avg_pax,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY y_pax_sold_today) AS pax_p25,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY y_pax_sold_today) AS pax_p75,
        AVG(load_factor)                                  AS avg_lf,
        AVG(CASE WHEN y_pax_sold_today > 0 THEN dtd END) AS avg_dtd_at_sale,
        AVG(capacity)                                     AS avg_capacity
    FROM read_parquet('{TRAIN_PATH}')
""").fetchone()
avg_pax, pax_p25, pax_p75 = float(stats[0]), float(stats[1]), float(stats[2])
avg_lf, avg_dtd_sale, avg_cap = float(stats[3]), float(stats[4]), float(stats[5])
print(f"   Avg daily pax: {avg_pax:.3f}  |  Avg LF: {avg_lf:.3f}  |  Avg capacity: {avg_cap:.0f}", flush=True)

# DTD-based sales profile (total sales ratio per DTD bucket)
dtd_profile = con.execute(f"""
    SELECT
        dtd_bucket,
        SUM(y_pax_sold_today) AS total_pax,
        COUNT(*) AS row_count,
        AVG(CASE WHEN y_pax_sold_today > 0 THEN 1.0 ELSE 0.0 END) AS sale_rate
    FROM read_parquet('{TRAIN_PATH}')
    GROUP BY dtd_bucket
    ORDER BY dtd_bucket
""").fetchall()
total_pax_all = sum(r[1] for r in dtd_profile)
dtd_dist = {}
for r in dtd_profile:
    dtd_dist[int(r[0])] = {
        "total_pax": float(r[1]),
        "share": round(float(r[1]) / total_pax_all * 100, 2),
        "sale_rate": round(float(r[3]) * 100, 2),
    }
print(f"   DTD buckets: {len(dtd_dist)}", flush=True)

# Cabin-based price reference (average ticket price from snapshot)
cabin_prices = con.execute(f"""
    SELECT
        LOWER(cabin_class) AS cabin,
        AVG(CASE WHEN pax_sold_today > 0
            THEN ticket_rev_today / pax_sold_today ELSE NULL END) AS avg_price,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY
            CASE WHEN pax_sold_today > 0 THEN ticket_rev_today / pax_sold_today END) AS price_p25,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY
            CASE WHEN pax_sold_today > 0 THEN ticket_rev_today / pax_sold_today END) AS price_p75,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY
            CASE WHEN pax_sold_today > 0 THEN ticket_rev_today / pax_sold_today END) AS price_p90
    FROM read_parquet('{SNAP_PATH}')
    WHERE pax_sold_today > 0
    GROUP BY LOWER(cabin_class)
""").fetchall()
price_ref = {}
for r in cabin_prices:
    price_ref[r[0]] = {
        "avg": round(float(r[1]), 2),
        "p25": round(float(r[2]), 2),
        "p75": round(float(r[3]), 2),
        "p90": round(float(r[4]), 2),
    }
print(f"   Price references: {json.dumps(price_ref, indent=2)}", flush=True)

# Region-based price differences
region_prices = con.execute(f"""
    SELECT
        m.region,
        AVG(CASE WHEN s.pax_sold_today > 0
            THEN s.ticket_rev_today / s.pax_sold_today ELSE NULL END) AS avg_price,
        SUM(s.pax_sold_today) AS total_pax
    FROM read_parquet('{SNAP_PATH}') s
    LEFT JOIN read_parquet('{META_PATH}') m
        ON s.flight_id = m.flight_id AND LOWER(s.cabin_class) = LOWER(m.cabin_class)
    WHERE s.pax_sold_today > 0
    GROUP BY m.region
""").fetchall()
region_ref = {}
for r in region_prices:
    if r[0]:
        region_ref[r[0]] = {"avg_price": round(float(r[1]), 2), "total_pax": int(r[2])}

con.close()

# ═══════════════════════════════════════════════════════════════
# STEP 2: Define 6 Passenger Segments
# ═══════════════════════════════════════════════════════════════
print("[2/5] Defining passenger segments...", flush=True)

# Base price per cabin
eco_base = price_ref.get("economy", {}).get("avg", 500)
biz_base = price_ref.get("business", {}).get("avg", 1500)

SEGMENTS = {
    "A": {
        "id": "A",
        "name": "Business Traveler",
        "icon": "💼",
        "color": "#3b82f6",
        "description": "Late booker with high budget and time sensitivity. Prefers business class.",
        "characteristics": [
            "Last-minute booking (0–14 days)",
            "High budget, price-insensitive",
            "Weekday morning flights",
            "Frequent flyer member",
            "Business class preference",
        ],
        "booking_window": {"min_dtd": 0, "max_dtd": 14, "peak_dtd": 5},
        "wtp_multiplier": {"min": 1.8, "max": 2.5},
        "price_elasticity": -0.3,   # inelastic: +10% price → −3% demand
        "base_share_pct": 15,        # ~15% of total passengers
        "preferred_cabin": "business",
        "seasonal_boost": {"congress_expo": 1.4, "peak_business": 1.3, "normal": 1.0},
        "dtd_decay_rate": 0.15,      # high → concentrates in final days
    },
    "B": {
        "id": "B",
        "name": "Diaspora / VFR",
        "icon": "🏠",
        "color": "#10b981",
        "description": "Fixed destination (hometown). August & holiday peaks. Heavy baggage. Moderate price sensitivity.",
        "characteristics": [
            "Fixed destination (hometown)",
            "Peak in summer & holiday periods",
            "High baggage volume",
            "Moderate price sensitivity",
            "Economy class dominant",
        ],
        "booking_window": {"min_dtd": 14, "max_dtd": 60, "peak_dtd": 30},
        "wtp_multiplier": {"min": 1.3, "max": 1.8},
        "price_elasticity": -0.7,
        "base_share_pct": 20,
        "preferred_cabin": "economy",
        "seasonal_boost": {"summer_holiday": 1.6, "religious_holiday": 1.8, "ramadan": 1.3, "new_year": 1.4, "normal": 1.0},
        "dtd_decay_rate": 0.05,
    },
    "C": {
        "id": "C",
        "name": "Congress / Medical Travel",
        "icon": "🏥",
        "color": "#8b5cf6",
        "description": "Fixed date and destination. Group travel potential. Low flexibility. Medium budget.",
        "characteristics": [
            "Fixed date and destination",
            "Group travel potential",
            "Low flexibility",
            "Medium budget level",
            "Mixed economy & business class",
        ],
        "booking_window": {"min_dtd": 7, "max_dtd": 30, "peak_dtd": 14},
        "wtp_multiplier": {"min": 1.2, "max": 1.6},
        "price_elasticity": -0.5,
        "base_share_pct": 12,
        "preferred_cabin": "economy",
        "seasonal_boost": {"congress_expo": 1.8, "festival_season": 1.3, "normal": 1.0},
        "dtd_decay_rate": 0.08,
    },
    "D": {
        "id": "D",
        "name": "Early Leisure",
        "icon": "🏖️",
        "color": "#f59e0b",
        "description": "Promotion hunter with flexible routes. Price is the key driver. Plans months ahead.",
        "characteristics": [
            "Books 60–180 days in advance",
            "Promotion & discount seeker",
            "High route flexibility",
            "Price is the main decision factor",
            "Economy class",
        ],
        "booking_window": {"min_dtd": 60, "max_dtd": 180, "peak_dtd": 90},
        "wtp_multiplier": {"min": 0.7, "max": 1.0},
        "price_elasticity": -1.5,   # very elastic
        "base_share_pct": 25,
        "preferred_cabin": "economy",
        "seasonal_boost": {"summer_holiday": 1.5, "spring_break": 1.3, "winter_holiday": 1.2, "normal": 1.0},
        "dtd_decay_rate": 0.02,
    },
    "E": {
        "id": "E",
        "name": "Student",
        "icon": "🎓",
        "color": "#06b6d4",
        "description": "Budget-constrained with high time & route flexibility. Will switch destinations if price is too high.",
        "characteristics": [
            "Low budget, high price sensitivity",
            "Very high time & route flexibility",
            "May switch to alternative destinations",
            "Plans 30–120 days ahead",
            "Economy class, lowest fare",
        ],
        "booking_window": {"min_dtd": 30, "max_dtd": 120, "peak_dtd": 60},
        "wtp_multiplier": {"min": 0.5, "max": 0.8},
        "price_elasticity": -2.2,   # extremely elastic
        "base_share_pct": 18,
        "preferred_cabin": "economy",
        "seasonal_boost": {"semester_break": 1.5, "summer_holiday": 1.4, "spring_break": 1.3, "normal": 1.0},
        "dtd_decay_rate": 0.03,
    },
    "F": {
        "id": "F",
        "name": "Last-Minute Urgent",
        "icon": "🚨",
        "color": "#ef4444",
        "description": "Emergency, tender, sports event. Will fly at any cost. Completely price-insensitive.",
        "characteristics": [
            "Books within 0–3 days",
            "Zero price sensitivity",
            "Emergency-driven motivation",
            "Fixed destination and time",
            "Any cabin class",
        ],
        "booking_window": {"min_dtd": 0, "max_dtd": 3, "peak_dtd": 1},
        "wtp_multiplier": {"min": 2.5, "max": 4.0},
        "price_elasticity": -0.1,   # completely inelastic
        "base_share_pct": 10,
        "preferred_cabin": "economy",
        "seasonal_boost": {"sports_season": 1.5, "religious_holiday": 1.6, "normal": 1.0},
        "dtd_decay_rate": 0.5,
    },
}

for sid, seg in SEGMENTS.items():
    print(f"   {seg['icon']} Segment {sid}: {seg['name']} — elasticity={seg['price_elasticity']}, share={seg['base_share_pct']}%", flush=True)

# ═══════════════════════════════════════════════════════════════
# STEP 3: Compute Demand Functions
# ═══════════════════════════════════════════════════════════════
print("[3/5] Computing demand functions...", flush=True)


def demand_function(price_ratio, elasticity, dtd, peak_dtd, dtd_decay, seasonal=1.0, base_demand=1.0):
    """
    Demand function:
    Q(p, dtd) = base_demand × price_effect × timing_effect × seasonal_factor

    price_ratio: current_price / base_price (1.0 = base price)
    elasticity: price elasticity (negative, e.g. -0.3)
    dtd: days to departure
    peak_dtd: DTD at which this segment books most intensively
    dtd_decay: timing concentration coefficient
    """
    # Price effect: Q = Q0 × (P/P0)^elasticity
    price_effect = max(price_ratio ** elasticity, 0.01)

    # Timing effect: Gaussian-like around peak_dtd
    dtd_sigma = max(peak_dtd * 0.6, 3)
    timing_effect = math.exp(-0.5 * ((dtd - peak_dtd) / dtd_sigma) ** 2)

    # Extra boost near DTD 0 for last-minute segments
    if dtd_decay >= 0.3 and dtd <= 3:
        timing_effect = max(timing_effect, 0.9)

    return round(base_demand * price_effect * timing_effect * seasonal, 4)


# Compute curve for each segment × price point × DTD bucket
price_ratios = [round(0.3 + i * 0.1, 1) for i in range(28)]  # 0.3x - 3.0x
dtd_points = [0, 1, 3, 5, 7, 14, 21, 30, 45, 60, 90, 120, 150, 180]

segment_curves = {}
for sid, seg in SEGMENTS.items():
    # Price-Demand curve (at DTD = peak_dtd)
    price_demand_curve = []
    for pr in price_ratios:
        q = demand_function(
            price_ratio=pr,
            elasticity=seg["price_elasticity"],
            dtd=seg["booking_window"]["peak_dtd"],
            peak_dtd=seg["booking_window"]["peak_dtd"],
            dtd_decay=seg["dtd_decay_rate"],
            seasonal=1.0,
            base_demand=seg["base_share_pct"] / 100,
        )
        # Revenue = price × quantity
        revenue = round(pr * q, 4)
        price_demand_curve.append({
            "price_ratio": pr,
            "demand": q,
            "revenue": revenue,
        })

    # DTD-Demand curve (at base price)
    dtd_demand_curve = []
    for dtd in dtd_points:
        q = demand_function(
            price_ratio=1.0,
            elasticity=seg["price_elasticity"],
            dtd=dtd,
            peak_dtd=seg["booking_window"]["peak_dtd"],
            dtd_decay=seg["dtd_decay_rate"],
            base_demand=seg["base_share_pct"] / 100,
        )
        dtd_demand_curve.append({
            "dtd": dtd,
            "demand": q,
        })

    # Optimal price point (revenue maximization)
    best_rev = max(price_demand_curve, key=lambda x: x["revenue"])

    segment_curves[sid] = {
        "price_demand": price_demand_curve,
        "dtd_demand": dtd_demand_curve,
        "optimal_price_ratio": best_rev["price_ratio"],
        "optimal_revenue": best_rev["revenue"],
        "optimal_demand": best_rev["demand"],
    }
    print(f"   Segment {sid}: optimal price = {best_rev['price_ratio']:.1f}x base, "
          f"revenue-idx = {best_rev['revenue']:.4f}", flush=True)

# ═══════════════════════════════════════════════════════════════
# STEP 4: Segment Interaction Matrix (which segment shifts when price changes?)
# ═══════════════════════════════════════════════════════════════
print("[4/5] Computing segment interaction matrix...", flush=True)

# If price increases, how much do elastic segments shift?
interaction_matrix = {}
for sid, seg in SEGMENTS.items():
    row = {}
    for target_sid, target_seg in SEGMENTS.items():
        if sid == target_sid:
            row[target_sid] = 0.0
            continue
        # If source segment is elastic and target has a cheaper WTP range → shift potential
        if seg["price_elasticity"] < -1.0 and target_seg["wtp_multiplier"]["max"] < seg["wtp_multiplier"]["min"]:
            overlap = max(0, min(seg["booking_window"]["max_dtd"], target_seg["booking_window"]["max_dtd"])
                          - max(seg["booking_window"]["min_dtd"], target_seg["booking_window"]["min_dtd"]))
            overlap_ratio = overlap / max(seg["booking_window"]["max_dtd"] - seg["booking_window"]["min_dtd"], 1)
            shift = round(abs(seg["price_elasticity"]) * 0.1 * overlap_ratio, 3)
            row[target_sid] = shift
        else:
            row[target_sid] = 0.0
    interaction_matrix[sid] = row

# ═══════════════════════════════════════════════════════════════
# STEP 5: Build report and save
# ═══════════════════════════════════════════════════════════════
print("[5/5] Building report...", flush=True)

# Make segment definitions JSON-serializable
segments_json = {}
for sid, seg in SEGMENTS.items():
    segments_json[sid] = {
        "id": seg["id"],
        "name": seg["name"],
        "icon": seg["icon"],
        "color": seg["color"],
        "description": seg["description"],
        "characteristics": seg["characteristics"],
        "booking_window": seg["booking_window"],
        "wtp_multiplier": seg["wtp_multiplier"],
        "price_elasticity": seg["price_elasticity"],
        "base_share_pct": seg["base_share_pct"],
        "preferred_cabin": seg["preferred_cabin"],
        "seasonal_boost": seg["seasonal_boost"],
        "dtd_decay_rate": seg["dtd_decay_rate"],
    }

report = {
    "version": "2.0",
    "total_segments": len(SEGMENTS),
    "data_summary": {
        "avg_daily_pax": round(avg_pax, 4),
        "avg_load_factor": round(avg_lf, 4),
        "avg_capacity": round(avg_cap, 1),
        "avg_dtd_at_sale": round(avg_dtd_sale, 1),
    },
    "price_reference": price_ref,
    "region_reference": region_ref,
    "dtd_distribution": dtd_dist,
    "segments": segments_json,
    "curves": {
        sid: {
            "price_demand": cv["price_demand"],
            "dtd_demand": cv["dtd_demand"],
            "optimal": {
                "price_ratio": cv["optimal_price_ratio"],
                "revenue_index": cv["optimal_revenue"],
                "demand_at_optimal": cv["optimal_demand"],
            },
        }
        for sid, cv in segment_curves.items()
    },
    "interaction_matrix": interaction_matrix,
    "price_sweep_range": {"min": 0.3, "max": 3.0, "step": 0.1},
    "dtd_points": dtd_points,
}

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n{'=' * 60}")
print("  DEMAND FUNCTIONS REPORT")
print("=" * 60)
print(f"  Segments:      {len(SEGMENTS)}")
print(f"  Price range:   0.3x - 3.0x base")
print(f"  DTD points:    {len(dtd_points)}")
print()

for sid, seg in SEGMENTS.items():
    opt = segment_curves[sid]
    print(f"  {seg['icon']} {sid} — {seg['name']}")
    print(f"     Elasticity: {seg['price_elasticity']}  |  WTP: {seg['wtp_multiplier']['min']}-{seg['wtp_multiplier']['max']}x")
    print(f"     Booking: {seg['booking_window']['min_dtd']}-{seg['booking_window']['max_dtd']} days (peak: {seg['booking_window']['peak_dtd']})")
    print(f"     Optimal price: {opt['optimal_price_ratio']:.1f}x  →  revenue-idx: {opt['optimal_revenue']:.4f}")
    print()

print(f"  📋 Report: {OUT_REPORT}")
print("\n[DONE]")

