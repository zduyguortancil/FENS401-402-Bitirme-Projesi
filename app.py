"""
Flight Snapshot Dashboard — Sprint 4
Flask + DuckDB backend.
V2 parquet (ticket + ancillary revenue) + metadata lookup.
Demand forecast via two-stage XGBoost (classifier + regressor).
"""

import os
import math
import json
import numpy as np
from flask import Flask, render_template, jsonify, request
import duckdb

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

# ─── FILE SELECTION (v2 default, v1 fallback) ────────────
SNAPSHOT_V2 = f"{BASE_DIR}/flight_snapshot_v2.parquet"
SNAPSHOT_V1 = f"{BASE_DIR}/flight_snapshot.parquet"
METADATA_PATH = f"{BASE_DIR}/flight_metadata.parquet"

if os.path.exists(SNAPSHOT_V2.replace("/", os.sep)):
    PARQUET_PATH = SNAPSHOT_V2
    USE_V2 = True
else:
    PARQUET_PATH = SNAPSHOT_V1
    USE_V2 = False

# ─── DEMAND FORECAST MODEL ────────────────────────────────
CLF_PATH = f"{BASE_DIR}/xgb_demand_classifier.pkl"
REG_PATH = f"{BASE_DIR}/xgb_demand_regressor.pkl"
FEAT_PATH = f"{BASE_DIR}/feature_list.json"

FORECAST_READY = False
DEMAND_CLF = None
DEMAND_REG = None
FEATURE_LIST = None

try:
    import joblib
    if all(os.path.exists(p.replace('/', os.sep)) for p in [CLF_PATH, REG_PATH, FEAT_PATH]):
        DEMAND_CLF = joblib.load(CLF_PATH.replace('/', os.sep))
        DEMAND_REG = joblib.load(REG_PATH.replace('/', os.sep))
        with open(FEAT_PATH.replace('/', os.sep), 'r') as _f:
            FEATURE_LIST = json.load(_f)["features"]
        FORECAST_READY = True
        print(f"[Forecast] Models loaded ({len(FEATURE_LIST)} features)")
    else:
        print("[Forecast] Model files not found, forecast disabled")
except Exception as e:
    print(f"[Forecast] Failed to load models: {e}")


def get_con():
    return duckdb.connect()


def _num(val):
    if val is None:
        return None
    try:
        if math.isnan(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


# ─── PAGES ───────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ─── SEARCH API ──────────────────────────────────────────
@app.route("/api/flights")
def api_flights():
    """Uçuş numarası veya havaalanı koduna göre arama (metadata'dan)."""
    q = request.args.get("q", "").strip().upper()
    con = get_con()
    meta = METADATA_PATH
    results = []

    if q:
        fn_rows = con.execute(f"""
            SELECT DISTINCT flight_number
            FROM read_parquet('{meta}')
            WHERE flight_number IS NOT NULL
              AND UPPER(flight_number) LIKE '%' || $1 || '%'
            ORDER BY flight_number
            LIMIT 15
        """, [q]).fetchall()
        for r in fn_rows:
            results.append({"type": "flight", "value": r[0], "label": r[0]})

        apt_rows = con.execute(f"""
            SELECT DISTINCT airport FROM (
                SELECT departure_airport AS airport FROM read_parquet('{meta}')
                WHERE departure_airport IS NOT NULL
                UNION
                SELECT arrival_airport AS airport FROM read_parquet('{meta}')
                WHERE arrival_airport IS NOT NULL
            )
            WHERE UPPER(airport) LIKE '%' || $1 || '%'
            ORDER BY airport
            LIMIT 10
        """, [q]).fetchall()
        for r in apt_rows:
            results.append({"type": "airport", "value": r[0],
                            "label": f"{r[0]} — Tüm Uçuşlar"})
    else:
        fn_rows = con.execute(f"""
            SELECT DISTINCT flight_number
            FROM read_parquet('{meta}')
            WHERE flight_number IS NOT NULL
            ORDER BY flight_number
            LIMIT 30
        """).fetchall()
        for r in fn_rows:
            results.append({"type": "flight", "value": r[0], "label": r[0]})

    con.close()
    return jsonify(results)


@app.route("/api/airport/<airport_code>")
def api_airport_flights(airport_code):
    con = get_con()
    rows = con.execute(f"""
        SELECT DISTINCT flight_number, departure_airport, arrival_airport
        FROM read_parquet('{METADATA_PATH}')
        WHERE departure_airport = $1 OR arrival_airport = $1
        ORDER BY flight_number
    """, [airport_code.upper()]).fetchall()
    con.close()
    return jsonify([{
        "flight_number": r[0],
        "departure_airport": r[1],
        "arrival_airport": r[2],
    } for r in rows])


@app.route("/api/flight/<flight_number>")
def api_flight_dates(flight_number):
    con = get_con()
    rows = con.execute(f"""
        SELECT DISTINCT
            flight_id,
            departure_datetime,
            departure_airport,
            arrival_airport,
            region
        FROM read_parquet('{METADATA_PATH}')
        WHERE flight_number = $1
        ORDER BY departure_datetime DESC
    """, [flight_number]).fetchall()
    con.close()
    return jsonify([{
        "flight_id": r[0],
        "departure_datetime": str(r[1]) if r[1] else None,
        "departure_airport": r[2],
        "arrival_airport": r[3],
        "region": r[4],
    } for r in rows])


@app.route("/api/flights/date")
def api_flights_by_date():
    """Return all flights on a given departure date."""
    date_str = request.args.get("date", "").strip()
    if not date_str:
        return jsonify([])
    con = get_con()
    rows = con.execute(f"""
        SELECT DISTINCT
            flight_id,
            flight_number,
            departure_airport,
            arrival_airport,
            departure_datetime,
            region
        FROM read_parquet('{METADATA_PATH}')
        WHERE CAST(departure_datetime AS DATE) = CAST($1 AS DATE)
        ORDER BY departure_datetime, flight_number
    """, [date_str]).fetchall()
    con.close()
    return jsonify([{
        "flight_id": r[0],
        "flight_number": r[1],
        "departure_airport": r[2],
        "arrival_airport": r[3],
        "departure_datetime": str(r[4]) if r[4] else None,
        "region": r[5],
    } for r in rows])


# ─── SNAPSHOT API ────────────────────────────────────────
@app.route("/api/snapshot/<path:flight_id>")
def api_snapshot(flight_id):
    cabin = request.args.get("cabin", "").strip()
    con = get_con()

    cabin_filter = ""
    params = [flight_id]
    if cabin:
        cabin_filter = "AND LOWER(s.cabin_class) = LOWER($2)"
        params.append(cabin)

    if USE_V2:
        # V2: JOIN snapshot with metadata
        rows = con.execute(f"""
            SELECT
                s.flight_id,
                s.cabin_class,
                s.dtd,
                s.pax_sold_today,
                s.pax_sold_cum,
                s.pax_last_7d,
                s.ticket_rev_today,
                s.ticket_rev_cum,
                s.anc_rev_today,
                s.anc_rev_cum,
                s.ticket_rev_today + s.anc_rev_today AS total_rev_today,
                s.ticket_rev_cum + s.anc_rev_cum AS total_rev_cum,
                m.flight_number,
                m.departure_airport,
                m.arrival_airport,
                m.departure_datetime,
                m.region,
                m.distance_km,
                m.flight_time_min,
                m.capacity,
                s.ff_gold_pct,
                s.ff_elite_pct
            FROM read_parquet('{PARQUET_PATH}') s
            LEFT JOIN read_parquet('{METADATA_PATH}') m
              ON s.flight_id = m.flight_id AND s.cabin_class = m.cabin_class
            WHERE s.flight_id = $1 {cabin_filter}
            ORDER BY s.cabin_class, s.dtd DESC
        """, params).fetchall()
    else:
        # V1 fallback: revenue_today/cum = ticket, anc = 0
        rows = con.execute(f"""
            SELECT
                flight_id,
                cabin_class,
                dtd,
                pax_sold_today,
                pax_sold_cum,
                pax_last_7d,
                revenue_today AS ticket_rev_today,
                revenue_cum AS ticket_rev_cum,
                0 AS anc_rev_today,
                0 AS anc_rev_cum,
                revenue_today AS total_rev_today,
                revenue_cum AS total_rev_cum,
                flight_number,
                departure_airport,
                arrival_airport,
                departure_datetime,
                region,
                distance_km,
                flight_time_min,
                capacity,
                NULL AS ff_gold_pct,
                NULL AS ff_elite_pct
            FROM read_parquet('{PARQUET_PATH}')
            WHERE flight_id = $1 {cabin_filter.replace('s.cabin_class','cabin_class')}
            ORDER BY cabin_class, dtd DESC
        """, params).fetchall()

    # Cabin classes
    cabins = con.execute(f"""
        SELECT DISTINCT cabin_class
        FROM read_parquet('{PARQUET_PATH}')
        WHERE flight_id = $1 AND cabin_class IS NOT NULL
    """, [flight_id]).fetchall()

    con.close()

    data = []
    for r in rows:
        capacity = _num(r[19])
        pax_cum = _num(r[4])
        remaining = max(int(capacity) - int(pax_cum), 0) if capacity is not None and pax_cum is not None else None
        load_factor = (pax_cum / capacity) if capacity is not None and capacity > 0 and pax_cum is not None else None
        ticket_today = _num(r[6])
        pax_today = _num(r[3])
        avg_fare = (ticket_today / pax_today) if ticket_today is not None and pax_today is not None and pax_today > 0 else None

        total_rev_cum = _num(r[11])
        anc_rev_cum = _num(r[9])
        anc_share = (anc_rev_cum / total_rev_cum) if anc_rev_cum is not None and total_rev_cum is not None and total_rev_cum > 0 else None

        data.append({
            "flight_id": r[0],
            "cabin_class": r[1],
            "dtd": r[2],
            "pax_sold_today": _num(r[3]),
            "pax_sold_cum": pax_cum,
            "pax_last_7d": _num(r[5]),
            "ticket_rev_today": ticket_today,
            "ticket_rev_cum": _num(r[7]),
            "anc_rev_today": _num(r[8]),
            "anc_rev_cum": anc_rev_cum,
            "total_rev_today": _num(r[10]),
            "total_rev_cum": total_rev_cum,
            "ancillary_share": _num(anc_share),
            "capacity": capacity,
            "remaining_seats": remaining,
            "load_factor": _num(load_factor),
            "avg_fare_today": _num(avg_fare),
            "flight_number": r[12],
            "departure_airport": r[13],
            "arrival_airport": r[14],
            "departure_datetime": str(r[15]) if r[15] else None,
            "region": r[16],
            "distance_km": _num(r[17]),
            "flight_time_min": _num(r[18]),
            "ff_gold_pct": _num(r[20]),
            "ff_elite_pct": _num(r[21]),
        })

    # Summary
    summary = {}
    if data:
        meta = data[0]
        summary = {
            "flight_number": meta["flight_number"],
            "departure_airport": meta["departure_airport"],
            "arrival_airport": meta["arrival_airport"],
            "departure_datetime": meta["departure_datetime"],
            "region": meta["region"],
            "distance_km": meta["distance_km"],
            "flight_time_min": meta["flight_time_min"],
        }

    return jsonify({
        "summary": summary,
        "cabins": [c[0] for c in cabins],
        "rows": data,
        "use_v2": USE_V2,
    })


# ─── FORECAST API ─────────────────────────────────────────
@app.route("/api/forecast/<path:flight_id>")
def api_forecast(flight_id):
    """Produce per-DTD demand forecast for a flight + cabin with horizon support."""
    if not FORECAST_READY:
        return jsonify({"error": "Forecast model not loaded"}), 503

    cabin = request.args.get("cabin", "").strip().lower() or None
    mode = request.args.get("mode", "remaining").strip().lower()
    try:
        current_dtd = int(request.args.get("current_dtd", 30))
    except:
        current_dtd = 30

    con = get_con()

    # ── Pull snapshot rows for this flight ──
    cabin_filter = "AND LOWER(s.cabin_class) = $2" if cabin else ""
    params = [flight_id, cabin] if cabin else [flight_id]

    rows = con.execute(f"""
        SELECT
            s.flight_id,
            LOWER(s.cabin_class) AS cabin_class,
            s.dtd,
            s.pax_sold_today,
            s.pax_sold_cum,
            s.pax_last_7d,
            s.ff_gold_pct,
            s.ff_elite_pct,
            m.capacity,
            m.distance_km,
            m.flight_time_min,
            m.region,
            m.departure_datetime
        FROM read_parquet('{PARQUET_PATH}') s
        LEFT JOIN read_parquet('{METADATA_PATH}') m
            ON s.flight_id = m.flight_id
            AND LOWER(s.cabin_class) = LOWER(m.cabin_class)
        WHERE s.flight_id = $1 {cabin_filter}
        ORDER BY s.cabin_class, s.dtd DESC
    """, params).fetchall()
    con.close()

    if not rows:
        return jsonify({"error": "No data found", "rows": []})

    col_names = ["flight_id", "cabin_class", "dtd", "pax_sold_today",
                 "pax_sold_cum", "pax_last_7d", "ff_gold_pct", "ff_elite_pct",
                 "capacity", "distance_km", "flight_time_min", "region",
                 "departure_datetime"]

    results = []
    X_rows = []
    
    # Track some metrics for summary
    max_dtd = 0
    actual_so_far = 0
    actual_final = 0
    capacity = 0

    for r in rows:
        rd = dict(zip(col_names, r))
        dtd = _num(rd["dtd"]) or 0
        if dtd > max_dtd: max_dtd = dtd
        
        pax_cum = _num(rd["pax_sold_cum"]) or 0
        cap = _num(rd["capacity"]) or 0
        if cap > capacity: capacity = cap
        
        # Operational actuals
        if dtd == 0: actual_final = pax_cum
        # Current DTD might not be exactly in the data (snapshots might skip days)
        # We'll pick the closest p_cum that is as-of >= current_dtd
        if dtd >= current_dtd:
            actual_so_far = pax_cum

        remaining = max(cap - pax_cum, 0) if cap else 0
        lf = (pax_cum / cap) if cap and cap > 0 else 0.0
        cabin_cls = (rd["cabin_class"] or "").lower()
        region = rd["region"] or ""

        # DTD bucket code (same as training)
        if dtd <= 3: dtd_bkt = 0
        elif dtd <= 7: dtd_bkt = 1
        elif dtd <= 14: dtd_bkt = 2
        elif dtd <= 30: dtd_bkt = 3
        elif dtd <= 60: dtd_bkt = 4
        elif dtd <= 90: dtd_bkt = 5
        else: dtd_bkt = 6

        # Calendar features
        dep_dt = rd["departure_datetime"]
        dep_year = dep_month = dep_dow = dep_hour = 0
        if dep_dt:
            from datetime import datetime
            if isinstance(dep_dt, str):
                try:
                    dep_dt = datetime.fromisoformat(dep_dt)
                except Exception:
                    dep_dt = None
            if dep_dt and hasattr(dep_dt, 'year'):
                dep_year = dep_dt.year
                dep_month = dep_dt.month
                dep_dow = dep_dt.weekday()
                dep_hour = dep_dt.hour

        # Build feature dict
        feat = {f: 0.0 for f in FEATURE_LIST}
        feat["dtd"] = float(dtd)
        feat["pax_sold_cum"] = float(pax_cum)
        feat["pax_last_7d"] = float(_num(rd["pax_last_7d"]) or 0)
        feat["capacity"] = float(cap)
        feat["remaining_seats"] = float(remaining)
        feat["load_factor"] = float(lf)
        feat["distance_km"] = float(_num(rd["distance_km"]) or 0)
        feat["flight_time_min"] = float(_num(rd["flight_time_min"]) or 0)
        feat["dep_year"] = float(dep_year)
        feat["dep_month"] = float(dep_month)
        feat["dep_dow"] = float(dep_dow)
        feat["dep_hour"] = float(dep_hour)
        feat["ff_gold_pct"] = float(_num(rd["ff_gold_pct"]) or 0)
        feat["ff_elite_pct"] = float(_num(rd["ff_elite_pct"]) or 0)

        # One-hot
        if f"cabin_class_{cabin_cls}" in feat: feat[f"cabin_class_{cabin_cls}"] = 1.0
        if f"region_{region}" in feat: feat[f"region_{region}"] = 1.0
        if f"dtd_bucket_{float(dtd_bkt)}" in feat: feat[f"dtd_bucket_{float(dtd_bkt)}"] = 1.0

        X_rows.append([feat[f] for f in FEATURE_LIST])
        results.append({
            "dtd": dtd,
            "cabin_class": cabin_cls,
            "pax_sold_today": _num(rd["pax_sold_today"]),
            "pax_sold_cum": pax_cum,
            "pax_last_7d": _num(rd["pax_last_7d"]),
        })

    # ── Predict ──
    X = np.array(X_rows, dtype=np.float32)
    p_sale = DEMAND_CLF.predict_proba(X)[:, 1]
    y_pos = np.clip(DEMAND_REG.predict(X), 0, None)
    y_pred = p_sale * y_pos

    # Calculate Summaries
    window_start = max_dtd if mode == "full" else current_dtd
    forecast_remaining = 0
    sum_p_sale = 0
    count_p_sale = 0

    for i, res in enumerate(results):
        res["y_pred"] = round(float(y_pred[i]), 4)
        res["p_sale"] = round(float(p_sale[i]), 4)
        
        # Add to windowed metrics
        if res["dtd"] <= window_start:
            forecast_remaining += res["y_pred"]
            sum_p_sale += res["p_sale"]
            count_p_sale += 1

    metadata = {
        "mode": mode,
        "current_dtd": current_dtd,
        "window_start": window_start,
        "window_end": 0,
        "actual_so_far": actual_so_far,
        "actual_final": actual_final,
        "forecast_remaining": round(forecast_remaining, 2),
        "forecast_total_est": round(actual_so_far + forecast_remaining, 2),
        "avg_p_sale": round(sum_p_sale / count_p_sale, 4) if count_p_sale > 0 else 0,
        "capacity": capacity
    }

    return jsonify({"rows": results, "metadata": metadata})


# ─── CLUSTER API ──────────────────────────────────────────
CLUSTER_PARQUET = f"{BASE_DIR}/passenger_clusters.parquet"
CLUSTER_REPORT  = f"{BASE_DIR}/cluster_report.json"

@app.route("/api/clusters")
def api_clusters():
    """Return cluster summary from pre-computed report."""
    report_path = CLUSTER_REPORT.replace("/", os.sep)
    if not os.path.exists(report_path):
        return jsonify({"error": "Cluster report not found"}), 404
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    return jsonify(report)


@app.route("/api/cluster/<int:cluster_id>")
def api_cluster_detail(cluster_id):
    """Return flights belonging to a specific cluster."""
    parquet_path = CLUSTER_PARQUET.replace("/", os.sep)
    if not os.path.exists(parquet_path):
        return jsonify({"error": "Cluster data not found"}), 404

    limit = request.args.get("limit", 50, type=int)
    con = get_con()
    rows = con.execute(f"""
        SELECT
            flight_id, cabin_class, cluster, cluster_label,
            avg_dtd_at_purchase, pct_last_minute, pct_early_bird,
            avg_daily_pax, max_load_factor, ff_gold_avg, ff_elite_avg,
            is_business, is_weekday, is_morning_flight, distance_km,
            total_pax, capacity
        FROM read_parquet('{CLUSTER_PARQUET}')
        WHERE cluster = $1
        ORDER BY total_pax DESC
        LIMIT $2
    """, [cluster_id, limit]).fetchall()
    con.close()

    cols = ["flight_id", "cabin_class", "cluster", "cluster_label",
            "avg_dtd_at_purchase", "pct_last_minute", "pct_early_bird",
            "avg_daily_pax", "max_load_factor", "ff_gold_avg", "ff_elite_avg",
            "is_business", "is_weekday", "is_morning_flight", "distance_km",
            "total_pax", "capacity"]
    data = [dict(zip(cols, [_num(v) if isinstance(v, (int, float)) else v for v in r])) for r in rows]
    return jsonify({"cluster_id": cluster_id, "rows": data})


# ─── TREND ANALYSIS API ──────────────────────────────────
TRAINING_PARQUET = f"{BASE_DIR}/demand_training.parquet"

@app.route("/api/trends")
def api_trends():
    """Monthly demand trend analysis — time series data."""
    year_filter = request.args.get("year", "").strip()
    cabin_filter = request.args.get("cabin", "").strip().lower()
    region_filter = request.args.get("region", "").strip()

    con = get_con()
    path = TRAINING_PARQUET

    where_clauses = ["dep_year IS NOT NULL", "dep_month IS NOT NULL"]
    params = []
    param_idx = 1

    if year_filter:
        where_clauses.append(f"dep_year = ${param_idx}")
        params.append(int(year_filter))
        param_idx += 1
    if cabin_filter:
        where_clauses.append(f"LOWER(cabin_class) = ${param_idx}")
        params.append(cabin_filter)
        param_idx += 1
    if region_filter:
        where_clauses.append(f"region = ${param_idx}")
        params.append(region_filter)
        param_idx += 1

    where_sql = " AND ".join(where_clauses)

    # 1. Monthly aggregation
    monthly = con.execute(f"""
        SELECT
            dep_year,
            dep_month,
            SUM(y_pax_sold_today)              AS total_pax,
            AVG(y_pax_sold_today)              AS avg_daily_pax,
            AVG(load_factor)                   AS avg_load_factor,
            COUNT(DISTINCT flight_id)          AS flight_count,
            SUM(CASE WHEN y_pax_sold_today > 0 THEN 1 ELSE 0 END) * 100.0
                / COUNT(*) AS sale_rate_pct
        FROM read_parquet('{path}')
        WHERE {where_sql}
        GROUP BY dep_year, dep_month
        ORDER BY dep_year, dep_month
    """, params).fetchall()

    # 2. Cabin breakdown by month
    cabin_monthly = con.execute(f"""
        SELECT
            dep_year, dep_month,
            LOWER(cabin_class) AS cabin,
            SUM(y_pax_sold_today) AS total_pax,
            AVG(y_pax_sold_today) AS avg_daily_pax
        FROM read_parquet('{path}')
        WHERE {where_sql}
        GROUP BY dep_year, dep_month, LOWER(cabin_class)
        ORDER BY dep_year, dep_month, cabin
    """, params).fetchall()

    # 3. Region breakdown by month
    region_monthly = con.execute(f"""
        SELECT
            dep_year, dep_month,
            region,
            SUM(y_pax_sold_today) AS total_pax
        FROM read_parquet('{path}')
        WHERE {where_sql}
        GROUP BY dep_year, dep_month, region
        ORDER BY dep_year, dep_month, region
    """, params).fetchall()

    # 4. Day-of-week pattern
    dow_pattern = con.execute(f"""
        SELECT
            dep_dow,
            SUM(y_pax_sold_today) AS total_pax,
            AVG(y_pax_sold_today) AS avg_pax
        FROM read_parquet('{path}')
        WHERE {where_sql}
        GROUP BY dep_dow
        ORDER BY dep_dow
    """, params).fetchall()

    # 5. Available filters
    years = con.execute(f"""
        SELECT DISTINCT dep_year FROM read_parquet('{path}')
        WHERE dep_year IS NOT NULL ORDER BY dep_year
    """).fetchall()
    cabins = con.execute(f"""
        SELECT DISTINCT LOWER(cabin_class) FROM read_parquet('{path}')
        WHERE cabin_class IS NOT NULL ORDER BY 1
    """).fetchall()
    regions = con.execute(f"""
        SELECT DISTINCT region FROM read_parquet('{path}')
        WHERE region IS NOT NULL ORDER BY region
    """).fetchall()

    con.close()

    month_names = ["", "Oca", "Şub", "Mar", "Nis", "May", "Haz",
                   "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]

    result = {
        "monthly": [{
            "year": r[0], "month": r[1],
            "month_name": month_names[r[1]] if 1 <= r[1] <= 12 else f"M{r[1]}",
            "label": f"{r[0]}-{r[1]:02d}",
            "total_pax": int(r[2]) if r[2] else 0,
            "avg_daily_pax": round(float(r[3]), 4) if r[3] else 0,
            "avg_load_factor": round(float(r[4]), 4) if r[4] else 0,
            "flight_count": int(r[5]) if r[5] else 0,
            "sale_rate_pct": round(float(r[6]), 2) if r[6] else 0,
        } for r in monthly],

        "cabin_monthly": [{
            "year": r[0], "month": r[1], "cabin": r[2],
            "total_pax": int(r[3]) if r[3] else 0,
            "avg_daily_pax": round(float(r[4]), 4) if r[4] else 0,
        } for r in cabin_monthly],

        "region_monthly": [{
            "year": r[0], "month": r[1], "region": r[2],
            "total_pax": int(r[3]) if r[3] else 0,
        } for r in region_monthly],

        "dow_pattern": [{
            "dow": r[0],
            "dow_name": ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"][r[0]] if 0 <= r[0] <= 6 else f"D{r[0]}",
            "total_pax": int(r[1]) if r[1] else 0,
            "avg_pax": round(float(r[2]), 4) if r[2] else 0,
        } for r in dow_pattern],

        "filters": {
            "years": [r[0] for r in years],
            "cabins": [r[0] for r in cabins],
            "regions": [r[0] for r in regions],
        }
    }

    return jsonify(result)


# ─── TOP ROUTES API ──────────────────────────────────────
@app.route("/api/top-routes")
def api_top_routes():
    """EDA: Top N routes by total pax with load-factor, revenue & DTD curves."""
    n = request.args.get("n", 10, type=int)
    sort_by = request.args.get("sort", "total_pax")  # total_pax | final_lf | total_rev
    cabin_filter = request.args.get("cabin", "").strip().lower()

    con = get_con()
    tp = TRAINING_PARQUET
    sp = PARQUET_PATH  # snapshot v2

    cabin_where = f"AND LOWER(t.cabin_class) = '{cabin_filter}'" if cabin_filter else ""

    # 1) Identify top routes with summary metrics
    order_col = {
        "total_pax": "total_pax DESC",
        "final_lf": "final_lf DESC",
        "total_rev": "total_rev DESC",
    }.get(sort_by, "total_pax DESC")

    top_routes = con.execute(f"""
        WITH route_summary AS (
            SELECT
                t.flight_id,
                t.cabin_class,
                SUM(t.y_pax_sold_today) AS total_pax,
                MAX(t.capacity) AS capacity,
                MAX(t.load_factor) AS final_lf,
                AVG(t.load_factor) AS avg_lf,
                MAX(t.distance_km) AS distance_km,
                MAX(t.region) AS region,
                MAX(t.dep_year) AS dep_year,
                MAX(t.dep_month) AS dep_month,
                MAX(t.dep_dow) AS dep_dow,
                MAX(t.dep_hour) AS dep_hour,
                MAX(t.ff_gold_pct) AS ff_gold_pct,
                MAX(t.ff_elite_pct) AS ff_elite_pct
            FROM read_parquet('{tp}') t
            WHERE 1=1 {cabin_where}
            GROUP BY t.flight_id, t.cabin_class
        ),
        route_rev AS (
            SELECT
                s.flight_id,
                s.cabin_class,
                SUM(s.ticket_rev_today) AS total_ticket_rev,
                SUM(s.anc_rev_today) AS total_anc_rev,
                SUM(s.ticket_rev_today) + SUM(s.anc_rev_today) AS total_rev,
                AVG(CASE WHEN s.pax_sold_today > 0
                    THEN s.ticket_rev_today / s.pax_sold_today ELSE NULL END) AS avg_ticket_price
            FROM read_parquet('{sp}') s
            GROUP BY s.flight_id, s.cabin_class
        )
        SELECT
            rs.flight_id, rs.cabin_class,
            rs.total_pax, rs.capacity, rs.final_lf, rs.avg_lf,
            rs.distance_km, rs.region, rs.dep_year, rs.dep_month,
            rs.dep_dow, rs.dep_hour, rs.ff_gold_pct, rs.ff_elite_pct,
            COALESCE(rr.total_ticket_rev, 0) AS total_ticket_rev,
            COALESCE(rr.total_anc_rev, 0) AS total_anc_rev,
            COALESCE(rr.total_rev, 0) AS total_rev,
            COALESCE(rr.avg_ticket_price, 0) AS avg_ticket_price
        FROM route_summary rs
        LEFT JOIN route_rev rr ON rs.flight_id = rr.flight_id AND rs.cabin_class = rr.cabin_class
        ORDER BY {order_col}
        LIMIT {n}
    """).fetchall()

    routes = []
    flight_ids = []
    for r in top_routes:
        fid = r[0]
        flight_ids.append(fid)
        routes.append({
            "flight_id": fid,
            "cabin_class": r[1],
            "total_pax": _num(r[2]),
            "capacity": int(r[3]) if r[3] else 0,
            "final_lf": round(float(r[4]) * 100, 1) if r[4] else 0,
            "avg_lf": round(float(r[5]) * 100, 1) if r[5] else 0,
            "distance_km": int(r[6]) if r[6] else 0,
            "region": r[7] or "",
            "dep_year": r[8],
            "dep_month": r[9],
            "dep_dow": r[10],
            "dep_hour": r[11],
            "ff_gold_pct": round(float(r[12]) * 100, 1) if r[12] else 0,
            "ff_elite_pct": round(float(r[13]) * 100, 1) if r[13] else 0,
            "total_ticket_rev": _num(r[14]),
            "total_anc_rev": _num(r[15]),
            "total_rev": _num(r[16]),
            "avg_ticket_price": round(float(r[17]), 2) if r[17] else 0,
        })

    # 2) DTD curves for top routes (load factor & revenue over DTD)
    if flight_ids:
        id_list = ",".join(f"'{fid}'" for fid in flight_ids)

        # LF curves from training data
        lf_curves_raw = con.execute(f"""
            SELECT flight_id, dtd, load_factor, pax_sold_cum, remaining_seats, y_pax_sold_today
            FROM read_parquet('{tp}')
            WHERE flight_id IN ({id_list})
            ORDER BY flight_id, dtd DESC
        """).fetchall()

        lf_curves = {}
        for row in lf_curves_raw:
            fid = row[0]
            if fid not in lf_curves:
                lf_curves[fid] = []
            lf_curves[fid].append({
                "dtd": int(row[1]),
                "load_factor": round(float(row[2]) * 100, 2) if row[2] else 0,
                "pax_cum": _num(row[3]),
                "remaining": _num(row[4]),
                "pax_today": _num(row[5]),
            })

        # Revenue curves from snapshot v2
        rev_curves_raw = con.execute(f"""
            SELECT flight_id, dtd,
                   ticket_rev_cum, anc_rev_cum,
                   ticket_rev_today, anc_rev_today,
                   CASE WHEN pax_sold_today > 0
                       THEN ticket_rev_today / pax_sold_today ELSE 0 END AS unit_price
            FROM read_parquet('{sp}')
            WHERE flight_id IN ({id_list})
            ORDER BY flight_id, dtd DESC
        """).fetchall()

        rev_curves = {}
        for row in rev_curves_raw:
            fid = row[0]
            if fid not in rev_curves:
                rev_curves[fid] = []
            rev_curves[fid].append({
                "dtd": int(row[1]),
                "ticket_rev_cum": _num(row[2]),
                "anc_rev_cum": _num(row[3]),
                "ticket_rev_today": _num(row[4]),
                "anc_rev_today": _num(row[5]),
                "unit_price": round(float(row[6]), 2) if row[6] else 0,
            })
    else:
        lf_curves = {}
        rev_curves = {}

    # 3) Available cabins for filter
    cabins = con.execute(f"""
        SELECT DISTINCT LOWER(cabin_class) FROM read_parquet('{tp}')
        WHERE cabin_class IS NOT NULL ORDER BY 1
    """).fetchall()

    con.close()

    return jsonify({
        "routes": routes,
        "lf_curves": lf_curves,
        "rev_curves": rev_curves,
        "filters": {
            "cabins": [c[0] for c in cabins],
            "sort_options": ["total_pax", "final_lf", "total_rev"],
        }
    })


# ─── EVENT / SENTIMENT ANALYSIS API ──────────────────────
@app.route("/api/events")
def api_events():
    """EDA: Event/sentiment category analysis from tagged training data."""
    con = get_con()
    tp = TRAINING_PARQUET

    # Check if event tags exist
    try:
        cols = con.execute(f"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{tp}'))").fetchall()
        col_names = [c[0] for c in cols]
        if 'primary_event' not in col_names:
            con.close()
            return jsonify({"error": "Event tags not found. Run add_event_tags.py first."}), 404
    except Exception as e:
        con.close()
        return jsonify({"error": str(e)}), 500

    tag_cols = [c for c in col_names if c.startswith('tag_')]

    # 1) Per-tag summary (all 15 tags)
    tag_stats = []
    for tag in tag_cols:
        name = tag.replace('tag_', '')
        r = con.execute(f"""
            SELECT
                SUM(CASE WHEN {tag} THEN 1 ELSE 0 END) AS cnt,
                AVG(CASE WHEN {tag} THEN y_pax_sold_today END) AS avg_pax,
                AVG(CASE WHEN {tag} THEN load_factor END) AS avg_lf,
                AVG(CASE WHEN NOT {tag} THEN y_pax_sold_today END) AS baseline_pax,
                AVG(CASE WHEN NOT {tag} THEN load_factor END) AS baseline_lf
            FROM read_parquet('{tp}')
        """).fetchone()
        total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{tp}')").fetchone()[0]
        tag_stats.append({
            "name": name,
            "label": name.replace('_', ' ').title(),
            "count": int(r[0]),
            "pct": round(r[0] / total * 100, 1),
            "avg_pax": round(float(r[1]), 3) if r[1] else 0,
            "avg_lf": round(float(r[2]) * 100, 1) if r[2] else 0,
            "baseline_pax": round(float(r[3]), 3) if r[3] else 0,
            "baseline_lf": round(float(r[4]) * 100, 1) if r[4] else 0,
            "pax_lift": round(float(r[1]) - float(r[3]), 3) if r[1] and r[3] else 0,
            "lf_lift": round((float(r[2]) - float(r[4])) * 100, 1) if r[2] and r[4] else 0,
        })

    # 2) Primary event distribution
    primary_dist = con.execute(f"""
        SELECT primary_event, COUNT(*) AS cnt,
               AVG(y_pax_sold_today) AS avg_pax,
               AVG(load_factor) AS avg_lf
        FROM read_parquet('{tp}')
        GROUP BY primary_event
        ORDER BY cnt DESC
    """).fetchall()

    primary_events = []
    for r in primary_dist:
        primary_events.append({
            "event": r[0],
            "label": r[0].replace('_', ' ').title(),
            "count": int(r[1]),
            "avg_pax": round(float(r[2]), 3) if r[2] else 0,
            "avg_lf": round(float(r[3]) * 100, 1) if r[3] else 0,
        })

    # 3) Monthly breakdown by primary event
    monthly = con.execute(f"""
        SELECT dep_year, dep_month, primary_event,
               SUM(y_pax_sold_today) AS total_pax,
               AVG(load_factor) AS avg_lf,
               COUNT(*) AS cnt
        FROM read_parquet('{tp}')
        GROUP BY dep_year, dep_month, primary_event
        ORDER BY dep_year, dep_month, primary_event
    """).fetchall()

    monthly_data = []
    for r in monthly:
        monthly_data.append({
            "year": r[0], "month": r[1], "event": r[2],
            "total_pax": _num(r[3]),
            "avg_lf": round(float(r[4]) * 100, 1) if r[4] else 0,
            "count": int(r[5]),
        })

    con.close()

    return jsonify({
        "tag_stats": tag_stats,
        "primary_events": primary_events,
        "monthly": monthly_data,
    })


# ─── RISK / OPPORTUNITY INDEX API ──────────────────────
@app.route("/api/risk-index")
def api_risk_index():
    """Flight risk/opportunity index with pricing action categories."""
    con = get_con()
    sp = PARQUET_PATH
    cabin_filter = request.args.get("cabin", "").strip().lower()
    cabin_where = f"AND LOWER(s.cabin_class) = '{cabin_filter}'" if cabin_filter else ""

    # Get latest snapshot per flight (DTD=0 or minimum DTD available)
    flights = con.execute(f"""
        WITH latest AS (
            SELECT flight_id, cabin_class, MIN(dtd) AS min_dtd
            FROM read_parquet('{sp}')
            WHERE dtd IS NOT NULL
            GROUP BY flight_id, cabin_class
        ),
        flight_data AS (
            SELECT
                s.flight_id,
                s.cabin_class,
                s.dtd,
                s.pax_sold_cum,
                m.capacity,
                CASE WHEN m.capacity > 0
                    THEN s.pax_sold_cum * 1.0 / m.capacity ELSE 0 END AS load_factor,
                GREATEST(m.capacity - s.pax_sold_cum, 0) AS remaining_seats,
                s.ticket_rev_cum,
                s.anc_rev_cum,
                s.ticket_rev_cum + s.anc_rev_cum AS total_rev_cum,
                CASE WHEN s.pax_sold_cum > 0
                    THEN (s.ticket_rev_cum + s.anc_rev_cum) / s.pax_sold_cum
                    ELSE 0 END AS rev_per_pax,
                m.region,
                m.departure_airport,
                m.arrival_airport,
                m.distance_km,
                s.pax_last_7d,
                s.pax_sold_today
            FROM read_parquet('{sp}') s
            INNER JOIN latest l ON s.flight_id = l.flight_id
                AND s.cabin_class = l.cabin_class AND s.dtd = l.min_dtd
            LEFT JOIN read_parquet('{METADATA_PATH}') m
                ON s.flight_id = m.flight_id AND s.cabin_class = m.cabin_class
            WHERE 1=1 {cabin_where}
        )
        SELECT * FROM flight_data
        ORDER BY flight_id, cabin_class
    """).fetchall()

    results = []
    categories = {"price_increase": [], "price_decrease": [], "cancel_risk": []}

    for r in flights:
        fid, cabin, dtd, pax_cum, capacity = r[0], r[1], r[2], r[3] or 0, r[4] or 1
        lf = float(r[5]) if r[5] else 0
        remaining = int(r[6]) if r[6] else 0
        total_rev = float(r[9]) if r[9] else 0
        rev_per_pax = float(r[10]) if r[10] else 0
        region = r[11] or ""
        dep_ap, arr_ap = r[12] or "", r[13] or ""
        distance = float(r[14]) if r[14] else 0
        pax_7d = float(r[15]) if r[15] else 0
        pax_today = float(r[16]) if r[16] else 0

        # === RISK/OPPORTUNITY SCORING ===
        # 1. DTD urgency (0-25): closer to departure = more urgent
        if dtd is None:
            dtd_score = 12.5
        elif dtd <= 3:
            dtd_score = 25
        elif dtd <= 7:
            dtd_score = 20
        elif dtd <= 14:
            dtd_score = 15
        elif dtd <= 30:
            dtd_score = 10
        else:
            dtd_score = 5

        # 2. Load factor score (0-25): low LF = more risk
        lf_score = max(0, 25 - lf * 25)  # LF=0 → 25, LF=1 → 0

        # 3. Revenue momentum (0-25): recent booking activity
        momentum = min(pax_7d / max(capacity, 1), 1.0)
        momentum_score = (1 - momentum) * 25  # low activity = high risk

        # 4. Capacity waste (0-25): empty seats cost money
        waste = remaining / max(capacity, 1)
        waste_score = waste * 25  # more empty = more risk

        risk_score = round(dtd_score + lf_score + momentum_score + waste_score, 1)
        opp_score = round(100 - risk_score, 1)

        # Revenue potential = remaining seats × avg revenue per pax
        rev_potential = round(remaining * rev_per_pax, 2)

        # === PRICING CATEGORY ===
        if lf >= 0.75 and (dtd is None or dtd >= 7):
            category = "price_increase"
            action = "Fiyat Artırım Fırsatı"
            reason = f"Doluluk %{lf*100:.0f}, {dtd or 0} gün kala yüksek talep"
        elif lf < 0.40 and (dtd is not None and dtd <= 14):
            category = "cancel_risk"
            action = "İptal Riski / Zarar Potansiyeli"
            reason = f"Doluluk %{lf*100:.0f}, {dtd} gün kala çok düşük"
        elif lf < 0.65 and (dtd is not None and dtd <= 30):
            category = "price_decrease"
            action = "Fiyat Düşürme Gerekli"
            reason = f"Doluluk %{lf*100:.0f}, {dtd} gün kala doluluk yakalanmalı"
        elif lf >= 0.65:
            category = "price_increase"
            action = "Fiyat Artırım Fırsatı"
            reason = f"İyi doluluk (%{lf*100:.0f}), fiyat optimize edilebilir"
        else:
            category = "price_decrease"
            action = "Fiyat Düşürme Gerekli"
            reason = f"Doluluk %{lf*100:.0f}, talep çekilmeli"

        flight_info = {
            "flight_id": fid,
            "cabin": cabin,
            "route": f"{dep_ap}-{arr_ap}",
            "region": region,
            "dtd": dtd,
            "pax_cum": int(pax_cum),
            "capacity": int(capacity),
            "load_factor": round(lf * 100, 1),
            "remaining_seats": remaining,
            "total_rev": round(total_rev, 2),
            "rev_potential": rev_potential,
            "rev_per_pax": round(rev_per_pax, 2),
            "risk_score": risk_score,
            "opp_score": opp_score,
            "category": category,
            "action": action,
            "reason": reason,
            "pax_7d": int(pax_7d),
            "pax_today": int(pax_today),
            "distance_km": int(distance),
        }
        results.append(flight_info)
        categories[category].append(flight_info)

    # Sort each category by risk score
    for cat in categories:
        categories[cat].sort(key=lambda x: x["risk_score"], reverse=True)

    # Summary stats
    summary = {
        "total_flights": len(results),
        "price_increase": {
            "count": len(categories["price_increase"]),
            "avg_lf": round(sum(f["load_factor"] for f in categories["price_increase"]) / max(len(categories["price_increase"]), 1), 1),
            "total_rev_potential": round(sum(f["rev_potential"] for f in categories["price_increase"]), 0),
        },
        "price_decrease": {
            "count": len(categories["price_decrease"]),
            "avg_lf": round(sum(f["load_factor"] for f in categories["price_decrease"]) / max(len(categories["price_decrease"]), 1), 1),
            "total_rev_potential": round(sum(f["rev_potential"] for f in categories["price_decrease"]), 0),
        },
        "cancel_risk": {
            "count": len(categories["cancel_risk"]),
            "avg_lf": round(sum(f["load_factor"] for f in categories["cancel_risk"]) / max(len(categories["cancel_risk"]), 1), 1),
            "total_rev_potential": round(sum(f["rev_potential"] for f in categories["cancel_risk"]), 0),
        },
        "avg_risk_score": round(sum(f["risk_score"] for f in results) / max(len(results), 1), 1),
    }

    # Available cabins
    cabins = con.execute(f"""
        SELECT DISTINCT LOWER(cabin_class) FROM read_parquet('{sp}')
        WHERE cabin_class IS NOT NULL ORDER BY 1
    """).fetchall()

    con.close()

    return jsonify({
        "flights": results[:200],  # limit response size
        "categories": {k: v[:50] for k, v in categories.items()},
        "summary": summary,
        "filters": {"cabins": [c[0] for c in cabins]},
    })


if __name__ == "__main__":
    v_label = "V2 (ticket + ancillary)" if USE_V2 else "V1 (legacy)"
    fc_label = "ON" if FORECAST_READY else "OFF"
    print(f"\nFlight Snapshot Dashboard -- {v_label} | Forecast: {fc_label}")
    print(f"   Snapshot: {PARQUET_PATH}")
    print(f"   Metadata: {METADATA_PATH}")
    print(f"   URL: http://localhost:5001\n")
    app.run(debug=True, port=5001)


