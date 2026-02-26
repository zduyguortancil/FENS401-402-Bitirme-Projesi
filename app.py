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


if __name__ == "__main__":
    v_label = "V2 (ticket + ancillary)" if USE_V2 else "V1 (legacy)"
    fc_label = "ON" if FORECAST_READY else "OFF"
    print(f"\nFlight Snapshot Dashboard -- {v_label} | Forecast: {fc_label}")
    print(f"   Snapshot: {PARQUET_PATH}")
    print(f"   Metadata: {METADATA_PATH}")
    print(f"   URL: http://localhost:5001\n")
    app.run(debug=True, port=5001)
