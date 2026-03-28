"""
Flight Snapshot Dashboard — Sprint 4
Flask + DuckDB backend.
V2 parquet (ticket + ancillary revenue) + metadata lookup.
Demand forecast via two-stage XGBoost (classifier + regressor).
Sentiment Intelligence via HuggingFace NLP (integrated).
"""

import os
import math
import json
import threading
import numpy as np
from flask import Flask, render_template, jsonify, request
import duckdb

# ─── .env dosyasından ortam değişkenlerini oku ───────────
def _load_dotenv():
    # dashboard/.env ve project root/.env — ikisini de dene
    base = os.path.dirname(os.path.abspath(__file__))
    for env_path in [os.path.join(base, ".env"), os.path.join(os.path.dirname(base), ".env")]:
        if not os.path.exists(env_path):
            continue
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if not os.environ.get(key):
                    os.environ[key] = val

_load_dotenv()

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PROJECT_DIR = os.path.dirname(BASE_DIR).replace("\\", "/")
DATA_DIR = f"{PROJECT_DIR}/data"

# ─── FILE SELECTION (v2 default, v1 fallback) ────────────
SNAPSHOT_V2 = f"{DATA_DIR}/raw/flight_snapshot_v2.parquet"
SNAPSHOT_V1 = f"{DATA_DIR}/raw/flight_snapshot.parquet"
METADATA_PATH = f"{DATA_DIR}/processed/flight_metadata.parquet"

if os.path.exists(SNAPSHOT_V2.replace("/", os.sep)):
    PARQUET_PATH = SNAPSHOT_V2
    USE_V2 = True
else:
    PARQUET_PATH = SNAPSHOT_V1
    USE_V2 = False

# ─── TFT DEMAND FORECAST (Route-Daily) ───────────────────
import pandas as pd
from datetime import datetime, timedelta

TFT_DATA_PATH = f"{DATA_DIR}/processed/tft_route_daily.parquet"
TFT_PRED_PATH = f"{DATA_DIR}/processed/tft_predictions_indexed.parquet"
FORECAST_READY = False
TFT_DATA = None
TFT_PRED = None
TFT_METRICS = {
    "test_mae": 14.03, "test_corr": 0.991,  # Kaggle test set (2026 Q2-Q4), 50 epoch
}

try:
    if os.path.exists(TFT_DATA_PATH.replace('/', os.sep)):
        TFT_DATA = pd.read_parquet(TFT_DATA_PATH.replace('/', os.sep))
        TFT_DATA["dep_date"] = pd.to_datetime(TFT_DATA["dep_date"])
        FORECAST_READY = True
        print(f"[Forecast] TFT route-daily loaded: {len(TFT_DATA):,} rows, "
              f"{TFT_DATA['entity_id'].nunique()} entities, MAE={TFT_METRICS['test_mae']}")
    else:
        print("[Forecast] tft_route_daily.parquet not found, forecast disabled")
    if os.path.exists(TFT_PRED_PATH.replace('/', os.sep)):
        TFT_PRED = pd.read_parquet(TFT_PRED_PATH.replace('/', os.sep))
        TFT_PRED["dep_date"] = pd.to_datetime(TFT_PRED["dep_date"])
        print(f"[Forecast] TFT predictions loaded: {len(TFT_PRED):,} rows, "
              f"{TFT_PRED['entity_id'].nunique()} entities")
    else:
        print("[Forecast] tft_predictions_indexed.parquet not found, using YoY baseline")
except Exception as e:
    print(f"[Forecast] Failed to load TFT data: {e}")


# ─── PICKUP XGBOOST (Flight-Level Demand) ─────────────────
import xgboost as xgb

PICKUP_MODEL_PATH = f"{DATA_DIR}/models/pickup_xgb.json"
PICKUP_FEATURES_PATH = f"{DATA_DIR}/models/pickup_feature_list.json"
PICKUP_MASTER_PATH = f"{DATA_DIR}/processed/pickup_master.parquet"
PICKUP_METRICS_PATH = f"{PROJECT_DIR}/reports/pickup_xgb_metrics.json"

PICKUP_READY = False
PICKUP_MODEL = None
PICKUP_FEATURES = None
PICKUP_METRICS = {}

try:
    if os.path.exists(PICKUP_MODEL_PATH.replace('/', os.sep)):
        PICKUP_MODEL = xgb.Booster()
        PICKUP_MODEL.load_model(PICKUP_MODEL_PATH.replace('/', os.sep))
        with open(PICKUP_FEATURES_PATH.replace('/', os.sep), 'r') as f:
            PICKUP_FEATURES = json.load(f)["features"]
        if os.path.exists(PICKUP_METRICS_PATH.replace('/', os.sep)):
            with open(PICKUP_METRICS_PATH.replace('/', os.sep), 'r') as f:
                PICKUP_METRICS = json.load(f)
        PICKUP_READY = True
        print(f"[Pickup] Model loaded: {len(PICKUP_FEATURES)} features, "
              f"MAE={PICKUP_METRICS.get('mae', '?')}, WAPE={PICKUP_METRICS.get('wape', '?')}%")
    else:
        print(f"[Pickup] pickup_xgb.json not found, pickup disabled")
except Exception as e:
    print(f"[Pickup] Failed to load: {e}")


# ─── TWO-STAGE XGBOOST (Daily Pax Sold) ──────────────────
import joblib

TWOSTAGE_CLF_PATH = f"{DATA_DIR}/models/xgb_demand_classifier.pkl"
TWOSTAGE_REG_PATH = f"{DATA_DIR}/models/xgb_demand_regressor.pkl"
TWOSTAGE_FEAT_PATH = f"{DATA_DIR}/models/feature_list.json"
TWOSTAGE_METRICS_PATH = f"{PROJECT_DIR}/reports/demand_metrics.json"

TWOSTAGE_READY = False
TWOSTAGE_CLF = None
TWOSTAGE_REG = None
TWOSTAGE_FEATURES = None
TWOSTAGE_METRICS = {}

try:
    if os.path.exists(TWOSTAGE_CLF_PATH.replace('/', os.sep)):
        TWOSTAGE_CLF = joblib.load(TWOSTAGE_CLF_PATH.replace('/', os.sep))
        TWOSTAGE_REG = joblib.load(TWOSTAGE_REG_PATH.replace('/', os.sep))
        with open(TWOSTAGE_FEAT_PATH.replace('/', os.sep), 'r') as f:
            TWOSTAGE_FEATURES = json.load(f)["features"]
        if os.path.exists(TWOSTAGE_METRICS_PATH.replace('/', os.sep)):
            with open(TWOSTAGE_METRICS_PATH.replace('/', os.sep), 'r') as f:
                TWOSTAGE_METRICS = json.load(f)
        TWOSTAGE_READY = True
        print(f"[TwoStage] Model loaded: {len(TWOSTAGE_FEATURES)} features, "
              f"MAE={TWOSTAGE_METRICS.get('two_stage_model', {}).get('mae', '?')}, "
              f"AUC={TWOSTAGE_METRICS.get('two_stage_model', {}).get('auc_sale_classifier', '?')}")
    else:
        print("[TwoStage] xgb_demand_classifier.pkl not found, two-stage disabled")
except Exception as e:
    print(f"[TwoStage] Failed to load: {e}")


# ─── SENTIMENT INTELLIGENCE v2 (GDELT + DeBERTa) ────────
SENTIMENT_READY = False
_SENT_CACHE = {"data": None, "loading": False, "last_update": None}
try:
    from sentiment import CITIES as SENT_CITIES, AIRPORT_TO_CITY
    from sentiment import init_db as _sent_init_db, load_cached_scores, start_scheduler

    _sent_init_db()

    # SQLite'tan son skorlari yukle (aninda hazir)
    _cached = load_cached_scores(max_age_hours=2)
    if _cached:
        _SENT_CACHE["data"] = _cached
        print(f"[Sentiment] Loaded {len(_cached)} cities from cache")

    # Arka plan scheduler baslat (GDELT fetch + DeBERTa classify)
    start_scheduler(_SENT_CACHE)
    SENTIMENT_READY = True
    print("[Sentiment] v2 ready (GDELT + DeBERTa)")
except Exception as e:
    print(f"[Sentiment] Module not available: {e}")
    import traceback; traceback.print_exc()
    SENT_CITIES = {}


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
    """TFT route-daily demand forecast for a flight.
    Maps flight_id -> route+cabin, returns daily demand time series
    with actual (current year) and seasonal baseline (previous year).
    """
    if not FORECAST_READY:
        return jsonify({"error": "Forecast model not loaded"}), 503

    cabin = request.args.get("cabin", "").strip().lower() or None
    window = int(request.args.get("window", 60))  # days around dep_date

    con = get_con()

    # ── 1. Look up flight metadata to get route + dep_date ──
    meta_rows = con.execute(f"""
        SELECT DISTINCT
            departure_airport, arrival_airport, departure_datetime,
            cabin_class, capacity, region, distance_km, flight_time_min,
            flight_number
        FROM read_parquet('{METADATA_PATH}')
        WHERE flight_id = $1
    """, [flight_id]).fetchall()
    con.close()

    if not meta_rows:
        return jsonify({"error": "Flight not found", "rows": []})

    # Pick cabin: use requested or first available
    meta_cols = ["dep_apt", "arr_apt", "dep_dt", "cabin_class", "capacity",
                 "region", "distance_km", "flight_time_min", "flight_number"]
    metas = [dict(zip(meta_cols, r)) for r in meta_rows]

    if cabin:
        metas = [m for m in metas if m["cabin_class"].lower() == cabin]
    if not metas:
        return jsonify({"error": "Cabin not found", "rows": []})

    cabins_result = {}

    for meta in metas:
        cab = meta["cabin_class"].lower()
        route = f"{meta['dep_apt']}_{meta['arr_apt']}"
        entity_id = f"{route}_{cab}"
        dep_dt = meta["dep_dt"]
        if isinstance(dep_dt, str):
            dep_dt = datetime.fromisoformat(dep_dt)
        dep_date = pd.Timestamp(dep_dt).normalize()
        cap = meta["capacity"] or (300 if cab == "economy" else 49)

        # ── 2. Get route demand data from TFT dataset ──
        entity_data = TFT_DATA[TFT_DATA["entity_id"] == entity_id].copy()
        if entity_data.empty:
            continue

        # Window: [dep_date - window, dep_date] — only show days up to departure
        date_start = dep_date - pd.Timedelta(days=window)
        date_end = dep_date
        windowed = entity_data[
            (entity_data["dep_date"] >= date_start) &
            (entity_data["dep_date"] <= date_end)
        ].sort_values("dep_date")

        # ── 3a. TFT Model Predictions (varsa) ──
        tft_lookup = {}
        tft_q10_lookup = {}
        tft_q90_lookup = {}
        has_tft = False
        if TFT_PRED is not None:
            pred_entity = TFT_PRED[TFT_PRED["entity_id"] == entity_id]
            if not pred_entity.empty:
                pred_window = pred_entity[
                    (pred_entity["dep_date"] >= date_start) &
                    (pred_entity["dep_date"] <= date_end)
                ]
                for _, pr in pred_window.iterrows():
                    ds = pr["dep_date"].strftime("%Y-%m-%d")
                    tft_lookup[ds] = float(pr["predicted"])
                    if "pred_q10" in pr and pd.notna(pr["pred_q10"]):
                        tft_q10_lookup[ds] = float(pr["pred_q10"])
                    if "pred_q90" in pr and pd.notna(pr["pred_q90"]):
                        tft_q90_lookup[ds] = float(pr["pred_q90"])
                has_tft = len(tft_lookup) > 0

        # ── 3b. YoY Baseline (yedek) ──
        baseline_start = date_start - pd.Timedelta(days=365)
        baseline_end = date_end - pd.Timedelta(days=365)
        baseline = entity_data[
            (entity_data["dep_date"] >= baseline_start) &
            (entity_data["dep_date"] <= baseline_end)
        ].sort_values("dep_date")

        baseline_lookup = {}
        for _, row in baseline.iterrows():
            shifted = row["dep_date"] + pd.Timedelta(days=365)
            baseline_lookup[shifted.strftime("%Y-%m-%d")] = float(row["total_pax"])

        # ── 4. Build response rows ──
        rows = []
        total_actual = 0
        total_forecast = 0
        dep_date_pax = None

        for _, row in windowed.iterrows():
            d = row["dep_date"]
            date_str = d.strftime("%Y-%m-%d")
            actual_pax = float(row["total_pax"])
            n_fl = int(row["n_flights"]) if row["n_flights"] else 1

            tft_pred = tft_lookup.get(date_str)
            yoy_base = baseline_lookup.get(date_str)
            forecast_pax = tft_pred if tft_pred is not None else yoy_base

            pax_per_flight = actual_pax / n_fl if n_fl > 0 else actual_pax
            load_factor = pax_per_flight / cap if cap > 0 else 0

            row_data = {
                "date": date_str,
                "actual_pax": round(actual_pax, 1),
                "forecast_pax": round(forecast_pax, 1) if forecast_pax is not None else None,
                "tft_forecast": round(tft_pred, 1) if tft_pred is not None else None,
                "pred_q10": round(tft_q10_lookup[date_str], 1) if date_str in tft_q10_lookup else None,
                "pred_q90": round(tft_q90_lookup[date_str], 1) if date_str in tft_q90_lookup else None,
                "yoy_baseline": round(yoy_base, 1) if yoy_base is not None else None,
                "pax_per_flight": round(pax_per_flight, 1),
                "load_factor": round(load_factor, 4),
                "n_flights": n_fl,
                "avg_fare": round(float(row["avg_fare"]), 2) if row["avg_fare"] else None,
                "is_special": bool(row["is_special_period"]),
                "special_period": row["special_period"] if row["special_period"] != "none" else None,
            }
            rows.append(row_data)

            total_actual += actual_pax
            if forecast_pax is not None:
                total_forecast += forecast_pax

            if date_str == dep_date.strftime("%Y-%m-%d"):
                dep_date_pax = actual_pax

        # ── 5. Compute accuracy metrics ──
        paired = [(r["actual_pax"], r["forecast_pax"])
                  for r in rows if r["forecast_pax"] is not None]
        if paired:
            actuals_arr = np.array([p[0] for p in paired])
            forecasts_arr = np.array([p[1] for p in paired])
            window_mae = float(np.mean(np.abs(actuals_arr - forecasts_arr)))
            sum_act = float(np.sum(np.abs(actuals_arr)))
            window_wape = float(np.sum(np.abs(actuals_arr - forecasts_arr)) / sum_act * 100) if sum_act > 0 else None
        else:
            window_mae = None
            window_wape = None

        cabins_result[cab] = {
            "rows": rows,
            "metadata": {
                "entity_id": entity_id,
                "route": route,
                "cabin_class": cab,
                "flight_id": flight_id,
                "flight_number": meta["flight_number"],
                "dep_date": dep_date.strftime("%Y-%m-%d"),
                "region": meta["region"],
                "distance_km": meta["distance_km"],
                "flight_time_min": meta["flight_time_min"],
                "capacity_per_flight": cap,
                "dep_date_pax": dep_date_pax,
                "dep_date_pax_per_flight": round(dep_date_pax / (rows[0]["n_flights"] if rows else 1), 1) if dep_date_pax else None,
                "dep_date_load_factor": round((dep_date_pax / (rows[0]["n_flights"] if rows else 1)) / cap, 4) if dep_date_pax and cap else None,
                "window_days": window,
                "total_actual": round(total_actual, 1),
                "total_forecast": round(total_forecast, 1),
                "forecast_source": "tft_model" if has_tft else "yoy_baseline",
                "window_mae": round(window_mae, 2) if window_mae is not None else None,
                "window_wape": round(window_wape, 1) if window_wape is not None else None,
                "tft_test_mae": TFT_METRICS.get("test_mae"),
                "tft_test_corr": TFT_METRICS.get("test_corr"),
            }
        }

    if not cabins_result:
        return jsonify({"error": "No route data found for this flight"})

    # If single cabin requested, return flat; otherwise return all
    if cabin and cabin in cabins_result:
        return jsonify(cabins_result[cabin])
    return jsonify(cabins_result)


# ─── TWO-STAGE DEMAND FORECAST API ────────────────────────
@app.route("/api/demand/<path:flight_id>")
def api_demand(flight_id):
    """Two-Stage XGBoost: predict daily pax sold at each DTD for a flight.
    Stage 1 (classifier): P(sale > 0)
    Stage 2 (regressor): E[pax | sale > 0]
    Combined: P * E[pax]
    """
    if not TWOSTAGE_READY:
        return jsonify({"error": "Two-stage demand model not loaded"}), 503

    cabin = request.args.get("cabin", "").strip().lower() or None
    con = get_con()

    meta_rows = con.execute(f"""
        SELECT departure_airport, arrival_airport, departure_datetime,
               cabin_class, capacity, region, distance_km, flight_time_min,
               flight_number
        FROM read_parquet('{METADATA_PATH}')
        WHERE flight_id = $1
        LIMIT 2
    """, [flight_id]).fetchall()
    con.close()

    if not meta_rows:
        return jsonify({"error": "Flight not found"})

    result = {}

    for row in meta_rows:
        cab = row[3].lower()
        if cabin and cab != cabin:
            continue
        cap = int(row[4]) if row[4] else (300 if cab == "economy" else 49)
        dep_dt = row[2]
        dep_year = dep_dt.year if hasattr(dep_dt, 'year') else int(str(dep_dt)[:4])
        region = row[5] or "Europe"
        dist_km = float(row[6]) if row[6] else 3000
        ft_min = float(row[7]) if row[7] else 300

        # Read snapshot data for this flight
        snap_con = get_con()
        snap_data = snap_con.execute(f"""
            SELECT dtd, pax_sold_cum, pax_last_7d, ff_gold_pct, ff_elite_pct
            FROM read_parquet('{PARQUET_PATH}')
            WHERE flight_id = $1 AND LOWER(cabin_class) = $2
            ORDER BY dtd DESC
        """, [flight_id, cab]).fetchall()
        snap_con.close()

        if not snap_data:
            continue

        # Build feature vectors for each DTD point
        rows_out = []
        for snap in snap_data:
            dtd_val = float(snap[0]) if snap[0] is not None else 0
            pax_cum = float(snap[1]) if snap[1] is not None else 0
            pax_7d = float(snap[2]) if snap[2] is not None else 0
            ff_gold = float(snap[3]) if snap[3] is not None else 0
            ff_elite = float(snap[4]) if snap[4] is not None else 0
            remaining = max(cap - pax_cum, 0)
            lf = pax_cum / cap if cap > 0 else 0

            # DTD bucket (0-6 encoding)
            dtd_bucket = min(int(dtd_val // 30), 6)

            # Build feature dict matching TWOSTAGE_FEATURES order
            feat = {
                'dtd': dtd_val,
                'pax_sold_cum': pax_cum,
                'pax_last_7d': pax_7d,
                'capacity': float(cap),
                'remaining_seats': remaining,
                'load_factor': lf,
                'distance_km': dist_km,
                'flight_time_min': ft_min,
                'dep_year': float(dep_year),
                'dep_month': float(dep_dt.month) if hasattr(dep_dt, 'month') else 1.0,
                'dep_dow': float(dep_dt.weekday()) if hasattr(dep_dt, 'weekday') else 0.0,
                'dep_hour': float(dep_dt.hour) if hasattr(dep_dt, 'hour') else 12.0,
                'ff_gold_pct': ff_gold,
                'ff_elite_pct': ff_elite,
                'cabin_class_business': 1.0 if cab == 'business' else 0.0,
                'cabin_class_economy': 1.0 if cab == 'economy' else 0.0,
                'cabin_class_nan': 0.0,
                'region_Africa': 1.0 if region == 'Africa' else 0.0,
                'region_Americas': 1.0 if region == 'Americas' else 0.0,
                'region_Asia': 1.0 if region == 'Asia' else 0.0,
                'region_Europe': 1.0 if region == 'Europe' else 0.0,
                'region_Middle East': 1.0 if region == 'Middle East' else 0.0,
                'region_nan': 0.0,
            }
            # DTD bucket one-hot
            for b in range(7):
                feat[f'dtd_bucket_{float(b)}'] = 1.0 if dtd_bucket == b else 0.0
            feat['dtd_bucket_nan'] = 0.0

            X = np.array([[feat.get(f, 0.0) for f in TWOSTAGE_FEATURES]], dtype=np.float32)

            # Stage 1: P(sale > 0)
            p_sale = float(TWOSTAGE_CLF.predict_proba(X)[0, 1])
            # Stage 2: E[pax | sale > 0]
            e_pax = max(float(TWOSTAGE_REG.predict(X)[0]), 0)
            # Combined
            predicted_pax = round(p_sale * e_pax, 2)

            rows_out.append({
                "dtd": int(dtd_val),
                "pax_sold_cum": int(pax_cum),
                "load_factor": round(lf, 4),
                "p_sale": round(p_sale, 4),
                "e_pax_given_sale": round(e_pax, 2),
                "predicted_daily_pax": predicted_pax,
                "remaining_seats": int(remaining),
            })

        # Model metrics
        ts_metrics = TWOSTAGE_METRICS.get("two_stage_model", {})

        result[cab] = {
            "rows": rows_out,
            "metadata": {
                "flight_id": flight_id,
                "flight_number": row[8],
                "cabin_class": cab,
                "departure_airport": row[0],
                "arrival_airport": row[1],
                "dep_date": str(dep_dt)[:10],
                "region": region,
                "capacity": cap,
                "model": "Two-Stage XGBoost (Classifier + Regressor)",
                "model_mae": ts_metrics.get("mae"),
                "model_auc": ts_metrics.get("auc_sale_classifier"),
            }
        }

    if not result:
        return jsonify({"error": "No data found"})
    if cabin and cabin in result:
        return jsonify(result[cabin])
    return jsonify(result)


# ─── PICKUP FORECAST API ─────────────────────────────────
@app.route("/api/pickup/<path:flight_id>")
def api_pickup(flight_id):
    """XGBoost pickup model: predict remaining_pax at each DTD for a flight.
    2025 flights: return actual data only.
    2026 flights: return model predictions.
    """
    if not PICKUP_READY:
        return jsonify({"error": "Pickup model not loaded"}), 503

    cabin = request.args.get("cabin", "").strip().lower() or None
    con = get_con()

    # Get flight metadata
    meta_row = con.execute(f"""
        SELECT departure_airport, arrival_airport, departure_datetime,
               cabin_class, capacity, region, distance_km, flight_time_min,
               flight_number
        FROM read_parquet('{METADATA_PATH}')
        WHERE flight_id = $1
        LIMIT 2
    """, [flight_id]).fetchall()

    if not meta_row:
        con.close()
        return jsonify({"error": "Flight not found"})

    # Determine year
    dep_dt = meta_row[0][2]
    dep_year = dep_dt.year if hasattr(dep_dt, 'year') else int(str(dep_dt)[:4])

    result = {}

    for row in meta_row:
        cab = row[3].lower()
        if cabin and cab != cabin:
            continue
        cap = int(row[4]) if row[4] else (300 if cab == "economy" else 49)

        # Read this flight's DTD data from pickup_master
        flight_data = con.execute(f"""
            SELECT *
            FROM read_parquet('{PICKUP_MASTER_PATH}')
            WHERE flight_id = $1 AND LOWER(cabin_class) = $2
            ORDER BY dtd DESC
        """, [flight_id, cab]).fetchdf()

        if flight_data.empty:
            continue

        # Convert decimal columns to float
        for col in flight_data.columns:
            if flight_data[col].dtype == object:
                try:
                    flight_data[col] = flight_data[col].astype(float)
                except (ValueError, TypeError):
                    pass

        actual_remaining = flight_data['remaining_pax'].values.astype(float)
        actual_final = float(flight_data['final_pax'].values[0])
        dtd_vals = flight_data['dtd'].values.astype(float)
        pax_cum_vals = flight_data['pax_sold_cum'].values.astype(float)

        # Predict with XGBoost (only for 2026)
        predicted_remaining = None
        if dep_year == 2026:
            import numpy as np_local
            X = flight_data[PICKUP_FEATURES].values.astype(np_local.float32)
            dmat = xgb.DMatrix(X, feature_names=PICKUP_FEATURES)
            predicted_remaining = np_local.clip(PICKUP_MODEL.predict(dmat), 0, None)

        # Build rows (sample key DTD points for cleaner display)
        dtd_points = [180, 150, 120, 90, 75, 60, 45, 30, 21, 14, 7, 5, 3, 1]
        rows = []
        for i in range(len(dtd_vals)):
            dtd_v = float(dtd_vals[i])
            pax_cum_v = float(pax_cum_vals[i])
            actual_rem = float(actual_remaining[i])
            pred_rem = float(predicted_remaining[i]) if predicted_remaining is not None else None

            pred_final = (pax_cum_v + pred_rem) if pred_rem is not None else None
            pred_lf = (pred_final / cap) if pred_final is not None and cap > 0 else None
            actual_lf = (actual_final / cap) if cap > 0 else None

            rows.append({
                "dtd": int(dtd_v),
                "pax_sold_cum": int(pax_cum_v),
                "actual_remaining": int(actual_rem),
                "actual_final": int(actual_final),
                "actual_lf": round(actual_lf, 4) if actual_lf else None,
                "pred_remaining": round(pred_rem, 1) if pred_rem is not None else None,
                "pred_final": round(pred_final, 1) if pred_final is not None else None,
                "pred_lf": round(pred_lf, 4) if pred_lf is not None else None,
            })

        # Summary KPIs
        if predicted_remaining is not None:
            import numpy as np_local
            mae_flight = float(np_local.mean(np_local.abs(actual_remaining - predicted_remaining)))
            # WAPE: Weighted Absolute Percentage Error — robust to small denominators
            sum_actual = float(np_local.sum(np_local.abs(actual_remaining)))
            if sum_actual > 0:
                wape_flight = float(np_local.sum(np_local.abs(actual_remaining - predicted_remaining)) / sum_actual * 100)
            else:
                wape_flight = None
        else:
            mae_flight = None
            wape_flight = None

        result[cab] = {
            "rows": rows,
            "metadata": {
                "flight_id": flight_id,
                "flight_number": row[8],
                "cabin_class": cab,
                "dep_year": dep_year,
                "departure_airport": row[0],
                "arrival_airport": row[1],
                "dep_date": str(dep_dt)[:10],
                "region": row[5],
                "distance_km": float(row[6]) if row[6] is not None else None,
                "flight_time_min": float(row[7]) if row[7] is not None else None,
                "capacity": int(cap),
                "actual_final_pax": int(actual_final),
                "actual_final_lf": round(float(actual_final / cap), 4) if cap > 0 else None,
                "is_prediction": dep_year == 2026,
            },
            "accuracy": {
                "flight_mae": round(float(mae_flight), 2) if mae_flight is not None else None,
                "flight_wape": round(float(wape_flight), 1) if wape_flight is not None else None,
                "model_mae": PICKUP_METRICS.get("mae"),
                "model_wape": PICKUP_METRICS.get("wape") or PICKUP_METRICS.get("mape"),
                "model_improvement": PICKUP_METRICS.get("improvement_mae_pct"),
            }
        }

    con.close()

    if not result:
        return jsonify({"error": "No data found for this flight"})

    if cabin and cabin in result:
        return jsonify(result[cabin])
    return jsonify(result)


# ─── CLUSTER API ──────────────────────────────────────────
CLUSTER_PARQUET = f"{DATA_DIR}/processed/passenger_clusters.parquet"
CLUSTER_REPORT  = f"{PROJECT_DIR}/reports/cluster_report.json"

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
TRAINING_PARQUET = f"{DATA_DIR}/processed/demand_training.parquet"

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


# ─── DEMAND FUNCTIONS API ─────────────────────────────
DEMAND_FUNCS_REPORT = f"{PROJECT_DIR}/reports/demand_functions_report.json"


@app.route("/api/demand-functions")
def api_demand_functions():
    """Return all segment definitions and pre-computed demand curves."""
    report_path = DEMAND_FUNCS_REPORT.replace("/", os.sep)
    if not os.path.exists(report_path):
        return jsonify({"error": "Demand functions report not found. Run build_demand_functions.py first."}), 404
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    return jsonify(report)


@app.route("/api/demand-curves")
def api_demand_curves():
    """Compute demand curves for a specific flight using segment models + actual price data."""
    flight_id = request.args.get("flight_id", "").strip()
    cabin = request.args.get("cabin", "economy").strip().lower()

    # Load demand functions report
    report_path = DEMAND_FUNCS_REPORT.replace("/", os.sep)
    if not os.path.exists(report_path):
        return jsonify({"error": "Demand functions report not found"}), 404
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    segments = report["segments"]
    price_ref = report.get("price_reference", {})
    base_price = price_ref.get(cabin, {}).get("avg", 500)

    # If flight_id provided, get actual price data for that flight
    flight_price = None
    flight_info = {}
    if flight_id:
        con = get_con()
        row = con.execute(f"""
            SELECT
                AVG(CASE WHEN s.pax_sold_today > 0
                    THEN s.ticket_rev_today / s.pax_sold_today ELSE NULL END) AS avg_price,
                MAX(m.capacity) AS capacity,
                MAX(m.departure_airport) AS dep_ap,
                MAX(m.arrival_airport) AS arr_ap,
                MAX(m.region) AS region,
                MAX(m.distance_km) AS distance_km
            FROM read_parquet('{PARQUET_PATH}') s
            LEFT JOIN read_parquet('{METADATA_PATH}') m
                ON s.flight_id = m.flight_id AND LOWER(s.cabin_class) = LOWER(m.cabin_class)
            WHERE s.flight_id = $1 AND LOWER(s.cabin_class) = $2
        """, [flight_id, cabin]).fetchone()
        con.close()

        if row and row[0]:
            flight_price = float(row[0])
            base_price = flight_price
            flight_info = {
                "flight_id": flight_id,
                "cabin": cabin,
                "avg_price": round(flight_price, 2),
                "capacity": int(row[1]) if row[1] else 0,
                "departure_airport": row[2],
                "arrival_airport": row[3],
                "region": row[4],
                "distance_km": _num(row[5]),
            }

    # Generate curves for each segment at this base price
    price_ratios = [round(0.3 + i * 0.1, 1) for i in range(28)]
    dtd_points = [0, 1, 3, 5, 7, 14, 21, 30, 45, 60, 90, 120, 150, 180]

    segment_curves = {}
    combined_demand = []
    combined_revenue = []

    for pr in price_ratios:
        total_q = 0
        total_rev = 0
        for sid, seg in segments.items():
            elast = seg["price_elasticity"]
            peak_dtd = seg["booking_window"]["peak_dtd"]
            dtd_decay = seg["dtd_decay_rate"]
            share = seg["base_share_pct"] / 100

            price_effect = max(pr ** elast, 0.01)
            q = share * price_effect
            rev = pr * base_price * q
            total_q += q
            total_rev += rev
        combined_demand.append({"price_ratio": pr, "price": round(pr * base_price, 2), "demand": round(total_q, 4)})
        combined_revenue.append({"price_ratio": pr, "price": round(pr * base_price, 2), "revenue": round(total_rev, 2)})

    for sid, seg in segments.items():
        elast = seg["price_elasticity"]
        peak_dtd = seg["booking_window"]["peak_dtd"]
        share = seg["base_share_pct"] / 100

        # Price curve
        seg_price_curve = []
        for pr in price_ratios:
            price_effect = max(pr ** elast, 0.01)
            q = share * price_effect
            seg_price_curve.append({
                "price_ratio": pr,
                "price": round(pr * base_price, 2),
                "demand": round(q, 4),
                "revenue": round(pr * base_price * q, 2),
            })

        # DTD curve
        seg_dtd_curve = []
        dtd_sigma = max(peak_dtd * 0.6, 3)
        for dtd in dtd_points:
            import math as _math
            timing = _math.exp(-0.5 * ((dtd - peak_dtd) / dtd_sigma) ** 2)
            if seg["dtd_decay_rate"] >= 0.3 and dtd <= 3:
                timing = max(timing, 0.9)
            seg_dtd_curve.append({
                "dtd": dtd,
                "demand": round(share * timing, 4),
            })

        best_rev = max(seg_price_curve, key=lambda x: x["revenue"])
        segment_curves[sid] = {
            "price_curve": seg_price_curve,
            "dtd_curve": seg_dtd_curve,
            "optimal": {
                "price_ratio": best_rev["price_ratio"],
                "price": best_rev["price"],
                "revenue": best_rev["revenue"],
                "demand": best_rev["demand"],
            },
        }

    # Overall optimal
    best_combined = max(combined_revenue, key=lambda x: x["revenue"])

    return jsonify({
        "base_price": round(base_price, 2),
        "cabin": cabin,
        "flight_info": flight_info,
        "segments": {sid: segments[sid] for sid in segments},
        "segment_curves": segment_curves,
        "combined_demand": combined_demand,
        "combined_revenue": combined_revenue,
        "optimal_combined": {
            "price_ratio": best_combined["price_ratio"],
            "price": best_combined["price"],
            "revenue": best_combined["revenue"],
        },
    })


# ─── FARE CLASSES API ─────────────────────────────────
@app.route("/api/fare-classes")
def api_fare_classes():
    """Return fare class structure, DTD×LF availability matrix, segment matching."""
    import math as _math

    # Fare class definitions
    fare_classes = {
        "V": {"name": "V — Promosyon", "name_short": "V", "multiplier": 0.5, "protection": 0.0, "open_until_lf": 0.40, "color": "#94a3b8", "description": "En düşük fiyat. Erken rezervasyon, fiyata duyarlı yolcular."},
        "K": {"name": "K — İndirimli", "name_short": "K", "multiplier": 0.75, "protection": 0.2, "open_until_lf": 0.60, "color": "#c9a227", "description": "Orta-düşük fiyat. Planlı seyahat, esnek tarih."},
        "M": {"name": "M — Esnek", "name_short": "M", "multiplier": 1.0, "protection": 0.4, "open_until_lf": 0.85, "color": "#6366f1", "description": "Standart fiyat. İptal/değişiklik esnekliği var."},
        "Y": {"name": "Y — Tam Fiyat", "name_short": "Y", "multiplier": 1.5, "protection": 0.6, "open_until_lf": 1.0, "color": "#ef4444", "description": "En yüksek fiyat. Tam esneklik, son dakika."},
    }

    # DTD rules
    dtd_rules = [
        {"dtd_min": 60, "dtd_max": 180, "open": ["V", "K", "M"], "label": "Erken Dönem"},
        {"dtd_min": 30, "dtd_max": 59,  "open": ["K", "M"],      "label": "Orta Dönem"},
        {"dtd_min": 14, "dtd_max": 29,  "open": ["K", "M", "Y"], "label": "Geç Dönem"},
        {"dtd_min": 7,  "dtd_max": 13,  "open": ["M", "Y"],      "label": "Son Hafta"},
        {"dtd_min": 0,  "dtd_max": 6,   "open": ["Y"],            "label": "Son Dakika"},
    ]

    # Build DTD × LF heatmap matrix
    dtd_points = [0, 1, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 150, 180]
    lf_points = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    heatmap = []
    for dtd in dtd_points:
        row = []
        for lf in lf_points:
            lf_ratio = lf / 100
            # Find DTD rule
            open_classes = ["Y"]
            for rule in dtd_rules:
                if rule["dtd_min"] <= dtd <= rule["dtd_max"]:
                    open_classes = rule["open"]
                    break
            # Filter by LF protection
            available = []
            for fc_id in open_classes:
                fc = fare_classes[fc_id]
                if lf_ratio < fc["open_until_lf"] or fc_id == "Y":
                    available.append(fc_id)
            if not available:
                available = ["Y"]
            # Best fare = cheapest available
            best = available[0]
            row.append({"dtd": dtd, "lf": lf, "available": available, "best_fare": best, "price_mult": fare_classes[best]["multiplier"]})
        heatmap.append(row)

    # Segment → fare class matching
    demand_path = DEMAND_FUNCS_REPORT.replace("/", os.sep)
    segment_matching = []
    if os.path.exists(demand_path):
        with open(demand_path, "r", encoding="utf-8") as f:
            dreport = json.load(f)
        segments = dreport.get("segments", {})
        for sid, seg in segments.items():
            wtp_avg = (seg["wtp_multiplier"]["min"] + seg["wtp_multiplier"]["max"]) / 2
            # Find the most expensive fare class the segment can afford (revenue max)
            best_fc = "V"
            for fc_id in ["V", "K", "M", "Y"]:
                if fare_classes[fc_id]["multiplier"] <= wtp_avg:
                    best_fc = fc_id
            segment_matching.append({
                "segment_id": sid,
                "segment_name": seg["name"],
                "icon": seg["icon"],
                "color": seg["color"],
                "wtp_range": f"{seg['wtp_multiplier']['min']}-{seg['wtp_multiplier']['max']}x",
                "wtp_avg": round(wtp_avg, 2),
                "preferred_fare": best_fc,
                "preferred_fare_name": fare_classes[best_fc]["name"],
                "elasticity": seg["price_elasticity"],
                "booking_window": f"{seg['booking_window']['min_dtd']}-{seg['booking_window']['max_dtd']} gün",
            })

    # Price examples per cabin
    price_ref = {}
    if os.path.exists(demand_path):
        price_ref = dreport.get("price_reference", {})

    price_examples = {}
    for cabin in ["economy", "business"]:
        base = price_ref.get(cabin, {}).get("avg", 500 if cabin == "economy" else 1500)
        price_examples[cabin] = {
            fc_id: {"price": round(base * fc["multiplier"], 2), "multiplier": fc["multiplier"]}
            for fc_id, fc in fare_classes.items()
        }
        price_examples[cabin]["base_price"] = round(base, 2)

    return jsonify({
        "fare_classes": fare_classes,
        "dtd_rules": dtd_rules,
        "heatmap": heatmap,
        "dtd_points": dtd_points,
        "lf_points": lf_points,
        "segment_matching": segment_matching,
        "price_examples": price_examples,
    })


# ─── SIMULATION API ───────────────────────────────────
SIM_REPORT = f"{BASE_DIR}/simulation_report.json"


@app.route("/api/simulation")
def api_simulation():
    """Return simulation results: static vs dynamic pricing comparison."""
    report_path = SIM_REPORT.replace("/", os.sep)
    if not os.path.exists(report_path):
        return jsonify({"error": "Simulation report not found. Run run_simulation.py first."}), 404
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Strip daily data if summary_only requested (lighter payload)
    summary_only = request.args.get("summary", "").strip().lower() == "true"
    if summary_only:
        routes = {}
        for k, v in report.get("routes", {}).items():
            route_copy = {key: val for key, val in v.items()}
            route_copy["static"] = {key: val for key, val in v["static"].items() if key != "daily"}
            route_copy["dynamic"] = {key: val for key, val in v["dynamic"].items() if key != "daily"}
            routes[k] = route_copy
        report_copy = {**report, "routes": routes}
        return jsonify(report_copy)

    return jsonify(report)


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


# ─── SENTIMENT PAGE & API ──────────────────────────────
@app.route("/sentiment")
def sentiment_page():
    return render_template("sentiment.html")


@app.route("/api/sentiment/status")
def api_sentiment_status():
    if not SENTIMENT_READY:
        return jsonify({"ready": False, "error": "sentiment module not available", "cities": []})
    return jsonify({
        "ready": True,
        "source": "GDELT + DeBERTa",
        "last_update": _SENT_CACHE.get("last_update"),
        "loading": _SENT_CACHE.get("loading", False),
        "cities_count": len(SENT_CITIES),
    })


@app.route("/api/sentiment/all")
def api_sentiment_all():
    """Tum sehirlerin sentiment ozetini doner. Bellekten aninda."""
    if not SENTIMENT_READY:
        return jsonify({"error": "sentiment module not available"}), 503

    if _SENT_CACHE["data"]:
        return jsonify(_SENT_CACHE["data"])

    if _SENT_CACHE["loading"]:
        return jsonify({"_loading": True, "_message": "Sentiment verileri yukleniyor..."}), 202

    return jsonify({"_loading": True, "_message": "Scheduler henuz calismadi"}), 202


@app.route("/api/sentiment/<city_key>")
def api_sentiment_city(city_key):
    if not SENTIMENT_READY:
        return jsonify({"error": "sentiment module not available"}), 503
    if city_key not in SENT_CITIES:
        return jsonify({"error": f"Unknown city: {city_key}"}), 404

    # Cache'ten dondur
    if _SENT_CACHE["data"] and city_key in _SENT_CACHE["data"]:
        return jsonify(_SENT_CACHE["data"][city_key])

    cfg = SENT_CITIES[city_key]
    return jsonify({
        "city": city_key, "label": cfg["label"],
        "flag": cfg["flag"], "color": cfg["color"],
        "aggregate": {"composite_score": 0, "alert_level": "low", "article_count": 0},
        "articles": [],
    })


# ─── DYNAMIC PRICING & SIMULATION ENGINE ─────────────────
SIM_READY = False
try:
    from pricing_engine import PricingEngine
    from simulation_engine import SimulationEngine

    # Route mesafeleri yukle
    _route_distances = {}
    _route_meta = {}
    try:
        _meta_con = duckdb.connect()
        _meta_rows = _meta_con.execute(f"""
            SELECT DISTINCT
                departure_airport || '_' || arrival_airport as route_key,
                distance_km, region, cabin_class, capacity
            FROM read_parquet('{METADATA_PATH}')
            WHERE departure_airport = 'IST'
        """).fetchall()
        _meta_con.close()
        for r in _meta_rows:
            _route_distances[r[0]] = r[1]
            _route_meta[r[0] + "_" + r[3]] = {"distance_km": r[1], "region": r[2], "capacity": r[4]}
    except Exception as e:
        print(f"[Pricing] Route metadata load failed: {e}")

    # Segment verileri yukle
    _segments = {}
    _dtd_curves = {}
    _demand_report_path = f"{PROJECT_DIR}/reports/demand_functions_report.json"
    if os.path.exists(_demand_report_path.replace("/", os.sep)):
        with open(_demand_report_path.replace("/", os.sep), encoding="utf-8") as f:
            _dreport = json.load(f)
        _segments = _dreport.get("segments", {})
        for sid in _dreport.get("curves", {}):
            _dtd_curves[sid] = _dreport["curves"][sid].get("dtd_demand", [])

    # Engine'leri olustur
    _pricing_engine = PricingEngine(
        segments=_segments,
        route_distances=_route_distances,
        sentiment_cache=_SENT_CACHE,
        airport_to_city=AIRPORT_TO_CITY if SENTIMENT_READY else {},
    )
    # ForecastBridge — modelleri simulasyona bagla
    _forecast_bridge = None
    try:
        from forecast_bridge import ForecastBridge
        _forecast_bridge = ForecastBridge(
            tft_predictions_df=TFT_PRED,
            twostage_clf=TWOSTAGE_CLF,
            twostage_reg=TWOSTAGE_REG,
            twostage_features=TWOSTAGE_FEATURES,
            pickup_model=PICKUP_MODEL,
            pickup_features=PICKUP_FEATURES,
            route_meta=_route_meta,
        )
        print(f"[Bridge] ForecastBridge ready: TFT={len(_forecast_bridge._tft_cache)} entries")
    except Exception as e:
        print(f"[Bridge] ForecastBridge not available: {e}")

    _sim_engine = SimulationEngine(pricing_engine=_pricing_engine, forecast_bridge=_forecast_bridge)

    SIM_READY = True
    print(f"[Pricing] Engine ready: {len(_route_distances)} routes, {len(_segments)} segments")
except Exception as e:
    print(f"[Pricing] Engine not available: {e}")
    import traceback; traceback.print_exc()


# ─── SIMULATION API ──────────────────────────────────────
@app.route("/simulation")
def simulation_page():
    return render_template("simulation.html")


@app.route("/booking")
def booking_page():
    return render_template("booking.html")


@app.route("/api/routes")
def api_routes():
    """Tum rotalari dondur."""
    routes = set()
    for rk in _route_meta:
        parts = rk.rsplit("_", 1)
        route = parts[0].replace("_", "-")
        routes.add(route)
    return jsonify({"routes": sorted(routes)})


@app.route("/api/sim/start", methods=["POST"])
def api_sim_start():
    if not SIM_READY:
        return jsonify({"error": "Simulation engine not available"}), 503
    data = request.get_json() or {}
    date_range = data.get("date_range", ["2026-07-01", "2026-12-31"])
    speed = data.get("speed", 1440)
    cabins = data.get("cabins", ["economy", "business"])

    # Rota filtresi (tek rota veya tumu)
    routes_filter = data.get("routes")  # ["IST-DXB"] veya None (tumu)

    # Ucuslari metadata'dan cek
    flights = []
    for rk, meta in _route_meta.items():
        parts = rk.rsplit("_", 1)
        route_key = parts[0]
        cabin = parts[1]
        if cabin not in cabins:
            continue
        route_str = route_key.replace("_", "-")
        # Rota filtresi
        if routes_filter and route_str not in routes_filter:
            continue
        start = datetime.strptime(date_range[0], "%Y-%m-%d").date()
        end = datetime.strptime(date_range[1], "%Y-%m-%d").date()
        d = start
        while d <= end:
            flights.append({
                "flight_id": f"{route_str}_{d.isoformat()}_{cabin}",
                "route": route_str,
                "cabin": cabin,
                "dep_date": d.isoformat(),
                "capacity": meta["capacity"],
                "distance_km": meta["distance_km"],
                "region": meta["region"],
            })
            d += timedelta(days=1)

    _sim_engine.initialize(flights, date_range, speed, _dtd_curves)
    _sim_engine.start()
    return jsonify({"status": "started", "flights": len(_sim_engine.inventory)})


@app.route("/api/sim/pause", methods=["POST"])
def api_sim_pause():
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    _sim_engine.pause()
    return jsonify({"status": _sim_engine.state})


@app.route("/api/sim/resume", methods=["POST"])
def api_sim_resume():
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    _sim_engine.resume()
    return jsonify({"status": _sim_engine.state})


@app.route("/api/sim/speed", methods=["POST"])
def api_sim_speed():
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    data = request.get_json() or {}
    _sim_engine.set_speed(data.get("speed", 1440))
    return jsonify({"speed": _sim_engine.clock.speed})


@app.route("/api/sim/jump", methods=["POST"])
def api_sim_jump():
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    data = request.get_json() or {}
    target = data.get("date")
    if not target:
        return jsonify({"error": "date required"}), 400
    _sim_engine.jump_to(target)
    return jsonify({"status": _sim_engine.state, "sim_date": _sim_engine.clock.today().isoformat()})


@app.route("/api/sim/status")
def api_sim_status():
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    return jsonify(_sim_engine.get_status())


@app.route("/api/sim/flights")
def api_sim_flights():
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    return jsonify({"flights": _sim_engine.get_flights_list()})


@app.route("/api/sim/flight/<path:flight_key>")
def api_sim_flight_detail(flight_key):
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    detail = _sim_engine.get_flight_detail(flight_key)
    if not detail:
        return jsonify({"error": "Flight not found"}), 404
    return jsonify(detail)


@app.route("/api/sim/inject", methods=["POST"])
def api_sim_inject():
    """Manuel bot enjeksiyonu."""
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    data = request.get_json() or {}
    sales = _sim_engine.inject_bots(
        data.get("flight_key", ""),
        data.get("segment", "D"),
        data.get("count", 10),
    )
    return jsonify({"sales": sales})


@app.route("/api/sim/override", methods=["POST"])
def api_sim_override():
    """Fare class elle ac/kapa."""
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    data = request.get_json() or {}
    ok = _sim_engine.override_fare_class(
        data.get("flight_key", ""),
        data.get("fare_class", "V"),
        data.get("action", "open"),
    )
    return jsonify({"success": ok})


@app.route("/api/pricing/quote")
def api_pricing_quote():
    """Canli fiyat teklifi (booking sayfasi icin)."""
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    flight_key = request.args.get("flight_key", "").strip()
    segment = request.args.get("segment", "").strip() or None

    inv = _sim_engine.inventory.get(flight_key)
    if not inv:
        return jsonify({"error": "Flight not found"}), 404

    dtd = _sim_engine.clock.dtd(inv["dep_date"])
    session_str = request.args.get("session", "")
    session_info = json.loads(session_str) if session_str else None

    quote = _pricing_engine.compute_price(inv, dtd, segment_id=segment, session_info=session_info)
    return jsonify(quote)


@app.route("/api/pricing/book", methods=["POST"])
def api_pricing_book():
    """Bilet satin alma (booking sayfasindan)."""
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    data = request.get_json() or {}
    result = _sim_engine.book_human(
        flight_key=data.get("flight_key", ""),
        fare_class=data.get("fare_class", "Y"),
        session_info=data.get("session"),
    )
    return jsonify(result)


if __name__ == "__main__":
    v_label = "V2 (ticket + ancillary)" if USE_V2 else "V1 (legacy)"
    fc_label = "ON" if FORECAST_READY else "OFF"
    sent_label = "ON" if SENTIMENT_READY else "OFF"
    sim_label = "ON" if SIM_READY else "OFF"
    print(f"\nFlight Snapshot Dashboard -- {v_label} | Forecast: {fc_label} | Sentiment: {sent_label} | Sim: {sim_label}")
    print(f"   Snapshot: {PARQUET_PATH}")
    print(f"   Metadata: {METADATA_PATH}")
    print(f"   URL: http://localhost:5005\n")
    app.run(debug=True, host="0.0.0.0", port=5005, use_reloader=False)


