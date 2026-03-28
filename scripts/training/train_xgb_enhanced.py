# -*- coding: utf-8 -*-
"""
Enhanced XGBoost - 16 GB RAM Uyumlu
dssd=====================================
Strateji: DuckDB join -> diske parquet yaz -> DuckDB kapat -> RAM bosalt
         -> parquet'tan numpy oku -> XGBoost egit
Ayni anda 2 buyuk nesne RAM'de OLMAZ. Tum 37M satir, veri kaybi YOK.
"""

import duckdb
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor
import joblib
import json
import time
import gc

BASE_DIR = Path(r"C:\Users\ahmet\OneDrive\Desktop\ptir")

DEMAND_PATH   = BASE_DIR / "demand_training.parquet"
METADATA_PATH = BASE_DIR / "flight_metadata.parquet"
ROUTE_DAILY   = BASE_DIR / "tft_route_daily.parquet"
TRAIN_PATH    = BASE_DIR / "_tmp_train.parquet"
TEST_PATH     = BASE_DIR / "_tmp_test.parquet"

# DuckDB icin forward-slash path (Windows backslash sorun yaratir)
DEMAND_STR   = str(DEMAND_PATH).replace("\\", "/")
METADATA_STR = str(METADATA_PATH).replace("\\", "/")
ROUTE_STR    = str(ROUTE_DAILY).replace("\\", "/")
TRAIN_STR    = str(TRAIN_PATH).replace("\\", "/")
TEST_STR     = str(TEST_PATH).replace("\\", "/")

start = time.time()

# ============================================
# FAZE 1: DuckDB join -> diske yaz -> kapat
# ============================================
print("=" * 50, flush=True)
print("FAZE 1: DuckDB Join + Diske Yaz", flush=True)
print("=" * 50, flush=True)

con = duckdb.connect()
# DuckDB bellek limiti
con.execute("SET memory_limit='6GB'")
con.execute("SET temp_directory='.'")

print("[1/3] Route mapping...", flush=True)
con.execute(f"""
    CREATE TABLE meta AS
    SELECT DISTINCT
        flight_id,
        departure_airport || '_' || arrival_airport AS route
    FROM read_parquet('{METADATA_STR}')
""")

print("[2/3] Route-daily features...", flush=True)
con.execute(f"""
    CREATE TABLE rdaily AS
    SELECT
        route, cabin_class, dep_date,
        total_pax AS tft_route_demand,
        n_flights AS route_n_flights,
        avg_fare, std_fare, max_fare, min_fare,
        n_bookings, avg_group_size,
        corporate_pct, agency_pct, connecting_pct,
        early_booking_pct, late_booking_pct,
        child_pct, halal_pct,
        CASE WHEN is_special_period THEN 1 ELSE 0 END AS is_special_period,
        CASE WHEN tag_yaz_tatili THEN 1 ELSE 0 END AS tag_yaz_tatili,
        CASE WHEN tag_kis_tatili THEN 1 ELSE 0 END AS tag_kis_tatili,
        CASE WHEN tag_bahar_tatili THEN 1 ELSE 0 END AS tag_bahar_tatili,
        CASE WHEN tag_ramazan THEN 1 ELSE 0 END AS tag_ramazan,
        CASE WHEN tag_kongre_fuar THEN 1 ELSE 0 END AS tag_kongre_fuar,
        CASE WHEN tag_ski_sezonu THEN 1 ELSE 0 END AS tag_ski_sezonu,
        CASE WHEN tag_festival THEN 1 ELSE 0 END AS tag_festival,
        CASE WHEN tag_is_seyahati THEN 1 ELSE 0 END AS tag_is_seyahati,
        CASE WHEN tag_hac_umre THEN 1 ELSE 0 END AS tag_hac_umre
    FROM read_parquet('{ROUTE_STR}')
""")

print("[3/3] Join + encode -> diske yaziliyor...", flush=True)

ENRICHED_SQL = f"""
    SELECT
        d.dep_year,
        d.y_pax_sold_today::FLOAT AS y_pax_sold_today,
        d.pax_last_7d::FLOAT AS pax_last_7d,
        d.dtd::FLOAT AS dtd,
        d.pax_sold_cum::FLOAT AS pax_sold_cum,
        d.capacity::FLOAT AS capacity,
        d.remaining_seats::FLOAT AS remaining_seats,
        d.load_factor::FLOAT AS load_factor,
        d.distance_km::FLOAT AS distance_km,
        d.flight_time_min::FLOAT AS flight_time_min,
        d.dep_month::FLOAT AS dep_month,
        d.dep_dow::FLOAT AS dep_dow,
        d.dep_hour::FLOAT AS dep_hour,
        d.ff_gold_pct::FLOAT AS ff_gold_pct,
        d.ff_elite_pct::FLOAT AS ff_elite_pct,
        COALESCE(r.tft_route_demand, 0)::FLOAT AS tft_route_demand,
        COALESCE(r.route_n_flights, 1)::FLOAT AS route_n_flights,
        COALESCE(r.avg_fare, 0)::FLOAT AS avg_fare,
        COALESCE(r.std_fare, 0)::FLOAT AS std_fare,
        COALESCE(r.max_fare, 0)::FLOAT AS max_fare,
        COALESCE(r.min_fare, 0)::FLOAT AS min_fare,
        COALESCE(r.n_bookings, 0)::FLOAT AS route_n_bookings,
        COALESCE(r.avg_group_size, 0)::FLOAT AS avg_group_size,
        COALESCE(r.corporate_pct, 0)::FLOAT AS corporate_pct,
        COALESCE(r.agency_pct, 0)::FLOAT AS agency_pct,
        COALESCE(r.connecting_pct, 0)::FLOAT AS connecting_pct,
        COALESCE(r.early_booking_pct, 0)::FLOAT AS early_booking_pct,
        COALESCE(r.late_booking_pct, 0)::FLOAT AS late_booking_pct,
        COALESCE(r.child_pct, 0)::FLOAT AS child_pct,
        COALESCE(r.halal_pct, 0)::FLOAT AS halal_pct,
        COALESCE(r.is_special_period, 0)::FLOAT AS is_special_period,
        COALESCE(r.tag_yaz_tatili, 0)::FLOAT AS tag_yaz_tatili,
        COALESCE(r.tag_kis_tatili, 0)::FLOAT AS tag_kis_tatili,
        COALESCE(r.tag_bahar_tatili, 0)::FLOAT AS tag_bahar_tatili,
        COALESCE(r.tag_ramazan, 0)::FLOAT AS tag_ramazan,
        COALESCE(r.tag_kongre_fuar, 0)::FLOAT AS tag_kongre_fuar,
        COALESCE(r.tag_ski_sezonu, 0)::FLOAT AS tag_ski_sezonu,
        COALESCE(r.tag_festival, 0)::FLOAT AS tag_festival,
        COALESCE(r.tag_is_seyahati, 0)::FLOAT AS tag_is_seyahati,
        COALESCE(r.tag_hac_umre, 0)::FLOAT AS tag_hac_umre,
        (CASE WHEN LOWER(d.cabin_class) = 'economy' THEN 1.0 ELSE 0.0 END)::FLOAT AS cabin_economy,
        (CASE WHEN LOWER(d.cabin_class) = 'business' THEN 1.0 ELSE 0.0 END)::FLOAT AS cabin_business,
        (CASE WHEN d.region = 'Domestic' THEN 1.0 ELSE 0.0 END)::FLOAT AS reg_domestic,
        (CASE WHEN d.region = 'Europe' THEN 1.0 ELSE 0.0 END)::FLOAT AS reg_europe,
        (CASE WHEN d.region = 'Middle East' THEN 1.0 ELSE 0.0 END)::FLOAT AS reg_mideast,
        (CASE WHEN d.region = 'Asia' THEN 1.0 ELSE 0.0 END)::FLOAT AS reg_asia,
        (CASE WHEN d.region = 'Americas' THEN 1.0 ELSE 0.0 END)::FLOAT AS reg_americas,
        (CASE WHEN d.region NOT IN ('Domestic','Europe','Middle East','Asia','Americas') THEN 1.0 ELSE 0.0 END)::FLOAT AS reg_other
    FROM read_parquet('{DEMAND_STR}') d
    LEFT JOIN meta m ON d.flight_id = m.flight_id
    LEFT JOIN rdaily r ON m.route = r.route
        AND LOWER(d.cabin_class) = LOWER(r.cabin_class)
        AND CAST(SPLIT_PART(SPLIT_PART(d.flight_id, '_', 2), ' ', 1) AS DATE) = r.dep_date
"""

# Train (2025) -> diske
con.execute(f"COPY ({ENRICHED_SQL} WHERE d.dep_year = 2025) TO '{TRAIN_STR}' (FORMAT PARQUET)")
train_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{TRAIN_STR}')").fetchone()[0]
print(f"  Train: {train_count:,} satir -> {TRAIN_PATH}", flush=True)

# Test (2026) -> diske
con.execute(f"COPY ({ENRICHED_SQL} WHERE d.dep_year = 2026) TO '{TEST_STR}' (FORMAT PARQUET)")
test_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{TEST_STR}')").fetchone()[0]
print(f"  Test:  {test_count:,} satir -> {TEST_PATH}", flush=True)

# DuckDB tamamen kapat - RAM bosalsin
con.close()
del con
gc.collect()
print(f"\n  DuckDB KAPATILDI - RAM bosaldi ({time.time()-start:.0f}s)", flush=True)

# ============================================
# FAZE 2: Parquet'tan oku -> XGBoost egit
# ============================================
print("\n" + "=" * 50, flush=True)
print("FAZE 2: XGBoost Egitimi", flush=True)
print("=" * 50, flush=True)

import pyarrow.parquet as pq

# Feature kolonlari (dep_year, y_pax_sold_today, pax_last_7d haric)
EXCLUDE = {'dep_year', 'y_pax_sold_today', 'pax_last_7d'}

# -- Train yukle --
print("[1/3] Train verisi yukleniyor...", flush=True)
train_table = pq.read_table(TRAIN_PATH)
feature_cols = [c for c in train_table.column_names if c not in EXCLUDE]
print(f"  Features: {len(feature_cols)}", flush=True)

X_train = np.column_stack([train_table.column(c).to_numpy().astype(np.float32) for c in feature_cols])
y_train = train_table.column('y_pax_sold_today').to_numpy().astype(np.float32)
del train_table
gc.collect()
print(f"  X_train: {X_train.shape} ({X_train.nbytes / 1e9:.1f} GB)", flush=True)

# -- Tek XGBRegressor (eval_set yok, RAM yetmiyor) --
print("\n[2/3] XGBRegressor egitiliyor (500 tree)...", flush=True)

reg = XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    eval_metric='mae', random_state=42, n_jobs=-1,
    tree_method='hist',
)
reg.fit(X_train, y_train, verbose=50)
print(f"  Regressor OK ({time.time()-start:.0f}s)", flush=True)

del X_train, y_train
gc.collect()
print("  Train RAM'den silindi", flush=True)

# -- Evaluate --
y_pred = np.clip(reg.predict(X_test), 0, None)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
baseline_mae  = mean_absolute_error(y_test, baseline_vals)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_vals))
imp_mae  = (1 - mae / baseline_mae) * 100
imp_rmse = (1 - rmse / baseline_rmse) * 100

print(f"\n{'='*50}")
print(f"{'ENHANCED XGBOOST SONUCLARI':^50}")
print(f"{'='*50}")
print(f"\n  Baseline (pace/7):  MAE={baseline_mae:.4f}  RMSE={baseline_rmse:.4f}")
print(f"  Enhanced XGBoost:   MAE={mae:.4f} ({imp_mae:+.1f}%)  RMSE={rmse:.4f} ({imp_rmse:+.1f}%)")
print(f"\n  Eski XGBoost:       MAE=0.780  RMSE=1.326  AUC=0.835")

# Feature importance top 15
print(f"\n  Top 15 Feature Importance:")
for name, imp in sorted(zip(feature_cols, reg.feature_importances_), key=lambda x: -x[1])[:15]:
    print(f"    {name:30s}: {imp:.4f}")

# ============================================
# KAYDET
# ============================================
print(f"\n  Kaydediliyor...", flush=True)
joblib.dump(reg, BASE_DIR / "xgb_enhanced_regressor.pkl")
with open(BASE_DIR / "enhanced_feature_list.json", "w") as f:
    json.dump({"features": feature_cols}, f)

metrics = {
    "mae": round(float(mae), 4), "rmse": round(float(rmse), 4),
    "baseline_mae": round(float(baseline_mae), 4), "baseline_rmse": round(float(baseline_rmse), 4),
    "improvement_mae_pct": round(float(imp_mae), 1), "improvement_rmse_pct": round(float(imp_rmse), 1),
    "train_rows": int(train_count), "test_rows": int(test_count),
    "n_features": len(feature_cols), "best_iteration": int(reg.best_iteration),
}
with open(BASE_DIR / "xgb_enhanced_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Temp dosyalari temizle
Path(TRAIN_PATH).unlink(missing_ok=True)
Path(TEST_PATH).unlink(missing_ok=True)

elapsed = time.time() - start
print(f"\n  Cikti:")
print(f"    xgb_enhanced_regressor.pkl")
print(f"    enhanced_feature_list.json")
print(f"    xgb_enhanced_metrics.json")
print(f"\n  Toplam sure: {elapsed:.0f} saniye")
print(f"\n[TAMAMLANDI]")
