"""
Enhanced XGBoost Demand Model — Kaggle Version
================================================
Ucus bazli gunluk artisal talep tahmini (y_pax_sold_today)
TFT rota tahmini + fare + channel + event features

Kaggle Dataset: "ptir-xgb-data"
  - demand_training.parquet (102 MB)
  - flight_metadata.parquet (1.4 MB)
  - tft_route_daily.parquet (6.8 MB)

Train: 2025, Test: 2026
Two-stage: XGBClassifier (sale yes/no) + XGBRegressor (how many pax)
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor
import joblib
import json
import time

# ── Kaggle Paths ──
INPUT_DIR  = Path("/kaggle/input/datasets/afgokbullut/ptir-xgb-data")
OUTPUT_DIR = Path("/kaggle/working")

DEMAND_PATH   = INPUT_DIR / "demand_training.parquet"
METADATA_PATH = INPUT_DIR / "flight_metadata.parquet"
ROUTE_DAILY   = INPUT_DIR / "tft_route_daily.parquet"

for p in [DEMAND_PATH, METADATA_PATH, ROUTE_DAILY]:
    assert p.exists(), f"Dosya bulunamadi: {p}"
print("Dosyalar OK")

start = time.time()
con = duckdb.connect()

# ── Step 1: Route mapping ──
print("[1/5] Route mapping...", flush=True)
con.execute(f"""
    CREATE TABLE meta AS
    SELECT DISTINCT
        flight_id,
        departure_airport || '_' || arrival_airport AS route,
        departure_airport,
        arrival_airport
    FROM read_parquet('{METADATA_PATH}')
""")

# ── Step 2: Route-daily features ──
print("[2/5] Route-daily features...", flush=True)
con.execute(f"""
    CREATE TABLE rdaily AS
    SELECT
        route, cabin_class, dep_date,
        total_pax AS tft_route_demand,
        n_flights AS route_n_flights,
        avg_fare, std_fare, max_fare, min_fare,
        n_bookings, avg_group_size,
        corporate_pct, agency_pct, website_pct, mobile_pct, connecting_pct,
        early_booking_pct, late_booking_pct,
        child_pct,
        gold_elite_pct AS route_gold_elite_pct,
        elite_pct AS route_elite_pct,
        halal_pct,
        is_special_period,
        tag_yaz_tatili, tag_kis_tatili, tag_bahar_tatili, tag_ramazan,
        tag_kongre_fuar, tag_ski_sezonu, tag_festival, tag_is_seyahati, tag_hac_umre
    FROM read_parquet('{ROUTE_DAILY}')
""")

# ── Step 3: Join ──
print("[3/5] Feature join (37M satir)...", flush=True)
con.execute(f"""
    CREATE TABLE enriched AS
    SELECT
        d.flight_id, d.cabin_class, d.dtd, d.y_pax_sold_today,
        d.pax_sold_cum, d.pax_last_7d, d.capacity, d.remaining_seats,
        d.load_factor, d.region, d.distance_km, d.flight_time_min,
        d.dep_year, d.dep_month, d.dep_dow, d.dep_hour,
        d.ff_gold_pct, d.ff_elite_pct,
        m.route,
        COALESCE(r.tft_route_demand, 0)   AS tft_route_demand,
        COALESCE(r.route_n_flights, 1)     AS route_n_flights,
        COALESCE(r.avg_fare, 0)            AS avg_fare,
        COALESCE(r.std_fare, 0)            AS std_fare,
        COALESCE(r.max_fare, 0)            AS max_fare,
        COALESCE(r.min_fare, 0)            AS min_fare,
        COALESCE(r.n_bookings, 0)          AS route_n_bookings,
        COALESCE(r.avg_group_size, 0)      AS avg_group_size,
        COALESCE(r.corporate_pct, 0)       AS corporate_pct,
        COALESCE(r.agency_pct, 0)          AS agency_pct,
        COALESCE(r.connecting_pct, 0)      AS connecting_pct,
        COALESCE(r.early_booking_pct, 0)   AS early_booking_pct,
        COALESCE(r.late_booking_pct, 0)    AS late_booking_pct,
        COALESCE(r.child_pct, 0)           AS child_pct,
        COALESCE(r.halal_pct, 0)           AS halal_pct,
        CASE WHEN r.is_special_period THEN 1 ELSE 0 END AS is_special_period,
        CASE WHEN r.tag_yaz_tatili THEN 1 ELSE 0 END    AS tag_yaz_tatili,
        CASE WHEN r.tag_kis_tatili THEN 1 ELSE 0 END    AS tag_kis_tatili,
        CASE WHEN r.tag_bahar_tatili THEN 1 ELSE 0 END  AS tag_bahar_tatili,
        CASE WHEN r.tag_ramazan THEN 1 ELSE 0 END       AS tag_ramazan,
        CASE WHEN r.tag_kongre_fuar THEN 1 ELSE 0 END   AS tag_kongre_fuar,
        CASE WHEN r.tag_ski_sezonu THEN 1 ELSE 0 END    AS tag_ski_sezonu,
        CASE WHEN r.tag_festival THEN 1 ELSE 0 END      AS tag_festival,
        CASE WHEN r.tag_is_seyahati THEN 1 ELSE 0 END   AS tag_is_seyahati,
        CASE WHEN r.tag_hac_umre THEN 1 ELSE 0 END      AS tag_hac_umre
    FROM read_parquet('{DEMAND_PATH}') d
    LEFT JOIN meta m ON d.flight_id = m.flight_id
    LEFT JOIN rdaily r ON m.route = r.route
        AND LOWER(d.cabin_class) = LOWER(r.cabin_class)
        AND CAST(SPLIT_PART(SPLIT_PART(d.flight_id, '_', 2), ' ', 1) AS DATE) = r.dep_date
""")

row_count = con.execute("SELECT COUNT(*) FROM enriched").fetchone()[0]
null_route = con.execute("SELECT COUNT(*) FROM enriched WHERE tft_route_demand = 0").fetchone()[0]
print(f"  {row_count:,} satir, {null_route:,} route-demand=0 ({null_route/row_count*100:.1f}%)", flush=True)

# ── Step 4: Train/Test Split + XGBoost ──
print("[4/5] XGBoost egitimi...", flush=True)

df = con.execute("SELECT * FROM enriched").fetchdf()
con.close()

exclude = ['flight_id', 'cabin_class', 'y_pax_sold_today', 'region', 'route']
feature_cols = [c for c in df.columns if c not in exclude]

df_encoded = pd.get_dummies(
    df[feature_cols + ['y_pax_sold_today', 'dep_year', 'cabin_class', 'region']],
    columns=['cabin_class', 'region'], drop_first=False
)
target = 'y_pax_sold_today'
feature_cols_final = [c for c in df_encoded.columns if c != target and c != 'flight_id']

train_mask = df_encoded['dep_year'] == 2025
test_mask  = df_encoded['dep_year'] == 2026

X_train = df_encoded.loc[train_mask, feature_cols_final].values.astype(np.float32)
y_train = df_encoded.loc[train_mask, target].values.astype(np.float32)
X_test  = df_encoded.loc[test_mask, feature_cols_final].values.astype(np.float32)
y_test  = df_encoded.loc[test_mask, target].values.astype(np.float32)

print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
print(f"  Features: {len(feature_cols_final)}")

# ── Stage 1: Classifier (sale yes/no) ──
print("\n  [Stage 1] Classifier...", flush=True)
y_train_cls = (y_train > 0).astype(int)
y_test_cls  = (y_test > 0).astype(int)

clf = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    eval_metric='logloss', random_state=42, n_jobs=-1,
    tree_method='hist',
)
clf.fit(X_train, y_train_cls, eval_set=[(X_test, y_test_cls)], verbose=50)

p_sale_train = clf.predict_proba(X_train)[:, 1]
p_sale_test  = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test_cls, p_sale_test)
print(f"\n  Classifier AUC: {auc:.4f}")

# ── Stage 2: Regressor (how many pax) ──
print("\n  [Stage 2] Regressor...", flush=True)
pos_mask_train = y_train > 0
X_train_pos = X_train[pos_mask_train]
y_train_pos = y_train[pos_mask_train]

reg = XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    eval_metric='rmse', random_state=42, n_jobs=-1,
    tree_method='hist',
)
reg.fit(X_train_pos, y_train_pos, verbose=50)

y_pos_pred = np.clip(reg.predict(X_test), 0, None)
y_pred = p_sale_test * y_pos_pred

# ── Step 5: Evaluation ──
print("\n[5/5] Degerlendirme...", flush=True)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Baseline: pace/7
df_test = df[df.dep_year == 2026].copy()
baseline = df_test['pax_last_7d'].values / 7.0
baseline_mae  = mean_absolute_error(y_test, baseline)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline))

improvement_mae  = (1 - mae / baseline_mae) * 100
improvement_rmse = (1 - rmse / baseline_rmse) * 100

print(f"\n{'='*50}")
print(f"{'ENHANCED XGBOOST SONUCLARI':^50}")
print(f"{'='*50}")
print(f"\n  Baseline (pace/7):")
print(f"    MAE:  {baseline_mae:.4f}")
print(f"    RMSE: {baseline_rmse:.4f}")
print(f"\n  Enhanced XGBoost:")
print(f"    MAE:  {mae:.4f}  ({improvement_mae:+.1f}% vs baseline)")
print(f"    RMSE: {rmse:.4f}  ({improvement_rmse:+.1f}% vs baseline)")
print(f"    AUC:  {auc:.4f}")
print(f"\n  Eski XGBoost (referans):")
print(f"    MAE:  0.780  (+1% vs baseline)")
print(f"    RMSE: 1.326  (+1% vs baseline)")
print(f"    AUC:  0.835")

# Feature importance (top 20)
print(f"\n  Top 20 Feature Importance (Classifier):")
for name, importance in sorted(zip(feature_cols_final, clf.feature_importances_), key=lambda x: -x[1])[:20]:
    print(f"    {name:30s}: {importance:.4f}")

print(f"\n  Top 20 Feature Importance (Regressor):")
for name, importance in sorted(zip(feature_cols_final, reg.feature_importances_), key=lambda x: -x[1])[:20]:
    print(f"    {name:30s}: {importance:.4f}")

# ── Save outputs ──
print(f"\n  Modeller kaydediliyor...", flush=True)
joblib.dump(clf, OUTPUT_DIR / "xgb_enhanced_classifier.pkl")
joblib.dump(reg, OUTPUT_DIR / "xgb_enhanced_regressor.pkl")
with open(OUTPUT_DIR / "enhanced_feature_list.json", "w") as f:
    json.dump({"features": feature_cols_final}, f)

# Save predictions for analysis
results = pd.DataFrame({
    'flight_id': df.loc[df.dep_year == 2026, 'flight_id'].values,
    'cabin_class': df.loc[df.dep_year == 2026, 'cabin_class'].values,
    'dtd': df.loc[df.dep_year == 2026, 'dtd'].values,
    'y_actual': y_test,
    'y_pred': y_pred,
    'p_sale': p_sale_test,
})
results.to_parquet(OUTPUT_DIR / "xgb_enhanced_predictions.parquet", index=False)

# Save metrics
metrics = {
    "mae": round(float(mae), 4),
    "rmse": round(float(rmse), 4),
    "auc": round(float(auc), 4),
    "baseline_mae": round(float(baseline_mae), 4),
    "baseline_rmse": round(float(baseline_rmse), 4),
    "improvement_mae_pct": round(float(improvement_mae), 1),
    "improvement_rmse_pct": round(float(improvement_rmse), 1),
    "train_rows": int(X_train.shape[0]),
    "test_rows": int(X_test.shape[0]),
    "n_features": len(feature_cols_final),
}
with open(OUTPUT_DIR / "xgb_enhanced_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

elapsed = time.time() - start
print(f"\n  Cikti dosyalari (/kaggle/working/):")
print(f"    xgb_enhanced_classifier.pkl")
print(f"    xgb_enhanced_regressor.pkl")
print(f"    enhanced_feature_list.json")
print(f"    xgb_enhanced_predictions.parquet")
print(f"    xgb_enhanced_metrics.json")
print(f"\n  Sure: {elapsed:.0f} saniye")
print(f"\n[TAMAMLANDI] — Dosyalari indir ve ptir/ klasorune koy")
