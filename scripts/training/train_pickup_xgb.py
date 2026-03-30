# -*- coding: utf-8 -*-
"""
Pickup XGBoost - Remaining Pax Prediction
==========================================
Target: remaining_pax = final_pax - pax_sold_cum
Train: 2025, Test: 2026
Faze 1: DuckDB'den train/test ayri parquet'lara yaz
Faze 2: Sirayla yukle, egit, evaluate
"""

import duckdb
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib
import json
import time
import gc

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "data" / "models"
REPORTS = BASE / "reports"

MASTER = str(DATA / "pickup_master.parquet").replace("\\", "/")
TMP_TRAIN = str(DATA / "_tmp_train.parquet").replace("\\", "/")
TMP_TEST  = str(DATA / "_tmp_test.parquet").replace("\\", "/")

start = time.time()

# ============================================
# FAZE 1: Train/Test ayir, diske yaz
# ============================================
print("=" * 50, flush=True)
print("FAZE 1: Train/Test Split", flush=True)
print("=" * 50, flush=True)

con = duckdb.connect()
con.execute("SET memory_limit='6GB'")

# DTD=0 satirlari cikar (remaining_pax=0, ogretici degil)
# flight_id, cabin_class, dep_year identifiers olarak kalsin ama feature olarak kullanilmasin
con.execute(f"COPY (SELECT * FROM read_parquet('{MASTER}') WHERE dep_year = 2025) TO '{TMP_TRAIN}' (FORMAT PARQUET)")
con.execute(f"COPY (SELECT * FROM read_parquet('{MASTER}') WHERE dep_year = 2026) TO '{TMP_TEST}' (FORMAT PARQUET)")

train_n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{TMP_TRAIN}')").fetchone()[0]
test_n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{TMP_TEST}')").fetchone()[0]
print(f"  Train: {train_n:,}  Test: {test_n:,}", flush=True)

con.close()
del con
gc.collect()
print(f"  DuckDB kapatildi ({time.time()-start:.0f}s)\n", flush=True)

# ============================================
# FAZE 2: XGBoost Egitimi
# ============================================
print("=" * 50, flush=True)
print("FAZE 2: XGBoost Egitimi", flush=True)
print("=" * 50, flush=True)

import xgboost as xgb
import pyarrow.parquet as pq

EXCLUDE = {'flight_id', 'cabin_class', 'dep_year', 'remaining_pax', 'final_pax'}

# -- Feature listesi al (sadece schema oku, RAM yemez) --
print("[1/4] Feature listesi...", flush=True)
schema = pq.read_schema(TMP_TRAIN)
feature_cols = [c for c in schema.names if c not in EXCLUDE]
print(f"  Features: {len(feature_cols)}", flush=True)

# -- Train -> DMatrix (parquet'tan chunk chunk oku) --
print("[2/4] Train DMatrix olusturuluyor (chunk)...", flush=True)
CHUNK = 3_000_000
reader = pq.ParquetFile(TMP_TRAIN)
X_chunks = []
y_chunks = []
for batch in reader.iter_batches(batch_size=CHUNK, columns=feature_cols + ['remaining_pax']):
    df_chunk = batch.to_pandas()
    # decimal128 kolonlari float'a cevir
    for col in df_chunk.columns:
        if df_chunk[col].dtype == object or str(df_chunk[col].dtype).startswith('object'):
            df_chunk[col] = df_chunk[col].astype(float)
    X_chunks.append(df_chunk[feature_cols].values.astype(np.float32))
    y_chunks.append(df_chunk['remaining_pax'].values.astype(np.float32))
    print(f"    chunk {len(X_chunks)}: {X_chunks[-1].shape[0]:,} satir", flush=True)
    del df_chunk
    gc.collect()

X_train = np.vstack(X_chunks)
y_train = np.concatenate(y_chunks)
del X_chunks, y_chunks
gc.collect()

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
del X_train, y_train
gc.collect()
print(f"  DMatrix OK, RAM bosaldi ({time.time()-start:.0f}s)", flush=True)

# -- Egit --
print("\n[3/4] XGBoost egitiliyor...", flush=True)

params = {
    'max_depth': 7,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 10,
    'eval_metric': 'mae',
    'tree_method': 'hist',
    'seed': 42,
}
bst = xgb.train(params, dtrain, num_boost_round=500, verbose_eval=50)
del dtrain
gc.collect()
print(f"  Egitim OK ({time.time()-start:.0f}s)\n", flush=True)

# -- Test yukle + evaluate (chunk) --
print("[4/4] Test + degerlendirme...", flush=True)
reader = pq.ParquetFile(TMP_TEST)
X_t_chunks, y_t_chunks, dtd_chunks, cum_chunks, fp_chunks = [], [], [], [], []
for batch in reader.iter_batches(batch_size=CHUNK, columns=feature_cols + ['remaining_pax', 'dtd', 'pax_sold_cum', 'final_pax']):
    df_c = batch.to_pandas()
    for col in df_c.columns:
        if df_c[col].dtype == object or str(df_c[col].dtype).startswith('object'):
            df_c[col] = df_c[col].astype(float)
    X_t_chunks.append(df_c[feature_cols].values.astype(np.float32))
    y_t_chunks.append(df_c['remaining_pax'].values.astype(np.float32))
    dtd_chunks.append(df_c['dtd'].values.astype(np.float32))
    cum_chunks.append(df_c['pax_sold_cum'].values.astype(np.float32))
    fp_chunks.append(df_c['final_pax'].values.astype(np.float32))
    del df_c
    gc.collect()

X_test = np.vstack(X_t_chunks); del X_t_chunks
y_test = np.concatenate(y_t_chunks); del y_t_chunks
dtd_test = np.concatenate(dtd_chunks); del dtd_chunks
pax_cum_test = np.concatenate(cum_chunks); del cum_chunks
final_pax_test = np.concatenate(fp_chunks); del fp_chunks
gc.collect()

dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
y_pred = np.clip(bst.predict(dtest), 0, None)
del dtest, X_test
gc.collect()

# -- Genel metrikler --
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape_mask = y_test > 0
mape = np.mean(np.abs((y_test[mape_mask] - y_pred[mape_mask]) / y_test[mape_mask])) * 100
wape = np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test)) * 100 if np.sum(np.abs(y_test)) > 0 else 0.0

# -- Baseline: naive (ortalama remaining_pax per DTD) --
# Basit baseline: remaining = final_pax * (dtd / 180)
baseline_pred = final_pax_test * (dtd_test.astype(np.float32) / 180.0)
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

imp_mae = (1 - mae / baseline_mae) * 100
imp_rmse = (1 - rmse / baseline_rmse) * 100

# -- DTD bazli analiz --
dtd_buckets = [(1, 7), (8, 30), (31, 60), (61, 90), (91, 120), (121, 180)]

print(f"\n{'='*60}")
print(f"{'PICKUP XGBOOST SONUCLARI':^60}")
print(f"{'='*60}")
print(f"\n  Genel:")
print(f"    MAE:   {mae:.2f} yolcu")
print(f"    RMSE:  {rmse:.2f} yolcu")
print(f"    MAPE:  %{mape:.1f}
    WAPE:  %{wape:.1f}")
print(f"\n  Baseline (linear DTD):")
print(f"    MAE:   {baseline_mae:.2f}")
print(f"    RMSE:  {baseline_rmse:.2f}")
print(f"\n  Iyilestirme:")
print(f"    MAE:   {imp_mae:+.1f}% vs baseline")
print(f"    RMSE:  {imp_rmse:+.1f}% vs baseline")

print(f"\n  DTD Bazli MAE:")
print(f"    {'DTD Aralik':<15} {'MAE':>8} {'Ort. Remaining':>16} {'Hata %':>8}")
print(f"    {'-'*50}")
for lo, hi in dtd_buckets:
    mask = (dtd_test >= lo) & (dtd_test <= hi)
    if mask.sum() > 0:
        bucket_mae = mean_absolute_error(y_test[mask], y_pred[mask])
        bucket_avg = y_test[mask].mean()
        bucket_pct = (bucket_mae / bucket_avg * 100) if bucket_avg > 0 else 0
        print(f"    {lo:>3}-{hi:<3} gun      {bucket_mae:>8.2f} {bucket_avg:>16.1f} {bucket_pct:>7.1f}%")

# Feature importance top 15
print(f"\n  Top 15 Feature Importance:")
importance = bst.get_score(importance_type='gain')
for name, imp in sorted(importance.items(), key=lambda x: -x[1])[:15]:
    print(f"    {name:30s}: {imp:.1f}")

# ============================================
# KAYDET
# ============================================
print(f"\n  Kaydediliyor...", flush=True)

model_path = MODELS / "pickup_xgb.json"
feature_path = MODELS / "pickup_feature_list.json"
metrics_path = REPORTS / "pickup_xgb_metrics.json"

bst.save_model(str(model_path))

with open(feature_path, "w") as f:
    json.dump({"features": feature_cols}, f)

metrics = {
    "mae": round(float(mae), 4),
    "rmse": round(float(rmse), 4),
    "mape": round(float(mape), 2),
    "wape": round(float(wape), 2),
    "baseline_mae": round(float(baseline_mae), 4),
    "baseline_rmse": round(float(baseline_rmse), 4),
    "improvement_mae_pct": round(float(imp_mae), 1),
    "improvement_rmse_pct": round(float(imp_rmse), 1),
    "train_rows": int(train_n),
    "test_rows": int(test_n),
    "n_features": len(feature_cols),
    "n_estimators": 500,
    "target": "remaining_pax",
}
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

# Temp temizle
Path(TMP_TRAIN).unlink(missing_ok=True)
Path(TMP_TEST).unlink(missing_ok=True)

elapsed = time.time() - start
print(f"\n  Cikti:")
print(f"    {model_path}")
print(f"    {feature_path}")
print(f"    {metrics_path}")
print(f"\n  Toplam sure: {elapsed:.0f} saniye")
print(f"\n[TAMAMLANDI]")
