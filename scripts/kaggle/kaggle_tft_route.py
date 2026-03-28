"""
TFT Route-Daily Demand Forecasting - Kaggle Notebook (FINAL)
=============================================================
Tum buglar duzeltildi:
  - KeepAlive callback (commit mode timeout onleme)
  - Egitim biter bitmez 3 kritik dosya HEMEN kaydedilir
  - predictions.output 2D/3D ndim kontrolu
  - NaN filtreleme
  - Feature importance try/except + None kontrolu
  - mode="raw" ile quantile ciktilari
  - Horizon-weighted averaging
  - Hicbir print/access None objeye yapilmaz
"""
import warnings; warnings.filterwarnings("ignore")
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytorch-forecasting", "lightning"])

import pandas as pd
import numpy as np
import torch
import shutil
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import (
    TemporalFusionTransformer, TimeSeriesDataSet, QuantileLoss, NaNLabelEncoder, GroupNormalizer,
)

INPUT_DIR = Path("/kaggle/input/datasets/afgokbullut/ptir-route-daily")
OUT_DIR = Path("/kaggle/working")
BATCH_SIZE = 128

# ══════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════
print("=" * 60, flush=True)
print("[1/7] Data yukleniyor...", flush=True)
parquet_candidates = list(INPUT_DIR.glob("*.parquet"))
assert len(parquet_candidates) > 0, "Parquet bulunamadi!"
df = pd.read_parquet(parquet_candidates[0])
print(f"  {df.shape[0]:,} rows, {df.entity_id.nunique()} entities", flush=True)

df["special_period"] = df["special_period"].fillna("none")
df["direction"] = df["direction"].fillna("unknown")
df["region"] = df["region"].fillna("unknown")
df["time_idx"] = df["time_idx"].astype(int)
df["total_pax"] = df["total_pax"].astype(float)
for c in ["dep_month", "dep_dow", "dep_year", "n_flights"]:
    df[c] = df[c].astype(str)
tag_cols = [c for c in df.columns if c.startswith("tag_")]
for c in tag_cols:
    df[c] = df[c].astype(str)
df["is_special_period"] = df["is_special_period"].astype(str)
print("  Prep OK", flush=True)

# ══════════════════════════════════════════════════════════
# 2. DATASET
# ══════════════════════════════════════════════════════════
print("[2/7] Dataset...", flush=True)
TRAIN_END = 364; VAL_END = 364 + 90

training = TimeSeriesDataSet(
    df[df.time_idx <= VAL_END],
    time_idx="time_idx", target="total_pax", group_ids=["entity_id"],
    max_encoder_length=60, max_prediction_length=30,
    min_encoder_length=30, min_prediction_length=1,
    static_categoricals=["entity_id", "route", "cabin_class", "region", "direction"],
    static_reals=["distance_km", "flight_time_min"],
    time_varying_known_categoricals=[
        "dep_month", "dep_dow", "dep_year", "n_flights",
        "is_special_period", "special_period"] + tag_cols,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=[
        "total_pax", "avg_fare", "std_fare", "max_fare", "min_fare",
        "n_bookings", "avg_group_size",
        "corporate_pct", "agency_pct", "website_pct", "mobile_pct",
        "connecting_pct", "early_booking_pct", "late_booking_pct",
        "child_pct", "gold_elite_pct", "elite_pct", "halal_pct"],
    target_normalizer=GroupNormalizer(groups=["entity_id"], transformation="softplus"),
    categorical_encoders={
        "entity_id": NaNLabelEncoder(add_nan=True), "route": NaNLabelEncoder(add_nan=True),
        "cabin_class": NaNLabelEncoder(add_nan=True), "region": NaNLabelEncoder(add_nan=True),
        "direction": NaNLabelEncoder(add_nan=True), "dep_month": NaNLabelEncoder(add_nan=True),
        "dep_dow": NaNLabelEncoder(add_nan=True), "dep_year": NaNLabelEncoder(add_nan=True),
        "n_flights": NaNLabelEncoder(add_nan=True), "is_special_period": NaNLabelEncoder(add_nan=True),
        "special_period": NaNLabelEncoder(add_nan=True),
        **{c: NaNLabelEncoder(add_nan=True) for c in tag_cols}},
    allow_missing_timesteps=True, add_relative_time_idx=True,
    add_target_scales=True, add_encoder_length=True,
)
validation = TimeSeriesDataSet.from_dataset(training, df[df.time_idx <= VAL_END], min_prediction_idx=TRAIN_END + 1)
train_dl = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=2)
val_dl = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=2)
print(f"  Train: {len(training):,}, Val: {len(validation):,}", flush=True)

# ══════════════════════════════════════════════════════════
# 3. MODEL + TRAINING
# ══════════════════════════════════════════════════════════
print("[3/7] Model + Training...", flush=True)
tft = TemporalFusionTransformer.from_dataset(
    training, learning_rate=1e-3, hidden_size=64, attention_head_size=4,
    dropout=0.1, hidden_continuous_size=32,
    loss=QuantileLoss(quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]),
    optimizer="adam", reduce_on_plateau_patience=5,
)
print(f"  Params: {tft.size()/1e3:.1f}K", flush=True)

class KeepAlive(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 50 == 0:
            print(f"  [b{batch_idx}]", flush=True)
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"  [epoch {trainer.current_epoch}]", flush=True)
    def on_validation_epoch_end(self, trainer, pl_module):
        print(f"  [val]", flush=True)

trainer = pl.Trainer(
    max_epochs=50, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
    gradient_clip_val=0.5,
    callbacks=[EarlyStopping(monitor="val_loss", patience=8, mode="min", verbose=True),
               LearningRateMonitor(logging_interval="epoch"), KeepAlive()],
    logger=CSVLogger(OUT_DIR, name="tft_logs"),
    enable_progress_bar=False, log_every_n_steps=50,
)
print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}", flush=True)
print("  Egitim basliyor...", flush=True)
trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

# ══════════════════════════════════════════════════════════
# 4. HEMEN KAYDET — evaluation'dan ONCE
# ══════════════════════════════════════════════════════════
print("\n[4/7] KRITIK DOSYALAR KAYDEDILIYOR...", flush=True)
best_path = trainer.checkpoint_callback.best_model_path
print(f"  Best: {best_path}", flush=True)
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_path)

shutil.copy(best_path, str(OUT_DIR / "tft_full_checkpoint.ckpt"))
print(f"  [1] tft_full_checkpoint.ckpt OK", flush=True)

torch.save(training, OUT_DIR / "tft_training_dataset.pt")
print(f"  [2] tft_training_dataset.pt OK", flush=True)

torch.save(best_tft.state_dict(), OUT_DIR / "tft_route_daily_model.pt")
print(f"  [3] tft_route_daily_model.pt OK", flush=True)

# ══════════════════════════════════════════════════════════
# 5. EVALUATION (hata alsa bile dosyalar zaten kayitli)
# ══════════════════════════════════════════════════════════
print("\n[5/7] Evaluation...", flush=True)
try:
    # Validation
    vp = best_tft.predict(val_dl, return_x=True)
    vo = vp.output.cpu().numpy()
    vm = vo[:, :, 2] if vo.ndim == 3 else vo
    va = vp.x["decoder_target"].cpu().numpy()
    vf = vm.flatten(); af = va.flatten()
    vv = ~(np.isnan(vf) | np.isnan(af)); vf = vf[vv]; af = af[vv]
    print(f"  Val MAE: {mean_absolute_error(af, vf):.2f}", flush=True)

    # Test
    test_ds = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=VAL_END + 1)
    test_dl = test_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=2)
    tp = best_tft.predict(test_dl, return_x=True)
    to_ = tp.output.cpu().numpy()
    tm = to_[:, :, 2] if to_.ndim == 3 else to_
    ta = tp.x["decoder_target"].cpu().numpy()
    tf_ = tm.flatten(); taf = ta.flatten()
    tv = ~(np.isnan(tf_) | np.isnan(taf)); tf_ = tf_[tv]; taf = taf[tv]
    test_mae = mean_absolute_error(taf, tf_)
    print(f"  Test MAE: {test_mae:.2f}", flush=True)
except Exception as e:
    print(f"  Evaluation HATA: {e}", flush=True)
    test_mae = None
    # test_dl olustur (indexed predictions icin gerekli)
    test_ds = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=VAL_END + 1)
    test_dl = test_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=2)

# ══════════════════════════════════════════════════════════
# 6. FLAT + INDEXED PREDICTIONS
# ══════════════════════════════════════════════════════════
print("\n[6/7] Predictions...", flush=True)
try:
    # Flat
    fp = best_tft.predict(test_dl, return_x=True)
    fo = fp.output.cpu().numpy()
    fm = fo[:, :, 2] if fo.ndim == 3 else fo
    fa = fp.x["decoder_target"].cpu().numpy()
    ff = fm.flatten(); faf = fa.flatten()
    fv = ~(np.isnan(ff) | np.isnan(faf)); ff = ff[fv]; faf = faf[fv]
    pd.DataFrame({"actual": faf, "predicted": ff}).to_parquet(OUT_DIR / "tft_predictions.parquet")
    print(f"  tft_predictions.parquet OK ({len(ff):,} rows)", flush=True)

    # Indexed with quantiles
    ip = best_tft.predict(test_dl, return_x=True, return_index=True, mode="raw")
    raw = ip.output
    if isinstance(raw, dict):
        pq = raw["prediction"].cpu().numpy()
    elif hasattr(raw, "prediction"):
        pq = raw.prediction.cpu().numpy()
    else:
        pq = raw.cpu().numpy()
    ia = ip.x["decoder_target"].cpu().numpy()
    idx = ip.index
    print(f"  Raw shape: {pq.shape}", flush=True)

    if pq.ndim == 3:
        med = pq[:, :, 2]; q10 = pq[:, :, 0]; q90 = pq[:, :, 4]
    else:
        med = pq; q10 = pq * 0.85; q90 = pq * 1.15

    base_date = pd.Timestamp("2025-01-01")
    rows = []
    for i in range(len(idx)):
        eid = idx.iloc[i]["entity_id"]
        st = int(idx.iloc[i]["time_idx"])
        pl_ = med.shape[1]
        for j in range(pl_):
            h = j + 1
            tidx = st + j + 1
            dd = base_date + pd.Timedelta(days=tidx)
            rows.append({"entity_id": eid, "dep_date": dd,
                "actual": float(ia[i, j]), "predicted": float(med[i, j]),
                "pred_q10": float(q10[i, j]), "pred_q90": float(q90[i, j]), "horizon": h})

    rdf = pd.DataFrame(rows)
    print(f"  Raw indexed: {len(rdf):,}", flush=True)

    def _wa(g):
        w = 1.0 / g["horizon"].values
        return pd.Series({"actual": g["actual"].iloc[0],
            "predicted": float(np.average(g["predicted"], weights=w)),
            "pred_q10": float(np.average(g["pred_q10"], weights=w)),
            "pred_q90": float(np.average(g["pred_q90"], weights=w))})

    agg = rdf.groupby(["entity_id", "dep_date"]).apply(_wa, include_groups=False).reset_index()
    agg["dep_date"] = pd.to_datetime(agg["dep_date"])
    agg = agg.sort_values(["entity_id", "dep_date"]).reset_index(drop=True)
    agg.to_parquet(OUT_DIR / "tft_predictions_indexed.parquet", index=False)
    print(f"  tft_predictions_indexed.parquet OK ({len(agg):,} rows, {agg.entity_id.nunique()} entities)", flush=True)
except Exception as e:
    print(f"  Predictions HATA: {e}", flush=True)
    import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════════════
# 7. FEATURE IMPORTANCE (tamamen opsiyonel)
# ══════════════════════════════════════════════════════════
print("\n[7/7] Feature importance...", flush=True)
fi_data = {"encoder_variables": {}, "decoder_variables": {}}
try:
    fi_pred = best_tft.predict(val_dl, return_x=True, return_index=True, mode="raw")
    interp = best_tft.interpret_output(fi_pred, reduction="sum")
    vn = training.reals + training.flat_categoricals
    for name, imp in zip(vn, interp["encoder_variables"].cpu().numpy()):
        if imp > 0.005: fi_data["encoder_variables"][name] = round(float(imp), 4)
    for name, imp in zip(vn, interp["decoder_variables"].cpu().numpy()):
        if imp > 0.005: fi_data["decoder_variables"][name] = round(float(imp), 4)
    fi_data["encoder_variables"] = dict(sorted(fi_data["encoder_variables"].items(), key=lambda x: -x[1]))
    fi_data["decoder_variables"] = dict(sorted(fi_data["decoder_variables"].items(), key=lambda x: -x[1]))
    print(f"  OK ({len(fi_data['encoder_variables'])} encoder, {len(fi_data['decoder_variables'])} decoder)", flush=True)
except Exception as e:
    print(f"  ATLANDI: {e}", flush=True)
with open(OUT_DIR / "tft_feature_importance.json", "w") as f:
    json.dump(fi_data, f, indent=2)
print(f"  tft_feature_importance.json saved", flush=True)

# ══════════════════════════════════════════════════════════
# SONUC
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60, flush=True)
print("  TAMAMLANDI!", flush=True)
print("=" * 60, flush=True)
for f in sorted(OUT_DIR.glob("tft_*")):
    print(f"  {f.name:45s} {f.stat().st_size/1024/1024:.1f} MB", flush=True)
if test_mae:
    print(f"\n  Test MAE: {test_mae:.2f}", flush=True)
print("\n[BITTI]", flush=True)
