"""
train_tft_model.py — Temporal Fusion Transformer ile talep tahmini
pytorch-forecasting + pytorch-lightning
Kaggle GPU notebook'ta çalıştırılmak üzere tasarlanmıştır.
"""
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, mean_absolute_error

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import TweedieLoss
from pytorch_forecasting.data import GroupNormalizer

warnings.filterwarnings("ignore", ".*does not have many workers.*")

# ─── CONFIG ────────────────────────────────────────────────
BASE_DIR = Path(".")   # Kaggle'da /kaggle/working/ olacak
DATA_PATH = BASE_DIR / "tft_dataset.parquet"
CONFIG_PATH = BASE_DIR / "tft_dataset_config.json"
MODEL_DIR = BASE_DIR / "tft_model"
METRICS_PATH = BASE_DIR / "tft_metrics.json"

MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 256
MAX_EPOCHS = 30
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 64
ATTENTION_HEAD_SIZE = 4
DROPOUT = 0.1
PATIENCE = 5
RANDOM_SEED = 42

pl.seed_everything(RANDOM_SEED)

# ─── 1. Veri yükleme ─────────────────────────────────────
print("[1/5] Veri yükleniyor...")
df = pd.read_parquet(DATA_PATH)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

print(f"    Toplam: {len(df):,} satır, {df['group_id'].nunique():,} grup")

# Categorical sütunları string yap (TimeSeriesDataSet gereksinimi)
for col in config["static_categoricals"]:
    df[col] = df[col].astype(str)

# ─── 2. Train / Val / Test split ─────────────────────────
print("[2/5] Train/Val/Test ayrımı yapılıyor...")
train_df = df[df["dep_year"] == config["train_year"]].copy()
test_df = df[df["dep_year"] == config["test_year"]].copy()

# Validation: 2025 gruplarının %20'si
train_groups = train_df["group_id"].unique()
np.random.seed(RANDOM_SEED)
val_groups = np.random.choice(train_groups, size=int(len(train_groups) * 0.2), replace=False)
val_mask = train_df["group_id"].isin(val_groups)

val_df = train_df[val_mask].copy()
train_df = train_df[~val_mask].copy()

print(f"    Train:  {len(train_df):,} satır ({train_df['group_id'].nunique():,} grup)")
print(f"    Val:    {len(val_df):,} satır ({val_df['group_id'].nunique():,} grup)")
print(f"    Test:   {len(test_df):,} satır ({test_df['group_id'].nunique():,} grup)")

# ─── 3. TimeSeriesDataSet oluşturma ──────────────────────
print("[3/5] TimeSeriesDataSet oluşturuluyor...")

training_dataset = TimeSeriesDataSet(
    train_df,
    time_idx=config["time_idx"],
    target=config["target"],
    group_ids=config["group_ids"],
    max_encoder_length=config["max_encoder_length"],
    max_prediction_length=config["max_prediction_length"],
    static_categoricals=config["static_categoricals"],
    static_reals=config["static_reals"],
    time_varying_known_reals=config["time_varying_known_reals"],
    time_varying_unknown_reals=config["time_varying_unknown_reals"] + [config["target"]],
    target_normalizer=GroupNormalizer(
        groups=config["group_ids"],
        transformation="softplus",
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=False,
)

validation_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset, val_df, predict=True, stop_randomization=True
)

test_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset, test_df, predict=True, stop_randomization=True
)

# DataLoader'lar
train_loader = training_dataset.to_dataloader(
    train=True, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True
)
val_loader = validation_dataset.to_dataloader(
    train=False, batch_size=BATCH_SIZE * 2, num_workers=2, pin_memory=True
)
test_loader = test_dataset.to_dataloader(
    train=False, batch_size=BATCH_SIZE * 2, num_workers=2, pin_memory=True
)

print(f"    Training samples: {len(training_dataset):,}")
print(f"    Validation samples: {len(validation_dataset):,}")
print(f"    Test samples: {len(test_dataset):,}")

# ─── 4. Model eğitimi ────────────────────────────────────
print("[4/5] TFT modeli eğitiliyor...")

tft = TemporalFusionTransformer.from_dataset(
    training_dataset,
    hidden_size=HIDDEN_SIZE,
    attention_head_size=ATTENTION_HEAD_SIZE,
    dropout=DROPOUT,
    hidden_continuous_size=32,
    loss=TweedieLoss(p=1.5),
    learning_rate=LEARNING_RATE,
    output_size=7,   # 7 quantile
    log_interval=10,
    reduce_on_plateau_patience=4,
)

print(f"    Model parametreleri: {tft.size()/1e6:.1f}M")

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        min_delta=1e-4,
        mode="min",
        verbose=True,
    ),
    LearningRateMonitor(),
    ModelCheckpoint(
        dirpath=str(MODEL_DIR),
        filename="tft-best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    ),
]

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",
    devices=1,
    gradient_clip_val=0.1,
    callbacks=callbacks,
    enable_progress_bar=True,
    log_every_n_steps=50,
)

trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

# En iyi checkpoint'ı yükle
best_path = trainer.checkpoint_callback.best_model_path
print(f"    Best checkpoint: {best_path}")
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_path)

# Training dataset'i kaydet (inference için gerekli)
torch.save(training_dataset, MODEL_DIR / "tft_training_dataset.pt")

# ─── 5. Test seti değerlendirme ──────────────────────────
print("[5/5] Test seti değerlendiriliyor...")

# Tahminler
raw_predictions = best_tft.predict(test_loader, mode="raw", return_x=True)

# Quantile tahminler: shape [n_samples, pred_length, 7_quantiles]
pred_quantiles = raw_predictions.output["prediction"]

# Median (index 3) = ana tahmin
y_pred_all = pred_quantiles[:, :, 3].cpu().numpy()  # [n_samples, pred_length]
y_pred_flat = np.clip(y_pred_all.flatten(), 0, None)

# Gerçek değerler
actuals = torch.cat([y for x, (y, weight) in iter(test_loader)]).cpu().numpy()
y_true_flat = actuals.flatten()

# Boyut eşitle (TFT bazen farklı boyut dönebilir)
min_len = min(len(y_pred_flat), len(y_true_flat))
y_pred_flat = y_pred_flat[:min_len]
y_true_flat = y_true_flat[:min_len]

# Metrikler
mae = float(mean_absolute_error(y_true_flat, y_pred_flat))
rmse = float(np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2)))

# AUC eşdeğeri: sıfır vs pozitif ayrımı
y_bin = (y_true_flat > 0).astype(int)
p_sale_proxy = np.clip(y_pred_flat, 0, None)
try:
    auc = float(roc_auc_score(y_bin, p_sale_proxy))
except ValueError:
    auc = None

print(f"\n{'='*50}")
print(f"TEST METRİKLERİ")
print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  AUC:  {auc:.4f}" if auc else "  AUC:  N/A")
print(f"{'='*50}")

# Quantile istatistikleri
y_low = np.clip(pred_quantiles[:, :, 1].cpu().numpy().flatten()[:min_len], 0, None)   # %10
y_high = np.clip(pred_quantiles[:, :, 5].cpu().numpy().flatten()[:min_len], 0, None)  # %90
coverage_90 = float(np.mean((y_true_flat >= y_low) & (y_true_flat <= y_high)))
print(f"  %90 güven aralığı kapsama: {coverage_90:.1%}")

# ─── Metrikleri kaydet ────────────────────────────────────
metrics = {
    "train_year": config["train_year"],
    "test_year": config["test_year"],
    "rows_train": int(len(train_df)),
    "rows_val": int(len(val_df)),
    "rows_test": int(len(test_df)),
    "zero_rate_test": float((y_true_flat == 0).mean()),
    "tft_model": {
        "mae": mae,
        "rmse": rmse,
        "auc_equivalent": auc,
        "coverage_90pct": coverage_90,
    },
    "xgb_baseline": {
        "mae": 0.7805,
        "rmse": 1.3262,
        "auc": 0.8354,
    },
    "improvement": {
        "mae_pct": round((0.7805 - mae) / 0.7805 * 100, 2),
        "rmse_pct": round((1.3262 - rmse) / 1.3262 * 100, 2),
    },
    "hyperparameters": {
        "hidden_size": HIDDEN_SIZE,
        "attention_head_size": ATTENTION_HEAD_SIZE,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "max_encoder_length": config["max_encoder_length"],
        "max_prediction_length": config["max_prediction_length"],
        "loss": "TweedieLoss(p=1.5)",
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "best_epoch": trainer.current_epoch,
    },
    "artifacts": {
        "checkpoint": str(Path(best_path).name),
        "dataset_params": "tft_training_dataset.pt",
        "config": str(CONFIG_PATH.name),
    },
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print(f"\ntft_metrics.json kaydedildi.")
print(f"Model checkpoint: {best_path}")

# ─── XGBoost karşılaştırma özeti ─────────────────────────
print(f"\n{'='*50}")
print("XGBoost vs TFT KARŞILAŞTIRMA")
print(f"{'='*50}")
print(f"  {'Metrik':<10} {'XGBoost':>10} {'TFT':>10} {'Fark':>10}")
print(f"  {'MAE':<10} {0.7805:>10.4f} {mae:>10.4f} {(0.7805-mae):>+10.4f}")
print(f"  {'RMSE':<10} {1.3262:>10.4f} {rmse:>10.4f} {(1.3262-rmse):>+10.4f}")
if auc:
    print(f"  {'AUC':<10} {0.8354:>10.4f} {auc:>10.4f} {(auc-0.8354):>+10.4f}")
print(f"{'='*50}")
