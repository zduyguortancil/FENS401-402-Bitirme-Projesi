"""
Kaggle GPU Notebook — TFT Talep Tahmini (v2 — hazır dataset)
==============================================================
1. Kaggle'da "New Dataset" ile tft_dataset.parquet + tft_dataset_config.json yükle
2. "New Notebook" → dataset'i ekle → GPU aç → Internet On
3. Bu kodu yapıştır, INPUT_DIR'i güncelle, Run All (~2-3 saat)
"""

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 1: Bağımlılık kurulumu                            ║
# ╚══════════════════════════════════════════════════════════╝
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "pytorch-forecasting", "lightning"])

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 2: Import'lar ve Config                           ║
# ╚══════════════════════════════════════════════════════════╝
import json, warnings, gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, mean_absolute_error

import torch
try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer

warnings.filterwarnings("ignore", ".*does not have many workers.*")

# ─── Kaggle path'leri ────────────────────────────────────
# Dataset'i yükledikten sonra bu path'i güncelle:
INPUT_DIR = Path("/kaggle/input/datasets/afgokbullut/ptir-demand-data")  # ← BUNU GÜNCELLE
WORK_DIR = Path("/kaggle/working")
MODEL_DIR = WORK_DIR / "tft_model"
MODEL_DIR.mkdir(exist_ok=True)

DATA_PATH = INPUT_DIR / "tft_dataset.parquet"
CONFIG_PATH = INPUT_DIR / "tft_dataset_config.json"

assert DATA_PATH.exists(), f"Dosya bulunamadı: {DATA_PATH}. Dataset adını kontrol et!"
assert CONFIG_PATH.exists(), f"Config bulunamadı: {CONFIG_PATH}"

# ─── Hyperparametreler ────────────────────────────────────
BATCH_SIZE = 128
MAX_EPOCHS = 30
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 64
ATTENTION_HEAD_SIZE = 4
DROPOUT = 0.1
PATIENCE = 5
RANDOM_SEED = 42

pl.seed_everything(RANDOM_SEED)

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 3: Veri Yükleme (DuckDB yok, direkt parquet)     ║
# ╚══════════════════════════════════════════════════════════╝
print("=" * 60)
print("ADIM 1: Veri Yükleme")
print("=" * 60)

print("[1/2] Dataset okunuyor...")
df = pd.read_parquet(DATA_PATH)
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

# Categorical'ları string yap
for col in config["static_categoricals"]:
    df[col] = df[col].astype(str)

print(f"    {len(df):,} satır, {df['group_id'].nunique():,} grup")
print(f"    Sütun sayısı: {len(df.columns)}")

# Train / Val / Test split
print("[2/2] Train/Val/Test ayrımı...")
train_df = df[df["dep_year"] == config["train_year"]].copy()
test_df = df[df["dep_year"] == config["test_year"]].copy()

train_groups = train_df["group_id"].unique()
np.random.seed(RANDOM_SEED)
val_groups = np.random.choice(train_groups, size=int(len(train_groups) * 0.2), replace=False)
val_mask = train_df["group_id"].isin(val_groups)
val_df = train_df[val_mask].copy()
train_df = train_df[~val_mask].copy()

print(f"    Train: {len(train_df):,} satır ({train_df['group_id'].nunique():,} grup)")
print(f"    Val:   {len(val_df):,} satır ({val_df['group_id'].nunique():,} grup)")
print(f"    Test:  {len(test_df):,} satır ({test_df['group_id'].nunique():,} grup)")

del df; gc.collect()

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 4: TimeSeriesDataSet + TFT Eğitimi               ║
# ╚══════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("ADIM 2: TFT Model Eğitimi")
print("=" * 60)

print("[1/3] TimeSeriesDataSet oluşturuluyor...")

# group_id encoder: bilinmeyen kategorileri (val/test grupları) kabul etsin
from pytorch_forecasting.data.encoders import NaNLabelEncoder
categorical_encoders = {"group_id": NaNLabelEncoder(add_nan=True)}
for col in config["static_categoricals"]:
    categorical_encoders[col] = NaNLabelEncoder(add_nan=True)

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
    categorical_encoders=categorical_encoders,
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

train_loader = training_dataset.to_dataloader(
    train=True, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True
)
val_loader = validation_dataset.to_dataloader(
    train=False, batch_size=BATCH_SIZE * 2, num_workers=0, pin_memory=True
)
test_loader = test_dataset.to_dataloader(
    train=False, batch_size=BATCH_SIZE * 2, num_workers=0, pin_memory=True
)

print(f"    Training samples: {len(training_dataset):,}")

# Model
print("[2/3] TFT modeli eğitiliyor...")
tft = TemporalFusionTransformer.from_dataset(
    training_dataset,
    hidden_size=HIDDEN_SIZE,
    attention_head_size=ATTENTION_HEAD_SIZE,
    dropout=DROPOUT,
    hidden_continuous_size=32,
    loss=QuantileLoss(quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]),
    learning_rate=LEARNING_RATE,
    log_interval=10,
    reduce_on_plateau_patience=4,
)
print(f"    Parametre sayısı: {tft.size()/1e6:.1f}M")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=PATIENCE, min_delta=1e-4, mode="min", verbose=True),
    LearningRateMonitor(),
    ModelCheckpoint(dirpath=str(MODEL_DIR), filename="tft-best", monitor="val_loss", mode="min", save_top_k=1),
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

best_path = trainer.checkpoint_callback.best_model_path
print(f"    Best checkpoint: {best_path}")
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_path)

print("[3/3] Training dataset kaydediliyor...")
torch.save(training_dataset, MODEL_DIR / "tft_training_dataset.pt")

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 5: Test Değerlendirme                             ║
# ╚══════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("ADIM 3: Test Değerlendirme")
print("=" * 60)

# Actuals
y_true_list = []
for x, (y, weight) in iter(test_loader):
    y_true_list.append(y)
y_true_flat = torch.cat(y_true_list).cpu().numpy().flatten()

# Quantile tahminler
predictions = best_tft.predict(test_loader, mode="quantiles")

if isinstance(predictions, dict):
    pred_quantiles = predictions["prediction"]
elif isinstance(predictions, torch.Tensor):
    pred_quantiles = predictions
else:
    pred_quantiles = predictions.output if hasattr(predictions, "output") else predictions

y_pred_median = np.clip(pred_quantiles[:, :, 3].cpu().numpy().flatten(), 0, None)

min_len = min(len(y_pred_median), len(y_true_flat))
y_pred_flat = y_pred_median[:min_len]
y_true_flat = y_true_flat[:min_len]

mae = float(mean_absolute_error(y_true_flat, y_pred_flat))
rmse = float(np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2)))

y_bin = (y_true_flat > 0).astype(int)
try:
    auc = float(roc_auc_score(y_bin, np.clip(y_pred_flat, 0, None)))
except ValueError:
    auc = None

y_low = np.clip(pred_quantiles[:, :, 1].cpu().numpy().flatten()[:min_len], 0, None)
y_high = np.clip(pred_quantiles[:, :, 5].cpu().numpy().flatten()[:min_len], 0, None)
coverage_90 = float(np.mean((y_true_flat >= y_low) & (y_true_flat <= y_high)))

# ╔══════════════════════════════════════════════════════════╗
# ║  CELL 6: Sonuçlar ve Kaydetme                           ║
# ╚══════════════════════════════════════════════════════════╝
# Config'i working'e de kopyala (indirmek için)
import shutil
shutil.copy2(CONFIG_PATH, WORK_DIR / "tft_dataset_config.json")

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
        "loss": "QuantileLoss(7 quantiles)",
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "best_epoch": trainer.current_epoch,
    },
    "artifacts": {
        "checkpoint": "tft-best.ckpt",
        "dataset_params": "tft_training_dataset.pt",
        "config": "tft_dataset_config.json",
    },
}

metrics_path = WORK_DIR / "tft_metrics.json"
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print("SONUÇLAR")
print(f"{'='*60}")
print(f"  {'Metrik':<12} {'XGBoost':>10} {'TFT':>10} {'Fark':>10}")
print(f"  {'---':<12} {'---':>10} {'---':>10} {'---':>10}")
print(f"  {'MAE':<12} {0.7805:>10.4f} {mae:>10.4f} {(0.7805-mae):>+10.4f}")
print(f"  {'RMSE':<12} {1.3262:>10.4f} {rmse:>10.4f} {(1.3262-rmse):>+10.4f}")
if auc:
    print(f"  {'AUC':<12} {0.8354:>10.4f} {auc:>10.4f} {(auc-0.8354):>+10.4f}")
print(f"  {'Coverage':<12} {'---':>10} {coverage_90:>9.1%} {'---':>10}")
print(f"{'='*60}")
print(f"\nMAE iyilesme:  %{metrics['improvement']['mae_pct']}")
print(f"RMSE iyilesme: %{metrics['improvement']['rmse_pct']}")
print(f"\nCiktilar:")
print(f"  {MODEL_DIR}/tft-best.ckpt")
print(f"  {MODEL_DIR}/tft_training_dataset.pt")
print(f"  {WORK_DIR}/tft_dataset_config.json")
print(f"  {metrics_path}")
print(f"\nBu 4 dosyayi indirip projeye koy!")
