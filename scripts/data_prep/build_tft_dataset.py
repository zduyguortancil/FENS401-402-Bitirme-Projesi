"""
build_tft_dataset.py — demand_training.parquet → TFT-uyumlu dataset
Grup bazlı stratified örnekleme + lag features + event tag temizliği.
Çıktı: tft_dataset.parquet + tft_dataset_config.json
"""
import json
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
INPUT_PATH = BASE_DIR / "demand_training.parquet"
OUTPUT_PATH = BASE_DIR / "tft_dataset.parquet"
CONFIG_PATH = BASE_DIR / "tft_dataset_config.json"

SAMPLE_FRAC = 0.15          # grup bazlı örnekleme oranı
RANDOM_SEED = 42
DTD_MAX = 180

# ─── 1. DuckDB ile oku + temizle ──────────────────────────
print("[1/6] DuckDB ile veri okunuyor...")
con = duckdb.connect()

# _1 duplikat sütunları ve primary_event çıkar, tag'leri INT'e çevir
tag_cols = [
    "tag_yaz_tatili", "tag_kis_tatili", "tag_yariyil_tatili",
    "tag_bahar_tatili", "tag_ramazan_donemi", "tag_bayram_donemi",
    "tag_yilbasi", "tag_sevgililer_gunu", "tag_futbol_sezonu",
    "tag_kongre_fuar", "tag_ski_sezonu", "tag_festival_sezonu",
    "tag_is_seyahati_yogun", "tag_hac_umre", "tag_gece_ucusu",
]

tag_cast = ", ".join(f"CAST({t} AS INT) AS {t}" for t in tag_cols)

sql = f"""
    SELECT
        flight_id, cabin_class, dtd,
        y_pax_sold_today,
        pax_sold_cum, pax_last_7d,
        capacity, remaining_seats, load_factor,
        region, distance_km, flight_time_min,
        dep_year, dep_month, dep_dow, dep_hour,
        ff_gold_pct, ff_elite_pct,
        {tag_cast},
        flight_id || '__' || cabin_class AS group_id,
        ({DTD_MAX} - dtd) AS time_idx
    FROM read_parquet('{INPUT_PATH}')
    WHERE dep_year IN (2025, 2026)
      AND dtd BETWEEN 0 AND {DTD_MAX}
"""

con.execute(f"CREATE TABLE raw AS {sql}")
total_rows = con.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
total_groups = con.execute("SELECT COUNT(DISTINCT group_id) FROM raw").fetchone()[0]
print(f"    Toplam: {total_rows:,} satır, {total_groups:,} grup")

# ─── 2. Stratified grup örneklemesi ───────────────────────
print(f"[2/6] Stratified örnekleme (%{int(SAMPLE_FRAC*100)})...")

# Her grubun region, cabin, çeyrek bilgisini al
con.execute("""
    CREATE TABLE group_meta AS
    SELECT DISTINCT group_id, region, cabin_class,
           CASE WHEN dep_month <= 3 THEN 1
                WHEN dep_month <= 6 THEN 2
                WHEN dep_month <= 9 THEN 3
                ELSE 4 END AS quarter,
           dep_year
    FROM raw
""")

# Stratified örnekleme: her (region, cabin, quarter) katmanından %15
con.execute(f"""
    CREATE TABLE sampled_groups AS
    SELECT group_id FROM group_meta
    USING SAMPLE 15 PERCENT (bernoulli)
""")

n_sampled = con.execute("SELECT COUNT(*) FROM sampled_groups").fetchone()[0]
print(f"    Örneklenen grup: {n_sampled:,} / {total_groups:,}")

# ─── 3. Pandas'a aktar ───────────────────────────────────
print("[3/6] Örneklenen veriler pandas'a aktarılıyor...")
df = con.execute("""
    SELECT r.*
    FROM raw r
    SEMI JOIN sampled_groups sg ON r.group_id = sg.group_id
    ORDER BY r.group_id, r.time_idx
""").fetchdf()

con.close()
print(f"    Boyut: {len(df):,} satır")

# ─── 4. Lag features ─────────────────────────────────────
print("[4/6] Lag features hesaplanıyor...")
df = df.sort_values(["group_id", "time_idx"]).reset_index(drop=True)

g = df.groupby("group_id", sort=False)
df["pax_sold_today_lag1"] = g["y_pax_sold_today"].shift(1).fillna(0.0)
df["pax_sold_today_lag7"] = g["y_pax_sold_today"].shift(7).fillna(0.0)
df["pax_cum_diff7"] = (df["pax_sold_cum"] - g["pax_sold_cum"].shift(7)).fillna(0.0)

# ─── 5. Ek feature'lar ───────────────────────────────────
print("[5/6] Ek feature'lar ekleniyor...")
df["is_weekend"] = (df["dep_dow"] >= 5).astype(int)

# Eksik zaman adımı olan grupları kontrol et ve at
steps_per_group = df.groupby("group_id")["time_idx"].nunique()
complete_groups = steps_per_group[steps_per_group == DTD_MAX + 1].index
n_before = df["group_id"].nunique()
df = df[df["group_id"].isin(complete_groups)].reset_index(drop=True)
n_after = df["group_id"].nunique()
if n_before != n_after:
    print(f"    Eksik adımlı {n_before - n_after} grup çıkarıldı")
else:
    print(f"    Tüm gruplar tam (181 adım)")

# ─── 6. Kaydet ────────────────────────────────────────────
print("[6/6] Kaydediliyor...")

# Categorical sütunları string olarak zorla (TFT gereksinimi)
df["cabin_class"] = df["cabin_class"].astype(str)
df["region"] = df["region"].astype(str)

df.to_parquet(OUTPUT_PATH, compression="zstd", index=False)

# TFT config
config = {
    "time_idx": "time_idx",
    "target": "y_pax_sold_today",
    "group_ids": ["group_id"],
    "static_categoricals": ["cabin_class", "region"],
    "static_reals": ["capacity", "distance_km", "flight_time_min"],
    "time_varying_known_reals": [
        "dtd", "dep_month", "dep_dow", "dep_hour", "is_weekend",
    ] + tag_cols,
    "time_varying_unknown_reals": [
        "pax_sold_cum", "pax_last_7d", "load_factor",
        "remaining_seats", "ff_gold_pct", "ff_elite_pct",
        "pax_sold_today_lag1", "pax_sold_today_lag7", "pax_cum_diff7",
    ],
    "max_encoder_length": 150,
    "max_prediction_length": 31,
    "dtd_max": DTD_MAX,
    "sample_frac": SAMPLE_FRAC,
    "train_year": 2025,
    "test_year": 2026,
    "total_groups": int(n_after),
    "total_rows": int(len(df)),
}

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"\n{'='*50}")
print(f"tft_dataset.parquet  : {len(df):,} satır, {n_after:,} grup")
print(f"tft_dataset_config.json kaydedildi")
print(f"{'='*50}")
