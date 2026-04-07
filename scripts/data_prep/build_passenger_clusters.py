"""
Sprint 5 - Passenger Segmentation via K-Means Clustering
Aggregates flight-cabin level behavioral features from demand_training.parquet,
then clusters them into meaningful passenger segments.
"""
import json
import numpy as np
import duckdb
from pathlib import Path

BASE_DIR = Path(__file__).parent
TRAIN_PATH = BASE_DIR / "demand_training.parquet"
META_PATH  = BASE_DIR / "flight_metadata.parquet"
OUT_PATH   = BASE_DIR / "passenger_clusters.parquet"
OUT_RPT    = BASE_DIR / "cluster_report.json"

con = duckdb.connect()

# ═══════════════════════════════════════════════════════════
# STEP 1: Feature engineering at (flight_id, cabin_class) level
# ═══════════════════════════════════════════════════════════
print("[1/5] Building flight-cabin profiles via DuckDB...", flush=True)

con.execute(f"""
    CREATE TABLE profiles AS
    SELECT
        flight_id,
        cabin_class,

        -- Timing behavior: when do passengers buy?
        AVG(dtd)                                          AS avg_dtd,
        MEDIAN(dtd)                                       AS median_dtd,
        AVG(CASE WHEN y_pax_sold_today > 0 THEN dtd END) AS avg_dtd_at_purchase,

        -- Last-minute vs early-bird ratios
        SUM(CASE WHEN dtd <= 7  THEN y_pax_sold_today ELSE 0 END) * 1.0
            / NULLIF(SUM(y_pax_sold_today), 0)            AS pct_last_minute,
        SUM(CASE WHEN dtd >= 30 THEN y_pax_sold_today ELSE 0 END) * 1.0
            / NULLIF(SUM(y_pax_sold_today), 0)            AS pct_early_bird,

        -- Demand intensity
        SUM(y_pax_sold_today)                              AS total_pax,
        AVG(y_pax_sold_today)                              AS avg_daily_pax,
        MAX(pax_sold_cum)                                  AS final_pax_cum,
        AVG(pax_last_7d)                                   AS avg_pax_7d,

        -- Load factor (take the last snapshot = dtd 0)
        MAX(load_factor)                                   AS max_load_factor,

        -- Loyalty / FF
        AVG(ff_gold_pct)                                   AS ff_gold_avg,
        AVG(ff_elite_pct)                                  AS ff_elite_avg,

        -- Cabin
        CASE WHEN LOWER(cabin_class) = 'business' THEN 1 ELSE 0 END AS is_business,

        -- Capacity
        MAX(capacity)                                      AS capacity,

        -- Route features
        MAX(distance_km)                                   AS distance_km,
        MAX(flight_time_min)                               AS flight_time_min,

        -- Calendar features
        MAX(dep_month)                                     AS dep_month,
        MAX(dep_dow)                                       AS dep_dow,
        MAX(dep_hour)                                      AS dep_hour,

        -- Derived calendar
        CASE WHEN MAX(dep_dow) < 5 THEN 1 ELSE 0 END      AS is_weekday,
        CASE WHEN MAX(dep_hour) BETWEEN 6 AND 10
             THEN 1 ELSE 0 END                             AS is_morning_flight,

        -- Region one-hot
        CASE WHEN MAX(region) = 'Europe'      THEN 1 ELSE 0 END AS region_europe,
        CASE WHEN MAX(region) = 'Asia'        THEN 1 ELSE 0 END AS region_asia,
        CASE WHEN MAX(region) = 'Americas'    THEN 1 ELSE 0 END AS region_americas,
        CASE WHEN MAX(region) = 'Middle East' THEN 1 ELSE 0 END AS region_mideast,
        CASE WHEN MAX(region) = 'Africa'      THEN 1 ELSE 0 END AS region_africa

    FROM read_parquet('{TRAIN_PATH}')
    WHERE flight_id IS NOT NULL
      AND cabin_class IS NOT NULL
    GROUP BY flight_id, cabin_class
""")

profile_count = con.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
print(f"   → {profile_count:,} flight-cabin profiles created", flush=True)

# ═══════════════════════════════════════════════════════════
# STEP 2: Pull into numpy for clustering
# ═══════════════════════════════════════════════════════════
print("[2/5] Loading features for clustering...", flush=True)

CLUSTER_FEATURES = [
    "avg_dtd_at_purchase",
    "pct_last_minute",
    "pct_early_bird",
    "avg_daily_pax",
    "max_load_factor",
    "ff_gold_avg",
    "ff_elite_avg",
    "is_business",
    "is_weekday",
    "is_morning_flight",
    "distance_km",
]

# Pull data
df = con.execute(f"""
    SELECT flight_id, cabin_class,
           {', '.join(CLUSTER_FEATURES)}
    FROM profiles
""").fetchdf()

# Fill NaN with 0 for clustering
X = df[CLUSTER_FEATURES].fillna(0).values.astype(np.float64)

print(f"   → Feature matrix: {X.shape[0]} samples × {X.shape[1]} features", flush=True)

# ═══════════════════════════════════════════════════════════
# STEP 3: StandardScaler + K-Means
# ═══════════════════════════════════════════════════════════
print("[3/5] Running K-Means clustering...", flush=True)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try k = 3..7 and pick best silhouette
results = {}
for k in range(3, 8):
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=10000, n_init=5)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels, sample_size=min(50000, len(X_scaled)), random_state=42)
    results[k] = {"model": km, "labels": labels, "silhouette": sil}
    print(f"   k={k}  silhouette={sil:.4f}", flush=True)

# Pick best k
best_k = max(results, key=lambda k: results[k]["silhouette"])
best = results[best_k]
print(f"   → Best k = {best_k} (silhouette = {best['silhouette']:.4f})", flush=True)

df["cluster"] = best["labels"]

# ═══════════════════════════════════════════════════════════
# STEP 4: Analyze clusters and assign labels
# ═══════════════════════════════════════════════════════════
print("[4/5] Analyzing cluster profiles...", flush=True)

cluster_profiles = {}
label_map = {}

for c in range(best_k):
    mask = df["cluster"] == c
    subset = df[mask]
    n = int(mask.sum())
    
    profile = {
        "size": n,
        "pct": round(n / len(df) * 100, 1),
        "avg_dtd_at_purchase": round(float(subset["avg_dtd_at_purchase"].mean()), 1),
        "pct_last_minute": round(float(subset["pct_last_minute"].mean()) * 100, 1),
        "pct_early_bird": round(float(subset["pct_early_bird"].mean()) * 100, 1),
        "avg_daily_pax": round(float(subset["avg_daily_pax"].mean()), 3),
        "max_load_factor": round(float(subset["max_load_factor"].mean()), 3),
        "ff_gold_avg": round(float(subset["ff_gold_avg"].mean()), 3),
        "ff_elite_avg": round(float(subset["ff_elite_avg"].mean()), 3),
        "is_business_pct": round(float(subset["is_business"].mean()) * 100, 1),
        "is_weekday_pct": round(float(subset["is_weekday"].mean()) * 100, 1),
        "is_morning_pct": round(float(subset["is_morning_flight"].mean()) * 100, 1),
        "distance_km": round(float(subset["distance_km"].mean()), 0),
    }
    cluster_profiles[c] = profile

# Auto-label clusters based on dominant characteristics
for c, p in cluster_profiles.items():
    scores = {}
    
    # Last-minute / urgent travelers
    scores["Son Dakikacılar (Acil Seyahat)"] = (
        p["pct_last_minute"] * 2 +
        (100 - p["avg_dtd_at_purchase"]) * 0.5 +
        p["max_load_factor"] * 50
    )
    
    # Early planners / price-sensitive
    scores["Erken Planlayıcılar (Fiyata Duyarlı)"] = (
        p["pct_early_bird"] * 2 +
        p["avg_dtd_at_purchase"] * 0.5 +
        (100 - p["is_business_pct"]) * 0.3
    )
    
    # Business travelers
    scores["İş Yolcuları"] = (
        p["is_business_pct"] * 2 +
        p["is_weekday_pct"] * 0.5 +
        p["is_morning_pct"] * 0.5 +
        p["ff_gold_avg"] * 200 +
        p["ff_elite_avg"] * 200
    )
    
    # Leisure / holiday travelers
    scores["Tatilciler / Leisure"] = (
        (100 - p["is_weekday_pct"]) * 1.5 +
        (100 - p["is_business_pct"]) * 0.5 +
        p["distance_km"] * 0.01 +
        (100 - p["ff_gold_avg"] * 1000) * 0.1
    )
    
    # High-demand routes
    scores["Yüksek Talep Rotaları"] = (
        p["max_load_factor"] * 100 +
        p["avg_daily_pax"] * 50 +
        p["pct_last_minute"] * 0.5
    )
    
    # Pick the top label; avoid duplicates
    sorted_labels = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for label, score in sorted_labels:
        if label not in label_map.values():
            label_map[c] = label
            break
    else:
        label_map[c] = sorted_labels[0][0] + f" (Küme {c})"
    
    cluster_profiles[c]["label"] = label_map[c]
    print(f"   Küme {c}: {label_map[c]} ({p['size']:,} uçuş, {p['pct']}%)", flush=True)

# Map labels to dataframe
df["cluster_label"] = df["cluster"].map(label_map)

# ═══════════════════════════════════════════════════════════
# STEP 5: Save results
# ═══════════════════════════════════════════════════════════
print("[5/5] Saving results...", flush=True)

# Register df in DuckDB and export
con.register("cluster_df", df)
con.execute(f"""
    COPY (
        SELECT p.*, c.cluster, c.cluster_label
        FROM profiles p
        JOIN cluster_df c
          ON p.flight_id = c.flight_id AND p.cabin_class = c.cabin_class
    ) TO '{OUT_PATH}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

# Build report
report = {
    "total_profiles": int(profile_count),
    "best_k": best_k,
    "silhouette_score": round(best['silhouette'], 4),
    "silhouette_all": {str(k): round(v["silhouette"], 4) for k, v in results.items()},
    "features_used": CLUSTER_FEATURES,
    "clusters": {}
}
for c in range(best_k):
    p = cluster_profiles[c]
    report["clusters"][str(c)] = p

with open(OUT_RPT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

con.close()

# ── Print summary ──
print("\n" + "=" * 60)
print("  YOLCU KÜMELEME SONUÇLARI")
print("=" * 60)
print(f"  Toplam profil:     {profile_count:,}")
print(f"  Küme sayısı (k):  {best_k}")
print(f"  Silhouette score:  {best['silhouette']:.4f}")
print()
for c in range(best_k):
    p = cluster_profiles[c]
    emoji = ["🔴", "🟢", "🔵", "🟡", "🟣", "🟠", "⚪"][c % 7]
    print(f"  {emoji} Küme {c}: {p['label']}")
    print(f"     Boyut: {p['size']:,} ({p['pct']}%)")
    print(f"     Ort. DTD: {p['avg_dtd_at_purchase']}  |  Son dakika: {p['pct_last_minute']}%  |  Erken: {p['pct_early_bird']}%")
    print(f"     Doluluk: {p['max_load_factor']:.1%}  |  Business: {p['is_business_pct']}%  |  Hafta içi: {p['is_weekday_pct']}%")
    print(f"     FF Gold: {p['ff_gold_avg']:.3f}  |  FF Elite: {p['ff_elite_avg']:.3f}  |  Mesafe: {p['distance_km']:.0f} km")
    print()

print(f"  📦 Parquet: {OUT_PATH}")
print(f"  📋 Rapor:   {OUT_RPT}")
print("\n[DONE]")
