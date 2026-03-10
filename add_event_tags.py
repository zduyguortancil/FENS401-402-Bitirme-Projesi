"""
Event / Sentiment Tags Builder
Adds 15 event categories to demand_training.parquet based on departure timing.
Each flight can have multiple tags (multi-label), plus a primary_event (single).
"""
import json
import duckdb
from pathlib import Path

BASE_DIR = Path(__file__).parent
PARQUET   = BASE_DIR / "demand_training.parquet"
OUT_PATH  = BASE_DIR / "demand_training.parquet"   # overwrite
REPORT    = BASE_DIR / "event_tags_report.json"

con = duckdb.connect()

print("[1/4] Reading training data...", flush=True)
row_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{PARQUET}')").fetchone()[0]
print(f"       {row_count:,} rows", flush=True)

print("[2/4] Adding event tags...", flush=True)

# We need the actual departure date from flight_id (format: TKxxxxx_YYYY-MM-DD HH:MM:SS)
# and the existing dep_month, dep_dow, dep_hour columns

con.execute(f"""
    CREATE TABLE tagged AS
    SELECT
        t.*,

        -- Extract day-of-month from flight_id for precise date matching
        CAST(SPLIT_PART(SPLIT_PART(SPLIT_PART(t.flight_id, '_', 2), ' ', 1), '-', 3) AS INT) AS _dep_day,

        -- ======= 15 EVENT TAGS (each is boolean) =======

        -- 1. yaz_tatili: Haziran-Ağustos
        CASE WHEN t.dep_month IN (6, 7, 8) THEN TRUE ELSE FALSE END AS tag_yaz_tatili,

        -- 2. kis_tatili: Aralık-Ocak
        CASE WHEN t.dep_month IN (12, 1) THEN TRUE ELSE FALSE END AS tag_kis_tatili,

        -- 3. yariyil_tatili: Ocak sonu - Şubat başı (15 Ocak - 10 Şubat)
        CASE WHEN (t.dep_month = 1 AND CAST(SPLIT_PART(SPLIT_PART(SPLIT_PART(t.flight_id, '_', 2), ' ', 1), '-', 3) AS INT) >= 15)
              OR  (t.dep_month = 2 AND CAST(SPLIT_PART(SPLIT_PART(SPLIT_PART(t.flight_id, '_', 2), ' ', 1), '-', 3) AS INT) <= 10)
             THEN TRUE ELSE FALSE END AS tag_yariyil_tatili,

        -- 4. bahar_tatili: Mart-Nisan
        CASE WHEN t.dep_month IN (3, 4) THEN TRUE ELSE FALSE END AS tag_bahar_tatili,

        -- 5. ramazan_donemi: ~Mart 2025, ~Şubat-Mart 2026
        CASE WHEN (t.dep_year = 2025 AND t.dep_month = 3)
              OR  (t.dep_year = 2026 AND t.dep_month IN (2, 3))
             THEN TRUE ELSE FALSE END AS tag_ramazan_donemi,

        -- 6. bayram_donemi: Ramazan Bayramı (~30 Mart 2025, ~20 Mart 2026) + Kurban (~7 Haziran 2025, ~27 Mayıs 2026)
        CASE WHEN (t.dep_year = 2025 AND t.dep_month = 3 AND CAST(SPLIT_PART(SPLIT_PART(SPLIT_PART(t.flight_id, '_', 2), ' ', 1), '-', 3) AS INT) BETWEEN 28 AND 31)
              OR  (t.dep_year = 2025 AND t.dep_month = 4 AND CAST(SPLIT_PART(SPLIT_PART(SPLIT_PART(t.flight_id, '_', 2), ' ', 1), '-', 3) AS INT) BETWEEN 1 AND 3)
              OR  (t.dep_year = 2025 AND t.dep_month = 6 AND CAST(SPLIT_PART(SPLIT_PART(SPLIT_PART(t.flight_id, '_', 2), ' ', 1), '-', 3) AS INT) BETWEEN 5 AND 10)
              OR  (t.dep_year = 2026 AND t.dep_month = 3 AND CAST(SPLIT_PART(SPLIT_PART(SPLIT_PART(t.flight_id, '_', 2), ' ', 1), '-', 3) AS INT) BETWEEN 18 AND 23)
              OR  (t.dep_year = 2026 AND t.dep_month = 5 AND CAST(SPLIT_PART(SPLIT_PART(SPLIT_PART(t.flight_id, '_', 2), ' ', 1), '-', 3) AS INT) BETWEEN 25 AND 30)
             THEN TRUE ELSE FALSE END AS tag_bayram_donemi,

        -- 7. yilbasi: 20 Aralık - 5 Ocak
        CASE WHEN (t.dep_month = 12 AND CAST(SPLIT_PART(SPLIT_PART(SPLIT_PART(t.flight_id, '_', 2), ' ', 1), '-', 3) AS INT) >= 20)
              OR  (t.dep_month = 1  AND CAST(SPLIT_PART(SPLIT_PART(SPLIT_PART(t.flight_id, '_', 2), ' ', 1), '-', 3) AS INT) <= 5)
             THEN TRUE ELSE FALSE END AS tag_yilbasi,

        -- 8. sevgililer_gunu: 10-16 Şubat
        CASE WHEN t.dep_month = 2 AND CAST(SPLIT_PART(SPLIT_PART(SPLIT_PART(t.flight_id, '_', 2), ' ', 1), '-', 3) AS INT) BETWEEN 10 AND 16
             THEN TRUE ELSE FALSE END AS tag_sevgililer_gunu,

        -- 9. futbol_sezonu: Ağustos-Mayıs (ligler aktif)
        CASE WHEN t.dep_month NOT IN (6, 7) THEN TRUE ELSE FALSE END AS tag_futbol_sezonu,

        -- 10. kongre_fuar: Eylül-Kasım (iş sezonu)
        CASE WHEN t.dep_month IN (9, 10, 11) THEN TRUE ELSE FALSE END AS tag_kongre_fuar,

        -- 11. ski_sezonu: Aralık-Mart
        CASE WHEN t.dep_month IN (12, 1, 2, 3) THEN TRUE ELSE FALSE END AS tag_ski_sezonu,

        -- 12. festival_sezonu: Haziran-Eylül
        CASE WHEN t.dep_month IN (6, 7, 8, 9) THEN TRUE ELSE FALSE END AS tag_festival_sezonu,

        -- 13. is_seyahati_yogun: Hafta içi + iş ayları (Eyl-Kas, Oca-Mar)
        CASE WHEN t.dep_dow BETWEEN 1 AND 5
              AND t.dep_month IN (1, 2, 3, 9, 10, 11)
             THEN TRUE ELSE FALSE END AS tag_is_seyahati_yogun,

        -- 14. hac_umre: Haziran 2025 (Hac) veya bölge Middle East + belirli aylar
        CASE WHEN (t.dep_year = 2025 AND t.dep_month = 6 AND t.region = 'Middle East')
              OR  (t.dep_year = 2026 AND t.dep_month = 5 AND t.region = 'Middle East')
              OR  (t.region = 'Middle East' AND t.dep_month IN (1, 2, 3, 11, 12))
             THEN TRUE ELSE FALSE END AS tag_hac_umre,

        -- 15. gece_ucusu: 22:00 - 06:00
        CASE WHEN t.dep_hour >= 22 OR t.dep_hour <= 6 THEN TRUE ELSE FALSE END AS tag_gece_ucusu,

        -- ======= PRIMARY EVENT (single best-match) =======
        CASE
            WHEN (t.dep_year = 2025 AND t.dep_month = 3 AND CAST(SPLIT_PART(SPLIT_PART(SPLIT_PART(t.flight_id, '_', 2), ' ', 1), '-', 3) AS INT) BETWEEN 28 AND 31)
              OR (t.dep_year = 2025 AND t.dep_month = 4 AND CAST(SPLIT_PART(SPLIT_PART(SPLIT_PART(t.flight_id, '_', 2), ' ', 1), '-', 3) AS INT) BETWEEN 1 AND 3)
              OR (t.dep_year = 2025 AND t.dep_month = 6 AND CAST(SPLIT_PART(SPLIT_PART(SPLIT_PART(t.flight_id, '_', 2), ' ', 1), '-', 3) AS INT) BETWEEN 5 AND 10)
              OR (t.dep_year = 2026 AND t.dep_month = 3 AND CAST(SPLIT_PART(SPLIT_PART(SPLIT_PART(t.flight_id, '_', 2), ' ', 1), '-', 3) AS INT) BETWEEN 18 AND 23)
              OR (t.dep_year = 2026 AND t.dep_month = 5 AND CAST(SPLIT_PART(SPLIT_PART(SPLIT_PART(t.flight_id, '_', 2), ' ', 1), '-', 3) AS INT) BETWEEN 25 AND 30)
                THEN 'bayram_donemi'
            WHEN (t.dep_month = 12 AND CAST(SPLIT_PART(SPLIT_PART(SPLIT_PART(t.flight_id, '_', 2), ' ', 1), '-', 3) AS INT) >= 20)
              OR (t.dep_month = 1  AND CAST(SPLIT_PART(SPLIT_PART(SPLIT_PART(t.flight_id, '_', 2), ' ', 1), '-', 3) AS INT) <= 5)
                THEN 'yilbasi'
            WHEN t.dep_month = 2 AND CAST(SPLIT_PART(SPLIT_PART(SPLIT_PART(t.flight_id, '_', 2), ' ', 1), '-', 3) AS INT) BETWEEN 10 AND 16
                THEN 'sevgililer_gunu'
            WHEN (t.dep_year = 2025 AND t.dep_month = 3)
              OR (t.dep_year = 2026 AND t.dep_month IN (2, 3))
                THEN 'ramazan_donemi'
            WHEN t.dep_month IN (6, 7, 8)   THEN 'yaz_tatili'
            WHEN t.dep_month IN (12, 1)      THEN 'kis_tatili'
            WHEN t.dep_month IN (3, 4)       THEN 'bahar_tatili'
            WHEN t.dep_month IN (9, 10, 11)  THEN 'kongre_fuar'
            WHEN t.dep_month = 5             THEN 'festival_sezonu'
            ELSE 'normal'
        END AS primary_event

    FROM read_parquet('{PARQUET}') t
""")

# Drop the helper _dep_day column
con.execute("ALTER TABLE tagged DROP COLUMN _dep_day")

new_count = con.execute("SELECT COUNT(*) FROM tagged").fetchone()[0]
print(f"       Tagged {new_count:,} rows", flush=True)
assert new_count == row_count, f"Row mismatch! {new_count} vs {row_count}"

print("[3/4] Writing updated parquet...", flush=True)
con.execute(f"COPY tagged TO '{OUT_PATH}' (FORMAT PARQUET, COMPRESSION ZSTD)")

# Build report
print("[4/4] Building report...", flush=True)

tag_names = [
    "tag_yaz_tatili", "tag_kis_tatili", "tag_yariyil_tatili", "tag_bahar_tatili",
    "tag_ramazan_donemi", "tag_bayram_donemi", "tag_yilbasi", "tag_sevgililer_gunu",
    "tag_futbol_sezonu", "tag_kongre_fuar", "tag_ski_sezonu", "tag_festival_sezonu",
    "tag_is_seyahati_yogun", "tag_hac_umre", "tag_gece_ucusu"
]

tag_stats = {}
for tag in tag_names:
    r = con.execute(f"""
        SELECT
            SUM(CASE WHEN {tag} THEN 1 ELSE 0 END) AS cnt,
            AVG(CASE WHEN {tag} THEN y_pax_sold_today END) AS avg_pax,
            AVG(CASE WHEN {tag} THEN load_factor END) AS avg_lf,
            AVG(CASE WHEN NOT {tag} THEN y_pax_sold_today END) AS avg_pax_no,
            AVG(CASE WHEN NOT {tag} THEN load_factor END) AS avg_lf_no
        FROM tagged
    """).fetchone()
    name = tag.replace("tag_", "")
    tag_stats[name] = {
        "count": int(r[0]),
        "pct": round(r[0] / new_count * 100, 1),
        "avg_pax_tagged": round(float(r[1]), 4) if r[1] else 0,
        "avg_lf_tagged": round(float(r[2]) * 100, 2) if r[2] else 0,
        "avg_pax_normal": round(float(r[3]), 4) if r[3] else 0,
        "avg_lf_normal": round(float(r[4]) * 100, 2) if r[4] else 0,
    }

primary_dist = {r[0]: int(r[1]) for r in
    con.execute("SELECT primary_event, COUNT(*) FROM tagged GROUP BY primary_event ORDER BY COUNT(*) DESC").fetchall()}

report = {
    "total_rows": new_count,
    "tag_columns": tag_names,
    "primary_events": list(primary_dist.keys()),
    "tag_stats": tag_stats,
    "primary_distribution": primary_dist,
}

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

con.close()

print("\n--- Event Tags Report ---")
for name, s in tag_stats.items():
    diff = s["avg_pax_tagged"] - s["avg_pax_normal"]
    arrow = "↑" if diff > 0 else "↓"
    print(f"  {name:25s}  {s['pct']:5.1f}% | pax {s['avg_pax_tagged']:.2f} ({arrow}{abs(diff):.2f}) | LF {s['avg_lf_tagged']:.1f}%")

print(f"\n  Primary distribution:")
for ev, cnt in primary_dist.items():
    print(f"    {ev:25s}  {cnt:>10,} ({cnt/new_count*100:.1f}%)")

print(f"\n  Output: {OUT_PATH}")
print(f"  Report: {REPORT}")
print("\n[DONE]")
