"""
Sprint 4 - Demand Training Dataset Builder (DuckDB edition)
Avoids pandas OOM on 37M rows by doing all transforms in DuckDB SQL.
"""
import json
import duckdb
from pathlib import Path

BASE_DIR  = Path(r"C:\Users\ahmet\OneDrive\Desktop\ptir")
SNAP_PATH = BASE_DIR / "flight_snapshot_v2.parquet"
META_PATH = BASE_DIR / "flight_metadata.parquet"
OUT_PATH  = BASE_DIR / "demand_training.parquet"
OUT_RPT   = BASE_DIR / "demand_training_report.json"

con = duckdb.connect()

print("[1/4] Building training table via DuckDB...", flush=True)

con.execute(f"""
    CREATE TABLE training AS
    SELECT
        s.flight_id,
        s.cabin_class,
        s.dtd,
        -- DTD bucket (int8 code: 0=0-3, 1=4-7, 2=8-14, 3=15-30, 4=31-60, 5=61-90, 6=90+)
        CASE
            WHEN s.dtd <= 3  THEN 0
            WHEN s.dtd <= 7  THEN 1
            WHEN s.dtd <= 14 THEN 2
            WHEN s.dtd <= 30 THEN 3
            WHEN s.dtd <= 60 THEN 4
            WHEN s.dtd <= 90 THEN 5
            ELSE 6
        END::TINYINT AS dtd_bucket,

        -- Target
        s.pax_sold_today AS y_pax_sold_today,

        -- Pax features
        s.pax_sold_cum,
        s.pax_last_7d,

        -- Capacity-derived features (from metadata)
        m.capacity,
        GREATEST(m.capacity - s.pax_sold_cum, 0) AS remaining_seats,
        CASE WHEN m.capacity > 0
             THEN s.pax_sold_cum * 1.0 / m.capacity
             ELSE NULL
        END AS load_factor,

        -- Route / flight features
        m.region,
        m.distance_km,
        m.flight_time_min,

        -- Calendar features
        EXTRACT(YEAR  FROM m.departure_datetime)::INT AS dep_year,
        EXTRACT(MONTH FROM m.departure_datetime)::INT AS dep_month,
        EXTRACT(DOW   FROM m.departure_datetime)::INT AS dep_dow,
        EXTRACT(HOUR  FROM m.departure_datetime)::INT AS dep_hour,

        -- Optional passenger profile
        s.ff_gold_pct,
        s.ff_elite_pct

    FROM read_parquet('{SNAP_PATH}') s
    LEFT JOIN read_parquet('{META_PATH}') m
        ON  s.flight_id   = m.flight_id
        AND LOWER(s.cabin_class) = LOWER(m.cabin_class)

    WHERE s.flight_id IS NOT NULL
      AND s.cabin_class IS NOT NULL
      AND s.dtd IS NOT NULL
      AND s.pax_sold_today IS NOT NULL
      AND s.dtd BETWEEN 0 AND 365
""")

row_count = con.execute("SELECT COUNT(*) FROM training").fetchone()[0]
print(f"[2/4] Training table: {row_count:,} rows", flush=True)

# Export to parquet
print("[3/4] Writing parquet...", flush=True)
con.execute(f"COPY training TO '{OUT_PATH}' (FORMAT PARQUET, COMPRESSION ZSTD)")

# Build report
print("[4/4] Building report...", flush=True)

stats = con.execute("""
    SELECT
        COUNT(*)                                      AS total_rows,
        AVG(y_pax_sold_today)                         AS y_mean,
        MEDIAN(y_pax_sold_today)                      AS y_median,
        STDDEV(y_pax_sold_today)                      AS y_std,
        MIN(y_pax_sold_today)                         AS y_min,
        MAX(y_pax_sold_today)                         AS y_max,
        AVG(CASE WHEN y_pax_sold_today = 0 THEN 1.0 ELSE 0.0 END) AS y_zero_rate,
        MIN(dtd)                                      AS dtd_min,
        MAX(dtd)                                      AS dtd_max
    FROM training
""").fetchone()

year_counts = {str(r[0]): int(r[1]) for r in
    con.execute("SELECT dep_year, COUNT(*) FROM training GROUP BY dep_year ORDER BY dep_year").fetchall()}

cabin_counts = {str(r[0]): int(r[1]) for r in
    con.execute("SELECT cabin_class, COUNT(*) FROM training GROUP BY cabin_class ORDER BY cabin_class").fetchall()}

null_rates = {}
cols = con.execute("SELECT column_name FROM information_schema.columns WHERE table_name='training'").fetchall()
for (col_name,) in cols:
    rate = con.execute(f"SELECT AVG(CASE WHEN \"{col_name}\" IS NULL THEN 1.0 ELSE 0.0 END) FROM training").fetchone()[0]
    null_rates[col_name] = float(rate)

report = {
    "rows": int(stats[0]),
    "cols": len(cols),
    "columns": [c[0] for c in cols],
    "y_stats": {
        "mean":      round(float(stats[1]), 4),
        "median":    round(float(stats[2]), 4),
        "std":       round(float(stats[3]), 4),
        "min":       float(stats[4]),
        "max":       float(stats[5]),
        "zero_rate": round(float(stats[6]), 4),
    },
    "dtd_min": int(stats[7]),
    "dtd_max": int(stats[8]),
    "year_counts": year_counts,
    "cabin_counts": cabin_counts,
    "null_rates": null_rates,
}

with open(OUT_RPT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

con.close()

print("\n--- Result ---")
print(f"  Rows:      {report['rows']:,}")
print(f"  Cols:      {report['cols']}")
print(f"  y mean:    {report['y_stats']['mean']}")
print(f"  y median:  {report['y_stats']['median']}")
print(f"  y zero%:   {report['y_stats']['zero_rate']:.1%}")
print(f"  DTD range: {report['dtd_min']} - {report['dtd_max']}")
print(f"  Years:     {report['year_counts']}")
print(f"  Cabins:    {report['cabin_counts']}")
print(f"\n  Parquet: {OUT_PATH}")
print(f"  Report:  {OUT_RPT}")
print("\n[DONE]")
