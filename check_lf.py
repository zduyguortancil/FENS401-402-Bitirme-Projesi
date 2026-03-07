#!/usr/bin/env python3
"""Quick check of load_factor distribution and available columns."""
import duckdb, json
from pathlib import Path

BASE = Path(__file__).parent
con = duckdb.connect()
p = str(BASE / "demand_training.parquet")

# 1) Load factor stats
r = con.execute(f"""
    SELECT 
        MIN(load_factor), AVG(load_factor),
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY load_factor),
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY load_factor),
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY load_factor),
        MAX(load_factor), COUNT(*),
        AVG(CASE WHEN dtd = 0 THEN load_factor END),
        MAX(CASE WHEN dtd = 0 THEN load_factor END)
    FROM read_parquet('{p}')
""").fetchone()

print("=== Load Factor Distribution (all DTDs) ===")
print(f"Min:    {r[0]:.4f}")
print(f"Avg:    {r[1]:.4f}")
print(f"Median: {r[2]:.4f}")
print(f"P75:    {r[3]:.4f}")
print(f"P90:    {r[4]:.4f}")
print(f"Max:    {r[5]:.4f}")
print(f"Rows:   {r[6]:,}")
if r[7]: print(f"Final LF avg (DTD=0): {r[7]:.4f}")
if r[8]: print(f"Final LF max (DTD=0): {r[8]:.4f}")

# 2) Columns
cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{p}')").fetchall()
print("\n=== Columns ===")
for c in cols:
    print(f"  {c[0]} ({c[1]})")

# 3) Top 10 routes by total pax
print("\n=== Top 10 Routes by Total Pax ===")
routes = con.execute(f"""
    SELECT flight_id,
           SUM(y_pax_sold_today) as total_pax,
           MAX(load_factor) as max_lf,
           AVG(load_factor) as avg_lf,
           MIN(dtd) as min_dtd,
           MAX(dtd) as max_dtd
    FROM read_parquet('{p}')
    GROUP BY flight_id
    ORDER BY total_pax DESC
    LIMIT 10
""").fetchall()
for rt in routes:
    print(f"  {rt[0]}: pax={rt[1]}, max_lf={rt[2]:.3f}, avg_lf={rt[3]:.3f}")

# 4) Check snapshot v2 columns for pricing
sp = str(BASE / "flight_snapshot_v2.parquet")
try:
    scols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{sp}')").fetchall()
    print("\n=== Snapshot V2 Columns ===")
    for c in scols:
        print(f"  {c[0]} ({c[1]})")
except:
    print("\nSnapshot V2 not found")

con.close()
