# -*- coding: utf-8 -*-
"""
Pickup Model Master Tablo
===========================
Target: remaining_pax = final_pax - pax_sold_cum
Her ucusun DTD=0 satirindan final_pax alinir.

Girdi:
  data/processed/demand_training.parquet  (37M satir)
  data/processed/flight_metadata.parquet  (204K satir)
  data/processed/tft_route_daily.parquet  (138K satir)

Cikti:
  data/processed/pickup_master.parquet
"""

import duckdb
import time
from pathlib import Path
import gc

BASE = Path(r"C:\Users\ahmet\OneDrive\Desktop\ptir")
DATA = BASE / "data" / "processed"

DEMAND   = str(DATA / "demand_training.parquet").replace("\\", "/")
METADATA = str(DATA / "flight_metadata.parquet").replace("\\", "/")
ROUTE    = str(DATA / "tft_route_daily.parquet").replace("\\", "/")
OUTPUT   = str(DATA / "pickup_master.parquet").replace("\\", "/")

start = time.time()
con = duckdb.connect()
con.execute("SET memory_limit='6GB'")
con.execute("SET temp_directory='.'")

# ── 1. Final pax: her ucusun DTD=0 satirindaki pax_sold_cum ──
print("[1/4] Final pax hesaplaniyor...", flush=True)
con.execute(f"""
    CREATE TABLE final_pax AS
    SELECT flight_id, cabin_class, pax_sold_cum AS final_pax
    FROM read_parquet('{DEMAND}')
    WHERE dtd = 0
""")
cnt = con.execute("SELECT COUNT(*) FROM final_pax").fetchone()[0]
print(f"  {cnt:,} ucus (flight x cabin)", flush=True)

# ── 2. Route mapping ──
print("[2/4] Route mapping...", flush=True)
con.execute(f"""
    CREATE TABLE meta AS
    SELECT DISTINCT
        flight_id,
        departure_airport || '_' || arrival_airport AS route
    FROM read_parquet('{METADATA}')
""")

# ── 3. Route-daily features ──
print("[3/4] Route-daily features...", flush=True)
con.execute(f"""
    CREATE TABLE rdaily AS
    SELECT
        route, cabin_class, dep_date,
        total_pax AS route_total_pax,
        n_flights AS route_n_flights,
        avg_fare, std_fare,
        n_bookings AS route_n_bookings,
        avg_group_size,
        corporate_pct, agency_pct, connecting_pct,
        early_booking_pct, late_booking_pct,
        child_pct, halal_pct,
        gold_elite_pct AS route_gold_pct,
        elite_pct AS route_elite_pct
    FROM read_parquet('{ROUTE}')
""")

# ── 4. Master tablo: join + remaining_pax ──
print("[4/4] Master tablo olusturuluyor...", flush=True)
con.execute(f"""
    COPY (
        SELECT
            -- Identifiers (model icin kullanilmaz, analiz icin)
            d.flight_id,
            d.cabin_class,
            d.dep_year,

            -- TARGET
            (f.final_pax - d.pax_sold_cum)::FLOAT AS remaining_pax,
            f.final_pax::FLOAT AS final_pax,

            -- Booking curve features
            d.dtd::FLOAT AS dtd,
            d.pax_sold_cum::FLOAT AS pax_sold_cum,
            d.pax_last_7d::FLOAT AS pax_last_7d,
            d.capacity::FLOAT AS capacity,
            d.remaining_seats::FLOAT AS remaining_seats,
            d.load_factor::FLOAT AS load_factor,

            -- Flight features
            d.distance_km::FLOAT AS distance_km,
            d.flight_time_min::FLOAT AS flight_time_min,
            d.dep_month::FLOAT AS dep_month,
            d.dep_dow::FLOAT AS dep_dow,
            d.dep_hour::FLOAT AS dep_hour,

            -- Passenger profile
            d.ff_gold_pct::FLOAT AS ff_gold_pct,
            d.ff_elite_pct::FLOAT AS ff_elite_pct,

            -- Event tags (demand_training'den - 15 adet)
            CASE WHEN d.tag_yaz_tatili THEN 1.0 ELSE 0.0 END AS tag_yaz_tatili,
            CASE WHEN d.tag_kis_tatili THEN 1.0 ELSE 0.0 END AS tag_kis_tatili,
            CASE WHEN d.tag_yariyil_tatili THEN 1.0 ELSE 0.0 END AS tag_yariyil_tatili,
            CASE WHEN d.tag_bahar_tatili THEN 1.0 ELSE 0.0 END AS tag_bahar_tatili,
            CASE WHEN d.tag_ramazan_donemi THEN 1.0 ELSE 0.0 END AS tag_ramazan,
            CASE WHEN d.tag_bayram_donemi THEN 1.0 ELSE 0.0 END AS tag_bayram,
            CASE WHEN d.tag_yilbasi THEN 1.0 ELSE 0.0 END AS tag_yilbasi,
            CASE WHEN d.tag_kongre_fuar THEN 1.0 ELSE 0.0 END AS tag_kongre_fuar,
            CASE WHEN d.tag_ski_sezonu THEN 1.0 ELSE 0.0 END AS tag_ski_sezonu,
            CASE WHEN d.tag_festival_sezonu THEN 1.0 ELSE 0.0 END AS tag_festival,
            CASE WHEN d.tag_is_seyahati_yogun THEN 1.0 ELSE 0.0 END AS tag_is_seyahati,
            CASE WHEN d.tag_hac_umre THEN 1.0 ELSE 0.0 END AS tag_hac_umre,
            CASE WHEN d.tag_gece_ucusu THEN 1.0 ELSE 0.0 END AS tag_gece_ucusu,

            -- Route-level features (tft_route_daily'den)
            COALESCE(r.route_total_pax, 0)::FLOAT AS route_total_pax,
            COALESCE(r.route_n_flights, 1)::FLOAT AS route_n_flights,
            COALESCE(r.avg_fare, 0)::FLOAT AS avg_fare,
            COALESCE(r.std_fare, 0)::FLOAT AS std_fare,
            COALESCE(r.route_n_bookings, 0)::FLOAT AS route_n_bookings,
            COALESCE(r.avg_group_size, 0)::FLOAT AS avg_group_size,
            COALESCE(r.corporate_pct, 0)::FLOAT AS corporate_pct,
            COALESCE(r.agency_pct, 0)::FLOAT AS agency_pct,
            COALESCE(r.connecting_pct, 0)::FLOAT AS connecting_pct,
            COALESCE(r.early_booking_pct, 0)::FLOAT AS early_booking_pct,
            COALESCE(r.late_booking_pct, 0)::FLOAT AS late_booking_pct,
            COALESCE(r.child_pct, 0)::FLOAT AS child_pct,
            COALESCE(r.halal_pct, 0)::FLOAT AS halal_pct,
            COALESCE(r.route_gold_pct, 0)::FLOAT AS route_gold_pct,
            COALESCE(r.route_elite_pct, 0)::FLOAT AS route_elite_pct,

            -- One-hot: cabin
            (CASE WHEN LOWER(d.cabin_class) = 'economy' THEN 1.0 ELSE 0.0 END)::FLOAT AS cabin_economy,
            (CASE WHEN LOWER(d.cabin_class) = 'business' THEN 1.0 ELSE 0.0 END)::FLOAT AS cabin_business,

            -- One-hot: region
            (CASE WHEN d.region = 'Domestic' THEN 1.0 ELSE 0.0 END)::FLOAT AS reg_domestic,
            (CASE WHEN d.region = 'Europe' THEN 1.0 ELSE 0.0 END)::FLOAT AS reg_europe,
            (CASE WHEN d.region = 'Middle East' THEN 1.0 ELSE 0.0 END)::FLOAT AS reg_mideast,
            (CASE WHEN d.region = 'Asia' THEN 1.0 ELSE 0.0 END)::FLOAT AS reg_asia,
            (CASE WHEN d.region = 'Americas' THEN 1.0 ELSE 0.0 END)::FLOAT AS reg_americas,
            (CASE WHEN d.region NOT IN ('Domestic','Europe','Middle East','Asia','Americas') THEN 1.0 ELSE 0.0 END)::FLOAT AS reg_other

        FROM read_parquet('{DEMAND}') d
        INNER JOIN final_pax f ON d.flight_id = f.flight_id AND d.cabin_class = f.cabin_class
        LEFT JOIN meta m ON d.flight_id = m.flight_id
        LEFT JOIN rdaily r ON m.route = r.route
            AND LOWER(d.cabin_class) = LOWER(r.cabin_class)
            AND CAST(SPLIT_PART(SPLIT_PART(d.flight_id, '_', 2), ' ', 1) AS DATE) = r.dep_date
        WHERE d.dtd > 0
    ) TO '{OUTPUT}' (FORMAT PARQUET)
""")

# Kontrol
info = con.execute(f"""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT flight_id || cabin_class) AS n_flights,
        AVG(remaining_pax) AS avg_remaining,
        MIN(remaining_pax) AS min_remaining,
        MAX(remaining_pax) AS max_remaining,
        AVG(CASE WHEN remaining_pax = 0 THEN 1.0 ELSE 0.0 END) * 100 AS zero_pct,
        SUM(CASE WHEN dep_year = 2025 THEN 1 ELSE 0 END) AS train_rows,
        SUM(CASE WHEN dep_year = 2026 THEN 1 ELSE 0 END) AS test_rows
    FROM read_parquet('{OUTPUT}')
""").fetchone()

cols = con.execute(f"SELECT COUNT(*) FROM (SELECT * FROM read_parquet('{OUTPUT}') LIMIT 1) t").fetchone()[0]
col_names = con.execute(f"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{OUTPUT}'))").fetchall()
n_cols = len(col_names)

con.close()
del con
gc.collect()

elapsed = time.time() - start

print(f"\n{'='*50}")
print(f"  PICKUP MASTER TABLO")
print(f"{'='*50}")
print(f"  Toplam satir:    {info[0]:,}")
print(f"  Ucus sayisi:     {info[1]:,}")
print(f"  Kolon sayisi:    {n_cols}")
print(f"  Train (2025):    {info[6]:,}")
print(f"  Test (2026):     {info[7]:,}")
print(f"")
print(f"  TARGET: remaining_pax")
print(f"    Ortalama:      {info[2]:.1f}")
print(f"    Min/Max:       {info[3]:.0f} / {info[4]:.0f}")
print(f"    Sifir orani:   %{info[5]:.1f}")
print(f"")
print(f"  Cikti: {OUTPUT}")
print(f"  Sure:  {elapsed:.0f} saniye")
print(f"\n[TAMAMLANDI]")
