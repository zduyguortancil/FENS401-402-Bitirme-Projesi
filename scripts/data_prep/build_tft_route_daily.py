"""
TFT Route-Daily Dataset Builder
================================
3 ham parquet dosyasini (flights, bookings, passengers) birlestirip
rota+kabin+gun bazli aggregate eder.

Cikti: tft_route_daily.parquet (~146K satir, ~200 entity x 730 gun)

Her satir = bir rotanin bir kabininin bir gunluk toplam talep + feature ozeti.
Hicbir veri atilmiyor, sampling yok.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
DATA_DIR   = BASE_DIR / "data" / "raw"
OUT_PATH   = BASE_DIR / "tft_route_daily.parquet"

FLIGHTS_PATH    = DATA_DIR / "flights.parquet"
BOOKINGS_PATH   = DATA_DIR / "bookings.parquet"
PASSENGERS_PATH = DATA_DIR / "passengers.parquet"

for p in [FLIGHTS_PATH, BOOKINGS_PATH, PASSENGERS_PATH]:
    assert p.exists(), f"Dosya bulunamadi: {p}"

con = duckdb.connect()

# ── Step 1: Passenger-level aggregates per PNR ──
print("[1/5] Passenger profilleri hazirlaniyor...", flush=True)
con.execute(f"""
    CREATE TABLE pax_profile AS
    SELECT
        pnr,
        COUNT(*)                                                          AS pax_count,
        SUM(CASE WHEN passenger_type = 'child'  THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS child_pct,
        SUM(CASE WHEN passenger_type = 'infant' THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS infant_pct,
        SUM(CASE WHEN frequent_flyer_tier IN ('gold','elite') THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS gold_elite_pct,
        SUM(CASE WHEN frequent_flyer_tier = 'elite' THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS elite_pct,
        SUM(CASE WHEN meal_preference = 'halal' THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS halal_pct
    FROM read_parquet('{PASSENGERS_PATH}')
    GROUP BY pnr
""")
pax_count = con.execute("SELECT COUNT(*) FROM pax_profile").fetchone()[0]
print(f"       {pax_count:,} PNR profili", flush=True)

# ── Step 2: Enrich bookings with passenger profiles ──
print("[2/5] Booking + passenger join...", flush=True)
con.execute(f"""
    CREATE TABLE bookings_enriched AS
    SELECT
        b.flight_number,
        b.flight_date,
        b.cabin_class,
        b.passenger_count,
        b.fare_per_pax,
        b.booking_channel,
        b.days_to_departure,
        b.is_connecting,
        b.special_period,
        COALESCE(p.child_pct, 0)       AS child_pct,
        COALESCE(p.infant_pct, 0)      AS infant_pct,
        COALESCE(p.gold_elite_pct, 0)  AS gold_elite_pct,
        COALESCE(p.elite_pct, 0)       AS elite_pct,
        COALESCE(p.halal_pct, 0)       AS halal_pct
    FROM read_parquet('{BOOKINGS_PATH}') b
    LEFT JOIN pax_profile p ON b.pnr = p.pnr
""")
booking_count = con.execute("SELECT COUNT(*) FROM bookings_enriched").fetchone()[0]
print(f"       {booking_count:,} enriched bookings", flush=True)

# ── Step 3: Flight metadata (route, region, etc.) ──
print("[3/5] Flight metadata hazirlaniyor...", flush=True)
con.execute(f"""
    CREATE TABLE flight_meta AS
    SELECT
        flight_number,
        departure_airport || '_' || arrival_airport  AS route,
        departure_datetime::DATE                     AS dep_date,
        region,
        direction,
        distance_km,
        flight_time_min,
        EXTRACT(YEAR FROM departure_datetime)        AS dep_year,
        EXTRACT(MONTH FROM departure_datetime)       AS dep_month,
        EXTRACT(DOW FROM departure_datetime)         AS dep_dow,
        EXTRACT(HOUR FROM departure_datetime)        AS dep_hour
    FROM read_parquet('{FLIGHTS_PATH}')
""")
flight_count = con.execute("SELECT COUNT(*) FROM flight_meta").fetchone()[0]
print(f"       {flight_count:,} flights", flush=True)

# ── Step 4: Aggregate to route + cabin + date level ──
print("[4/5] Route-cabin-gun bazli aggregate...", flush=True)
con.execute("""
    CREATE TABLE route_daily AS
    WITH per_flight AS (
        SELECT
            fm.route,
            fm.dep_date,
            fm.region,
            fm.direction,
            fm.distance_km,
            fm.flight_time_min,
            fm.dep_year,
            fm.dep_month,
            fm.dep_dow,
            be.cabin_class,
            -- Target
            SUM(be.passenger_count)                     AS total_pax,
            COUNT(DISTINCT fm.flight_number)             AS n_flights,
            -- Fare features
            AVG(be.fare_per_pax)                        AS avg_fare,
            STDDEV(be.fare_per_pax)                     AS std_fare,
            MAX(be.fare_per_pax)                        AS max_fare,
            MIN(be.fare_per_pax)                        AS min_fare,
            -- Booking count
            COUNT(*)                                    AS n_bookings,
            -- Group size
            AVG(be.passenger_count)                     AS avg_group_size,
            -- Channel mix
            SUM(CASE WHEN be.booking_channel = 'corporate'    THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS corporate_pct,
            SUM(CASE WHEN be.booking_channel = 'travel_agency' THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS agency_pct,
            SUM(CASE WHEN be.booking_channel = 'website'      THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS website_pct,
            SUM(CASE WHEN be.booking_channel = 'mobile_app'   THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS mobile_pct,
            -- Connecting pct
            SUM(CASE WHEN be.is_connecting THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS connecting_pct,
            -- DTD distribution (early vs late bookers)
            SUM(CASE WHEN be.days_to_departure > 90 THEN be.passenger_count ELSE 0 END)::DOUBLE
                / NULLIF(SUM(be.passenger_count), 0)    AS early_booking_pct,
            SUM(CASE WHEN be.days_to_departure <= 14 THEN be.passenger_count ELSE 0 END)::DOUBLE
                / NULLIF(SUM(be.passenger_count), 0)    AS late_booking_pct,
            -- Passenger profiles (weighted avg)
            AVG(be.child_pct)                           AS child_pct,
            AVG(be.gold_elite_pct)                      AS gold_elite_pct,
            AVG(be.elite_pct)                           AS elite_pct,
            AVG(be.halal_pct)                           AS halal_pct,
            -- Special period
            MAX(be.special_period)                      AS special_period
        FROM bookings_enriched be
        JOIN flight_meta fm ON be.flight_number = fm.flight_number
        GROUP BY
            fm.route, fm.dep_date, fm.region, fm.direction,
            fm.distance_km, fm.flight_time_min,
            fm.dep_year, fm.dep_month, fm.dep_dow,
            be.cabin_class
    )
    SELECT
        *,
        -- Event tags (takvim bazli, add_event_tags.py mantigi)
        CASE WHEN dep_month IN (6,7,8) THEN TRUE ELSE FALSE END                AS tag_yaz_tatili,
        CASE WHEN dep_month IN (12,1) THEN TRUE ELSE FALSE END                 AS tag_kis_tatili,
        CASE WHEN dep_month IN (3,4) THEN TRUE ELSE FALSE END                  AS tag_bahar_tatili,
        CASE WHEN (dep_year=2025 AND dep_month=3)
              OR  (dep_year=2026 AND dep_month IN (2,3)) THEN TRUE ELSE FALSE END AS tag_ramazan,
        CASE WHEN dep_month IN (9,10,11) THEN TRUE ELSE FALSE END              AS tag_kongre_fuar,
        CASE WHEN dep_month IN (12,1,2,3) THEN TRUE ELSE FALSE END             AS tag_ski_sezonu,
        CASE WHEN dep_month IN (6,7,8,9) THEN TRUE ELSE FALSE END              AS tag_festival,
        CASE WHEN dep_dow BETWEEN 1 AND 5
              AND dep_month IN (1,2,3,9,10,11) THEN TRUE ELSE FALSE END        AS tag_is_seyahati,
        CASE WHEN (dep_year=2025 AND dep_month=6 AND region='Middle East')
              OR  (dep_year=2026 AND dep_month=5 AND region='Middle East')
              OR  (region='Middle East' AND dep_month IN (1,2,3,11,12))
             THEN TRUE ELSE FALSE END                                           AS tag_hac_umre,
        -- Special period flag
        CASE WHEN special_period IS NOT NULL THEN TRUE ELSE FALSE END           AS is_special_period
    FROM per_flight
    ORDER BY route, cabin_class, dep_date
""")

row_count = con.execute("SELECT COUNT(*) FROM route_daily").fetchone()[0]
entity_count = con.execute("SELECT COUNT(DISTINCT route || '_' || cabin_class) FROM route_daily").fetchone()[0]
print(f"       {row_count:,} satir, {entity_count} entity", flush=True)

# ── Step 5: Add time_idx and entity_id, write output ──
print("[5/5] Final duzenleme ve yazma...", flush=True)
con.execute(f"""
    COPY (
        SELECT
            route || '_' || cabin_class                             AS entity_id,
            route,
            cabin_class,
            dep_date,
            -- time_idx: 0-based from first date
            (dep_date - DATE '2025-01-01')::INT                    AS time_idx,
            -- All features
            region,
            direction,
            distance_km,
            flight_time_min,
            dep_year,
            dep_month,
            dep_dow,
            n_flights,
            -- Target
            total_pax,
            -- Fare
            avg_fare,
            std_fare,
            max_fare,
            min_fare,
            -- Booking
            n_bookings,
            avg_group_size,
            -- Channel
            corporate_pct,
            agency_pct,
            website_pct,
            mobile_pct,
            connecting_pct,
            -- DTD profile
            early_booking_pct,
            late_booking_pct,
            -- Passenger profile
            child_pct,
            gold_elite_pct,
            elite_pct,
            halal_pct,
            -- Events
            is_special_period,
            special_period,
            tag_yaz_tatili,
            tag_kis_tatili,
            tag_bahar_tatili,
            tag_ramazan,
            tag_kongre_fuar,
            tag_ski_sezonu,
            tag_festival,
            tag_is_seyahati,
            tag_hac_umre
        FROM route_daily
        ORDER BY entity_id, dep_date
    ) TO '{OUT_PATH}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

# ── Verification ──
df = pd.read_parquet(OUT_PATH)
print(f"\n=== SONUC ===")
print(f"Dosya: {OUT_PATH}")
print(f"Boyut: {OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")
print(f"Satir: {len(df):,}")
print(f"Entity: {df.entity_id.nunique()}")
print(f"Tarih araligi: {df.dep_date.min()} - {df.dep_date.max()}")
print(f"Sutunlar ({len(df.columns)}): {list(df.columns)}")
print(f"\nTarget (total_pax) istatistikleri:")
print(df.total_pax.describe().round(1))
print(f"\nEntity ornekleri: {sorted(df.entity_id.unique())[:5]}")
print(f"\n[DONE]")

con.close()
