"""
calibrate_from_data.py — Pricing engine katsayilarini veriden ogren.

Elle belirlenmis katsayilar yerine, historik fiyat verisinden
regresyon ve istatistiksel analiz ile katsayilari turetir.

Cikti: reports/calibration_report.json
"""
import os
import json
import numpy as np
import duckdb
from sklearn.linear_model import LinearRegression

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")
SNAPSHOT = os.path.join(DATA_DIR, "raw", "flight_snapshot_v2.parquet")
METADATA = os.path.join(DATA_DIR, "processed", "flight_metadata.parquet")

os.makedirs(REPORTS_DIR, exist_ok=True)


def connect():
    return duckdb.connect()


def calibrate_base_price(con):
    """
    Economy ve Business icin mesafe-fiyat iliskisini regresyonla ogren.
    Sonuc: price_per_km katsayisi + intercept
    """
    print("[1/8] Base price calibration (distance -> rev/pax)...")
    df = con.execute(f"""
        SELECT m.cabin_class, m.distance_km,
               s.ticket_rev_cum / NULLIF(s.pax_sold_cum, 0) AS rev_per_pax
        FROM read_parquet('{SNAPSHOT}') s
        JOIN read_parquet('{METADATA}') m
            ON s.flight_id = m.flight_id AND s.cabin_class = m.cabin_class
        WHERE s.dtd = 0
          AND s.pax_sold_cum > 10
          AND s.ticket_rev_cum > 0
    """).fetchdf()

    result = {}
    for cabin in ["economy", "business"]:
        sub = df[df["cabin_class"].str.lower() == cabin].dropna()
        if len(sub) < 10:
            continue
        X = sub[["distance_km"]].values
        y = sub["rev_per_pax"].values
        reg = LinearRegression(fit_intercept=True)
        reg.fit(X, y)
        coef = float(reg.coef_[0])
        intercept = float(reg.intercept_)
        r2 = float(reg.score(X, y))
        result[cabin] = {
            "price_per_km": round(coef, 6),
            "intercept": round(intercept, 2),
            "r_squared": round(r2, 4),
            "n_samples": len(sub),
            "formula": f"base = {intercept:.2f} + distance_km x {coef:.6f}"
        }
        print(f"  {cabin}: base = {intercept:.2f} + dist x {coef:.6f}  (R²={r2:.4f}, n={len(sub)})")
    return result


def calibrate_season_factors(con):
    """Ay bazli sezon faktorlerini veriden hesapla.
    Gunluk bilet fiyatini DTD 30-60 arasinda olcerek sezon etkisini izole ederiz."""
    print("[2/8] Season factors (monthly price ratios, DTD-controlled)...")
    df = con.execute(f"""
        SELECT EXTRACT(MONTH FROM m.departure_datetime) AS dep_month,
               AVG(s.ticket_rev_today / NULLIF(s.pax_sold_today, 0)) AS avg_daily_price
        FROM read_parquet('{SNAPSHOT}') s
        JOIN read_parquet('{METADATA}') m
            ON s.flight_id = m.flight_id AND s.cabin_class = m.cabin_class
        WHERE s.dtd BETWEEN 30 AND 60
          AND s.pax_sold_today > 0
          AND LOWER(s.cabin_class) = 'economy'
        GROUP BY dep_month
        ORDER BY dep_month
    """).fetchdf()

    yearly_avg = df["avg_daily_price"].mean()
    factors = {}
    for _, row in df.iterrows():
        month = int(row["dep_month"])
        factor = round(float(row["avg_daily_price"]) / yearly_avg, 4)
        factors[str(month)] = factor
        print(f"  Month {month:2d}: {factor:.4f}")
    return {"factors": factors, "yearly_avg": round(float(yearly_avg), 2)}


def calibrate_dow_factors(con):
    """Hafta gunu bazli fiyat faktorleri (DTD-controlled)."""
    print("[3/8] Day-of-week factors (DTD-controlled)...")
    df = con.execute(f"""
        SELECT EXTRACT(DOW FROM m.departure_datetime) AS dep_dow,
               AVG(s.ticket_rev_today / NULLIF(s.pax_sold_today, 0)) AS avg_rev
        FROM read_parquet('{SNAPSHOT}') s
        JOIN read_parquet('{METADATA}') m
            ON s.flight_id = m.flight_id AND s.cabin_class = m.cabin_class
        WHERE s.dtd BETWEEN 30 AND 60
          AND s.pax_sold_today > 0
          AND LOWER(s.cabin_class) = 'economy'
        GROUP BY dep_dow
        ORDER BY dep_dow
    """).fetchdf()

    overall = df["avg_rev"].mean()
    factors = {}
    dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    for _, row in df.iterrows():
        dow = int(row["dep_dow"])
        factor = round(float(row["avg_rev"]) / overall, 4)
        factors[str(dow)] = factor
        print(f"  {dow_names.get(dow, dow)}: {factor:.4f}")
    return factors


def calibrate_region_factors(con):
    """Bolge bazli fiyat faktorleri."""
    print("[4/8] Region factors...")
    df = con.execute(f"""
        SELECT m.region,
               AVG(s.ticket_rev_cum / NULLIF(s.pax_sold_cum, 0)) AS avg_rev,
               COUNT(*) AS n
        FROM read_parquet('{SNAPSHOT}') s
        JOIN read_parquet('{METADATA}') m
            ON s.flight_id = m.flight_id AND s.cabin_class = m.cabin_class
        WHERE s.dtd = 0
          AND s.pax_sold_cum > 10
          AND LOWER(s.cabin_class) = 'economy'
        GROUP BY m.region
    """).fetchdf()

    overall = df["avg_rev"].mean()
    factors = {}
    for _, row in df.iterrows():
        region = row["region"]
        factor = round(float(row["avg_rev"]) / overall, 4)
        factors[region] = factor
        print(f"  {region}: {factor:.4f} (n={int(row['n'])})")
    return factors


def calibrate_dtd_factors(con):
    """DTD bazli fiyat egrileri — son dakika primi vs erken indirim."""
    print("[5/8] DTD price curve...")
    df = con.execute(f"""
        SELECT
            CASE
                WHEN s.dtd <= 3 THEN '0-3'
                WHEN s.dtd <= 7 THEN '4-7'
                WHEN s.dtd <= 14 THEN '8-14'
                WHEN s.dtd <= 30 THEN '15-30'
                WHEN s.dtd <= 60 THEN '31-60'
                WHEN s.dtd <= 90 THEN '61-90'
                WHEN s.dtd <= 120 THEN '91-120'
                ELSE '121+'
            END AS dtd_bucket,
            MIN(s.dtd) AS dtd_min,
            AVG(s.ticket_rev_today / NULLIF(s.pax_sold_today, 0)) AS avg_price_today
        FROM read_parquet('{SNAPSHOT}') s
        JOIN read_parquet('{METADATA}') m
            ON s.flight_id = m.flight_id AND s.cabin_class = m.cabin_class
        WHERE s.pax_sold_today > 0
          AND LOWER(s.cabin_class) = 'economy'
        GROUP BY dtd_bucket
        ORDER BY dtd_min
    """).fetchdf()

    # baseline = mid-range DTD (31-60)
    baseline_row = df[df["dtd_bucket"] == "31-60"]
    baseline = float(baseline_row["avg_price_today"].values[0]) if len(baseline_row) > 0 else df["avg_price_today"].mean()

    factors = {}
    for _, row in df.iterrows():
        bucket = row["dtd_bucket"]
        factor = round(float(row["avg_price_today"]) / baseline, 4)
        factors[bucket] = factor
        print(f"  DTD {bucket:>6s}: {factor:.4f} (avg ${float(row['avg_price_today']):.1f})")
    return {"factors": factors, "baseline_bucket": "31-60", "baseline_price": round(baseline, 2)}


def calibrate_route_factors(con):
    """Rota bazli fiyat faktorleri — mesafe normalize edildikten sonra rota spesifik etki."""
    print("[6/8] Route-specific factors...")
    df = con.execute(f"""
        SELECT
            m.departure_airport || '_' || m.arrival_airport AS route_key,
            m.distance_km,
            AVG(s.ticket_rev_cum / NULLIF(s.pax_sold_cum, 0)) AS avg_rev,
            COUNT(*) as n
        FROM read_parquet('{SNAPSHOT}') s
        JOIN read_parquet('{METADATA}') m
            ON s.flight_id = m.flight_id AND s.cabin_class = m.cabin_class
        WHERE s.dtd = 0
          AND s.pax_sold_cum > 10
          AND LOWER(s.cabin_class) = 'economy'
        GROUP BY route_key, m.distance_km
        HAVING COUNT(*) > 3
    """).fetchdf()

    # Mesafe etkisini cikar — geriye rota spesifik faktor kalir
    X = df[["distance_km"]].values
    y = df["avg_rev"].values
    reg = LinearRegression()
    reg.fit(X, y)
    predicted = reg.predict(X)

    factors = {}
    for i, row in df.iterrows():
        rk = row["route_key"]
        factor = round(float(row["avg_rev"]) / float(predicted[i]), 4) if predicted[i] > 0 else 1.0
        factors[rk] = factor

    # Top 5 / Bottom 5
    sorted_f = sorted(factors.items(), key=lambda x: x[1], reverse=True)
    print(f"  Total routes: {len(factors)}")
    print(f"  Top 3: {sorted_f[:3]}")
    print(f"  Bottom 3: {sorted_f[-3:]}")
    return factors


def calibrate_load_factor_curve(con):
    """Load factor vs fiyat iliskisi — supply multiplier icin."""
    print("[7/8] Load factor -> price relationship...")
    df = con.execute(f"""
        SELECT
            CASE
                WHEN s.pax_sold_cum * 1.0 / NULLIF(m.capacity, 0) < 0.30 THEN 'LF<30'
                WHEN s.pax_sold_cum * 1.0 / NULLIF(m.capacity, 0) < 0.50 THEN 'LF30-50'
                WHEN s.pax_sold_cum * 1.0 / NULLIF(m.capacity, 0) < 0.70 THEN 'LF50-70'
                WHEN s.pax_sold_cum * 1.0 / NULLIF(m.capacity, 0) < 0.85 THEN 'LF70-85'
                WHEN s.pax_sold_cum * 1.0 / NULLIF(m.capacity, 0) < 0.95 THEN 'LF85-95'
                ELSE 'LF95+'
            END AS lf_bucket,
            AVG(s.ticket_rev_today / NULLIF(s.pax_sold_today, 0)) AS avg_price
        FROM read_parquet('{SNAPSHOT}') s
        JOIN read_parquet('{METADATA}') m
            ON s.flight_id = m.flight_id AND s.cabin_class = m.cabin_class
        WHERE s.pax_sold_today > 0
          AND LOWER(s.cabin_class) = 'economy'
        GROUP BY lf_bucket
        ORDER BY lf_bucket
    """).fetchdf()

    # LF<30 as baseline (=1.0)
    baseline_row = df[df["lf_bucket"] == "LF<30"]
    baseline = float(baseline_row["avg_price"].values[0]) if len(baseline_row) > 0 else df["avg_price"].min()

    factors = {}
    for _, row in df.iterrows():
        bucket = row["lf_bucket"]
        factor = round(float(row["avg_price"]) / baseline, 4)
        factors[bucket] = factor
        print(f"  {bucket:>8s}: {factor:.4f} (avg ${float(row['avg_price']):.1f})")
    return {"factors": factors, "baseline": round(baseline, 2)}


def calibrate_cabin_ratio(con):
    """Economy vs Business yolcu orani."""
    print("[8/8] Cabin ratios...")
    df = con.execute(f"""
        SELECT LOWER(s.cabin_class) as cabin,
               AVG(s.pax_sold_cum) AS avg_pax,
               AVG(m.capacity) AS avg_cap,
               AVG(s.pax_sold_cum * 1.0 / NULLIF(m.capacity, 0)) AS avg_lf
        FROM read_parquet('{SNAPSHOT}') s
        JOIN read_parquet('{METADATA}') m
            ON s.flight_id = m.flight_id AND s.cabin_class = m.cabin_class
        WHERE s.dtd = 0 AND s.pax_sold_cum > 0
        GROUP BY cabin
    """).fetchdf()

    result = {}
    for _, row in df.iterrows():
        cab = row["cabin"]
        result[cab] = {
            "avg_pax": round(float(row["avg_pax"]), 1),
            "avg_capacity": round(float(row["avg_cap"]), 1),
            "avg_load_factor": round(float(row["avg_lf"]), 4),
        }
        print(f"  {cab}: avg_pax={row['avg_pax']:.0f}, cap={row['avg_cap']:.0f}, LF={row['avg_lf']:.2%}")
    return result


def main():
    print("=" * 60)
    print("SeatWise — Pricing Calibration from Data")
    print("=" * 60)
    print()

    con = connect()

    report = {
        "_meta": {
            "description": "Pricing engine katsayilari — veriden ogrenilmis",
            "source": "flight_snapshot_v2.parquet + flight_metadata.parquet",
            "method": "Regresyon + istatistiksel oran analizi",
        },
        "base_price": calibrate_base_price(con),
        "season_factors": calibrate_season_factors(con),
        "dow_factors": calibrate_dow_factors(con),
        "region_factors": calibrate_region_factors(con),
        "dtd_factors": calibrate_dtd_factors(con),
        "route_factors": calibrate_route_factors(con),
        "lf_curve": calibrate_load_factor_curve(con),
        "cabin_ratios": calibrate_cabin_ratio(con),
    }

    con.close()

    # Kaydet
    out_path = os.path.join(REPORTS_DIR, "calibration_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print(f"Calibration report saved: {out_path}")
    print(f"Toplam {len(report) - 1} katsayi grubu ogrenildi.")


if __name__ == "__main__":
    main()
