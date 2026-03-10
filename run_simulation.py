"""
Sprint 7 — Simülasyon Motoru: Sabit Fiyat vs Dinamik Fiyat
3 rota × 2 kabin × 180 gün simülasyonu.
Talep fonksiyonlarından segment bazlı booking davranışı.
Fare class yönetimi ile gelir optimizasyonu.
Çıktı: simulation_report.json (ROI karşılaştırması)
"""
import json
import math
import random
import duckdb
from pathlib import Path

BASE_DIR = Path(__file__).parent
DEMAND_REPORT = BASE_DIR / "demand_functions_report.json"
SNAP_PATH = BASE_DIR / "flight_snapshot_v2.parquet"
META_PATH = BASE_DIR / "flight_metadata.parquet"
OUT_REPORT = BASE_DIR / "simulation_report.json"

random.seed(42)

# ═══════════════════════════════════════════════════════════
# STEP 1: Verileri yükle
# ═══════════════════════════════════════════════════════════
print("[1/5] Veriler yükleniyor...", flush=True)

with open(DEMAND_REPORT, "r", encoding="utf-8") as f:
    demand_data = json.load(f)

SEGMENTS = demand_data["segments"]
PRICE_REF = demand_data["price_reference"]

# Gerçek rota fiyatlarını çek
con = duckdb.connect()

route_prices = con.execute(f"""
    SELECT
        m.departure_airport || '-' || m.arrival_airport AS route,
        m.departure_airport, m.arrival_airport, m.region,
        LOWER(s.cabin_class) AS cabin,
        AVG(m.capacity) AS capacity,
        AVG(m.distance_km) AS distance_km,
        AVG(CASE WHEN s.pax_sold_today > 0
            THEN s.ticket_rev_today / s.pax_sold_today END) AS avg_price,
        MIN(CASE WHEN s.pax_sold_today > 0
            THEN s.ticket_rev_today / s.pax_sold_today END) AS min_price,
        MAX(CASE WHEN s.pax_sold_today > 0
            THEN s.ticket_rev_today / s.pax_sold_today END) AS max_price,
        SUM(s.pax_sold_today) AS total_pax
    FROM read_parquet('{SNAP_PATH}') s
    LEFT JOIN read_parquet('{META_PATH}') m
        ON s.flight_id = m.flight_id AND LOWER(s.cabin_class) = LOWER(m.cabin_class)
    WHERE s.pax_sold_today > 0
      AND m.departure_airport IN ('IST')
      AND m.arrival_airport IN ('LHR', 'CDG', 'AUH')
    GROUP BY route, m.departure_airport, m.arrival_airport, m.region, cabin
    ORDER BY route, cabin
""").fetchall()
con.close()

# 3 simülasyon rotası
SIM_ROUTES = {}
for r in route_prices:
    route_key = r[0]
    cabin = r[4]
    if route_key not in SIM_ROUTES:
        SIM_ROUTES[route_key] = {
            "route": route_key,
            "dep": r[1], "arr": r[2], "region": r[3],
            "cabins": {},
        }
    SIM_ROUTES[route_key]["cabins"][cabin] = {
        "capacity": int(r[5]),
        "distance_km": int(r[6]),
        "avg_price": round(float(r[7]), 2),
        "min_price": round(float(r[8]), 2),
        "max_price": round(float(r[9]), 2),
        "total_pax": int(r[10]),
    }

print(f"   {len(SIM_ROUTES)} rota yüklendi:", flush=True)
for rk, rv in SIM_ROUTES.items():
    for cb, cv in rv["cabins"].items():
        print(f"     {rk} {cb}: avg=${cv['avg_price']:.0f}, cap={cv['capacity']}", flush=True)


# ═══════════════════════════════════════════════════════════
# STEP 2: Fare Class Tanımları
# ═══════════════════════════════════════════════════════════
print("\n[2/5] Fare class yapısı tanımlanıyor...", flush=True)

# 4 fare class: V (en ucuz) → K → M → Y (en pahalı)
FARE_CLASSES = {
    "V": {"name": "V — Promosyon", "multiplier": 0.5, "protection": 0,    "open_until_lf": 0.40},
    "K": {"name": "K — İndirimli",  "multiplier": 0.75, "protection": 0.2, "open_until_lf": 0.60},
    "M": {"name": "M — Esnek",      "multiplier": 1.0,  "protection": 0.4, "open_until_lf": 0.85},
    "Y": {"name": "Y — Tam Fiyat",  "multiplier": 1.5,  "protection": 0.6, "open_until_lf": 1.0},
}

# Hangi DTD aralığında hangi fare class'lar açık?
# Erken: V, K açık | Orta: K, M açık | Son: M, Y açık
DTD_FARE_RULES = [
    {"dtd_min": 60, "dtd_max": 180, "open": ["V", "K", "M"]},
    {"dtd_min": 30, "dtd_max": 59,  "open": ["K", "M"]},
    {"dtd_min": 14, "dtd_max": 29,  "open": ["K", "M", "Y"]},
    {"dtd_min": 7,  "dtd_max": 13,  "open": ["M", "Y"]},
    {"dtd_min": 0,  "dtd_max": 6,   "open": ["Y"]},
]

for fc_id, fc in FARE_CLASSES.items():
    print(f"   {fc['name']}: {fc['multiplier']:.2f}x base, protection={fc['protection']:.0%}", flush=True)


# ═══════════════════════════════════════════════════════════
# STEP 3: Talep Fonksiyonu
# ═══════════════════════════════════════════════════════════
print("\n[3/5] Simülasyon başlıyor...", flush=True)


def calc_segment_demand(segment, price_ratio, dtd, seasonal=1.0):
    """Bir segment için belirli DTD ve fiyat oranında günlük talep."""
    elast = segment["price_elasticity"]
    peak_dtd = segment["booking_window"]["peak_dtd"]
    min_dtd = segment["booking_window"]["min_dtd"]
    max_dtd = segment["booking_window"]["max_dtd"]
    share = segment["base_share_pct"] / 100

    # DTD booking window dışındaysa talep çok düşük
    if dtd < min_dtd or dtd > max_dtd:
        return 0.0

    # Price effect: Q = Q0 × (P/P0)^elasticity
    price_effect = max(price_ratio ** elast, 0.01)

    # Timing effect: Gaussian around peak_dtd
    dtd_sigma = max(peak_dtd * 0.6, 3)
    timing = math.exp(-0.5 * ((dtd - peak_dtd) / dtd_sigma) ** 2)

    # Son dakikacılar boost
    if segment["dtd_decay_rate"] >= 0.3 and dtd <= 3:
        timing = max(timing, 0.9)

    base_demand = share * price_effect * timing * seasonal

    # Stokastik noise
    noise = random.gauss(1.0, 0.15)
    return max(base_demand * noise, 0)


def get_open_fare_classes(dtd, load_factor):
    """DTD ve doluluk oranına göre hangi fare class'lar açık."""
    # DTD kuralından açık olan class'ları bul
    open_classes = []
    for rule in DTD_FARE_RULES:
        if rule["dtd_min"] <= dtd <= rule["dtd_max"]:
            open_classes = rule["open"]
            break
    if not open_classes:
        open_classes = ["Y"]  # fallback

    # Doluluk yüksekse düşük class'ları kapat (protection)
    filtered = []
    for fc_id in open_classes:
        fc = FARE_CLASSES[fc_id]
        if load_factor < fc["open_until_lf"]:
            filtered.append(fc_id)
        elif fc_id == "Y":
            filtered.append(fc_id)  # Y her zaman açık

    return filtered if filtered else ["Y"]


def match_segment_to_fare(segment, open_fares):
    """Segment'in WTP'sine göre en uygun fare class'ı seç."""
    wtp_avg = (segment["wtp_multiplier"]["min"] + segment["wtp_multiplier"]["max"]) / 2

    best_fc = None
    best_price = None

    # Segment'in ödeyeceği en yüksek fare class'ı bul (gelir maksimize)
    for fc_id in reversed(open_fares):  # Y'den V'ye
        fc = FARE_CLASSES[fc_id]
        if fc["multiplier"] <= wtp_avg:
            best_fc = fc_id
            best_price = fc["multiplier"]
            break

    if best_fc is None:
        # Segment'in WTP'si hiçbir açık class'a yetmiyorsa en ucuz olanı al
        best_fc = open_fares[0]
        fc = FARE_CLASSES[best_fc]
        if wtp_avg < fc["multiplier"] * 0.7:
            return None, None  # hiç alamaz
        best_price = fc["multiplier"]

    return best_fc, best_price


# ═══════════════════════════════════════════════════════════
# STEP 4: Her rota için simülasyon çalıştır
# ═══════════════════════════════════════════════════════════

SIM_DAYS = 181  # DTD 180 → 0
results = {}

for route_key, route_info in SIM_ROUTES.items():
    print(f"\n  ✈️  {route_key} ({route_info['region']})", flush=True)

    for cabin, cabin_data in route_info["cabins"].items():
        capacity = cabin_data["capacity"]
        base_price = cabin_data["avg_price"]

        # Kabin kapasitesi: economy tipik %70, business %30
        if cabin == "economy":
            cabin_cap = int(capacity * 0.70)
        else:
            cabin_cap = int(capacity * 0.30)

        # ─── SENARYO A: SABİT FİYAT ─────────────
        fixed_revenue = 0
        fixed_pax = 0
        fixed_daily = []

        remaining_fixed = cabin_cap
        for dtd in range(SIM_DAYS - 1, -1, -1):
            day_pax = 0
            day_rev = 0

            for sid, seg in SEGMENTS.items():
                demand = calc_segment_demand(seg, 1.0, dtd)

                # Kapasiteden fazla satılamaz
                actual = min(demand, remaining_fixed)
                actual = max(actual, 0)

                if actual > 0:
                    rev = actual * base_price
                    day_pax += actual
                    day_rev += rev
                    remaining_fixed -= actual

            fixed_pax += day_pax
            fixed_revenue += day_rev
            lf = 1 - (remaining_fixed / cabin_cap) if cabin_cap > 0 else 0
            fixed_daily.append({
                "dtd": dtd,
                "pax": round(day_pax, 2),
                "revenue": round(day_rev, 2),
                "cumulative_rev": round(fixed_revenue, 2),
                "load_factor": round(lf, 4),
                "price": base_price,
            })

        # ─── SENARYO B: DİNAMİK FİYAT ──────────
        dynamic_revenue = 0
        dynamic_pax = 0
        dynamic_daily = []
        fare_class_revenue = {fc: 0 for fc in FARE_CLASSES}
        fare_class_pax = {fc: 0 for fc in FARE_CLASSES}

        remaining_dynamic = cabin_cap
        for dtd in range(SIM_DAYS - 1, -1, -1):
            day_pax = 0
            day_rev = 0
            lf = 1 - (remaining_dynamic / cabin_cap) if cabin_cap > 0 else 0

            # Açık fare class'ları belirle
            open_fares = get_open_fare_classes(dtd, lf)

            for sid, seg in SEGMENTS.items():
                # Segment'e uygun fare class bul
                fc_id, price_mult = match_segment_to_fare(seg, open_fares)
                if fc_id is None:
                    continue  # segment bu fiyatı ödeyemez

                actual_price = base_price * price_mult
                demand = calc_segment_demand(seg, price_mult, dtd)

                actual = min(demand, remaining_dynamic)
                actual = max(actual, 0)

                if actual > 0:
                    rev = actual * actual_price
                    day_pax += actual
                    day_rev += rev
                    remaining_dynamic -= actual
                    fare_class_revenue[fc_id] += rev
                    fare_class_pax[fc_id] += actual

            dynamic_pax += day_pax
            dynamic_revenue += day_rev
            lf = 1 - (remaining_dynamic / cabin_cap) if cabin_cap > 0 else 0

            # En yüksek açık fare class'ın fiyatı
            current_price = base_price * FARE_CLASSES[open_fares[-1]]["multiplier"]

            dynamic_daily.append({
                "dtd": dtd,
                "pax": round(day_pax, 2),
                "revenue": round(day_rev, 2),
                "cumulative_rev": round(dynamic_revenue, 2),
                "load_factor": round(lf, 4),
                "price": round(current_price, 2),
                "open_fares": open_fares,
            })

        # ─── SONUÇLAR ───────────────────────────
        delta = dynamic_revenue - fixed_revenue
        roi_pct = (delta / fixed_revenue * 100) if fixed_revenue > 0 else 0

        result_key = f"{route_key}_{cabin}"
        results[result_key] = {
            "route": route_key,
            "departure": route_info["dep"],
            "arrival": route_info["arr"],
            "region": route_info["region"],
            "cabin": cabin,
            "capacity": cabin_cap,
            "base_price": base_price,
            "static": {
                "total_revenue": round(fixed_revenue, 2),
                "total_pax": round(fixed_pax, 2),
                "load_factor": round(1 - remaining_fixed / cabin_cap, 4) if cabin_cap > 0 else 0,
                "avg_price": round(fixed_revenue / fixed_pax, 2) if fixed_pax > 0 else 0,
                "daily": fixed_daily,
            },
            "dynamic": {
                "total_revenue": round(dynamic_revenue, 2),
                "total_pax": round(dynamic_pax, 2),
                "load_factor": round(1 - remaining_dynamic / cabin_cap, 4) if cabin_cap > 0 else 0,
                "avg_price": round(dynamic_revenue / dynamic_pax, 2) if dynamic_pax > 0 else 0,
                "daily": dynamic_daily,
                "fare_class_breakdown": {
                    fc: {
                        "revenue": round(fare_class_revenue[fc], 2),
                        "pax": round(fare_class_pax[fc], 2),
                        "avg_price": round(fare_class_revenue[fc] / fare_class_pax[fc], 2) if fare_class_pax[fc] > 0 else 0,
                    }
                    for fc in FARE_CLASSES
                },
            },
            "comparison": {
                "revenue_delta": round(delta, 2),
                "roi_pct": round(roi_pct, 2),
                "pax_delta": round(dynamic_pax - fixed_pax, 2),
            },
        }

        emoji = "📈" if delta > 0 else "📉"
        print(f"     {cabin:10s}  Sabit: ${fixed_revenue:>10,.0f}  |  Dinamik: ${dynamic_revenue:>10,.0f}  |  "
              f"{emoji} Δ = ${delta:>+8,.0f} ({roi_pct:>+.1f}%)", flush=True)


# ═══════════════════════════════════════════════════════════
# STEP 5: Rapor oluştur
# ═══════════════════════════════════════════════════════════
print("\n[5/5] Rapor oluşturuluyor...", flush=True)

# Toplam ROI
total_static = sum(r["static"]["total_revenue"] for r in results.values())
total_dynamic = sum(r["dynamic"]["total_revenue"] for r in results.values())
total_delta = total_dynamic - total_static
total_roi = (total_delta / total_static * 100) if total_static > 0 else 0

report = {
    "version": "1.0",
    "simulation_config": {
        "days": SIM_DAYS,
        "routes": list(SIM_ROUTES.keys()),
        "cabins": ["economy", "business"],
        "fare_classes": {fc_id: {
            "name": fc["name"],
            "multiplier": fc["multiplier"],
            "protection": fc["protection"],
        } for fc_id, fc in FARE_CLASSES.items()},
        "dtd_fare_rules": DTD_FARE_RULES,
        "segments_used": list(SEGMENTS.keys()),
    },
    "summary": {
        "total_static_revenue": round(total_static, 2),
        "total_dynamic_revenue": round(total_dynamic, 2),
        "total_delta": round(total_delta, 2),
        "total_roi_pct": round(total_roi, 2),
        "interpretation": (
            f"Bu sistem kullanılmasaydı ${total_static:,.0f} kazanılacaktı. "
            f"Dinamik fiyatlama sistemi ile ${total_dynamic:,.0f} kazanıldı. "
            f"Sistemin sağladığı net katma değer: ${total_delta:,.0f} (%{total_roi:.1f} artış)."
        ),
    },
    "routes": results,
}

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

# ─── ÇIKTI ────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("  SİMÜLASYON SONUÇLARI — SABİT FİYAT vs DİNAMİK FİYATLAMA")
print("=" * 70)
print(f"\n  🏦 Toplam Sabit Fiyat Geliri:    ${total_static:>12,.0f}")
print(f"  🚀 Toplam Dinamik Fiyat Geliri:  ${total_dynamic:>12,.0f}")
print(f"  {'─' * 46}")
print(f"  💰 Net Katma Değer:              ${total_delta:>+12,.0f}")
print(f"  📊 ROI:                          {total_roi:>+11.1f}%")
print(f"\n  💬 {report['summary']['interpretation']}")
print(f"\n  📋 Rapor: {OUT_REPORT}")
print("\n[DONE]")
