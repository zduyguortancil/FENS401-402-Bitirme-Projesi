"""
Sprint 6 — Yolcu Segmentleri ve Talep Fonksiyonları (Demand Functions)
Emre'nin toplantıda belirlediği 6 yolcu segmenti için:
  - Willingness-to-pay (WTP) aralıkları
  - Booking window profilleri
  - Fiyat elastikiyeti fonksiyonları
  - Fiyat–talep eğrileri (price sweep)
Mevcut cluster ve training verisinden davranışsal parametreler türetilir.
"""
import json
import math
import duckdb
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
TRAIN_PATH = BASE_DIR / "demand_training.parquet"
META_PATH = BASE_DIR / "flight_metadata.parquet"
SNAP_PATH = BASE_DIR / "flight_snapshot_v2.parquet"
CLUSTER_PATH = BASE_DIR / "passenger_clusters.parquet"
OUT_REPORT = BASE_DIR / "demand_functions_report.json"

con = duckdb.connect()

# ═══════════════════════════════════════════════════════════════
# STEP 1: Mevcut veriden davranışsal parametreleri çıkar
# ═══════════════════════════════════════════════════════════════
print("[1/5] Mevcut veriden davranış parametreleri çıkarılıyor...", flush=True)

# Genel istatistikler
stats = con.execute(f"""
    SELECT
        AVG(y_pax_sold_today)                             AS avg_pax,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY y_pax_sold_today) AS pax_p25,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY y_pax_sold_today) AS pax_p75,
        AVG(load_factor)                                  AS avg_lf,
        AVG(CASE WHEN y_pax_sold_today > 0 THEN dtd END) AS avg_dtd_at_sale,
        AVG(capacity)                                     AS avg_capacity
    FROM read_parquet('{TRAIN_PATH}')
""").fetchone()
avg_pax, pax_p25, pax_p75 = float(stats[0]), float(stats[1]), float(stats[2])
avg_lf, avg_dtd_sale, avg_cap = float(stats[3]), float(stats[4]), float(stats[5])
print(f"   Ort. günlük pax: {avg_pax:.3f}  |  Ort. LF: {avg_lf:.3f}  |  Ort. kapasite: {avg_cap:.0f}", flush=True)

# DTD-bazlı satış profili (her DTD bucket'ta toplam satış oranı)
dtd_profile = con.execute(f"""
    SELECT
        dtd_bucket,
        SUM(y_pax_sold_today) AS total_pax,
        COUNT(*) AS row_count,
        AVG(CASE WHEN y_pax_sold_today > 0 THEN 1.0 ELSE 0.0 END) AS sale_rate
    FROM read_parquet('{TRAIN_PATH}')
    GROUP BY dtd_bucket
    ORDER BY dtd_bucket
""").fetchall()
total_pax_all = sum(r[1] for r in dtd_profile)
dtd_dist = {}
for r in dtd_profile:
    dtd_dist[int(r[0])] = {
        "total_pax": float(r[1]),
        "share": round(float(r[1]) / total_pax_all * 100, 2),
        "sale_rate": round(float(r[3]) * 100, 2),
    }
print(f"   DTD buckets: {len(dtd_dist)}", flush=True)

# Kabin bazlı fiyat referansı (snapshot'tan ortalama bilet fiyatı)
cabin_prices = con.execute(f"""
    SELECT
        LOWER(cabin_class) AS cabin,
        AVG(CASE WHEN pax_sold_today > 0
            THEN ticket_rev_today / pax_sold_today ELSE NULL END) AS avg_price,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY
            CASE WHEN pax_sold_today > 0 THEN ticket_rev_today / pax_sold_today END) AS price_p25,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY
            CASE WHEN pax_sold_today > 0 THEN ticket_rev_today / pax_sold_today END) AS price_p75,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY
            CASE WHEN pax_sold_today > 0 THEN ticket_rev_today / pax_sold_today END) AS price_p90
    FROM read_parquet('{SNAP_PATH}')
    WHERE pax_sold_today > 0
    GROUP BY LOWER(cabin_class)
""").fetchall()
price_ref = {}
for r in cabin_prices:
    price_ref[r[0]] = {
        "avg": round(float(r[1]), 2),
        "p25": round(float(r[2]), 2),
        "p75": round(float(r[3]), 2),
        "p90": round(float(r[4]), 2),
    }
print(f"   Fiyat referansları: {json.dumps(price_ref, indent=2)}", flush=True)

# Bölge bazlı fiyat farkları
region_prices = con.execute(f"""
    SELECT
        m.region,
        AVG(CASE WHEN s.pax_sold_today > 0
            THEN s.ticket_rev_today / s.pax_sold_today ELSE NULL END) AS avg_price,
        SUM(s.pax_sold_today) AS total_pax
    FROM read_parquet('{SNAP_PATH}') s
    LEFT JOIN read_parquet('{META_PATH}') m
        ON s.flight_id = m.flight_id AND LOWER(s.cabin_class) = LOWER(m.cabin_class)
    WHERE s.pax_sold_today > 0
    GROUP BY m.region
""").fetchall()
region_ref = {}
for r in region_prices:
    if r[0]:
        region_ref[r[0]] = {"avg_price": round(float(r[1]), 2), "total_pax": int(r[2])}

con.close()

# ═══════════════════════════════════════════════════════════════
# STEP 2: 6 Yolcu Segmenti Tanımla
# ═══════════════════════════════════════════════════════════════
print("[2/5] Yolcu segmentleri tanımlanıyor...", flush=True)

# Her kabin için base fiyat
eco_base = price_ref.get("economy", {}).get("avg", 500)
biz_base = price_ref.get("business", {}).get("avg", 1500)

SEGMENTS = {
    "A": {
        "id": "A",
        "name": "İş Yolcusu",
        "name_en": "Business Traveler",
        "icon": "💼",
        "color": "#3b82f6",
        "description": "Son dakika alır, yüksek bütçe, zaman hassasiyeti yüksek. Business class tercih eder.",
        "characteristics": [
            "Son dakika rezervasyon (0-14 gün)",
            "Yüksek bütçe, fiyata duyarsız",
            "Hafta içi, sabah uçuşları",
            "Frequent flyer üyesi",
            "Business class tercih",
        ],
        "booking_window": {"min_dtd": 0, "max_dtd": 14, "peak_dtd": 5},
        "wtp_multiplier": {"min": 1.8, "max": 2.5},
        "price_elasticity": -0.3,   # inelastic: fiyat %10 artarsa talep %3 düşer
        "base_share_pct": 15,        # bu segment toplam yolcunun ~%15'i
        "preferred_cabin": "business",
        "seasonal_boost": {"kongre_fuar": 1.4, "is_seyahati_yogun": 1.3, "normal": 1.0},
        "dtd_decay_rate": 0.15,      # yüksek → son günlerde yoğunlaşır
    },
    "B": {
        "id": "B",
        "name": "Gurbetçi (VFR)",
        "name_en": "Diaspora / VFR",
        "icon": "🏠",
        "color": "#10b981",
        "description": "Destinasyon sabit (memleket). Ağustos + bayram peak. Bagaj ağır. Fiyata orta duyarlı.",
        "characteristics": [
            "Sabit destinasyon (memleket)",
            "Yaz + bayram döneminde yoğun",
            "Yüksek bagaj hacmi",
            "Fiyata orta düzey duyarlı",
            "Economy class ağırlıklı",
        ],
        "booking_window": {"min_dtd": 14, "max_dtd": 60, "peak_dtd": 30},
        "wtp_multiplier": {"min": 1.3, "max": 1.8},
        "price_elasticity": -0.7,
        "base_share_pct": 20,
        "preferred_cabin": "economy",
        "seasonal_boost": {"yaz_tatili": 1.6, "bayram_donemi": 1.8, "ramazan_donemi": 1.3, "yilbasi": 1.4, "normal": 1.0},
        "dtd_decay_rate": 0.05,
    },
    "C": {
        "id": "C",
        "name": "Kongre / Tıbbi Seyahat",
        "name_en": "Congress / Medical",
        "icon": "🏥",
        "color": "#8b5cf6",
        "description": "Belirli tarih ve destinasyon. Grup potansiyeli. Esneklik düşük. Orta bütçe.",
        "characteristics": [
            "Tarih ve destinasyon sabit",
            "Grup halinde seyahat potansiyeli",
            "Esneklik düşük",
            "Orta bütçe seviyesi",
            "Economy + Business karışık",
        ],
        "booking_window": {"min_dtd": 7, "max_dtd": 30, "peak_dtd": 14},
        "wtp_multiplier": {"min": 1.2, "max": 1.6},
        "price_elasticity": -0.5,
        "base_share_pct": 12,
        "preferred_cabin": "economy",
        "seasonal_boost": {"kongre_fuar": 1.8, "festival_sezonu": 1.3, "normal": 1.0},
        "dtd_decay_rate": 0.08,
    },
    "D": {
        "id": "D",
        "name": "Erken Tatilci",
        "name_en": "Early Leisure",
        "icon": "🏖️",
        "color": "#f59e0b",
        "description": "Promosyon kovalar, rota esnek, fiyat belirleyici. Aylar öncesinden planlar.",
        "characteristics": [
            "60-180 gün önceden rezervasyon",
            "Promosyon ve indirim takipçisi",
            "Rota esnekliği yüksek",
            "Fiyat ana karar faktörü",
            "Economy class",
        ],
        "booking_window": {"min_dtd": 60, "max_dtd": 180, "peak_dtd": 90},
        "wtp_multiplier": {"min": 0.7, "max": 1.0},
        "price_elasticity": -1.5,   # very elastic
        "base_share_pct": 25,
        "preferred_cabin": "economy",
        "seasonal_boost": {"yaz_tatili": 1.5, "bahar_tatili": 1.3, "kis_tatili": 1.2, "normal": 1.0},
        "dtd_decay_rate": 0.02,
    },
    "E": {
        "id": "E",
        "name": "Öğrenci",
        "name_en": "Student",
        "icon": "🎓",
        "color": "#06b6d4",
        "description": "Bütçe kısıtlı, zaman ve rota esnek. İspanya pahalıysa Yunanistan'a gider.",
        "characteristics": [
            "Düşük bütçe, yüksek fiyat duyarlılığı",
            "Zaman ve rota esnekliği çok yüksek",
            "Alternatif destinasyonlara kayabilir",
            "30-120 gün önceden planlar",
            "Economy class, en düşük fare",
        ],
        "booking_window": {"min_dtd": 30, "max_dtd": 120, "peak_dtd": 60},
        "wtp_multiplier": {"min": 0.5, "max": 0.8},
        "price_elasticity": -2.2,   # extremely elastic
        "base_share_pct": 18,
        "preferred_cabin": "economy",
        "seasonal_boost": {"yariyil_tatili": 1.5, "yaz_tatili": 1.4, "bahar_tatili": 1.3, "normal": 1.0},
        "dtd_decay_rate": 0.03,
    },
    "F": {
        "id": "F",
        "name": "Son Dakika Acil",
        "name_en": "Last-Minute Urgent",
        "icon": "🚨",
        "color": "#ef4444",
        "description": "Acil durum, ihale, futbol maçı. Ne pahasına olursa olsun biner. Bagajda bile gider.",
        "characteristics": [
            "0-3 gün içinde bilet alır",
            "Fiyat duyarlılığı sıfır",
            "Acil durum motivasyonu",
            "Destinasyon ve zaman sabit",
            "Herhangi bir kabin sınıfı",
        ],
        "booking_window": {"min_dtd": 0, "max_dtd": 3, "peak_dtd": 1},
        "wtp_multiplier": {"min": 2.5, "max": 4.0},
        "price_elasticity": -0.1,   # completely inelastic
        "base_share_pct": 10,
        "preferred_cabin": "economy",
        "seasonal_boost": {"futbol_sezonu": 1.5, "bayram_donemi": 1.6, "normal": 1.0},
        "dtd_decay_rate": 0.5,
    },
}

for sid, seg in SEGMENTS.items():
    print(f"   {seg['icon']} Segment {sid}: {seg['name']} — elasticity={seg['price_elasticity']}, share={seg['base_share_pct']}%", flush=True)

# ═══════════════════════════════════════════════════════════════
# STEP 3: Talep Fonksiyonlarını Hesapla
# ═══════════════════════════════════════════════════════════════
print("[3/5] Talep fonksiyonları hesaplanıyor...", flush=True)


def demand_function(price_ratio, elasticity, dtd, peak_dtd, dtd_decay, seasonal=1.0, base_demand=1.0):
    """
    Talep fonksiyonu:
    Q(p, dtd) = base_demand × price_effect × timing_effect × seasonal_factor

    price_ratio: current_price / base_price (1.0 = base fiyat)
    elasticity: fiyat elastikiyeti (negatif, örn: -0.3)
    dtd: kalkışa gün sayısı
    peak_dtd: segmentin en yoğun alım yaptığı DTD
    dtd_decay: zamanlama yoğunlaşma katsayısı
    """
    # Price effect: Q = Q0 × (P/P0)^elasticity
    price_effect = max(price_ratio ** elasticity, 0.01)

    # Timing effect: Gaussian-like around peak_dtd
    dtd_sigma = max(peak_dtd * 0.6, 3)
    timing_effect = math.exp(-0.5 * ((dtd - peak_dtd) / dtd_sigma) ** 2)

    # Son dakikacılar için DTD 0'a yakın ek boost
    if dtd_decay >= 0.3 and dtd <= 3:
        timing_effect = max(timing_effect, 0.9)

    return round(base_demand * price_effect * timing_effect * seasonal, 4)


# Her segment × her fiyat noktası × her DTD bucket için eğri hesapla
price_ratios = [round(0.3 + i * 0.1, 1) for i in range(28)]  # 0.3x - 3.0x
dtd_points = [0, 1, 3, 5, 7, 14, 21, 30, 45, 60, 90, 120, 150, 180]

segment_curves = {}
for sid, seg in SEGMENTS.items():
    # Fiyat-Talep eğrisi (DTD = peak_dtd'de)
    price_demand_curve = []
    for pr in price_ratios:
        q = demand_function(
            price_ratio=pr,
            elasticity=seg["price_elasticity"],
            dtd=seg["booking_window"]["peak_dtd"],
            peak_dtd=seg["booking_window"]["peak_dtd"],
            dtd_decay=seg["dtd_decay_rate"],
            seasonal=1.0,
            base_demand=seg["base_share_pct"] / 100,
        )
        # Gelir = fiyat × miktar
        revenue = round(pr * q, 4)
        price_demand_curve.append({
            "price_ratio": pr,
            "demand": q,
            "revenue": revenue,
        })

    # DTD-Talep eğrisi (base fiyatta)
    dtd_demand_curve = []
    for dtd in dtd_points:
        q = demand_function(
            price_ratio=1.0,
            elasticity=seg["price_elasticity"],
            dtd=dtd,
            peak_dtd=seg["booking_window"]["peak_dtd"],
            dtd_decay=seg["dtd_decay_rate"],
            base_demand=seg["base_share_pct"] / 100,
        )
        dtd_demand_curve.append({
            "dtd": dtd,
            "demand": q,
        })

    # Optimal fiyat noktası (gelir maksimizasyonu)
    best_rev = max(price_demand_curve, key=lambda x: x["revenue"])

    segment_curves[sid] = {
        "price_demand": price_demand_curve,
        "dtd_demand": dtd_demand_curve,
        "optimal_price_ratio": best_rev["price_ratio"],
        "optimal_revenue": best_rev["revenue"],
        "optimal_demand": best_rev["demand"],
    }
    print(f"   Segment {sid}: optimal fiyat = {best_rev['price_ratio']:.1f}x base, "
          f"gelir-idx = {best_rev['revenue']:.4f}", flush=True)

# ═══════════════════════════════════════════════════════════════
# STEP 4: Segment Etkileşim Matrisi (Fiyat değişirse hangi segment kayar?)
# ═══════════════════════════════════════════════════════════════
print("[4/5] Segment etkileşim matrisi hesaplanıyor...", flush=True)

# Fiyat artarsa elastic segmentler ne kadar kayar?
interaction_matrix = {}
for sid, seg in SEGMENTS.items():
    row = {}
    for target_sid, target_seg in SEGMENTS.items():
        if sid == target_sid:
            row[target_sid] = 0.0
            continue
        # Eğer kaynak segment elastic ise ve hedef segment daha ucuz aralıktaysa → kayma potansiyeli
        if seg["price_elasticity"] < -1.0 and target_seg["wtp_multiplier"]["max"] < seg["wtp_multiplier"]["min"]:
            overlap = max(0, min(seg["booking_window"]["max_dtd"], target_seg["booking_window"]["max_dtd"])
                          - max(seg["booking_window"]["min_dtd"], target_seg["booking_window"]["min_dtd"]))
            overlap_ratio = overlap / max(seg["booking_window"]["max_dtd"] - seg["booking_window"]["min_dtd"], 1)
            shift = round(abs(seg["price_elasticity"]) * 0.1 * overlap_ratio, 3)
            row[target_sid] = shift
        else:
            row[target_sid] = 0.0
    interaction_matrix[sid] = row

# ═══════════════════════════════════════════════════════════════
# STEP 5: Rapor oluştur ve kaydet
# ═══════════════════════════════════════════════════════════════
print("[5/5] Rapor oluşturuluyor...", flush=True)

# Segment tanımlarını JSON-serializable yap
segments_json = {}
for sid, seg in SEGMENTS.items():
    segments_json[sid] = {
        "id": seg["id"],
        "name": seg["name"],
        "name_en": seg["name_en"],
        "icon": seg["icon"],
        "color": seg["color"],
        "description": seg["description"],
        "characteristics": seg["characteristics"],
        "booking_window": seg["booking_window"],
        "wtp_multiplier": seg["wtp_multiplier"],
        "price_elasticity": seg["price_elasticity"],
        "base_share_pct": seg["base_share_pct"],
        "preferred_cabin": seg["preferred_cabin"],
        "seasonal_boost": seg["seasonal_boost"],
        "dtd_decay_rate": seg["dtd_decay_rate"],
    }

report = {
    "version": "1.0",
    "total_segments": len(SEGMENTS),
    "data_summary": {
        "avg_daily_pax": round(avg_pax, 4),
        "avg_load_factor": round(avg_lf, 4),
        "avg_capacity": round(avg_cap, 1),
        "avg_dtd_at_sale": round(avg_dtd_sale, 1),
    },
    "price_reference": price_ref,
    "region_reference": region_ref,
    "dtd_distribution": dtd_dist,
    "segments": segments_json,
    "curves": {
        sid: {
            "price_demand": cv["price_demand"],
            "dtd_demand": cv["dtd_demand"],
            "optimal": {
                "price_ratio": cv["optimal_price_ratio"],
                "revenue_index": cv["optimal_revenue"],
                "demand_at_optimal": cv["optimal_demand"],
            },
        }
        for sid, cv in segment_curves.items()
    },
    "interaction_matrix": interaction_matrix,
    "price_sweep_range": {"min": 0.3, "max": 3.0, "step": 0.1},
    "dtd_points": dtd_points,
}

with open(OUT_REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n{'=' * 60}")
print("  TALEP FONKSİYONLARI RAPORU")
print("=" * 60)
print(f"  Segment sayısı:    {len(SEGMENTS)}")
print(f"  Fiyat aralığı:     0.3x - 3.0x base")
print(f"  DTD noktaları:     {len(dtd_points)}")
print()

for sid, seg in SEGMENTS.items():
    opt = segment_curves[sid]
    print(f"  {seg['icon']} {sid} — {seg['name']}")
    print(f"     Elastikiyet: {seg['price_elasticity']}  |  WTP: {seg['wtp_multiplier']['min']}-{seg['wtp_multiplier']['max']}x")
    print(f"     Booking: {seg['booking_window']['min_dtd']}-{seg['booking_window']['max_dtd']} gün (peak: {seg['booking_window']['peak_dtd']})")
    print(f"     Optimal fiyat: {opt['optimal_price_ratio']:.1f}x  →  gelir-idx: {opt['optimal_revenue']:.4f}")
    print()

print(f"  📋 Rapor: {OUT_REPORT}")
print("\n[DONE]")
