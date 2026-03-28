"""
Dynamic Pricing Engine — Fiyat hesaplama motoru.

Formula: fiyat = baz_fiyat x arz_carpani x talep_carpani x sentiment_carpani x musteri_carpani

Girisler:
  - XGBoost Pickup: kalan yolcu tahmini (arz)
  - TFT Forecast: rota bazli talep trendi (talep)
  - Sentiment: destinasyon sehir skoru (dis etken)
  - Musteri segmenti: WTP + elastikiyet (musteri)
  - Fare class: DTD + LF bazli class yonetimi
"""
import json
import math
import os
from datetime import datetime, timedelta

# ── Fare class tanimlari ──────────────────────────────────────
FARE_CLASSES = {
    "V": {"name": "V — Promosyon",    "multiplier": 0.50, "open_until_lf": 0.40, "quota_pct": 0.15, "color": "#94a3b8",
           "features": "En dusuk fiyat. Degisiklik/iptal yok. Bagaj sinirli."},
    "K": {"name": "K — Indirimli",    "multiplier": 0.75, "open_until_lf": 0.60, "quota_pct": 0.25, "color": "#c9a227",
           "features": "Indirimli fiyat. Ucretli degisiklik. 1 bagaj."},
    "M": {"name": "M — Esnek",        "multiplier": 1.00, "open_until_lf": 0.85, "quota_pct": 0.35, "color": "#6366f1",
           "features": "Standart fiyat. Ucretsiz degisiklik. 2 bagaj."},
    "Y": {"name": "Y — Tam Fiyat",    "multiplier": 1.50, "open_until_lf": 1.00, "quota_pct": 1.00, "color": "#ef4444",
           "features": "Tam esneklik. Ucretsiz iptal/degisiklik. 2 bagaj + ozel."},
}
# quota_pct: kapasitenin yuzde kaci bu class'ta satilabilir
# V: %15 = economy 300'de 45 koltuk, dolunca V kapanir
# K: %25 = 75 koltuk (V+K toplam)
# M: %35 = 105 koltuk (V+K+M toplam)
# Y: %100 = sinir yok, her zaman acik

# DTD kurallari: hangi fare class'lar hangi donemde acik
DTD_RULES = [
    {"dtd_min": 60,  "dtd_max": 999, "open": ["V", "K", "M"],      "label": "Erken Donem"},
    {"dtd_min": 30,  "dtd_max": 59,  "open": ["K", "M"],            "label": "Orta Donem"},
    {"dtd_min": 14,  "dtd_max": 29,  "open": ["K", "M", "Y"],       "label": "Gec Donem"},
    {"dtd_min": 7,   "dtd_max": 13,  "open": ["M", "Y"],            "label": "Son Hafta"},
    {"dtd_min": 0,   "dtd_max": 6,   "open": ["Y"],                  "label": "Son Dakika"},
]

# Baz fiyat formulu (sentetik veri ureticisiyle ayni)
BASE_PRICE_FORMULAS = {
    "economy":  lambda dist_km: max(dist_km * 0.08, 150),
    "business": lambda dist_km: max(dist_km * 0.35, 800),
}

# Bolge fiyat faktorleri
REGION_FACTORS = {
    "Europe": 1.00, "Middle East": 1.10, "Africa": 0.95,
    "Asia": 1.05, "Americas": 1.15,
}

# Sezon faktorleri (ay bazli)
SEASON_FACTORS = {
    1: 0.85, 2: 0.85, 3: 1.00, 4: 1.05, 5: 1.10, 6: 1.25,
    7: 1.30, 8: 1.25, 9: 1.05, 10: 1.00, 11: 0.90, 12: 1.20,
}

# Ozel gunler
SPECIAL_PERIODS = {
    # 2026
    (2026, 3, 18): ("Ramazan Bayrami", 1.5),
    (2026, 3, 19): ("Ramazan Bayrami", 1.5),
    (2026, 3, 20): ("Ramazan Bayrami", 1.5),
    (2026, 3, 21): ("Ramazan Bayrami", 1.5),
    (2026, 3, 22): ("Ramazan Bayrami", 1.5),
    (2026, 3, 23): ("Ramazan Bayrami", 1.5),
    (2026, 5, 25): ("Kurban Bayrami", 1.6),
    (2026, 5, 26): ("Kurban Bayrami", 1.6),
    (2026, 5, 27): ("Kurban Bayrami", 1.6),
    (2026, 5, 28): ("Kurban Bayrami", 1.6),
    (2026, 5, 29): ("Kurban Bayrami", 1.6),
    (2026, 5, 30): ("Kurban Bayrami", 1.6),
    (2026, 10, 29): ("Cumhuriyet Bayrami", 1.25),
    (2026, 10, 30): ("Cumhuriyet Bayrami", 1.25),
    (2026, 12, 28): ("Yilbasi", 1.4),
    (2026, 12, 29): ("Yilbasi", 1.4),
    (2026, 12, 30): ("Yilbasi", 1.4),
    (2026, 12, 31): ("Yilbasi", 1.4),
}

# Hafta gunu faktorleri
DOW_FACTORS = {0: 1.05, 1: 1.00, 2: 1.00, 3: 1.10, 4: 1.15, 5: 1.10, 6: 1.05}


class PricingEngine:
    """Dinamik fiyatlandirma motoru."""

    def __init__(self, segments, route_distances, sentiment_cache=None, airport_to_city=None):
        """
        segments: demand_functions_report.json'dan segment tanimlari
        route_distances: {route_key: distance_km} dict
        sentiment_cache: _SENT_CACHE referansi
        airport_to_city: AIRPORT_TO_CITY mapping
        """
        self.segments = segments
        self.route_distances = route_distances
        self.sentiment_cache = sentiment_cache or {}
        self.airport_to_city = airport_to_city or {}

        # Rota bazli fiyat faktorleri (veriden turetilmis)
        self.route_price_factors = {}
        try:
            import json
            factors_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', 'reports', 'route_price_factors.json')
            if os.path.exists(factors_path):
                with open(factors_path, encoding='utf-8') as f:
                    self.route_price_factors = json.load(f)
        except Exception:
            pass

    # ── ANA FONKSIYON ─────────────────────────────────────────
    def compute_price(self, inventory, dtd, segment_id=None, session_info=None, predicted_remaining=None):
        """
        Bir ucus+kabin icin fiyat hesapla.

        Args:
            inventory: dict — ucus envanter durumu (capacity, sold, route, cabin, dep_date, ...)
            dtd: int — kalkisa kalan gun
            segment_id: str — musteri segmenti (A-F), None ise genel fiyat
            session_info: dict — musteri davranis verileri (booking.html'den)

        Returns: dict — {
            prices: {V: $, K: $, M: $, Y: $},
            open_fares: ["V", "K"],
            best_fare: "V",
            best_price: 175.50,
            multipliers: {supply, demand, sentiment, customer},
            baseline_price: 350.0,
            demand_pressure: 1.2,
        }
        """
        cabin = inventory.get("cabin", "economy")
        route = inventory.get("route", "IST-LHR")
        dep_date = inventory.get("dep_date")
        lf = inventory.get("load_factor", 0.0)
        remaining = inventory.get("capacity", 300) - inventory.get("sold", 0)

        # Baz fiyat
        dist_km = self._get_distance(route)
        base_price = self._compute_base_price(cabin, dist_km, route)

        # 4 carpan
        supply_mult = self._supply_multiplier(lf, remaining, dtd, inventory, predicted_remaining)
        demand_mult = self._demand_multiplier(dep_date, route, cabin)
        sentiment_mult = self._sentiment_multiplier(route)
        customer_mult = self._customer_multiplier(segment_id, session_info)

        # Bilesik carpan
        combined = supply_mult * demand_mult * sentiment_mult * customer_mult

        # Fare class yonetimi
        demand_pressure = supply_mult * demand_mult
        open_fares = self._get_open_fares(dtd, lf, demand_pressure, inventory)

        # Her fare class icin fiyat hesapla
        prices = {}
        for fc_id in ["V", "K", "M", "Y"]:
            fc_mult = FARE_CLASSES[fc_id]["multiplier"]
            prices[fc_id] = round(base_price * fc_mult * combined, 2)

        # En ucuz acik fare
        best_fare = open_fares[0] if open_fares else "Y"
        best_price = prices[best_fare]

        # Baseline (eski formul) karsilastirma
        baseline = self.compute_baseline_price(route, cabin, dtd, dep_date)

        return {
            "prices": prices,
            "open_fares": open_fares,
            "best_fare": best_fare,
            "best_price": best_price,
            "base_price": round(base_price, 2),
            "multipliers": {
                "supply": round(supply_mult, 4),
                "demand": round(demand_mult, 4),
                "sentiment": round(sentiment_mult, 4),
                "customer": round(customer_mult, 4),
                "combined": round(combined, 4),
            },
            "baseline_price": round(baseline, 2),
            "delta_pct": round((best_price - baseline) / max(baseline, 1) * 100, 1),
            "demand_pressure": round(demand_pressure, 4),
            "fare_classes": {
                fc_id: {
                    "name": fc["name"],
                    "price": prices[fc_id],
                    "open": fc_id in open_fares,
                    "color": fc["color"],
                    "features": fc["features"],
                }
                for fc_id, fc in FARE_CLASSES.items()
            },
        }

    # ── ARZ CARPANI ───────────────────────────────────────────
    def _supply_multiplier(self, lf, remaining, dtd, inventory, predicted_remaining=None):
        """
        Doluluk + kalan koltuk + DTD durumuna gore arz carpani.
        predicted_remaining varsa: model-based expected_final_lf kullanir.
        """
        if remaining <= 0:
            return 2.5  # sold out, max

        dtd_boost = 1.0
        if dtd <= 3:
            dtd_boost = 1.3
        elif dtd <= 7:
            dtd_boost = 1.15
        elif dtd <= 14:
            dtd_boost = 1.05

        # ── MODEL-DRIVEN: XGBoost Pickup ──
        if predicted_remaining is not None:
            cap = inventory.get("capacity", 300)
            sold = inventory.get("sold", 0)
            expected_final_lf = (sold + predicted_remaining) / cap if cap > 0 else 0
            if expected_final_lf >= 0.95:
                lf_mult = 1.80
            elif expected_final_lf >= 0.85:
                lf_mult = 1.40
            elif expected_final_lf >= 0.70:
                lf_mult = 1.15
            elif expected_final_lf >= 0.50:
                lf_mult = 1.00
            elif expected_final_lf >= 0.35:
                lf_mult = 0.90
            else:
                lf_mult = 0.80
            return lf_mult * dtd_boost

        # ── FALLBACK: mevcut LF heuristic ──
        if lf >= 0.95:
            lf_mult = 2.00
        elif lf >= 0.85:
            lf_mult = 1.50 + (lf - 0.85) * 5.0
        elif lf >= 0.70:
            lf_mult = 1.20 + (lf - 0.70) * 2.0
        elif lf >= 0.50:
            lf_mult = 1.05 + (lf - 0.50) * 0.75
        elif lf >= 0.30:
            lf_mult = 1.00 + (lf - 0.30) * 0.25
        else:
            lf_mult = 1.00

        return lf_mult * dtd_boost

    # ── TALEP CARPANI ─────────────────────────────────────────
    def _demand_multiplier(self, dep_date, route, cabin):
        """
        Sezonsellik + ozel gun + hafta gunu etkisi.
        TFT trend entegrasyonu sonraki fazda eklenecek.
        """
        if dep_date is None:
            return 1.0

        if isinstance(dep_date, str):
            dep_date = datetime.strptime(dep_date, "%Y-%m-%d").date()
        elif isinstance(dep_date, datetime):
            dep_date = dep_date.date()

        # Sezon
        season = SEASON_FACTORS.get(dep_date.month, 1.0)

        # Ozel gun
        special_key = (dep_date.year, dep_date.month, dep_date.day)
        special = SPECIAL_PERIODS.get(special_key)
        special_mult = special[1] if special else 1.0

        # Hafta gunu
        dow = DOW_FACTORS.get(dep_date.weekday(), 1.0)

        # Bolge faktoru artik baz fiyatta (rota bazli), burada tekrar uygulanmaz
        raw = season * max(special_mult, 1.0) * dow
        # Dampen: demand etkisini azalt — ana fark supply'dan gelmeli
        return 1.0 + (raw - 1.0) * 0.3

    # ── SENTIMENT CARPANI ─────────────────────────────────────
    def _sentiment_multiplier(self, route):
        """
        Destinasyon sehir sentiment skoru.
        Negatif olay = fiyat dusur (talep azalir)
        Pozitif olay = fiyat arttir (talep artar)
        """
        arr = route.split("-")[1] if "-" in route else ""
        city_key = self.airport_to_city.get(arr)

        if not city_key or not self.sentiment_cache.get("data"):
            return 1.0

        city_data = self.sentiment_cache["data"].get(city_key, {})
        score = city_data.get("aggregate", {}).get("composite_score", 0.0)

        # score: -1.0 (cok negatif) → +1.0 (cok pozitif)
        # Mapping: -1.0 → 0.85, 0.0 → 1.0, +1.0 → 1.15
        return 1.0 + score * 0.15

    # ── MUSTERI CARPANI ───────────────────────────────────────
    def _customer_multiplier(self, segment_id=None, session_info=None):
        """
        Musteri segmenti + davranis analizi.
        Is yolcusu = fiyat artir, ogrenci = fiyat dusur.
        """
        mult = 1.0

        # Segment bazli
        if segment_id and segment_id in self.segments:
            seg = self.segments[segment_id]
            wtp = seg.get("wtp_multiplier", {})
            wtp_avg = (wtp.get("min", 1.0) + wtp.get("max", 1.0)) / 2
            # WTP'yi carpana cevir: ogrenci (0.65) → 0.88, is (2.15) → 1.30
            mult = 0.85 + (wtp_avg - 0.5) * 0.2
            mult = max(0.80, min(mult, 1.40))

        # Davranis bazli (booking.html'den gelen session verileri)
        if session_info:
            # Geri donen musteri = ilgili = fiyat artir
            if session_info.get("return_visit"):
                mult *= 1.05

            # Uzun sure sayfada = tereddut = korkutma
            time_on_page = session_info.get("time_on_page", 0)
            if time_on_page > 180:
                mult *= 0.97  # uzun bekleme, biraz indir
            elif time_on_page > 60:
                mult *= 1.02  # ilgili ama kararsiz

            # Cok tarih aradi = fiyata duyarli
            search_count = session_info.get("search_count", 0)
            if search_count > 5:
                mult *= 0.98  # fiyat avciisi
            elif search_count <= 1:
                mult *= 1.03  # kararlı, tek arama

            # Odeme sayfasina gidip dondu = kesin alacak
            if session_info.get("abandoned_cart"):
                mult *= 1.08

            # Login durumu (FF uye)
            ff_tier = session_info.get("ff_tier", "none")
            if ff_tier in ("gold", "elite"):
                mult *= 1.06
            elif ff_tier == "silver":
                mult *= 1.02

            # Mobil vs desktop
            if session_info.get("device") == "mobile":
                mult *= 1.02  # mobil = acil

            # Yolcu sayisi
            pax_count = session_info.get("pax_count", 1)
            if pax_count >= 4:
                mult *= 0.97  # aile, fiyat hassas
            elif pax_count == 1:
                mult *= 1.01  # tek kisi, is yolcusu olabilir

        return round(max(0.90, min(mult, 1.20)), 4)

    # ── FARE CLASS YONETIMI ───────────────────────────────────
    def _get_open_fares(self, dtd, lf, demand_pressure=1.0, inventory=None):
        """
        Hangi fare class'lar acik?
        DTD + LF + kota + talep basincina gore dinamik.
        """
        # DTD bazli temel kurallar
        base_open = ["Y"]  # Y her zaman acik
        for rule in DTD_RULES:
            if rule["dtd_min"] <= dtd <= rule["dtd_max"]:
                base_open = rule["open"]
                break

        # LF filtresi: doluluk arttikca ucuz class kapanir
        filtered = []
        for fc_id in base_open:
            fc = FARE_CLASSES[fc_id]
            if lf < fc["open_until_lf"] or fc_id == "Y":
                filtered.append(fc_id)

        if not filtered:
            filtered = ["Y"]

        # Kota kontrolu: her fare class'in kapasiteye gore satilabilecek limiti var
        if inventory:
            capacity = inventory.get("capacity", 300)
            fc_sold = inventory.get("fare_class_sold", {})
            quota_filtered = []
            for fc_id in filtered:
                fc = FARE_CLASSES[fc_id]
                quota = int(capacity * fc["quota_pct"])
                sold_in_class = fc_sold.get(fc_id, 0)
                if sold_in_class < quota or fc_id == "Y":
                    quota_filtered.append(fc_id)
            filtered = quota_filtered if quota_filtered else ["Y"]

        # Talep basinci ayari
        if demand_pressure > 1.3 and len(filtered) > 1:
            filtered = filtered[1:]

        if demand_pressure < 0.6 and dtd > 7:
            cheapest = filtered[0]
            cheaper_map = {"Y": "M", "M": "K", "K": "V"}
            cheaper = cheaper_map.get(cheapest)
            if cheaper and cheaper not in filtered:
                filtered.insert(0, cheaper)

        return filtered

    # ── BASELINE FIYAT (eski formul) ──────────────────────────
    def compute_baseline_price(self, route, cabin, dtd, dep_date):
        """
        Sentetik veri ureticisindeki eski fiyat formulu.
        Karsilastirma icin kullanilir.
        """
        dist_km = self._get_distance(route)
        base = BASE_PRICE_FORMULAS.get(cabin, BASE_PRICE_FORMULAS["economy"])(dist_km)

        # DTD faktoru (eski)
        if dtd <= 3:
            dtd_f = 2.0
        elif dtd <= 7:
            dtd_f = 1.7
        elif dtd <= 14:
            dtd_f = 1.4
        elif dtd <= 30:
            dtd_f = 1.15
        elif dtd <= 60:
            dtd_f = 0.95
        elif dtd <= 90:
            dtd_f = 0.85
        else:
            dtd_f = 0.75

        # Sezon (eski — basit)
        if dep_date:
            if isinstance(dep_date, str):
                dep_date = datetime.strptime(dep_date, "%Y-%m-%d").date()
            elif isinstance(dep_date, datetime):
                dep_date = dep_date.date()
            season_f = SEASON_FACTORS.get(dep_date.month, 1.0)
        else:
            season_f = 1.0

        # Bolge
        arr = route.split("-")[1] if "-" in route else ""
        region = self._get_region(arr)
        region_f = REGION_FACTORS.get(region, 1.0)

        return base * dtd_f * season_f * region_f

    # ── SATIN ALMA OLASILIGI ──────────────────────────────────
    def purchase_probability(self, segment_id, offered_price, base_price):
        """
        Musteri bu fiyatta alir mi? (Botlar + UI gosterimi icin)
        Segment WTP araligi icerisinde sigmoid benzeri olasilik.
        """
        if segment_id not in self.segments:
            return 0.5

        seg = self.segments[segment_id]
        wtp = seg.get("wtp_multiplier", {"min": 0.8, "max": 1.2})
        wtp_max_price = wtp["max"] * base_price
        wtp_min_price = wtp["min"] * base_price

        if offered_price > wtp_max_price:
            return 0.0  # cok pahali, almaz
        elif offered_price <= wtp_min_price:
            return 0.95  # cok ucuz, kesin alir

        # Aradaki bolge: sigmoid benzeri dusus
        ratio = (offered_price - wtp_min_price) / (wtp_max_price - wtp_min_price)
        return max(0.0, 0.95 * (1 - ratio ** 0.7))

    # ── YARDIMCI FONKSIYONLAR ─────────────────────────────────
    def _compute_base_price(self, cabin, dist_km, route):
        """Baz fiyat hesapla — rota bazli faktor (veriden turetilmis)."""
        formula = BASE_PRICE_FORMULAS.get(cabin, BASE_PRICE_FORMULAS["economy"])
        base = formula(dist_km)
        # Rota bazli faktor: bolge faktoru yerine veriden turetilmis rota spesifik faktor
        route_key = route.replace("-", "_")
        route_factor = self.route_price_factors.get(route_key, 1.0)
        return base * route_factor

    def _get_distance(self, route):
        """Rota mesafesini getir."""
        # IST-LHR formatindan key olustur
        key = route.replace("-", "_")
        dist = self.route_distances.get(key)
        if dist:
            return dist
        # Ters yonu dene
        parts = route.split("-")
        if len(parts) == 2:
            rev_key = f"{parts[1]}_{parts[0]}"
            dist = self.route_distances.get(rev_key)
            if dist:
                return dist
        return 3000  # default

    def _get_region(self, airport_code):
        """Havalimani kodundan bolge bul."""
        # Basit mapping
        AIRPORT_REGIONS = {
            "LHR": "Europe", "LGW": "Europe", "STN": "Europe", "MAN": "Europe",
            "CDG": "Europe", "ORY": "Europe", "NCE": "Europe",
            "FRA": "Europe", "MUC": "Europe",
            "FCO": "Europe", "MXP": "Europe",
            "MAD": "Europe", "BCN": "Europe",
            "DXB": "Middle East", "DWC": "Middle East", "AUH": "Middle East",
            "RUH": "Middle East", "JED": "Middle East",
            "DOH": "Middle East", "BAH": "Middle East", "KWI": "Middle East",
            "AMM": "Middle East", "TLV": "Middle East", "BEY": "Middle East",
            "CAI": "Africa", "HRG": "Africa", "CMN": "Africa", "RAK": "Africa",
            "JNB": "Africa", "CPT": "Africa", "NBO": "Africa", "MBA": "Africa",
            "LOS": "Africa", "ABV": "Africa",
            "NRT": "Asia", "HND": "Asia", "KIX": "Asia",
            "ICN": "Asia", "PEK": "Asia", "PVG": "Asia",
            "SIN": "Asia", "BKK": "Asia", "HKT": "Asia",
            "DEL": "Asia", "BOM": "Asia",
            "JFK": "Americas", "LAX": "Americas", "MIA": "Americas", "ORD": "Americas",
            "YYZ": "Americas", "YVR": "Americas",
            "MEX": "Americas", "GRU": "Americas", "EZE": "Americas", "GIG": "Americas",
        }
        return AIRPORT_REGIONS.get(airport_code, "Europe")
