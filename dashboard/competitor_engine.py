"""
Competitor Airline Engine — Rakip havayolu simulasyonu.

Pegasus (PC) ve Emirates (EK) icin:
- Sabit fiyatlandirma (DTD-bazli merdiven)
- Golge envanter takibi (kapasite, doluluk, gelir)
- Utility-bazli yolcu secim modeli (discrete choice)
"""
import hashlib
import json
import random
from collections import defaultdict
from datetime import date


class CompetitorAirline:
    """Tek bir rakip havayolunu temsil eder."""

    def __init__(self, config):
        self.code = config["code"]
        self.name = config["name"]
        self.airline_type = config["type"]
        self.cabins = set(config["cabins"])
        self.capacity = config["capacity"]
        self.price_range = config["price_range"]
        self.brand_premium = config.get("brand_premium", 0.0)
        self.no_show_rate = config.get("no_show_rate", 0.05)
        self.route_presence = set(config["route_presence"])
        self.segment_preference = config.get("segment_preference", {})
        raw_ladder = config.get("price_ladder", {"0": 1.0})
        self.price_ladder = sorted(
            [(int(k), v) for k, v in raw_ladder.items()],
            key=lambda x: x[0], reverse=True,
        )
        self.inventory = {}
        self._price_cache = {}
        # Rota bazli sabit oran cache (seed-driven, tum sim boyunca sabit)
        self._route_ratio_cache = {}

    def is_present(self, route, cabin):
        arr = route.split("-")[1] if "-" in route else ""
        return arr in self.route_presence and cabin in self.cabins

    def _dtd_mult(self, dtd):
        for threshold, mult in self.price_ladder:
            if dtd >= threshold:
                return mult
        return self.price_ladder[-1][1] if self.price_ladder else 1.0

    def _get_route_ratio(self, route, cabin):
        """Rota bazli sabit ucuzluk/pahalilik orani — seed-driven, degismez."""
        cache_key = f"{route}_{cabin}"
        if cache_key not in self._route_ratio_cache:
            seed_str = f"{self.code}_{route}_{cabin}_ratio"
            seed_val = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed_val)
            pr = self.price_range.get(cabin, [1.0, 1.0])
            self._route_ratio_cache[cache_key] = rng.uniform(pr[0], pr[1])
        return self._route_ratio_cache[cache_key]

    def compute_price(self, route, cabin, dtd, our_base_price):
        """Deterministik sabit fiyat: rota bazli sabit oran x DTD merdiveni."""
        if cabin not in self.cabins:
            return None
        ratio = self._get_route_ratio(route, cabin)
        dtd_m = self._dtd_mult(dtd)
        return round(our_base_price * ratio * dtd_m, 2)

    def init_flight(self, inv_key, cabin):
        cap = self.capacity.get(cabin, 0)
        if cap == 0:
            return
        self.inventory[inv_key] = {
            "capacity": cap,
            "sold": 0,
            "load_factor": 0.0,
            "revenue": 0.0,
            "current_price": 0.0,
            "bookings": [],
        }

    def record_sale(self, inv_key, segment_id, price, dtd, timestamp):
        inv = self.inventory.get(inv_key)
        if not inv:
            return False
        if inv["sold"] >= inv["capacity"]:
            return False
        inv["sold"] += 1
        inv["load_factor"] = inv["sold"] / inv["capacity"]
        inv["revenue"] += price
        inv["bookings"].append({
            "segment": segment_id,
            "price": round(price, 2),
            "dtd": dtd,
            "timestamp": timestamp,
        })
        return True

    def is_sold_out(self, inv_key):
        inv = self.inventory.get(inv_key)
        if not inv:
            return True
        return inv["sold"] >= inv["capacity"]


class CompetitorManager:
    """Tum rakipleri yonetir, yolcu secim modelini calistirir."""

    def __init__(self, competitors, segments):
        self.competitors = {c.code: c for c in competitors}
        self.segments = segments or {}

    def initialize_inventory(self, our_inventory):
        """Bizim ucus listesinden rakip envanterlerini olustur."""
        for comp in self.competitors.values():
            comp.inventory.clear()
            comp._price_cache.clear()
        for key, inv in our_inventory.items():
            route = inv["route"]
            cabin = inv.get("cabin", "economy")
            for comp in self.competitors.values():
                if comp.is_present(route, cabin):
                    comp.init_flight(key, cabin)

    def update_daily_prices(self, sim_day, our_inventory, pricing_engine):
        """Her rakibin gunluk fiyatini hesapla ve cache'le."""
        for comp in self.competitors.values():
            comp._price_cache.clear()
            for key in comp.inventory:
                inv = our_inventory.get(key)
                if not inv:
                    continue
                route = inv["route"]
                cabin = inv.get("cabin", "economy")
                dep_date = inv["dep_date"]
                dep_date_obj = dep_date if isinstance(dep_date, date) else date.fromisoformat(str(dep_date))
                dtd = (dep_date_obj - sim_day).days
                if dtd < 0:
                    continue
                our_base = pricing_engine._compute_base_price(
                    cabin,
                    inv.get("distance_km", 3000),
                    route,
                )
                price = comp.compute_price(route, cabin, dtd, our_base)
                if price:
                    comp._price_cache[key] = price
                    comp.inventory[key]["current_price"] = price

    def choose_airline(self, route, cabin, dep_date, dtd, segment_id,
                       our_price, our_base_price, personal_wtp, inv_key):
        """
        Utility-bazli discrete choice model.
        U = -alpha * (price / base) + beta * brand_pref + epsilon
        En yuksek U kazanir.
        """
        seg_data = self.segments.get(segment_id, {})
        elasticity = abs(seg_data.get("price_elasticity", 1.0))

        options = [("us", our_price, 0.0)]
        for code, comp in self.competitors.items():
            if not comp.is_present(route, cabin):
                continue
            if comp.is_sold_out(inv_key):
                continue
            comp_price = comp._price_cache.get(inv_key)
            if not comp_price:
                continue
            brand_val = comp.segment_preference.get(segment_id, 0.1)
            options.append((code, comp_price, brand_val))

        if len(options) == 1:
            return ("us", None)

        best_label = "us"
        best_util = float("-inf")
        for label, price, brand in options:
            price_util = -elasticity * (price / our_base_price)
            brand_util = brand * 2.0
            noise = random.gauss(0, 0.15)
            u = price_util + brand_util + noise
            if label == "us":
                u += 0.05
            if u > best_util:
                best_util = u
                best_label = label

        if best_label == "us":
            return ("us", None)
        return ("competitor", best_label)

    def get_competitor_prices(self, inv_key):
        prices = {}
        for code, comp in self.competitors.items():
            p = comp._price_cache.get(inv_key)
            if p:
                prices[code] = p
        return prices

    def get_summary(self):
        """Pazar payi, rakip LF, gelir ozeti."""
        result = {"competitors": {}}
        for code, comp in self.competitors.items():
            total_sold = sum(inv["sold"] for inv in comp.inventory.values())
            total_rev = sum(inv["revenue"] for inv in comp.inventory.values())
            total_cap = sum(inv["capacity"] for inv in comp.inventory.values())
            avg_lf = (total_sold / total_cap * 100) if total_cap > 0 else 0
            # Ortalama guncel fiyat (cache'deki fiyatlardan)
            prices = [p for p in comp._price_cache.values() if p > 0]
            avg_price = round(sum(prices) / len(prices), 2) if prices else 0
            result["competitors"][code] = {
                "name": comp.name,
                "type": comp.airline_type,
                "total_sold": total_sold,
                "total_revenue": round(total_rev, 2),
                "total_capacity": total_cap,
                "avg_lf": round(avg_lf, 1),
                "avg_price": avg_price,
                "routes_served": len(comp.inventory),
            }
        return result

    def get_flight_competition(self, inv_key, our_inv):
        """Tek ucus icin 3 havayolunun durumu."""
        result = {
            "us": {
                "name": "THY",
                "sold": our_inv["sold"],
                "capacity": our_inv["capacity"],
                "lf": round(our_inv.get("load_factor", 0) * 100, 1),
                "price": our_inv.get("current_prices", {}).get(
                    (our_inv.get("fare_classes_open") or ["Y"])[0], 0
                ),
                "bookings": our_inv.get("bookings", []),
            },
        }
        for code, comp in self.competitors.items():
            cinv = comp.inventory.get(inv_key)
            if cinv:
                result[code] = {
                    "name": comp.name,
                    "sold": cinv["sold"],
                    "capacity": cinv["capacity"],
                    "lf": round(cinv["load_factor"] * 100, 1),
                    "price": cinv["current_price"],
                    "bookings": cinv["bookings"],
                }
        return result
