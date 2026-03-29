"""
Network Optimizer — O&D Revenue Management

EMSR-b fare class optimizasyonu + bid price + fare proration.
Connecting vs lokal yolcu karari, displacement tracking.
"""
import math
import random
import numpy as np
from scipy.stats import norm


FARE_MULTIPLIERS = {"V": 0.50, "K": 0.75, "M": 1.00, "Y": 1.50}
FARE_ORDER = ["V", "K", "M", "Y"]


class NetworkOptimizer:

    def __init__(self, route_distances, segments, connecting_pcts=None):
        self.route_distances = route_distances
        self.segments = segments
        self.connecting_pcts = connecting_pcts or {}

        # Inbound rota agirliklari (connecting origin secimi icin)
        total_dist = sum(route_distances.values()) or 1
        self.origin_weights = {}
        for rk, dist in route_distances.items():
            self.origin_weights[rk] = dist / total_dist
        self._origin_keys = list(self.origin_weights.keys())
        self._origin_probs = [self.origin_weights[k] for k in self._origin_keys]

        self.class_demand = self._estimate_class_demand()

    def _estimate_class_demand(self):
        """Segment WTP'den fare class bazli talep orani turet."""
        class_shares = {"V": 0.0, "K": 0.0, "M": 0.0, "Y": 0.0}
        seg_map = {
            "A": {"Y": 0.50, "M": 0.50},
            "B": {"M": 0.40, "K": 0.60},
            "C": {"M": 0.40, "K": 0.60},
            "D": {"K": 0.40, "V": 0.60},
            "E": {"V": 1.00},
            "F": {"Y": 1.00},
        }
        for seg_id, seg in self.segments.items():
            share = seg.get("base_share_pct", 10) / 100
            mapping = seg_map.get(seg_id, {"M": 1.0})
            for fc, fc_pct in mapping.items():
                class_shares[fc] += share * fc_pct
        return class_shares

    # ══════════════════════════════════════════════════════
    # EMSR-b
    # ══════════════════════════════════════════════════════

    def compute_protection_levels(self, base_price, capacity, current_sold=0,
                                    expected_total_demand=None):
        """
        EMSR-b ile fare class koruma seviyeleri.
        expected_total_demand: TFT/Pickup'tan gelen toplam talep tahmini (varsa).
        Returns: {fc: {"quota": int, "protected": int}}
        """
        remaining = max(capacity - current_sold, 0)
        if remaining == 0:
            return {fc: {"quota": 0, "protected": capacity} for fc in FARE_ORDER}

        prices = {fc: base_price * FARE_MULTIPLIERS[fc] for fc in FARE_ORDER}

        # Talep tahmini: forecast varsa onu kullan, yoksa kapasiteyi baz al
        demand_base = expected_total_demand if expected_total_demand else capacity
        demands = {}
        for fc in FARE_ORDER:
            share = self.class_demand.get(fc, 0.1)
            mean_d = demand_base * share
            std_d = max(mean_d * 0.4, 1.0)
            demands[fc] = {"mean": mean_d, "std": std_d}

        protections = {}
        for i in range(len(FARE_ORDER) - 1, 0, -1):
            high = FARE_ORDER[i]
            low = FARE_ORDER[i - 1]
            ratio = max(0.01, min(prices[low] / prices[high], 0.99))
            d = demands[high]
            z = norm.ppf(1 - ratio)
            prot = d["mean"] + d["std"] * z
            prot = max(0, min(prot, remaining))
            protections[high] = int(round(prot))

        protections["V"] = 0

        result = {}
        cum = 0
        for fc in reversed(FARE_ORDER):
            prot = protections.get(fc, 0)
            cum += prot
            result[fc] = {
                "protected": min(int(cum), remaining),
                "quota": max(remaining - int(cum), 0),
            }

        total_prot = sum(protections.values())
        result["V"]["quota"] = max(remaining - int(total_prot), 0)
        result["V"]["protected"] = 0

        return result

    def get_bid_price(self, base_price, capacity, sold, open_fares, current_prices=None):
        """Koltugun minimum kabul edilebilir fiyati.
        current_prices varsa gercek fiyatlari kullan (tum carpanlar dahil).
        Yoksa base x multiplier fallback."""
        if current_prices and open_fares:
            lowest = open_fares[0]
            return current_prices.get(lowest, base_price * FARE_MULTIPLIERS.get(lowest, 1.0))
        if not open_fares:
            return base_price * FARE_MULTIPLIERS["Y"]
        lowest = open_fares[0]
        return base_price * FARE_MULTIPLIERS[lowest]

    # ══════════════════════════════════════════════════════
    # FARE PRORATION
    # ══════════════════════════════════════════════════════

    def prorate_fare(self, origin_route, dest_route, cabin="economy"):
        """
        Baglantili yolcu itinerary fiyati + bacak katkisi.
        Returns: {"origin", "total_fare", "leg_contribution", ...}
        """
        origin_dist = self._get_distance(origin_route.replace("-", "_"))
        dest_dist = self._get_distance(dest_route.replace("-", "_"))
        total_dist = origin_dist + dest_dist
        if total_dist == 0:
            return None

        if cabin == "business":
            o_base = max(origin_dist * 0.35, 800)
            d_base = max(dest_dist * 0.35, 800)
        else:
            o_base = max(origin_dist * 0.08, 150)
            d_base = max(dest_dist * 0.08, 150)

        total_fare = (o_base + d_base) * 0.85

        parts = origin_route.replace("-", "_").split("_")
        origin_apt = parts[0] if parts[0] != "IST" else (parts[1] if len(parts) > 1 else "UNK")

        return {
            "origin": origin_apt,
            "total_distance": total_dist,
            "total_fare": round(total_fare, 2),
            "leg_contribution": round(total_fare * dest_dist / total_dist, 2),
            "origin_contribution": round(total_fare * origin_dist / total_dist, 2),
        }

    def pick_random_origin(self):
        """Agirikli rastgele inbound origin sec."""
        if not self._origin_keys:
            return None
        idx = random.choices(range(len(self._origin_keys)), weights=self._origin_probs, k=1)[0]
        return self._origin_keys[idx]

    def get_connecting_pct(self, route):
        """Rota icin connecting yolcu orani. Minimum %15 (hub etkisi)."""
        key = route.replace("-", "_")
        raw = self.connecting_pcts.get(key, 0.15)
        # IST hub'inda connecting minimum %10
        return max(raw, 0.10)

    def evaluate_od(self, is_connecting, fare, leg_contribution, bid_price):
        """Kabul/red karari. Returns: (kabul, reason)"""
        if is_connecting:
            if leg_contribution >= bid_price:
                return True, "connecting_accepted"
            return False, "connecting_below_bid"
        else:
            if fare >= bid_price:
                return True, "local_accepted"
            return False, "local_below_bid"

    def _get_distance(self, route_key):
        dist = self.route_distances.get(route_key)
        if dist:
            return dist
        parts = route_key.split("_")
        if len(parts) == 2:
            dist = self.route_distances.get(f"{parts[1]}_{parts[0]}")
            if dist:
                return dist
        return 3000
