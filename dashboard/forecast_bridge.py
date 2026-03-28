"""
ForecastBridge — ML modelleri ile simulasyon arasindaki kopru.

Uc model, uc rol:
  Two-Stage XGBoost → gunluk bot sayisi (gaz pedali)
  TFT → tavan/taban regulatoru (hiz limiti)
  XGBoost Pickup → pricing supply multiplier (direksiyon)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from datetime import datetime, date


class ForecastBridge:

    def __init__(self, tft_predictions_df, twostage_clf, twostage_reg, twostage_features,
                 pickup_model, pickup_features, route_meta=None):
        """
        tft_predictions_df: DataFrame (entity_id, dep_date, predicted) veya None
        twostage_clf: XGBClassifier (.pkl)
        twostage_reg: XGBRegressor (.pkl)
        twostage_features: list of 31 feature names
        pickup_model: xgb.Booster (.json)
        pickup_features: list of 49 feature names
        route_meta: dict {route_key_cabin: {distance_km, region, capacity, ...}}
        """
        self.twostage_clf = twostage_clf
        self.twostage_reg = twostage_reg
        self.twostage_features = twostage_features
        self.pickup_model = pickup_model
        self.pickup_features = pickup_features
        self.route_meta = route_meta or {}

        # TFT cache: (entity_id, date_str) → predicted
        self._tft_cache = {}
        if tft_predictions_df is not None:
            for _, row in tft_predictions_df.iterrows():
                ds = row["dep_date"]
                if isinstance(ds, pd.Timestamp):
                    ds = ds.strftime("%Y-%m-%d")
                elif isinstance(ds, (date, datetime)):
                    ds = ds.isoformat()[:10]
                self._tft_cache[(row["entity_id"], ds)] = float(row["predicted"])

        # Gunluk batch predict cache — her gun sifirlanir
        self._daily_cache = {}
        self._cache_day = None

    # ══════════════════════════════════════════════════════
    # TFT — tavan/taban regulatoru
    # ══════════════════════════════════════════════════════

    def get_tft_total(self, route, cabin, dep_date):
        """TFT'nin rota-gun toplam yolcu tahmini. None = TFT yok."""
        entity_id = f"{route.replace('-', '_')}_{cabin}"
        if isinstance(dep_date, (date, datetime)):
            dep_date = dep_date.isoformat()[:10]
        return self._tft_cache.get((entity_id, dep_date))

    def get_tft_band(self, route, cabin, dep_date, dtd):
        """
        TFT tavan/taban: bu DTD'de kumulatif ne kadar satilmis olmali?
        Returns: (daily_floor, daily_ceiling, cumulative_floor, cumulative_ceiling) veya None
        """
        tft_total = self.get_tft_total(route, cabin, dep_date)
        if tft_total is None:
            return None

        # S-curve: kumulatif beklenen oran
        cum_fraction = 1.0 - (dtd / 180.0) ** 1.5 if dtd <= 180 else 0.0
        cum_fraction = max(0.0, min(1.0, cum_fraction))

        # Gunluk pay (dunku - bugunku fark)
        cum_tomorrow = 1.0 - ((dtd + 1) / 180.0) ** 1.5 if dtd < 180 else 0.0
        cum_tomorrow = max(0.0, min(1.0, cum_tomorrow))
        daily_fraction = cum_fraction - cum_tomorrow
        daily_fraction = max(0.001, daily_fraction)  # sifir olmasin

        tft_daily = tft_total * daily_fraction
        daily_floor = tft_daily * 0.3
        daily_ceiling = tft_daily * 2.0

        cumulative_expected = tft_total * cum_fraction
        cumulative_floor = cumulative_expected * 0.5
        cumulative_ceiling = cumulative_expected * 1.3

        return {
            "daily_floor": daily_floor,
            "daily_ceiling": daily_ceiling,
            "cumulative_floor": cumulative_floor,
            "cumulative_ceiling": cumulative_ceiling,
            "tft_total": tft_total,
            "cum_fraction": cum_fraction,
        }

    # ══════════════════════════════════════════════════════
    # BATCH PREDICT — gunde 1 kez tum aktif ucuslar icin
    # ══════════════════════════════════════════════════════

    def predict_daily_batch(self, inventory, sim_day):
        """
        Tum aktif ucuslar icin Two-Stage + Pickup toplu predict.
        Gunde 1 kez cagirilir, sonuc cache'lenir.
        """
        # Ayni gun icin tekrar cagrilirsa cache'den don
        day_str = sim_day.isoformat() if isinstance(sim_day, (date, datetime)) else str(sim_day)
        if self._cache_day == day_str:
            return self._daily_cache
        self._cache_day = day_str
        self._daily_cache = {}

        active_keys = []
        ts_features = []  # Two-Stage: 31 feature
        pk_features = []  # Pickup: 49 feature

        for key, inv in inventory.items():
            dep = inv["dep_date"]
            if isinstance(dep, datetime):
                dep = dep.date()
            dtd = (dep - sim_day).days if isinstance(sim_day, date) else 0
            if dtd < 0 or dtd > 180:
                continue
            if inv["sold"] >= inv["capacity"]:
                continue

            # Feature dict olustur
            feat = self._build_features(inv, dtd, sim_day)
            active_keys.append(key)

            # Two-Stage features (31)
            ts_row = [feat.get(f, 0.0) for f in self.twostage_features]
            ts_features.append(ts_row)

            # Pickup features (49)
            pk_row = [feat.get(f, 0.0) for f in self.pickup_features]
            pk_features.append(pk_row)

        if not active_keys:
            return self._daily_cache

        # ── Two-Stage batch predict ──
        X_ts = np.array(ts_features, dtype=np.float32)
        p_sale = self.twostage_clf.predict_proba(X_ts)[:, 1]
        e_pax = np.clip(self.twostage_reg.predict(X_ts), 0, None)
        daily_demand = p_sale * e_pax

        # ── Pickup batch predict ──
        X_pk = np.array(pk_features, dtype=np.float32)
        dmat = xgb.DMatrix(X_pk, feature_names=self.pickup_features)
        remaining = np.clip(self.pickup_model.predict(dmat), 0, None)

        # Cache
        for i, key in enumerate(active_keys):
            self._daily_cache[key] = {
                "daily_demand": float(daily_demand[i]),
                "p_sale": float(p_sale[i]),
                "e_pax": float(e_pax[i]),
                "predicted_remaining": float(remaining[i]),
            }

        return self._daily_cache

    # ══════════════════════════════════════════════════════
    # FEATURE MAPPING — simulasyon state → model features
    # ══════════════════════════════════════════════════════

    def _build_features(self, inv, dtd, sim_day):
        """Simulasyon inventory'den model feature dict olustur."""
        dep_date = inv["dep_date"]
        if isinstance(dep_date, datetime):
            dep_date = dep_date.date()
        cabin = inv.get("cabin", "economy")
        region = inv.get("region", "Europe")
        capacity = inv.get("capacity", 300)
        sold = inv.get("sold", 0)

        # pax_last_7d: booking gecmisinden turet
        bookings = inv.get("bookings", [])
        pax_last_7d = 0
        if bookings and isinstance(sim_day, date):
            for b in bookings:
                b_dtd = b.get("dtd", 999)
                if dtd <= b_dtd <= dtd + 7:
                    pax_last_7d += 1

        # Route metadata'dan default'lar
        route_key = inv.get("route", "IST-LHR").replace("-", "_")
        meta_key = f"{route_key}_{cabin}"
        meta = self.route_meta.get(meta_key, {})
        flight_time_min = meta.get("flight_time_min", 300.0) if meta else 300.0
        distance_km = float(inv.get("distance_km", 3000))

        # DTD bucket (0-6)
        dtd_bucket = min(int(dtd // 30), 6)

        feat = {
            # Ortak features
            "dtd": float(dtd),
            "pax_sold_cum": float(sold),
            "pax_last_7d": float(pax_last_7d),
            "capacity": float(capacity),
            "remaining_seats": float(max(capacity - sold, 0)),
            "load_factor": sold / capacity if capacity > 0 else 0.0,
            "distance_km": distance_km,
            "flight_time_min": float(flight_time_min),
            "dep_year": float(dep_date.year),
            "dep_month": float(dep_date.month),
            "dep_dow": float(dep_date.weekday()),
            "dep_hour": 12.0,  # default — simulasyonda saat yok
            "ff_gold_pct": 0.15,  # default
            "ff_elite_pct": 0.05,  # default

            # Cabin one-hot
            "cabin_class_business": 1.0 if cabin == "business" else 0.0,
            "cabin_class_economy": 1.0 if cabin == "economy" else 0.0,
            "cabin_class_nan": 0.0,

            # Region one-hot
            "region_Africa": 1.0 if region == "Africa" else 0.0,
            "region_Americas": 1.0 if region == "Americas" else 0.0,
            "region_Asia": 1.0 if region == "Asia" else 0.0,
            "region_Europe": 1.0 if region == "Europe" else 0.0,
            "region_Middle East": 1.0 if region == "Middle East" else 0.0,
            "region_nan": 0.0,

            # DTD bucket one-hot
            "dtd_bucket_0.0": 1.0 if dtd_bucket == 0 else 0.0,
            "dtd_bucket_1.0": 1.0 if dtd_bucket == 1 else 0.0,
            "dtd_bucket_2.0": 1.0 if dtd_bucket == 2 else 0.0,
            "dtd_bucket_3.0": 1.0 if dtd_bucket == 3 else 0.0,
            "dtd_bucket_4.0": 1.0 if dtd_bucket == 4 else 0.0,
            "dtd_bucket_5.0": 1.0 if dtd_bucket == 5 else 0.0,
            "dtd_bucket_6.0": 1.0 if dtd_bucket == 6 else 0.0,
            "dtd_bucket_nan": 0.0,

            # Pickup-only features (default'lar)
            "is_weekend": 1.0 if dep_date.weekday() >= 5 else 0.0,
            "pax_sold_today": float(pax_last_7d / 7.0) if pax_last_7d > 0 else 0.0,
            "ticket_rev_cum": float(inv.get("revenue_dynamic", 0)) * 0.85,
            "anc_rev_cum": float(inv.get("revenue_dynamic", 0)) * 0.15,
        }

        return feat
