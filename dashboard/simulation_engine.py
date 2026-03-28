"""
Simulation Engine — Canli ucus simulasyonu.

- Secilen tarih araligindaki ucuslari bos olarak baslatir
- Bot yolcular segment dagilimina gore gelir
- Pricing engine fiyat belirler, bot WTP'sine gore alir/almaz
- Zaman hizlandirmasi (1dk = 1gun vs.)
- Duraklat / devam / tarihe atla / manuel mudahale
"""
import json
import math
import os
import random
import threading
import time
from datetime import datetime, timedelta, date
from collections import defaultdict


class SimClock:
    """Simulasyon saati — hizlandirma destekli."""

    def __init__(self):
        self.mode = "paused"          # paused | running
        self.speed = 1440.0           # 1440 = 1 dakika = 1 gun
        self.sim_datetime = None      # suanki simulasyon zamani
        self.wall_start = None        # gercek baslangic zamani
        self.sim_start = None         # simulasyon baslangic zamani
        self._pause_sim_dt = None     # duraklama anindaki sim zamani

    def configure(self, start_date, speed=1440):
        """Simulasyonu baslat."""
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        elif isinstance(start_date, date) and not isinstance(start_date, datetime):
            start_date = datetime.combine(start_date, datetime.min.time())

        self.sim_start = start_date
        self.sim_datetime = start_date
        self.speed = speed
        self.mode = "paused"

    def start(self):
        self.wall_start = time.time()
        self._pause_sim_dt = None
        self.mode = "running"

    def pause(self):
        if self.mode == "running":
            self._pause_sim_dt = self.now()
            self.mode = "paused"

    def resume(self):
        if self.mode == "paused" and self._pause_sim_dt:
            self.sim_start = self._pause_sim_dt
            self.wall_start = time.time()
            self._pause_sim_dt = None
            self.mode = "running"

    def set_speed(self, speed):
        # Hiz degistirirken mevcut zamani koru
        current = self.now()
        self.speed = speed
        self.sim_start = current
        self.wall_start = time.time()

    def jump_to(self, target_date):
        """Belirli bir tarihe atla."""
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, "%Y-%m-%d")
        elif isinstance(target_date, date) and not isinstance(target_date, datetime):
            target_date = datetime.combine(target_date, datetime.min.time())
        self.sim_start = target_date
        self.sim_datetime = target_date
        self.wall_start = time.time()
        self._pause_sim_dt = target_date

    def now(self):
        """Suanki simulasyon zamani."""
        if self.mode == "paused":
            return self._pause_sim_dt or self.sim_datetime or datetime(2026, 7, 1)
        if self.wall_start is None:
            return self.sim_datetime or datetime(2026, 7, 1)
        elapsed_real = time.time() - self.wall_start
        sim_elapsed = elapsed_real * self.speed
        return self.sim_start + timedelta(seconds=sim_elapsed)

    def today(self):
        return self.now().date()

    def dtd(self, dep_date):
        """Kalkisa kalan gun."""
        if isinstance(dep_date, str):
            dep_date = datetime.strptime(dep_date, "%Y-%m-%d").date()
        elif isinstance(dep_date, datetime):
            dep_date = dep_date.date()
        return (dep_date - self.today()).days


class SimulationEngine:
    """Ana simulasyon motoru."""

    def __init__(self, pricing_engine, forecast_bridge=None):
        self.pricing = pricing_engine
        self.bridge = forecast_bridge  # ForecastBridge veya None (fallback rule-based)
        self.clock = SimClock()
        self.state = "idle"  # idle | ready | running | paused | completed
        self.inventory = {}  # flight_key -> envanter dict
        self.lock = threading.Lock()
        self.thread = None

        # Istatistikler
        self.stats = {
            "total_bots": 0,
            "total_sales": 0,
            "total_rejected": 0,
            "total_revenue_dynamic": 0.0,
            "total_revenue_baseline": 0.0,
        }

        # Event log (son 100 olay)
        self.event_log = []
        self._max_events = 200

        # Segment verileri (pricing engine'den)
        self.segments = pricing_engine.segments

        # DTD demand curves (sonra yuklenir)
        self.dtd_curves = {}

    def initialize(self, flights_data, date_range, speed=1440, dtd_curves=None, seed=42):
        """
        Simulasyonu hazirla.

        flights_data: list of dicts — her ucus icin:
            {flight_id, route, cabin, dep_date, capacity, distance_km, region}
        date_range: (start_str, end_str) — "2026-07-01", "2026-12-31"
        speed: hiz carpani
        dtd_curves: demand_functions_report'tan DTD talep egrileri
        seed: random seed — ayni parametrelerle ayni sonuc icin
        """
        # Onceki simulasyonu durdur
        if self.state in ("running", "paused"):
            self.state = "idle"
            time.sleep(0.2)  # thread'in durmasini bekle

        random.seed(seed)
        self.state = "initializing"
        self.inventory = {}
        self.stats = {k: 0 for k in self.stats}
        self.stats["total_revenue_dynamic"] = 0.0
        self.stats["total_revenue_baseline"] = 0.0
        self.event_log = []

        if dtd_curves:
            self.dtd_curves = dtd_curves

        start_date = datetime.strptime(date_range[0], "%Y-%m-%d").date()
        end_date = datetime.strptime(date_range[1], "%Y-%m-%d").date()

        for f in flights_data:
            dep = f["dep_date"]
            if isinstance(dep, str):
                dep = datetime.strptime(dep, "%Y-%m-%d").date()
            elif isinstance(dep, datetime):
                dep = dep.date()

            if dep < start_date or dep > end_date:
                continue

            key = f"{f['route']}_{dep.isoformat()}_{f['cabin']}"
            self.inventory[key] = {
                "flight_id": f.get("flight_id", key),
                "route": f["route"],
                "cabin": f["cabin"],
                "dep_date": dep,
                "capacity": f["capacity"],
                "distance_km": f.get("distance_km", 3000),
                "region": f.get("region", "Europe"),
                "sold": 0,
                "load_factor": 0.0,
                "fare_class_sold": {"V": 0, "K": 0, "M": 0, "Y": 0},
                "revenue_dynamic": 0.0,
                "revenue_baseline": 0.0,
                "bookings": [],
                "price_history": [],
                "fare_classes_open": [],
                "current_prices": {},
            }

        # Saat ayarla — booking window 180 gun oncesinden baslar
        sim_start = start_date - timedelta(days=180)
        self.clock.configure(sim_start, speed)

        self.state = "ready"
        print(f"[Sim] Initialized: {len(self.inventory)} flights, "
              f"{date_range[0]} to {date_range[1]}, speed={speed}x", flush=True)

    def start(self):
        """Simulasyonu baslat."""
        if self.state not in ("ready", "paused"):
            return
        self.clock.start() if self.state == "ready" else self.clock.resume()
        self.state = "running"

        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._run_loop, daemon=True, name="sim-loop")
            self.thread.start()
        print(f"[Sim] Running at {self.clock.speed}x speed", flush=True)

    def pause(self):
        if self.state == "running":
            self.clock.pause()
            self.state = "paused"
            print(f"[Sim] Paused at {self.clock.today()}", flush=True)

    def resume(self):
        if self.state == "paused":
            self.clock.resume()
            self.state = "running"
            print(f"[Sim] Resumed", flush=True)

    def set_speed(self, speed):
        self.clock.set_speed(speed)
        print(f"[Sim] Speed changed to {speed}x", flush=True)

    def jump_to(self, target_date):
        """Belirli tarihe atla — aradaki gunleri hizlica simule et."""
        was_running = self.state == "running"
        if was_running:
            self.pause()

        target = datetime.strptime(target_date, "%Y-%m-%d").date() if isinstance(target_date, str) else target_date
        current = self.clock.today()

        # Aradaki gunleri simule et
        while current < target:
            self._process_day(current)
            current += timedelta(days=1)

        self.clock.jump_to(target_date)
        if was_running:
            self.resume()
        print(f"[Sim] Jumped to {target_date}", flush=True)

    # ── MANUEL MUDAHALE ───────────────────────────────────────
    def override_fare_class(self, flight_key, fare_class, action):
        """Fare class'i elle ac/kapa."""
        with self.lock:
            inv = self.inventory.get(flight_key)
            if not inv:
                return False
            if action == "open" and fare_class not in inv["fare_classes_open"]:
                inv["fare_classes_open"].append(fare_class)
                inv["fare_classes_open"].sort(key=lambda x: ["V","K","M","Y"].index(x))
            elif action == "close" and fare_class in inv["fare_classes_open"]:
                inv["fare_classes_open"].remove(fare_class)
            self._log_event(flight_key, "manual_override", f"Fare {fare_class} {action}")
            return True

    def inject_bots(self, flight_key, segment_id, count):
        """Belirli bir ucusa belirli sayida bot gonder."""
        with self.lock:
            inv = self.inventory.get(flight_key)
            if not inv:
                return 0
            dtd = self.clock.dtd(inv["dep_date"])
            sales = 0
            for _ in range(count):
                if self._process_bot(flight_key, inv, segment_id, dtd):
                    sales += 1
            return sales

    def book_human(self, flight_key, fare_class, session_info=None):
        """Gercek kullanici bilet alimi (booking.html'den)."""
        with self.lock:
            inv = self.inventory.get(flight_key)
            if not inv:
                return {"error": "Flight not found"}
            if inv["sold"] >= inv["capacity"]:
                return {"error": "Sold out"}
            if fare_class not in inv.get("fare_classes_open", ["Y"]):
                return {"error": f"Fare class {fare_class} closed"}

            dtd = self.clock.dtd(inv["dep_date"])
            quote = self.pricing.compute_price(inv, dtd, session_info=session_info)
            price = quote["prices"].get(fare_class)
            if price is None:
                return {"error": "Invalid fare class"}

            # Satis yap
            inv["sold"] += 1
            inv["load_factor"] = inv["sold"] / inv["capacity"]
            if "fare_class_sold" in inv:
                inv["fare_class_sold"][fare_class] = inv["fare_class_sold"].get(fare_class, 0) + 1
            inv["revenue_dynamic"] += price

            from pricing_engine import FARE_CLASSES
            baseline_full = self.pricing.compute_baseline_price(
                inv["route"], inv["cabin"], dtd,
                inv["dep_date"].isoformat() if isinstance(inv["dep_date"], date) else inv["dep_date"]
            )
            fc_mult = FARE_CLASSES.get(fare_class, {}).get("multiplier", 1.0)
            baseline = baseline_full * fc_mult
            inv["revenue_baseline"] += baseline

            booking = {
                "timestamp": self.clock.now().isoformat(),
                "dtd": dtd,
                "segment": "HUMAN",
                "fare_class": fare_class,
                "price": round(price, 2),
                "baseline_price": round(baseline, 2),
                "is_bot": False,
                "session_info": session_info,
            }
            inv["bookings"].append(booking)

            self.stats["total_sales"] += 1
            self.stats["total_revenue_dynamic"] += price
            self.stats["total_revenue_baseline"] += baseline

            self._log_event(flight_key, "human_booking",
                f"Fare {fare_class} ${price:.0f} (baseline ${baseline:.0f})")

            return {
                "success": True,
                "price": round(price, 2),
                "fare_class": fare_class,
                "seats_remaining": inv["capacity"] - inv["sold"],
                "load_factor": round(inv["load_factor"], 4),
            }

    # ── ANA DONGU ─────────────────────────────────────────────
    def _run_loop(self):
        """Her gun sirayla isle. Hiz = gunler arasi bekleme."""
        # Baslangic: kalkis tarihinden 180 gun once
        dep_dates = [inv["dep_date"] for inv in self.inventory.values()]
        if not dep_dates:
            self.state = "completed"
            return
        earliest_dep = min(dep_dates)
        current_day = earliest_dep - timedelta(days=180)
        end_day = max(dep_dates)

        while current_day <= end_day and self.state in ("running", "paused"):
            if self.state == "paused":
                time.sleep(0.1)
                continue

            # jump_to sonrasi clock ilerlemis olabilir — sync et
            clock_day = self.clock.today()
            if clock_day > current_day:
                current_day = clock_day
                continue

            self._process_day(current_day)

            # Clock'u guncelle
            self.clock.jump_to(current_day.isoformat())

            current_day += timedelta(days=1)

            # Hiz = "1 dakikada kac gun" demek
            # 1 gunun suresi = 60 / speed saniye
            # speed=720  → 1 gun = 83ms  → 180 gun = 15sn  (yavas)
            # speed=1440 → 1 gun = 42ms  → 180 gun = 7.5sn (normal)
            # speed=4320 → 1 gun = 14ms  → 180 gun = 2.5sn (hizli)
            # speed=8640 → 1 gun = 7ms   → 180 gun = 1.3sn (cok hizli)
            # speed=14400→ 1 gun = 4ms   → 180 gun = 0.7sn (ultra)
            seconds_per_day = 60.0 / self.clock.speed
            delay = max(0.003, seconds_per_day)
            time.sleep(delay)

        if self.state != "paused":
            self.state = "completed"
            print(f"[Sim] Completed!", flush=True)

    def _process_day(self, sim_day):
        """Bir gunluk bot uretimi + fiyat guncelleme."""
        # Gunluk batch predict (bridge varsa)
        daily_preds = {}
        if self.bridge:
            try:
                daily_preds = self.bridge.predict_daily_batch(self.inventory, sim_day)
            except Exception:
                daily_preds = {}

        with self.lock:
            for key, inv in self.inventory.items():
                dep = inv["dep_date"]
                dtd = (dep - sim_day).days

                if dtd < 0 or dtd > 180:
                    continue
                if inv["sold"] >= inv["capacity"]:
                    continue

                # Bot uret (bridge varsa model-driven, yoksa rule-based)
                pred = daily_preds.get(key)
                self._generate_daily_bots(key, inv, dtd, pred)

                # Fiyatlari guncelle (bridge varsa pickup-informed)
                self._update_prices(key, inv, dtd, pred)

    # Bolgesel sezon faktorleri — sentetik veri ureticisiyle uyumlu
    REGION_SEASON = {
        "Europe":      {1:0.70,2:0.70,3:0.85,4:0.95,5:1.10,6:1.35,7:1.40,8:1.35,9:1.05,10:0.90,11:0.75,12:0.85},
        "Middle East": {1:1.10,2:1.05,3:1.00,4:0.90,5:0.80,6:0.70,7:0.65,8:0.70,9:0.90,10:1.00,11:1.10,12:1.20},
        "Africa":      {1:0.85,2:0.90,3:0.95,4:0.95,5:1.00,6:1.10,7:1.15,8:1.20,9:1.00,10:0.90,11:0.85,12:0.95},
        "Asia":        {1:0.90,2:0.85,3:0.95,4:1.00,5:1.00,6:1.05,7:1.10,8:1.05,9:1.00,10:1.05,11:1.05,12:1.10},
        "Americas":    {1:0.80,2:0.80,3:0.90,4:0.95,5:1.00,6:1.15,7:1.25,8:1.25,9:1.00,10:0.90,11:0.85,12:1.05},
    }

    # Ozel gun talep faktorleri — pricing_engine.SPECIAL_PERIODS'tan turetilir (tek kaynak)
    from pricing_engine import SPECIAL_PERIODS as _SP
    SPECIAL_DEMAND = {k: v[1] for k, v in _SP.items()}

    # Hafta gunu talep faktorleri
    DOW_DEMAND = {0:1.05, 1:1.00, 2:1.00, 3:1.10, 4:1.15, 5:1.10, 6:1.05}

    # Kabin oranları (veriden turetilmis)
    BIZ_RATIO = {"A":0.174, "B":0.174, "C":0.174, "D":0.299, "E":0.174, "F":0.176}

    def _generate_daily_bots(self, key, inv, dtd, pred=None):
        """
        Bir ucus icin bir gunluk bot yolcu uretimi.
        Tum faktorler talebe etki eder:
        - DTD curve (segment bazli booking pattern)
        - Bolgesel sezon (yaz/kis etkisi)
        - Ozel gunler (bayram, yilbasi)
        - Hafta gunu
        - Sentiment (destinasyon haberleri)
        - Kabin orani (veriden)
        """
        capacity = inv["capacity"]
        remaining = capacity - inv["sold"]
        if remaining <= 0:
            return

        cabin = inv.get("cabin", "economy")
        region = inv.get("region", "Europe")
        dep_date = inv["dep_date"]

        # ── Sentiment faktoru (her zaman real-time) ──
        sentiment_factor = 1.0
        if self.pricing.sentiment_cache and self.pricing.sentiment_cache.get("data"):
            arr = inv["route"].split("-")[1] if "-" in inv["route"] else ""
            city_key = self.pricing.airport_to_city.get(arr)
            if city_key and city_key in self.pricing.sentiment_cache["data"]:
                score = self.pricing.sentiment_cache["data"][city_key].get("aggregate", {}).get("composite_score", 0)
                sentiment_factor = 1.0 + score * 0.30

        # ══════════════════════════════════════════════
        # MODEL-DRIVEN yol (bridge + Two-Stage + TFT)
        # ══════════════════════════════════════════════
        if pred is not None and self.bridge:
            ts_demand = pred["daily_demand"]  # Two-Stage tahmini

            # TFT tavan/taban kontrolu
            band = self.bridge.get_tft_band(inv["route"], cabin, dep_date, dtd)
            if band:
                # Kumulatif kontrol: cok mu sattik / az mi sattik?
                if inv["sold"] > band["cumulative_ceiling"]:
                    ts_demand *= 0.3  # cok fazla satilmis, yavasla
                elif inv["sold"] < band["cumulative_floor"] and dtd < 60:
                    ts_demand *= 1.5  # cok az satilmis, hizlan

                # Gunluk band: clip
                ts_demand = max(band["daily_floor"], min(ts_demand, band["daily_ceiling"]))

            # Sentiment hala etkili
            ts_demand *= sentiment_factor
            # Noise
            ts_demand *= random.gauss(1.0, 0.15)
            ts_demand = max(0, ts_demand)

            # Segmentlere dagit
            for seg_id, seg in self.segments.items():
                dtd_weight = self._get_dtd_demand(seg_id, dtd)
                if dtd_weight <= 0:
                    continue
                share = seg.get("base_share_pct", 10) / 100
                seg_demand = ts_demand * share * dtd_weight

                if cabin == "business":
                    seg_demand *= self.BIZ_RATIO.get(seg_id, 0.18)
                else:
                    seg_demand *= (1.0 - self.BIZ_RATIO.get(seg_id, 0.18))

                num_bots = int(seg_demand)
                if random.random() < (seg_demand - num_bots):
                    num_bots += 1
                for _ in range(min(num_bots, remaining)):
                    if inv["sold"] >= capacity:
                        break
                    self._process_bot(key, inv, seg_id, dtd)
                    remaining = capacity - inv["sold"]
            return

        # ══════════════════════════════════════════════
        # FALLBACK: eski rule-based yol (bridge yoksa)
        # ══════════════════════════════════════════════
        month = dep_date.month
        season_factor = self.REGION_SEASON.get(region, {}).get(month, 1.0)
        special_key = (dep_date.year, dep_date.month, dep_date.day)
        special_factor = self.SPECIAL_DEMAND.get(special_key, 1.0)
        dow_factor = self.DOW_DEMAND.get(dep_date.weekday(), 1.0)
        demand_multiplier = season_factor * special_factor * dow_factor * sentiment_factor

        for seg_id, seg in self.segments.items():
            # Bu segment bu DTD'de talep gosteriyor mu?
            demand = self._get_dtd_demand(seg_id, dtd)
            if demand <= 0:
                continue

            # Segmentin genel paylarina gore olcekle
            share = seg.get("base_share_pct", 10) / 100
            daily_demand = demand * share * (capacity / 300)

            # Kabin orani (veriden)
            if cabin == "business":
                daily_demand *= self.BIZ_RATIO.get(seg_id, 0.18)
            else:
                daily_demand *= (1.0 - self.BIZ_RATIO.get(seg_id, 0.18))

            # TUM DIS FAKTORLERI UYGULA
            daily_demand *= demand_multiplier

            # Noise ekle
            daily_demand *= random.gauss(1.0, 0.15)
            daily_demand = max(0, daily_demand)

            num_bots = int(daily_demand)
            if random.random() < (daily_demand - num_bots):
                num_bots += 1

            for _ in range(min(num_bots, remaining)):
                if inv["sold"] >= capacity:
                    break
                self._process_bot(key, inv, seg_id, dtd)
                remaining = capacity - inv["sold"]

    def _process_bot(self, key, inv, segment_id, dtd):
        """Bir bot fiyati degerlendirir ve alir/almaz."""
        self.stats["total_bots"] += 1

        # Fiyat al
        quote = self.pricing.compute_price(inv, dtd, segment_id=segment_id)
        open_fares = quote["open_fares"]
        if not open_fares:
            self.stats["total_rejected"] += 1
            return False

        # Botun kisisel WTP'si
        seg = self.segments[segment_id]
        wtp = seg.get("wtp_multiplier", {"min": 0.8, "max": 1.2})
        personal_wtp = random.uniform(wtp["min"], wtp["max"])
        base_price = quote["base_price"]
        max_willing = base_price * personal_wtp

        # En pahali karsilayabilecegi fare class'i bul (gelir max)
        chosen_fare = None
        chosen_price = None
        for fc_id in reversed(open_fares):
            fc_price = quote["prices"][fc_id]
            if fc_price <= max_willing:
                chosen_fare = fc_id
                chosen_price = fc_price
                break

        if chosen_fare is None:
            self.stats["total_rejected"] += 1
            return False

        # Elastikiyet kontrolu
        prob = self.pricing.purchase_probability(segment_id, chosen_price, base_price)
        if random.random() > prob:
            self.stats["total_rejected"] += 1
            return False

        # SATIS!
        from pricing_engine import FARE_CLASSES
        baseline_full = self.pricing.compute_baseline_price(
            inv["route"], inv["cabin"], dtd,
            inv["dep_date"].isoformat() if isinstance(inv["dep_date"], date) else inv["dep_date"]
        )
        # Adil karsilastirma: baseline'a da ayni fare class multiplier uygula
        fc_mult = FARE_CLASSES.get(chosen_fare, {}).get("multiplier", 1.0)
        baseline = baseline_full * fc_mult

        inv["sold"] += 1
        inv["load_factor"] = inv["sold"] / inv["capacity"]
        if "fare_class_sold" in inv:
            inv["fare_class_sold"][chosen_fare] = inv["fare_class_sold"].get(chosen_fare, 0) + 1
        inv["revenue_dynamic"] += chosen_price
        inv["revenue_baseline"] += baseline

        inv["bookings"].append({
            "timestamp": self.clock.now().isoformat(),
            "dtd": dtd,
            "segment": segment_id,
            "fare_class": chosen_fare,
            "price": round(chosen_price, 2),
            "baseline_price": round(baseline, 2),
            "is_bot": True,
        })

        self.stats["total_sales"] += 1
        self.stats["total_revenue_dynamic"] += chosen_price
        self.stats["total_revenue_baseline"] += baseline

        return True

    def _update_prices(self, key, inv, dtd, pred=None):
        """Fiyatlari guncelle ve gecmise kaydet."""
        # Pickup-informed: predicted_remaining'i pricing engine'e ilet
        predicted_remaining = None
        if pred:
            predicted_remaining = pred.get("predicted_remaining")
        quote = self.pricing.compute_price(inv, dtd, predicted_remaining=predicted_remaining)
        inv["fare_classes_open"] = quote["open_fares"]
        inv["current_prices"] = quote["prices"]

        # Her gun icin fiyat gecmisi kaydet
        inv["price_history"].append({
            "sim_date": self.clock.today().isoformat(),
            "dtd": dtd,
            "load_factor": round(inv["load_factor"], 4),
            "sold": inv["sold"],
            "prices": {k: round(v, 2) for k, v in quote["prices"].items()},
            "open_fares": quote["open_fares"],
            "best_price": round(quote["best_price"], 2),
            "baseline_price": round(quote["baseline_price"], 2),
            "multipliers": quote["multipliers"],
        })

    def _get_dtd_demand(self, segment_id, dtd):
        """Segment icin DTD bazli talep degeri (0-1 arasi)."""
        if segment_id in self.dtd_curves:
            curve = self.dtd_curves[segment_id]
            # En yakin DTD noktasini bul + interpolasyon
            best = None
            best_dist = 999
            for pt in curve:
                d = abs(pt["dtd"] - dtd)
                if d < best_dist:
                    best_dist = d
                    best = pt["demand"]
            # Olcekle — gercekci ~%85 doluluk icin
            return (best or 0) * 200.0

        # Fallback: segment booking_window'una gore basit egri
        seg = self.segments.get(segment_id, {})
        bw = seg.get("booking_window", {"min_dtd": 0, "max_dtd": 180, "peak_dtd": 30})
        min_dtd = bw["min_dtd"]
        max_dtd = bw["max_dtd"]
        peak_dtd = bw["peak_dtd"]

        if dtd < min_dtd or dtd > max_dtd:
            return 0

        # Gaussian benzeri egri
        sigma = (max_dtd - min_dtd) / 4
        if sigma <= 0:
            sigma = 10
        demand = math.exp(-0.5 * ((dtd - peak_dtd) / sigma) ** 2)
        return demand * 40.0  # olcekleme — gercekci doluluk icin

    def _log_event(self, flight_key, event_type, detail):
        self.event_log.append({
            "timestamp": self.clock.now().isoformat(),
            "flight_key": flight_key,
            "type": event_type,
            "detail": detail,
        })
        if len(self.event_log) > self._max_events:
            self.event_log = self.event_log[-self._max_events:]

    # ── OZET VERILERI ─────────────────────────────────────────
    def get_status(self):
        """Dashboard icin genel durum."""
        with self.lock:
            rev_dyn = sum(i["revenue_dynamic"] for i in self.inventory.values())
            rev_base = sum(i["revenue_baseline"] for i in self.inventory.values())
            total_sold = sum(i["sold"] for i in self.inventory.values())
            total_cap = sum(i["capacity"] for i in self.inventory.values())
            stats_copy = dict(self.stats)
            events_copy = list(self.event_log[-20:])
            n_flights = len(self.inventory)

        return {
            "state": self.state,
            "clock": {
                "sim_date": self.clock.today().isoformat() if self.clock.sim_start else None,
                "sim_datetime": self.clock.now().isoformat() if self.clock.sim_start else None,
                "speed": self.clock.speed,
                "mode": self.clock.mode,
            },
            "stats": stats_copy,
            "summary": {
                "total_flights": n_flights,
                "total_capacity": total_cap,
                "total_sold": total_sold,
                "avg_load_factor": round(total_sold / max(total_cap, 1), 4),
                "revenue_dynamic": round(rev_dyn, 2),
                "revenue_baseline": round(rev_base, 2),
                "revenue_delta": round(rev_dyn - rev_base, 2),
                "revenue_delta_pct": round((rev_dyn - rev_base) / max(rev_base, 1) * 100, 1),
            },
            "recent_events": events_copy,
        }

    def get_flights_list(self):
        """Tum ucuslarin ozet listesi."""
        with self.lock:
            items = list(self.inventory.items())
        flights = []
        for key, inv in items:
            dtd = self.clock.dtd(inv["dep_date"])
            delta = inv["revenue_dynamic"] - inv["revenue_baseline"]
            flights.append({
                "key": key,
                "route": inv["route"],
                "cabin": inv["cabin"],
                "dep_date": inv["dep_date"].isoformat(),
                "dtd": dtd,
                "capacity": inv["capacity"],
                "sold": inv["sold"],
                "load_factor": round(inv["load_factor"], 4),
                "revenue_dynamic": round(inv["revenue_dynamic"], 2),
                "revenue_baseline": round(inv["revenue_baseline"], 2),
                "delta": round(delta, 2),
                "delta_pct": round(delta / max(inv["revenue_baseline"], 1) * 100, 1),
                "current_prices": inv["current_prices"],
                "open_fares": inv.get("fare_classes_open", []),
            })
        return sorted(flights, key=lambda x: x["dep_date"])

    def get_flight_detail(self, flight_key):
        """Tek ucusun detayli durumu."""
        with self.lock:
            inv = self.inventory.get(flight_key)
            if not inv:
                return None
            # Serializable kopya (lock icinde al)
            detail = {k: v for k, v in inv.items()}
            bookings_copy = list(inv["bookings"])
        detail["dep_date"] = inv["dep_date"].isoformat()
        detail["dtd"] = self.clock.dtd(inv["dep_date"])

        # Segment dagilimi
        seg_dist = defaultdict(int)
        for b in bookings_copy:
            seg_dist[b["segment"]] += 1
        detail["segment_distribution"] = dict(seg_dist)

        # Fare class dagilimi
        fc_dist = defaultdict(int)
        fc_rev = defaultdict(float)
        for b in bookings_copy:
            fc_dist[b["fare_class"]] += 1
            fc_rev[b["fare_class"]] += b["price"]
        detail["fare_class_distribution"] = dict(fc_dist)
        detail["fare_class_revenue"] = {k: round(v, 2) for k, v in fc_rev.items()}

        return detail
