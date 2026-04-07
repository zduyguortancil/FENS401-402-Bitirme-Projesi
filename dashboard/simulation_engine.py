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
import numpy as np


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

    def __init__(self, pricing_engine, forecast_bridge=None, network_optimizer=None):
        self.pricing = pricing_engine
        self.bridge = forecast_bridge
        self.network = network_optimizer  # NetworkOptimizer veya None
        self.clock = SimClock()
        self.state = "idle"
        self.inventory = {}
        self.lock = threading.Lock()
        self.thread = None

        # Istatistikler
        self.stats = {
            "total_bots": 0,
            "total_sales": 0,
            "total_rejected": 0,
            "total_revenue_dynamic": 0.0,
            "total_revenue_baseline": 0.0,
            "local_sales": 0,
            "connecting_sales": 0,
            "displacement_count": 0,
            "displacement_revenue_saved": 0.0,
            "cancellations": 0,
            "cancellation_refunds": 0.0,
            "no_shows": 0,
            "denied_boardings": 0,
            "denied_boarding_cost": 0.0,
            "shadow_lost_to_competitor": 0,
            "shadow_displacement": 0,
            "wtp_forced_purchase": 0,
            "guaranteed_no_fares": 0,
            "lost_to_PC": 0,
            "lost_to_EK": 0,
            "stolen_from_PC": 0,
            "stolen_from_EK": 0,
        }
        self.competitor_manager = None

        # Segment bazli no-show oranlari
        self.NO_SHOW_RATES = {
            "A": 0.15, "B": 0.05, "C": 0.08,
            "D": 0.03, "E": 0.07, "F": 0.20, "HUMAN": 0.02,
        }
        # Fare class iptal oranlari (toplam, booking omru boyunca)
        self.CANCEL_RATES = {"V": 0.01, "K": 0.03, "M": 0.08, "Y": 0.12}
        # Fare class refund oranlari
        self.REFUND_RATES = {"V": 0.0, "K": 0.50, "M": 0.80, "Y": 1.00}
        self.DENIED_BOARDING_COST = 400  # $ per pax

        # Route-group overbooking & no-show oranlari
        # Gercek havayolu pratigiyle uyumlu — rota tipi bazli farkli oranlar
        self.ROUTE_GROUPS = {
            # Europe business: yuksek frekans, esnek bilet, kolay rebooking
            "europe_biz": {
                "airports": {"LHR","FRA","CDG","LGW","STN","MAN"},
                "no_show": 0.10, "ob_eco": 0.07, "ob_biz": 0.03,
            },
            # Europe leisure: tatil yolcusu, dusuk iptal
            "europe_leisure": {
                "airports": {"BCN","FCO","MXP","NCE","ATH","PMI"},
                "no_show": 0.04, "ob_eco": 0.03, "ob_biz": 0.015,
            },
            # Europe mixed: VFR + is karisimi
            "europe_mixed": {
                "airports": {"MUC","MAD"},
                "no_show": 0.06, "ob_eco": 0.04, "ob_biz": 0.02,
            },
            # Middle East hub: transit, missed connection riski
            "me_hub": {
                "airports": {"DXB","DOH","AUH"},
                "no_show": 0.07, "ob_eco": 0.05, "ob_biz": 0.025,
            },
            # Middle East VFR/dini: hac/umre + VFR
            "me_vfr": {
                "airports": {"JED","RUH","AMM","BEY","TLV","KWI","BAH"},
                "no_show": 0.05, "ob_eco": 0.035, "ob_biz": 0.015,
            },
            # Africa: dusuk frekans, alternatif yok
            "africa": {
                "airports": {"JNB","CPT","NBO","MBA","LOS","ABV","CAI","HRG","CMN","RAK"},
                "no_show": 0.04, "ob_eco": 0.03, "ob_biz": 0.015,
            },
            # Asia leisure: uzun mesafe tatil
            "asia_leisure": {
                "airports": {"BKK","HKT","SIN"},
                "no_show": 0.03, "ob_eco": 0.02, "ob_biz": 0.01,
            },
            # Asia business/VFR: kurumsal + diaspora
            "asia_biz": {
                "airports": {"BOM","DEL","ICN","PEK","PVG","NRT","KIX"},
                "no_show": 0.06, "ob_eco": 0.04, "ob_biz": 0.02,
            },
            # Americas: ultra uzun mesafe, yuksek bilet fiyati
            "americas": {
                "airports": {"JFK","LAX","ORD","MIA","YYZ","YVR","MEX","GRU","EZE","GIG"},
                "no_show": 0.05, "ob_eco": 0.035, "ob_biz": 0.02,
            },
        }
        # Flat lookup: airport -> group key
        self._airport_group = {}
        for gk, gv in self.ROUTE_GROUPS.items():
            for apt in gv["airports"]:
                self._airport_group[apt] = gk

        # Event log (son 100 olay)
        self.event_log = []
        self._max_events = 200

        # Segment verileri (pricing engine'den)
        self.segments = pricing_engine.segments

        # Rekabet yogunlugu: rota bazli (0=monopol, 1=tam rekabet)
        # Yuksek rekabet: Avrupa trunk rotalar; Dusuk: Afrika/niche rotalar
        self.COMPETITION_INTENSITY = self._build_competition_map()

        # DTD demand curves (sonra yuklenir)
        self.dtd_curves = {}

    def _build_competition_map(self):
        """Rota bazli rekabet yogunlugu haritasi olustur.
        Avrupa trunk = yuksek, niche = dusuk, Afrika = cok dusuk."""
        comp = {}
        # Airport gruplarina gore rekabet yogunlugu
        HIGH_COMP = {"LHR","CDG","FRA","MUC","FCO","BCN","MAD","MXP"}  # 0.8
        MED_COMP = {"DXB","DOH","AUH","JFK","LAX","MIA","ORD","SIN","BKK","ICN"}  # 0.5
        LOW_COMP = {"JNB","NBO","LOS","ABV","CPT","MBA","CAI","HRG","CMN","RAK",
                    "AMM","BEY","TLV","KWI","BAH","RUH","JED"}  # 0.25
        for route_key in (self.pricing.route_distances if hasattr(self.pricing, 'route_distances') else {}):
            parts = route_key.split("_")
            arr = parts[1] if len(parts) == 2 else ""
            if arr in HIGH_COMP:
                comp[route_key] = 0.8
            elif arr in MED_COMP:
                comp[route_key] = 0.5
            elif arr in LOW_COMP:
                comp[route_key] = 0.25
            else:
                comp[route_key] = 0.4  # default orta
        return comp

    def _calc_overbooking_limit(self, route, cabin, capacity):
        """Route-group bazli overbooking limiti hesapla."""
        arr = route.split("-")[1] if "-" in route else ""
        group_key = self._airport_group.get(arr)
        if group_key:
            g = self.ROUTE_GROUPS[group_key]
            ob_pct = g["ob_biz"] if cabin == "business" else g["ob_eco"]
        else:
            ob_pct = 0.02 if cabin == "business" else 0.04
        return math.ceil(capacity * (1 + ob_pct))

    def _get_route_noshow(self, route):
        """Route-group bazli no-show orani."""
        arr = route.split("-")[1] if "-" in route else ""
        group_key = self._airport_group.get(arr)
        if group_key:
            return self.ROUTE_GROUPS[group_key]["no_show"]
        return 0.05  # default

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
        self.stats = {
            "total_bots": 0,
            "total_sales": 0,
            "total_rejected": 0,
            "total_revenue_dynamic": 0.0,
            "total_revenue_baseline": 0.0,
            "local_sales": 0,
            "connecting_sales": 0,
            "displacement_count": 0,
            "displacement_revenue_saved": 0.0,
            "cancellations": 0,
            "cancellation_refunds": 0.0,
            "no_shows": 0,
            "denied_boardings": 0,
            "denied_boarding_cost": 0.0,
            "shadow_lost_to_competitor": 0,
            "shadow_displacement": 0,
            "wtp_forced_purchase": 0,
            "guaranteed_no_fares": 0,
            "lost_to_PC": 0,
            "lost_to_EK": 0,
            "stolen_from_PC": 0,
            "stolen_from_EK": 0,
        }
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
                "max_lf_reached": 0.0,
                "max_fc_sold": {"V": 0, "K": 0, "M": 0, "Y": 0},
                "revenue_dynamic": 0.0,
                "revenue_baseline": 0.0,
                "bookings": [],
                "price_history": [],
                "fare_classes_open": [],
                "current_prices": {},
                "local_sold": 0,
                "connecting_sold": 0,
                "overbooking_limit": self._calc_overbooking_limit(f["route"], f["cabin"], f["capacity"]),
            }

        # Saat ayarla — booking window 180 gun oncesinden baslar
        sim_start = start_date - timedelta(days=180)
        self.clock.configure(sim_start, speed)

        # Rakip havayollarini yukle
        try:
            import os as _os
            _comp_path = _os.path.join(_os.path.dirname(__file__), '..', 'reports', 'competitor_config.json')
            _comp_path = _comp_path.replace("/", _os.sep)
            if _os.path.exists(_comp_path):
                from competitor_engine import CompetitorAirline, CompetitorManager
                with open(_comp_path, encoding="utf-8") as _cf:
                    _cc = json.load(_cf)
                comps = [CompetitorAirline(c) for c in _cc["competitors"]]
                self.competitor_manager = CompetitorManager(comps, self.segments)
                self.competitor_manager.initialize_inventory(self.inventory)
                print(f"[Sim] Competitors loaded: {[c.code for c in comps]}", flush=True)
        except Exception as e:
            print(f"[Sim] Competitor init skipped: {e}", flush=True)
            self.competitor_manager = None

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

            self._process_day(current_day)

            # Clock'u deterministik guncelle — current_day'i dogrudan set et
            self.clock.sim_datetime = datetime.combine(current_day, datetime.min.time())
            self.clock._pause_sim_dt = self.clock.sim_datetime
            self.clock.sim_start = self.clock.sim_datetime
            self.clock.wall_start = time.time()

            current_day += timedelta(days=1)

            # Hiz gecikme
            seconds_per_day = 60.0 / self.clock.speed
            delay = max(0.003, seconds_per_day)
            time.sleep(delay)

        if self.state != "paused":
            # Simulasyon bitti — no-show ve denied boarding hesapla
            self._process_departure()
            self.state = "completed"
            print(f"[Sim] Completed!", flush=True)

    def _process_day(self, sim_day):
        """Bir gunluk bot uretimi + iptal + fiyat guncelleme."""
        # Gunluk batch predict (bridge varsa)
        daily_preds = {}
        if self.bridge:
            try:
                daily_preds = self.bridge.predict_daily_batch(self.inventory, sim_day)
            except Exception:
                daily_preds = {}

        # Iptal kontrolu — mevcut bookingleri tara
        self._process_cancellations(sim_day)

        # Rakip fiyatlarini guncelle (gunde 1 kez)
        if self.competitor_manager:
            try:
                self.competitor_manager.update_daily_prices(sim_day, self.inventory, self.pricing)
            except Exception:
                pass

        with self.lock:
            for key, inv in self.inventory.items():
                dep = inv["dep_date"]
                dtd = (dep - sim_day).days

                if dtd < 0 or dtd > 180:
                    continue
                # Dolu ucuslarda sadece bot uretme — fiyat guncellemeye devam et
                is_full = inv["sold"] >= inv.get("overbooking_limit", inv["capacity"])

                # Bot uret (bridge varsa model-driven, yoksa rule-based)
                pred = daily_preds.get(key)
                if not is_full:
                    self._generate_daily_bots(key, inv, dtd, pred)

                # Fiyatlari HER ZAMAN guncelle (dolu olsa bile DTD ilerliyor)
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
        sell_limit = inv.get("overbooking_limit", capacity)  # overbooking dahil limit
        remaining = sell_limit - inv["sold"]
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
        # MODEL-DRIVEN yol (TFT = ana kaynak, Two-Stage = modulasyon)
        # ══════════════════════════════════════════════
        if pred is not None and self.bridge:
            band = self.bridge.get_tft_band(inv["route"], cabin, dep_date, dtd)

            if band and band.get("tft_total") is not None:
                # ── TFT-DRIVEN: ana talep kaynagi ──
                tft_total = band["tft_total"]

                # S-curve ile bugunun gunluk hedefi
                cum_today = band.get("cum_fraction", 0)
                cum_tomorrow = 1.0 - ((dtd + 1) / 180.0) ** 1.5 if dtd < 179 else 0.0
                cum_tomorrow = max(0.0, min(1.0, cum_tomorrow))
                daily_target = tft_total * max(cum_today - cum_tomorrow, 0)

                # Catch-up: gerideysen hizlan, ama gunluk hedefi 2x'ten fazla sisirme
                expected_sold = tft_total * cum_today
                gap = expected_sold - inv["sold"]
                catch_up = max(0, gap * 0.10)
                catch_up = min(catch_up, daily_target * 2)  # max 2x daily target

                daily_bots = daily_target + catch_up

                # Two-Stage p_sale — TFT zaten gerceklesmis talebi modeller,
                # p_sale ile carpma double-counting yapar. Sadece log icin tutuluyor.
                # p_sale = pred.get("p_sale", 0.7)
                # daily_bots *= (0.5 + 0.5 * p_sale)

            else:
                # TFT yoksa Two-Stage fallback
                daily_bots = pred["daily_demand"]

            # Sentiment + stokastik noise (Negative Binomial)
            daily_bots *= sentiment_factor
            daily_bots = self._stochastic_demand(daily_bots)

            # ── Segment dagilim agirliklari (normalize) ──
            seg_weights = {}
            total_w = 0.0
            for seg_id, seg in self.segments.items():
                dtd_w = self._get_dtd_demand(seg_id, dtd)
                if dtd_w <= 0:
                    continue
                share = seg.get("base_share_pct", 10) / 100
                if cabin == "business":
                    cab_r = self.BIZ_RATIO.get(seg_id, 0.18)
                else:
                    cab_r = 1.0 - self.BIZ_RATIO.get(seg_id, 0.18)
                w = dtd_w * share * cab_r
                seg_weights[seg_id] = max(w, 0)
                total_w += max(w, 0)

            if total_w <= 0:
                return

            # O&D ayirimi
            conn_pct = 0.0
            if self.network:
                conn_pct = self.network.get_connecting_pct(inv["route"])
            local_bots = daily_bots * (1.0 - conn_pct)
            connecting_bots = daily_bots * conn_pct

            # LOKAL botlar — daily_bots segmentlere dagitilir
            for seg_id, w in seg_weights.items():
                seg_demand = local_bots * (w / total_w)
                num_bots = int(seg_demand)
                if random.random() < (seg_demand - num_bots):
                    num_bots += 1
                for _ in range(min(num_bots, remaining)):
                    if inv["sold"] >= sell_limit:
                        break
                    self._process_bot(key, inv, seg_id, dtd, is_connecting=False, guaranteed=True)
                    remaining = sell_limit - inv["sold"]

            # CONNECTING botlar
            if self.network and connecting_bots > 0:
                for seg_id, w in seg_weights.items():
                    seg_demand = connecting_bots * (w / total_w)
                    num_bots = int(seg_demand)
                    if random.random() < (seg_demand - num_bots):
                        num_bots += 1
                    for _ in range(min(num_bots, remaining)):
                        if inv["sold"] >= sell_limit:
                            break
                        self._process_bot(key, inv, seg_id, dtd, is_connecting=True, guaranteed=True)
                        remaining = sell_limit - inv["sold"]
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

            # Stokastik noise (Negative Binomial)
            daily_demand = self._stochastic_demand(daily_demand)

            num_bots = int(daily_demand)
            if random.random() < (daily_demand - num_bots):
                num_bots += 1

            for _ in range(min(num_bots, remaining)):
                if inv["sold"] >= sell_limit:
                    break
                self._process_bot(key, inv, seg_id, dtd)
                remaining = sell_limit - inv["sold"]

    def _process_bot(self, key, inv, segment_id, dtd, is_connecting=False, guaranteed=False):
        """
        Bot satin alma karari.
        guaranteed=True: TFT-driven bot, MUTLAKA alir (gerceklesmis talep).
        guaranteed=False: fallback bot, tum filtrelerden gecer.
        """
        self.stats["total_bots"] += 1
        from pricing_engine import FARE_CLASSES

        # Fiyat al
        quote = self.pricing.compute_price(inv, dtd, segment_id=segment_id)
        open_fares = inv.get("fare_classes_open") or quote["open_fares"]
        if not open_fares:
            self.stats["total_rejected"] += 1
            if guaranteed:
                self.stats["guaranteed_no_fares"] += 1
            return False

        # Connecting yolcu icin fare proration
        leg_contribution = None
        od_info = None
        if is_connecting and self.network:
            origin_key = self.network.pick_random_origin()
            if origin_key:
                od_info = self.network.prorate_fare(origin_key, inv["route"], inv.get("cabin", "economy"))
                if od_info:
                    leg_contribution = od_info["leg_contribution"]

        # O&D bid price kontrolu — sadece LF > %70 sonrasi
        if self.network and inv["load_factor"] > 0.70:
            bid = self.network.get_bid_price(quote["base_price"], inv["capacity"], inv["sold"], open_fares, quote["prices"])
            check_fare = leg_contribution if is_connecting and leg_contribution else quote["best_price"]
            accepted, reason = self.network.evaluate_od(is_connecting, check_fare, leg_contribution or check_fare, bid)
            if not accepted:
                self.stats["shadow_displacement"] += 1
                if not guaranteed:
                    self.stats["total_rejected"] += 1
                    self.stats["displacement_count"] += 1
                    self.stats["displacement_revenue_saved"] += max(0, bid - check_fare)
                    return False

        # WTP hesapla (choice model icin once gerekli)
        seg = self.segments[segment_id]
        wtp = seg.get("wtp_multiplier", {"min": 0.8, "max": 1.2})
        personal_wtp = random.uniform(wtp["min"], wtp["max"])
        base_price = quote["base_price"]
        combined = quote["multipliers"].get("combined", 1.0)
        max_willing = base_price * combined * personal_wtp

        # ── REKABET MODELI: 3-yollu havayolu secimi ──
        if self.competitor_manager:
            best_our_price = quote["prices"].get(quote.get("best_fare", "M"), quote["base_price"])
            chosen, comp_code = self.competitor_manager.choose_airline(
                route=inv["route"], cabin=inv.get("cabin", "economy"),
                dep_date=inv["dep_date"], dtd=dtd, segment_id=segment_id,
                our_price=best_our_price, our_base_price=base_price,
                personal_wtp=personal_wtp, inv_key=key,
            )
            if chosen == "competitor":
                comp_price = self.competitor_manager.competitors[comp_code]._price_cache.get(key, best_our_price)
                ts = self.clock.now().isoformat()
                self.competitor_manager.competitors[comp_code].record_sale(key, segment_id, comp_price, dtd, ts)
                self.stats["shadow_lost_to_competitor"] += 1
                self.stats[f"lost_to_{comp_code}"] = self.stats.get(f"lost_to_{comp_code}", 0) + 1
                if not guaranteed:
                    self.stats["total_rejected"] += 1
                    return False
            else:
                # Biz kazandik — stolen tracking
                for cc in self.competitor_manager.competitors:
                    comp = self.competitor_manager.competitors[cc]
                    if comp.is_present(inv["route"], inv.get("cabin", "economy")) and not comp.is_sold_out(key):
                        self.stats[f"stolen_from_{cc}"] = self.stats.get(f"stolen_from_{cc}", 0) + 1
        else:
            # Fallback: eski basit rekabet modeli
            route_key = inv["route"].replace("-", "_")
            comp_intensity = self.COMPETITION_INTENSITY.get(route_key, 0.5)
            if comp_intensity > 0:
                competitor_fare = quote["base_price"] * random.uniform(0.85, 1.15)
                brand_premium = 0.05
                best_our_price = quote["prices"].get(quote.get("best_fare", "M"), quote["base_price"])
                if best_our_price > competitor_fare * (1 + brand_premium):
                    loss_prob = comp_intensity * 0.6
                    if random.random() < loss_prob:
                        self.stats["shadow_lost_to_competitor"] += 1
                        if not guaranteed:
                            self.stats["total_rejected"] += 1
                            return False

        # En pahali karsilayabilecegi fare class'i bul
        chosen_fare = None
        chosen_price = None
        for fc_id in reversed(open_fares):
            fc_price = quote["prices"][fc_id]
            if fc_price <= max_willing:
                chosen_fare = fc_id
                chosen_price = fc_price
                break

        # Karsilayamiyorsa → almaz (fiyat elastikiyeti)
        if chosen_fare is None:
            self.stats["total_rejected"] += 1
            if guaranteed:
                self.stats["wtp_priced_out"] = self.stats.get("wtp_priced_out", 0) + 1
            return False

        # Kota kontrolu — dolduysa bir ust sinifa kaydir
        fc_def = FARE_CLASSES.get(chosen_fare, {})
        if chosen_fare != "Y":
            quota = int(inv["capacity"] * fc_def.get("quota_pct", 1.0))
            sold_in_class = inv.get("fare_class_sold", {}).get(chosen_fare, 0)
            if sold_in_class >= quota:
                upper = {"V": "K", "K": "M", "M": "Y"}
                next_fc = upper.get(chosen_fare)
                if next_fc and next_fc in open_fares:
                    chosen_fare = next_fc
                    chosen_price = quote["prices"].get(next_fc, chosen_price)
                elif next_fc:
                    # Bir ust sinif kapali — daha yukari bak
                    for fc_try in ["K", "M", "Y"]:
                        if fc_try in open_fares:
                            chosen_fare = fc_try
                            chosen_price = quote["prices"].get(fc_try, chosen_price)
                            break

        # SATIS!
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

        booking_record = {
            "timestamp": self.clock.now().isoformat(),
            "dtd": dtd,
            "segment": segment_id,
            "fare_class": chosen_fare,
            "price": round(chosen_price, 2),
            "baseline_price": round(baseline, 2),
            "is_bot": True,
            "is_connecting": is_connecting,
        }
        if od_info:
            booking_record["origin"] = od_info["origin"]
            booking_record["total_itinerary_fare"] = od_info["total_fare"]
            booking_record["leg_contribution"] = od_info["leg_contribution"]
        inv["bookings"].append(booking_record)

        self.stats["total_sales"] += 1
        self.stats["total_revenue_dynamic"] += chosen_price
        self.stats["total_revenue_baseline"] += baseline
        if is_connecting:
            self.stats["connecting_sales"] += 1
            inv["connecting_sold"] = inv.get("connecting_sold", 0) + 1
        else:
            self.stats["local_sales"] += 1
            inv["local_sold"] = inv.get("local_sold", 0) + 1

        return True

    def _update_prices(self, key, inv, dtd, pred=None):
        """Fiyatlari guncelle ve gecmise kaydet.
        EMSR-b kota bazli fare class yonetimi + booking pace kontrolu.
        """
        # Pickup-informed: predicted_remaining'i pricing engine'e ilet
        predicted_remaining = None
        if pred:
            predicted_remaining = pred.get("predicted_remaining")
        comp_prices = self.competitor_manager.get_competitor_prices(key) if self.competitor_manager else None
        quote = self.pricing.compute_price(inv, dtd, predicted_remaining=predicted_remaining, competitor_prices=comp_prices)

        # Fare class yonetimi: pricing engine DTD+LF kurallarini verir,
        # kota + LF kalici kapatma burada (gunde 1 kez) uygulanir.
        open_fares = quote["open_fares"]

        # Monotonic fare class: max_lf_reached ile en yuksek LF'yi takip et.
        # LF asla dusmez (max), dolayisiyla fare class toggle olmaz.
        max_lf = inv.get("max_lf_reached", 0.0)
        current_lf = inv.get("load_factor", 0.0)
        if current_lf > max_lf:
            max_lf = current_lf
            inv["max_lf_reached"] = max_lf

        # Max LF'ye gore fare class filtrele (pricing engine'in LF thresholdlari)
        from pricing_engine import FARE_CLASSES
        fc_closed = set()
        if max_lf >= 0.40:
            fc_closed.add("V")
        if max_lf >= 0.70:
            fc_closed.add("K")
        if max_lf >= 0.85:
            fc_closed.add("M")

        # Kota kontrolu (bagimsiz, max_sold ile)
        capacity = inv["capacity"]
        fc_sold = inv.get("fare_class_sold", {})
        max_fc_sold = inv.get("max_fc_sold", {"V": 0, "K": 0, "M": 0, "Y": 0})
        for fc_id in ["V", "K", "M"]:
            sold_now = fc_sold.get(fc_id, 0)
            if sold_now > max_fc_sold.get(fc_id, 0):
                max_fc_sold[fc_id] = sold_now
            quota = int(capacity * FARE_CLASSES[fc_id]["quota_pct"])
            if max_fc_sold[fc_id] >= quota:
                fc_closed.add(fc_id)
        inv["max_fc_sold"] = max_fc_sold

        # Nesting
        if "M" in fc_closed:
            fc_closed.add("K")
            fc_closed.add("V")
        elif "K" in fc_closed:
            fc_closed.add("V")

        open_fares = [fc for fc in open_fares if fc not in fc_closed]
        if not open_fares:
            open_fares = ["Y"]

        inv["fare_classes_open"] = open_fares
        inv["current_prices"] = quote["prices"]

        # Her gun icin fiyat gecmisi kaydet
        inv["price_history"].append({
            "sim_date": self.clock.today().isoformat(),
            "dtd": dtd,
            "load_factor": round(inv["load_factor"], 4),
            "sold": inv["sold"],
            "prices": {k: round(v, 2) for k, v in quote["prices"].items()},
            "open_fares": open_fares,
            "quote_fares": quote["open_fares"],  # pricing engine'in dondugu (ever_closed oncesi)
            "fc_closed": sorted(fc_closed),
            "max_lf": round(max_lf, 4),
            "fc_sold": dict(inv.get("fare_class_sold", {})),
            "best_price": round(quote["best_price"], 2),
            "baseline_price": round(quote["baseline_price"], 2),
            "multipliers": quote["multipliers"],
        })

    def _process_cancellations(self, sim_day):
        """Mevcut bookingleri tara, iptal olasiliklarini hesapla.
        DTD-conditional: erken booking = yuksek iptal, gec booking = dusuk iptal.
        Fare class + booking_dtd etkilesimi gercekci iptal modeli saglar.
        """
        with self.lock:
            for key, inv in self.inventory.items():
                dep = inv["dep_date"]
                dtd = (dep - sim_day).days
                if dtd < 0 or dtd > 180:
                    continue
                bookings = inv.get("bookings", [])
                if not bookings:
                    continue
                to_remove = []
                for i, b in enumerate(bookings):
                    if b.get("cancelled"):
                        continue
                    fc = b.get("fare_class", "M")
                    base_cancel_rate = self.CANCEL_RATES.get(fc, 0.05)

                    # DTD-conditional: kalkisa uzak = iptal olasiligi yuksek
                    # Erken donem (dtd>90): base rate x 1.8
                    # Orta donem (30-90): base rate x 1.2
                    # Gec donem (7-30): base rate x 0.6
                    # Son hafta (0-7): base rate x 0.2 (neredeyse iptal etmez)
                    if dtd > 90:
                        dtd_factor = 1.8
                    elif dtd > 30:
                        dtd_factor = 1.2
                    elif dtd > 7:
                        dtd_factor = 0.6
                    else:
                        dtd_factor = 0.2

                    total_cancel_rate = base_cancel_rate * dtd_factor
                    daily_rate = total_cancel_rate / 180
                    if random.random() < daily_rate:
                        to_remove.append(i)
                # Iptalleri uygula
                for i in reversed(to_remove):
                    b = bookings[i]
                    b["cancelled"] = True
                    fc = b.get("fare_class", "M")
                    refund = b["price"] * self.REFUND_RATES.get(fc, 0.5)
                    inv["sold"] -= 1
                    inv["load_factor"] = inv["sold"] / inv["capacity"] if inv["capacity"] > 0 else 0
                    inv["revenue_dynamic"] -= refund
                    # fare_class_sold'dan da dusur — tutarlilik icin
                    if "fare_class_sold" in inv:
                        inv["fare_class_sold"][fc] = max(0, inv["fare_class_sold"].get(fc, 0) - 1)
                    self.stats["cancellations"] += 1
                    self.stats["cancellation_refunds"] += refund

    def _process_departure(self):
        """Kalkis gunu: no-show ve denied boarding hesapla."""
        with self.lock:
            for key, inv in self.inventory.items():
                bookings = [b for b in inv.get("bookings", []) if not b.get("cancelled")]
                total_booked = len(bookings)
                capacity = inv["capacity"]
                route_noshow = self._get_route_noshow(inv["route"])
                # Her yolcu icin no-show kontrolu
                # Rota bazli oran ile segment bazli oranin ortalamasi
                showed_up = 0
                for b in bookings:
                    seg = b.get("segment", "D")
                    seg_rate = self.NO_SHOW_RATES.get(seg, 0.05)
                    # Blend: %50 segment, %50 rota
                    no_show_rate = 0.5 * seg_rate + 0.5 * route_noshow
                    if random.random() > no_show_rate:
                        showed_up += 1
                    else:
                        self.stats["no_shows"] += 1
                # Denied boarding
                if showed_up > capacity:
                    denied = showed_up - capacity
                    self.stats["denied_boardings"] += denied
                    self.stats["denied_boarding_cost"] += denied * self.DENIED_BOARDING_COST

    # ── STOKASTIK TALEP MODELI ─────────────────────────────
    def _stochastic_demand(self, expected):
        """Beklenen talep etrafinda Negative Binomial noise uygula.
        NegBin, Poisson'dan daha gercekci: overdispersion (varyans > ortalama).
        Dispersion parametresi r=5: orta duzey varyans."""
        if expected <= 0:
            return 0.0
        # NegBin parametreleri: mean=expected, variance=expected + expected^2/r
        r = 5.0  # dispersion (dusuk r = daha fazla varyans)
        p = r / (r + expected)
        try:
            realized = float(np.random.negative_binomial(r, p))
        except (ValueError, Exception):
            realized = max(0, expected * random.gauss(1.0, 0.25))
        return realized

    # ── MONTE CARLO RUNNER ────────────────────────────────
    def run_monte_carlo(self, flights_data, date_range, n_runs=50,
                        speed=14400, dtd_curves=None, on_progress=None):
        """
        Ayni parametrelerle N kez simulasyon calistir, istatistikleri topla.
        Her run farkli random seed ile farkli talep realizasyonu gorecek.

        Returns: {
            "n_runs": 50,
            "revenue_dynamic": {"mean": X, "std": Y, "min": A, "max": B, "ci95": (lo, hi)},
            "revenue_baseline": {"mean": X, "std": Y},
            "revenue_delta_pct": {"mean": X, "std": Y, "ci95": (lo, hi)},
            "avg_load_factor": {"mean": X, "std": Y},
            "fare_mix": {"V": mean_pct, "K": ..., "M": ..., "Y": ...},
            "lost_to_competitor": {"mean": X},
            "denied_boardings": {"mean": X},
            "runs": [per-run summaries]
        }
        """
        import numpy as np
        results = []
        print(f"[MC] Starting {n_runs} Monte Carlo runs...", flush=True)

        for run_idx in range(n_runs):
            seed = 1000 + run_idx  # farkli seed her run icin
            self.initialize(flights_data, date_range, speed=speed,
                           dtd_curves=dtd_curves, seed=seed)

            # Senkron calistir (thread yerine dogrudan)
            dep_dates = [inv["dep_date"] for inv in self.inventory.values()]
            if not dep_dates:
                continue
            earliest_dep = min(dep_dates)
            current_day = earliest_dep - timedelta(days=180)
            end_day = max(dep_dates)
            self.state = "running"
            self.clock.start()

            while current_day <= end_day:
                self._process_day(current_day)
                self.clock.sim_datetime = datetime.combine(current_day, datetime.min.time())
                self.clock._pause_sim_dt = self.clock.sim_datetime
                self.clock.sim_start = self.clock.sim_datetime
                self.clock.wall_start = time.time()
                current_day += timedelta(days=1)

            self._process_departure()
            self.state = "completed"

            # Sonuclari topla
            status = self.get_status()
            summary = status["summary"]
            stats = status["stats"]

            # Fare mix hesapla
            total_fc = {"V": 0, "K": 0, "M": 0, "Y": 0}
            for inv in self.inventory.values():
                for fc, cnt in inv.get("fare_class_sold", {}).items():
                    total_fc[fc] = total_fc.get(fc, 0) + cnt
            total_pax = max(sum(total_fc.values()), 1)

            results.append({
                "seed": seed,
                "revenue_dynamic": summary["revenue_dynamic"],
                "revenue_baseline": summary["revenue_baseline"],
                "revenue_delta_pct": summary["revenue_delta_pct"],
                "avg_load_factor": summary["avg_load_factor"],
                "total_sold": summary["total_sold"],
                "denied_boardings": stats.get("denied_boardings", 0),
                "lost_to_competitor": stats.get("lost_to_competitor", 0),
                "cancellations": stats.get("cancellations", 0),
                "fare_mix": {fc: round(cnt / total_pax * 100, 1) for fc, cnt in total_fc.items()},
            })

            if on_progress:
                on_progress(run_idx + 1, n_runs)
            if (run_idx + 1) % 10 == 0:
                print(f"[MC] Completed {run_idx + 1}/{n_runs} runs", flush=True)

        # Aggregate
        rev_dyn = np.array([r["revenue_dynamic"] for r in results])
        rev_base = np.array([r["revenue_baseline"] for r in results])
        delta_pct = np.array([r["revenue_delta_pct"] for r in results])
        lf = np.array([r["avg_load_factor"] for r in results])

        def stats_summary(arr):
            return {
                "mean": round(float(np.mean(arr)), 2),
                "std": round(float(np.std(arr)), 2),
                "min": round(float(np.min(arr)), 2),
                "max": round(float(np.max(arr)), 2),
                "ci95": (round(float(np.percentile(arr, 2.5)), 2),
                         round(float(np.percentile(arr, 97.5)), 2)),
            }

        # Fare mix ortalamasi
        avg_fm = {}
        for fc in ["V", "K", "M", "Y"]:
            vals = [r["fare_mix"].get(fc, 0) for r in results]
            avg_fm[fc] = round(float(np.mean(vals)), 1)

        mc_result = {
            "n_runs": n_runs,
            "revenue_dynamic": stats_summary(rev_dyn),
            "revenue_baseline": stats_summary(rev_base),
            "revenue_delta_pct": stats_summary(delta_pct),
            "avg_load_factor": stats_summary(lf),
            "fare_mix": avg_fm,
            "denied_boardings_mean": round(float(np.mean([r["denied_boardings"] for r in results])), 1),
            "lost_to_competitor_mean": round(float(np.mean([r["lost_to_competitor"] for r in results])), 1),
            "runs": results,
        }

        print(f"\n[MC] ═══ MONTE CARLO SONUC ({n_runs} run) ═══")
        print(f"  Revenue Delta: {mc_result['revenue_delta_pct']['mean']}% "
              f"(95% CI: {mc_result['revenue_delta_pct']['ci95']})")
        print(f"  Avg LF: {mc_result['avg_load_factor']['mean']}")
        print(f"  Fare Mix: {avg_fm}")
        print(f"  Denied Boardings (mean): {mc_result['denied_boardings_mean']}")
        print(f"  Lost to Competitor (mean): {mc_result['lost_to_competitor_mean']}")
        print(f"══════════════════════════════════════════\n", flush=True)

        return mc_result

    # Overlapping segment booking windows — gercekci S-curve icin
    # Her segment DTD araligi boyunca aktif, Gaussian ile peak civarinda yogun
    _SEGMENT_WINDOWS = {
        "D": {"min": 0, "max": 180, "peak": 120, "sigma": 50},   # Tatilci: genis pencere, erken peak
        "E": {"min": 0, "max": 150, "peak": 75,  "sigma": 40},   # Ogrenci: genis, orta peak
        "B": {"min": 0, "max": 120, "peak": 45,  "sigma": 30},   # Erken planlayan: orta pencere
        "C": {"min": 0, "max": 60,  "peak": 21,  "sigma": 15},   # Kongre/is: dar, gec peak
        "A": {"min": 0, "max": 21,  "peak": 7,   "sigma": 6},    # Is yolcusu: son dakika
        "F": {"min": 0, "max": 7,   "peak": 2,   "sigma": 2},    # Acil: en son
    }

    def _get_dtd_demand(self, segment_id, dtd):
        """Segment icin DTD bazli talep agirligi. Overlapping Gaussian."""
        if segment_id in self.dtd_curves:
            curve = self.dtd_curves[segment_id]
            best = None
            best_dist = 999
            for pt in curve:
                d = abs(pt["dtd"] - dtd)
                if d < best_dist:
                    best_dist = d
                    best = pt["demand"]
            return (best or 0) * 200.0

        # Overlapping Gaussian windows
        win = self._SEGMENT_WINDOWS.get(segment_id)
        if not win:
            # Bilinmeyen segment — demand_functions_report'tan fallback
            seg = self.segments.get(segment_id, {})
            bw = seg.get("booking_window", {"min_dtd": 0, "max_dtd": 180, "peak_dtd": 30})
            win = {"min": bw["min_dtd"], "max": bw["max_dtd"], "peak": bw["peak_dtd"],
                   "sigma": max((bw["max_dtd"] - bw["min_dtd"]) / 4, 5)}

        if dtd < win["min"] or dtd > win["max"]:
            return 0

        sigma = win["sigma"]
        demand = math.exp(-0.5 * ((dtd - win["peak"]) / sigma) ** 2)
        return demand

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
            "competition": self.competitor_manager.get_summary() if self.competitor_manager else None,
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
            # Serializable kopya (lock icinde al) — set'leri list'e cevir
            detail = {}
            for k, v in inv.items():
                if isinstance(v, set):
                    detail[k] = list(v)
                else:
                    detail[k] = v
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
