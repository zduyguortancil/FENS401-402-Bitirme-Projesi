import math
import os
import requests
import json
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, session, redirect, url_for

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.environ.get("BILETBUL_SECRET", "bb-desktop-secret-2026")

# ─── AUTH HELPERS ─────────────────────────────────────────
import hashlib, uuid, re
from functools import wraps

BB_USERS_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users_db.json")

def _load_bb_users():
    if not os.path.exists(BB_USERS_DB):
        return {"users": []}
    with open(BB_USERS_DB, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_bb_users(db):
    with open(BB_USERS_DB, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def _hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("bb_user"):
            return redirect(url_for("bb_login_page"))
        return f(*args, **kwargs)
    return decorated

DASHBOARD_URL = "http://localhost:5005"
REQUEST_TIMEOUT = 4

# ─── PERSISTENT PRICE CACHE ─────────────────────────────────────────
# Stores the last known simulation price for each flight key.
# Format: { "IST-CDG_2026-05-10_economy": {"thy": 86.04, "pc": 103.82, "ek": 213.36, "min_price": 86.04} }
# Once a price is fetched from the simulation, it persists here even when the simulation is stopped.
PRICE_CACHE = {}

# Havayolu bazli gunluk sefer saatleri (sim'de saat verisi olmadigi icin deterministik uretim)
# Format: (dep_hour, dep_min, duration_min)
AIRLINE_SCHEDULES = {
    "THY": [
        (7, 30, 0),   # sabah
        (13, 15, 0),  # oglen
        (21, 45, 0),  # gece
    ],
    "PC": [
        (6, 0, 0),    # cok erken
        (17, 30, 0),  # aksam
    ],
    "EK": [
        (2, 30, 0),   # gece (Dubai hub)
        (22, 0, 0),   # gece
    ],
}

# Rota mesafelerine gore ucus suresi (dakika)
ROUTE_DURATIONS = {
    "LHR": 235, "CDG": 235, "FRA": 215, "MUC": 210, "MAD": 285, "BCN": 270,
    "FCO": 200, "MXP": 205, "NCE": 230, "MAN": 250,
    "DXB": 225, "DOH": 215, "AUH": 220, "RUH": 200, "JED": 195,
    "AMM": 150, "CAI": 140, "TLV": 155, "BEY": 145, "KWI": 195, "BAH": 200,
    "HRG": 130, "CMN": 300, "RAK": 305,
    "JNB": 500, "CPT": 540, "NBO": 440, "MBA": 460, "LOS": 480, "ABV": 470,
    "NRT": 720, "KIX": 715, "ICN": 680, "PEK": 650, "PVG": 640,
    "SIN": 680, "BKK": 600, "HKT": 620, "DEL": 430, "BOM": 380,
    "JFK": 660, "LAX": 800, "ORD": 720, "MIA": 690,
    "YYZ": 720, "YVR": 860, "MEX": 840, "GRU": 1080, "EZE": 1100, "GIG": 1090,
}

def _get_duration(dest):
    return ROUTE_DURATIONS.get(dest, 300)

def _fmt_time(h, m):
    return f"{h:02d}:{m:02d}"

def _add_minutes(h, m, mins):
    total = h * 60 + m + mins
    return (total // 60) % 24, total % 60

def _get_flight_times(airline_code, dest, flight_idx=0):
    """Havayolu ve destinasyona gore ucus saatini getir/uret."""
    schedules = AIRLINE_SCHEDULES.get(airline_code, [(10, 30, 0)])
    sched = schedules[flight_idx % len(schedules)]
    dep_h, dep_m = sched[0], sched[1]
    
    # Base duration for the route
    base_dur = _get_duration(dest)
    
    # Airline specific offset (Low cost carriers might take slightly longer)
    airline_offset = {"THY": 0, "PC": 15, "EK": -5}.get(airline_code, 0)
    
    # Deterministic jitter per flight (-5 to +10 mins)
    jitter = ((dep_h * 60 + dep_m + flight_idx * 17) % 15) - 5
    
    dur = base_dur + airline_offset + jitter
    
    arr_h, arr_m = _add_minutes(dep_h, dep_m, dur)
    return _fmt_time(dep_h, dep_m), _fmt_time(arr_h, arr_m), dur

def _get_synthetic_price(origin, dest, dep_date_str, cabin, code):
    try:
        from pricing_engine import BASE_PRICE_FORMULAS, SEASON_FACTORS, DOW_FACTORS
    except ImportError:
        return 500.0 if cabin == "economy" else 1500.0
    from datetime import datetime
    import hashlib as _hl
    
    dur = _get_duration(dest)
    dist_km = dur / 60.0 * 900
    base_formula = BASE_PRICE_FORMULAS.get(cabin, BASE_PRICE_FORMULAS["economy"])
    route_base = base_formula(dist_km)
    
    d = datetime.strptime(dep_date_str, "%Y-%m-%d")
    seed_val = int(_hl.md5(f"{dest}{dep_date_str}".encode()).hexdigest(), 16) % 1000
    noise = (seed_val - 500) / 500.0
    season_f = SEASON_FACTORS.get(d.month, 1.0)
    dow_f = DOW_FACTORS.get(d.weekday(), 1.0)
    
    thy_p = route_base * season_f * dow_f * (1.0 + noise * 0.15)
    thy_p = round(max(thy_p, 80.0), 0)
    
    if code == "THY":
        return thy_p
    elif code == "PC":
        return round(thy_p * 0.82, 0) if cabin == "economy" else None
    elif code == "EK":
        return round(thy_p * 1.18, 0)
    return thy_p

SIM_ROUTE_FALLBACK = [
    "IST-LHR", "IST-MAD", "IST-JFK", "IST-DXB", "IST-CDG", "IST-FRA", "IST-BCN", "IST-FCO", "IST-MUC", "IST-NCE",
    "IST-AMM", "IST-CAI", "IST-MAN", "IST-TLV", "IST-BEY", "IST-JED", "IST-RUH", "IST-DOH", "IST-BAH", "IST-KWI",
    "IST-AUH", "IST-HRG", "IST-CMN", "IST-RAK", "IST-NBO", "IST-MBA", "IST-LOS", "IST-ABV", "IST-JNB", "IST-CPT",
    "IST-NRT", "IST-KIX", "IST-ICN", "IST-PEK", "IST-PVG", "IST-SIN", "IST-BKK", "IST-HKT", "IST-DEL", "IST-BOM",
    "IST-LAX", "IST-ORD", "IST-MIA", "IST-YYZ", "IST-YVR", "IST-MEX", "IST-GRU", "IST-EZE", "IST-GIG", "IST-MXP",
]

# Konfigürasyon: Bagaj ve Pet ücretleri (sabit)
AIRLINE_EXTRAS = {
    "THY": {
        "economy": {"baggage_kg": 30, "pet_fee": 45.0, "extra_bag_per_kg": 8.0, "type": "full-service", "name": "Turkish Airlines"},
        "business": {"baggage_kg": 40, "pet_fee": 65.0, "extra_bag_per_kg": 0.0, "type": "full-service", "name": "Turkish Airlines"}
    },
    "PC": {
        "economy": {"baggage_kg": 15, "pet_fee": None, "extra_bag_per_kg": 12.0, "type": "budget", "name": "Pegasus Airlines"}
    },
    "EK": {
        "economy": {"baggage_kg": 40, "pet_fee": 55.0, "extra_bag_per_kg": 6.0, "type": "premium", "name": "Emirates"},
        "business": {"baggage_kg": 50, "pet_fee": 80.0, "extra_bag_per_kg": 0.0, "type": "premium", "name": "Emirates"}
    }
}

# Soldout_tracker: { "compCode_flightKey": {"dtd_at_soldout": 28, "sim_date": "2026-07-28"} }
_soldout_tracker = {}

def _dashboard_get(path, timeout=REQUEST_TIMEOUT):
    """Safe helper for dashboard JSON GET requests."""
    try:
        response = requests.get(f"{DASHBOARD_URL}{path}", timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return None


def _dashboard_post(path, payload, timeout=REQUEST_TIMEOUT):
    """Safe helper for dashboard JSON POST requests."""
    try:
        return requests.post(f"{DASHBOARD_URL}{path}", json=payload, timeout=timeout)
    except requests.RequestException:
        return None


def get_sim_status():
    """Check if the simulation is running on the main dashboard."""
    return _dashboard_get("/api/sim/status")

def get_live_data():
    """RM Dashboard'dan canli snapshot ceker."""
    try:
        r1 = _dashboard_get("/api/sim/flights")
        r2 = _dashboard_get("/api/sim/competition")
        
        if r1 and r2:
            flights_list = r1.get("flights", [])
            flights_dict = {f["key"]: f for f in flights_list} if isinstance(flights_list, list) else flights_list
            return flights_dict, r2.get("competitors", {})
    except Exception as e:
        print(f"Error connecting to RM Dashboard: {e}")
    return None, None

def get_flight_live_data(flight_key):
    """RM Dashboard'dan tek ucus icin detay ceker."""
    f_inv = _dashboard_get(f"/api/sim/flight/{flight_key}")
    comp_data = _dashboard_get(f"/api/sim/competition/flight/{flight_key}")
    return f_inv, comp_data

def calc_extras(code, cabin, pax, wants_pet, extra_bag_kg):
    cfg = AIRLINE_EXTRAS.get(code, {}).get(cabin)
    if not cfg:
        return 0.0, False # available=False
        
    fee = 0.0
    if wants_pet:
        if cfg["pet_fee"] is None:
            return 0.0, False # Pet disallowed
        fee += cfg["pet_fee"]
        
    if extra_bag_kg > 0:
        fee += extra_bag_kg * cfg["extra_bag_per_kg"]
        
    return fee * pax, True

def build_single_airline_result(origin, dest, dep_date_str, cabin, pax, wants_pet, extra_bag_kg,
                                  code, flight_idx=0, inv_data=None, comp_data=None,
                                  current_dtd=180, sim_running=False, flight_key=None):
    """Tek havayolu icin tek sefer sonucu uret."""
    dep_time, arr_time, dur_min = _get_flight_times(code, dest, flight_idx)
    cfg = AIRLINE_EXTRAS.get(code, {}).get(cabin)
    if not cfg:
        return None
    if code == "THY":
        if inv_data and flight_key and flight_key in inv_data:
            f_inv = inv_data[flight_key]
            prices = f_inv.get("current_prices", {})
            open_fares = f_inv.get("fare_classes_open", [])
            best_fare = open_fares[0] if open_fares else "Y"
            base_ticket_price = prices.get(best_fare, 500.0)
            extras_fee, valid_extras = calc_extras(code, cabin, pax, wants_pet, extra_bag_kg)
            thy_seats = f_inv.get("capacity", 0) - f_inv.get("sold", 0)
            return {
                "code": code, "available": valid_extras,
                "name": cfg["name"], "type": cfg.get("type", "full-service"),
                "price_per_pax": base_ticket_price,
                "total_with_extras": (base_ticket_price * pax) + extras_fee,
                "best_fare": best_fare, "open_fares": open_fares,
                "sold_out": thy_seats <= 0, "seats_remaining": max(thy_seats, 0),
                "load_factor": f_inv.get("load_factor", 0),
                "last_seats": 0 < thy_seats < 15,
                "baggage_kg": cfg["baggage_kg"], "pet_allowed": cfg["pet_fee"] is not None,
                "pet_fee": cfg["pet_fee"], "extra_baggage_fee": extras_fee,
                "flight_key": flight_key, "prices_breakdown": prices,
                "departure_time": dep_time, "arrival_time": arr_time,
                "duration_min": dur_min, "flight_idx": flight_idx,
            }
        else:
            base = _get_synthetic_price(origin, dest, dep_date_str, cabin, code)
            extras_fee, valid_extras = calc_extras(code, cabin, pax, wants_pet, extra_bag_kg)
            return {
                "code": code, "available": valid_extras,
                "name": cfg["name"], "type": cfg.get("type", "full-service"),
                "price_per_pax": base, "total_with_extras": (base * pax) + extras_fee,
                "best_fare": "M", "open_fares": ["M", "Y"],
                "sold_out": False, "seats_remaining": 150, "load_factor": 0.5,
                "last_seats": False, "baggage_kg": cfg["baggage_kg"],
                "pet_allowed": cfg["pet_fee"] is not None, "pet_fee": cfg["pet_fee"],
                "extra_baggage_fee": extras_fee, "flight_key": None, "prices_breakdown": {},
                "departure_time": dep_time, "arrival_time": arr_time,
                "duration_min": dur_min, "flight_idx": flight_idx,
            }
    else:
        c_ticket_price = _get_synthetic_price(origin, dest, dep_date_str, cabin, code)
        if c_ticket_price is None:
            c_ticket_price = 450.0 if code == "PC" else 600.0
        c_sold_out = False
        c_seats = 100
        days_ago = None
        if comp_data and code in comp_data:
            c_inv = comp_data[code]
            if c_inv and "price" in c_inv:
                c_ticket_price = c_inv.get("price", c_ticket_price)
                c_seats = c_inv.get("capacity", 0) - c_inv.get("sold", 0)
                c_sold_out = c_seats <= 0
                trk_key = f"{code}_{flight_key}"
                if c_sold_out:
                    if trk_key not in _soldout_tracker:
                        _soldout_tracker[trk_key] = current_dtd
                    soldout_dtd = _soldout_tracker[trk_key]
                    days_ago = soldout_dtd - current_dtd
                elif trk_key in _soldout_tracker:
                    del _soldout_tracker[trk_key]
        extras_fee, valid_extras = calc_extras(code, cabin, pax, wants_pet, extra_bag_kg)
        if not valid_extras:
            return None
        return {
            "code": code, "available": True,
            "name": cfg["name"], "type": cfg.get("type", "full-service"),
            "price_per_pax": c_ticket_price,
            "total_with_extras": (c_ticket_price * pax) + extras_fee,
            "sold_out": c_sold_out, "seats_remaining": max(c_seats, 0),
            "last_seats": 0 < c_seats < 10,
            "days_since_soldout": days_ago if days_ago and days_ago > 0 else None,
            "baggage_kg": cfg["baggage_kg"], "pet_allowed": cfg["pet_fee"] is not None,
            "pet_fee": cfg["pet_fee"], "extra_baggage_fee": extras_fee,
            "flight_key": flight_key,
            "departure_time": dep_time, "arrival_time": arr_time,
            "duration_min": dur_min, "flight_idx": flight_idx,
        }


def build_leg_result(origin, dest, dep_date_str, cabin, pax, wants_pet, extra_bag_kg):
    dep_date = datetime.strptime(dep_date_str, "%Y-%m-%d").date()

    sim_status = get_sim_status()
    sim_running = False
    current_dtd = 180
    if sim_status:
        clock_info = sim_status.get("clock", {})
        if clock_info.get("sim_datetime"):
            sim_dt = datetime.fromisoformat(clock_info["sim_datetime"].split("T")[0]).date()
            current_dtd = (dep_date - sim_dt).days
        else:
            current_dtd = (dep_date - datetime.today().date()).days
        sim_running = sim_status.get("state") in ("running", "paused", "completed")
    else:
        current_dtd = (dep_date - datetime.today().date()).days

    if current_dtd < 0:
        return {"error": f"Date in the past (DTD: {current_dtd})"}

    route = f"{origin}-{dest}"
    flight_key = f"{route}_{dep_date_str}_{cabin}"

    inv_data, comp_data = None, None
    if sim_running:
        f_inv, comp_data = get_flight_live_data(flight_key)
        inv_data = {flight_key: f_inv} if f_inv and "error" not in f_inv else {}

    # Her havayolu icin tum seferleri uret
    airlines_flights = {}
    for code in ["THY", "PC", "EK"]:
        cfg = AIRLINE_EXTRAS.get(code, {}).get(cabin)
        schedules = AIRLINE_SCHEDULES.get(code, [(10, 30, 0)])
        flights_list = []
        if not cfg:
            airlines_flights[code] = [{"available": False, "reason": "Cabin not offered",
                                        "name": AIRLINE_EXTRAS[code]["economy"]["name"]}]
            continue
        for idx in range(len(schedules)):
            f = build_single_airline_result(
                origin, dest, dep_date_str, cabin, pax, wants_pet, extra_bag_kg,
                code, flight_idx=idx, inv_data=inv_data, comp_data=comp_data,
                current_dtd=current_dtd, sim_running=sim_running, flight_key=flight_key
            )
            if f:
                flights_list.append(f)
        airlines_flights[code] = flights_list if flights_list else [{"available": False,
                                                                      "reason": "No flights available",
                                                                      "name": cfg["name"]}]

    return {
        "route": route, "dep_date": dep_date_str, "cabin": cabin,
        "dtd": current_dtd, "simulation_live": sim_running,
        "airlines": airlines_flights,
        # Eski API uyumlulugu icin thy/PC/EK aliaslar (ilk seferi dondur)
        "thy": airlines_flights["THY"][0] if airlines_flights.get("THY") else None,
        "PC": airlines_flights["PC"][0] if airlines_flights.get("PC") else None,
        "EK": airlines_flights["EK"][0] if airlines_flights.get("EK") else None,
    }

@app.route("/")
def landing():
    if session.get("bb_user"):
        return render_template("loading.html")
    return redirect(url_for("bb_login_page"))


@app.route("/search")
@login_required
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET"])
def bb_login_page():
    if session.get("bb_user"):
        return redirect(url_for("landing"))
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def bb_login_post():
    data = request.get_json(force=True) or {}
    identifier = data.get("username", "").strip()
    password = data.get("password", "").strip()
    if not identifier or not password:
        return jsonify({"success": False, "message": "Username and password are required."}), 400
    db = _load_bb_users()
    pw_hash = _hash_pw(password)
    user = next((u for u in db["users"] if (u["username"] == identifier or u["email"] == identifier) and u["password_hash"] == pw_hash), None)
    if not user:
        return jsonify({"success": False, "message": "Invalid username or password."}), 401
    session["bb_user"] = {"id": user["id"], "username": user["username"], "email": user["email"]}
    return jsonify({"success": True, "redirect": "/"})


@app.route("/register", methods=["POST"])
def bb_register_post():
    data = request.get_json(force=True) or {}
    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()
    if not re.match(r'^[a-zA-Z0-9]{3,20}$', username):
        return jsonify({"success": False, "message": "Invalid username."}), 400
    if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
        return jsonify({"success": False, "message": "Invalid email address."}), 400
    if not re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&_\-.])[A-Za-z\d@$!%*?&_\-.]{8,32}$', password):
        return jsonify({"success": False, "message": "Password does not meet requirements."}), 400
    db = _load_bb_users()
    if any(u["username"] == username for u in db["users"]):
        return jsonify({"success": False, "message": "This username is already taken."}), 409
    if any(u["email"] == email for u in db["users"]):
        return jsonify({"success": False, "message": "This email address is already registered."}), 409
    from datetime import datetime as _dt
    new_user = {"id": str(uuid.uuid4()), "username": username, "email": email, "password_hash": _hash_pw(password), "created_at": _dt.now().isoformat()}
    db["users"].append(new_user)
    _save_bb_users(db)
    return jsonify({"success": True, "message": "Registration successful."})


@app.route("/logout")
def bb_logout():
    session.pop("bb_user", None)
    return redirect(url_for("bb_login_page"))

@app.route("/api/search")
def api_search():
    try:
        origin = request.args.get("origin", "IST").upper()
        dest = request.args.get("destination", "MAD").upper()
        dep_date = request.args.get("dep_date")
        ret_date = request.args.get("return_date")
        cabin = request.args.get("cabin", "economy").lower()
        pax = int(request.args.get("pax", 1))
        wants_pet = request.args.get("pet", "false").lower() == "true"
        extra_bag_kg = int(request.args.get("extra_baggage", 0))

        if not dep_date:
            return jsonify({"error": "dep_date required"}), 400

        outbound = build_leg_result(origin, dest, dep_date, cabin, pax, wants_pet, extra_bag_kg)

        res = {"outbound": outbound}
        if ret_date:
            # Round trip: destination back to origin
            inbound = build_leg_result(dest, origin, ret_date, cabin, pax, wants_pet, extra_bag_kg)
            res["return"] = inbound

        return jsonify(res)
    except Exception as exc:
        print(f"Customer search failed: {exc}", flush=True)
        return jsonify({"error": "Customer search failed. Please retry."}), 500

@app.route("/api/routes")
def api_routes():
    """B5: Dynamically fetch available routes from simulation."""
    fallback_routes = list(SIM_ROUTE_FALLBACK)
    try:
        payload = _dashboard_get("/api/routes", timeout=5)
        if payload:
            live_routes = [route for route in payload.get("routes", []) if route]
            merged = []
            seen = set()
            for route in live_routes + fallback_routes:
                if route not in seen:
                    seen.add(route)
                    merged.append(route)
            return jsonify({"routes": merged, "source": "dashboard"})
    except Exception as e:
        print(f"Route sync fallback triggered: {e}")
    return jsonify({"routes": fallback_routes, "source": "fallback"})

@app.route("/api/price-calendar")
def api_price_calendar():
    """Monthly price overview — fetches live prices per day from simulation and caches them."""
    origin = request.args.get("origin", "IST").upper()
    dest = request.args.get("destination", "MAD").upper()
    cabin = request.args.get("cabin", "economy").lower()
    month = request.args.get("month")

    if not month:
        return jsonify({"error": "month required"}), 400

    y, m = month.split("-")

    # Check if simulation is currently running
    sim_status = get_sim_status()
    sim_running = bool(sim_status and sim_status.get("state") in ("running", "paused", "completed"))

    # Fallback synthetic pricing (only used if cache is empty AND sim is not running)
    from pricing_engine import BASE_PRICE_FORMULAS, SEASON_FACTORS, DOW_FACTORS
    dur = _get_duration(dest)
    dist_km = dur / 60.0 * 900
    base_formula = BASE_PRICE_FORMULAS.get(cabin, BASE_PRICE_FORMULAS["economy"])
    route_base = base_formula(dist_km)

    calendar = []
    cheapest_price = float("inf")
    cheapest_date = None

    start_date = datetime(int(y), int(m), 1)
    if int(m) == 12:
        end_date = datetime(int(y) + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(int(y), int(m) + 1, 1) - timedelta(days=1)

    d = start_date
    while d <= end_date:
        ds = d.strftime("%Y-%m-%d")
        k = f"{origin}-{dest}_{ds}_{cabin}"

        day_info = {
            "date": ds, "min_price": None,
            "thy": None, "pc": None, "ek": None,
            "pc_sold_out": False, "is_cheapest": False,
            "source": "synthetic"
        }

        # ── STEP 1: Try live simulation data ──────────────────────────────
        if sim_running:
            try:
                f_inv = _dashboard_get(f"/api/sim/flight/{k}", timeout=2)
                if f_inv and "error" not in f_inv:
                    prices = f_inv.get("current_prices", {})
                    open_fares = f_inv.get("fare_classes_open", [])
                    best_fare = open_fares[0] if open_fares else None
                    thy_p = prices.get(best_fare) if best_fare else None

                    # Try to get competitor prices from the same endpoint
                    comp_flight = _dashboard_get(f"/api/sim/competition/flight/{k}", timeout=2)
                    pc_p = None
                    ek_p = None
                    pc_sold = False
                    if comp_flight:
                        pc_info = comp_flight.get("PC", {})
                        ek_info = comp_flight.get("EK", {})
                        if isinstance(pc_info, dict):
                            pc_p = pc_info.get("price")
                            pc_sold = pc_info.get("sold_out", False)
                        if isinstance(ek_info, dict):
                            ek_p = ek_info.get("price")

                    if thy_p:
                        day_info["thy"] = thy_p
                        day_info["pc"] = pc_p
                        day_info["ek"] = ek_p
                        day_info["pc_sold_out"] = pc_sold
                        day_info["source"] = "simulation"

                        # ── Cache the live price for persistence ──────────────
                        PRICE_CACHE[k] = {
                            "thy": thy_p, "pc": pc_p, "ek": ek_p,
                            "pc_sold_out": pc_sold
                        }
            except Exception:
                pass  # Fall through to cache or synthetic

        # ── STEP 2: Use cached price if live fetch failed or sim is stopped ─
        if day_info["thy"] is None and k in PRICE_CACHE:
            cached = PRICE_CACHE[k]
            day_info["thy"] = cached.get("thy")
            day_info["pc"] = cached.get("pc")
            day_info["ek"] = cached.get("ek")
            day_info["pc_sold_out"] = cached.get("pc_sold_out", False)
            day_info["source"] = "cached"

        # ── STEP 3: Synthetic fallback only if no cache exists ─────────────
        if day_info["thy"] is None:
            day_info["thy"] = _get_synthetic_price(origin, dest, ds, cabin, "THY")
            day_info["pc"] = _get_synthetic_price(origin, dest, ds, cabin, "PC")
            day_info["ek"] = _get_synthetic_price(origin, dest, ds, cabin, "EK")
            day_info["source"] = "synthetic"

        ps = [p for p in (day_info["thy"], day_info["pc"], day_info["ek"]) if p is not None]
        if ps:
            day_info["min_price"] = min(ps)
            if day_info["min_price"] < cheapest_price:
                cheapest_price = day_info["min_price"]
                cheapest_date = ds

        calendar.append(day_info)
        d += timedelta(days=1)

    for day_info in calendar:
        if day_info["date"] == cheapest_date:
            day_info["is_cheapest"] = True

    return jsonify({"calendar": calendar, "cheapest_date": cheapest_date,
                    "cheapest_price": cheapest_price if cheapest_price < float("inf") else None,
                    "sim_live": sim_running})

@app.route("/api/flight-detail")
def api_flight_detail():
    """Price trend — sadece sim'de varsa gercek veri, yoksa rota bazli sentetik."""
    flight_key = request.args.get("flight_key")
    if not flight_key or flight_key == "demo":
        return jsonify({"history": [], "source": "no_data"})

    parts = flight_key.split("_")
    if len(parts) < 2:
        return jsonify({"history": [], "source": "invalid_key"})

    route = parts[0]  # e.g. IST-MAD
    dest = route.split("-")[1] if "-" in route else "MAD"

    # Gercek sim verisi dene
    try:
        flight_data = _dashboard_get(f"/api/sim/flight/{flight_key}")
        if flight_data and not flight_data.get("error"):
            price_history = flight_data.get("price_history", [])
            if price_history and len(price_history) > 3:
                history = []
                for entry in price_history:
                    dtd = entry.get("dtd", 0)
                    prices = entry.get("prices", {})
                    open_fares = entry.get("open_fares", ["Y"])
                    thy_p = prices.get(open_fares[0], 0) if open_fares else 0
                    history.append({
                        "dtd": dtd, "thy": thy_p,
                        "pc": thy_p * 0.80,
                        "ek": thy_p * 1.18,
                        "load_factor": entry.get("load_factor", 0),
                    })
                try:
                    comp_data = _dashboard_get(f"/api/sim/competition/flight/{flight_key}")
                    if comp_data:
                        for h in history:
                            if "PC" in comp_data and "price" in comp_data["PC"]:
                                h["pc"] = comp_data["PC"]["price"]
                            if "EK" in comp_data and "price" in comp_data["EK"]:
                                h["ek"] = comp_data["EK"]["price"]
                except Exception:
                    pass
                return jsonify({"history": history, "source": "simulation"})
    except Exception:
        pass

    # Rota bazli deterministik sentetik egri
    from pricing_engine import BASE_PRICE_FORMULAS, SEASON_FACTORS
    dur = _get_duration(dest)
    dist_km = dur / 60.0 * 900
    cabin = parts[2] if len(parts) >= 3 else "economy"
    base_formula = BASE_PRICE_FORMULAS.get(cabin, BASE_PRICE_FORMULAS["economy"])
    route_base = base_formula(dist_km)
    dep_month = int(parts[1].split("-")[1]) if len(parts) >= 2 and "-" in parts[1] else 7
    season_f = SEASON_FACTORS.get(dep_month, 1.0)
    base_p = route_base * season_f

    history = []
    for i in range(180, -1, -5):
        thy_p = base_p * (0.70 + 0.30 * (1 - i / 180.0)) + (base_p * 0.10 if i < 30 else 0)
        pc_p = thy_p * 0.82
        ek_p = thy_p * 1.18
        history.append({
            "dtd": i,
            "thy": round(thy_p, 2),
            "pc": round(pc_p, 2),
            "ek": round(ek_p, 2)
        })

    return jsonify({"history": history, "source": "estimated"})

@app.route("/api/sim-status")
def api_sim_status():
    """B1: Endpoint for auto-refresh polling — returns sim state."""
    status = get_sim_status()
    if status:
        return jsonify({
            "sim_running": status.get("state") in ("running", "paused", "completed"),
            "state": status.get("state"),
            "clock": status.get("clock", {}),
        })
    return jsonify({"sim_running": False, "state": "offline"})


@app.route("/api/user-info")
def api_user_info():
    """Oturumdaki kullanici bilgisini dondur."""
    user = session.get("bb_user")
    if user:
        return jsonify({"logged_in": True, "username": user.get("username", "User")})
    return jsonify({"logged_in": False})

@app.route("/api/book", methods=["POST"])
def api_book():
    """B3: Booking proxy — for the confirmation flow."""
    response = _dashboard_post("/api/pricing/book", request.json, timeout=4)
    if response is None:
        return jsonify({"error": "Could not connect to booking engine"}), 500
    return jsonify(response.json()), response.status_code
