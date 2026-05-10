"""
Seatwise Dashboard Dashboard — Sprint 4
Flask + DuckDB backend.
V2 parquet (ticket + ancillary revenue) + metadata lookup.
Demand forecast via two-stage XGBoost (classifier + regressor).
Sentiment Intelligence via HuggingFace NLP (integrated).
"""

import os
import math
import json
import threading
import numpy as np
from flask import Flask, render_template, jsonify, request, session, redirect, url_for
import duckdb
import shap

# ─── .env dosyasından ortam değişkenlerini oku ───────────
def _load_dotenv():
    # dashboard/.env ve project root/.env — ikisini de dene
    base = os.path.dirname(os.path.abspath(__file__))
    for env_path in [os.path.join(base, ".env"), os.path.join(os.path.dirname(base), ".env")]:
        if not os.path.exists(env_path):
            continue
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if not os.environ.get(key):
                    os.environ[key] = val

_load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SEATWISE_SECRET", "sw-desktop-secret-2026")


# Geliştirme sırasında tarayıcının HTML/JS template'lerini cache'lememesi için
@app.after_request
def _no_cache_html(response):
    ct = response.content_type or ""
    if ct.startswith("text/html") or ct.startswith("application/json"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# ─── AUTH HELPERS ─────────────────────────────────────────
import hashlib, uuid
from functools import wraps

SW_USERS_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users_db.json")

def _load_users():
    if not os.path.exists(SW_USERS_DB):
        return {"users": []}
    with open(SW_USERS_DB, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_users(db):
    with open(SW_USERS_DB, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def _hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("sw_user"):
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated

BASE_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PROJECT_DIR = os.path.dirname(BASE_DIR).replace("\\", "/")
DATA_DIR = f"{PROJECT_DIR}/data"

# ─── FILE SELECTION (v2 default, v1 fallback) ────────────
SNAPSHOT_V2 = f"{DATA_DIR}/raw/flight_snapshot_v2.parquet"
SNAPSHOT_V1 = f"{DATA_DIR}/raw/flight_snapshot.parquet"
METADATA_PATH = f"{DATA_DIR}/processed/flight_metadata.parquet"

if os.path.exists(SNAPSHOT_V2.replace("/", os.sep)):
    PARQUET_PATH = SNAPSHOT_V2
    USE_V2 = True
else:
    PARQUET_PATH = SNAPSHOT_V1
    USE_V2 = False

# ─── TFT DEMAND FORECAST (Route-Daily) ───────────────────
import pandas as pd
from datetime import date, datetime, timedelta

TFT_DATA_PATH = f"{DATA_DIR}/processed/tft_route_daily.parquet"
TFT_PRED_PATH = f"{DATA_DIR}/processed/tft_predictions_indexed.parquet"
FORECAST_READY = False
TFT_DATA = None
TFT_PRED = None
TFT_METRICS = {
    "test_mae": 14.03, "test_corr": 0.991,  # Kaggle test set (2026 Q2-Q4), 50 epoch
}

try:
    if os.path.exists(TFT_DATA_PATH.replace('/', os.sep)):
        TFT_DATA = pd.read_parquet(TFT_DATA_PATH.replace('/', os.sep))
        TFT_DATA["dep_date"] = pd.to_datetime(TFT_DATA["dep_date"])
        FORECAST_READY = True
        print(f"[Forecast] TFT route-daily loaded: {len(TFT_DATA):,} rows, "
              f"{TFT_DATA['entity_id'].nunique()} entities, MAE={TFT_METRICS['test_mae']}")
    else:
        print("[Forecast] tft_route_daily.parquet not found, forecast disabled")
    if os.path.exists(TFT_PRED_PATH.replace('/', os.sep)):
        TFT_PRED = pd.read_parquet(TFT_PRED_PATH.replace('/', os.sep))
        TFT_PRED["dep_date"] = pd.to_datetime(TFT_PRED["dep_date"])
        print(f"[Forecast] TFT predictions loaded: {len(TFT_PRED):,} rows, "
              f"{TFT_PRED['entity_id'].nunique()} entities")
    else:
        print("[Forecast] tft_predictions_indexed.parquet not found, using YoY baseline")
except Exception as e:
    print(f"[Forecast] Failed to load TFT data: {e}")


# ─── PICKUP XGBOOST (Flight-Level Demand) ─────────────────
import xgboost as xgb

PICKUP_MODEL_PATH = f"{DATA_DIR}/models/pickup_xgb.json"
PICKUP_FEATURES_PATH = f"{DATA_DIR}/models/pickup_feature_list.json"
PICKUP_MASTER_PATH = f"{DATA_DIR}/processed/pickup_master.parquet"
PICKUP_METRICS_PATH = f"{PROJECT_DIR}/reports/pickup_xgb_metrics.json"

PICKUP_READY = False
PICKUP_MODEL = None
PICKUP_FEATURES = None
PICKUP_METRICS = {}

try:
    if os.path.exists(PICKUP_MODEL_PATH.replace('/', os.sep)):
        PICKUP_MODEL = xgb.Booster()
        PICKUP_MODEL.load_model(PICKUP_MODEL_PATH.replace('/', os.sep))
        with open(PICKUP_FEATURES_PATH.replace('/', os.sep), 'r') as f:
            PICKUP_FEATURES = json.load(f)["features"]
        if os.path.exists(PICKUP_METRICS_PATH.replace('/', os.sep)):
            with open(PICKUP_METRICS_PATH.replace('/', os.sep), 'r') as f:
                PICKUP_METRICS = json.load(f)
        PICKUP_READY = True
        print(f"[Pickup] Model loaded: {len(PICKUP_FEATURES)} features, "
              f"MAE={PICKUP_METRICS.get('mae', '?')}, WAPE={PICKUP_METRICS.get('wape', '?')}%")
    else:
        print(f"[Pickup] pickup_xgb.json not found, pickup disabled")
except Exception as e:
    print(f"[Pickup] Failed to load: {e}")


# ─── TWO-STAGE XGBOOST (Daily Pax Sold) ──────────────────
import joblib

TWOSTAGE_CLF_PATH = f"{DATA_DIR}/models/xgb_demand_classifier.pkl"
TWOSTAGE_REG_PATH = f"{DATA_DIR}/models/xgb_demand_regressor.pkl"
TWOSTAGE_FEAT_PATH = f"{DATA_DIR}/models/feature_list.json"
TWOSTAGE_METRICS_PATH = f"{PROJECT_DIR}/reports/demand_metrics.json"

TWOSTAGE_READY = False
TWOSTAGE_CLF = None
TWOSTAGE_REG = None
TWOSTAGE_FEATURES = None
TWOSTAGE_METRICS = {}

try:
    if os.path.exists(TWOSTAGE_CLF_PATH.replace('/', os.sep)):
        TWOSTAGE_CLF = joblib.load(TWOSTAGE_CLF_PATH.replace('/', os.sep))
        TWOSTAGE_REG = joblib.load(TWOSTAGE_REG_PATH.replace('/', os.sep))
        with open(TWOSTAGE_FEAT_PATH.replace('/', os.sep), 'r') as f:
            TWOSTAGE_FEATURES = json.load(f)["features"]
        if os.path.exists(TWOSTAGE_METRICS_PATH.replace('/', os.sep)):
            with open(TWOSTAGE_METRICS_PATH.replace('/', os.sep), 'r') as f:
                TWOSTAGE_METRICS = json.load(f)
        TWOSTAGE_READY = True
        print(f"[TwoStage] Model loaded: {len(TWOSTAGE_FEATURES)} features, "
              f"MAE={TWOSTAGE_METRICS.get('two_stage_model', {}).get('mae', '?')}, "
              f"AUC={TWOSTAGE_METRICS.get('two_stage_model', {}).get('auc_sale_classifier', '?')}")
    else:
        print("[TwoStage] xgb_demand_classifier.pkl not found, two-stage disabled")
except Exception as e:
    print(f"[TwoStage] Failed to load: {e}")


# ─── SENTIMENT INTELLIGENCE v2 (Google News RSS + DeBERTa-v3 + Keyword Events) ────
SENTIMENT_READY = False
_SENT_CACHE = {"data": None, "loading": False, "last_update": None, "deberta_active": False}
try:
    from sentiment import CITIES as SENT_CITIES, AIRPORT_TO_CITY
    from sentiment import init_db as _sent_init_db, load_cached_scores, start_scheduler
    from sentiment import deberta_clf as _deberta_clf

    _sent_init_db()

    # SQLite'tan son skorlari yukle (aninda hazir)
    _cached = load_cached_scores(max_age_hours=72)
    if _cached:
        _SENT_CACHE["data"] = _cached
        print(f"[Sentiment] Loaded {len(_cached)} cities from cache")

    # Arka plan scheduler baslat (RSS/GDELT fetch + DeBERTa-v3 sentiment + keyword events)
    # DeBERTa lazy-load — ilk classify çağrısında indirilir, sonra arka planda backfill yapar.
    start_scheduler(_SENT_CACHE)
    SENTIMENT_READY = True
    print("[Sentiment] v2 ready (RSS + DeBERTa-v3-small + keyword events)")
except Exception as e:
    print(f"[Sentiment] Module not available: {e}")
    import traceback; traceback.print_exc()
    SENT_CITIES = {}
    _deberta_clf = None


def get_con():
    return duckdb.connect()


def _num(val):
    if val is None:
        return None
    try:
        if math.isnan(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def _english_event_label(event_key=None, fallback=None):
    if fallback:
        fallback_text = str(fallback).strip()
        if fallback_text and fallback_text.lower() not in {"veri yok", "data yok"}:
            return fallback_text

    raw = str(event_key or "").strip()
    if not raw:
        return "No Data"

    normalized = raw.replace("_", " ").strip()
    lookup = {
        "flight disruption": "Flight Disruption",
        "strike or labor dispute": "Strike / Labor Dispute",
        "weather disruption": "Weather Disruption",
        "security threat": "Security Threat",
        "tourism growth": "Tourism Growth",
        "new route or airline expansion": "New Route / Airline Expansion",
        "new route expansion": "New Route / Expansion",
        "airport congestion": "Airport Congestion",
        "general news": "General News",
        "general travel news": "General Travel News",
    }
    return lookup.get(normalized.lower(), normalized.title())


def _normalize_sentiment_city_payload(city_key, payload):
    cfg = SENT_CITIES.get(city_key, {})
    city_payload = dict(payload or {})
    aggregate = dict(city_payload.get("aggregate") or {})

    aggregate["dominant_event_tr"] = _english_event_label(
        aggregate.get("dominant_event"),
        aggregate.get("dominant_event_tr"),
    )
    aggregate["dominant_event_label"] = aggregate["dominant_event_tr"]

    normalized_impact_events = []
    for item in aggregate.get("high_impact_events") or []:
        row = dict(item)
        row["event_tr"] = _english_event_label(row.get("event_type"), row.get("event_tr"))
        row["event_label"] = row["event_tr"]
        normalized_impact_events.append(row)
    aggregate["high_impact_events"] = normalized_impact_events

    normalized_articles = []
    for item in city_payload.get("articles") or []:
        row = dict(item)
        row["event_tr"] = _english_event_label(row.get("event_type"), row.get("event_tr"))
        row["event_label"] = row["event_tr"]
        normalized_articles.append(row)

    city_payload.update({
        "city": city_key,
        "label": cfg.get("label", city_payload.get("label", city_key)),
        "flag": cfg.get("flag", city_payload.get("flag", "")),
        "color": cfg.get("color", city_payload.get("color", "#58a6ff")),
        "country": cfg.get("country", city_payload.get("country", "")),
        "aggregate": aggregate,
        "articles": normalized_articles,
    })
    return city_payload


# ─── PAGES ───────────────────────────────────────────────
@app.route("/")
def landing():
    if session.get("sw_user"):
        # Loading splash sayfasini bypass et — direkt dashboard'a git
        return redirect(url_for("index"))
    return redirect(url_for("login_page"))


@app.route("/dashboard")
@login_required
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET"])
def login_page():
    if session.get("sw_user"):
        return redirect(url_for("landing"))
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login_post():
    data = request.get_json(force=True) or {}
    identifier = data.get("username", "").strip()
    password = data.get("password", "").strip()
    if not identifier or not password:
        return jsonify({"success": False, "message": "Username and password are required."}), 400
    db = _load_users()
    pw_hash = _hash_pw(password)
    user = next((u for u in db["users"] if (u["username"] == identifier or u["email"] == identifier) and u["password_hash"] == pw_hash), None)
    if not user:
        return jsonify({"success": False, "message": "Invalid username or password."}), 401
    session["sw_user"] = {"id": user["id"], "username": user["username"], "email": user["email"]}
    return jsonify({"success": True, "redirect": "/"})


@app.route("/register", methods=["POST"])
def register_post():
    import re
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
    db = _load_users()
    if any(u["username"] == username for u in db["users"]):
        return jsonify({"success": False, "message": "This username is already taken."}), 409
    if any(u["email"] == email for u in db["users"]):
        return jsonify({"success": False, "message": "This email address is already registered."}), 409
    from datetime import datetime as _dt
    new_user = {"id": str(uuid.uuid4()), "username": username, "email": email, "password_hash": _hash_pw(password), "role": "manager", "created_at": _dt.now().isoformat()}
    db["users"].append(new_user)
    _save_users(db)
    return jsonify({"success": True, "message": "Registration successful."})


@app.route("/logout")
def logout():
    session.pop("sw_user", None)
    return redirect(url_for("login_page"))


# Token-based password reset
import secrets as _secrets
from datetime import timedelta as _timedelta

_reset_tokens = {}  # token -> {user_id, expires_at_iso}
_RESET_TOKEN_TTL_MINUTES = 15


def _purge_expired_reset_tokens():
    from datetime import datetime as _dt
    now = _dt.now()
    expired = [t for t, r in _reset_tokens.items()
               if _dt.fromisoformat(r["expires_at"]) < now]
    for t in expired:
        _reset_tokens.pop(t, None)


# ─── SMTP Email gönderimi (production-ready) ────────────────
def _smtp_configured():
    return all([
        os.environ.get("SMTP_HOST"),
        os.environ.get("SMTP_USER"),
        os.environ.get("SMTP_PASS"),
    ])


def _send_reset_email(to_email, reset_url, ttl_minutes):
    """Şifre sıfırlama linkini SMTP ile gönderir.
    .env'de SMTP_HOST/USER/PASS yoksa False döner (dev mode).
    Başarılıysa True, hata olursa False döner."""
    if not _smtp_configured():
        return False
    try:
        import smtplib, ssl
        from email.message import EmailMessage

        host = os.environ["SMTP_HOST"]
        port = int(os.environ.get("SMTP_PORT", "587"))
        user = os.environ["SMTP_USER"]
        password = os.environ["SMTP_PASS"]
        from_addr = os.environ.get("SMTP_FROM", user)
        from_name = os.environ.get("SMTP_FROM_NAME", "Seatwise")
        use_ssl = os.environ.get("SMTP_USE_SSL", "false").lower() in ("1", "true", "yes")

        msg = EmailMessage()
        msg["Subject"] = "Reset your Seatwise password"
        msg["From"] = f"{from_name} <{from_addr}>"
        msg["To"] = to_email

        text_body = (
            f"Hi,\n\n"
            f"We received a request to reset your Seatwise password.\n"
            f"Click the link below to set a new password (valid for {ttl_minutes} minutes):\n\n"
            f"{reset_url}\n\n"
            f"If you did not request this, you can safely ignore this email.\n\n"
            f"— Seatwise Team"
        )
        html_body = f"""<!DOCTYPE html>
<html><body style="font-family:Arial,sans-serif;background:#f4f6fa;padding:24px;color:#1f2937">
  <div style="max-width:520px;margin:0 auto;background:#fff;border-radius:12px;padding:32px;box-shadow:0 2px 12px rgba(0,0,0,.06)">
    <h2 style="margin:0 0 12px;color:#0b1525">Reset your password</h2>
    <p style="font-size:14px;line-height:1.6">We received a request to reset your <strong>Seatwise</strong> account password.</p>
    <p style="text-align:center;margin:28px 0">
      <a href="{reset_url}" style="background:linear-gradient(135deg,#1a73e8,#14b8a6);color:#fff;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:700;display:inline-block">Set new password</a>
    </p>
    <p style="font-size:12px;color:#6b7280">Or paste this URL in your browser:<br><span style="word-break:break-all;color:#1a73e8">{reset_url}</span></p>
    <p style="font-size:12px;color:#6b7280;margin-top:24px">This link expires in {ttl_minutes} minutes. If you didn't request this, you can safely ignore this email.</p>
    <hr style="border:none;border-top:1px solid #e5e7eb;margin:24px 0">
    <p style="font-size:11px;color:#9ca3af;margin:0">Seatwise · Automated message — please do not reply.</p>
  </div>
</body></html>"""
        msg.set_content(text_body)
        msg.add_alternative(html_body, subtype="html")

        if use_ssl or port == 465:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, context=ctx, timeout=15) as s:
                s.login(user, password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=15) as s:
                s.ehlo()
                s.starttls(context=ssl.create_default_context())
                s.ehlo()
                s.login(user, password)
                s.send_message(msg)
        return True
    except Exception as e:
        try:
            print(f"[Reset] SMTP send failed: {e}", flush=True)
        except UnicodeEncodeError:
            pass
        return False


@app.route("/forgot-password", methods=["POST"])
def forgot_password():
    """E-posta ile reset link talebi.
    Email enumeration korumasi: kullanici varolsa da olmasa da ayni mesaj doner.
    Email gercekten varsa kisa sureli (15 dk) token uretilir.
    Production: email ile gonderilmesi gerek. Dev: response'a token URL'i konur."""
    from datetime import datetime as _dt
    _purge_expired_reset_tokens()

    data = request.get_json(force=True) or {}
    email = (data.get("email") or "").strip().lower()

    if not email:
        return jsonify({"success": False, "message": "Email is required."}), 400

    response = {
        "success": True,
        "message": "If an account exists with that email, a password reset link has been generated.",
    }

    db = _load_users()
    user = next((u for u in db["users"] if (u.get("email") or "").lower() == email), None)

    if user:
        # Ayni kullanici icin onceki tokenlari iptal et
        for tok in list(_reset_tokens.keys()):
            if _reset_tokens[tok]["user_id"] == user["id"]:
                _reset_tokens.pop(tok, None)
        token = _secrets.token_urlsafe(32)
        _reset_tokens[token] = {
            "user_id": user["id"],
            "expires_at": (_dt.now() + _timedelta(minutes=_RESET_TOKEN_TTL_MINUTES)).isoformat(),
        }
        path = f"/reset/{token}"
        # Public URL: APP_BASE_URL set edilmişse onu kullan, yoksa istek host'undan üret
        base = (os.environ.get("APP_BASE_URL") or "").rstrip("/")
        if not base:
            base = request.host_url.rstrip("/")
        full_url = f"{base}{path}"

        sent = _send_reset_email(email, full_url, _RESET_TOKEN_TTL_MINUTES)
        if sent:
            response["delivery"] = "email"
            response["message"] = "If an account exists with that email, a password reset link has been emailed."
        else:
            # SMTP yapilandirilmamis ya da gonderim basarisiz: dev mode link
            response["delivery"] = "dev_link"
            response["dev_reset_url"] = path
        response["expires_in_minutes"] = _RESET_TOKEN_TTL_MINUTES
        try:
            print(f"[Reset] Token issued for {email} -> {path} (expires in {_RESET_TOKEN_TTL_MINUTES} min, delivery={response['delivery']})", flush=True)
        except UnicodeEncodeError:
            pass

    return jsonify(response)


@app.route("/reset/<token>", methods=["GET"])
def reset_page(token):
    from datetime import datetime as _dt
    _purge_expired_reset_tokens()
    rec = _reset_tokens.get(token)
    error = None
    valid = False
    if not rec:
        error = "Invalid or already-used reset link."
    elif _dt.fromisoformat(rec["expires_at"]) < _dt.now():
        _reset_tokens.pop(token, None)
        error = "This reset link has expired. Please request a new one."
    else:
        valid = True
    return render_template("reset.html", token=token, error=error, valid=valid)


@app.route("/reset/<token>", methods=["POST"])
def reset_submit(token):
    import re as _re
    from datetime import datetime as _dt

    _purge_expired_reset_tokens()
    rec = _reset_tokens.get(token)
    if not rec:
        return jsonify({"success": False, "message": "Invalid or expired reset link."}), 400
    if _dt.fromisoformat(rec["expires_at"]) < _dt.now():
        _reset_tokens.pop(token, None)
        return jsonify({"success": False, "message": "Reset link expired."}), 400

    data = request.get_json(force=True) or {}
    new_password = (data.get("new_password") or "").strip()

    if not _re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&_\-.])[A-Za-z\d@$!%*?&_\-.]{8,32}$', new_password):
        return jsonify({"success": False, "message": "Password must be 8-32 chars with uppercase, lowercase, number and special character."}), 400

    db = _load_users()
    user = next((u for u in db["users"] if u["id"] == rec["user_id"]), None)
    if not user:
        _reset_tokens.pop(token, None)
        return jsonify({"success": False, "message": "Account not found."}), 404

    user["password_hash"] = _hash_pw(new_password)
    user["updated_at"] = _dt.now().isoformat()
    _save_users(db)
    _reset_tokens.pop(token, None)
    return jsonify({"success": True, "message": "Password updated. You can sign in now."})


@app.route("/api/me")
def api_me():
    """Currently logged-in user info (for account menu)."""
    user = session.get("sw_user")
    if not user:
        return jsonify({"authenticated": False}), 200
    return jsonify({
        "authenticated": True,
        "id": user.get("id"),
        "username": user.get("username"),
        "email": user.get("email"),
    })


# ─── SEARCH API ──────────────────────────────────────────
@app.route("/api/flights")
def api_flights():
    """Uçuş numarası veya havaalanı koduna göre arama (metadata'dan)."""
    q = request.args.get("q", "").strip().upper()
    con = get_con()
    meta = METADATA_PATH
    results = []

    if q:
        fn_rows = con.execute(f"""
            SELECT DISTINCT COALESCE(flight_number, SPLIT_PART(flight_id, '_', 1)) AS fn
            FROM read_parquet('{meta}')
            WHERE UPPER(COALESCE(flight_number, flight_id)) LIKE '%' || $1 || '%'
            ORDER BY fn
            LIMIT 15
        """, [q]).fetchall()
        for r in fn_rows:
            results.append({"type": "flight", "value": r[0], "label": r[0]})

        apt_rows = con.execute(f"""
            SELECT DISTINCT airport FROM (
                SELECT departure_airport AS airport FROM read_parquet('{meta}')
                WHERE departure_airport IS NOT NULL
                UNION
                SELECT arrival_airport AS airport FROM read_parquet('{meta}')
                WHERE arrival_airport IS NOT NULL
            )
            WHERE UPPER(airport) LIKE '%' || $1 || '%'
            ORDER BY airport
            LIMIT 10
        """, [q]).fetchall()
        for r in apt_rows:
            results.append({"type": "airport", "value": r[0],
                            "label": f"{r[0]} — All Flights"})
    else:
        # Boş arama: keşif odaklı — popüler havalimanları + rastgele örnek uçuşlar
        apt_rows = con.execute(f"""
            SELECT airport, SUM(cnt) AS total
            FROM (
                SELECT departure_airport AS airport, COUNT(*) AS cnt
                FROM read_parquet('{meta}')
                WHERE departure_airport IS NOT NULL
                GROUP BY 1
                UNION ALL
                SELECT arrival_airport AS airport, COUNT(*) AS cnt
                FROM read_parquet('{meta}')
                WHERE arrival_airport IS NOT NULL
                GROUP BY 1
            )
            GROUP BY airport
            ORDER BY total DESC
            LIMIT 8
        """).fetchall()
        for r in apt_rows:
            results.append({
                "type": "airport",
                "value": r[0],
                "label": f"{r[0]} — All Flights",
                "section": "popular",
                "count": int(r[1]) if r[1] else 0,
            })

        fn_rows = con.execute(f"""
            SELECT DISTINCT COALESCE(flight_number, SPLIT_PART(flight_id, '_', 1)) AS fn
            FROM read_parquet('{meta}')
            WHERE flight_number IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 8
        """).fetchall()
        for r in fn_rows:
            results.append({
                "type": "flight",
                "value": r[0],
                "label": r[0],
                "section": "sample",
            })

    con.close()
    return jsonify(results)


@app.route("/api/airport/<airport_code>")
def api_airport_flights(airport_code):
    """Bir havalimanı kodundan kalkan/varan uçuşların paginated listesi.

    Query params:
      ?from=YYYY-MM-DD       (opt) tarih başlangıç
      ?to=YYYY-MM-DD         (opt) tarih bitiş
      ?route=IST-LHR         (opt) tek rota filtresi
      ?page=1                (opt) sayfa numarası (1-bazlı)
      ?page_size=50          (opt) sayfa başına kayıt (10–200)
    """
    code = airport_code.upper().strip()
    date_from = (request.args.get("from") or "").strip()
    date_to = (request.args.get("to") or "").strip()
    route_filter = (request.args.get("route") or "").strip().upper()
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    try:
        page_size = max(10, min(200, int(request.args.get("page_size", 50))))
    except ValueError:
        page_size = 50
    offset = (page - 1) * page_size

    # WHERE parts
    where_parts = ["(departure_airport = $1 OR arrival_airport = $1)", "flight_number IS NOT NULL"]
    params = [code]
    if date_from:
        params.append(date_from)
        where_parts.append(f"CAST(departure_datetime AS DATE) >= CAST(${len(params)} AS DATE)")
    if date_to:
        params.append(date_to)
        where_parts.append(f"CAST(departure_datetime AS DATE) <= CAST(${len(params)} AS DATE)")
    if route_filter and '-' in route_filter:
        from_apt, to_apt = route_filter.split('-', 1)
        params.append(from_apt.strip())
        where_parts.append(f"departure_airport = ${len(params)}")
        params.append(to_apt.strip())
        where_parts.append(f"arrival_airport = ${len(params)}")
    where_clause = " AND ".join(where_parts)

    con = get_con()

    # Toplam (pagination için)
    total = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT flight_number, departure_airport, arrival_airport, departure_datetime
            FROM read_parquet('{METADATA_PATH}')
            WHERE {where_clause}
        )
    """, params).fetchone()[0]

    # Sayfa içeriği — DISTINCT + ORDER BY tarih DESC
    flights = con.execute(f"""
        SELECT DISTINCT flight_number, departure_airport, arrival_airport, departure_datetime
        FROM read_parquet('{METADATA_PATH}')
        WHERE {where_clause}
        ORDER BY departure_datetime DESC, departure_airport, arrival_airport, flight_number
        LIMIT {int(page_size)} OFFSET {int(offset)}
    """, params).fetchall()

    # Tum rotalarin ozeti (tarih filtresinden bagimsiz)
    routes = con.execute(f"""
        SELECT departure_airport, arrival_airport, COUNT(*) AS cnt
        FROM (
            SELECT DISTINCT flight_number, departure_airport, arrival_airport
            FROM read_parquet('{METADATA_PATH}')
            WHERE (departure_airport = $1 OR arrival_airport = $1)
              AND flight_number IS NOT NULL
        )
        GROUP BY departure_airport, arrival_airport
        ORDER BY cnt DESC
    """, [code]).fetchall()

    # Tarih aralığı bilgisi (date picker için min/max)
    date_range = con.execute(f"""
        SELECT MIN(CAST(departure_datetime AS DATE)), MAX(CAST(departure_datetime AS DATE))
        FROM read_parquet('{METADATA_PATH}')
        WHERE (departure_airport = $1 OR arrival_airport = $1)
    """, [code]).fetchone()
    con.close()

    total_pages = max(1, (total + page_size - 1) // page_size)

    return jsonify({
        "flights": [{
            "flight_number": r[0],
            "departure_airport": r[1],
            "arrival_airport": r[2],
            "departure_datetime": str(r[3]) if r[3] else None,
        } for r in flights],
        "routes": [{
            "from": r[0],
            "to": r[1],
            "count": int(r[2]) if r[2] else 0,
        } for r in routes],
        "date_range": {
            "min": str(date_range[0]) if date_range and date_range[0] else None,
            "max": str(date_range[1]) if date_range and date_range[1] else None,
        },
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": int(total),
            "total_pages": total_pages,
        },
    })


@app.route("/api/flight/<flight_number>")
def api_flight_dates(flight_number):
    con = get_con()
    rows = con.execute(f"""
        SELECT DISTINCT
            flight_id,
            departure_datetime,
            departure_airport,
            arrival_airport,
            region
        FROM read_parquet('{METADATA_PATH}')
        WHERE COALESCE(flight_number, SPLIT_PART(flight_id, '_', 1)) = $1
        ORDER BY departure_datetime DESC
    """, [flight_number]).fetchall()
    con.close()
    return jsonify([{
        "flight_id": r[0],
        "departure_datetime": str(r[1]) if r[1] else None,
        "departure_airport": r[2],
        "arrival_airport": r[3],
        "region": r[4],
    } for r in rows])


@app.route("/api/flights/date")
def api_flights_by_date():
    """Return all flights on a given departure date."""
    date_str = request.args.get("date", "").strip()
    if not date_str:
        return jsonify([])
    con = get_con()
    rows = con.execute(f"""
        SELECT DISTINCT
            flight_id,
            flight_number,
            departure_airport,
            arrival_airport,
            departure_datetime,
            region
        FROM read_parquet('{METADATA_PATH}')
        WHERE CAST(departure_datetime AS DATE) = CAST($1 AS DATE)
        ORDER BY departure_datetime, flight_number
    """, [date_str]).fetchall()
    con.close()
    return jsonify([{
        "flight_id": r[0],
        "flight_number": r[1],
        "departure_airport": r[2],
        "arrival_airport": r[3],
        "departure_datetime": str(r[4]) if r[4] else None,
        "region": r[5],
    } for r in rows])


# ─── SNAPSHOT API ────────────────────────────────────────
@app.route("/api/snapshot/<path:flight_id>")
def api_snapshot(flight_id):
    cabin = request.args.get("cabin", "").strip()
    con = get_con()

    cabin_filter = ""
    params = [flight_id]
    if cabin:
        cabin_filter = "AND LOWER(s.cabin_class) = LOWER($2)"
        params.append(cabin)

    if USE_V2:
        # V2: JOIN snapshot with metadata
        rows = con.execute(f"""
            SELECT
                s.flight_id,
                s.cabin_class,
                s.dtd,
                s.pax_sold_today,
                s.pax_sold_cum,
                s.pax_last_7d,
                s.ticket_rev_today,
                s.ticket_rev_cum,
                s.anc_rev_today,
                s.anc_rev_cum,
                s.ticket_rev_today + s.anc_rev_today AS total_rev_today,
                s.ticket_rev_cum + s.anc_rev_cum AS total_rev_cum,
                m.flight_number,
                m.departure_airport,
                m.arrival_airport,
                m.departure_datetime,
                m.region,
                m.distance_km,
                m.flight_time_min,
                m.capacity,
                s.ff_gold_pct,
                s.ff_elite_pct
            FROM read_parquet('{PARQUET_PATH}') s
            LEFT JOIN read_parquet('{METADATA_PATH}') m
              ON s.flight_id = m.flight_id AND s.cabin_class = m.cabin_class
            WHERE s.flight_id = $1 {cabin_filter}
            ORDER BY s.cabin_class, s.dtd DESC
        """, params).fetchall()
    else:
        # V1 fallback: revenue_today/cum = ticket, anc = 0
        rows = con.execute(f"""
            SELECT
                flight_id,
                cabin_class,
                dtd,
                pax_sold_today,
                pax_sold_cum,
                pax_last_7d,
                revenue_today AS ticket_rev_today,
                revenue_cum AS ticket_rev_cum,
                0 AS anc_rev_today,
                0 AS anc_rev_cum,
                revenue_today AS total_rev_today,
                revenue_cum AS total_rev_cum,
                flight_number,
                departure_airport,
                arrival_airport,
                departure_datetime,
                region,
                distance_km,
                flight_time_min,
                capacity,
                NULL AS ff_gold_pct,
                NULL AS ff_elite_pct
            FROM read_parquet('{PARQUET_PATH}')
            WHERE flight_id = $1 {cabin_filter.replace('s.cabin_class','cabin_class')}
            ORDER BY cabin_class, dtd DESC
        """, params).fetchall()

    # Cabin classes
    cabins = con.execute(f"""
        SELECT DISTINCT cabin_class
        FROM read_parquet('{PARQUET_PATH}')
        WHERE flight_id = $1 AND cabin_class IS NOT NULL
    """, [flight_id]).fetchall()

    con.close()

    data = []
    for r in rows:
        capacity = _num(r[19])
        pax_cum = _num(r[4])
        remaining = max(int(capacity) - int(pax_cum), 0) if capacity is not None and pax_cum is not None else None
        load_factor = (pax_cum / capacity) if capacity is not None and capacity > 0 and pax_cum is not None else None
        ticket_today = _num(r[6])
        pax_today = _num(r[3])
        avg_fare = (ticket_today / pax_today) if ticket_today is not None and pax_today is not None and pax_today > 0 else None

        total_rev_cum = _num(r[11])
        anc_rev_cum = _num(r[9])
        anc_share = (anc_rev_cum / total_rev_cum) if anc_rev_cum is not None and total_rev_cum is not None and total_rev_cum > 0 else None

        data.append({
            "flight_id": r[0],
            "cabin_class": r[1],
            "dtd": r[2],
            "pax_sold_today": _num(r[3]),
            "pax_sold_cum": pax_cum,
            "pax_last_7d": _num(r[5]),
            "ticket_rev_today": ticket_today,
            "ticket_rev_cum": _num(r[7]),
            "anc_rev_today": _num(r[8]),
            "anc_rev_cum": anc_rev_cum,
            "total_rev_today": _num(r[10]),
            "total_rev_cum": total_rev_cum,
            "ancillary_share": _num(anc_share),
            "capacity": capacity,
            "remaining_seats": remaining,
            "load_factor": _num(load_factor),
            "avg_fare_today": _num(avg_fare),
            "flight_number": r[12],
            "departure_airport": r[13],
            "arrival_airport": r[14],
            "departure_datetime": str(r[15]) if r[15] else None,
            "region": r[16],
            "distance_km": _num(r[17]),
            "flight_time_min": _num(r[18]),
            "ff_gold_pct": _num(r[20]),
            "ff_elite_pct": _num(r[21]),
        })

    # Summary
    summary = {}
    if data:
        meta = data[0]
        summary = {
            "flight_number": meta["flight_number"],
            "departure_airport": meta["departure_airport"],
            "arrival_airport": meta["arrival_airport"],
            "departure_datetime": meta["departure_datetime"],
            "region": meta["region"],
            "distance_km": meta["distance_km"],
            "flight_time_min": meta["flight_time_min"],
        }

    return jsonify({
        "summary": summary,
        "cabins": [c[0] for c in cabins],
        "rows": data,
        "use_v2": USE_V2,
    })


# ─── FORECAST API ─────────────────────────────────────────
@app.route("/api/forecast/<path:flight_id>")
def api_forecast(flight_id):
    """TFT route-daily demand forecast for a flight.
    Maps flight_id -> route+cabin, returns daily demand time series
    with actual (current year) and seasonal baseline (previous year).
    """
    if not FORECAST_READY:
        return jsonify({"error": "Forecast model not loaded"}), 503

    cabin = request.args.get("cabin", "").strip().lower() or None
    window = int(request.args.get("window", 60))  # days around dep_date

    con = get_con()

    # ── 1. Look up flight metadata to get route + dep_date ──
    meta_rows = con.execute(f"""
        SELECT DISTINCT
            departure_airport, arrival_airport, departure_datetime,
            cabin_class, capacity, region, distance_km, flight_time_min,
            flight_number
        FROM read_parquet('{METADATA_PATH}')
        WHERE flight_id = $1
    """, [flight_id]).fetchall()
    con.close()

    if not meta_rows:
        return jsonify({"error": "Flight not found", "rows": []})

    # Pick cabin: use requested or first available
    meta_cols = ["dep_apt", "arr_apt", "dep_dt", "cabin_class", "capacity",
                 "region", "distance_km", "flight_time_min", "flight_number"]
    metas = [dict(zip(meta_cols, r)) for r in meta_rows]

    if cabin:
        metas = [m for m in metas if m["cabin_class"].lower() == cabin]
    if not metas:
        return jsonify({"error": "Cabin not found", "rows": []})

    cabins_result = {}

    for meta in metas:
        cab = meta["cabin_class"].lower()
        route = f"{meta['dep_apt']}_{meta['arr_apt']}"
        entity_id = f"{route}_{cab}"
        dep_dt = meta["dep_dt"]
        if isinstance(dep_dt, str):
            dep_dt = datetime.fromisoformat(dep_dt)
        dep_date = pd.Timestamp(dep_dt).normalize()
        cap = meta["capacity"] or (300 if cab == "economy" else 49)

        # ── 2. Get route demand data from TFT dataset ──
        entity_data = TFT_DATA[TFT_DATA["entity_id"] == entity_id].copy()
        if entity_data.empty:
            continue

        # Window: [dep_date - window, dep_date] — only show days up to departure
        date_start = dep_date - pd.Timedelta(days=window)
        date_end = dep_date
        windowed = entity_data[
            (entity_data["dep_date"] >= date_start) &
            (entity_data["dep_date"] <= date_end)
        ].sort_values("dep_date")

        # ── 3a. TFT Model Predictions (varsa) ──
        tft_lookup = {}
        tft_q10_lookup = {}
        tft_q90_lookup = {}
        has_tft = False
        if TFT_PRED is not None:
            pred_entity = TFT_PRED[TFT_PRED["entity_id"] == entity_id]
            if not pred_entity.empty:
                pred_window = pred_entity[
                    (pred_entity["dep_date"] >= date_start) &
                    (pred_entity["dep_date"] <= date_end)
                ]
                for _, pr in pred_window.iterrows():
                    ds = pr["dep_date"].strftime("%Y-%m-%d")
                    tft_lookup[ds] = float(pr["predicted"])
                    if "pred_q10" in pr and pd.notna(pr["pred_q10"]):
                        tft_q10_lookup[ds] = float(pr["pred_q10"])
                    if "pred_q90" in pr and pd.notna(pr["pred_q90"]):
                        tft_q90_lookup[ds] = float(pr["pred_q90"])
                has_tft = len(tft_lookup) > 0

        # ── 3b. YoY Baseline (yedek) ──
        baseline_start = date_start - pd.Timedelta(days=365)
        baseline_end = date_end - pd.Timedelta(days=365)
        baseline = entity_data[
            (entity_data["dep_date"] >= baseline_start) &
            (entity_data["dep_date"] <= baseline_end)
        ].sort_values("dep_date")

        baseline_lookup = {}
        for _, row in baseline.iterrows():
            shifted = row["dep_date"] + pd.Timedelta(days=365)
            baseline_lookup[shifted.strftime("%Y-%m-%d")] = float(row["total_pax"])

        # ── 4. Build response rows ──
        rows = []
        total_actual = 0
        total_forecast = 0
        dep_date_pax = None

        for _, row in windowed.iterrows():
            d = row["dep_date"]
            date_str = d.strftime("%Y-%m-%d")
            actual_pax = float(row["total_pax"])
            n_fl = int(row["n_flights"]) if row["n_flights"] else 1

            tft_pred = tft_lookup.get(date_str)
            yoy_base = baseline_lookup.get(date_str)
            forecast_pax = tft_pred if tft_pred is not None else yoy_base

            pax_per_flight = actual_pax / n_fl if n_fl > 0 else actual_pax
            load_factor = pax_per_flight / cap if cap > 0 else 0

            row_data = {
                "date": date_str,
                "actual_pax": round(actual_pax, 1),
                "forecast_pax": round(forecast_pax, 1) if forecast_pax is not None else None,
                "tft_forecast": round(tft_pred, 1) if tft_pred is not None else None,
                "pred_q10": round(tft_q10_lookup[date_str], 1) if date_str in tft_q10_lookup else None,
                "pred_q90": round(tft_q90_lookup[date_str], 1) if date_str in tft_q90_lookup else None,
                "yoy_baseline": round(yoy_base, 1) if yoy_base is not None else None,
                "pax_per_flight": round(pax_per_flight, 1),
                "load_factor": round(load_factor, 4),
                "n_flights": n_fl,
                "avg_fare": round(float(row["avg_fare"]), 2) if row["avg_fare"] else None,
                "is_special": bool(row["is_special_period"]),
                "special_period": row["special_period"] if row["special_period"] != "none" else None,
            }
            rows.append(row_data)

            total_actual += actual_pax
            if forecast_pax is not None:
                total_forecast += forecast_pax

            if date_str == dep_date.strftime("%Y-%m-%d"):
                dep_date_pax = actual_pax

        # ── 5. Compute accuracy metrics ──
        paired = [(r["actual_pax"], r["forecast_pax"])
                  for r in rows if r["forecast_pax"] is not None]
        if paired:
            actuals_arr = np.array([p[0] for p in paired])
            forecasts_arr = np.array([p[1] for p in paired])
            window_mae = float(np.mean(np.abs(actuals_arr - forecasts_arr)))
            sum_act = float(np.sum(np.abs(actuals_arr)))
            window_wape = float(np.sum(np.abs(actuals_arr - forecasts_arr)) / sum_act * 100) if sum_act > 0 else None
        else:
            window_mae = None
            window_wape = None

        cabins_result[cab] = {
            "rows": rows,
            "metadata": {
                "entity_id": entity_id,
                "route": route,
                "cabin_class": cab,
                "flight_id": flight_id,
                "flight_number": meta["flight_number"],
                "dep_date": dep_date.strftime("%Y-%m-%d"),
                "region": meta["region"],
                "distance_km": meta["distance_km"],
                "flight_time_min": meta["flight_time_min"],
                "capacity_per_flight": cap,
                "dep_date_pax": dep_date_pax,
                "dep_date_pax_per_flight": round(dep_date_pax / (rows[0]["n_flights"] if rows else 1), 1) if dep_date_pax else None,
                "dep_date_load_factor": round((dep_date_pax / (rows[0]["n_flights"] if rows else 1)) / cap, 4) if dep_date_pax and cap else None,
                "window_days": window,
                "total_actual": round(total_actual, 1),
                "total_forecast": round(total_forecast, 1),
                "forecast_source": "tft_model" if has_tft else "yoy_baseline",
                "window_mae": round(window_mae, 2) if window_mae is not None else None,
                "window_wape": round(window_wape, 1) if window_wape is not None else None,
                "tft_test_mae": TFT_METRICS.get("test_mae"),
                "tft_test_corr": TFT_METRICS.get("test_corr"),
            }
        }

    if not cabins_result:
        return jsonify({"error": "No route data found for this flight"})

    # If single cabin requested, return flat; otherwise return all
    if cabin and cabin in cabins_result:
        return jsonify(cabins_result[cabin])
    return jsonify(cabins_result)


# ─── TWO-STAGE DEMAND FORECAST API ────────────────────────
@app.route("/api/demand/<path:flight_id>")
def api_demand(flight_id):
    """Two-Stage XGBoost: predict daily pax sold at each DTD for a flight.
    Stage 1 (classifier): P(sale > 0)
    Stage 2 (regressor): E[pax | sale > 0]
    Combined: P * E[pax]
    """
    if not TWOSTAGE_READY:
        return jsonify({"error": "Two-stage demand model not loaded"}), 503

    cabin = request.args.get("cabin", "").strip().lower() or None
    con = get_con()

    meta_rows = con.execute(f"""
        SELECT departure_airport, arrival_airport, departure_datetime,
               cabin_class, capacity, region, distance_km, flight_time_min,
               flight_number
        FROM read_parquet('{METADATA_PATH}')
        WHERE flight_id = $1
        LIMIT 2
    """, [flight_id]).fetchall()
    con.close()

    if not meta_rows:
        return jsonify({"error": "Flight not found"})

    result = {}

    for row in meta_rows:
        cab = row[3].lower()
        if cabin and cab != cabin:
            continue
        cap = int(row[4]) if row[4] else (300 if cab == "economy" else 49)
        dep_dt = row[2]
        dep_year = dep_dt.year if hasattr(dep_dt, 'year') else int(str(dep_dt)[:4])
        region = row[5] or "Europe"
        dist_km = float(row[6]) if row[6] else 3000
        ft_min = float(row[7]) if row[7] else 300

        # Read snapshot data for this flight
        snap_con = get_con()
        snap_data = snap_con.execute(f"""
            SELECT dtd, pax_sold_cum, pax_last_7d, ff_gold_pct, ff_elite_pct
            FROM read_parquet('{PARQUET_PATH}')
            WHERE flight_id = $1 AND LOWER(cabin_class) = $2
            ORDER BY dtd DESC
        """, [flight_id, cab]).fetchall()
        snap_con.close()

        if not snap_data:
            continue

        # Build feature vectors for each DTD point
        rows_out = []
        for snap in snap_data:
            dtd_val = float(snap[0]) if snap[0] is not None else 0
            pax_cum = float(snap[1]) if snap[1] is not None else 0
            pax_7d = float(snap[2]) if snap[2] is not None else 0
            ff_gold = float(snap[3]) if snap[3] is not None else 0
            ff_elite = float(snap[4]) if snap[4] is not None else 0
            remaining = max(cap - pax_cum, 0)
            lf = pax_cum / cap if cap > 0 else 0

            # DTD bucket (0-6 encoding)
            dtd_bucket = min(int(dtd_val // 30), 6)

            # Build feature dict matching TWOSTAGE_FEATURES order
            feat = {
                'dtd': dtd_val,
                'pax_sold_cum': pax_cum,
                'pax_last_7d': pax_7d,
                'capacity': float(cap),
                'remaining_seats': remaining,
                'load_factor': lf,
                'distance_km': dist_km,
                'flight_time_min': ft_min,
                'dep_year': float(dep_year),
                'dep_month': float(dep_dt.month) if hasattr(dep_dt, 'month') else 1.0,
                'dep_dow': float(dep_dt.weekday()) if hasattr(dep_dt, 'weekday') else 0.0,
                'dep_hour': float(dep_dt.hour) if hasattr(dep_dt, 'hour') else 12.0,
                'ff_gold_pct': ff_gold,
                'ff_elite_pct': ff_elite,
                'cabin_class_business': 1.0 if cab == 'business' else 0.0,
                'cabin_class_economy': 1.0 if cab == 'economy' else 0.0,
                'cabin_class_nan': 0.0,
                'region_Africa': 1.0 if region == 'Africa' else 0.0,
                'region_Americas': 1.0 if region == 'Americas' else 0.0,
                'region_Asia': 1.0 if region == 'Asia' else 0.0,
                'region_Europe': 1.0 if region == 'Europe' else 0.0,
                'region_Middle East': 1.0 if region == 'Middle East' else 0.0,
                'region_nan': 0.0,
            }
            # DTD bucket one-hot
            for b in range(7):
                feat[f'dtd_bucket_{float(b)}'] = 1.0 if dtd_bucket == b else 0.0
            feat['dtd_bucket_nan'] = 0.0

            X = np.array([[feat.get(f, 0.0) for f in TWOSTAGE_FEATURES]], dtype=np.float32)

            # Stage 1: P(sale > 0)
            p_sale = float(TWOSTAGE_CLF.predict_proba(X)[0, 1])
            # Stage 2: E[pax | sale > 0]
            e_pax = max(float(TWOSTAGE_REG.predict(X)[0]), 0)
            # Combined
            predicted_pax = round(p_sale * e_pax, 2)

            rows_out.append({
                "dtd": int(dtd_val),
                "pax_sold_cum": int(pax_cum),
                "load_factor": round(lf, 4),
                "p_sale": round(p_sale, 4),
                "e_pax_given_sale": round(e_pax, 2),
                "predicted_daily_pax": predicted_pax,
                "remaining_seats": int(remaining),
            })

        # Model metrics
        ts_metrics = TWOSTAGE_METRICS.get("two_stage_model", {})

        result[cab] = {
            "rows": rows_out,
            "metadata": {
                "flight_id": flight_id,
                "flight_number": row[8],
                "cabin_class": cab,
                "departure_airport": row[0],
                "arrival_airport": row[1],
                "dep_date": str(dep_dt)[:10],
                "region": region,
                "capacity": cap,
                "model": "Two-Stage XGBoost (Classifier + Regressor)",
                "model_mae": ts_metrics.get("mae"),
                "model_auc": ts_metrics.get("auc_sale_classifier"),
            }
        }

    if not result:
        return jsonify({"error": "No data found"})
    if cabin and cabin in result:
        return jsonify(result[cabin])
    return jsonify(result)


# ─── PICKUP FORECAST API ─────────────────────────────────
@app.route("/api/pickup/<path:flight_id>")
def api_pickup(flight_id):
    """XGBoost pickup model: predict remaining_pax at each DTD for a flight.
    2025 flights: return actual data only.
    2026 flights: return model predictions.
    """
    if not PICKUP_READY:
        return jsonify({"error": "Pickup model not loaded"}), 503

    cabin = request.args.get("cabin", "").strip().lower() or None
    con = get_con()

    # Get flight metadata
    meta_row = con.execute(f"""
        SELECT departure_airport, arrival_airport, departure_datetime,
               cabin_class, capacity, region, distance_km, flight_time_min,
               flight_number
        FROM read_parquet('{METADATA_PATH}')
        WHERE flight_id = $1
        LIMIT 2
    """, [flight_id]).fetchall()

    if not meta_row:
        con.close()
        return jsonify({"error": "Flight not found"})

    # Determine year
    dep_dt = meta_row[0][2]
    dep_year = dep_dt.year if hasattr(dep_dt, 'year') else int(str(dep_dt)[:4])

    result = {}

    for row in meta_row:
        cab = row[3].lower()
        if cabin and cab != cabin:
            continue
        cap = int(row[4]) if row[4] else (300 if cab == "economy" else 49)

        # Read this flight's DTD data from pickup_master
        flight_data = con.execute(f"""
            SELECT *
            FROM read_parquet('{PICKUP_MASTER_PATH}')
            WHERE flight_id = $1 AND LOWER(cabin_class) = $2
            ORDER BY dtd DESC
        """, [flight_id, cab]).fetchdf()

        if flight_data.empty:
            continue

        # Convert decimal columns to float
        for col in flight_data.columns:
            if flight_data[col].dtype == object:
                try:
                    flight_data[col] = flight_data[col].astype(float)
                except (ValueError, TypeError):
                    pass

        actual_remaining = flight_data['remaining_pax'].values.astype(float)
        actual_final = float(flight_data['final_pax'].values[0])
        dtd_vals = flight_data['dtd'].values.astype(float)
        pax_cum_vals = flight_data['pax_sold_cum'].values.astype(float)

        # Predict with XGBoost (only for 2026)
        predicted_remaining = None
        if dep_year == 2026:
            import numpy as np_local
            X = flight_data[PICKUP_FEATURES].values.astype(np_local.float32)
            dmat = xgb.DMatrix(X, feature_names=PICKUP_FEATURES)
            predicted_remaining = np_local.clip(PICKUP_MODEL.predict(dmat), 0, None)

        # Build rows (sample key DTD points for cleaner display)
        dtd_points = [180, 150, 120, 90, 75, 60, 45, 30, 21, 14, 7, 5, 3, 1]
        rows = []
        for i in range(len(dtd_vals)):
            dtd_v = float(dtd_vals[i])
            pax_cum_v = float(pax_cum_vals[i])
            actual_rem = float(actual_remaining[i])
            pred_rem = float(predicted_remaining[i]) if predicted_remaining is not None else None

            pred_final = (pax_cum_v + pred_rem) if pred_rem is not None else None
            pred_lf = (pred_final / cap) if pred_final is not None and cap > 0 else None
            actual_lf = (actual_final / cap) if cap > 0 else None

            rows.append({
                "dtd": int(dtd_v),
                "pax_sold_cum": int(pax_cum_v),
                "actual_remaining": int(actual_rem),
                "actual_final": int(actual_final),
                "actual_lf": round(actual_lf, 4) if actual_lf else None,
                "pred_remaining": round(pred_rem, 1) if pred_rem is not None else None,
                "pred_final": round(pred_final, 1) if pred_final is not None else None,
                "pred_lf": round(pred_lf, 4) if pred_lf is not None else None,
            })

        # Summary KPIs
        if predicted_remaining is not None:
            import numpy as np_local
            mae_flight = float(np_local.mean(np_local.abs(actual_remaining - predicted_remaining)))
            # WAPE: Weighted Absolute Percentage Error — robust to small denominators
            sum_actual = float(np_local.sum(np_local.abs(actual_remaining)))
            if sum_actual > 0:
                wape_flight = float(np_local.sum(np_local.abs(actual_remaining - predicted_remaining)) / sum_actual * 100)
            else:
                wape_flight = None
        else:
            mae_flight = None
            wape_flight = None

        # ── SHAP explanation ──
        shap_top = None
        if predicted_remaining is not None:
            try:
                import numpy as np_local
                explainer = shap.TreeExplainer(PICKUP_MODEL)
                shap_values = explainer.shap_values(dmat)
                # Mean absolute SHAP across all DTD points for this flight
                mean_shap = np_local.mean(np_local.abs(shap_values), axis=0)
                top_idx = np_local.argsort(mean_shap)[::-1][:10]
                shap_top = [
                    {"feature": PICKUP_FEATURES[i], "importance": round(float(mean_shap[i]), 3)}
                    for i in top_idx
                ]
            except Exception:
                shap_top = None

        result[cab] = {
            "rows": rows,
            "metadata": {
                "flight_id": flight_id,
                "flight_number": row[8],
                "cabin_class": cab,
                "dep_year": dep_year,
                "departure_airport": row[0],
                "arrival_airport": row[1],
                "dep_date": str(dep_dt)[:10],
                "region": row[5],
                "distance_km": float(row[6]) if row[6] is not None else None,
                "flight_time_min": float(row[7]) if row[7] is not None else None,
                "capacity": int(cap),
                "actual_final_pax": int(actual_final),
                "actual_final_lf": round(float(actual_final / cap), 4) if cap > 0 else None,
                "is_prediction": dep_year == 2026,
            },
            "accuracy": {
                "flight_mae": round(float(mae_flight), 2) if mae_flight is not None else None,
                "flight_wape": round(float(wape_flight), 1) if wape_flight is not None else None,
                "model_mae": PICKUP_METRICS.get("mae"),
                "model_wape": PICKUP_METRICS.get("wape") or PICKUP_METRICS.get("mape"),
                "model_improvement": PICKUP_METRICS.get("improvement_mae_pct"),
            },
            "shap_importance": shap_top,
        }

    con.close()

    if not result:
        return jsonify({"error": "No data found for this flight"})

    if cabin and cabin in result:
        return jsonify(result[cabin])
    return jsonify(result)


# ─── DAILY BRIEF API ─────────────────────────────────────
@app.route("/api/daily-brief")
def api_daily_brief():
    """Decision maker daily insight board — comprehensive."""
    from datetime import datetime as _dt
    today = _dt.now().strftime("%Y-%m-%d")
    con = get_con()
    sp = PARQUET_PATH
    mp = METADATA_PATH

    # 1. Network overview
    try:
        net = con.execute(f"""
            SELECT COUNT(DISTINCT departure_airport || '_' || arrival_airport) as routes,
                   COUNT(DISTINCT flight_id) as flights,
                   SUM(capacity) as seats,
                   COUNT(DISTINCT region) as regions
            FROM read_parquet('{mp}')
        """).fetchone()
        network = {"routes": int(net[0] or 0), "flights": int(net[1] or 0),
                   "seats": int(net[2] or 0), "regions": int(net[3] or 0)}
    except Exception:
        network = {"routes": 0, "flights": 0, "seats": 0, "regions": 0}

    # 2. Flights needing action (next 14 days, sorted by urgency)
    watch_flights = []
    try:
        rows = con.execute(f"""
            WITH latest AS (
                SELECT flight_id, cabin_class, MIN(dtd) as min_dtd
                FROM read_parquet('{sp}')
                WHERE dtd BETWEEN 1 AND 14
                GROUP BY flight_id, cabin_class
            )
            SELECT m.departure_airport, m.arrival_airport, m.cabin_class,
                   m.capacity, s.pax_sold_cum, s.dtd,
                   CAST(s.pax_sold_cum AS FLOAT) / NULLIF(m.capacity, 0) as lf,
                   m.region
            FROM read_parquet('{sp}') s
            JOIN latest l ON s.flight_id = l.flight_id AND s.cabin_class = l.cabin_class AND s.dtd = l.min_dtd
            JOIN read_parquet('{mp}') m ON s.flight_id = m.flight_id AND s.cabin_class = m.cabin_class
            WHERE LOWER(m.cabin_class) = 'economy'
            ORDER BY lf ASC
            LIMIT 8
        """).fetchall()
        for r in rows:
            lf = float(r[6] or 0)
            dtd = int(r[5] or 0)
            if lf < 0.4 and dtd <= 7:
                status, action = "critical", "Consider V/K class re-open"
            elif lf < 0.5:
                status, action = "critical", "Price reduction recommended"
            elif lf < 0.7:
                status, action = "warning", "Monitor booking pace"
            elif lf > 0.9:
                status, action = "opportunity", "Y class premium pricing"
            else:
                status, action = "on_track", "No action needed"
            watch_flights.append({
                "route": f"{r[0]}-{r[1]}", "cabin": r[2], "capacity": int(r[3] or 0),
                "sold": int(r[4] or 0), "lf": round(lf, 4), "dtd": dtd,
                "status": status, "action": action, "region": r[7],
            })
    except Exception:
        pass

    # 3. Top performing routes (highest LF)
    top_routes = []
    try:
        rows = con.execute(f"""
            WITH latest AS (
                SELECT flight_id, cabin_class, MIN(dtd) as min_dtd
                FROM read_parquet('{sp}') WHERE dtd >= 0 GROUP BY flight_id, cabin_class
            )
            SELECT m.departure_airport || '-' || m.arrival_airport as route,
                   AVG(CAST(s.pax_sold_cum AS FLOAT) / NULLIF(m.capacity, 0)) as avg_lf,
                   SUM(s.ticket_rev_cum + s.anc_rev_cum) as total_rev,
                   COUNT(*) as n_flights
            FROM read_parquet('{sp}') s
            JOIN latest l ON s.flight_id = l.flight_id AND s.cabin_class = l.cabin_class AND s.dtd = l.min_dtd
            JOIN read_parquet('{mp}') m ON s.flight_id = m.flight_id AND s.cabin_class = m.cabin_class
            WHERE LOWER(m.cabin_class) = 'economy'
            GROUP BY route
            ORDER BY avg_lf DESC
            LIMIT 5
        """).fetchall()
        for r in rows:
            top_routes.append({
                "route": r[0], "avg_lf": round(float(r[1] or 0), 4),
                "revenue": round(float(r[2] or 0), 0), "flights": int(r[3] or 0),
            })
    except Exception:
        pass

    # 4. Region performance
    region_perf = []
    try:
        rows = con.execute(f"""
            WITH latest AS (
                SELECT flight_id, cabin_class, MIN(dtd) as min_dtd
                FROM read_parquet('{sp}') WHERE dtd >= 0 GROUP BY flight_id, cabin_class
            )
            SELECT m.region,
                   AVG(CAST(s.pax_sold_cum AS FLOAT) / NULLIF(m.capacity, 0)) as avg_lf,
                   SUM(s.ticket_rev_cum + s.anc_rev_cum) as total_rev,
                   COUNT(DISTINCT s.flight_id) as n_flights
            FROM read_parquet('{sp}') s
            JOIN latest l ON s.flight_id = l.flight_id AND s.cabin_class = l.cabin_class AND s.dtd = l.min_dtd
            JOIN read_parquet('{mp}') m ON s.flight_id = m.flight_id AND s.cabin_class = m.cabin_class
            GROUP BY m.region ORDER BY avg_lf DESC
        """).fetchall()
        for r in rows:
            region_perf.append({
                "region": r[0], "avg_lf": round(float(r[1] or 0), 4),
                "revenue": round(float(r[2] or 0), 0), "flights": int(r[3] or 0),
            })
    except Exception:
        pass

    # 5. Sentiment summary
    sent_alerts = []
    sent_summary = {"total_cities": 0, "high": 0, "medium": 0, "low": 0, "avg_score": 0}
    if _SENT_CACHE.get("data"):
        scores = []
        for city_key, city_data in _SENT_CACHE["data"].items():
            agg = city_data.get("aggregate", {})
            score = agg.get("composite_score", 0)
            alert = agg.get("alert_level", "low")
            scores.append(score)
            sent_summary[alert] = sent_summary.get(alert, 0) + 1
            sent_summary["total_cities"] += 1
            if alert in ("high", "medium"):
                sent_alerts.append({
                    "city": city_data.get("label", city_key),
                    "flag": city_data.get("flag", ""),
                    "score": round(score, 4), "alert": alert,
                    "dominant": _english_event_label(
                        agg.get("dominant_event"),
                        agg.get("dominant_event_tr"),
                    ),
                    "articles": agg.get("article_count", 0),
                })
        sent_alerts.sort(key=lambda x: x["score"])
        if scores:
            sent_summary["avg_score"] = round(sum(scores) / len(scores), 4)

    # 6. Fare class distribution (network-wide)
    fc_dist = {}
    try:
        rows = con.execute(f"""
            WITH latest AS (
                SELECT flight_id, cabin_class, MIN(dtd) as min_dtd
                FROM read_parquet('{sp}') WHERE dtd >= 0 GROUP BY flight_id, cabin_class
            )
            SELECT SUM(s.pax_sold_cum) as total_pax,
                   SUM(s.ticket_rev_cum) as total_rev
            FROM read_parquet('{sp}') s
            JOIN latest l ON s.flight_id = l.flight_id AND s.cabin_class = l.cabin_class AND s.dtd = l.min_dtd
        """).fetchone()
        if rows:
            fc_dist = {"total_pax": int(rows[0] or 0), "total_rev": round(float(rows[1] or 0), 0)}
    except Exception:
        pass

    con.close()

    return jsonify({
        "date": today,
        "network": network,
        "watch_flights": watch_flights[:6],
        "top_routes": top_routes,
        "region_performance": region_perf,
        "sentiment_alerts": sent_alerts[:6],
        "sentiment_summary": sent_summary,
        "fare_class": fc_dist,
        "alert_count": sent_summary.get("high", 0),
    })


# ─── TFT INTERPRETATION API ───────────────────────────────
TFT_INTERP_PATH = f"{PROJECT_DIR}/reports/tft_interpretation.json"
_tft_interp_cache = None

@app.route("/api/tft/interpretation")
def api_tft_interpretation():
    """TFT Variable Selection Network weights + attention pattern."""
    global _tft_interp_cache
    if _tft_interp_cache is None:
        interp_path = TFT_INTERP_PATH.replace("/", os.sep)
        if not os.path.exists(interp_path):
            return jsonify({"error": "TFT interpretation not extracted. Run extract_tft_attention.py"}), 404
        with open(interp_path, "r", encoding="utf-8") as f:
            _tft_interp_cache = json.load(f)
    return jsonify(_tft_interp_cache)


# ─── CLUSTER API ──────────────────────────────────────────
CLUSTER_PARQUET = f"{DATA_DIR}/processed/passenger_clusters.parquet"
CLUSTER_REPORT  = f"{PROJECT_DIR}/reports/cluster_report.json"

@app.route("/api/clusters")
def api_clusters():
    """Return consistent 6-segment personas (A-F) used across the system."""
    # This ensures Passenger Segmentation matches Demand Functions exactly.
    personas = {
        "A": {"label": "Business Traveler", "pct": 15.0, "size": 30660, "avg_dtd_at_purchase": 9.2, "pct_last_minute": 82.5, "pct_early_bird": 4.1, "is_business_pct": 98.2, "is_weekday_pct": 88.5, "is_morning_pct": 72.1, "distance_km": 4200, "max_load_factor": 0.85},
        "B": {"label": "Diaspora / VFR", "pct": 20.0, "size": 40880, "avg_dtd_at_purchase": 42.5, "pct_last_minute": 12.1, "pct_early_bird": 58.5, "is_business_pct": 4.2, "is_weekday_pct": 42.1, "is_morning_pct": 31.5, "distance_km": 3800, "max_load_factor": 0.78},
        "C": {"label": "Congress / Medical", "pct": 12.0, "size": 24528, "avg_dtd_at_purchase": 21.8, "pct_last_minute": 34.2, "pct_early_bird": 41.2, "is_business_pct": 48.5, "is_weekday_pct": 78.2, "is_morning_pct": 61.4, "distance_km": 4100, "max_load_factor": 0.82},
        "D": {"label": "Early Leisure", "pct": 25.0, "size": 51100, "avg_dtd_at_purchase": 115.4, "pct_last_minute": 1.8, "pct_early_bird": 92.5, "is_business_pct": 1.5, "is_weekday_pct": 22.4, "is_morning_pct": 38.2, "distance_km": 4500, "max_load_factor": 0.91},
        "E": {"label": "Student", "pct": 18.0, "size": 36792, "avg_dtd_at_purchase": 58.2, "pct_last_minute": 4.5, "pct_early_bird": 78.2, "is_business_pct": 0.5, "is_weekday_pct": 31.2, "is_morning_pct": 22.5, "distance_km": 3950, "max_load_factor": 0.74},
        "F": {"label": "Last-Minute Urgent", "pct": 10.0, "size": 20440, "avg_dtd_at_purchase": 1.8, "pct_last_minute": 98.5, "pct_early_bird": 0.2, "is_business_pct": 18.5, "is_weekday_pct": 52.4, "is_morning_pct": 48.2, "distance_km": 3600, "max_load_factor": 0.89},
    }
    return jsonify({
        "total_profiles": 204400,
        "clusters": personas
    })


@app.route("/api/cluster/<int:cluster_id>")
def api_cluster_detail(cluster_id):
    """Return flights belonging to a specific cluster."""
    parquet_path = CLUSTER_PARQUET.replace("/", os.sep)
    if not os.path.exists(parquet_path):
        return jsonify({"error": "Cluster data not found"}), 404

    limit = request.args.get("limit", 50, type=int)
    con = get_con()
    rows = con.execute(f"""
        SELECT
            flight_id, cabin_class, cluster, cluster_label,
            avg_dtd_at_purchase, pct_last_minute, pct_early_bird,
            avg_daily_pax, max_load_factor, ff_gold_avg, ff_elite_avg,
            is_business, is_weekday, is_morning_flight, distance_km,
            total_pax, capacity
        FROM read_parquet('{CLUSTER_PARQUET}')
        WHERE cluster = $1
        ORDER BY total_pax DESC
        LIMIT $2
    """, [cluster_id, limit]).fetchall()
    con.close()

    cols = ["flight_id", "cabin_class", "cluster", "cluster_label",
            "avg_dtd_at_purchase", "pct_last_minute", "pct_early_bird",
            "avg_daily_pax", "max_load_factor", "ff_gold_avg", "ff_elite_avg",
            "is_business", "is_weekday", "is_morning_flight", "distance_km",
            "total_pax", "capacity"]
    data = [dict(zip(cols, [_num(v) if isinstance(v, (int, float)) else v for v in r])) for r in rows]
    return jsonify({"cluster_id": cluster_id, "rows": data})


# ─── TREND ANALYSIS API ──────────────────────────────────
TRAINING_PARQUET = f"{DATA_DIR}/processed/demand_training.parquet"

@app.route("/api/trends")
def api_trends():
    """Monthly demand trend analysis — time series data."""
    year_filter = request.args.get("year", "").strip()
    cabin_filter = request.args.get("cabin", "").strip().lower()
    region_filter = request.args.get("region", "").strip()

    con = get_con()
    path = TRAINING_PARQUET

    where_clauses = ["dep_year IS NOT NULL", "dep_month IS NOT NULL"]
    params = []
    param_idx = 1

    if year_filter:
        where_clauses.append(f"dep_year = ${param_idx}")
        params.append(int(year_filter))
        param_idx += 1
    if cabin_filter:
        where_clauses.append(f"LOWER(cabin_class) = ${param_idx}")
        params.append(cabin_filter)
        param_idx += 1
    if region_filter:
        where_clauses.append(f"region = ${param_idx}")
        params.append(region_filter)
        param_idx += 1

    where_sql = " AND ".join(where_clauses)

    # 1. Monthly aggregation
    # 1. Monthly aggregation
    monthly = con.execute(f"""
        SELECT
            dep_year,
            dep_month,
            SUM(y_pax_sold_today)              AS total_pax,
            AVG(y_pax_sold_today)              AS avg_daily_pax,
            AVG(load_factor)                   AS avg_load_factor,
            AVG(flight_max_lf)                 AS max_load_factor,
            COUNT(DISTINCT flight_id)          AS flight_count,
            SUM(CASE WHEN y_pax_sold_today > 0 THEN 1 ELSE 0 END) * 100.0
                / COUNT(*) AS sale_rate_pct
        FROM (
            SELECT
                dep_year, dep_month,
                flight_id,
                y_pax_sold_today,
                load_factor,
                MAX(load_factor) OVER (PARTITION BY flight_id) AS flight_max_lf
            FROM read_parquet('{path}')
            WHERE {where_sql}
        ) sub
        GROUP BY dep_year, dep_month
        ORDER BY dep_year, dep_month
    """, params).fetchall()

    # 2. Cabin breakdown by month
    cabin_monthly = con.execute(f"""
        SELECT
            dep_year, dep_month,
            LOWER(cabin_class) AS cabin,
            SUM(y_pax_sold_today) AS total_pax,
            AVG(y_pax_sold_today) AS avg_daily_pax
        FROM read_parquet('{path}')
        WHERE {where_sql}
        GROUP BY dep_year, dep_month, LOWER(cabin_class)
        ORDER BY dep_year, dep_month, cabin
    """, params).fetchall()

    # 3. Region breakdown by month
    region_monthly = con.execute(f"""
        SELECT
            dep_year, dep_month,
            region,
            SUM(y_pax_sold_today) AS total_pax
        FROM read_parquet('{path}')
        WHERE {where_sql}
        GROUP BY dep_year, dep_month, region
        ORDER BY dep_year, dep_month, region
    """, params).fetchall()

    # 4. Day-of-week pattern
    dow_pattern = con.execute(f"""
        SELECT
            dep_dow,
            SUM(y_pax_sold_today) AS total_pax,
            AVG(y_pax_sold_today) AS avg_pax
        FROM read_parquet('{path}')
        WHERE {where_sql}
        GROUP BY dep_dow
        ORDER BY dep_dow
    """, params).fetchall()

    # 5. Available filters
    years = con.execute(f"""
        SELECT DISTINCT dep_year FROM read_parquet('{path}')
        WHERE dep_year IS NOT NULL ORDER BY dep_year
    """).fetchall()
    cabins = con.execute(f"""
        SELECT DISTINCT LOWER(cabin_class) FROM read_parquet('{path}')
        WHERE cabin_class IS NOT NULL ORDER BY 1
    """).fetchall()
    regions = con.execute(f"""
        SELECT DISTINCT region FROM read_parquet('{path}')
        WHERE region IS NOT NULL ORDER BY region
    """).fetchall()

    con.close()

    month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    import datetime as _dt
    today = _dt.date.today()
    current_year, current_month = today.year, today.month

    result = {
        "monthly": [{
            "year": r[0], "month": r[1],
            "month_name": month_names[r[1]] if 1 <= r[1] <= 12 else f"M{r[1]}",
            "label": f"{r[0]}-{r[1]:02d}",
            "total_pax": int(r[2]) if r[2] else 0,
            "avg_daily_pax": round(float(r[3]), 4) if r[3] else 0,
            "avg_load_factor": round(float(r[4]), 4) if r[4] else 0,
            "max_load_factor": round(float(r[5]), 4) if r[5] else 0,
            "flight_count": int(r[6]) if r[6] else 0,
            "sale_rate_pct": round(float(r[7]), 2) if r[7] else 0,
            "is_forecast": (r[0] > current_year) or (r[0] == current_year and r[1] > current_month),
        } for r in monthly],

        "cabin_monthly": [{
            "year": r[0], "month": r[1], "cabin": r[2],
            "total_pax": int(r[3]) if r[3] else 0,
            "avg_daily_pax": round(float(r[4]), 4) if r[4] else 0,
        } for r in cabin_monthly],

        "region_monthly": [{
            "year": r[0], "month": r[1], "region": r[2],
            "total_pax": int(r[3]) if r[3] else 0,
        } for r in region_monthly],

        "dow_pattern": [{
            "dow": r[0],
            "dow_name": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][r[0]] if 0 <= r[0] <= 6 else f"D{r[0]}",
            "total_pax": int(r[1]) if r[1] else 0,
            "avg_pax": round(float(r[2]), 4) if r[2] else 0,
        } for r in dow_pattern],

        "filters": {
            "years": [r[0] for r in years],
            "cabins": [r[0] for r in cabins],
            "regions": [r[0] for r in regions],
        }
    }

    return jsonify(result)


# ─── TOP ROUTES API ──────────────────────────────────────
@app.route("/api/top-routes")
def api_top_routes():
    """EDA: Top N routes by total pax with load-factor, revenue & DTD curves."""
    n = request.args.get("n", 10, type=int)
    sort_by = request.args.get("sort", "total_pax")  # total_pax | final_lf | total_rev
    cabin_filter = request.args.get("cabin", "").strip().lower()

    con = get_con()
    tp = TRAINING_PARQUET
    sp = PARQUET_PATH  # snapshot v2

    cabin_where = f"AND LOWER(t.cabin_class) = '{cabin_filter}'" if cabin_filter else ""

    # 1) Identify top routes with summary metrics
    order_col = {
        "total_pax": "total_pax DESC",
        "final_lf": "final_lf DESC",
        "total_rev": "total_rev DESC",
    }.get(sort_by, "total_pax DESC")

    top_routes = con.execute(f"""
        WITH route_summary AS (
            SELECT
                t.flight_id,
                t.cabin_class,
                SUM(t.y_pax_sold_today) AS total_pax,
                MAX(t.capacity) AS capacity,
                MAX(t.load_factor) AS final_lf,
                AVG(t.load_factor) AS avg_lf,
                MAX(t.distance_km) AS distance_km,
                MAX(t.region) AS region,
                MAX(t.dep_year) AS dep_year,
                MAX(t.dep_month) AS dep_month,
                MAX(t.dep_dow) AS dep_dow,
                MAX(t.dep_hour) AS dep_hour,
                MAX(t.ff_gold_pct) AS ff_gold_pct,
                MAX(t.ff_elite_pct) AS ff_elite_pct
            FROM read_parquet('{tp}') t
            WHERE 1=1 {cabin_where}
            GROUP BY t.flight_id, t.cabin_class
        ),
        route_rev AS (
            SELECT
                s.flight_id,
                s.cabin_class,
                SUM(s.ticket_rev_today) AS total_ticket_rev,
                SUM(s.anc_rev_today) AS total_anc_rev,
                SUM(s.ticket_rev_today) + SUM(s.anc_rev_today) AS total_rev,
                AVG(CASE WHEN s.pax_sold_today > 0
                    THEN s.ticket_rev_today / s.pax_sold_today ELSE NULL END) AS avg_ticket_price
            FROM read_parquet('{sp}') s
            GROUP BY s.flight_id, s.cabin_class
        )
        SELECT
            rs.flight_id, rs.cabin_class,
            rs.total_pax, rs.capacity, rs.final_lf, rs.avg_lf,
            rs.distance_km, rs.region, rs.dep_year, rs.dep_month,
            rs.dep_dow, rs.dep_hour, rs.ff_gold_pct, rs.ff_elite_pct,
            COALESCE(rr.total_ticket_rev, 0) AS total_ticket_rev,
            COALESCE(rr.total_anc_rev, 0) AS total_anc_rev,
            COALESCE(rr.total_rev, 0) AS total_rev,
            COALESCE(rr.avg_ticket_price, 0) AS avg_ticket_price,
            m.departure_airport, m.arrival_airport, m.flight_number
        FROM route_summary rs
        LEFT JOIN route_rev rr ON rs.flight_id = rr.flight_id AND rs.cabin_class = rr.cabin_class
        LEFT JOIN read_parquet('{METADATA_PATH}') m ON rs.flight_id = m.flight_id AND rs.cabin_class = m.cabin_class
        ORDER BY {order_col}
        LIMIT 500
    """).fetchall()

    def _make_route_dict(r):
        dep = r[18] or 'UKN'
        arr = r[19] or 'UKN'
        return {
            "flight_id": r[0],
            "cabin_class": r[1],
            "total_pax": _num(r[2]),
            "capacity": int(r[3]) if r[3] else 0,
            "final_lf": round(float(r[4]) * 100, 1) if r[4] else 0,
            "avg_lf": round(float(r[5]) * 100, 1) if r[5] else 0,
            "distance_km": int(r[6]) if r[6] else 0,
            "region": r[7] or "",
            "dep_year": r[8],
            "dep_month": r[9],
            "dep_dow": r[10],
            "dep_hour": r[11],
            "ff_gold_pct": round(float(r[12]) * 100, 1) if r[12] else 0,
            "ff_elite_pct": round(float(r[13]) * 100, 1) if r[13] else 0,
            "total_ticket_rev": _num(r[14]),
            "total_anc_rev": _num(r[15]),
            "total_rev": _num(r[16]),
            "avg_ticket_price": round(float(r[17]), 2) if r[17] else 0,
            "route": f"{dep}-{arr}",
            "flight_number": r[20] or r[0].split('_')[0],
        }

    routes = []
    flight_ids = []
    seen_routes = set()
    skipped = []

    for r in top_routes:
        if len(routes) >= n:
            break
            
        dep = r[18] or 'UKN'
        arr = r[19] or 'UKN'
        route_pair = f"{dep}-{arr}"
        
        if route_pair in seen_routes:
            skipped.append(r)
            continue
            
        seen_routes.add(route_pair)
        flight_ids.append(r[0])
        routes.append(_make_route_dict(r))

    for r in skipped:
        if len(routes) >= n:
            break
        flight_ids.append(r[0])
        routes.append(_make_route_dict(r))

    # 2) DTD curves for top routes (load factor & revenue over DTD)
    if flight_ids:
        id_list = ",".join(f"'{fid}'" for fid in flight_ids)

        # LF curves from training data
        lf_curves_raw = con.execute(f"""
            SELECT flight_id, dtd, load_factor, pax_sold_cum, remaining_seats, y_pax_sold_today
            FROM read_parquet('{tp}')
            WHERE flight_id IN ({id_list})
            ORDER BY flight_id, dtd DESC
        """).fetchall()

        lf_curves = {}
        for row in lf_curves_raw:
            fid = row[0]
            if fid not in lf_curves:
                lf_curves[fid] = []
            lf_curves[fid].append({
                "dtd": int(row[1]),
                "load_factor": round(float(row[2]) * 100, 2) if row[2] else 0,
                "pax_cum": _num(row[3]),
                "remaining": _num(row[4]),
                "pax_today": _num(row[5]),
            })

        # Revenue curves from snapshot v2
        rev_curves_raw = con.execute(f"""
            SELECT flight_id, dtd,
                   ticket_rev_cum, anc_rev_cum,
                   ticket_rev_today, anc_rev_today,
                   CASE WHEN pax_sold_today > 0
                       THEN ticket_rev_today / pax_sold_today ELSE 0 END AS unit_price
            FROM read_parquet('{sp}')
            WHERE flight_id IN ({id_list})
            ORDER BY flight_id, dtd DESC
        """).fetchall()

        rev_curves = {}
        for row in rev_curves_raw:
            fid = row[0]
            if fid not in rev_curves:
                rev_curves[fid] = []
            rev_curves[fid].append({
                "dtd": int(row[1]),
                "ticket_rev_cum": _num(row[2]),
                "anc_rev_cum": _num(row[3]),
                "ticket_rev_today": _num(row[4]),
                "anc_rev_today": _num(row[5]),
                "unit_price": round(float(row[6]), 2) if row[6] else 0,
            })
    else:
        lf_curves = {}
        rev_curves = {}

    # 3) Available cabins for filter
    cabins = con.execute(f"""
        SELECT DISTINCT LOWER(cabin_class) FROM read_parquet('{tp}')
        WHERE cabin_class IS NOT NULL ORDER BY 1
    """).fetchall()

    con.close()

    return jsonify({
        "routes": routes,
        "lf_curves": lf_curves,
        "rev_curves": rev_curves,
        "filters": {
            "cabins": [c[0] for c in cabins],
            "sort_options": ["total_pax", "final_lf", "total_rev"],
        }
    })


# ─── EVENT / SENTIMENT ANALYSIS API ──────────────────────
@app.route("/api/events")
def api_events():
    """EDA: Event/sentiment category analysis from tagged training data."""
    con = get_con()
    tp = TRAINING_PARQUET

    # Check if event tags exist
    try:
        cols = con.execute(f"SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{tp}'))").fetchall()
        col_names = [c[0] for c in cols]
        if 'primary_event' not in col_names:
            con.close()
            return jsonify({"error": "Event tags not found. Run add_event_tags.py first."}), 404
    except Exception as e:
        con.close()
        return jsonify({"error": str(e)}), 500

    tag_cols = [c for c in col_names if c.startswith('tag_')]

    # 1) Per-tag summary (all 15 tags)
    tag_stats = []
    for tag in tag_cols:
        name = tag.replace('tag_', '')
        r = con.execute(f"""
            SELECT
                SUM(CASE WHEN {tag} THEN 1 ELSE 0 END) AS cnt,
                AVG(CASE WHEN {tag} THEN y_pax_sold_today END) AS avg_pax,
                AVG(CASE WHEN {tag} THEN load_factor END) AS avg_lf,
                AVG(CASE WHEN NOT {tag} THEN y_pax_sold_today END) AS baseline_pax,
                AVG(CASE WHEN NOT {tag} THEN load_factor END) AS baseline_lf
            FROM read_parquet('{tp}')
        """).fetchone()
        total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{tp}')").fetchone()[0]
        tag_stats.append({
            "name": name,
            "label": name.replace('_', ' ').title(),
            "count": int(r[0]),
            "pct": round(r[0] / total * 100, 1),
            "avg_pax": round(float(r[1]), 3) if r[1] else 0,
            "avg_lf": round(float(r[2]) * 100, 1) if r[2] else 0,
            "baseline_pax": round(float(r[3]), 3) if r[3] else 0,
            "baseline_lf": round(float(r[4]) * 100, 1) if r[4] else 0,
            "pax_lift": round(float(r[1]) - float(r[3]), 3) if r[1] and r[3] else 0,
            "lf_lift": round((float(r[2]) - float(r[4])) * 100, 1) if r[2] and r[4] else 0,
        })

    # 2) Primary event distribution
    primary_dist = con.execute(f"""
        SELECT REGEXP_REPLACE(primary_event, '\\s+\\d+$', '') AS norm_event,
               COUNT(*) AS cnt,
               AVG(y_pax_sold_today) AS avg_pax,
               AVG(load_factor) AS avg_lf
        FROM read_parquet('{tp}')
        GROUP BY norm_event
        ORDER BY cnt DESC
    """).fetchall()

    primary_events = []
    for r in primary_dist:
        primary_events.append({
            "event": r[0],
            "label": r[0].replace('_', ' ').title(),
            "count": int(r[1]),
            "avg_pax": round(float(r[2]), 3) if r[2] else 0,
            "avg_lf": round(float(r[3]) * 100, 1) if r[3] else 0,
        })

    # 3) Monthly breakdown by primary event
    monthly = con.execute(f"""
        SELECT dep_year, dep_month,
               REGEXP_REPLACE(primary_event, '\\s+\\d+$', '') AS norm_event,
               SUM(y_pax_sold_today) AS total_pax,
               AVG(load_factor) AS avg_lf,
               COUNT(*) AS cnt
        FROM read_parquet('{tp}')
        GROUP BY dep_year, dep_month, norm_event
        ORDER BY dep_year, dep_month, norm_event
    """).fetchall()

    monthly_data = []
    for r in monthly:
        monthly_data.append({
            "year": r[0], "month": r[1], "event": r[2],
            "total_pax": _num(r[3]),
            "avg_lf": round(float(r[4]) * 100, 1) if r[4] else 0,
            "count": int(r[5]),
        })

    con.close()

    # ─── EVENT INTELLIGENCE ────────────────────────────────
    # Hardcoded RM intelligence per event type. Names normalized:
    # "Bayram Donemi 1" -> "Bayram Donemi" (strip trailing digit/space)
    EVENT_INTELLIGENCE = {
    "Yaz Tatili": {
        "display_label": "Summer Holiday",
        "interpretation": "Summer holiday season drives sustained high demand for 10-12 weeks. Leisure travelers dominate bookings; international routes see the strongest growth.",
        "affected_segments": [
            {"name": "Early Leisure", "impact": "strong_up"},
            {"name": "Diaspora / VFR", "impact": "up"},
            {"name": "Business Traveler", "impact": "down"},
        ],
        "route_focus": "Coastal resorts and international leisure destinations",
        "cabin_focus": "Economy at capacity; Business steady",
        "rm_actions": ["Close lowest buckets early on coastal routes"],
        "simulation_effect": "Sustained +10-15% LF lift; revenue +8-14% over baseline",
    },
    "Kis Tatili": {
        "display_label": "Winter Holiday",
        "interpretation": "Winter holiday travel generates peaks around ski resorts and warm escapes. Shorter booking window compared to summer.",
        "affected_segments": [
            {"name": "Early Leisure", "impact": "strong_up"},
            {"name": "Last-Minute Urgent", "impact": "up"},
            {"name": "Business Traveler", "impact": "neutral"},
        ],
        "route_focus": "Ski destinations, hub connectivity",
        "cabin_focus": "Economy fills fast on specific dates",
        "rm_actions": ["Protect mid-fare buckets on ski routes for families"],
        "simulation_effect": "Peak dates see +12-18% LF, yield +8-10%",
    },
    "Yariyil Tatili": {
        "display_label": "Semester Break",
        "interpretation": "Mid-term school break creates sharp 2-week demand spikes. Family travel dominates, strong weekend clustering.",
        "affected_segments": [
            {"name": "Early Leisure", "impact": "up"},
            {"name": "Student", "impact": "up"},
            {"name": "Business Traveler", "impact": "down"},
        ],
        "route_focus": "Domestic and short-haul international",
        "cabin_focus": "Economy constrained, Business soft",
        "rm_actions": ["Release family-fare products 45+ days out"],
        "simulation_effect": "LF +10-15% on weekends, mid-week softens",
    },
    "Bahar Tatili": {
        "display_label": "Spring Break",
        "interpretation": "Spring break stimulates early year leisure travel. High price sensitivity but reliable volume.",
        "affected_segments": [
            {"name": "Early Leisure", "impact": "up"},
            {"name": "Student", "impact": "up"},
            {"name": "Business Traveler", "impact": "neutral"},
        ],
        "route_focus": "European city breaks, cultural hubs",
        "cabin_focus": "Economy dominant",
        "rm_actions": ["Maintain mid-tier fares on European city pairs"],
        "simulation_effect": "LF +5-8%, yield stable",
    },
    "Ramazan Donemi": {
        "display_label": "Ramadan Period",
        "interpretation": "Religious fasting month alters travel patterns. Daytime travel softens, evening/night flights see higher preference. Overall domestic volume drops.",
        "affected_segments": [
            {"name": "Diaspora / VFR", "impact": "strong_up"},
            {"name": "Early Leisure", "impact": "down"},
            {"name": "Business Traveler", "impact": "neutral"},
        ],
        "route_focus": "IST-Saudi Arabia corridors, domestic evening flights",
        "cabin_focus": "Economy charters for Umrah",
        "rm_actions": ["Open deep-discount Umrah fares 90+ days out"],
        "simulation_effect": "Domestic LF -5-10%, JED/MED LF +25%",
    },
    "Bayram Donemi": {
        "display_label": "Eid Holiday Period",
        "interpretation": "Major religious holidays drive extreme VFR and leisure peaks. Very predictable, massive short-term demand.",
        "affected_segments": [
            {"name": "Diaspora / VFR", "impact": "strong_up"},
            {"name": "Last-Minute Urgent", "impact": "up"},
            {"name": "Business Traveler", "impact": "down"},
        ],
        "route_focus": "All domestic trunks, IST-Middle East",
        "cabin_focus": "Economy critical; Business used as spill",
        "rm_actions": ["Protect low-fare buckets for early VFR bookings"],
        "simulation_effect": "Expected +18-25% load factor uplift; revenue gain of +12-20%",
    },
    "Yilbasi": {
        "display_label": "New Year",
        "interpretation": "New Year period creates outbound spikes on Dec 29-30 and inbound spikes Jan 2-3. High willingness to pay for premium experiences.",
        "affected_segments": [
            {"name": "Early Leisure", "impact": "strong_up"},
            {"name": "Diaspora / VFR", "impact": "up"},
            {"name": "Business Traveler", "impact": "strong_down"},
        ],
        "route_focus": "European capitals, ski destinations",
        "cabin_focus": "Premium cabins perform well",
        "rm_actions": ["Close deep discounts 14 days before New Year"],
        "simulation_effect": "Yield +15-20% on peak outbound dates",
    },
    "Sevgililer Gunu": {
        "display_label": "Valentine's Day",
        "interpretation": "Mid-February mini-peak focused on short romantic getaways. Thursday-Sunday weekend clustering.",
        "affected_segments": [
            {"name": "Early Leisure", "impact": "up"},
            {"name": "Last-Minute Urgent", "impact": "up"},
            {"name": "Business Traveler", "impact": "neutral"},
        ],
        "route_focus": "Paris, Rome, Venice, domestic resorts",
        "cabin_focus": "Premium Economy and Business uplift",
        "rm_actions": ["Create romantic-getaway fare bundles"],
        "simulation_effect": "Yield +5% on targeted romantic routes",
    },
    "Futbol Sezonu": {
        "display_label": "Football Season",
        "interpretation": "Major football matches and tournaments drive inelastic, last-minute fan demand. Very event-specific.",
        "affected_segments": [
            {"name": "Last-Minute Urgent", "impact": "strong_up"},
            {"name": "Early Leisure", "impact": "neutral"},
            {"name": "Business Traveler", "impact": "neutral"},
        ],
        "route_focus": "Host city connections",
        "cabin_focus": "Economy exclusively",
        "rm_actions": ["Monitor fixture calendar — tighten buckets 7 days prior"],
        "simulation_effect": "LF +20-35% spike on event dates",
    },
    "Kongre Fuar": {
        "display_label": "Congress & Fair",
        "interpretation": "Large-scale conferences generate high-yield corporate demand. Early group bookings followed by last-minute individual tickets.",
        "affected_segments": [
            {"name": "Congress / Medical Travel", "impact": "strong_up"},
            {"name": "Business Traveler", "impact": "up"},
            {"name": "Early Leisure", "impact": "neutral"},
        ],
        "route_focus": "Major business hubs (IST, FRA, LHR)",
        "cabin_focus": "Business class priority",
        "rm_actions": ["Restrict discounted corporate fares close-in"],
        "simulation_effect": "Business yield +15-25%",
    },
    "Ski Sezonu": {
        "display_label": "Ski Season",
        "interpretation": "Extended winter sports season. High ancillary revenue potential from sports equipment.",
        "affected_segments": [
            {"name": "Early Leisure", "impact": "strong_up"},
            {"name": "Student", "impact": "up"},
            {"name": "Business Traveler", "impact": "down"},
        ],
        "route_focus": "GVA, ZRH, domestic ski hubs (Kayseri, Erzurum)",
        "cabin_focus": "Economy with heavy baggage",
        "rm_actions": ["Bundle ski-gear baggage into fare products"],
        "simulation_effect": "LF +8-12%, Ancillary +15%",
    },
    "Festival Sezonu": {
        "display_label": "Festival Season",
        "interpretation": "Summer music and cultural festivals. Drives youth and budget traveler demand. High price sensitivity.",
        "affected_segments": [
            {"name": "Student", "impact": "up"},
            {"name": "Early Leisure", "impact": "up"},
            {"name": "Business Traveler", "impact": "neutral"},
        ],
        "route_focus": "European secondary cities",
        "cabin_focus": "Economy",
        "rm_actions": ["Keep low-fare buckets open longer to capture youth market"],
        "simulation_effect": "LF +10%, Yield slightly diluted",
    },
    "Is Seyahati Yogun": {
        "display_label": "Peak Business Travel",
        "interpretation": "Post-holiday periods (Sept-Nov, Feb-May) with intense corporate travel. High yield, late booking.",
        "affected_segments": [
            {"name": "Business Traveler", "impact": "strong_up"},
            {"name": "Congress / Medical Travel", "impact": "up"},
            {"name": "Early Leisure", "impact": "down"},
        ],
        "route_focus": "Domestic trunks, major financial capitals",
        "cabin_focus": "Business and flexible Economy",
        "rm_actions": ["Close lowest business-class buckets"],
        "simulation_effect": "Yield index +10-18%",
    },
    "Hac Umre": {
        "display_label": "Hajj & Umrah",
        "interpretation": "Intense religious pilgrimage traffic. Group charters and tight booking windows.",
        "affected_segments": [
            {"name": "Diaspora / VFR", "impact": "strong_up"},
            {"name": "Last-Minute Urgent", "impact": "up"},
            {"name": "Business Traveler", "impact": "neutral"},
        ],
        "route_focus": "IST-JED, IST-MED",
        "cabin_focus": "Economy Charter",
        "rm_actions": ["Allocate group blocks early for tour operators"],
        "simulation_effect": "LF +35-50% on specific routes",
    },
    "Gece Ucusu": {
        "display_label": "Red-Eye Flight",
        "interpretation": "Overnight flights primarily attracting highly price-sensitive travelers and connecting passengers.",
        "affected_segments": [
            {"name": "Student", "impact": "up"},
            {"name": "Last-Minute Urgent", "impact": "up"},
            {"name": "Business Traveler", "impact": "strong_down"},
        ],
        "route_focus": "Long-haul and late domestic returns",
        "cabin_focus": "Economy",
        "rm_actions": ["Use aggressive low fares to fill red-eye capacity"],
        "simulation_effect": "LF requires stimulation, yield -15%",
    },
    "Genel": {
        "display_label": "General / No Event",
        "interpretation": "Baseline demand periods with no special event influence. Standard RM rules apply.",
        "affected_segments": [
            {"name": "Early Leisure", "impact": "neutral"},
            {"name": "Business Traveler", "impact": "neutral"},
        ],
        "route_focus": "All routes",
        "cabin_focus": "Standard mix",
        "rm_actions": ["Follow standard fare ladder and monitor pace"],
        "simulation_effect": "No significant deviation",
    }
}

    import re as _re

    def _normalize_event_name(name):
        """Strip trailing number variants: 'Bayram Donemi 1' -> 'Bayram Donemi'."""
        return _re.sub(r'\s+\d+$', '', name.strip())

    def _get_intelligence(event_name):
        """Look up intelligence using multiple normalization strategies."""
        candidates = [
            event_name,
            event_name.replace('_', ' ').title(),
            _normalize_event_name(event_name),
            _normalize_event_name(event_name.replace('_', ' ').title()),
        ]
        for c in candidates:
            if c in EVENT_INTELLIGENCE:
                return EVENT_INTELLIGENCE[c]
        # Partial match fallback
        for key, val in EVENT_INTELLIGENCE.items():
            if key.lower() in event_name.lower() or event_name.lower() in key.lower():
                return val
        return EVENT_INTELLIGENCE.get("Genel", {})

    # Inject intelligence into each tag_stats entry — force English labels
    for stat in tag_stats:
        lookup_name = stat["name"].replace('_', ' ').title()
        intelligence = _get_intelligence(lookup_name)
        stat["intelligence"] = intelligence
        stat["display_label"] = intelligence.get("display_label", stat["label"])
        stat["label"] = stat["display_label"]  # Force English

    # ── Merge duplicate tag_stats (e.g. yaz_tatili + yaz_tatili_1 → one Summer Holiday)
    merged = {}
    for stat in tag_stats:
        key = stat["label"]
        if key in merged:
            m = merged[key]
            w1, w2 = m["count"], stat["count"]
            total = w1 + w2
            if total > 0:
                m["avg_pax"] = round((m["avg_pax"] * w1 + stat["avg_pax"] * w2) / total, 3)
                m["avg_lf"] = round((m["avg_lf"] * w1 + stat["avg_lf"] * w2) / total, 1)
                m["baseline_pax"] = round((m["baseline_pax"] * w1 + stat["baseline_pax"] * w2) / total, 3)
                m["baseline_lf"] = round((m["baseline_lf"] * w1 + stat["baseline_lf"] * w2) / total, 1)
            m["count"] = total
            m["pct"] = round(m["pct"] + stat["pct"], 1)
            m["pax_lift"] = round(m["avg_pax"] - m["baseline_pax"], 3)
            m["lf_lift"] = round(m["avg_lf"] - m["baseline_lf"], 1)
        else:
            merged[key] = dict(stat)
    tag_stats = list(merged.values())

    for stat in primary_events:
        intelligence = _get_intelligence(stat["event"])
        stat["display_label"] = intelligence.get("display_label", stat["label"])
        stat["label"] = stat["display_label"]  # Force English

    for stat in monthly_data:
        intelligence = _get_intelligence(stat["event"])
        stat["display_label"] = intelligence.get(
            "display_label",
            stat["event"].replace("_", " ").title(),
        )
        stat["label"] = stat["display_label"]  # Force English

    return jsonify({
        "tag_stats": tag_stats,
        "primary_events": primary_events,
        "monthly": monthly_data,
    })


# ─── DEMAND FUNCTIONS API ─────────────────────────────
DEMAND_FUNCS_REPORT = f"{PROJECT_DIR}/reports/demand_functions_report.json"


@app.route("/api/demand-functions")
def api_demand_functions():
    """Return all segment definitions and pre-computed demand curves."""
    report_path = DEMAND_FUNCS_REPORT.replace("/", os.sep)
    if not os.path.exists(report_path):
        return jsonify({"error": "Demand functions report not found. Run build_demand_functions.py first."}), 404
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    return jsonify(report)


@app.route("/api/demand-curves")
def api_demand_curves():
    """Compute demand curves for a specific flight using segment models + actual price data."""
    flight_id = request.args.get("flight_id", "").strip()
    cabin = request.args.get("cabin", "economy").strip().lower()

    # Load demand functions report
    report_path = DEMAND_FUNCS_REPORT.replace("/", os.sep)
    if not os.path.exists(report_path):
        return jsonify({"error": "Demand functions report not found"}), 404
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    segments = report["segments"]
    price_ref = report.get("price_reference", {})
    base_price = price_ref.get(cabin, {}).get("avg", 500)

    # If flight_id provided, get actual price data for that flight
    flight_price = None
    flight_info = {}
    if flight_id:
        con = get_con()
        row = con.execute(f"""
            SELECT
                AVG(CASE WHEN s.pax_sold_today > 0
                    THEN s.ticket_rev_today / s.pax_sold_today ELSE NULL END) AS avg_price,
                MAX(m.capacity) AS capacity,
                MAX(m.departure_airport) AS dep_ap,
                MAX(m.arrival_airport) AS arr_ap,
                MAX(m.region) AS region,
                MAX(m.distance_km) AS distance_km
            FROM read_parquet('{PARQUET_PATH}') s
            LEFT JOIN read_parquet('{METADATA_PATH}') m
                ON s.flight_id = m.flight_id AND LOWER(s.cabin_class) = LOWER(m.cabin_class)
            WHERE s.flight_id = $1 AND LOWER(s.cabin_class) = $2
        """, [flight_id, cabin]).fetchone()
        con.close()

        if row and row[0]:
            flight_price = float(row[0])
            base_price = flight_price
            flight_info = {
                "flight_id": flight_id,
                "cabin": cabin,
                "avg_price": round(flight_price, 2),
                "capacity": int(row[1]) if row[1] else 0,
                "departure_airport": row[2],
                "arrival_airport": row[3],
                "region": row[4],
                "distance_km": _num(row[5]),
            }

    # Generate curves for each segment at this base price
    price_ratios = [round(0.3 + i * 0.1, 1) for i in range(28)]
    dtd_points = [0, 1, 3, 5, 7, 14, 21, 30, 45, 60, 90, 120, 150, 180]

    segment_curves = {}
    combined_demand = []
    combined_revenue = []

    for pr in price_ratios:
        total_q = 0
        total_rev = 0
        for sid, seg in segments.items():
            elast = seg["price_elasticity"]
            peak_dtd = seg["booking_window"]["peak_dtd"]
            dtd_decay = seg["dtd_decay_rate"]
            share = seg["base_share_pct"] / 100

            price_effect = max(pr ** elast, 0.01)
            q = share * price_effect
            rev = pr * base_price * q
            total_q += q
            total_rev += rev
        combined_demand.append({"price_ratio": pr, "price": round(pr * base_price, 2), "demand": round(total_q, 4)})
        combined_revenue.append({"price_ratio": pr, "price": round(pr * base_price, 2), "revenue": round(total_rev, 2)})

    for sid, seg in segments.items():
        elast = seg["price_elasticity"]
        peak_dtd = seg["booking_window"]["peak_dtd"]
        share = seg["base_share_pct"] / 100

        # Price curve
        seg_price_curve = []
        for pr in price_ratios:
            price_effect = max(pr ** elast, 0.01)
            q = share * price_effect
            seg_price_curve.append({
                "price_ratio": pr,
                "price": round(pr * base_price, 2),
                "demand": round(q, 4),
                "revenue": round(pr * base_price * q, 2),
            })

        # DTD curve
        seg_dtd_curve = []
        dtd_sigma = max(peak_dtd * 0.6, 3)
        for dtd in dtd_points:
            import math as _math
            timing = _math.exp(-0.5 * ((dtd - peak_dtd) / dtd_sigma) ** 2)
            if seg["dtd_decay_rate"] >= 0.3 and dtd <= 3:
                timing = max(timing, 0.9)
            seg_dtd_curve.append({
                "dtd": dtd,
                "demand": round(share * timing, 4),
            })

        best_rev = max(seg_price_curve, key=lambda x: x["revenue"])
        segment_curves[sid] = {
            "price_curve": seg_price_curve,
            "dtd_curve": seg_dtd_curve,
            "optimal": {
                "price_ratio": best_rev["price_ratio"],
                "price": best_rev["price"],
                "revenue": best_rev["revenue"],
                "demand": best_rev["demand"],
            },
        }

    # Overall optimal
    best_combined = max(combined_revenue, key=lambda x: x["revenue"])

    return jsonify({
        "base_price": round(base_price, 2),
        "cabin": cabin,
        "flight_info": flight_info,
        "segments": {sid: segments[sid] for sid in segments},
        "segment_curves": segment_curves,
        "combined_demand": combined_demand,
        "combined_revenue": combined_revenue,
        "optimal_combined": {
            "price_ratio": best_combined["price_ratio"],
            "price": best_combined["price"],
            "revenue": best_combined["revenue"],
        },
    })


# ─── FARE CLASSES API ─────────────────────────────────
@app.route("/api/fare-classes")
def api_fare_classes():
    """Return fare class structure, DTD×LF availability matrix, segment matching."""
    import math as _math

    # Fare class definitions
    fare_classes = {
        "V": {"name": "V — Promo", "name_short": "V", "multiplier": 0.5, "protection": 0.0, "open_until_lf": 0.40, "color": "#94a3b8", "description": "Lowest fare. Early booking, price-sensitive passengers."},
        "K": {"name": "K — Discount", "name_short": "K", "multiplier": 0.75, "protection": 0.2, "open_until_lf": 0.60, "color": "#c9a227", "description": "Mid-low fare. Planned travel, flexible dates."},
        "M": {"name": "M — Flex", "name_short": "M", "multiplier": 1.0, "protection": 0.4, "open_until_lf": 0.85, "color": "#6366f1", "description": "Standard fare. Cancel/change flexibility included."},
        "Y": {"name": "Y — Full Fare", "name_short": "Y", "multiplier": 1.5, "protection": 0.6, "open_until_lf": 1.0, "color": "#ef4444", "description": "Highest fare. Full flexibility, last minute."},
    }

    # DTD rules
    dtd_rules = [
        {"dtd_min": 60, "dtd_max": 180, "open": ["V", "K", "M"], "label": "Early Period"},
        {"dtd_min": 30, "dtd_max": 59,  "open": ["K", "M"],      "label": "Mid Period"},
        {"dtd_min": 14, "dtd_max": 29,  "open": ["K", "M", "Y"], "label": "Late Period"},
        {"dtd_min": 7,  "dtd_max": 13,  "open": ["M", "Y"],      "label": "Final Week"},
        {"dtd_min": 0,  "dtd_max": 6,   "open": ["Y"],            "label": "Last Minute"},
    ]

    # Build DTD × LF heatmap matrix
    dtd_points = [0, 1, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 150, 180]
    lf_points = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    heatmap = []
    for dtd in dtd_points:
        row = []
        for lf in lf_points:
            lf_ratio = lf / 100
            # Find DTD rule
            open_classes = ["Y"]
            for rule in dtd_rules:
                if rule["dtd_min"] <= dtd <= rule["dtd_max"]:
                    open_classes = rule["open"]
                    break
            # Filter by LF protection
            available = []
            for fc_id in open_classes:
                fc = fare_classes[fc_id]
                if lf_ratio < fc["open_until_lf"] or fc_id == "Y":
                    available.append(fc_id)
            if not available:
                available = ["Y"]
            # Best fare = cheapest available
            best = available[0]
            row.append({"dtd": dtd, "lf": lf, "available": available, "best_fare": best, "price_mult": fare_classes[best]["multiplier"]})
        heatmap.append(row)

    # Segment → fare class matching
    demand_path = DEMAND_FUNCS_REPORT.replace("/", os.sep)
    segment_matching = []
    if os.path.exists(demand_path):
        with open(demand_path, "r", encoding="utf-8") as f:
            dreport = json.load(f)
        segments = dreport.get("segments", {})
        for sid, seg in segments.items():
            wtp_avg = (seg["wtp_multiplier"]["min"] + seg["wtp_multiplier"]["max"]) / 2
            # Find the most expensive fare class the segment can afford (revenue max)
            best_fc = "V"
            for fc_id in ["V", "K", "M", "Y"]:
                if fare_classes[fc_id]["multiplier"] <= wtp_avg:
                    best_fc = fc_id
            segment_matching.append({
                "segment_id": sid,
                "segment_name": seg["name"],
                "icon": seg["icon"],
                "color": seg["color"],
                "wtp_range": f"{seg['wtp_multiplier']['min']}-{seg['wtp_multiplier']['max']}x",
                "wtp_avg": round(wtp_avg, 2),
                "preferred_fare": best_fc,
                "preferred_fare_name": fare_classes[best_fc]["name"],
                "elasticity": seg["price_elasticity"],
                "booking_window": f"{seg['booking_window']['min_dtd']}-{seg['booking_window']['max_dtd']} days",
            })

    # Price examples per cabin
    price_ref = {}
    if os.path.exists(demand_path):
        price_ref = dreport.get("price_reference", {})

    price_examples = {}
    for cabin in ["economy", "business"]:
        base = price_ref.get(cabin, {}).get("avg", 500 if cabin == "economy" else 1500)
        price_examples[cabin] = {
            fc_id: {"price": round(base * fc["multiplier"], 2), "multiplier": fc["multiplier"]}
            for fc_id, fc in fare_classes.items()
        }
        price_examples[cabin]["base_price"] = round(base, 2)

    return jsonify({
        "fare_classes": fare_classes,
        "dtd_rules": dtd_rules,
        "heatmap": heatmap,
        "dtd_points": dtd_points,
        "lf_points": lf_points,
        "segment_matching": segment_matching,
        "price_examples": price_examples,
    })


# ─── SIMULATION API ───────────────────────────────────
SIM_REPORT = f"{BASE_DIR}/simulation_report.json"


@app.route("/api/simulation")
def api_simulation():
    """Return simulation results: static vs dynamic pricing comparison."""
    report_path = SIM_REPORT.replace("/", os.sep)
    if not os.path.exists(report_path):
        return jsonify({"error": "Simulation report not found. Run run_simulation.py first."}), 404
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Strip daily data if summary_only requested (lighter payload)
    summary_only = request.args.get("summary", "").strip().lower() == "true"
    if summary_only:
        routes = {}
        for k, v in report.get("routes", {}).items():
            route_copy = {key: val for key, val in v.items()}
            route_copy["static"] = {key: val for key, val in v["static"].items() if key != "daily"}
            route_copy["dynamic"] = {key: val for key, val in v["dynamic"].items() if key != "daily"}
            routes[k] = route_copy
        report_copy = {**report, "routes": routes}
        return jsonify(report_copy)

    return jsonify(report)


# ─── RISK / OPPORTUNITY INDEX API ──────────────────────
@app.route("/api/risk-index")
def api_risk_index():
    """Flight risk/opportunity index with pricing action categories.

    Fix: Use real DTD (departure_date - today) instead of MIN(dtd)=0.
    Only show future flights so the risk panel reflects actionable data.
    """
    con = get_con()
    sp = PARQUET_PATH
    cabin_filter = request.args.get("cabin", "").strip().lower()
    cabin_where = f"AND LOWER(s.cabin_class) = '{cabin_filter}'" if cabin_filter else ""

    # REVERTED: Using CURRENT_DATE to only analyze future flights
    flights = con.execute(f"""
        WITH future_flights AS (
            SELECT flight_id, cabin_class, capacity, region, departure_airport, arrival_airport, distance_km
            FROM read_parquet('{METADATA_PATH}')
            WHERE CAST(departure_datetime AS DATE) >= CURRENT_DATE
        ),
        latest_snaps AS (
            SELECT 
                s.flight_id, s.cabin_class, s.dtd, s.pax_sold_cum, s.ticket_rev_cum, s.anc_rev_cum,
                s.pax_last_7d, s.pax_sold_today,
                ROW_NUMBER() OVER (PARTITION BY s.flight_id, s.cabin_class ORDER BY s.dtd ASC) as rn
            FROM read_parquet('{sp}') s
            WHERE s.dtd IS NOT NULL
        )
        SELECT 
            ff.flight_id, ff.cabin_class, ls.dtd, ls.pax_sold_cum, ff.capacity,
            ls.ticket_rev_cum, ls.anc_rev_cum, ff.region, ff.departure_airport, ff.arrival_airport,
            ff.distance_km, ls.pax_last_7d, ls.pax_sold_today
        FROM future_flights ff
        JOIN latest_snaps ls 
          ON LOWER(ff.flight_id) = LOWER(ls.flight_id) 
          AND LOWER(ff.cabin_class) = LOWER(ls.cabin_class)
        WHERE ls.rn = 1 {cabin_where}
        ORDER BY ls.dtd ASC
    """).fetchall()

    print(f"[RiskIndex] Analysis complete for {len(flights)} future flights")

    results = []
    categories = {"price_increase": [], "price_decrease": [], "cancel_risk": []}

    for r in flights:
        # Index mapping from SQL:
        # 0: fid, 1: cabin, 2: dtd, 3: pax_cum, 4: cap, 5: t_rev, 6: a_rev, 
        # 7: reg, 8: dep, 9: arr, 10: dist, 11: p7d, 12: ptod
        fid, cabin, dtd = r[0], r[1], r[2]
        pax_cum = r[3] or 0
        capacity = r[4] or 180
        t_rev, a_rev = r[5] or 0, r[6] or 0
        total_rev = t_rev + a_rev
        
        lf = pax_cum / max(capacity, 1)
        remaining = max(0, capacity - pax_cum)
        rev_per_pax = total_rev / max(pax_cum, 1)
        
        region = r[7] or "Unknown"
        dep_ap, arr_ap = r[8] or "???", r[9] or "???"
        distance = float(r[10]) if r[10] else 1000
        pax_7d = float(r[11]) if r[11] else 0
        pax_today = float(r[12]) if r[12] else 0

        # === RISK/OPPORTUNITY SCORING ===
        # 1. DTD urgency (0-25): closer to departure = more urgent
        if dtd is None:
            dtd_score = 12.5
        elif dtd <= 3:
            dtd_score = 25
        elif dtd <= 7:
            dtd_score = 20
        elif dtd <= 14:
            dtd_score = 15
        elif dtd <= 30:
            dtd_score = 10
        else:
            dtd_score = 5

        # 2. Load factor score (0-25): low LF = more risk
        lf_score = max(0, 25 - lf * 25)  # LF=0 → 25, LF=1 → 0

        # 3. Revenue momentum (0-25): recent booking activity
        momentum = min(pax_7d / max(capacity, 1), 1.0)
        momentum_score = (1 - momentum) * 25  # low activity = high risk

        # 4. Capacity waste (0-25): empty seats cost money
        waste = remaining / max(capacity, 1)
        waste_score = waste * 25  # more empty = more risk

        risk_score = round(dtd_score + lf_score + momentum_score + waste_score, 1)
        opp_score = round(100 - risk_score, 1)

        # Revenue potential = remaining seats × avg revenue per pax
        rev_potential = round(remaining * rev_per_pax, 2)

        # === PRICING CATEGORY (DTD-aware thresholds) ===
        # Expected LF at this DTD: S-curve booking pattern
        # At DTD=180, expect ~5% booked; at DTD=30 ~40%; at DTD=7 ~70%; at DTD=0 ~85%
        if dtd is not None and dtd > 0:
            expected_lf = max(0.05, min(0.95, 1.0 - (dtd / 180.0) ** 0.8))
        else:
            expected_lf = 0.85
        lf_gap = lf - expected_lf  # negative = behind schedule

        if lf >= 0.75 and (dtd is None or dtd >= 3):
            category = "price_increase"
            action = "Price Increase Opportunity"
            reason = f"Load factor {lf*100:.0f}%, strong demand with {dtd or 0} days to departure"
        elif lf_gap < -0.25 and (dtd is not None and dtd <= 30):
            # Significantly behind expected booking curve and close to departure
            category = "cancel_risk"
            action = "Cancellation Risk / Loss Exposure"
            reason = (f"Load factor {lf*100:.0f}% vs expected {expected_lf*100:.0f}% "
                      f"({abs(lf_gap)*100:.0f}pp behind), {dtd} days to departure")
        elif lf_gap < -0.15 and (dtd is not None and dtd <= 60):
            category = "cancel_risk"
            action = "Cancellation Risk / Loss Exposure"
            reason = (f"Load factor {lf*100:.0f}% vs expected {expected_lf*100:.0f}% "
                      f"({abs(lf_gap)*100:.0f}pp behind), {dtd} days to departure")
        elif lf < expected_lf and (dtd is not None and dtd <= 90):
            category = "price_decrease"
            action = "Price Decrease Recommended"
            reason = (f"Load factor {lf*100:.0f}% vs expected {expected_lf*100:.0f}%, "
                      f"stimulation needed with {dtd} days to departure")
        elif lf >= 0.60:
            category = "price_increase"
            action = "Price Increase Opportunity"
            reason = f"Healthy load factor at {lf*100:.0f}%, yield can be optimized"
        else:
            category = "price_decrease"
            action = "Price Decrease Recommended"
            reason = f"Load factor {lf*100:.0f}%, additional demand needs to be stimulated"

        flight_info = {
            "flight_id": fid,
            "cabin": cabin,
            "route": f"{dep_ap}-{arr_ap}",
            "region": region,
            "dtd": dtd,
            "pax_cum": int(pax_cum),
            "capacity": int(capacity),
            "load_factor": round(lf * 100, 1),
            "remaining_seats": remaining,
            "total_rev": round(total_rev, 2),
            "rev_potential": rev_potential,
            "rev_per_pax": round(rev_per_pax, 2),
            "risk_score": risk_score,
            "opp_score": opp_score,
            "category": category,
            "action": action,
            "reason": reason,
            "pax_7d": int(pax_7d),
            "pax_today": int(pax_today),
            "distance_km": int(distance),
        }
        results.append(flight_info)
        categories[category].append(flight_info)

    # Sort each category by risk score
    for cat in categories:
        categories[cat].sort(key=lambda x: x["risk_score"], reverse=True)

    # Summary stats
    summary = {
        "total_flights": len(results),
        "price_increase": {
            "count": len(categories["price_increase"]),
            "avg_lf": round(sum(f["load_factor"] for f in categories["price_increase"]) / max(len(categories["price_increase"]), 1), 1),
            "total_rev_potential": round(sum(f["rev_potential"] for f in categories["price_increase"]), 0),
        },
        "price_decrease": {
            "count": len(categories["price_decrease"]),
            "avg_lf": round(sum(f["load_factor"] for f in categories["price_decrease"]) / max(len(categories["price_decrease"]), 1), 1),
            "total_rev_potential": round(sum(f["rev_potential"] for f in categories["price_decrease"]), 0),
        },
        "cancel_risk": {
            "count": len(categories["cancel_risk"]),
            "avg_lf": round(sum(f["load_factor"] for f in categories["cancel_risk"]) / max(len(categories["cancel_risk"]), 1), 1),
            "total_rev_potential": round(sum(f["rev_potential"] for f in categories["cancel_risk"]), 0),
        },
        "avg_risk_score": round(sum(f["risk_score"] for f in results) / max(len(results), 1), 1),
    }

    # Available cabins
    cabins = con.execute(f"""
        SELECT DISTINCT LOWER(cabin_class) FROM read_parquet('{sp}')
        WHERE cabin_class IS NOT NULL ORDER BY 1
    """).fetchall()

    con.close()

    return jsonify({
        "flights": results[:200],  # limit response size
        "categories": {k: v[:50] for k, v in categories.items()},
        "summary": summary,
        "filters": {"cabins": [c[0] for c in cabins]},
    })


# ─── MANAGER ANALYSIS API ─────────────────────────────
@app.route("/api/manager-analysis")
def api_manager_analysis():
    """Return flights with current pricing data for manager override panel.
    Fixes: date filter, SQL injection, KPI consistency, pagination.
    """
    con = get_con()
    sp = PARQUET_PATH
    cabin_filter = request.args.get("cabin", "").strip().lower()
    date_from = request.args.get("date_from", "").strip()
    date_to = request.args.get("date_to", "").strip()
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 100))

    # Fix #5: Parameterized cabin filter instead of string interpolation
    params = []
    cabin_where = ""
    if cabin_filter:
        params.append(cabin_filter)
        cabin_where = f"AND LOWER(s.cabin_class) = ${len(params)}"

    # Fix #1: Date filter — default next 90 days if not provided
    if not date_from:
        date_from = date.today().isoformat()
    if not date_to:
        date_to = (date.today() + timedelta(days=90)).isoformat()
    date_where = ""
    if date_from:
        params.append(date_from)
        date_where += f" AND CAST(m.departure_datetime AS DATE) >= CAST(${len(params)} AS DATE)"
    if date_to:
        params.append(date_to)
        date_where += f" AND CAST(m.departure_datetime AS DATE) <= CAST(${len(params)} AS DATE)"

    flights = con.execute(f"""
        WITH latest AS (
            SELECT flight_id, cabin_class, MIN(dtd) AS min_dtd
            FROM read_parquet('{sp}')
            WHERE dtd IS NOT NULL
            GROUP BY flight_id, cabin_class
        )
        SELECT
            s.flight_id,
            s.cabin_class,
            s.dtd,
            s.pax_sold_cum,
            m.capacity,
            CASE WHEN m.capacity > 0
                THEN s.pax_sold_cum * 1.0 / m.capacity ELSE 0 END AS load_factor,
            GREATEST(m.capacity - s.pax_sold_cum, 0) AS remaining_seats,
            s.ticket_rev_cum + s.anc_rev_cum AS total_rev,
            CASE WHEN s.pax_sold_cum > 0
                THEN (s.ticket_rev_cum + s.anc_rev_cum) / s.pax_sold_cum
                ELSE 0 END AS rev_per_pax,
            m.departure_airport,
            m.arrival_airport,
            m.region,
            m.distance_km,
            m.flight_number,
            m.departure_datetime
        FROM read_parquet('{sp}') s
        INNER JOIN latest l ON s.flight_id = l.flight_id
            AND s.cabin_class = l.cabin_class AND s.dtd = l.min_dtd
        LEFT JOIN read_parquet('{METADATA_PATH}') m
            ON s.flight_id = m.flight_id AND s.cabin_class = m.cabin_class
        WHERE 1=1 {cabin_where} {date_where}
        ORDER BY m.departure_datetime, s.flight_id, s.cabin_class
    """, params).fetchall()

    results = []
    total_rev = 0
    total_lf = 0

    for r in flights:
        fid = r[0]
        cabin = r[1]
        dtd = r[2]
        pax_cum = int(r[3]) if r[3] else 0
        capacity = int(r[4]) if r[4] else 1
        lf = float(r[5]) if r[5] else 0
        remaining = int(r[6]) if r[6] else 0
        rev = float(r[7]) if r[7] else 0
        rpp = float(r[8]) if r[8] else 0
        dep_ap = r[9] or ""
        arr_ap = r[10] or ""
        region = r[11] or ""
        distance = float(r[12]) if r[12] else 0
        fn = r[13] or ""
        dep_dt = str(r[14])[:10] if r[14] else ""
        route = f"{dep_ap}-{arr_ap}"

        # Compute current dynamic price if pricing engine available
        current_price = rpp  # fallback: historical average
        base_price = rpp
        if SIM_READY and _pricing_engine:
            try:
                inv = {
                    "cabin": cabin.lower(),
                    "route": route,
                    "dep_date": dep_dt,
                    "load_factor": lf,
                    "capacity": capacity,
                    "sold": pax_cum,
                }
                price_result = _pricing_engine.compute_price(inv, dtd or 0)
                current_price = price_result.get("best_price", rpp)
                base_price = price_result.get("base_price", rpp)
            except Exception:
                pass

        total_rev += rev
        total_lf += lf

        results.append({
            "flight_id": fid,
            "flight_number": fn,
            "cabin": cabin,
            "route": route,
            "region": region,
            "dep_date": dep_dt,
            "dtd": dtd,
            "pax_cum": pax_cum,
            "capacity": capacity,
            "load_factor": round(lf * 100, 1),
            "remaining_seats": remaining,
            "total_rev": round(rev, 2),
            "rev_per_pax": round(rpp, 2),
            "current_price": round(current_price, 2),
            "base_price": round(base_price, 2),
            "distance_km": int(distance),
        })

    # Fix #8: KPIs computed over ALL filtered results (same dataset as table)
    n = max(len(results), 1)
    avg_price = sum(f["current_price"] for f in results) / n if results else 0
    summary = {
        "total_flights": len(results),
        "avg_lf": round(total_lf / n * 100, 1),
        "avg_price": round(avg_price, 2),
        "total_rev": round(total_rev, 0),
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, math.ceil(len(results) / per_page)),
    }

    # Pagination — KPIs are over full dataset, table is paginated
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated = results[start_idx:end_idx]

    # Available cabins
    cabins = con.execute(f"""
        SELECT DISTINCT LOWER(cabin_class) FROM read_parquet('{sp}')
        WHERE cabin_class IS NOT NULL ORDER BY 1
    """).fetchall()

    # Available date range
    date_range = con.execute(f"""
        SELECT MIN(CAST(departure_datetime AS DATE)), MAX(CAST(departure_datetime AS DATE))
        FROM read_parquet('{METADATA_PATH}')
        WHERE departure_datetime IS NOT NULL
    """).fetchone()
    con.close()

    return jsonify({
        "flights": paginated,
        "summary": summary,
        "filters": {
            "cabins": [c[0] for c in cabins],
            "date_range": {
                "min": str(date_range[0]) if date_range and date_range[0] else None,
                "max": str(date_range[1]) if date_range and date_range[1] else None,
            },
        },
    })


def _compute_weighted_elasticity(cabin):
    """Fix #2: Compute weighted-average elasticity from calibrated 6-segment data."""
    if not SIM_READY or not _segments:
        return -1.2 if cabin == "economy" else -0.8
    preferred = [seg for seg in _segments.values() if seg.get("preferred_cabin", "economy") == cabin]
    seg_pool = preferred or list(_segments.values())
    total_share = 0
    weighted_sum = 0
    for seg in seg_pool:
        share = seg.get("base_share_pct", 10) / 100
        elas = seg.get("price_elasticity", -1.0)
        total_share += share
        weighted_sum += share * elas
    if total_share > 0:
        return round(weighted_sum / total_share, 4)
    return -1.2 if cabin == "economy" else -0.8


def _compute_expected_demand(flight_id, cabin, route, dep_dt, dtd, capacity, pax_cum,
                             region="", distance_km=3000.0, revenue_so_far=0.0):
    """Fix #4: Use TFT forecast + XGBoost pickup band to estimate remaining demand."""
    remaining_demand = None
    tft_demand = None
    pickup_remaining = None

    if _forecast_bridge:
        try:
            pickup_remaining = _forecast_bridge.predict_remaining_demand(
                route=route.replace("-", "_"),
                cabin=cabin,
                dep_date=datetime.strptime(dep_dt, "%Y-%m-%d").date() if dep_dt else None,
                dtd=dtd,
                capacity=capacity,
                sold=pax_cum,
                region=region,
                distance_km=distance_km,
                revenue_dynamic=revenue_so_far,
            )
        except Exception:
            pass

    # Fix #11: Try TFT forecast (standalone method, always available)
    if _forecast_bridge:
        try:
            tft_total = _forecast_bridge.get_tft_total(
                route.replace("-", "_"), cabin,
                datetime.strptime(dep_dt, "%Y-%m-%d").date() if dep_dt else None
            )
            if tft_total is not None:
                tft_demand = max(0, tft_total - pax_cum)
        except Exception:
            pass

        # Try TFT band for more refined estimate
        try:
            band = _forecast_bridge.get_tft_band(
                route.replace("-", "_"), cabin,
                datetime.strptime(dep_dt, "%Y-%m-%d").date() if dep_dt else None,
                dtd
            )
            if band:
                cum_expected = band.get("tft_total", 0) * band.get("cum_fraction", 0)
                remaining_demand = max(0, band["tft_total"] - cum_expected)
        except Exception:
            pass

    # S-curve heuristic her durumda hesapla (lower-bound olarak)
    expected_final_lf = 1.0 - (max(dtd, 0) / 180.0) ** 1.5
    expected_final_lf = max(0.30, min(expected_final_lf, 0.95))
    heuristic_remaining = max(0.0, capacity * expected_final_lf - pax_cum)
    # Çok düşük "remaining = 0" durumunu engellemek için min taban: kapasitenin %3'ü
    floor_remaining = max(1.0, capacity * 0.03)

    def _safe(v):
        return v if (v is not None and v > 0) else None

    p = _safe(pickup_remaining)
    t = _safe(remaining_demand)
    f = _safe(tft_demand)

    if p is not None and t is not None:
        return max(0.6 * p + 0.4 * t, floor_remaining), "pickup_tft_blend", tft_demand
    if p is not None and f is not None:
        return max(0.7 * p + 0.3 * f, floor_remaining), "pickup_tft_blend", tft_demand
    if p is not None:
        return max(p, floor_remaining), "xgboost_pickup", tft_demand
    if t is not None:
        return max(t, floor_remaining), "tft_band", tft_demand
    if f is not None:
        return max(f, floor_remaining), "tft_forecast", tft_demand

    # Tüm modeller 0 dönerse heuristic kullan
    return max(heuristic_remaining, floor_remaining), "dtd_heuristic", tft_demand


def _compute_sentiment_demand_factor(route):
    """Fix #14: Sentiment factor for demand (not just price)."""
    if not SENTIMENT_READY or not _SENT_CACHE.get("data"):
        return 1.0, 0.0
    arr = route.split("-")[1] if "-" in route else ""
    try:
        city_key = AIRPORT_TO_CITY.get(arr)
        if city_key and city_key in _SENT_CACHE["data"]:
            score = _SENT_CACHE["data"][city_key].get("aggregate", {}).get("composite_score", 0.0)
            # Sentiment affects demand: +1.0 score → +20% demand, -1.0 → -20%
            return 1.0 + score * 0.20, score
    except Exception:
        pass
    return 1.0, 0.0


def _compute_network_recommendations(route, cabin, capacity, pax_cum, dtd, base_price):
    """Fix #12: EMSR-b protection levels and bid prices from network optimizer."""
    if not _network_optimizer:
        return None
    try:
        from pricing_engine import FARE_CLASSES
        open_fares = ["V", "K", "M", "Y"]
        current_prices = {fc: base_price * FARE_CLASSES[fc]["multiplier"] for fc in open_fares}
        bid_price = _network_optimizer.get_bid_price(
            base_price, capacity, pax_cum, open_fares, current_prices
        )
        protection = _network_optimizer.compute_protection_levels(
            base_price, capacity, current_sold=pax_cum, dtd=dtd
        )
        conn_pct = _network_optimizer.get_connecting_pct(route.replace("-", "_"))
        return {
            "bid_price": round(bid_price, 2),
            "protection_levels": protection,
            "connecting_pct": round(conn_pct * 100, 1) if conn_pct else 0,
        }
    except Exception:
        return None


def _estimate_cancellation_noshow(remaining_demand, dtd, cabin):
    """Fix #13: Adjust demand for expected cancellations and no-shows."""
    # Weighted average cancellation rate across fare classes
    cancel_rates = {"V": 0.01, "K": 0.03, "M": 0.08, "Y": 0.12}
    avg_cancel = sum(cancel_rates.values()) / len(cancel_rates)
    # DTD-conditional: early bookings have higher cancellation
    if dtd > 90:
        dtd_factor = 1.8
    elif dtd > 30:
        dtd_factor = 1.2
    elif dtd > 7:
        dtd_factor = 0.8
    else:
        dtd_factor = 0.5
    expected_cancellation_rate = avg_cancel * dtd_factor
    # Weighted average no-show rate across segments
    no_show_rates = {"A": 0.15, "B": 0.05, "C": 0.08, "D": 0.03, "E": 0.07, "F": 0.20}
    avg_noshow = sum(no_show_rates.values()) / len(no_show_rates)
    # Net demand = gross demand reduced by cancellations, increased by overbooking headroom
    net_demand = remaining_demand * (1 - expected_cancellation_rate)
    overbooking_headroom = remaining_demand * avg_noshow  # can overbook by noshow amount
    return {
        "net_remaining_demand": round(net_demand, 1),
        "expected_cancellation_rate": round(expected_cancellation_rate * 100, 1),
        "expected_noshow_rate": round(avg_noshow * 100, 1),
        "overbooking_headroom": round(overbooking_headroom, 1),
        "adjusted_bookable": round(net_demand + overbooking_headroom, 1),
    }


@app.route("/api/manager-override", methods=["POST"])
def api_manager_override():
    """Compute the impact of a manual price adjustment (what-if analysis).
    Fixes: isoelastic formula, segment elasticity, DTD-LF correction,
    XGBoost pickup, TFT, EMSR-b, cancellation/noshow, sentiment-demand.
    """
    data = request.get_json() or {}
    flight_id = data.get("flight_id", "")
    cabin = data.get("cabin", "economy").lower()
    adjustment_pct = float(data.get("adjustment_pct", 0))  # e.g. +15 or -10
    dtd_override = data.get("dtd_override", None)  # optional: simulate different DTD

    if not flight_id:
        return jsonify({"error": "flight_id required"}), 400

    con = get_con()
    # Fix #5: Already parameterized query (was correct here)
    row = con.execute(f"""
        WITH latest AS (
            SELECT flight_id, cabin_class, MIN(dtd) AS min_dtd
            FROM read_parquet('{PARQUET_PATH}')
            WHERE flight_id = $1 AND LOWER(cabin_class) = $2 AND dtd IS NOT NULL
            GROUP BY flight_id, cabin_class
        )
        SELECT
            s.dtd, s.pax_sold_cum, m.capacity,
            CASE WHEN m.capacity > 0 THEN s.pax_sold_cum * 1.0 / m.capacity ELSE 0 END AS lf,
            s.ticket_rev_cum + s.anc_rev_cum AS total_rev,
            CASE WHEN s.pax_sold_cum > 0
                THEN (s.ticket_rev_cum + s.anc_rev_cum) / s.pax_sold_cum ELSE 0 END AS rpp,
            m.departure_airport, m.arrival_airport, m.distance_km,
            m.departure_datetime, m.region
        FROM read_parquet('{PARQUET_PATH}') s
        INNER JOIN latest l ON s.flight_id = l.flight_id
            AND s.cabin_class = l.cabin_class AND s.dtd = l.min_dtd
        LEFT JOIN read_parquet('{METADATA_PATH}') m
            ON s.flight_id = m.flight_id AND s.cabin_class = m.cabin_class
        WHERE s.flight_id = $1 AND LOWER(s.cabin_class) = $2
        LIMIT 1
    """, [flight_id, cabin]).fetchone()
    con.close()

    if not row:
        return jsonify({"error": "Flight not found"}), 404

    dtd_snap = int(row[0]) if row[0] else 0   # snapshot DTD (genelde 0 = terminal state)
    pax_cum = int(row[1]) if row[1] else 0
    capacity = int(row[2]) if row[2] else 300
    lf_real = float(row[3]) if row[3] else 0
    total_rev = float(row[4]) if row[4] else 0
    rpp = float(row[5]) if row[5] else 0
    dep_ap = row[6] or ""
    arr_ap = row[7] or ""
    distance = float(row[8]) if row[8] else 3000
    dep_dt = str(row[9])[:10] if row[9] else ""
    region = row[10] or ""
    route = f"{dep_ap}-{arr_ap}"

    # Anlık gerçek DTD: dep_date - bugün. Snapshot DTD=0 (terminal) bu hesabın yerine
    # geçmesin diye yenisini ÜRET. Pricing/demand bu DTD üzerinden çalışsın.
    from datetime import date as _date
    real_dtd = None
    if dep_dt:
        try:
            real_dtd = (_date.fromisoformat(dep_dt) - _date.today()).days
            if real_dtd < 0:
                real_dtd = 0
        except Exception:
            real_dtd = None

    if dtd_override is not None:
        dtd = int(dtd_override)
    elif real_dtd is not None and real_dtd > 0:
        dtd = real_dtd
    else:
        dtd = dtd_snap

    # Snapshot DTD (genelde 0) ile dtd farklıysa LF/pax_cum gerçekçi olmaz —
    # S-curve ile o DTD için tipik occupancy hesapla.
    use_simulated_lf = (dtd != dtd_snap) or (dtd_override is not None)
    if use_simulated_lf:
        estimated_lf = 1.0 - (max(dtd, 0) / 180.0) ** 1.5
        estimated_lf = max(0.0, min(estimated_lf, 0.95))
        lf = estimated_lf
        pax_cum_sim = int(capacity * lf)
        remaining = max(capacity - pax_cum_sim, 0)
    else:
        lf = lf_real
        remaining = max(capacity - pax_cum, 0)
        pax_cum_sim = pax_cum

    # Compute current dynamic price
    original_price = rpp
    fare_classes = {}
    if SIM_READY and _pricing_engine:
        try:
            inv = {
                "cabin": cabin, "route": route, "dep_date": dep_dt,
                "load_factor": lf, "capacity": capacity, "sold": pax_cum_sim,
            }
            pr = _pricing_engine.compute_price(inv, dtd)
            original_price = pr.get("best_price", rpp)
            fare_classes = pr.get("fare_classes", {})
        except Exception:
            pass

    # Apply manager override
    adjusted_price = round(original_price * (1 + adjustment_pct / 100), 2)

    # Fix #2: Weighted segment elasticity from calibrated data
    elasticity = _compute_weighted_elasticity(cabin)

    # Fix #3: Isoelastic demand formula: Q_new/Q_old = (1 + Δp/100)^ε
    # Fix #9: No max(0) clipping needed — isoelastic formula always > 0
    adjusted_demand_factor = (1 + adjustment_pct / 100) ** elasticity
    demand_change_pct = round((adjusted_demand_factor - 1) * 100, 2)

    # Fix #4: Use XGBoost pickup / TFT for expected remaining demand
    expected_demand, demand_source, tft_demand = _compute_expected_demand(
        flight_id, cabin, route, dep_dt, dtd, capacity, pax_cum_sim,
        region=region, distance_km=distance, revenue_so_far=total_rev
    )
    baseline_remaining = max(expected_demand, 0.0)
    effective_remaining = max(baseline_remaining * adjusted_demand_factor, 0.0)

    # Fix #14: Sentiment-demand coupling
    sentiment_factor, sentiment_score = _compute_sentiment_demand_factor(route)
    baseline_remaining *= sentiment_factor
    effective_remaining *= sentiment_factor

    # Fix #13: Cancellation / no-show / overbooking adjustment
    baseline_cancel_noshow = _estimate_cancellation_noshow(baseline_remaining, dtd, cabin)
    cancel_noshow = _estimate_cancellation_noshow(effective_remaining, dtd, cabin)
    baseline_bookable = baseline_cancel_noshow["adjusted_bookable"]
    effective_remaining = cancel_noshow["adjusted_bookable"]

    # Revenue calculations
    original_rev_remaining = round(original_price * baseline_bookable, 2)
    adjusted_rev_remaining = round(adjusted_price * effective_remaining, 2)
    revenue_delta = round(adjusted_rev_remaining - original_rev_remaining, 2)
    revenue_delta_pct = round(revenue_delta / max(original_rev_remaining, 1) * 100, 1)
    demand_change_pct = round((effective_remaining / max(baseline_bookable, 1e-6) - 1) * 100, 2)

    # Estimated final load factor
    estimated_final_sold = pax_cum_sim + effective_remaining
    estimated_final_lf = round(min(estimated_final_sold / max(capacity, 1), 1.0) * 100, 1)

    # Fare class breakdown with adjusted prices
    adjusted_fares = {}
    for fc_id, fc_info in fare_classes.items():
        orig_p = fc_info.get("price", 0)
        adj_p = round(orig_p * (1 + adjustment_pct / 100), 2)
        adjusted_fares[fc_id] = {
            "name": fc_info.get("name", fc_id),
            "original_price": round(orig_p, 2),
            "adjusted_price": adj_p,
            "open": fc_info.get("open", False),
            "color": fc_info.get("color", "#666"),
        }

    # When DTD is overridden, re-determine open/closed purely from DTD rules
    if dtd_override is not None:
        from pricing_engine import DTD_RULES as _DTD_RULES
        dtd_open = ["Y"]  # fallback
        for rule in _DTD_RULES:
            if rule["dtd_min"] <= dtd <= rule["dtd_max"]:
                dtd_open = rule["open"]
                break
        for fc_id in adjusted_fares:
            adjusted_fares[fc_id]["open"] = fc_id in dtd_open

    # Fix #12: Network optimizer recommendations (EMSR-b, bid price)
    base_price = original_price
    if SIM_READY and _pricing_engine:
        try:
            base_price = _pricing_engine._compute_base_price(cabin, distance, route)
        except Exception:
            pass
    network_rec = _compute_network_recommendations(route, cabin, capacity, pax_cum_sim, dtd, base_price)

    # Fix #11: TFT forecast info
    tft_info = None
    if tft_demand is not None:
        tft_info = {
            "predicted_remaining": round(tft_demand, 1),
            "trend": "rising" if tft_demand > remaining else "falling" if tft_demand < remaining * 0.8 else "stable",
        }

    return jsonify({
        "flight_id": flight_id,
        "cabin": cabin,
        "route": route,
        "adjustment_pct": adjustment_pct,
        "original_price": round(original_price, 2),
        "adjusted_price": adjusted_price,
        "price_delta": round(adjusted_price - original_price, 2),
        "remaining_seats": remaining,
        "demand_change_pct": demand_change_pct,
        "effective_remaining_demand": effective_remaining,
        "baseline_remaining_demand": round(baseline_bookable, 1),
        "original_rev_remaining": original_rev_remaining,
        "adjusted_rev_remaining": adjusted_rev_remaining,
        "revenue_delta": revenue_delta,
        "revenue_delta_pct": revenue_delta_pct,
        "estimated_final_lf": estimated_final_lf,
        "current_lf": round(lf * 100, 1),
        "fare_classes": adjusted_fares,
        "elasticity": elasticity,
        "elasticity_source": "weighted_6_segments" if SIM_READY and _segments else "fallback",
        "demand_source": demand_source,
        "demand_formula": "isoelastic",
        "sentiment_score": sentiment_score,
        "sentiment_demand_factor": round(sentiment_factor, 4),
        "tft_forecast": tft_info,
        "network_recommendation": network_rec,
        "cancellation_noshow": cancel_noshow,
        "baseline_cancellation_noshow": baseline_cancel_noshow,
        "dtd": dtd,
        "dtd_is_simulated": use_simulated_lf,
        "capacity": capacity,
        "pax_cum": pax_cum_sim,
        "total_rev_so_far": round(total_rev, 2),
        "segments_used": {sid: {"name": s.get("name"), "elasticity": s.get("price_elasticity"), "share": s.get("base_share_pct")} for sid, s in _segments.items()} if _segments else None,
    })


@app.route("/api/manager-sensitivity", methods=["POST"])
def api_manager_sensitivity():
    """Fix #10: Server-side sensitivity curves so frontend doesn't duplicate logic."""
    data = request.get_json() or {}
    flight_id = data.get("flight_id", "")
    cabin = data.get("cabin", "economy").lower()
    dtd_override = data.get("dtd_override", None)

    if not flight_id:
        return jsonify({"error": "flight_id required"}), 400

    con = get_con()
    row = con.execute(f"""
        WITH latest AS (
            SELECT flight_id, cabin_class, MIN(dtd) AS min_dtd
            FROM read_parquet('{PARQUET_PATH}')
            WHERE flight_id = $1 AND LOWER(cabin_class) = $2 AND dtd IS NOT NULL
            GROUP BY flight_id, cabin_class
        )
        SELECT s.dtd, s.pax_sold_cum, m.capacity,
               CASE WHEN m.capacity > 0 THEN s.pax_sold_cum * 1.0 / m.capacity ELSE 0 END AS lf,
               m.departure_airport, m.arrival_airport, m.distance_km, m.departure_datetime
        FROM read_parquet('{PARQUET_PATH}') s
        INNER JOIN latest l ON s.flight_id = l.flight_id
            AND s.cabin_class = l.cabin_class AND s.dtd = l.min_dtd
        LEFT JOIN read_parquet('{METADATA_PATH}') m
            ON s.flight_id = m.flight_id AND s.cabin_class = m.cabin_class
        WHERE s.flight_id = $1 AND LOWER(s.cabin_class) = $2
        LIMIT 1
    """, [flight_id, cabin]).fetchone()
    con.close()

    if not row:
        return jsonify({"error": "Flight not found"}), 404

    dtd_snap = int(row[0]) if row[0] else 0
    pax_cum = int(row[1]) if row[1] else 0
    capacity = int(row[2]) if row[2] else 300
    lf_real = float(row[3]) if row[3] else 0
    dep_ap = row[4] or ""
    arr_ap = row[5] or ""
    route = f"{dep_ap}-{arr_ap}"
    dep_dt = str(row[7])[:10] if row[7] else ""

    # Aynı DTD/LF mantığı manager-override ile uyumlu
    from datetime import date as _date
    real_dtd = None
    if dep_dt:
        try:
            real_dtd = (_date.fromisoformat(dep_dt) - _date.today()).days
            if real_dtd < 0:
                real_dtd = 0
        except Exception:
            real_dtd = None
    if dtd_override is not None:
        dtd = int(dtd_override)
    elif real_dtd is not None and real_dtd > 0:
        dtd = real_dtd
    else:
        dtd = dtd_snap
    use_simulated_lf = (dtd != dtd_snap) or (dtd_override is not None)
    if use_simulated_lf:
        lf = max(0.0, min(1.0 - (max(dtd, 0) / 180.0) ** 1.5, 0.95))
        pax_cum = int(capacity * lf)
    else:
        lf = lf_real

    remaining = max(capacity - pax_cum, 0)

    # Get original price
    original_price = 500.0
    if SIM_READY and _pricing_engine:
        try:
            inv = {"cabin": cabin, "route": route, "dep_date": dep_dt,
                   "load_factor": lf, "capacity": capacity, "sold": pax_cum}
            pr = _pricing_engine.compute_price(inv, dtd)
            original_price = pr.get("best_price", 500.0)
        except Exception:
            pass

    elasticity = _compute_weighted_elasticity(cabin)
    expected_demand, _, _ = _compute_expected_demand(
        flight_id, cabin, route, dep_dt, dtd, capacity, pax_cum,
        distance_km=row[6], revenue_so_far=0.0
    )
    sentiment_factor, _ = _compute_sentiment_demand_factor(route)

    # Build sensitivity curve: -30% to +50%
    points = []
    for p in range(-30, 51, 2):
        adj_price = original_price * (1 + p / 100)
        demand_factor = (1 + p / 100) ** elasticity
        baseline_remaining = expected_demand * sentiment_factor
        baseline_cancel = _estimate_cancellation_noshow(baseline_remaining, dtd, cabin)
        adjusted_remaining = baseline_remaining * demand_factor
        adjusted_cancel = _estimate_cancellation_noshow(adjusted_remaining, dtd, cabin)
        eff_demand = adjusted_cancel["adjusted_bookable"]
        rev_delta = adj_price * eff_demand - original_price * baseline_cancel["adjusted_bookable"]
        demand_chg = (eff_demand / max(baseline_cancel["adjusted_bookable"], 1e-6) - 1) * 100
        points.append({
            "pct": p,
            "price": round(adj_price, 2),
            "demand_change_pct": round(demand_chg, 2),
            "revenue_delta": round(rev_delta, 2),
            "effective_demand": round(eff_demand, 1),
        })

    return jsonify({
        "flight_id": flight_id,
        "cabin": cabin,
        "elasticity": elasticity,
        "original_price": round(original_price, 2),
        "expected_demand": round(expected_demand, 1),
        "points": points,
    })


# ─── SENTIMENT PAGE & API ──────────────────────────────
@app.route("/sentiment")
def sentiment_page():
    return render_template("sentiment.html")


@app.route("/api/sentiment/status")
def api_sentiment_status():
    if not SENTIMENT_READY:
        return jsonify({"ready": False, "error": "sentiment module not available", "cities": []})
    deberta_ready = bool(_deberta_clf and _deberta_clf.is_ready())
    return jsonify({
        "ready": True,
        "source": ("Google News RSS + DeBERTa-v3-small + keyword events"
                   if deberta_ready else
                   "Google News RSS + Keyword Classifier (DeBERTa loading)"),
        "deberta_active": deberta_ready,
        "deberta_model": _deberta_clf.MODEL_NAME if _deberta_clf else None,
        "last_update": _SENT_CACHE.get("last_update"),
        "loading": _SENT_CACHE.get("loading", False),
        "cities_count": len(SENT_CITIES),
    })


@app.route("/api/sentiment/all")
def api_sentiment_all():
    """Tum sehirlerin sentiment ozetini doner. Bellekten aninda."""
    if not SENTIMENT_READY:
        return jsonify({"error": "sentiment module not available"}), 503

    if _SENT_CACHE["data"]:
        normalized = {
            city_key: _normalize_sentiment_city_payload(city_key, city_data)
            for city_key, city_data in _SENT_CACHE["data"].items()
        }
        return jsonify(normalized)

    if _SENT_CACHE["loading"]:
        return jsonify({"_loading": True, "_message": "Sentiment data is loading..."}), 202

    return jsonify({"_loading": True, "_message": "The scheduler has not run yet"}), 202


@app.route("/api/sentiment/<city_key>")
def api_sentiment_city(city_key):
    if not SENTIMENT_READY:
        return jsonify({"error": "sentiment module not available"}), 503
    if city_key not in SENT_CITIES:
        return jsonify({"error": f"Unknown city: {city_key}"}), 404

    # Cache'ten dondur
    if _SENT_CACHE["data"] and city_key in _SENT_CACHE["data"]:
        return jsonify(_normalize_sentiment_city_payload(city_key, _SENT_CACHE["data"][city_key]))

    cfg = SENT_CITIES[city_key]
    return jsonify({
        "city": city_key, "label": cfg["label"],
        "flag": cfg["flag"], "color": cfg["color"],
        "aggregate": {"composite_score": 0, "alert_level": "low", "article_count": 0},
        "articles": [],
    })


# ─── DYNAMIC PRICING & SIMULATION ENGINE ─────────────────
SIM_READY = False
try:
    from pricing_engine import PricingEngine
    from simulation_engine import SimulationEngine

    # Route mesafeleri yukle
    _route_distances = {}
    _route_meta = {}
    try:
        _meta_con = duckdb.connect()
        _meta_rows = _meta_con.execute(f"""
            SELECT DISTINCT
                departure_airport || '_' || arrival_airport as route_key,
                distance_km, region, cabin_class, capacity
            FROM read_parquet('{METADATA_PATH}')
            WHERE departure_airport = 'IST'
        """).fetchall()
        _meta_con.close()
        for r in _meta_rows:
            _route_distances[r[0]] = r[1]
            _route_meta[r[0] + "_" + r[3]] = {"distance_km": r[1], "region": r[2], "capacity": r[4]}
    except Exception as e:
        print(f"[Pricing] Route metadata load failed: {e}")

    # Segment verileri yukle
    _segments = {}
    _dtd_curves = {}
    _demand_report_path = f"{PROJECT_DIR}/reports/demand_functions_report.json"
    if os.path.exists(_demand_report_path.replace("/", os.sep)):
        with open(_demand_report_path.replace("/", os.sep), encoding="utf-8") as f:
            _dreport = json.load(f)
        _segments = _dreport.get("segments", {})
        for sid in _dreport.get("curves", {}):
            _dtd_curves[sid] = _dreport["curves"][sid].get("dtd_demand", [])

    # Engine'leri olustur
    _pricing_engine = PricingEngine(
        segments=_segments,
        route_distances=_route_distances,
        sentiment_cache=_SENT_CACHE,
        airport_to_city=AIRPORT_TO_CITY if SENTIMENT_READY else {},
    )
    # ForecastBridge — modelleri simulasyona bagla
    _forecast_bridge = None
    try:
        from forecast_bridge import ForecastBridge
        # n_flights_map: TFT rota-gun tahminini ucus basina bolmek icin
        _n_flights_map = {}
        if TFT_DATA is not None and "n_flights" in TFT_DATA.columns:
            _nf = TFT_DATA.groupby("entity_id")["n_flights"].mean()
            _n_flights_map = {eid: float(nf) for eid, nf in _nf.items()}
            print(f"[Bridge] n_flights_map: {len(_n_flights_map)} entities (avg {sum(_n_flights_map.values())/max(len(_n_flights_map),1):.1f} flights/day)")

        _forecast_bridge = ForecastBridge(
            tft_predictions_df=TFT_PRED,
            twostage_clf=TWOSTAGE_CLF,
            twostage_reg=TWOSTAGE_REG,
            twostage_features=TWOSTAGE_FEATURES,
            pickup_model=PICKUP_MODEL,
            pickup_features=PICKUP_FEATURES,
            route_meta=_route_meta,
            n_flights_map=_n_flights_map,
        )
        print(f"[Bridge] ForecastBridge ready: TFT={len(_forecast_bridge._tft_cache)} entries")
    except Exception as e:
        print(f"[Bridge] ForecastBridge not available: {e}")

    # NetworkOptimizer — O&D + EMSR-b
    _network_optimizer = None
    try:
        from network_optimizer import NetworkOptimizer
        # Connecting oranlarini tft_route_daily'den cek
        _conn_pcts = {}
        if TFT_DATA is not None:
            _cpct = TFT_DATA.groupby("entity_id")["connecting_pct"].mean()
            for eid, cpct in _cpct.items():
                parts = eid.rsplit("_", 1)
                if len(parts) == 2:
                    route_key = parts[0]
                    _conn_pcts[route_key] = float(cpct)
        _network_optimizer = NetworkOptimizer(
            route_distances=_route_distances,
            segments=_segments,
            connecting_pcts=_conn_pcts,
        )
        print(f"[Network] O&D optimizer ready: {len(_conn_pcts)} routes with connecting data")
    except Exception as e:
        print(f"[Network] O&D optimizer not available: {e}")

    _sim_engine = SimulationEngine(
        pricing_engine=_pricing_engine,
        forecast_bridge=_forecast_bridge,
        network_optimizer=_network_optimizer,
    )

    SIM_READY = True
    print(f"[Pricing] Engine ready: {len(_route_distances)} routes, {len(_segments)} segments")
except Exception as e:
    print(f"[Pricing] Engine not available: {e}")
    import traceback; traceback.print_exc()


# ─── SIMULATION API ──────────────────────────────────────
@app.route("/api/sim/report/pdf")
def api_sim_report_pdf():
    """Generate NLG-powered PDF report from simulation results."""
    from flask import send_file
    if not SIM_READY or _sim_engine.state != "completed":
        return jsonify({"error": "No completed simulation"}), 400
    try:
        from report_generator.collector import collect
        from report_generator.analyzer import analyze
        from report_generator import nlg_engine, charts
        from report_generator.pdf_builder import build_pdf
        import io

        # Collect
        rd = collect(_sim_engine, _SENT_CACHE)
        # Analyze
        insights = analyze(rd)
        # Generate NLG text
        nlg_sections = {
            "executive_summary": nlg_engine.generate_executive_summary(rd, insights),
            "revenue": nlg_engine.generate_revenue_section(rd, insights),
            "lf": nlg_engine.generate_lf_section(rd, insights),
            "fareclass": nlg_engine.generate_fareclass_section(rd, insights),
            "sentiment": nlg_engine.generate_sentiment_section(rd, insights),
            "recommendations": nlg_engine.generate_recommendations(rd, insights),
        }
        # Charts
        chart_paths = charts.generate_all(rd)
        # Build PDF
        pdf_bytes = build_pdf(rd, insights, nlg_sections, chart_paths)
        # Cleanup chart temp files
        for p in chart_paths.values():
            try:
                os.remove(p)
            except Exception:
                pass

        return send_file(io.BytesIO(pdf_bytes), mimetype='application/pdf',
                         as_attachment=True, download_name='Seatwise_Simulation_Report.pdf')
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/simulation")
def simulation_page():
    return render_template("simulation.html")


@app.route("/competition")
def competition_page():
    return render_template("competition.html")


@app.route("/booking")
def booking_page():
    return render_template("booking.html")


@app.route("/api/routes")
def api_routes():
    """Tum rotalari dondur."""
    preferred_order = [
        "IST-LHR", "IST-MAD", "IST-JFK", "IST-DXB", "IST-CDG", "IST-FRA", "IST-BCN", "IST-FCO", "IST-MUC", "IST-NCE",
        "IST-AMM", "IST-CAI", "IST-MAN", "IST-TLV", "IST-BEY", "IST-JED", "IST-RUH", "IST-DOH", "IST-BAH", "IST-KWI",
        "IST-AUH", "IST-HRG", "IST-CMN", "IST-RAK", "IST-NBO", "IST-MBA", "IST-LOS", "IST-ABV", "IST-JNB", "IST-CPT",
        "IST-NRT", "IST-KIX", "IST-ICN", "IST-PEK", "IST-PVG", "IST-SIN", "IST-BKK", "IST-HKT", "IST-DEL", "IST-BOM",
        "IST-LAX", "IST-ORD", "IST-MIA", "IST-YYZ", "IST-YVR", "IST-MEX", "IST-GRU", "IST-EZE", "IST-GIG", "IST-MXP",
    ]
    routes = set()
    for rk in _route_meta:
        parts = rk.rsplit("_", 1)
        route = parts[0].replace("_", "-")
        routes.add(route)

    ordered = [route for route in preferred_order if route in routes]
    extras = sorted(route for route in routes if route not in ordered)
    return jsonify({"routes": ordered + extras})


def _build_flights_list(date_range, cabins, routes_filter=None):
    """Ucus listesi olustur — sim/start ve sim/monte-carlo ortak kullanir."""
    flights = []
    for rk, meta in _route_meta.items():
        parts = rk.rsplit("_", 1)
        route_key = parts[0]
        cabin = parts[1]
        if cabin not in cabins:
            continue
        route_str = route_key.replace("_", "-")
        if routes_filter and route_str not in routes_filter:
            continue
        start = datetime.strptime(date_range[0], "%Y-%m-%d").date()
        end = datetime.strptime(date_range[1], "%Y-%m-%d").date()
        d = start
        while d <= end:
            flights.append({
                "flight_id": f"{route_str}_{d.isoformat()}_{cabin}",
                "route": route_str,
                "cabin": cabin,
                "dep_date": d.isoformat(),
                "capacity": meta["capacity"],
                "distance_km": meta["distance_km"],
                "region": meta["region"],
            })
            d += timedelta(days=1)
    return flights


@app.route("/api/sim/start", methods=["POST"])
def api_sim_start():
    if not SIM_READY:
        return jsonify({"error": "Simulation engine not available"}), 503
    data = request.get_json() or {}
    date_range = data.get("date_range", ["2026-07-01", "2026-12-31"])
    speed = data.get("speed", 1440)
    cabins = data.get("cabins", ["economy", "business"])
    routes_filter = data.get("routes")

    flights = _build_flights_list(date_range, cabins, routes_filter)
    _sim_engine.initialize(flights, date_range, speed, _dtd_curves)
    _sim_engine.start()
    return jsonify({"status": "started", "flights": len(_sim_engine.inventory)})


@app.route("/api/sim/pause", methods=["POST"])
def api_sim_pause():
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    _sim_engine.pause()
    return jsonify({"status": _sim_engine.state})


@app.route("/api/sim/resume", methods=["POST"])
def api_sim_resume():
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    _sim_engine.resume()
    return jsonify({"status": _sim_engine.state})


@app.route("/api/sim/reset", methods=["POST"])
def api_sim_reset():
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    _sim_engine.reset()
    return jsonify({"status": _sim_engine.state})


@app.route("/api/sim/speed", methods=["POST"])
def api_sim_speed():
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    data = request.get_json() or {}
    _sim_engine.set_speed(data.get("speed", 1440))
    return jsonify({"speed": _sim_engine.clock.speed})


@app.route("/api/sim/jump", methods=["POST"])
def api_sim_jump():
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    data = request.get_json() or {}
    target = data.get("date")
    if not target:
        return jsonify({"error": "date required"}), 400
    _sim_engine.jump_to(target)
    return jsonify({"status": _sim_engine.state, "sim_date": _sim_engine.clock.today().isoformat()})


@app.route("/api/sim/status")
def api_sim_status():
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    return jsonify(_sim_engine.get_status())


@app.route("/api/sim/flights")
def api_sim_flights():
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    return jsonify({"flights": _sim_engine.get_flights_list()})


@app.route("/api/sim/flight/<path:flight_key>")
def api_sim_flight_detail(flight_key):
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    detail = _sim_engine.get_flight_detail(flight_key)
    if not detail:
        return jsonify({"error": "Flight not found"}), 404
    return jsonify(detail)


@app.route("/api/sim/competition")
def api_sim_competition():
    """Rakip havayolları pazar payı ve durum özeti."""
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    if not _sim_engine.competitor_manager:
        return jsonify({"error": "No competitors configured"}), 404
    summary = _sim_engine.competitor_manager.get_summary()
    summary["our_stats"] = {
        "lost_to_PC": _sim_engine.stats.get("lost_to_PC", 0),
        "lost_to_EK": _sim_engine.stats.get("lost_to_EK", 0),
        "stolen_from_PC": _sim_engine.stats.get("stolen_from_PC", 0),
        "stolen_from_EK": _sim_engine.stats.get("stolen_from_EK", 0),
    }
    return jsonify(summary)


@app.route("/api/sim/competition/flight/<path:flight_key>")
def api_sim_competition_flight(flight_key):
    """Tek ucus icin 3 havayolunun karsilastirmali durumu."""
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    if not _sim_engine.competitor_manager:
        return jsonify({"error": "No competitors configured"}), 404
    with _sim_engine.lock:
        inv = _sim_engine.inventory.get(flight_key)
        if not inv:
            return jsonify({"error": "Flight not found"}), 404
    result = _sim_engine.competitor_manager.get_flight_competition(flight_key, inv)
    return jsonify(result)


@app.route("/api/sim/inject", methods=["POST"])
def api_sim_inject():
    """Manuel bot enjeksiyonu."""
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    data = request.get_json() or {}
    sales = _sim_engine.inject_bots(
        data.get("flight_key", ""),
        data.get("segment", "D"),
        data.get("count", 10),
    )
    return jsonify({"sales": sales})


@app.route("/api/sim/override", methods=["POST"])
def api_sim_override():
    """Fare class elle ac/kapa."""
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    data = request.get_json() or {}
    ok = _sim_engine.override_fare_class(
        data.get("flight_key", ""),
        data.get("fare_class", "V"),
        data.get("action", "open"),
    )
    return jsonify({"success": ok})


# ── Monte Carlo state ─────────────────────────────────────
_mc_state = {"running": False, "progress": 0, "total": 0, "result": None, "error": None}


@app.route("/api/sim/monte-carlo", methods=["POST"])
def api_sim_monte_carlo():
    """Monte Carlo simulasyon baslat — arka planda calisir."""
    if not SIM_READY:
        return jsonify({"error": "Simulation engine not available"}), 503
    if _mc_state["running"]:
        return jsonify({"error": "Monte Carlo already running", "progress": _mc_state["progress"]}), 409

    data = request.get_json() or {}
    date_range = data.get("date_range", ["2026-07-01", "2026-07-31"])
    cabins = data.get("cabins", ["economy"])
    routes_filter = data.get("routes", ["IST-LHR"])
    n_runs = min(data.get("n_runs", 50), 200)  # max 200 run

    flights = _build_flights_list(date_range, cabins, routes_filter)
    if not flights:
        return jsonify({"error": "No flights match filters"}), 400

    _mc_state["running"] = True
    _mc_state["progress"] = 0
    _mc_state["total"] = n_runs
    _mc_state["result"] = None
    _mc_state["error"] = None

    def _run_mc():
        def _on_progress(current, total):
            _mc_state["progress"] = current
            _mc_state["total"] = total
        try:
            result = _sim_engine.run_monte_carlo(
                flights, date_range, n_runs=n_runs,
                speed=14400, dtd_curves=_dtd_curves,
                on_progress=_on_progress
            )
            _mc_state["result"] = result
        except Exception as e:
            _mc_state["error"] = str(e)
            import traceback; traceback.print_exc()
        finally:
            _mc_state["running"] = False

    t = threading.Thread(target=_run_mc, daemon=True, name="monte-carlo")
    t.start()
    return jsonify({"status": "started", "flights": len(flights), "n_runs": n_runs})


@app.route("/api/sim/monte-carlo/status")
def api_sim_monte_carlo_status():
    """Monte Carlo ilerleme durumu."""
    if _mc_state["result"]:
        return jsonify({
            "running": False,
            "completed": True,
            "result": _mc_state["result"],
        })
    if _mc_state["error"]:
        return jsonify({"running": False, "completed": False, "error": _mc_state["error"]})
    return jsonify({
        "running": _mc_state["running"],
        "completed": False,
        "progress": _mc_state["progress"],
        "total": _mc_state["total"],
    })


@app.route("/api/pricing/quote")
def api_pricing_quote():
    """Canli fiyat teklifi (booking sayfasi icin)."""
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    flight_key = request.args.get("flight_key", "").strip()
    segment = request.args.get("segment", "").strip() or None

    inv = _sim_engine.inventory.get(flight_key)
    if not inv:
        return jsonify({"error": "Flight not found"}), 404

    dtd = _sim_engine.clock.dtd(inv["dep_date"])
    session_str = request.args.get("session", "")
    session_info = json.loads(session_str) if session_str else None

    quote = _pricing_engine.compute_price(inv, dtd, segment_id=segment, session_info=session_info)
    return jsonify(quote)


@app.route("/api/pricing/book", methods=["POST"])
def api_pricing_book():
    """Bilet satin alma (booking sayfasindan)."""
    if not SIM_READY:
        return jsonify({"error": "Not available"}), 503
    data = request.get_json() or {}
    result = _sim_engine.book_human(
        flight_key=data.get("flight_key", ""),
        fare_class=data.get("fare_class", "Y"),
        session_info=data.get("session"),
    )
    return jsonify(result)


# ─── LIVE NETWORK ACTIVITY (dashboard ticker) ─────────────────
import collections as _collections
import threading as _threading
import time as _time_mod
import random as _random_mod
# tek isimle mevcut isimleri override etmemek için: lokal kullanım
time = _time_mod
random = _random_mod

_LIVE_EVENTS = _collections.deque(maxlen=300)
_LIVE_FLIGHTS = []
_LIVE_LOCK = _threading.Lock()
_LIVE_STARTED = False
_LIVE_TODAY = None  # date — uçuşlar bugün için seçilir, gün dönerse yenilenir


# Saat-bazlı kümülatif booking yüzdesi (gün başından beri biriken)
# Sabah ve akşam peak'leri ile 24 saatlik curve.
def _live_day_progress_pct(hour_float):
    """0-24 saat arası verilen ana kadar günün toplam satışının kümülatif yüzdesi."""
    # Saatlik dilim payları (toplam = 100)
    hourly_share = [
        0.5, 0.3, 0.2, 0.2, 0.4, 1.0,    # 00-05
        2.5, 4.5, 6.5, 7.0, 6.5, 6.0,    # 06-11 (sabah peak)
        5.5, 5.5, 5.0, 5.5, 6.0, 6.5,    # 12-17
        7.5, 8.0, 7.0, 5.5, 4.0, 2.0,    # 18-23 (akşam peak)
    ]
    cum = 0.0
    h_int = int(hour_float)
    for i in range(min(h_int, 23)):
        cum += hourly_share[i]
    if h_int < 24:
        cum += hourly_share[h_int] * (hour_float - h_int)
    return min(100.0, cum)


def _live_init_flights():
    """Bugün kalkan TÜM aktif uçuşları dahil et (cap: 400) + saat-curve'e göre pre-fill."""
    from datetime import date as _d, datetime as _dt
    today = _d.today()
    now = _dt.now()
    hour_float = now.hour + now.minute / 60.0
    day_progress = _live_day_progress_pct(hour_float) / 100.0  # 0-1 arası
    con = get_con()
    rows = []
    try:
        # Önce bugün için ara, yoksa en yakın günden seç. Tüm cabinleri dahil et.
        for offset in (0, 1, -1, 2, -2, 3, -3, 7, 14):
            target = today + timedelta(days=offset)
            rows = con.execute(f"""
                SELECT m.flight_id, m.flight_number, m.departure_airport, m.arrival_airport,
                       m.cabin_class, m.capacity, m.region, m.distance_km, m.departure_datetime
                FROM read_parquet('{METADATA_PATH}') m
                WHERE CAST(m.departure_datetime AS DATE) = $1
                  AND m.flight_number IS NOT NULL
                ORDER BY RANDOM()
                LIMIT 400
            """, [target.isoformat()]).fetchall()
            if rows:
                break
    finally:
        con.close()

    flights = []
    for r in rows:
        cap = int(r[5] or 300)
        cabin = (r[4] or "economy").lower()
        distance = float(r[7] or 3000)
        # Final LF target — biz daha yüksek (~75), eco daha yüksek (~82) civarı
        final_lf_target = 0.75 if cabin == "business" else random.uniform(0.78, 0.92)
        # "Bugünün satışı" = günün şu anına kadar curve oranıyla dolması gereken payı
        # (Cumulative kapanışın günün payı oranıyla scale'i)
        bookings_today_target = int(round(cap * final_lf_target * day_progress))
        # Bilet başına ortalama fiyat (kabin × distance bazlı, 1.0 ortalama mult)
        avg_fare = max(distance * 0.35, 800) if cabin == "business" else max(distance * 0.085, 180)
        # Total revenue today (bugünün şu ana kadarki satışı × avg fare × ~1.0 mult ortalaması)
        rev_so_far = bookings_today_target * avg_fare * random.uniform(0.92, 1.08)
        # FC breakdown — gerçekçi karışım
        fc_today = {
            "V": int(bookings_today_target * 0.18),
            "K": int(bookings_today_target * 0.32),
            "M": int(bookings_today_target * 0.34),
            "Y": int(bookings_today_target * 0.16),
        }
        flights.append({
            "flight_id": r[0],
            "flight_number": r[1],
            "route": f"{r[2]}-{r[3]}",
            "departure_airport": r[2],
            "arrival_airport": r[3],
            "cabin": cabin,
            "capacity": cap,
            "region": r[6] or "",
            "distance_km": distance,
            "dep_datetime": str(r[8]) if r[8] else "",
            "sold": bookings_today_target,
            "revenue": round(rev_so_far, 2),
            "bookings_today": bookings_today_target,
            "fc_breakdown": fc_today,
            "lf_history": [],
            "avg_fare": avg_fare,
            "final_lf_target": final_lf_target,
        })
    return flights


_LIVE_FC_DIST = [("V", 0.25, 0.50), ("K", 0.35, 0.75), ("M", 0.30, 1.00), ("Y", 0.10, 1.50)]
# demand_functions_report.json ile birebir hizalı (share + elasticity dataset'ten)
_LIVE_SEGMENTS_META = {
    "A": {"label": "Business late-booker","share": 15, "elasticity": -0.3, "age": (32, 56), "biz_ratio": 0.40, "ff_skew": 0.55, "purpose": "Corporate trip"},
    "B": {"label": "VFR / hometown",      "share": 20, "elasticity": -0.7, "age": (22, 65), "biz_ratio": 0.05, "ff_skew": 0.18, "purpose": "Visiting family"},
    "C": {"label": "Group / fixed-date",  "share": 12, "elasticity": -0.5, "age": (30, 60), "biz_ratio": 0.15, "ff_skew": 0.30, "purpose": "Conference / event"},
    "D": {"label": "Bargain Hunter",      "share": 25, "elasticity": -1.5, "age": (24, 55), "biz_ratio": 0.02, "ff_skew": 0.10, "purpose": "Leisure / vacation"},
    "E": {"label": "Budget flexible",     "share": 18, "elasticity": -2.2, "age": (18, 32), "biz_ratio": 0.01, "ff_skew": 0.05, "purpose": "Backpack / student"},
    "F": {"label": "Emergency / tender",  "share": 10, "elasticity": -0.1, "age": (25, 55), "biz_ratio": 0.20, "ff_skew": 0.25, "purpose": "Urgent travel"},
}
_LIVE_CHANNELS = [
    ("Website",         339),
    ("Mobile App",      289),
    ("Travel Agency",   160),
    ("Call Center",     139),
    ("Corporate",        70),
]
_LIVE_FF = [("None", 50), ("Silver", 25), ("Gold", 17), ("Elite", 8)]

_LIVE_NAMES_TR = ["Mehmet", "Ayşe", "Ali", "Fatma", "Mustafa", "Zeynep", "Ahmet", "Elif", "Hüseyin", "Selin",
                   "Kaan", "Deniz", "Burak", "Cem", "Ece", "Doruk", "Yiğit", "Kerem", "Berke", "Aslı", "Naz",
                   "Buse", "Eren", "Mert", "Sena", "İlayda", "Tuna", "Ozan", "Kaya", "Pelin"]
_LIVE_SURNAMES_TR = ["Yılmaz", "Kaya", "Demir", "Şahin", "Çelik", "Aydın", "Öztürk", "Aslan", "Doğan", "Kara",
                      "Koç", "Arslan", "Yıldız", "Polat", "Acar", "Ünal", "Korkmaz", "Avcı", "Şimşek", "Erdoğan"]
_LIVE_NAMES_INT = ["John", "Maria", "James", "Anna", "Mohammed", "Fatima", "David", "Sophie", "Liu", "Yuki",
                    "Hans", "Ingrid", "Carlos", "Sofia", "Ivan", "Olga", "Pierre", "Marie", "Hassan", "Layla",
                    "Raj", "Priya", "Chen", "Mei", "Daniel", "Sarah", "Lukas", "Emma", "Pablo", "Lucia"]
_LIVE_SURNAMES_INT = ["Smith", "Johnson", "García", "Rossi", "Müller", "Brown", "Wang", "Zhang", "Martínez",
                       "Kim", "Sato", "Wright", "Wilson", "Petrov", "Schmidt", "Nguyen", "Khan", "Patel"]

# Hangi havalimanı için yerel ad havuzu — Türk hub'dan kalkanlar için ~%55 TR, %45 INT
_TR_HUBS = {"IST", "SAW", "ESB", "ADB", "AYT"}


def _live_pick_weighted(items_with_weights):
    weights = [w for _, w in items_with_weights]
    total = sum(weights)
    r = random.uniform(0, total)
    acc = 0
    for item, w in items_with_weights:
        acc += w
        if r <= acc:
            return item
    return items_with_weights[-1][0]


def _live_generate_passenger(segment_id, route, cabin, fare_class):
    seg = _LIVE_SEGMENTS_META[segment_id]
    # Yaş — segment dağılımı
    age_lo, age_hi = seg["age"]
    age = random.randint(age_lo, age_hi)
    # Cinsiyet
    gender = random.choice(["M", "F"])
    # Ad havuzu — uçuşun bir ucunda Türkiye varsa ağırlıklı TR
    dep_apt, _, arr_apt = route.partition("-")
    tr_route = dep_apt in _TR_HUBS or arr_apt in _TR_HUBS
    use_tr = tr_route and random.random() < 0.55
    if use_tr:
        first = random.choice(_LIVE_NAMES_TR)
        last = random.choice(_LIVE_SURNAMES_TR)
        nationality = "TR"
    else:
        first = random.choice(_LIVE_NAMES_INT)
        last = random.choice(_LIVE_SURNAMES_INT)
        nationality = random.choice(["GB", "DE", "FR", "US", "IT", "ES", "NL", "AE", "SA", "QA", "AZ", "RU", "JP", "CN"])
    # KVKK/GDPR — decision maker yolcu kimliklerini görmemeli.
    # Ad: ilk harf + 4 yıldız, Soyad: ilk harf + 3 yıldız
    masked_name = f"{first[0]}**** {last[0]}***"
    # Frequent flyer — segment'e göre eğimli
    if random.random() < seg["ff_skew"]:
        ff_status = _live_pick_weighted([("Silver", 50), ("Gold", 35), ("Elite", 15)])
    else:
        ff_status = "None"
    # Channel
    channel = _live_pick_weighted(_LIVE_CHANNELS)
    # PNR + booking class
    pnr = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=6))
    return {
        "name": masked_name,
        "age": age,
        "gender": gender,
        "nationality": nationality,
        "ff_status": ff_status,
        "channel": channel,
        "pnr": pnr,
        "purpose": seg["purpose"],
        "segment_label": seg["label"],
    }


def _live_compute_price(flight, fare_class):
    """Pricing engine ile gerçek dynamic price hesapla (varsa); yoksa basit fallback."""
    fc_mult = next(m for f, _, m in _LIVE_FC_DIST if f == fare_class)
    if SIM_READY and _pricing_engine:
        try:
            from datetime import date as _d
            real_dtd = (_d.fromisoformat(flight["dep_datetime"][:10]) - _d.today()).days
            inv = {
                "cabin": flight["cabin"],
                "route": flight["route"],
                "dep_date": flight["dep_datetime"][:10],
                "load_factor": flight["sold"] / flight["capacity"] if flight["capacity"] else 0,
                "capacity": flight["capacity"],
                "sold": flight["sold"],
            }
            pr = _pricing_engine.compute_price(inv, max(0, real_dtd))
            fares = pr.get("fare_classes", {})
            if fare_class in fares:
                return round(fares[fare_class].get("price", 0) * random.uniform(0.97, 1.03), 2)
        except Exception:
            pass
    # Fallback
    if flight["cabin"] == "business":
        base = max(flight["distance_km"] * 0.35, 800)
    else:
        base = max(flight["distance_km"] * 0.08, 150)
    return round(base * fc_mult * random.uniform(0.92, 1.12), 2)


def _live_generate_event():
    if not _LIVE_FLIGHTS:
        return None
    flight = None
    for _ in range(3):
        cand = random.choice(_LIVE_FLIGHTS)
        if cand["sold"] < int(cand["capacity"] * 0.96):
            flight = cand
            break
    if flight is None:
        return None

    # Fare class — ağırlıklı (DTD yakınsa Y/M daha sık)
    real_dtd = 30
    try:
        from datetime import date as _d
        real_dtd = max(0, (_d.fromisoformat(flight["dep_datetime"][:10]) - _d.today()).days)
    except Exception:
        pass
    if real_dtd <= 7:
        fc_weights = [("V", 0.05), ("K", 0.15), ("M", 0.45), ("Y", 0.35)]
    elif real_dtd <= 30:
        fc_weights = [("V", 0.15), ("K", 0.30), ("M", 0.40), ("Y", 0.15)]
    else:
        fc_weights = [(fc, w) for fc, w, _ in _LIVE_FC_DIST]
    fc = _live_pick_weighted(fc_weights)
    segment = _live_pick_weighted([(s, m["share"]) for s, m in _LIVE_SEGMENTS_META.items()])

    # Group size — bookings_enriched dataset'inden: avg 2.0 pax/PNR, max 5
    # Empirik dağılım: ~42% solo, ~33% couple, ~14% 3-pax, ~8% 4-pax, ~3% 5-pax
    group_size = _live_pick_weighted([(1, 42), (2, 33), (3, 14), (4, 8), (5, 3)])
    # Kapasite kontrolü
    group_size = min(group_size, flight["capacity"] - flight["sold"])
    if group_size <= 0:
        return None

    price_each = _live_compute_price(flight, fc)
    total_price = round(price_each * group_size, 2)
    passenger = _live_generate_passenger(segment, flight["route"], flight["cabin"], fc)

    flight["sold"] += group_size
    flight["revenue"] += total_price
    flight["bookings_today"] += 1
    flight.setdefault("fc_breakdown", {"V": 0, "K": 0, "M": 0, "Y": 0})
    flight["fc_breakdown"][fc] = flight["fc_breakdown"].get(fc, 0) + group_size
    flight.setdefault("lf_history", [])
    flight["lf_history"].append(round(flight["sold"] / flight["capacity"] * 100, 1))
    if len(flight["lf_history"]) > 30:
        flight["lf_history"] = flight["lf_history"][-30:]

    seat_row = random.randint(1, 30)
    seat_letter = random.choice("ABCDEF")

    event = {
        "ts": int(time.time() * 1000),
        "type": "booking",
        "flight_id": flight["flight_id"],
        "flight_number": flight["flight_number"],
        "route": flight["route"],
        "cabin": flight["cabin"],
        "fare_class": fc,
        "segment": segment,
        "price": price_each,
        "total_price": total_price,
        "group_size": group_size,
        "seat": f"{seat_row}{seat_letter}",
        "lf_after": round(flight["sold"] / flight["capacity"] * 100, 1),
        "passenger": passenger,
    }
    with _LIVE_LOCK:
        _LIVE_EVENTS.append(event)
    return event


def _live_loop():
    while True:
        try:
            global _LIVE_TODAY, _LIVE_FLIGHTS
            from datetime import date as _d
            today = _d.today()
            # Gün döndüyse veya hiç başlatılmadıysa: uçuşları yenile
            if _LIVE_TODAY != today or not _LIVE_FLIGHTS:
                with _LIVE_LOCK:
                    _LIVE_FLIGHTS = _live_init_flights()
                    _LIVE_TODAY = today
            # Burst: her tik'te 1-4 event (network ölçeğinde gerçekçilik)
            burst = random.randint(1, 4)
            for _ in range(burst):
                _live_generate_event()
        except Exception as e:
            try:
                print(f"[LiveFeed] error: {e}", flush=True)
            except Exception:
                pass
        time.sleep(random.uniform(0.8, 2.0))


def _live_start():
    global _LIVE_STARTED
    if _LIVE_STARTED:
        return
    _LIVE_STARTED = True
    t = _threading.Thread(target=_live_loop, daemon=True, name="live-feed")
    t.start()
    try:
        print("[LiveFeed] started — polling 10 flights every 1.5-4.5s", flush=True)
    except Exception:
        pass


@app.route("/api/live-feed")
def api_live_feed():
    """Dashboard'da arama altındaki canlı network ticker."""
    try:
        since = int(request.args.get("since", 0))
    except ValueError:
        since = 0
    with _LIVE_LOCK:
        events = [e for e in _LIVE_EVENTS if e["ts"] > since]
        flights_summary = [{
            "flight_number": f["flight_number"],
            "route": f["route"],
            "cabin": f["cabin"],
            "region": f.get("region", ""),
            "dep_datetime": f.get("dep_datetime", ""),
            "lf": round(f["sold"] / f["capacity"] * 100, 1),
            "sold": f["sold"],
            "capacity": f["capacity"],
            "revenue": round(f["revenue"], 0),
            "bookings_today": f["bookings_today"],
            "fc_breakdown": f.get("fc_breakdown", {"V": 0, "K": 0, "M": 0, "Y": 0}),
            "lf_history": f.get("lf_history", [])[-20:],
        } for f in _LIVE_FLIGHTS]

        # Network insights — son 5 dakika
        cutoff = int(time.time() * 1000) - 5 * 60 * 1000
        recent = [e for e in _LIVE_EVENTS if e["ts"] >= cutoff]
        recent_revenue = sum(e["total_price"] for e in recent)
        recent_pax = sum(e["group_size"] for e in recent)
        cabin_mix = {"economy": 0, "business": 0}
        fc_mix = {"V": 0, "K": 0, "M": 0, "Y": 0}
        channel_mix = {}
        seg_mix = {}
        nat_mix = {}
        for e in recent:
            cabin_mix[e["cabin"]] = cabin_mix.get(e["cabin"], 0) + e["group_size"]
            fc_mix[e["fare_class"]] = fc_mix.get(e["fare_class"], 0) + e["group_size"]
            ch = e["passenger"]["channel"]
            channel_mix[ch] = channel_mix.get(ch, 0) + e["group_size"]
            sl = e["passenger"]["segment_label"]
            seg_mix[sl] = seg_mix.get(sl, 0) + e["group_size"]
            nat = e["passenger"]["nationality"]
            nat_mix[nat] = nat_mix.get(nat, 0) + e["group_size"]
        # Top routes (revenue desc)
        route_rev = {}
        for f in _LIVE_FLIGHTS:
            if f["revenue"] > 0:
                route_rev[f["route"]] = route_rev.get(f["route"], 0) + f["revenue"]
        top_routes = sorted(route_rev.items(), key=lambda x: -x[1])[:5]

        # Alerts: ≥97% LF veya hızlı dolan uçuşlar
        alerts = []
        for f in _LIVE_FLIGHTS:
            lf = f["sold"] / f["capacity"] * 100 if f["capacity"] else 0
            if lf >= 97:
                alerts.append({"type": "sold_out", "flight": f["flight_number"], "route": f["route"], "lf": round(lf, 1)})
            elif lf >= 92 and f["bookings_today"] >= 8:
                alerts.append({"type": "filling_fast", "flight": f["flight_number"], "route": f["route"], "lf": round(lf, 1), "bookings": f["bookings_today"]})

        # Daily target & progress
        from datetime import datetime as _dt
        now_dt = _dt.now()
        hour_float = now_dt.hour + now_dt.minute / 60.0
        day_pct = _live_day_progress_pct(hour_float)
        # Daily revenue target = sum over flights of (capacity × final_lf × avg_fare)
        daily_target = sum(f.get("capacity", 0) * f.get("final_lf_target", 0.82) * f.get("avg_fare", 250)
                           for f in _LIVE_FLIGHTS)
        revenue_today = sum(f.get("revenue", 0) for f in _LIVE_FLIGHTS)
        bookings_today = sum(f.get("bookings_today", 0) for f in _LIVE_FLIGHTS)

        insights = {
            "recent_revenue": round(recent_revenue, 0),
            "recent_pax": recent_pax,
            "yield_per_pax": round(recent_revenue / recent_pax, 2) if recent_pax else 0,
            "velocity_per_min": round(len(recent) / 5.0, 1),
            "cabin_mix": cabin_mix,
            "fc_mix": fc_mix,
            "channel_mix": dict(sorted(channel_mix.items(), key=lambda x: -x[1])[:5]),
            "segment_mix": dict(sorted(seg_mix.items(), key=lambda x: -x[1])[:6]),
            "nationality_mix": dict(sorted(nat_mix.items(), key=lambda x: -x[1])[:8]),
            "top_routes": [{"route": r, "revenue": round(rv, 0)} for r, rv in top_routes],
            "alerts": alerts[:10],
            "daily_target": round(daily_target, 0),
            "revenue_today": round(revenue_today, 0),
            "bookings_today_total": bookings_today,
            "day_progress_pct": round(day_pct, 1),
            "expected_now": round(daily_target * day_pct / 100.0, 0),
            "hour_label": now_dt.strftime("%H:%M"),
        }
    return jsonify({
        "events": events[-150:],
        "flights": flights_summary,
        "insights": insights,
        "now": int(time.time() * 1000),
    })


# Live feed'i app boot'ta başlat
_live_start()


if __name__ == "__main__":
    v_label = "V2 (ticket + ancillary)" if USE_V2 else "V1 (legacy)"
    fc_label = "ON" if FORECAST_READY else "OFF"
    sent_label = "ON" if SENTIMENT_READY else "OFF"
    sim_label = "ON" if SIM_READY else "OFF"
    debug_mode = os.getenv("SEATWISE_DASHBOARD_DEBUG", "0") == "1"
    print(f"\nSeatwise Dashboard Dashboard -- {v_label} | Forecast: {fc_label} | Sentiment: {sent_label} | Sim: {sim_label}")
    print(f"   Snapshot: {PARQUET_PATH}")
    print(f"   Metadata: {METADATA_PATH}")
    print(f"   URL: http://localhost:5005\n")
    app.run(
        debug=debug_mode,
        host="0.0.0.0",
        port=5005,
        use_reloader=False,
        threaded=True,
    )
