#!/usr/bin/env python3
# =========================
# Fuel/EV-aware route planner — Part 1/3
# Flask app skeleton + config + helpers + parsing + geocoding + form UI
# =========================

from gettext import npgettext
import os
import json
import math
import re
import pickle
from typing import List, Dict, Tuple, Optional

from flask import Flask, request, render_template_string, jsonify
from dotenv import load_dotenv
import googlemaps
import polyline

# --------------------------
#  Load .env
# --------------------------
load_dotenv()

API_KEY = (os.getenv("GOOGLE_MAPS_API_KEY") or "").strip()
MAPBOX_TOKEN = (os.getenv("MAPBOX_TOKEN") or "").strip()

# --------------------------
#  ML config (no Lincub anywhere)
# --------------------------
ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", "ml_route_model.pkl")
ML_TRAIN_FILE = os.getenv("ML_TRAIN_FILE", "ml_training_data.jsonl")

# --------------------------
#  Configurables (defaults)
# --------------------------
DEFAULT_STEP_INTERVAL = 5          # places_nearby every Nth step to control API usage
DEFAULT_PLACES_RADIUS_M = 5000     # meters

# ICE defaults
DEFAULT_FUEL_THRESHOLD_PCT = 25
DEFAULT_TANK_CAPACITY_L = 50.0
DEFAULT_KM_PER_L = 15.0

# EV defaults
DEFAULT_CHARGE_THRESHOLD_PCT = 20
DEFAULT_BATTERY_KWH = 64.0
DEFAULT_KWH_PER_100KM = 16.5       # 16.5 kWh/100km → 6.06 km/kWh

# Services for "Take Break"
ALL_SERVICE_TYPES = [
    ("rest_stop", "Rest stop"),
    ("cafe", "Cafe"),
    ("restaurant", "Restaurant"),
    ("lodging", "Lodging"),
    ("tourist_attraction", "Attraction"),
    ("convenience_store", "Convenience"),
    ("supermarket", "Supermarket"),
    ("pharmacy", "Pharmacy"),
    ("park", "Park"),
    ("bar", "Bar"),
]

UK_POSTCODE_RE = re.compile(
    r"^\s*([A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}|GIR\s*0AA|BFPO\s*\d{1,4}|[A-Z]{1,2}\d[A-Z]?)\s*$",
    re.IGNORECASE
)

# --------------------------
#  Helpers
# --------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def unique_places(results: List[Dict]) -> List[Dict]:
    seen = set(); out = []
    for r in results:
        key = (r.get('name','?'), round(r['lat'], 6), round(r['lng'], 6))
        if key not in seen:
            seen.add(key); out.append(r)
    return out

def normalize_uk_query(q: str) -> str:
    q = q.strip()
    if UK_POSTCODE_RE.match(q) and "uk" not in q.lower():
        return f"{q}, UK"
    return q

def best_place_label_from_components(components):
    prios = ["locality", "postal_town", "administrative_area_level_2", "administrative_area_level_1"]
    by_type = {}
    for c in components:
        for t in c.get("types", []):
            by_type.setdefault(t, c.get("long_name") or c.get("short_name"))
    for t in prios:
        if t in by_type: return by_type[t]
    return None

def reverse_place_label(gmaps_client, lat, lng):
    try:
        resp = gmaps_client.reverse_geocode(
            (lat, lng),
            result_type=["locality", "postal_town", "administrative_area_level_2", "administrative_area_level_1"]
        ) or gmaps_client.reverse_geocode((lat, lng))
        if resp:
            label = best_place_label_from_components(resp[0].get("address_components", []))
            if label: return label
            fmt = resp[0].get("formatted_address", "")
            if fmt: return fmt.split(",")[0]
    except Exception:
        pass
    return None

def is_latlng(s: str) -> bool:
    return bool(re.match(r"^\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*$", s or ""))

def geocode_gb(gmaps_client, query: str) -> Tuple[float,float,str]:
    """Accepts UK address/postcode OR 'lat,lng' string."""
    if is_latlng(query):
        lat, lng = [float(x) for x in query.split(",")]
        label = reverse_place_label(gmaps_client, lat, lng) or f"{lat:.5f},{lng:.5f}"
        return (lat, lng, label)
    query = normalize_uk_query(query)
    resp = gmaps_client.geocode(query, components={"country": "GB"})
    if not resp:
        raise RuntimeError(f"Could not locate '{query}'. Try 'SW1A 1AA', a city, or allow geolocation.")
    loc = resp[0]["geometry"]["location"]
    return (loc["lat"], loc["lng"], resp[0].get("formatted_address", query))

# ---- Efficiency parsing
def parse_efficiency_to_kmpl(s: str, default_kmpl: float = DEFAULT_KM_PER_L) -> float:
    """ICE: accepts '15', '15 km/l', '40 mpg', '40 usmpg', '40 ukmpg', '6.8 l/100km' -> km/l"""
    if not s:
        return default_kmpl
    txt = s.strip().lower().replace(" ", "")
    m = re.match(r"^([0-9]*\.?[0-9]+)l/?100km$", txt)
    if m:
        l_per_100 = float(m.group(1))
        return 100.0 / l_per_100 if l_per_100 > 0 else default_kmpl
    m = re.match(r"^([0-9]*\.?[0-9]+)(km/?l)?$", txt)
    if m and (txt.endswith("km/l") or not any(u in txt for u in ["mpg", "l/100"])):  # default to km/l
        return float(m.group(1))
    m_us = re.match(r"^([0-9]*\.?[0-9]+)(us)?mpg$", txt)
    m_uk = re.match(r"^([0-9]*\.?[0-9]+)(uk)?mpg$", txt)
    if m_us:
        mpg = float(m_us.group(1))
        return mpg * 1.609344 / 3.78541
    if m_uk:
        mpg = float(m_uk.group(1))
        return mpg * 1.609344 / 4.54609
    try:
        return float(s)
    except Exception:
        return default_kmpl

def parse_ev_efficiency_to_kwh_per_100km(s: str, default_kwh_per_100km: float = DEFAULT_KWH_PER_100KM) -> float:
    """
    EV: accepts:
      '16.5' or '16.5 kwh/100km' -> kWh/100km
      '6 mi/kwh'                 -> kWh/100km
      '155 wh/km'                -> kWh/100km
      '6.2 km/kwh'               -> kWh/100km
    """
    if not s: return default_kwh_per_100km
    txt = s.strip().lower().replace(" ", "")
    m = re.match(r"^([0-9]*\.?[0-9]+)(kwh/?100km)?$", txt)
    if m and "mi/kwh" not in txt and "wh/km" not in txt and "km/kwh" not in txt:
        return float(m.group(1))
    m = re.match(r"^([0-9]*\.?[0-9]+)mi/?kwh$", txt)
    if m:
        km_per_kwh = float(m.group(1)) * 1.609344
        return 100.0 / km_per_kwh if km_per_kwh > 0 else default_kwh_per_100km
    m = re.match(r"^([0-9]*\.?[0-9]+)km/?kwh$", txt)
    if m:
        km_per_kwh = float(m.group(1))
        return 100.0 / km_per_kwh if km_per_kwh > 0 else default_kwh_per_100km
    m = re.match(r"^([0-9]*\.?[0-9]+)wh/?km$", txt)
    if m:
        wh_per_km = float(m.group(1))
        return (wh_per_km / 1000.0) * 100.0
    try:
        return float(s)
    except Exception:
        return default_kwh_per_100km

# --------------------------
#  Stronger & faster ML utils
#  - Ridge regression (closed-form) + standardization
#  - Adam online updates for /ml/rate (few steps)
#  - Quick-fit on recent rows
# --------------------------
import pickle
try:
    import numpy as _np
except Exception:
    _np = None

ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", "ml_route_model.pkl")
ML_TRAIN_FILE = os.getenv("ML_TRAIN_FILE", "ml_training_data.jsonl")

# ---- I/O helpers ----
def _safe_dump_jsonl(path: str, row: Dict):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:
        pass

def _load_jsonl_rows(path: str) -> List[Dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line:
                    try: rows.append(json.loads(line))
                    except Exception: pass
    except Exception:
        return []
    return rows

# ---- Features (fixed order) ----
_FEATURE_KEYS = ["total_km", "duration_s", "vehicle_type", "stops", "avg_leg_km"]

def _features_to_vector(features: Dict) -> Tuple[List[str], List[float]]:
    return _FEATURE_KEYS, [float(features.get(k, 0.0)) for k in _FEATURE_KEYS]

# ---- Model object (dict for pickle-compat) ----
def _new_model(d: int):
    return {
        "w": [0.0]*d,       # weights
        "b": 0.0,           # bias
        "mu": [0.0]*d,      # feature means
        "sigma": [1.0]*d,   # feature stds (>=1e-8)
        "lambda": 1e-2,     # ridge strength
        # Adam state for online updates
        "adam_m": [0.0]*d,
        "adam_v": [0.0]*d,
        "adam_b": 0.0,      # bias' momentum (separate scalar)
        "adam_bv": 0.0,     # bias' second moment
        "t": 0              # adam step count
    }

def ml_save_model(M: Dict, path: str = ML_MODEL_PATH) -> bool:
    try:
        with open(path, "wb") as f:
            pickle.dump(M, f)
        return True
    except Exception:
        return False

def ml_load_model(path: str = ML_MODEL_PATH) -> Optional[Dict]:
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return None

from typing import List

def _standardize_vec(x: List[float], mu: List[float], sigma: List[float]) -> List[float]:
    """Return a standardized feature vector as a plain Python list (works with or without NumPy)."""
    if _np is None:
        out = []
        for i in range(len(x)):
            s = sigma[i] if sigma[i] > 1e-8 else 1.0
            out.append((x[i] - mu[i]) / s)
        return out
    # NumPy fast path, then convert to list for consistency
    xn = _np.array(x, dtype=float)
    mu_n = _np.array(mu, dtype=float)
    sg_n = _np.array(sigma, dtype=float)
    sg_n = _np.where(sg_n < 1e-8, 1.0, sg_n)
    return ((xn - mu_n) / sg_n).tolist()


def _fit_standardizer(X: List[List[float]]) -> Tuple[List[float], List[float]]:
    if not X:
        return [0.0]*len(_FEATURE_KEYS), [1.0]*len(_FEATURE_KEYS)
    if _np is None:
        d = len(X[0])
        mu = [sum(row[j] for row in X)/len(X) for j in range(d)]
        # std
        sigma = []
        for j in range(d):
            var = sum((row[j]-mu[j])**2 for row in X)/max(1,len(X)-1)
            sigma.append(math.sqrt(var) if var>1e-16 else 1.0)
        return mu, sigma
    Xn = _np.array(X, dtype=float)
    mu = Xn.mean(axis=0)
    sg = Xn.std(axis=0, ddof=1)
    sg = _np.where(sg < 1e-8, 1.0, sg)
    return mu.tolist(), sg.tolist()

# ---- Ridge fit (closed-form): minimize ||y - (Xz @ w + b)||^2 + λ||w||^2
def _ridge_fit(X: List[List[float]], y: List[float], lam: float = 1e-2):
    d = len(X[0])
    mu, sigma = _fit_standardizer(X)

    # standardize X
    Xz = [ _standardize_vec(row, mu, sigma) for row in X ]
    if _np is None:
        # small, pure-Python normal eq (use Gauss-Jordan on (X^T X + λI))
        # Build augmented matrix for w plus bias by augmenting with ones:
        # Solve [Xz 1] θ = y  with ridge only on first d weights
        # Implementation detail: do ridge on weights; handle bias separately by centering y.
        ym = sum(y)/len(y)
        y0 = [yi-ym for yi in y]
        # Build G = Xz^T Xz + λI  and rhs = Xz^T y0
        G = [[0.0]*d for _ in range(d)]
        rhs = [0.0]*d
        for i in range(len(Xz)):
            xi = Xz[i]
            for a in range(d):
                rhs[a] += xi[a]*y0[i]
                for b in range(d):
                    G[a][b] += xi[a]*xi[b]
        for j in range(d):
            G[j][j] += lam
        # Solve G w = rhs (naive Gauss-Jordan)
        w = rhs[:]
        # Forward elimination
        for k in range(d):
            piv = G[k][k] if abs(G[k][k])>1e-12 else 1e-12
            inv = 1.0/piv
            for j in range(k,d):
                G[k][j]*=inv
            w[k]*=inv
            for i in range(d):
                if i==k: continue
                m = G[i][k]
                for j in range(k,d):
                    G[i][j]-=m*G[k][j]
                w[i]-=m*w[k]
        b = ym  # bias recovers mean
        return w, b, mu, sigma
    else:
        Xn = _np.array(Xz, dtype=float)
        yn = _np.array(y, dtype=float)
        n, d = Xn.shape
        I = _np.eye(d)
        A = Xn.T @ Xn + lam * I
        w = _np.linalg.solve(A, Xn.T @ yn)
        # recover bias as intercept on standardized space:
        # Fit b to minimize MSE: mean residual with w fixed
        b = float((yn - Xn @ w).mean())
        return w.tolist(), b, mu, sigma

# ---- Prediction ----
def _linear_predict(M: Dict, x: List[float]) -> float:
    w = M["w"]; b = float(M["b"])
    z = _standardize_vec(x, M["mu"], M["sigma"])
    if _np is None:
        y = sum(w[j]*z[j] for j in range(len(w))) + b
    else:
        y = float(_np.dot(_np.array(w), _np.array(z)) + b)
    # squash to [0,1]
    return max(0.0, min(1.0, y))

# Runtime model (load once)
_ML_MODEL = ml_load_model() or _new_model(len(_FEATURE_KEYS))

# ---- Optional: auto-fit after K new logs ----
AUTO_FIT_EVERY = int(os.getenv("ML_AUTO_FIT_EVERY", "50"))  # retrain after every 50 new rows
def _maybe_autofit():
    try:
        if AUTO_FIT_EVERY <= 0: return
        rows = _load_jsonl_rows(ML_TRAIN_FILE)
        if not rows: return
        if len(rows) % AUTO_FIT_EVERY == 0:
            M = ml_fit_ridge(rows, lam=_ML_MODEL.get("lambda", 1e-2))
            ml_save_model(M, ML_MODEL_PATH)
            # Runtime global model, loaded once at startup
            _ML_MODEL = ml_load_model() or _new_model(len(_FEATURE_KEYS))

    except Exception:
        pass

def predict_route_score(features: Dict) -> float:
    keys, vec = _features_to_vector(features)
    try:
        return float(_linear_predict(_ML_MODEL, vec))
    except Exception:
        return 0.5

def log_user_route_features(features: Dict, meta: Dict = None):
    _safe_dump_jsonl(ML_TRAIN_FILE, {"features": features, "meta": (meta or {})})

# ---- Label extraction with better heuristics ----
def _row_to_training_example(row: Dict) -> Optional[Tuple[List[float], float]]:
    try:
        feat = row.get("features") or {}
        meta = row.get("meta") or {}
        # explicit label first
        for k in ("label", "score", "rating"):
            if k in meta:
                y = float(meta[k])
                if 0.0 <= y <= 1.0:
                    _, x = _features_to_vector(feat)
                    return x, y
        # robust heuristic: shorter+faster+fewer stops; EV/ICE treated equally.
        total_km = float(feat.get("total_km", 0.0))
        duration_s = float(feat.get("duration_s", 0.0))
        stops = float(feat.get("stops", 0.0))
        km_term  = 1.0 - min(total_km / 800.0, 1.0)          # cap at 800 km
        dur_term = 1.0 - min(duration_s / (10*3600.0), 1.0)  # cap at 10 h
        stop_term= 1.0 - min(stops / 8.0, 1.0)               # cap at 8 stops
        y = 0.55*km_term + 0.35*dur_term + 0.10*stop_term
        y = max(0.0, min(1.0, y))
        _, x = _features_to_vector(feat)
        return x, y
    except Exception:
        return None

# ---- Public fit helpers ----
def ml_fit_ridge(rows: List[Dict], lam: float = 1e-2) -> Dict:
    pairs = []
    for r in rows:
        ex = _row_to_training_example(r)
        if ex: pairs.append(ex)
    if not pairs:
        raise RuntimeError("No usable training rows.")
    X, y = zip(*pairs)
    w, b, mu, sigma = _ridge_fit(list(X), list(y), lam)
    M = _new_model(len(_FEATURE_KEYS))
    M.update({"w": w, "b": b, "mu": mu, "sigma": sigma, "lambda": float(lam)})
    return M

# ---- Adam online step on a single labeled example ----
def _adam_step(M: Dict, x: List[float], y_true: float, lr: float = 1e-2, beta1=0.9, beta2=0.999, eps=1e-8, lam: float = None):
    # standardize
    z = _standardize_vec(x, M["mu"], M["sigma"])
    # forward
    if _np is None:
        y_pred = sum(M["w"][j]*z[j] for j in range(len(z))) + M["b"]
    else:
        y_pred = float(_np.dot(_np.array(M["w"]), _np.array(z)) + M["b"])
    # loss: 0.5 (y_pred - y)^2 + 0.5 * lam * ||w||^2
    e = (y_pred - y_true)
    dL_dw = [e * z[j] + (lam if (lam is not None) else M["lambda"]) * M["w"][j] for j in range(len(z))]
    dL_db = e

    # Adam state
    M["t"] += 1
    t = M["t"]
    for j in range(len(z)):
        M["adam_m"][j] = beta1*M["adam_m"][j] + (1-beta1)*dL_dw[j]
        M["adam_v"][j] = beta2*M["adam_v"][j] + (1-beta2)*(dL_dw[j]**2)
        m_hat = M["adam_m"][j] / (1 - beta1**t)
        v_hat = M["adam_v"][j] / (1 - beta2**t)
        M["w"][j] -= lr * m_hat / (math.sqrt(v_hat) + eps)
    # bias
    M["adam_b"]  = beta1*M["adam_b"]  + (1-beta1)*dL_db
    M["adam_bv"] = beta2*M["adam_bv"] + (1-beta2)*(dL_db**2)
    mb_hat = M["adam_b"]  / (1 - beta1**t)
    vb_hat = M["adam_bv"] / (1 - beta2**t)
    M["b"] -= lr * mb_hat / (math.sqrt(vb_hat) + eps)

    # clamp predictions to [0,1] implicitly by not clamping params; we clamp on predict.

# global runtime model
_ML_MODEL = ml_load_model() or _new_model(len(_FEATURE_KEYS))

# --------------------------
#  HTML (Form)
# --------------------------
FORM_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Fuel/EV Aware Route Planner — Plan a trip</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    html, body { height: 100%; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background:#0b1220; color:#e6edf3; margin:0; }
    .wrap { max-width: 1000px; margin: 40px auto; padding: 0 16px; }
    .card { background: #0f172a; border:1px solid #1f2a44; border-radius:16px; padding:24px; box-shadow: 0 10px 30px rgba(0,0,0,.3); }
    h1 { margin:0 0 14px; font-size: 28px; }
    p.lead { margin:0 0 20px; opacity:.9 }
    .grid { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:16px; }
    label { font-weight:600; display:block; margin-bottom:6px; }
    input[type=text], input[type=number], select {
      width:100%; background:#0b1220; color:#e6edf3; border:1px solid #23314f; border-radius:10px; padding:10px 12px; outline:none;
    }
    input[type=text]:focus, input[type=number]:focus, select:focus { border-color:#4f46e5; box-shadow: 0 0 0 3px rgba(79,70,229,.25); }
    fieldset { border:1px solid #23314f; border-radius:12px; padding:12px; }
    legend { opacity:.8; padding:0 6px; }
    .svc { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap:10px; margin-top:8px; }
    .row { display:flex; gap:8px; align-items:center; }
    .btn { background:#4f46e5; color:#fff; border:none; border-radius:10px; padding:10px 14px; font-weight:700; cursor:pointer; }
    .btn:hover { filter:brightness(1.1); }
    .muted { font-size:12px; opacity:.8; }
    .danger { color:#ef4444; }
    .note { margin-top:10px; font-size:12px; opacity:.8; }
    .footer { margin-top:16px; display:flex; gap:10px; justify-content:flex-end; }
    .geo-btn { margin-top:6px;background:#0ea5e9;color:#fff;border:none;padding:6px 10px;border-radius:6px;cursor:pointer; }
    .geo-btn:hover { filter:brightness(1.1); }
    .subgrid { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:12px; }
    .help { font-size: 12px; opacity: .85; margin-top: 4px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Fuel/EV-aware route planner</h1>
      <p class="lead">Enter your trip details. Use your current location for origin or destination if you like.</p>

      {% if error %}
        <p class="danger"><b>Error:</b> {{ error }}</p>
      {% endif %}

      <form method="post" action="{{ url_for('plan') }}">
        <div class="grid">
          <!-- Origin / Destination -->
          <div>
            <label>Origin (UK address, postcode, or "lat,lng")</label>
            <input type="text" id="origin-input" name="origin" placeholder="e.g., SW1A 1AA or '51.501,-0.142'" required value="{{ defaults.origin }}">
            <button type="button" id="btn-my-origin" class="geo-btn" color="red">📍 Use My Location (Origin)</button>
          </div>
          <div>
            <label>Destination (UK address, postcode, or "lat,lng")</label>
            <input type="text" id="destination-input" name="destination" placeholder="e.g., BS1 5TR or '51.45,-2.58'" required value="{{ defaults.destination }}">
             
            <button type="button" id="btn-my-dest" class="geo-btn" style="background-color: red">📍 Use My Location (Destination)</button>
          </div>

          <!-- Vehicle Type + Mode -->
          <div>
            <label>Vehicle type</label>
            <select name="vehicle_type" id="vehicle-type">
              <option value="ice" {% if defaults.vehicle_type == 'ice' %}selected{% endif %}>ICE (Petrol/Diesel)</option>
              <option value="ev"  {% if defaults.vehicle_type == 'ev' %}selected{% endif %}>EV (Electric)</option>
            </select>
          </div>
          <div>
            <label>Car mode (baseline)</label>
            <select name="car_mode">
              <option value="normal" {% if defaults.car_mode == 'normal' %}selected{% endif %}>Normal</option>
              <option value="eco"    {% if defaults.car_mode == 'eco' %}selected{% endif %}>Eco</option>
              <option value="sport"  {% if defaults.car_mode == 'sport' %}selected{% endif %}>Sport</option>
              <option value="heavy"  {% if defaults.car_mode == 'heavy' %}selected{% endif %}>Heavy load</option>
            </select>
            <div class="help">Mode affects efficiency baseline. You can tune further in the HUD later.</div>
          </div>

          <!-- Search Radius + Sampling -->
          <div>
            <label>Places search radius (m)</label>
            <input type="number" name="radius_m" min="1000" step="100" value="{{ defaults.radius_m }}">
          </div>
          <div>
            <label>Places sampling (every Nth step)</label>
            <input type="number" name="step_interval" min="1" step="1" value="{{ defaults.step_interval }}">
          </div>
        </div>

        <!-- ICE Settings -->
        <div id="ice-fields">
          <fieldset style="margin-top:20px;">
            <legend>ICE Settings</legend>
            <div class="subgrid">
              <div>
                <label>Fuel level at start (%)</label>
                <input type="number" name="fuel_pct" min="0" max="100" step="1" value="{{ defaults.fuel_pct }}">
              </div>
              <div>
                <label>Tank capacity (L)</label>
                <input type="number" name="tank_l" min="1" step="0.1" value="{{ defaults.tank_l }}">
              </div>
              <div>
                <label>Fuel efficiency</label>
                <input type="text" name="eff_ice" value="{{ defaults.eff_ice }}" placeholder="15 km/l, 40 mpg, 6.8 l/100km">
              </div>
              <div>
                <label>Fuel threshold (%)</label>
                <input type="number" name="threshold_pct" min="1" max="80" step="1" value="{{ defaults.threshold_pct }}">
              </div>
            </div>
          </fieldset>
        </div>

        <!-- EV Settings -->
        <div id="ev-fields">
          <fieldset style="margin-top:20px;">
            <legend>EV Settings</legend>
            <div class="subgrid">
              <div>
                <label>Start SoC (%)</label>
                <input type="number" name="soc_pct" min="0" max="100" step="1" value="{{ defaults.soc_pct }}">
              </div>
              <div>
                <label>Battery capacity (kWh)</label>
                <input type="number" name="battery_kwh" min="10" step="0.1" value="{{ defaults.battery_kwh }}">
              </div>
              <div>
                <label>EV efficiency</label>
                <input type="text" name="eff_ev" value="{{ defaults.eff_ev }}" placeholder="16.5 kWh/100km, 6 mi/kWh, 155 Wh/km, 6.2 km/kWh">
              </div>
              <div>
                <label>Charge threshold (%)</label>
                <input type="number" name="charge_threshold_pct" min="1" max="80" step="1" value="{{ defaults.charge_threshold_pct }}">
              </div>
            </div>
          </fieldset>
        </div>

        <!-- Break Services -->
        <fieldset style="margin-top:20px;">
          <legend>Break services to include</legend>
          <div class="svc">
            <label class="row">
              <input type="checkbox" name="service_types" value="electric_vehicle_charging_station" {% if 'electric_vehicle_charging_station' in defaults.service_types %}checked{% endif %}>
              <span>EV Charging</span>
            </label>
            <label class="row">
              <input type="checkbox" name="service_types" value="gas_station" {% if 'gas_station' in defaults.service_types %}checked{% endif %}>
              <span>Fuel (Petrol/Diesel)</span>
            </label>
            {% for val,label in services %}
              <label class="row">
                <input type="checkbox" name="service_types" value="{{val}}" {% if val in defaults.service_types %}checked{% endif %}>
                <span>{{ label }}</span>
              </label>
            {% endfor %}
          </div>
        </fieldset>

        <div class="footer">
          <button class="btn" type="submit">Plan Route</button>
        </div>

        <p class="note">
          ICE: <i>km/l</i>, <i>mpg</i> (US/UK), or <i>l/100km</i> • EV: <i>kWh/100km</i>, <i>mi/kWh</i>, <i>Wh/km</i>, <i>km/kWh</i><br>
          You can paste coordinates as "lat,lng".
        </p>
      </form>

      <p class="muted">Server keys: Google Maps {{ '✅' if has_gmaps else '❌' }} • Mapbox {{ '✅' if has_mapbox else '❌' }}</p>
    </div>
  </div>

  <script>
    function setFromGeolocation(inputId) {
      if (!navigator.geolocation) { alert("Geolocation not supported by your browser."); return; }
      navigator.geolocation.getCurrentPosition(function(pos) {
        const lat = pos.coords.latitude.toFixed(6);
        const lng = pos.coords.longitude.toFixed(6);
        document.getElementById(inputId).value = lat + "," + lng;
      }, function(err) {
        alert("Could not get location: " + err.message);
      }, { enableHighAccuracy: true, timeout: 10000 });
    }

    document.getElementById("btn-my-origin").onclick = function(){ setFromGeolocation("origin-input"); };
    document.getElementById("btn-my-dest").onclick   = function(){ setFromGeolocation("destination-input"); };

    // Show only the active vehicle's fields and disable the hidden set.
    (function(){
      const vehSel = document.getElementById('vehicle-type');
      const ice = document.getElementById('ice-fields');
      const ev  = document.getElementById('ev-fields');

      function setDisabled(container, disabled) {
        container.querySelectorAll('input,select,textarea').forEach(el => { el.disabled = disabled; });
      }

      function refresh() {
        const isEV = (vehSel.value === 'ev');
        ev.style.display  = isEV ? 'block' : 'none';
        ice.style.display = isEV ? 'none'  : 'block';
        setDisabled(ev,  !isEV);
        setDisabled(ice,  isEV);
      }

      vehSel.addEventListener('change', refresh);
      refresh(); // initial
    })();
  </script>
</body>
</html>
"""

# --------------------------
#  Flask app
# --------------------------
app = Flask(__name__)

from typing import List, Dict, Tuple

# ---- Waypoint "optimizer" (no external deps) ----
def optimize_route_baseline(waypoints: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], bool]:
    """
    Keep original order (origin -> planned stops -> destination).
    Returns (waypoints, used_optimizer_flag=False).
    """
    return waypoints, False

# --------------------------
#  Result Page (Mapbox UI)
# --------------------------
RESULT_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Route — Next Refuel/Charge + Break Detours</title>
<meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no" />
<script src='https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.js'></script>
<link href='https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.css' rel='stylesheet' />
<script src="https://cdn.jsdelivr.net/npm/@turf/turf@6/turf.min.js"></script>
<style>
html, body { height:100%; }
body { margin:0; padding:0; font-family:system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;}
#map { position:absolute; top:0; bottom:0; width:100%; min-height:420px; }
#nav-back { position:absolute; top:14px; left:14px; z-index:12; }
#nav-back a {
  background:#4f46e5; color:#fff; padding:8px 14px; border-radius:10px;
  font-weight:700; text-decoration:none; box-shadow:0 2px 6px rgba(0,0,0,.35);
}
#nav-back a:hover { filter:brightness(1.1); }

#hud { position:absolute; top:60px; left:10px; z-index:10; color:#fff; background:rgba(0,0,0,.55);
       padding:12px 14px; border-radius:12px; font-family:sans-serif; min-width:600px; backdrop-filter: blur(4px); }
#toggle-hud { position:absolute; top:5px; right:5px; z-index:15; }
#hud-content { transition: all .2s ease; }
#controls { margin-top:8px; display:flex; gap:6px; flex-wrap:wrap; align-items:center; }
.button { cursor:pointer; border:none; padding:6px 10px; border-radius:8px; box-shadow:0 1px 2px rgba(0,0,0,.3); }
.button:hover { filter:brightness(1.05); }
#btn-slow { background:#2563eb; color:#fff; }
#btn-fast { background:#16a34a; color:#fff; }
#btn-toggle { background:#f59e0b; color:#111; }
#btn-follow { background:#0ea5e9; color:#fff; }
#btn-fit { background:#9333ea; color:#fff; }
#btn-break { background:#dc2626; color:#fff; }
#btn-cancel-break { background:#475569; color:#fff; }
#btn-export { background:#64748b; color:#fff; }
#btn-export-charges { background:#0ea5e9; color:#fff; }
#btn-reset { background:#ef4444; color:#fff; }
#speed-slider { accent-color:#16a34a; }
.legend { margin-top:8px; font-size:12px; opacity:.9; }
.dot-fuel { display:inline-block; width:10px; height:10px; background:#ef4444; border-radius:50%; margin-right:4px; vertical-align:middle; }
.dot-charge { display:inline-block; width:10px; height:10px; background:#60a5fa; border-radius:50%; margin-right:4px; vertical-align:middle; }
.readout { margin-top:6px; font-size:13px; }
.energybar { width:100%; height:12px; background:rgba(255,255,255,.15); border-radius:8px; overflow:hidden; margin-top:4px; }
.energybar-fill { height:100%; width:0%; background:#22c55e; transition: width .15s linear; }
#progress-wrap { margin-top:10px; }
#progress-bar { width:100%; height:10px; background:rgba(255,255,255,.15); border-radius:999px; overflow:hidden; }
#progress-fill { width:0%; height:100%; background:#22c55e; transition: width .15s linear; }
#progress-label { margin-top:4px; font-size:12px; opacity:.95; }
#toasts { position:absolute; right:16px; bottom:16px; z-index:20; display:flex; flex-direction:column; gap:8px; }
.toast { background:rgba(0,0,0,.8); color:#fff; padding:10px 12px; border-radius:10px; min-width:220px; box-shadow:0 2px 10px rgba(0,0,0,.4); font-size:13px; }
.toast b { color:#ffd166; }
#stops-box { margin-top:8px; padding:8px; border-radius:8px; background:rgba(255,255,255,.08); }
#stops-list { margin:6px 0 0 16px; padding:0; }
#stops-list li { margin:3px 0; }
#next-stop { margin-top:8px; padding:8px; border-radius:8px; background:rgba(255,255,255,.08); }
#break-panel {
  position:absolute; right:10px; top:10px; z-index:11;
  background:rgba(0,0,0,.55); color:#fff; padding:12px 14px; border-radius:12px; width:360px;
  backdrop-filter: blur(4px); display:none;
}
.break-row { display:flex; align-items:center; justify-content:space-between; gap:8px; margin:6px 0; }
.break-name { font-size:13px; line-height:1.2; }
.break-dist { font-size:12px; opacity:.9; }
.small-note { font-size:12px; opacity:.85; }

/* Optimizer badge */
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:700; }
.badge-ok { background:#16a34a; color:#fff; }
.badge-muted { background:#475569; color:#fff; }
</style>
</head>
<body>

<div id="map"></div>
<div id="nav-back"><a href="{{ url_for('index') }}">◀ Plan another trip</a></div>

<div id="hud">
  <button id="toggle-hud" class="button">−</button>
  <div id="hud-content">
    <div><b>From:</b> {{ from_label }}</div>
    <div><b>To:</b> {{ to_label }}</div>
    <div><b>Total:</b> {{ total_km_str }} km, {{ total_min_str }} min</div>

    <div class="readout">
      <b>Vehicle:</b> <span id="veh-type">{{ vehicle_type_label }}</span>
    </div>

    <div class="readout"><b>AI route score:</b> {{ route_score | round(3) }}</div>
    <div class="readout"><b>Optimizer:</b> <span id="opt-badge" class="badge badge-muted">—</span></div>

    <div class="readout">
      <b>Car mode:</b>
      <select id="car-mode">
        <option value="normal" selected>Normal</option>
        <option value="eco">Eco</option>
        <option value="sport">Sport</option>
        <option value="heavy">Heavy load</option>
      </select>
      <span class="small-note">— affects efficiency & range live</span>
    </div>
    <div class="readout small-note">
      <label>Mode multiplier:
        <input id="mode-mult" type="number" step="0.01" min="0.5" max="1.5" value="1.00" style="width:80px;">
      </label>
      <span>(applied on top of selected mode)</span>
    </div>

    <div><b>Planned stops (fuel/charge):</b> {{ planned_count }}</div>

    <div id="next-stop" class="readout">
      <div><b>Next stop ahead:</b> <span id="nr-name">—</span></div>
      <div><b>Distance:</b> <span id="nr-dist">—</span></div>
    </div>

    <div class="readout"><b>Speed:</b> <span id="speed-value">0.0</span> km/h (<span id="speed-label">slow</span>)</div>

    <div class="readout">
      <b id="energy-label">Fuel:</b> <span id="energy-text">--</span>
      <div class="energybar"><div id="energybar-fill" class="energybar-fill"></div></div>
    </div>

    <div id="controls">
      <button id="btn-slow" class="button">Slow</button>
      <button id="btn-fast" class="button">Fast</button>
      <input id="speed-slider" type="range" min="1" max="20" value="1" />
      <button id="btn-toggle" class="button">Pause</button>
      <button id="btn-follow" class="button">Follow: On</button>
      <button id="btn-fit" class="button">Fit Route</button>
      <button id="btn-break" class="button">Take Break</button>
      <button id="btn-cancel-break" class="button">Cancel</button>
      <button id="btn-export" class="button">Export Refuels CSV</button>
      <button id="btn-export-charges" class="button">Export Charges CSV</button>
      <button id="btn-reset" class="button">Reset</button>
    </div>

    <div id="stops-box">
      <div><b>Stops:</b> <span id="stop-count">0</span></div>
      <ul id="stops-list"></ul>
    </div>

    <div id="progress-wrap">
      <div id="progress-bar"><div id="progress-fill"></div></div>
      <div id="progress-label">0% — 0.0 / {{ total_km_str }} km — ETA: --:--:--</div>
    </div>

    <div class="legend">
      <span class="dot-fuel"></span>Planned fuel stop
      &nbsp;&nbsp;&nbsp;
      <span class="dot-charge"></span>Planned charge stop
    </div>
  </div>
</div>
<div id="break-panel">
  <div style="font-weight:700">Services ahead on route</div>
  <div style="font-size:12px;opacity:.9;margin-top:2px">Pick a stop — we’ll detour (red) and rejoin ahead.</div>
  <div id="break-list" style="margin-top:8px"></div>
</div>

<div id="toasts"></div>

<script>
mapboxgl.accessToken = '{{ mapbox_token }}';

// ===== Data from server =====
const VEHICLE_TYPE = "{{ vehicle_type }}"; // 'ice' or 'ev'
const ROUTE_COORDS = {{ route_coords | safe }};
const ROUTE_LATLNG = {{ route_latlng | safe }};
const STOPS_LIST  = {{ stops_js | safe }};        // planned fuel/charge stops
const SERVICES_LIST = {{ services_js | safe }};   // includes chargers & fuel & extras
const USED_OPTIMIZER = {{ 'true' if used_optimizer else 'false' }};

// Energy capacity/efficiency, unified
const CAPACITY = {{ capacity }};                 // L for ICE, kWh for EV
const BASE_KM_PER_UNIT = {{ km_per_unit }};     // km per L (ICE) or km per kWh (EV)
let KM_PER_UNIT = BASE_KM_PER_UNIT;
const THRESHOLD_PCT = {{ threshold_pct }};      // fuel% or soc%

// Start amount
let energyUnits = {{ init_units }};             // liters or kWh
const TOTAL_KM = {{ total_km }};
const TOTAL_DUR_S = {{ total_dur_s }};

// ===== Labels based on vehicle type =====
const LABELS = (VEHICLE_TYPE === 'ev')
  ? { energy: 'Battery', units: 'kWh', stopWord: 'Charge', color:'#60a5fa' }
  : { energy: 'Fuel',    units: 'L',   stopWord: 'Refuel', color:'#ef4444' };

// ===== Car mode logic =====
const MODE_MULTIPLIER = {
  eco: 1.15,
  normal: 1.00,
  sport: 0.85,
  heavy: 0.75
};
function recalcKmplFromControls() {
  const mode = (document.getElementById('car-mode').value || 'normal');
  const baseMult = MODE_MULTIPLIER[mode] || 1.0;
  const userMult = Math.max(0.5, Math.min(1.5, Number(document.getElementById('mode-mult').value) || 1.0));
  KM_PER_UNIT = BASE_KM_PER_UNIT * baseMult * userMult;
}
function applyCarMode() {
  recalcKmplFromControls();
  addToast('<b>Efficiency:</b> ' + KM_PER_UNIT.toFixed(2) + ' km/' + LABELS.units);
  updateRangeCircle(lastLng, lastLat);
  updateEnergyUI();
}
document.addEventListener('change', function(e){
  if (e.target && (e.target.id === 'car-mode' || e.target.id === 'mode-mult')) {
    applyCarMode();
  }
});

// ===== Map init =====
const map = new mapboxgl.Map({
  container:'map',
  style:'mapbox://styles/mapbox/streets-v12',
  center: ROUTE_COORDS[0],
  zoom: 15,
  pitch: 60,
  bearing: -10,
  antialias: true
});
map.on('error', (e)=> console.error('Mapbox error:', e && e.error));

// ===== Helpers =====
function toRad(deg) { return deg * Math.PI / 180; }
function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371.0;
  const dlat = toRad(lat2 - lat1);
  const dlon = toRad(lon2 - lon1);
  const a = Math.sin(dlat/2)**2 + Math.cos(toRad(lat1))*Math.cos(toRad(lat2))*Math.sin(dlon/2)**2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return R * c;
}
function bearingDeg(lat1, lon1, lat2, lon2) {
  const φ1 = toRad(lat1), φ2 = toRad(lat2);
  const Δλ = toRad(lon2 - lon1);
  const y = Math.sin(Δλ) * Math.cos(φ2);
  const x = Math.cos(φ1)*Math.sin(φ2) - Math.sin(φ1)*Math.cos(φ2)*Math.cos(Δλ);
  let θ = Math.atan2(y, x);
  return (θ * 180/Math.PI + 360) % 360;
}
function fmt(n, d=1) { return Number(n).toFixed(d); }

// ===== UI helpers =====
function updateEnergyUI() {
  const pct = Math.max(0, Math.min(100, (energyUnits / CAPACITY) * 100));
  const fill = document.getElementById('energybar-fill');
  if (fill) {
    fill.style.width = pct + '%';
    fill.style.background = pct < 25 ? '#e11d48' : (pct < 50 ? '#f59e0b' : '#22c55e');
  }
  document.getElementById('energy-label').textContent = LABELS.energy + ':';
  document.getElementById('energy-text').textContent =
    fmt(energyUnits,1) + ' ' + LABELS.units + ' (' + pct.toFixed(0) + '%) — ' + KM_PER_UNIT.toFixed(2) + ' km/' + LABELS.units;
}
function addToast(html) {
  const wrap = document.getElementById('toasts');
  const t = document.createElement('div');
  t.className = 'toast';
  t.innerHTML = html;
  wrap.appendChild(t);
  setTimeout(function() {
    t.style.opacity = '0';
    t.style.transform = 'translateY(10px)';
    setTimeout(function() { t.remove(); }, 350);
  }, 3500);
}
const stopHistory = []; // unified: {type:'fuel'|'charge', name, place, timeISO}
function addStopHistory(kind, name, place) {
  const ul = document.getElementById('stops-list');
  const li = document.createElement('li');
  const when = new Date();
  li.textContent = '[' + (kind === 'charge' ? '⚡' : '⛽') + '] ' + (name || '') + (place ? ', ' + place : '') + ' — ' + when.toLocaleTimeString();
  ul.appendChild(li);
  const c = document.getElementById('stop-count');
  c.textContent = String(Number(c.textContent) + 1);
  stopHistory.push({ type: kind, name: (name || ''), place: (place || ''), timeISO: when.toISOString() });
}
function exportStopsCSV(kind) {
  const rows = stopHistory.filter(s => s.type === kind);
  const header = 'type,name,place,time\\n';
  const body = rows.map(function(r) {
    const name = String(r.name || "").replace(/"/g,'""');
    const place = String(r.place || "").replace(/"/g,'""');
    return [r.type, '"' + name + '"', '"' + place + '"', r.timeISO].join(',');
  }).join('\\n');
  const blob = new Blob([header + body], {type: 'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = (kind === 'charge' ? 'charges.csv' : 'refuels.csv'); a.click();
  URL.revokeObjectURL(url);
}

// ===== Next stop helpers =====
function computeNextStop(travelledKm) {
  for (var i=0;i<STOPS_LIST.length;i++) {
    var s = STOPS_LIST[i];
    if (s.kmFromStart > travelledKm + 0.02) return s;
  }
  return null;
}
function setNextStopUI(next) {
  const nameEl = document.getElementById('nr-name');
  const distEl = document.getElementById('nr-dist');
  if (!next) { nameEl.textContent = 'None'; distEl.textContent = '—'; return; }
  nameEl.textContent = next.name + (next.place ? ', ' + next.place : '') + (next.kind === 'charge' ? ' (⚡)' : ' (⛽)');
  const distAheadKm = Math.max(0, next.kmFromStart - travelledKm);
  distEl.textContent = fmt(distAheadKm,1) + ' km';
}

// Suggest nearest ahead by type from SERVICES_LIST
function nearestAhead(travelledKm, typeKey) {
  const ahead = SERVICES_LIST
    .filter(s => s.ptype === typeKey && s.kmFromStart > travelledKm + 0.02)
    .sort((a,b) => a.kmFromStart - b.kmFromStart);
  return ahead.length ? ahead[0] : null;
}

// ===== Fit route =====
function fitRouteBounds() {
  const b = new mapboxgl.LngLatBounds(ROUTE_COORDS[0], ROUTE_COORDS[0]);
  for (var i=0;i<ROUTE_COORDS.length;i++) b.extend(ROUTE_COORDS[i]);
  map.fitBounds(b, { padding: 60, pitch: 60, bearing: -10, duration: 600 });
}

// ===== Car marker =====
const carEl = document.createElement('div');
carEl.style.width = '36px';
carEl.style.height = '36px';
carEl.style.backgroundImage = "url('https://cdn-icons-png.flaticon.com/512/744/744465.png')";
carEl.style.backgroundSize = 'contain';
carEl.style.backgroundRepeat = 'no-repeat';
carEl.style.transform = 'translate(-50%, -50%)';
const car = new mapboxgl.Marker(carEl).setLngLat([ROUTE_LATLNG[0][1], ROUTE_LATLNG[0][0]]);

// ===== Animation state =====
let idx = 0;
let stepsPerTick = 1;
let paused = false;
let followCar = true;

let lastTs = performance.now();
let lastLat = ROUTE_LATLNG[0][0];
let lastLng = ROUTE_LATLNG[0][1];

let travelledKm = 0.0;
let etaSec = TOTAL_DUR_S;

const STOP_RADIUS_KM = 0.15;
const completedStopKeys = new Set();
let warnedRangeShortfall = false;

const PROGRESS_LATLNG = [ROUTE_LATLNG[0]];
function updateProgressLine() {
  if (!map.getSource('progress')) return;
  const coords = PROGRESS_LATLNG.map(function(p) { return [p[1], p[0]]; });
  map.getSource('progress').setData({ type:'Feature', geometry:{ type:'LineString', coordinates: coords }});
}

// ===== Export buttons =====
document.getElementById('btn-export').onclick = function(){ exportStopsCSV('fuel'); };
document.getElementById('btn-export-charges').onclick = function(){ exportStopsCSV('charge'); };

// ===== Reset / progress UI =====
function updateProgressUI(pct, travelledKm, etaSec) {
  const fill = document.getElementById('progress-fill');
  const label = document.getElementById('progress-label');
  fill.style.width = pct.toFixed(1) + '%';
  const h = Math.floor(etaSec/3600), m = Math.floor((etaSec%3600)/60), s = Math.floor(etaSec%60);
  const etaStr = String(h).padStart(2,'0') + ':' + String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0');
  label.textContent = pct.toFixed(1) + '% — ' + fmt(travelledKm,1) + ' / ' + fmt(TOTAL_KM,1) + ' km — ETA: ' + etaStr;
}
function updateSpeedLabel(spdTag) { document.getElementById('speed-label').textContent = spdTag; }

function resetRun() {
  idx = 0;
  lastLat = ROUTE_LATLNG[0][0];
  lastLng = ROUTE_LATLNG[0][1];
  travelledKm = 0;
  etaSec = TOTAL_DUR_S;
  warnedRangeShortfall = false;
  completedStopKeys.clear();
  document.getElementById('stop-count').textContent = '0';
  document.getElementById('stops-list').innerHTML = '';
  energyUnits = {{ init_units }};
  applyCarMode();
  updateEnergyUI();
  PROGRESS_LATLNG.length = 0;
  PROGRESS_LATLNG.push(ROUTE_LATLNG[0]);
  updateProgressLine();
  car.setLngLat([ROUTE_LATLNG[0][1], ROUTE_LATLNG[0][0]]);
  map.setCenter([ROUTE_COORDS[0][0], ROUTE_COORDS[0][1]]);
  updateProgressUI(0, 0, TOTAL_DUR_S);
  setNextStopUI(computeNextStop(0));
  clearDetour();
}
document.getElementById('btn-reset').onclick = resetRun;

// ===== Stop detection =====
function checkStop(lat, lng) {
  for (var i=0;i<STOPS_LIST.length;i++) {
    var s = STOPS_LIST[i];
    const key = s.lat.toFixed(5) + ',' + s.lng.toFixed(5) + ',' + s.kind;
    if (completedStopKeys.has(key)) continue;
    const d = haversineKm(lat, lng, s.lat, s.lng);
    if (d <= STOP_RADIUS_KM) {
      completedStopKeys.add(key);
      if (s.kind === 'fuel' || s.kind === 'charge') {
        energyUnits = CAPACITY; // always fill to 100%
      }

      updateEnergyUI();
      addToast('<b>' + LABELS.stopWord + ':</b> ' + s.name + (s.place ? ', ' + s.place : ''));
      addStopHistory(s.kind, s.name, s.place || '');
      updateRangeCircle(lng, lat);
    }
  }
}

// ===== Detour engine (Mapbox Directions) =====
let detourRoute = null;
let detourIdx = 0;
let detourRejoinIdx = null;

async function fetchMapboxLeg(fromLng, fromLat, toLng, toLat) {
  const url = "https://api.mapbox.com/directions/v5/mapbox/driving/" +
              fromLng + "," + fromLat + ";" + toLng + "," + toLat +
              "?geometries=geojson&access_token=" + encodeURIComponent(mapboxgl.accessToken);
  const res = await fetch(url);
  if (!res.ok) throw new Error('Directions fetch failed');
  const data = await res.json();
  const coords = (data.routes && data.routes[0] && data.routes[0].geometry && data.routes[0].geometry.coordinates) || [];
  return coords.map(function(c) { return [c[1], c[0]]; }); // [lat,lng]
}

function forwardNearestRouteIndex(lat, lng, startIdx) {
  let bestI = startIdx;
  let bestD = Infinity;
  for (let i = startIdx; i < ROUTE_LATLNG.length; i++) {
    const p = ROUTE_LATLNG[i];
    const d = haversineKm(lat, lng, p[0], p[1]);
    if (d < bestD) { bestD = d; bestI = i; }
  }
  return bestI;
}

function showBreakPanel(show) {
  document.getElementById('break-panel').style.display = show ? 'block' : 'none';
}

function servicesAhead(travelledKm, limit) {
  if (typeof limit === 'undefined') limit = 12;
  const ahead = SERVICES_LIST
    .filter(function(s) { return s.kmFromStart > travelledKm + 0.02; })
    .sort(function(a,b) { return a.kmFromStart - b.kmFromStart; });
  return ahead.slice(0, limit);
}

function renderBreakList() {
  const list = document.getElementById('break-list');
  list.innerHTML = '';
  const items = servicesAhead(travelledKm, 12);
  if (items.length === 0) {
    list.textContent = 'No suitable places ahead right now.';
    return;
  }
  items.forEach(function(p) {
    const row = document.createElement('div');
    row.className = 'break-row';

    const left = document.createElement('div');
    left.className = 'break-name';
    left.textContent = p.name + (p.place ? ', ' + p.place : '');

    const sub = document.createElement('div');
    sub.className = 'break-dist';
    sub.textContent = fmt(Math.max(0,p.kmFromStart - travelledKm),1) + ' km ahead • ' + (p.ptype || 'service');

    const leftWrap = document.createElement('div');
    leftWrap.appendChild(left);
    leftWrap.appendChild(sub);

    const btn = document.createElement('button');
    btn.textContent = 'Go';
    btn.className = 'button';
    btn.onclick = function() { planDetour(p); };

    row.appendChild(leftWrap);
    row.appendChild(btn);
    list.appendChild(row);
  });
}

function clearDetour() {
  detourRoute = null;
  detourIdx = 0;
  detourRejoinIdx = null;
  if (map.getSource('detour')) {
    map.getSource('detour').setData({ type:'Feature', geometry: { type:'LineString', coordinates: [] } });
  }
}

async function planDetour(place) {
  try {
    addToast('<b>Planning detour</b> to ' + place.name + ' ...');

    const currLat = lastLat, currLng = lastLng;
    const rejoinIdx = forwardNearestRouteIndex(place.lat, place.lng, idx);
    const rejoinLat = ROUTE_LATLNG[rejoinIdx][0];
    const rejoinLng = ROUTE_LATLNG[rejoinIdx][1];

    const partA = await fetchMapboxLeg(currLng, currLat, place.lng, place.lat);
    const partB = await fetchMapboxLeg(place.lng, place.lat, rejoinLng, rejoinLat);

    detourRoute = partA.concat(partB.slice(1));
    detourIdx = 0;
    detourRejoinIdx = rejoinIdx;

    const detourCoords = detourRoute.map(function(p) { return [p[1], p[0]]; });
    map.getSource('detour').setData({ type:'Feature', geometry: { type:'LineString', coordinates: detourCoords } });

    addToast('<b>Detour ready:</b> red path to your stop/break, then rejoin ahead.');
    showBreakPanel(false);
  } catch (e) {
    addToast('Could not plan detour: ' + e.message);
  }
}

// ===== Main loop =====
function animateTick() {
  const now = performance.now();
  const dtSec = (now - lastTs) / 1000.0;
  let movedKm = 0.0;

  if (!paused) {
    for (let k = 0; k < stepsPerTick; k++) {
      let lat, lng;

      if (detourRoute) {
        if (detourIdx >= detourRoute.length) break;
        lat = detourRoute[detourIdx][0];
        lng = detourRoute[detourIdx][1];
        detourIdx++;
      } else {
        if (idx >= ROUTE_LATLNG.length) break;
        lat = ROUTE_LATLNG[idx][0];
        lng = ROUTE_LATLNG[idx][1];
        idx++;
      }

      let angle;
      if (detourRoute) {
        const next = detourIdx < detourRoute.length ? detourRoute[detourIdx] : [lat,lng];
        angle = bearingDeg(lat, lng, next[0], next[1]);
      } else {
        const next = idx < ROUTE_LATLNG.length ? ROUTE_LATLNG[idx] : [lat,lng];
        angle = bearingDeg(lat, lng, next[0], next[1]);
      }
      carEl.style.transform = 'translate(-50%, -50%) rotate(' + angle + 'deg)';
      car.setLngLat([lng, lat]);
      if (followCar) map.setCenter([lng, lat]);

      const stepKm = haversineKm(lastLat, lastLng, lat, lng);
      movedKm += stepKm;

      checkStop(lat, lng);

      PROGRESS_LATLNG.push([lat, lng]);
      updateProgressLine();

      lastLat = lat; lastLng = lng;
    }
  }

  const speedKmh = dtSec > 0 ? (movedKm / dtSec) * 3600.0 : 0.0;
  document.getElementById('speed-value').textContent = speedKmh.toFixed(1);

  // Consumption using current live efficiency
  energyUnits = Math.max(0, energyUnits - (movedKm / KM_PER_UNIT));
  updateEnergyUI();
  updateRangeCircle(lastLng, lastLat);

  travelledKm += movedKm;
  const pct = TOTAL_KM > 0 ? Math.min(100, (travelledKm / TOTAL_KM) * 100) : 0;
  const remainingRatio = Math.max(0, 1 - (travelledKm / TOTAL_KM));
  etaSec = Math.round(TOTAL_DUR_S * remainingRatio);
  updateProgressUI(pct, travelledKm, etaSec);

  const nextStop = computeNextStop(travelledKm);
  setNextStopUI(nextStop);

  // === Mode-aware range guard + suggestion ===
  if (nextStop) {
    const distAhead = Math.max(0, nextStop.kmFromStart - travelledKm);
    const rangeKm = energyUnits * KM_PER_UNIT;
    const bufferKm = 5.0;
    if (!warnedRangeShortfall && rangeKm + 1e-6 < distAhead + bufferKm) {
      warnedRangeShortfall = true;
      const typeKey = (nextStop.kind === 'charge') ? 'electric_vehicle_charging_station' : 'gas_station';
      const alt = nearestAhead(travelledKm, typeKey);
      if (alt) {
        addToast('<b>Warning:</b> Range ' + fmt(rangeKm,1) + ' km < next stop ' + fmt(distAhead,1) +
                 ' km. Try nearer: <b>' + alt.name + '</b>' + (alt.place ? ', ' + alt.place : '') +
                 ' (' + fmt(Math.max(0, alt.kmFromStart - travelledKm),1) + ' km ahead).');
      } else {
        addToast('<b>Warning:</b> Range shortfall and no nearer stop known ahead.');
      }
    }
  }

  if (detourRoute && detourIdx >= detourRoute.length) {
    if (detourRejoinIdx != null && detourRejoinIdx > idx) {
      idx = detourRejoinIdx;
      lastLat = ROUTE_LATLNG[idx][0];
      lastLng = ROUTE_LATLNG[idx][1];
      car.setLngLat([lastLng, lastLat]);
    }
    clearDetour();
    addToast('<b>Back on route</b> — resuming to destination.');
  }

  const finishedOriginal = (idx >= ROUTE_LATLNG.length);
  const finishedDetour = (!detourRoute);
  if (finishedOriginal && finishedDetour && !window.__arrived__) {
    window.__arrived__ = true;
    addToast('<b>Arrived!</b> ' + fmt(travelledKm,1) + ' km — Stops: ' + document.getElementById('stop-count').textContent);
  }

  lastTs = now;
  setTimeout(animateTick, 50);
}

// Range circle
function updateRangeCircle(centerLng, centerLat) {
  if (!map.getSource('energy-range')) return;
  const radiusKm = Math.max(0, energyUnits * KM_PER_UNIT);
  const circle = turf.circle([centerLng, centerLat], radiusKm, {steps:64, units:'kilometers'});
  map.getSource('energy-range').setData(circle);
}

map.on('load', function() {
  map.addSource('mapbox-dem', { type:'raster-dem', url:'mapbox://mapbox.terrain-rgb', tileSize:512, maxzoom:14 });
  map.setTerrain({ source:'mapbox-dem', exaggeration:1.3 });

  const layers = map.getStyle().layers;
  let labelLayerId = null;
  for (let j = 0; j < layers.length; j++) {
    const l = layers[j];
    if (l.type === 'symbol' && l.layout && l.layout['text-field']) { labelLayerId = l.id; break; }
  }
  map.addLayer({
    'id': '3d-buildings',
    'source': 'composite',
    'source-layer': 'building',
    'filter': ['==', 'extrude', 'true'],
    'type': 'fill-extrusion',
    'minzoom': 15,
    'paint': {
      'fill-extrusion-color': '#aaa',
      'fill-extrusion-height': ['interpolate', ['linear'], ['zoom'], 15, 0, 16, ['get', 'height']],
      'fill-extrusion-base':  ['interpolate', ['linear'], ['zoom'], 15, 0, 16, ['get', 'min_height']],
      'fill-extrusion-opacity': 0.6
    }
  }, labelLayerId);

  map.addSource('route', {
    type: 'geojson',
    data: { type:'Feature', geometry: { type:'LineString', coordinates: ROUTE_COORDS } }
  });
  map.addLayer({ id:'route-line', type:'line', source:'route',
    paint: { 'line-color':'#0074D9', 'line-width':5 }});

  map.addSource('progress', {
    type: 'geojson',
    data: { type:'Feature', geometry: { type:'LineString', coordinates: [ROUTE_COORDS[0]] } }
  });
  map.addLayer({ id:'progress-line', type:'line', source:'progress',
    paint: { 'line-color':'#10b981', 'line-width':6, 'line-opacity':0.95, 'line-blur':0.3 }});

  map.addSource('detour', { type:'geojson', data: { type:'Feature', geometry: { type:'LineString', coordinates: [] } } });
  map.addLayer({
    id:'detour-line', type:'line', source:'detour',
    paint: { 'line-color':'#dc2626', 'line-width':4, 'line-dasharray':[2,2] }
  });

  const initCenter = ROUTE_COORDS[0];
  const initRadiusKm = Math.max(0, energyUnits * KM_PER_UNIT);
  const initCircle = turf.circle(initCenter, initRadiusKm, {steps:64, units:'kilometers'});
  map.addSource('energy-range', { type:'geojson', data: initCircle });
  map.addLayer({ id:'energy-range-fill', type:'fill', source:'energy-range',
    paint: { 'fill-color':'#22c55e', 'fill-opacity':0.12 }});
  map.addLayer({ id:'energy-range-outline', type:'line', source:'energy-range',
    paint: { 'line-color':'#16a34a', 'line-width':2, 'line-opacity':0.6 }});

  new mapboxgl.Marker({color:'#22c55e'}).setLngLat(ROUTE_COORDS[0]).setPopup(new mapboxgl.Popup().setText('Start')).addTo(map);
  new mapboxgl.Marker({color:'#ef4444'}).setLngLat(ROUTE_COORDS[ROUTE_COORDS.length-1]).setPopup(new mapboxgl.Popup().setText('Destination')).addTo(map);

  car.addTo(map);

  // Planned stops pins + HUD list
  STOPS_LIST.forEach(function(s) {
    const el = document.createElement('div');
    el.style.width = '24px';
    el.style.height = '24px';
    el.style.backgroundImage = (s.kind === 'charge')
      ? "url('https://cdn-icons-png.flaticon.com/512/4329/4329559.png')"  // bolt
      : "url('https://cdn-icons-png.flaticon.com/512/888/888879.png')";    // pump
    el.style.backgroundSize = 'contain';
    el.style.backgroundRepeat = 'no-repeat';
    el.style.transform = 'translate(-50%, -50%)';
    const popupText = (s.kind === 'charge' ? 'Charge: ' : 'Refuel: ')
      + s.name + (s.place ? ', ' + s.place : '');
    new mapboxgl.Marker(el).setLngLat([s.lng, s.lat]).setPopup(new mapboxgl.Popup().setText(popupText)).addTo(map);

    const ul = document.getElementById('stops-list');
    const li = document.createElement('li');
    li.textContent = (s.kind === 'charge' ? '⚡ ' : '⛽ ') + s.name + (s.place ? ', ' + s.place : '') +
                     ' — at ' + (s.kmFromStart).toFixed(1) + ' km';
    ul.appendChild(li);
    const c = document.getElementById('stop-count');
    c.textContent = String(Number(c.textContent) + 1);
  });

  document.getElementById('btn-slow').onclick = function() { stepsPerTick = 1; document.getElementById('speed-slider').value = 1; updateSpeedLabel('slow'); };
  document.getElementById('btn-fast').onclick = function() { stepsPerTick = 5; document.getElementById('speed-slider').value = 5; updateSpeedLabel('x5'); };
  document.getElementById('btn-toggle').onclick = function(e) { paused = !paused; e.currentTarget.textContent = paused ? 'Resume' : 'Pause'; };
  document.getElementById('btn-follow').onclick = function(e) { followCar = !followCar; e.currentTarget.textContent = 'Follow: ' + (followCar ? 'On' : 'Off'); };
  document.getElementById('btn-fit').onclick = fitRouteBounds;

  document.getElementById('btn-break').onclick = function() { renderBreakList(); showBreakPanel(true); };
  document.getElementById('btn-cancel-break').onclick = function() { clearDetour(); addToast('Detour cancelled.'); showBreakPanel(false); };

  const slider = document.getElementById('speed-slider');
  slider.oninput = function() { stepsPerTick = Number(slider.value); updateSpeedLabel(stepsPerTick === 1 ? 'slow' : 'x' + stepsPerTick); };

  // Vehicle label
  document.getElementById('veh-type').textContent = (VEHICLE_TYPE === 'ev') ? 'EV' : 'ICE';

  // Optimizer badge (we don’t actually optimize; we just show "Baseline sequence")
  const optBadge = document.getElementById('opt-badge');
  if (USED_OPTIMIZER) {
    optBadge.textContent = 'Optimized';
    optBadge.classList.remove('badge-muted');
    optBadge.classList.add('badge-ok');
  } else {
    optBadge.textContent = 'Baseline sequence';
    optBadge.classList.remove('badge-ok');
    optBadge.classList.add('badge-muted');
  }

  // Apply initial mode (Normal + 1.00 user multiplier)
  applyCarMode();

  updateEnergyUI();
  updateSpeedLabel('slow');
  fitRouteBounds();
  setNextStopUI(computeNextStop(0));
  animateTick();
});

// ===== HUD Minimize/Maximize =====
const toggleBtn = document.getElementById("toggle-hud");
const hudContent = document.getElementById("hud-content");
let hudVisible = true;
toggleBtn.onclick = function () {
  hudVisible = !hudVisible;
  hudContent.style.display = hudVisible ? "block" : "none";
  toggleBtn.textContent = hudVisible ? "−" : "+";
};
</script>
</body>
</html>
"""

# --------------------------
#  Planner helpers
# --------------------------
def _stitch_route_and_enrich(
    gmaps_client: "googlemaps.Client",
    decoded: List[Tuple[float, float]],
    steps: List[Dict],
    stops: List[Dict],
    o_fmt: str,
    d_fmt: str,
    vehicle_type: str,
    km_per_unit_in: float,
    capacity_in: float,
    threshold_pct_in: int,
    init_units_in: float,
):
    """
    Build final route via waypoints (origin -> planned stops -> destination),
    compute cumulative km, enrich stops with kmFromStart & place labels, and
    return: combined_points, cumulative_km, total_dur_s, total_dist_m, enriched_stops, used_optimizer
    """
    # Build via waypoints (keep baseline order)
    start = decoded[0]
    dest = decoded[-1]
    waypoints = [start] + [(s['lat'], s['lng']) for s in stops] + [dest]
    waypoints, used_optimizer = optimize_route_baseline(waypoints)

    combined_points: List[Tuple[float,float]] = []
    total_dist_m = 0.0
    total_dur_s = 0

    # Stitch path by requesting Google Directions for each leg
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        try:
            seg_dir = googlemaps.Client(key=API_KEY).directions(a, b, mode="driving", region="gb")
        except Exception:
            seg_dir = None
        if not seg_dir:
            if not combined_points: combined_points.append(a)
            combined_points.append(b)
            continue
        seg_leg = seg_dir[0]['legs'][0]
        total_dist_m += float(seg_leg['distance']['value'])
        total_dur_s  += int(seg_leg['duration']['value'])
        seg_pts = polyline.decode(seg_dir[0]['overview_polyline']['points'])
        combined_points.extend(seg_pts if not combined_points else seg_pts[1:])

    if not combined_points:
        # Fallback to original decoded route
        combined_points = decoded
        # Approximate distance by haversine
        total_dist_m = 0.0
        for i in range(1, len(decoded)):
            a = decoded[i-1]; b = decoded[i]
            total_dist_m += haversine_km(a[0], a[1], b[0], b[1]) * 1000.0
        # Leave total_dur_s as 0 if unavailable

    # Cumulative distance along the final route
    cumulative_km = [0.0]
    for i in range(1, len(combined_points)):
        a = combined_points[i-1]; b = combined_points[i]
        cumulative_km.append(cumulative_km[-1] + haversine_km(a[0], a[1], b[0], b[1]))

    # Reverse geocode helper
    def _label(lat, lng):
        return reverse_place_label(gmaps_client, lat, lng) or ""

    # Enrich planned stops with kmFromStart + place label
    enriched_stops = []
    for s in stops:
        nearest_idx = 0; best_d = float("inf")
        for i, (lat, lng) in enumerate(combined_points):
            d = haversine_km(lat, lng, s['lat'], s['lng'])
            if d < best_d: best_d = d; nearest_idx = i
        place = _label(s['lat'], s['lng'])
        enriched_stops.append({
            "name": s["name"], "lat": float(s["lat"]), "lng": float(s["lng"]), "place": place,
            "kmFromStart": cumulative_km[nearest_idx] if cumulative_km else 0.0,
            "kind": s["kind"]
        })

    return combined_points, cumulative_km, int(total_dur_s), float(total_dist_m), enriched_stops, used_optimizer

def _finalize_payload(
    combined_points: List[Tuple[float, float]],
    cumulative_km: List[float],
    total_dur_s: int,
    total_dist_m: float,
    enriched_stops: List[Dict],
    services: List[Dict],
    o_fmt: str,
    d_fmt: str,
    vehicle_type: str,
    km_per_unit: float,
    capacity: float,
    threshold_pct: int,
    init_units: float,
    used_optimizer: bool,
):
    # Enrich services with kmFromStart + place label
    enriched_services = []
    for s in services:
        nearest_idx = 0; best_d = float("inf")
        for i, (lat, lng) in enumerate(combined_points):
            d = haversine_km(lat, lng, s['lat'], s['lng'])
            if d < best_d: best_d = d; nearest_idx = i
        # Try to preserve any existing 'place' (if upstream added it)
        place = s.get("place") or reverse_place_label(googlemaps.Client(key=API_KEY), s["lat"], s["lng"]) or ""
        enriched_services.append({
            "name": s.get("name","Place"),
            "lat": float(s["lat"]), "lng": float(s["lng"]),
            "place": place, "ptype": s.get("ptype","service"),
            "kmFromStart": cumulative_km[nearest_idx] if cumulative_km else 0.0
        })

    # Ensure forward order along the final route
    enriched_stops.sort(key=lambda s: s["kmFromStart"])
    enriched_services.sort(key=lambda s: s["kmFromStart"])

    # ML features + scoring
    total_km = cumulative_km[-1] if cumulative_km else (total_dist_m / 1000.0)
    features = {
        "total_km": float(total_km or 0.0),
        "duration_s": float(total_dur_s or 0.0),
        "vehicle_type": 0.0 if vehicle_type == 'ice' else 1.0,
        "stops": float(len(enriched_stops)),
        "avg_leg_km": (float(total_km) / float(max(1, len(enriched_stops)+1))) if total_km else 0.0,
    }
    route_score = predict_route_score(features)
    log_user_route_features(features, meta={"origin": o_fmt, "dest": d_fmt})

    # Prep JS data
    route_coords = [[lng, lat] for lat, lng in combined_points]
    route_latlng = [[lat, lng] for lat, lng in combined_points]
    stops_js = json.dumps(enriched_stops)
    services_js = json.dumps(enriched_services)

    vehicle_type_label = "EV" if vehicle_type == 'ev' else "ICE"
    vtype_key = "ev" if vehicle_type == 'ev' else "ice"

    return {
        "from_label": o_fmt,
        "to_label": d_fmt,
        "vehicle_type": vtype_key,
        "vehicle_type_label": vehicle_type_label,
        "km_per_unit": float(f"{km_per_unit:.6f}"),
        "capacity": float(f"{capacity:.6f}"),
        "threshold_pct": int(threshold_pct),
        "init_units": float(f"{init_units:.6f}"),
        "route_coords": route_coords,
        "route_latlng": route_latlng,
        "stops_js": stops_js,
        "services_js": services_js,
        "total_km": float(f"{total_km:.6f}"),
        "total_dur_s": int(total_dur_s),
        "planned_count": len(enriched_stops),
        "route_score": float(route_score),
        "used_optimizer": bool(used_optimizer),
    }

# --------------------------
#  Main planner
# --------------------------
def plan_route_unified(
    origin_q: str,
    dest_q: str,
    vehicle_type: str,
    fuel_pct: float,
    eff_ice_raw: str,
    tank_l: float,
    threshold_pct_ice: int,
    soc_pct: float,
    eff_ev_raw: str,
    battery_kwh: float,
    charge_threshold_pct: int,
    step_interval: int,
    radius_m: int,
    service_types: List[str],
):
    """
    Compute a plan from origin to destination, with planned fuel/charge stops and services.
    Returns data structures later consumed by the result template.
    """
    if not API_KEY:
        raise RuntimeError("Google Maps API key is missing. Add GOOGLE_MAPS_API_KEY to .env.")
    gmaps_client = googlemaps.Client(key=API_KEY)

    # Efficiencies
    km_per_l = parse_efficiency_to_kmpl(eff_ice_raw, default_kmpl=DEFAULT_KM_PER_L)
    kwh_per_100km = parse_ev_efficiency_to_kwh_per_100km(eff_ev_raw, default_kwh_per_100km=DEFAULT_KWH_PER_100KM)
    km_per_kwh = 100.0 / kwh_per_100km if kwh_per_100km > 0 else (100.0 / DEFAULT_KWH_PER_100KM)

    # Capacities
    tank_capacity_l = max(1.0, float(tank_l) if tank_l else DEFAULT_TANK_CAPACITY_L)
    battery_capacity_kwh = max(1.0, float(battery_kwh) if battery_kwh else DEFAULT_BATTERY_KWH)

    # Geocode
    o_lat, o_lng, o_fmt = geocode_gb(gmaps_client, origin_q)
    d_lat, d_lng, d_fmt = geocode_gb(gmaps_client, dest_q)

    # Baseline route from Google Directions
    directions = gmaps_client.directions((o_lat, o_lng), (d_lat, d_lng), mode="driving", region="gb")
    if not directions:
        raise RuntimeError(f"No route found between '{o_fmt}' and '{d_fmt}'.")
    route0 = directions[0]
    leg = route0['legs'][0]
    steps = leg['steps']

    # Decode overview polyline; defensive fallback to per-step polylines
    ov = (route0.get('overview_polyline') or {}).get('points')
    if ov:
        decoded = polyline.decode(ov)
    else:
        decoded = []
        for st in steps:
            pts = (st.get('polyline') or {}).get('points')
            if pts:
                seg = polyline.decode(pts)
                decoded.extend(seg if not decoded else seg[1:])
    if not decoded:
        raise RuntimeError("Could not decode route geometry.")

    # Collect services along the route corridor
    svc_types = set(service_types or [])
    svc_types.update({'gas_station', 'electric_vehicle_charging_station'})  # enforce energy types presence

    chargers, stations, services = [], [], []
    for i, step in enumerate(steps):
        if i % max(1, int(step_interval)) != 0:
            continue
        lat = step['end_location']['lat']; lng = step['end_location']['lng']
        for t in list(svc_types):
            try:
                resp = gmaps_client.places_nearby(location=(lat, lng), radius=int(radius_m), type=t)
            except Exception:
                continue
            if not isinstance(resp, dict):
                continue
            for res in (resp.get('results') or []):
                geom = res.get('geometry') or {}
                loc = geom.get('location') or {}
                if not isinstance(loc, dict):
                    continue
                if 'lat' in loc and 'lng' in loc:
                    entry = {
                        'name': res.get('name', 'Place'),
                        'lat': float(loc['lat']),
                        'lng': float(loc['lng']),
                        'ptype': t
                    }
                    services.append(entry)
                    if t == 'gas_station':
                        stations.append({'name': entry['name'], 'lat': entry['lat'], 'lng': entry['lng']})
                    elif t == 'electric_vehicle_charging_station':
                        chargers.append({'name': entry['name'], 'lat': entry['lat'], 'lng': entry['lng']})

    chargers = unique_places(chargers)
    stations = unique_places(stations)
    services = unique_places(services)

    # Choose stops based on vehicle type & thresholds
    stops: List[Dict] = []
    if vehicle_type.lower() == 'ev':
        km_per_unit = km_per_kwh
        capacity = battery_capacity_kwh
        threshold_pct = int(charge_threshold_pct)
        init_units = max(0.0, min(100.0, float(soc_pct))) * 0.01 * battery_capacity_kwh

        energy = init_units
        for i in range(len(decoded) - 1):
            lat1, lon1 = decoded[i]; lat2, lon2 = decoded[i + 1]
            seg_km = haversine_km(lat1, lon1, lat2, lon2)
            if km_per_kwh > 0:
                energy -= (seg_km / km_per_kwh)
            pct_now = (energy / battery_capacity_kwh) * 100.0 if battery_capacity_kwh > 0 else 0.0
            if pct_now < charge_threshold_pct:
                if not chargers:
                    break
                nearest = min(chargers, key=lambda s: haversine_km(lat2, lon2, s['lat'], s['lng']))
                if (not stops) or (nearest['lat'], nearest['lng']) != (stops[-1]['lat'], stops[-1]['lng']):
                    stops.append({'name': nearest['name'], 'lat': nearest['lat'], 'lng': nearest['lng'], 'kind': 'charge'})
                energy = battery_capacity_kwh  # top-up to 100% for simulation

    else:
        km_per_unit = km_per_l
        capacity = tank_capacity_l
        threshold_pct = int(threshold_pct_ice)
        init_units = max(0.0, min(100.0, float(fuel_pct))) * 0.01 * tank_capacity_l

        fuel_l = init_units
        for i in range(len(decoded) - 1):
            lat1, lon1 = decoded[i]; lat2, lon2 = decoded[i + 1]
            seg_km = haversine_km(lat1, lon1, lat2, lon2)
            if km_per_l > 0:
                fuel_l -= (seg_km / km_per_l)
            pct_now = (fuel_l / tank_capacity_l) * 100.0 if tank_capacity_l > 0 else 0.0
            if pct_now < threshold_pct_ice:
                if not stations:
                    break
                nearest = min(stations, key=lambda s: haversine_km(lat2, lon2, s['lat'], s['lng']))
                if (not stops) or (nearest['lat'], nearest['lng']) != (stops[-1]['lat'], stops[-1]['lng']):
                    stops.append({'name': nearest['name'], 'lat': nearest['lat'], 'lng': nearest['lng'], 'kind': 'fuel'})
                fuel_l = tank_capacity_l  # fill to full for simulation

    # Stitch & enrich
    combined_points, cumulative_km, total_dur_s, total_dist_m, enriched_stops, used_optimizer = _stitch_route_and_enrich(
        gmaps_client, decoded, steps, stops, o_fmt, d_fmt, vehicle_type.lower(),
        km_per_unit, capacity, threshold_pct, init_units
    )

    # Final payload
    return _finalize_payload(
        combined_points=combined_points,
        cumulative_km=cumulative_km,
        total_dur_s=total_dur_s,
        total_dist_m=total_dist_m,
        enriched_stops=enriched_stops,
        services=services,
        o_fmt=o_fmt,
        d_fmt=d_fmt,
        vehicle_type=vehicle_type.lower(),
        km_per_unit=km_per_unit,
        capacity=capacity,
        threshold_pct=threshold_pct,
        init_units=init_units,
        used_optimizer=used_optimizer,
    )
# =========================
# Fuel/EV-aware route planner — Part 3/3
# Flask routes (index, plan) + lightweight ML training & management endpoints
# (Requires Part 1 and Part 2 to be loaded first.)
# =========================

import time
from flask import Response

# --------------------------
#  Defaults for the form
# --------------------------
def _default_form_values():
    return {
        "origin": "SW1A 1AA",
        "destination": "BS1 5TR",
        "vehicle_type": "ice",
        "car_mode": "normal",

        # ICE defaults
        "fuel_pct": 65,
        "tank_l": DEFAULT_TANK_CAPACITY_L,
        "eff_ice": f"{DEFAULT_KM_PER_L} km/l",
        "threshold_pct": DEFAULT_FUEL_THRESHOLD_PCT,

        # EV defaults
        "soc_pct": 70,
        "battery_kwh": DEFAULT_BATTERY_KWH,
        "eff_ev": f"{DEFAULT_KWH_PER_100KM} kWh/100km",
        "charge_threshold_pct": DEFAULT_CHARGE_THRESHOLD_PCT,

        "radius_m": DEFAULT_PLACES_RADIUS_M,
        "step_interval": DEFAULT_STEP_INTERVAL,
        "service_types": ["electric_vehicle_charging_station", "gas_station"] + [t for t,_ in ALL_SERVICE_TYPES][:3],
    }

# --------------------------
#  Web pages
# --------------------------
@app.get("/")
def index():
    defaults = _default_form_values()
    return render_template_string(
        FORM_HTML,
        defaults=defaults,
        services=ALL_SERVICE_TYPES,
        has_gmaps=bool(API_KEY),
        has_mapbox=bool(MAPBOX_TOKEN),
        error=None
    )

@app.post("/plan")
def plan():
    try:
        origin = (request.form.get("origin","") or "").strip()
        dest   = (request.form.get("destination","") or "").strip()
        if not origin or not dest:
            raise RuntimeError("Please enter both origin and destination.")

        vehicle_type = (request.form.get("vehicle_type","ice") or "ice").lower().strip()

        # ICE inputs
        fuel_pct = float(request.form.get("fuel_pct","65") or 65)
        tank_l = float(request.form.get("tank_l", str(DEFAULT_TANK_CAPACITY_L)) or DEFAULT_TANK_CAPACITY_L)
        eff_ice_raw = request.form.get("eff_ice", f"{DEFAULT_KM_PER_L} km/l")
        threshold_pct_ice = int(request.form.get("threshold_pct", str(DEFAULT_FUEL_THRESHOLD_PCT)) or DEFAULT_FUEL_THRESHOLD_PCT)

        # EV inputs
        soc_pct = float(request.form.get("soc_pct","70") or 70)
        battery_kwh = float(request.form.get("battery_kwh", str(DEFAULT_BATTERY_KWH)) or DEFAULT_BATTERY_KWH)
        eff_ev_raw = request.form.get("eff_ev", f"{DEFAULT_KWH_PER_100KM} kWh/100km")
        charge_threshold_pct = int(request.form.get("charge_threshold_pct", str(DEFAULT_CHARGE_THRESHOLD_PCT)) or DEFAULT_CHARGE_THRESHOLD_PCT)

        # Search config
        radius_m = int(request.form.get("radius_m", str(DEFAULT_PLACES_RADIUS_M)) or DEFAULT_PLACES_RADIUS_M)
        step_interval = int(request.form.get("step_interval", str(DEFAULT_STEP_INTERVAL)) or DEFAULT_STEP_INTERVAL)

        # Services
        service_types = request.form.getlist("service_types") or ["electric_vehicle_charging_station", "gas_station"]

        if not API_KEY:
            raise RuntimeError("Google Maps API key is missing. Add GOOGLE_MAPS_API_KEY to .env (or API.env).")
        if not MAPBOX_TOKEN:
            raise RuntimeError("Mapbox token is missing. Add MAPBOX_TOKEN to .env (or API.env).")

        res = plan_route_unified(
            origin_q=origin,
            dest_q=dest,
            vehicle_type=vehicle_type,
            fuel_pct=fuel_pct,
            eff_ice_raw=eff_ice_raw,
            tank_l=tank_l,
            threshold_pct_ice=threshold_pct_ice,
            soc_pct=soc_pct,
            eff_ev_raw=eff_ev_raw,
            battery_kwh=battery_kwh,
            charge_threshold_pct=charge_threshold_pct,
            step_interval=step_interval,
            radius_m=radius_m,
            service_types=service_types
        )
        _maybe_autofit()


        return render_template_string(
            RESULT_HTML,
            mapbox_token=MAPBOX_TOKEN,
            from_label=res["from_label"],
            to_label=res["to_label"],
            vehicle_type=res["vehicle_type"],
            vehicle_type_label=res["vehicle_type_label"],
            total_km_str=f"{res['total_km']:.1f}",
            total_min_str=f"{res['total_dur_s']/60:.0f}",
            planned_count=res["planned_count"],
            route_coords=json.dumps(res["route_coords"]),
            route_latlng=json.dumps(res["route_latlng"]),
            stops_js=res.get("stops_js","[]"),
            services_js=res.get("services_js","[]"),
            capacity=res["capacity"],
            km_per_unit=res["km_per_unit"],
            threshold_pct=res["threshold_pct"],
            init_units=res["init_units"],
            total_km=res["total_km"],
            total_dur_s=res["total_dur_s"],
            route_score=res.get("route_score", 0.5),
            used_optimizer=res.get("used_optimizer", False),
        )
    except Exception as e:
        # Refill the form with whatever the user submitted
        defaults = {
            "origin": request.form.get("origin",""),
            "destination": request.form.get("destination",""),
            "vehicle_type": request.form.get("vehicle_type","ice"),
            "car_mode": request.form.get("car_mode","normal"),

            "fuel_pct": request.form.get("fuel_pct","65"),
            "tank_l": request.form.get("tank_l", str(DEFAULT_TANK_CAPACITY_L)),
            "eff_ice": request.form.get("eff_ice", f"{DEFAULT_KM_PER_L} km/l"),
            "threshold_pct": request.form.get("threshold_pct", str(DEFAULT_FUEL_THRESHOLD_PCT)),

            "soc_pct": request.form.get("soc_pct","70"),
            "battery_kwh": request.form.get("battery_kwh", str(DEFAULT_BATTERY_KWH)),
            "eff_ev": request.form.get("eff_ev", f"{DEFAULT_KWH_PER_100KM} kWh/100km"),
            "charge_threshold_pct": request.form.get("charge_threshold_pct", str(DEFAULT_CHARGE_THRESHOLD_PCT)),

            "radius_m": request.form.get("radius_m", str(DEFAULT_PLACES_RADIUS_M)),
            "step_interval": request.form.get("step_interval", str(DEFAULT_STEP_INTERVAL)),
            "service_types": request.form.getlist("service_types") or [],
        }
        return render_template_string(
            FORM_HTML,
            defaults=defaults,
            services=ALL_SERVICE_TYPES,
            has_gmaps=bool(API_KEY),
            has_mapbox=bool(MAPBOX_TOKEN),
            error=str(e)
        ), 400

# --------------------------
#  ML Training & Management (stronger)
# --------------------------
import time
from flask import jsonify

@app.post("/ml/fit")
def ml_fit():
    """
    Full retrain on ALL rows using Ridge + standardization.
    Optional JSON body: {"lambda": 0.01}
    """
    body = request.get_json(silent=True) or {}
    lam = float(body.get("lambda", 1e-2))
    rows = _load_jsonl_rows(ML_TRAIN_FILE)
    if not rows:
        return jsonify({"ok": False, "error": f"No training rows in {ML_TRAIN_FILE}"}), 400
    try:
        M = ml_fit_ridge(rows, lam=lam)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    saved = ml_save_model(M, ML_MODEL_PATH)
    global _ML_MODEL
    _ML_MODEL = M

    return jsonify({
        "ok": True,
        "trained_on": len(rows),
        "lambda": lam,
        "weights": M["w"],
        "bias": M["b"],
        "mu": M["mu"],
        "sigma": M["sigma"],
        "saved": bool(saved),
        "model_path": ML_MODEL_PATH,
    })

@app.post("/ml/fit/quick")
def ml_fit_quick():
    """
    Retrain on the LAST N rows only (fast adaptation).
    JSON: {"n": 200, "lambda": 0.02}
    """
    body = request.get_json(silent=True) or {}
    n = int(body.get("n", 200))
    lam = float(body.get("lambda", 2e-2))
    rows = _load_jsonl_rows(ML_TRAIN_FILE)
    if not rows:
        return jsonify({"ok": False, "error": "No training rows."}), 400
    rows = rows[-n:]
    try:
        M = ml_fit_ridge(rows, lam=lam)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    saved = ml_save_model(M, ML_MODEL_PATH)
    global _ML_MODEL
    _ML_MODEL = M
    return jsonify({"ok": True, "trained_on": len(rows), "lambda": lam, "saved": bool(saved)})

@app.get("/ml/status")
def ml_status():
    model_loaded = isinstance(_ML_MODEL, dict) and "w" in _ML_MODEL
    mtime = None
    try:
        if os.path.exists(ML_MODEL_PATH):
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(ML_MODEL_PATH)))
    except Exception:
        mtime = None

    rows = _load_jsonl_rows(ML_TRAIN_FILE)
    return jsonify({
        "ok": True,
        "model_loaded": model_loaded,
        "lambda": _ML_MODEL.get("lambda") if model_loaded else None,
        "model_file": ML_MODEL_PATH if os.path.exists(ML_MODEL_PATH) else None,
        "model_mtime": mtime,
        "train_rows": len(rows),
        "train_file": ML_TRAIN_FILE if os.path.exists(ML_TRAIN_FILE) else None,
    })

@app.post("/ml/predict")
def ml_predict():
    try:
        payload = request.get_json(force=True) or {}
        features = payload.get("features") or {}
        _, x = _features_to_vector(features)
        score = predict_route_score(features)
        return jsonify({"ok": True, "score": float(score)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.post("/ml/rate")
def ml_rate():
    """
    Accept JSON:
      {
        "features": {...},
        "rating": 0..1,
        "meta": {...},
        "online_steps": 10,          # optional: small Adam steps now
        "online_lr": 0.01            # optional learning rate
      }
    Appends to JSONL and performs a few online Adam steps so the model learns immediately.
    """
    try:
        data = request.get_json(force=True) or {}
        features = data.get("features") or {}
        rating = data.get("rating")
        meta = data.get("meta") or {}
        if rating is None:
            return jsonify({"ok": False, "error": "Missing 'rating'"}), 400
        y = float(rating)
        if not (0.0 <= y <= 1.0):
            return jsonify({"ok": False, "error": "Rating must be in [0,1]."}), 400

        # persist for future offline training
        meta = dict(meta); meta["label"] = y
        _safe_dump_jsonl(ML_TRAIN_FILE, {"features": features, "meta": meta})

        # immediate online update
        steps = int(data.get("online_steps", 10))
        lr = float(data.get("online_lr", 1e-2))
        _, x = _features_to_vector(features)
        global _ML_MODEL
        for _ in range(max(1, steps)):
            _adam_step(_ML_MODEL, x, y, lr=lr)

        # save updated model (optional but handy)
        ml_save_model(_ML_MODEL, ML_MODEL_PATH)
        return jsonify({"ok": True, "online_steps": steps})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.post("/ml/reset")
def ml_reset():
    try:
        if os.path.exists(ML_MODEL_PATH):
            os.remove(ML_MODEL_PATH)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Failed to remove model: {e}"}), 500
    global _ML_MODEL
    _ML_MODEL = _new_model(len(_FEATURE_KEYS))
    return jsonify({"ok": True, "message": "Model reset; no model loaded."})

# --------------------------
#  Entrypoint
# --------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    # For local dev, you can set host="0.0.0.0" if you need LAN access.
    app.run(host="127.0.0.1", port=port, debug=os.getenv("FLASK_DEBUG", "0") == "1")
