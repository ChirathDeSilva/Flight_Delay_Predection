"""



Flight delay prediction backend (Flask).







Run:   python app.py        then open  http://127.0.0.1:5000



Needs: backend/artifacts/ created by the "Part 10" export cell of the notebook.



"""



import json



import math



import re



from pathlib import Path







import joblib



import numpy as np



import pandas as pd



from flask import Flask, jsonify, render_template, request, send_file







BASE = Path(__file__).resolve().parent



ART = BASE / "artifacts"







# ----------------------------------------------------------------------------------------



# Load the trained model + the same preprocessing objects used in the notebook



# ----------------------------------------------------------------------------------------



bundle = joblib.load(ART / "model_bundle.joblib")



MODEL, ENCODER = bundle["model"], bundle["encoder"]



MULT = np.asarray(bundle["multipliers"], dtype=float)      # class multipliers tuned on validation



CLASS_W = np.asarray(bundle["class_w"], dtype=float)        # balanced class weights used in training



FEATURES, CAT_COLS = bundle["features"], bundle["cat_cols"]



CLASS_NAMES = bundle["class_names"]



MODEL_NAME = bundle["model_name"]







with open(ART / "meta.json", encoding="utf-8") as f:



    META = json.load(f)







_ctx = np.load(ART / "context_tables.npz")



APT = [_ctx["apt_n"], _ctx["apt_s"], _ctx["apt_l"]]          # cumulative airport tables



CAR = [_ctx["car_n"], _ctx["car_s"], _ctx["car_l"]]          # cumulative airline tables



AIRPORT_IDX = {a: i for i, a in enumerate(META["airport_order"])}



CARRIER_IDX = {c: i for i, c in enumerate(META["carrier_order"])}



H, LAG_H, WINDOWS = META["H"], META["lag_h"], tuple(META["windows"])



START = pd.Timestamp(META["start_date"])



DATE_MIN, DATE_MAX = pd.Timestamp(META["date_min"]), pd.Timestamp(META["date_max"])



WEATHER_COLS = ["O_TEMP", "O_PRCP", "O_WSPD", "D_TEMP", "D_PRCP", "D_WSPD"]











def _build_rate_map(cum, win=3, n_min=5):



    """Learn from the stored 2024 tables how 'average delay' relates to 'share of flights 15+ min late'.



    Used when a user types an average delay by hand and the share has to be estimated."""



    cn, cs, cl = cum



    n, s_, l_ = cn[:, win:] - cn[:, :-win], cs[:, win:] - cs[:, :-win], cl[:, win:] - cl[:, :-win]



    ok = n >= n_min



    if ok.sum() < 200:



        return np.array([0.0, 30.0]), np.array([0.2, 0.6]), 0.0



    mean, rate = s_[ok] / n[ok], l_[ok] / n[ok]



    edges = np.unique(np.quantile(mean, np.linspace(0, 1, 41)))



    idx = np.clip(np.digitize(mean, edges[1:-1]), 0, len(edges) - 2)



    xs, ys = [], []



    for b in range(len(edges) - 1):



        sel = idx == b



        if sel.sum() >= 20:



            xs.append(float(np.median(mean[sel])))



            ys.append(float(np.median(rate[sel])))



    return np.array(xs), np.maximum.accumulate(np.array(ys)), float(np.median(mean))











_apt_map = _build_rate_map(APT)



RATE_MAP = {"o": _apt_map, "d": _apt_map, "c": _build_rate_map(CAR, n_min=20)}



TYPICAL_RECENT = {p: round(RATE_MAP[p][2], 1) for p in ("o", "d", "c")}



RECENT_RANGE = (-30.0, 300.0)











def rate_from_mean(prefix, mean_delay):



    xs, ys, _ = RATE_MAP[prefix]



    return float(np.interp(mean_delay, xs, ys))











def typical_n(cum, code, win):



    """Typical number of flights in a window of `win` hours for this airport / airline."""



    cn = cum[0]



    n = cn[code, win:] - cn[code, :-win]



    n = n[n > 0]



    return float(np.median(n)) if n.size else 0.0











def typical_sched(code, hour):



    """Typical number of scheduled departures at this airport in this hour of the day."""



    per_hour = np.diff(APT[0][code])



    sel = per_hour[np.arange(len(per_hour)) % 24 == hour]



    return float(np.median(sel)) if sel.size else float("nan")







CLASS_INFO = {



    "On-time": "arrives less than 15 minutes late",



    "Minor": "arrives 15 to 44 minutes late",



    "Moderate": "arrives 45 to 119 minutes late",



    "Severe": "arrives 2 hours or more late",



}



TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")









# Display names only; model input codes remain unchanged.

import csv

CARRIER_NAMES = {

    "9E": "Endeavor Air", "AA": "American Airlines", "AS": "Alaska Airlines",

    "B6": "JetBlue Airways", "DL": "Delta Air Lines", "F9": "Frontier Airlines",

    "G4": "Allegiant Air", "HA": "Hawaiian Airlines", "MQ": "Envoy Air",

    "NK": "Spirit Airlines", "OH": "PSA Airlines", "OO": "SkyWest Airlines",

    "UA": "United Airlines", "WN": "Southwest Airlines", "YX": "Republic Airways",

    "YV": "Mesa Airlines", "QX": "Horizon Air"

}

AIRPORT_NAMES = {

    "ATL": "Atlanta — Hartsfield–Jackson Atlanta International Airport",

    "JFK": "New York — John F. Kennedy International Airport",

    "LAX": "Los Angeles — Los Angeles International Airport",

    "ORD": "Chicago — Chicago O'Hare International Airport",

    "DFW": "Dallas–Fort Worth — Dallas Fort Worth International Airport",

    "DEN": "Denver — Denver International Airport",

    "SFO": "San Francisco — San Francisco International Airport",

    "SEA": "Seattle — Seattle–Tacoma International Airport",

    "MIA": "Miami — Miami International Airport",

    "BOS": "Boston — Logan International Airport",

    "EWR": "Newark — Newark Liberty International Airport",

    "LGA": "New York — LaGuardia Airport",

    "LAS": "Las Vegas — Harry Reid International Airport",

    "MCO": "Orlando — Orlando International Airport",

    "PHX": "Phoenix — Phoenix Sky Harbor International Airport",

    "IAD": "Washington — Washington Dulles International Airport",

    "DCA": "Washington — Ronald Reagan Washington National Airport",

    "PHL": "Philadelphia — Philadelphia International Airport",

    "CLT": "Charlotte — Charlotte Douglas International Airport",

    "IAH": "Houston — George Bush Intercontinental Airport",

    "HOU": "Houston — William P. Hobby Airport",

    "DTW": "Detroit — Detroit Metropolitan Wayne County Airport",

    "MSP": "Minneapolis — Minneapolis–Saint Paul International Airport",

    "SLC": "Salt Lake City — Salt Lake City International Airport",

    "SAN": "San Diego — San Diego International Airport",

    "TPA": "Tampa — Tampa International Airport",

    "FLL": "Fort Lauderdale — Fort Lauderdale–Hollywood International Airport",

    "BWI": "Baltimore — Baltimore/Washington International Airport",

    "PDX": "Portland — Portland International Airport",

    "HNL": "Honolulu — Daniel K. Inouye International Airport"

}

reference = BASE / "airports.csv"

if reference.exists():

    with reference.open(encoding="utf-8-sig", newline="") as f:

        for a in csv.DictReader(f):

            code = a.get("iata_code", "").strip().upper()

            name = a.get("name", "").strip()

            if code and name and a.get("iso_country") == "US":

                city = a.get("municipality", "").strip()

                AIRPORT_NAMES[code] = f"{city} — {name}" if city else name



def display_label(code, names):

    name = names.get(code)

    return f"{name} ({code})" if name else f"{code} (name unavailable)"



app = Flask(__name__)





def performance_data():

    """Load exported evaluation artifacts without inventing fallback metrics."""

    comparison_path = BASE.parent / "outputs_4class" / "model_comparison_validation.csv"

    comparison = []

    if comparison_path.exists():

        try:

            frame = pd.read_csv(comparison_path).replace({np.nan: None})

            comparison = frame.to_dict(orient="records")

        except (OSError, ValueError, pd.errors.ParserError):

            app.logger.exception("Could not load model comparison artifact")

    return {

        "metrics": META.get("test_metrics") or {},

        "comparison": comparison,

        "confusion_matrix_available": (BASE.parent / "outputs_4class" / "confusion_matrix_test.png").exists(),

    }











# ----------------------------------------------------------------------------------------



# Input validation



# ----------------------------------------------------------------------------------------



def parse_time(value):



    m = TIME_RE.match(str(value).strip())



    return (int(m.group(1)), int(m.group(2))) if m else None











def read_number(payload, key, errors, lo=None, hi=None):



    v = payload.get(key)



    if v is None or (isinstance(v, str) and v.strip() == ""):



        return None



    try:



        x = float(v)



    except (TypeError, ValueError):



        errors[key] = "Must be a number."



        return None



    if not math.isfinite(x):



        errors[key] = "Must be a finite number."



        return None



    if lo is not None and not (lo <= x <= hi):



        errors[key] = f"Outside the range seen in the data ({lo:.4g} to {hi:.4g})."



        return None



    return x











def validate(payload):



    errors, warnings, c = {}, [], {}







    # ---- required fields ----



    raw_date = str(payload.get("flight_date", "")).strip()



    try:



        c["date"] = pd.Timestamp(raw_date)



        if not re.match(r"^\d{4}-\d{2}-\d{2}$", raw_date):



            raise ValueError



    except Exception:



        errors["flight_date"] = "Enter a valid date (YYYY-MM-DD)."







    carrier = str(payload.get("carrier", "")).strip().upper()



    if not carrier:



        errors["carrier"] = "Choose an airline."



    elif carrier not in CARRIER_IDX:



        errors["carrier"] = "Unknown airline code."



    c["carrier"] = carrier







    for key, label in [("origin", "origin airport"), ("dest", "destination airport")]:



        code = str(payload.get(key, "")).strip().upper()



        if not code:



            errors[key] = f"Enter the {label} code (e.g. ATL)."



        elif code not in AIRPORT_IDX or code not in META["coords"]:



            errors[key] = f"Unknown {label} code."



        c[key] = code



    if c["origin"] and c["origin"] == c["dest"] and "origin" not in errors and "dest" not in errors:



        errors["dest"] = "Destination must be different from the origin."







    dep = parse_time(payload.get("dep_time", ""))



    if dep is None:



        errors["dep_time"] = "Enter the scheduled departure time as HH:MM (24-hour)."



    c["dep"] = dep







    # ---- optional fields ----



    arr_raw = str(payload.get("arr_time", "") or "").strip()



    arr = None



    if arr_raw:



        arr = parse_time(arr_raw)



        if arr is None:



            errors["arr_time"] = "Enter the scheduled arrival time as HH:MM (24-hour)."



    c["arr"] = arr



    c["duration"] = read_number(payload, "duration", errors, 20, 1500)







    c["weather"] = {}



    for col in WEATHER_COLS:



        lo, hi = META["weather_range"][col]



        pad = 0.1 * (hi - lo)



        c["weather"][col] = read_number(payload, col.lower(), errors, lo - pad, hi + pad)







    c["recent"] = {}



    for p, key in [("o", "recent_o_delay"), ("d", "recent_d_delay"), ("c", "recent_c_delay")]:



        c["recent"][p] = read_number(payload, key, errors, *RECENT_RANGE)







    # ---- warnings (not errors) ----



    if "flight_date" not in errors:



        if not (pd.Timestamp("2000-01-01") <= c["date"] <= pd.Timestamp("2100-12-31")):



            errors["flight_date"] = "Enter a date between 2000 and 2100."



        elif not (DATE_MIN <= c["date"] <= DATE_MAX):



            names = {"o": "origin airport", "d": "destination airport", "c": "airline"}



            missing = [n for p, n in names.items() if c["recent"][p] is None]



            if len(missing) == 3:



                warnings.append(



                    f"No stored recent-delay data exists for this date (the data covers {DATE_MIN.date()} to "



                    f"{DATE_MAX.date()}). Enter the recent delays in the optional section for a better estimate; "



                    "without them the prediction is less accurate.")



            elif missing:



                warnings.append("No stored recent-delay data exists for this date. Recent delay was not given for: "



                                + ", ".join(missing) + ", so those parts are left out.")



    if not errors and f"{c['origin']}-{c['dest']}" not in META["route_dur"]:



        warnings.append("This route does not appear in the 2024 data, so the estimate is less reliable.")



    return c, errors, warnings











# ----------------------------------------------------------------------------------------



# Feature building (must match the notebook's make_features / context_features exactly)



# ----------------------------------------------------------------------------------------



def haversine_km(a, b):



    lat1, lon1, lat2, lon2 = map(math.radians, [a[0], a[1], b[0], b[1]])



    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2



    return 6371 * 2 * math.asin(math.sqrt(min(max(h, 0), 1)))











def window_stats(cum, code, hb, win):



    cn, cs, cl = cum



    hi = int(min(max(hb - LAG_H + 1, 0), H))



    lo = int(min(max(hb - LAG_H - win + 1, 0), H))



    n = cn[code, hi] - cn[code, lo]



    if n < 1:



        return np.nan, np.nan, 0.0



    return (cs[code, hi] - cs[code, lo]) / n, (cl[code, hi] - cl[code, lo]) / n, float(n)











def context_features(origin, dest, carrier, hb, dep_hour, manual):



    """Recent-delay features. Priority: value typed by the user > stored 2024 history > not available."""



    o, d, c = AIRPORT_IDX.get(origin), AIRPORT_IDX.get(dest), CARRIER_IDX.get(carrier)



    out, source = {}, {}



    for prefix, cum, code in [("o", APT, o), ("d", APT, d), ("c", CAR, c)]:



        typed = manual.get(prefix)



        use_typed = typed is not None and code is not None



        source[prefix] = "your input" if use_typed else ("2024 history" if (hb is not None and code is not None) else "not available")



        for w in WINDOWS:



            if use_typed:



                m, r = float(typed), rate_from_mean(prefix, float(typed))



                n = typical_n(cum, code, w) if w == WINDOWS[0] else 0.0



            elif hb is None or code is None:



                m, r, n = np.nan, np.nan, 0.0



            else:



                m, r, n = window_stats(cum, code, hb, w)



            out[f"{prefix}_delay_mean_{w}h"] = m



            out[f"{prefix}_late_rate_{w}h"] = r



            if w == WINDOWS[0]:



                out[f"{prefix}_n_{w}h"] = n



    for prefix, code in [("o", o), ("d", d)]:



        if code is None:



            out[f"{prefix}_sched_n_same_hour"] = np.nan



        elif hb is None:



            out[f"{prefix}_sched_n_same_hour"] = typical_sched(code, dep_hour)



        else:



            b = min(max(hb, 0), H - 1)



            out[f"{prefix}_sched_n_same_hour"] = APT[0][code, b + 1] - APT[0][code, b]



    return out, source











def time_feats(prefix, hour, minute):



    mod = hour * 60 + minute



    return {f"{prefix}_hour": hour, f"{prefix}_minute": minute,



            f"{prefix}_sin": math.sin(2 * math.pi * mod / 1440),



            f"{prefix}_cos": math.cos(2 * math.pi * mod / 1440)}











def build_features(c):



    """Returns (1-row DataFrame in model order, assumptions made for missing inputs, source of recent-delay data)."""



    assumptions = []



    date, (dh, dm) = c["date"], c["dep"]



    oc, dc = META["coords"][c["origin"]], META["coords"][c["dest"]]



    dist = haversine_km(oc, dc)







    duration = c["duration"]



    if duration is None:



        duration = META["route_dur"].get(f"{c['origin']}-{c['dest']}")



        if duration is None:



            duration = 30 + dist / 13.5



            assumptions.append(f"Scheduled duration estimated from distance: {duration:.0f} min.")



        else:



            assumptions.append(f"Scheduled duration set to the typical value for this route: {duration:.0f} min.")







    if c["arr"] is None:



        total = int(round(dh * 60 + dm + duration)) % 1440



        ah, am = divmod(total, 60)



        assumptions.append(f"Scheduled arrival time calculated as {ah:02d}:{am:02d}.")



    else:



        ah, am = c["arr"]







    weather, filled = {}, False



    for col in WEATHER_COLS:



        v = c["weather"][col]



        if v is None:



            side, airport = col[0], (c["origin"] if col[0] == "O" else c["dest"])



            vals = META["weather_by_airport_month"].get(f"{side}|{airport}|{date.month}")



            idx = ["TEMP", "PRCP", "WSPD"].index(col[2:])



            v = vals[idx] if vals and vals[idx] is not None else META["weather_global"][col]



            filled = True



        weather[col] = v



    if filled:



        assumptions.append("Weather left blank: typical values for that airport and month were used.")







    hb = None



    if DATE_MIN <= date <= DATE_MAX:



        hb = int((date - START).days) * 24 + dh







    row = {"OP_CARRIER": c["carrier"], "ORIGIN": c["origin"], "DEST": c["dest"],



           "CRS_ELAPSED_TIME": duration, **weather,



           "month": date.month, "day_of_month": date.day, "day_of_week": date.dayofweek,



           "month_sin": math.sin(2 * math.pi * date.month / 12),



           "month_cos": math.cos(2 * math.pi * date.month / 12),



           "distance_km": dist}



    row.update(time_feats("CRS_DEP_TIME", dh, dm))



    row.update(time_feats("CRS_ARR_TIME", ah, am))



    ctx, source = context_features(c["origin"], c["dest"], c["carrier"], hb, dh, c["recent"])



    row.update(ctx)



    labels = {"o": "the origin airport", "d": "the destination airport", "c": "the airline"}



    for p in ("o", "d", "c"):



        if source[p] == "your input":



            assumptions.append(f"Recent delay at {labels[p]} taken from your input ({c['recent'][p]:.0f} min); "



                               "the share of late flights was estimated from 2024 patterns.")



    if hb is None:



        assumptions.append("Scheduled traffic per hour estimated from typical 2024 levels (date outside the stored period).")







    X = pd.DataFrame([row])



    X[CAT_COLS] = ENCODER.transform(X[CAT_COLS].astype(str))



    return X.reindex(columns=FEATURES).astype(np.float32), assumptions, source











def predict_from_clean(c):



    X, assumptions, source = build_features(c)



    proba = MODEL.predict_proba(X)[0]



    pred = int(np.argmax(proba * MULT))                      # decision rule tuned for macro-F1



    est = proba / CLASS_W                                    # undo the balanced training weights



    est = est / est.sum()                                    # -> estimated real-world chances



    risk = float(1 - est[0])



    level = "Low" if risk < 0.15 else ("Medium" if risk < 0.30 else "High")



    name = CLASS_NAMES[pred]



    return {



        "prediction": name,



        "class_index": pred,



        "description": CLASS_INFO[name],



        "delay_risk": round(risk, 4),



        "risk_level": level,



        "probabilities": {n: round(float(p), 4) for n, p in zip(CLASS_NAMES, est)},



        "assumptions": assumptions,



        "context_source": {"origin": source["o"], "destination": source["d"], "airline": source["c"]},



    }











# ----------------------------------------------------------------------------------------



# Routes



# ----------------------------------------------------------------------------------------



@app.get("/")



def home():



    return render_page("home")











def render_page(page):

    return render_template("index.html", page=page, model_name=MODEL_NAME,

                           metrics=META.get("test_metrics") or {},

                           performance=performance_data(),

                           date_min=META["date_min"], date_max=META["date_max"],

                           class_info=CLASS_INFO,

                           airport_count=len(AIRPORT_IDX), airline_count=len(CARRIER_IDX))





@app.get("/prediction")

def prediction_page():

    return render_page("prediction")





@app.get("/performance")

def performance_page():

    return render_page("performance")





@app.get("/about")

def about_page():

    return render_page("about")





@app.get("/api/health")



def health():



    return jsonify({"status": "ok", "model": MODEL_NAME})











@app.get("/api/meta")



def meta():



    return jsonify({



        "carriers": sorted(META["carrier_order"]),

        "carrier_names": {c: display_label(c, CARRIER_NAMES) for c in META["carrier_order"]},

        "airport_names": {a: display_label(a, AIRPORT_NAMES) for a in META["airport_order"] if a in META["coords"]},



        "airports": sorted(a for a in META["airport_order"] if a in META["coords"]),



        "date_min": META["date_min"], "date_max": META["date_max"],



        "weather_global": META["weather_global"], "weather_range": META["weather_range"],



        "model": MODEL_NAME, "test_metrics": META.get("test_metrics"),



        "class_info": CLASS_INFO,



        "typical_recent": TYPICAL_RECENT, "recent_range": list(RECENT_RANGE),



    })





@app.get("/artifacts/confusion-matrix")

def confusion_matrix():

    path = BASE.parent / "outputs_4class" / "confusion_matrix_test.png"

    if not path.exists():

        return jsonify({"error": "Confusion matrix artifact is not available."}), 404

    return send_file(path, mimetype="image/png")











@app.post("/api/predict")



def predict():



    payload = request.get_json(silent=True)



    if not isinstance(payload, dict):



        return jsonify({"ok": False, "errors": {"_form": "Send the inputs as a JSON object."}}), 400



    clean, errors, warnings = validate(payload)



    if errors:



        return jsonify({"ok": False, "errors": errors}), 400



    try:



        result = predict_from_clean(clean)



    except Exception as exc:                                  # never expose a stack trace to the user



        app.logger.exception("Prediction failed")



        return jsonify({"ok": False, "errors": {"_form": "The prediction failed. Please check the inputs."}}), 500



    result["warnings"] = warnings



    result["ok"] = True



    return jsonify(result)











if __name__ == "__main__":



    app.run(host="127.0.0.1", port=5000, debug=False)
