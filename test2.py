import os
import json
import pickle
import requests
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from river import linear_model, preprocessing, metrics, naive_bayes
from math import radians, cos, sin, asin, sqrt

# ----------------------------
# Config & Setup
# ----------------------------
st.set_page_config(page_title="Gerçek Zamanlı Trafik: Google + TomTom", page_icon="🚗", layout="wide")
load_dotenv()

# ----------------------------
# Helper Fonksiyonlar
# ----------------------------

def get_api_keys():
    """Google & TomTom anahtarlarını al"""
    google_key = st.session_state.get("google_api_key") or os.getenv("GOOGLE_API_KEY", "").strip()
    tomtom_key = st.session_state.get("tomtom_api_key") or os.getenv("TOMTOM_API_KEY", "").strip()
    return google_key, tomtom_key

# Haversine: iki koordinat arası km
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# ----------------------------
# Google Geocode
# ----------------------------
def geocode_google(address, api_key):
    """Google Maps Geocoding API — Adres → Koordinat"""
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": api_key, "region": "tr"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    if data["status"] != "OK":
        raise ValueError(f"Google geocode başarısız: {data['status']}")

    result = data["results"][0]
    loc = result["geometry"]["location"]
    formatted = result["formatted_address"]

    # Güvenlik filtresi: Türkiye dışı koordinatları engelle
    if not (35 <= loc["lat"] <= 43 and 25 <= loc["lng"] <= 45):
        raise ValueError(f"Adres Türkiye dışında: {formatted}")

    return loc["lat"], loc["lng"], formatted

# ----------------------------
# TomTom Routing
# ----------------------------
def route_with_traffic(start_lat, start_lon, end_lat, end_lon, api_key):
    """TomTom Routing API - Trafik etkili rota bilgisi"""
    dist = haversine(start_lat, start_lon, end_lat, end_lon)
    if dist > 1500:
        raise ValueError(f"Mesafe çok uzak ({dist:.1f} km). Lütfen aynı bölgede iki konum seçin.")

    base = "https://api.tomtom.com/routing/1/calculateRoute"
    path = f"{start_lat},{start_lon}:{end_lat},{end_lon}/json"
    params = {
        "key": api_key,
        "traffic": "true",
        "computeTravelTimeFor": "all",
        "routeType": "fastest"
    }
    r = requests.get(f"{base}/{path}", params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    routes = data.get("routes", [])
    if not routes:
        raise ValueError("Rota bulunamadı.")
    summary = routes[0]["summary"]
    return {
        "length_m": summary.get("lengthInMeters"),
        "travel_time_s": summary.get("travelTimeInSeconds"),
        "freeflow_time_s": summary.get("noTrafficTravelTimeInSeconds"),
        "traffic_delay_s": summary.get("trafficDelayInSeconds")
    }

# ----------------------------
# Model işlemleri
# ----------------------------
def build_features(now, route_info):
    return {
        "length_km": (route_info["length_m"] or 0) / 1000.0,
        "freeflow_min": (route_info["freeflow_time_s"] or 0) / 60.0,
        "hour": now.hour,
        "dow": now.weekday()
    }

def label_from_delay_ratio(r):
    if r < 1.15:
        return "Düşük"
    elif r < 1.5:
        return "Orta"
    else:
        return "Yüksek"

def load_or_init_models():
    reg_path = "models/traffic_reg.pkl"
    clf_path = "models/traffic_clf.pkl"
    os.makedirs("models", exist_ok=True)
    if os.path.exists(reg_path):
        with open(reg_path, "rb") as f:
            reg_model = pickle.load(f)
    else:
        reg_model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    if os.path.exists(clf_path):
        with open(clf_path, "rb") as f:
            clf_model = pickle.load(f)
    else:
        clf_model = naive_bayes.GaussianNB()
    return reg_model, clf_model, reg_path, clf_path

def save_models(reg_model, clf_model, reg_path, clf_path):
    with open(reg_path, "wb") as f:
        pickle.dump(reg_model, f)
    with open(clf_path, "wb") as f:
        pickle.dump(clf_model, f)

def init_history():
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "reg_metric" not in st.session_state:
        st.session_state["reg_metric"] = metrics.MAE()

def append_history(row):
    st.session_state["history"].append(row)
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame([row])
    file = "data/traffic_stream_log.csv"
    header = not os.path.exists(file)
    df.to_csv(file, mode="a", header=header, index=False)

# ----------------------------
# UI
# ----------------------------
st.title("🚗 Gerçek Zamanlı Trafik Tahmini")

with st.sidebar:
    st.subheader("API Ayarları")
    google_api_key = st.text_input("Google Maps API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password")
    tomtom_api_key = st.text_input("TomTom API Key", value=os.getenv("TOMTOM_API_KEY", ""), type="password")
    st.session_state["google_api_key"] = google_api_key
    st.session_state["tomtom_api_key"] = tomtom_api_key
    st.caption("💡 Google konum doğruluğu sağlar, TomTom trafik verisini hesaplar.")

col1, col2 = st.columns(2)
with col1:
    start_addr = st.text_input("Başlangıç adresi", value="Gaziantep Üniversitesi")
with col2:
    end_addr = st.text_input("Bitiş adresi", value="Sanko AVM")

update = st.button("🔁 Güncelle")

init_history()
reg_model, clf_model, reg_path, clf_path = load_or_init_models()

# ----------------------------
# Main Logic
# ----------------------------
def run_once():
    google_key, tomtom_key = get_api_keys()
    if not google_key or not tomtom_key:
        st.error("Google ve TomTom API anahtarları gerekli.")
        return

    try:
        s_lat, s_lon, s_fmt = geocode_google(start_addr, google_key)
        e_lat, e_lon, e_fmt = geocode_google(end_addr, google_key)
        st.info(f"📍 Başlangıç: {s_fmt}\n\n📍 Bitiş: {e_fmt}")
    except Exception as e:
        st.error(f"Google Geocode hatası: {e}")
        return

    try:
        route_info = route_with_traffic(s_lat, s_lon, e_lat, e_lon, tomtom_key)
    except Exception as e:
        st.error(f"Rota isteği hatası: {e}")
        return

    now = datetime.now()
    x = build_features(now, route_info)
    y_time = route_info["travel_time_s"] or 0.0
    y_ff = route_info["freeflow_time_s"] or max(1.0, y_time)
    delay_ratio = y_time / y_ff if y_ff > 0 else 1.0
    y_class = label_from_delay_ratio(delay_ratio)

    y_pred_time = reg_model.predict_one(x) or 0.0
    y_pred_class = clf_model.predict_one(x) or "Düşük"

    metric = st.session_state["reg_metric"]
    metric.update(y_time, y_pred_time)

    reg_model.learn_one(x, y_time)
    clf_model.learn_one(x, y_class)
    save_models(reg_model, clf_model, reg_path, clf_path)

    hist_df = pd.DataFrame(st.session_state["history"][-20:])
    hist_df = hist_df.rename(columns={
        "timestamp": "Zaman Damgası",
        "start": "Başlangıç Adresi",
        "end": "Bitiş Adresi",
        "length_m": "Mesafe (m)",
        "travel_time_s_obs": "Gözlenen Süre (sn)",
        "freeflow_time_s_obs": "Serbest Akış Süresi (sn)",
        "delay_ratio_obs": "Gecikme Oranı",
        "reg_pred_time_s": "Tahmin Edilen Süre (sn)",
        "clf_pred_class": "Tahmin Edilen Sınıf",
        "mae_seconds": "Ortalama Hata (sn)"
    })
    st.dataframe(hist_df, use_container_width=True)

    row = {
        "Zaman Başlangıcı": now.strftime("%Y-%m-%d %H:%M:%S"),
        "Başlangıç": start_addr,
        "Bitiş": end_addr,
        "length_m": route_info["length_m"],
        "travel_time_s_obs": y_time,
        "freeflow_time_s_obs": y_ff,
        "delay_ratio_obs": delay_ratio,
        "reg_pred_time_s": float(y_pred_time or 0.0),
        "clf_pred_class": y_pred_class,
        "mae_seconds": metric.get()
    }
    append_history(row)

    st.success("✅ Veri çekildi, tahmin yapıldı ve model güncellendi.")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Mesafe (km)", f"{(route_info['length_m'] or 0)/1000:.2f}")
    with kpi2:
        st.metric("Gözlenen Süre (dk)", f"{y_time/60:.1f}")
    with kpi3:
        st.metric("Serbest Akış Süresi (dk)", f"{y_ff/60:.1f}")
    with kpi4:
        st.metric("Gecikme Oranı", f"{delay_ratio:.2f}")

    st.write("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Regresyon (Süre) — Tahmin vs. Gözlem")
        st.write(f"🔮 Tahmin (dk): **{(y_pred_time/60):.1f}**")
        st.write(f"🎯 Gerçek (dk): **{(y_time/60):.1f}**")
        st.caption(f"MAE (s): {metric.get():.1f}")
    with c2:
        st.subheader("Sınıflandırma (Yoğunluk)")
        st.write(f"🔮 Tahmin edilen: **{y_pred_class}**")
        st.write(f"🎯 Gerçek sınıf: **{y_class}**")

    st.write("---")
    st.subheader("📊 Geçmiş (Son 20)")
    hist_df = pd.DataFrame(st.session_state["history"][-20:])
    st.dataframe(hist_df, width='stretch')

# ----------------------------
# Güncelle Butonu
# ----------------------------
if update:
    run_once()
else:
    st.info("🔁 Yeni trafik verisini almak için 'Güncelle' butonuna bas.")
