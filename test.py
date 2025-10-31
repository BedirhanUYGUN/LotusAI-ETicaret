# import os
# import json
# import pickle
# import requests
# import pandas as pd
# import streamlit as st
# from datetime import datetime
# from dotenv import load_dotenv
#
# # River
# from river import linear_model, preprocessing, metrics, naive_bayes
#
# # ----------------------------
# # Config & Setup
# # ----------------------------
# st.set_page_config(page_title="Gerçek Zamanlı Trafik: Tahmin & Sınıflandırma", page_icon="🚗", layout="wide")
# load_dotenv()
#
# # ----------------------------
# # Helper Fonksiyonlar
# # ----------------------------
#
# def get_api_key():
#     ui_key = st.session_state.get("api_key_ui")
#     if ui_key:
#         return ui_key.strip()
#     env_key = os.getenv("TOMTOM_API_KEY", "").strip()
#     return env_key
#
#
# def geocode(address, api_key):
#     """Adres → Koordinat"""
#     url = f"https://api.tomtom.com/search/2/geocode/{requests.utils.quote(address)}.json"
#     params = {"key": api_key, "limit": 1}
#     r = requests.get(url, params=params, timeout=10)
#     r.raise_for_status()
#     data = r.json()
#     results = data.get("results", [])
#     if not results:
#         raise ValueError(f"Adres bulunamadı: {address}")
#     pos = results[0]["position"]
#     return pos["lat"], pos["lon"]
#
#
# def route_with_traffic(start_lat, start_lon, end_lat, end_lon, api_key):
#     """TomTom Routing API - Trafik etkili rota bilgisi"""
#     # Çok uzak mesafe (kıtalar arası) kontrolü
#     from math import radians, cos, sin, asin, sqrt
#     def haversine(lat1, lon1, lat2, lon2):
#         R = 6371
#         dlat = radians(lat2 - lat1)
#         dlon = radians(lon2 - lon1)
#         a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
#         return 2 * R * asin(sqrt(a))
#     dist = haversine(start_lat, start_lon, end_lat, end_lon)
#     if dist > 1500:
#         raise ValueError(f"Mesafe çok uzak ({dist:.1f} km). Lütfen aynı bölgede iki konum seçin.")
#
#     base = "https://api.tomtom.com/routing/1/calculateRoute"
#     path = f"{start_lat},{start_lon}:{end_lat},{end_lon}/json"
#     params = {
#         "key": api_key,
#         "traffic": "true",
#         "computeTravelTimeFor": "all",
#         "routeType": "fastest"
#     }
#     r = requests.get(f"{base}/{path}", params=params, timeout=15)
#     r.raise_for_status()
#     data = r.json()
#     routes = data.get("routes", [])
#     if not routes:
#         raise ValueError("Rota bulunamadı.")
#     summary = routes[0]["summary"]
#     return {
#         "length_m": summary.get("lengthInMeters"),
#         "travel_time_s": summary.get("travelTimeInSeconds"),
#         "freeflow_time_s": summary.get("noTrafficTravelTimeInSeconds"),
#         "traffic_delay_s": summary.get("trafficDelayInSeconds")
#     }
#
#
# def build_features(now, route_info):
#     """Model özellikleri"""
#     return {
#         "length_km": (route_info["length_m"] or 0) / 1000.0,
#         "freeflow_min": (route_info["freeflow_time_s"] or 0) / 60.0,
#         "hour": now.hour,
#         "dow": now.weekday()
#     }
#
#
# def label_from_delay_ratio(r):
#     """Gecikme oranına göre sınıf etiketi"""
#     if r < 1.15:
#         return "Düşük"
#     elif r < 1.5:
#         return "Orta"
#     else:
#         return "Yüksek"
#
#
# def load_or_init_models():
#     """Model dosyaları varsa yükle, yoksa oluştur"""
#     reg_path = "models/traffic_reg.pkl"
#     clf_path = "models/traffic_clf.pkl"
#     os.makedirs("models", exist_ok=True)
#
#     if os.path.exists(reg_path):
#         with open(reg_path, "rb") as f:
#             reg_model = pickle.load(f)
#     else:
#         reg_model = preprocessing.StandardScaler() | linear_model.LinearRegression()
#
#     if os.path.exists(clf_path):
#         with open(clf_path, "rb") as f:
#             clf_model = pickle.load(f)
#     else:
#         clf_model = naive_bayes.GaussianNB()
#
#     return reg_model, clf_model, reg_path, clf_path
#
#
# def save_models(reg_model, clf_model, reg_path, clf_path):
#     with open(reg_path, "wb") as f:
#         pickle.dump(reg_model, f)
#     with open(clf_path, "wb") as f:
#         pickle.dump(clf_model, f)
#
#
# def init_history():
#     if "history" not in st.session_state:
#         st.session_state["history"] = []
#     if "reg_metric" not in st.session_state:
#         st.session_state["reg_metric"] = metrics.MAE()
#
#
# def append_history(row):
#     st.session_state["history"].append(row)
#     os.makedirs("data", exist_ok=True)
#     df = pd.DataFrame([row])
#     file = "data/traffic_stream_log.csv"
#     header = not os.path.exists(file)
#     df.to_csv(file, mode="a", header=header, index=False)
#
# # ----------------------------
# # UI
# # ----------------------------
# st.title("🚗 Gerçek Zamanlı Trafik Tahmini (TomTom + River)")
#
# with st.sidebar:
#     st.subheader("Ayarlar")
#     api_key_ui = st.text_input("TomTom API Key", value=get_api_key(), type="password",
#                                help="Çevre değişkenine TOMTOM_API_KEY olarak da koyabilirsin.")
#     st.session_state["api_key_ui"] = api_key_ui
#     st.caption("🕹️ 'Güncelle' butonuna bastığında yeni veri çekilir ve model güncellenir.")
#
# col1, col2 = st.columns(2)
# with col1:
#     start_addr = st.text_input("Başlangıç adresi", value="Gaziantep Üniversitesi")
# with col2:
#     end_addr = st.text_input("Bitiş adresi", value="Sanko AVM")
#
# update = st.button("🔁 Güncelle")
#
# # Başlangıç ayarları
# init_history()
# reg_model, clf_model, reg_path, clf_path = load_or_init_models()
#
# # ----------------------------
# # Main Logic
# # ----------------------------
# def run_once():
#     api_key = get_api_key()
#     if not api_key:
#         st.error("TomTom API anahtarı gerekli.")
#         return
#
#     try:
#         s_lat, s_lon = geocode(start_addr, api_key)
#         e_lat, e_lon = geocode(end_addr, api_key)
#     except Exception as e:
#         st.error(f"Geocoding hatası: {e}")
#         return
#
#     try:
#         route_info = route_with_traffic(s_lat, s_lon, e_lat, e_lon, api_key)
#     except Exception as e:
#         st.error(f"Rota isteği hatası: {e}")
#         return
#
#     now = datetime.now()
#     x = build_features(now, route_info)
#
#     # Hedef değişkenler
#     y_time = route_info["travel_time_s"] or 0.0
#     y_ff = route_info["freeflow_time_s"] or max(1.0, y_time)
#     delay_ratio = y_time / y_ff if y_ff > 0 else 1.0
#     y_class = label_from_delay_ratio(delay_ratio)
#
#     # Tahminler (öğrenmeden önce)
#     y_pred_time = reg_model.predict_one(x) or 0.0
#     y_pred_class = clf_model.predict_one(x) or "Düşük"
#
#     metric = st.session_state["reg_metric"]
#     metric.update(y_time, y_pred_time)
#
#     # Öğrenme
#     reg_model.learn_one(x, y_time)
#     clf_model.learn_one(x, y_class)
#
#     save_models(reg_model, clf_model, reg_path, clf_path)
#
#     row = {
#         "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
#         "start": start_addr,
#         "end": end_addr,
#         "length_m": route_info["length_m"],
#         "travel_time_s_obs": y_time,
#         "freeflow_time_s_obs": y_ff,
#         "delay_ratio_obs": delay_ratio,
#         "reg_pred_time_s": float(y_pred_time or 0.0),
#         "clf_pred_class": y_pred_class,
#         "mae_seconds": metric.get()
#     }
#     append_history(row)
#
#     # Görseller
#     st.success("✅ Veri çekildi, tahmin yapıldı ve model güncellendi.")
#
#     kpi1, kpi2, kpi3, kpi4 = st.columns(4)
#     with kpi1:
#         st.metric("Mesafe (km)", f"{(route_info['length_m'] or 0)/1000:.2f}")
#     with kpi2:
#         st.metric("Gözlenen Süre (dk)", f"{y_time/60:.1f}")
#     with kpi3:
#         st.metric("Serbest Akış Süresi (dk)", f"{y_ff/60:.1f}")
#     with kpi4:
#         st.metric("Gecikme Oranı", f"{delay_ratio:.2f}")
#
#     st.write("---")
#     c1, c2 = st.columns(2)
#     with c1:
#         st.subheader("Regresyon (Süre) — Tahmin vs. Gözlem")
#         st.write(f"🔮 Tahmin (dk): **{(y_pred_time/60):.1f}**")
#         st.write(f"🎯 Gerçek (dk): **{(y_time/60):.1f}**")
#         st.caption(f"MAE (s): {metric.get():.1f}")
#     with c2:
#         st.subheader("Sınıflandırma (Yoğunluk)")
#         st.write(f"🔮 Tahmin edilen: **{y_pred_class}**")
#         st.write(f"🎯 Gerçek sınıf: **{y_class}**")
#
#     st.write("---")
#     st.subheader("📊 Geçmiş (Son 20)")
#     hist_df = pd.DataFrame(st.session_state["history"][-20:])
#     st.dataframe(hist_df, use_container_width=True)
#
#
# # ----------------------------
# # Güncelle Butonu
# # ----------------------------
# if update:
#     run_once()
# else:
#     st.info("🔁 Yeni trafik verisini almak için 'Güncelle' butonuna bas.")


import os, json, pickle, requests, pandas as pd, streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from river import linear_model, preprocessing, metrics, naive_bayes

st.set_page_config(page_title="Gerçek Zamanlı Trafik Tahmini", page_icon="🚗", layout="wide")
load_dotenv()

# ------------------- Yardımcı Fonksiyonlar -------------------
def get_api_key():
    return st.session_state.get("api_key_ui") or os.getenv("TOMTOM_API_KEY", "").strip()

def geocode(address, api_key):
    """Adres → Koordinat (otomatik Gaziantep, Türkiye ekler)"""
    full_addr = f"{address}, Gaziantep, Türkiye"
    url = f"https://api.tomtom.com/search/2/geocode/{requests.utils.quote(full_addr)}.json"
    r = requests.get(url, params={"key": api_key, "limit": 1}, timeout=10)
    r.raise_for_status()
    data = r.json()
    res = data.get("results", [])
    if not res:
        raise ValueError(f"Adres bulunamadı: {full_addr}")
    pos = res[0]["position"]
    return pos["lat"], pos["lon"]

def route_with_traffic(lat1, lon1, lat2, lon2, key):
    from math import radians, cos, sin, asin, sqrt
    def haversine(a,b,c,d):
        R=6371;dlon=radians(d-b);dlat=radians(c-a)
        return 2*R*asin(sqrt(sin(dlat/2)**2+cos(radians(a))*cos(radians(c))*sin(dlon/2)**2))
    if haversine(lat1,lon1,lat2,lon2)>1500:
        raise ValueError("Mesafe çok uzak, aynı bölgede iki konum seçin.")
    base="https://api.tomtom.com/routing/1/calculateRoute"
    path=f"{lat1},{lon1}:{lat2},{lon2}/json"
    p={"key":key,"traffic":"true","computeTravelTimeFor":"all","routeType":"fastest"}
    r=requests.get(f"{base}/{path}",params=p,timeout=15);r.raise_for_status()
    s=r.json()["routes"][0]["summary"]
    return {"length_m":s["lengthInMeters"],"travel_time_s":s["travelTimeInSeconds"],
            "freeflow_time_s":s["noTrafficTravelTimeInSeconds"],"traffic_delay_s":s["trafficDelayInSeconds"]}

def build_features(now, info):
    return {"length_km":(info["length_m"] or 0)/1000,
            "freeflow_min":(info["freeflow_time_s"] or 0)/60,
            "hour":now.hour,"dow":now.weekday()}

def label_from_delay_ratio(r):
    return "Düşük" if r<1.15 else "Orta" if r<1.5 else "Yüksek"

def load_or_init_models():
    os.makedirs("models",exist_ok=True)
    def load(p,init):
        return pickle.load(open(p,"rb")) if os.path.exists(p) else init
    reg=load("models/traffic_reg.pkl",preprocessing.StandardScaler()|linear_model.LinearRegression())
    clf=load("models/traffic_clf.pkl",naive_bayes.GaussianNB())
    return reg,clf

def save_models(reg,clf):
    pickle.dump(reg,open("models/traffic_reg.pkl","wb"))
    pickle.dump(clf,open("models/traffic_clf.pkl","wb"))

def append_history(row):
    st.session_state["history"].append(row)
    df=pd.DataFrame([row])
    os.makedirs("data",exist_ok=True)
    file="data/traffic_log.csv";df.to_csv(file,mode="a",header=not os.path.exists(file),index=False)

# ------------------- UI -------------------
st.title("🚗 Gerçek Zamanlı Trafik Tahmini (TomTom + River)")
with st.sidebar:
    api_key_ui=st.text_input("TomTom API Key",value=get_api_key(),type="password")
    st.session_state["api_key_ui"]=api_key_ui
    st.caption("🔁 'Güncelle' yeni veriyi alır ve modeli eğitir.\n🔮 'Tahmin Et' yalnızca tahmin yapar ve adresleri sıfırlar.")

col1,col2=st.columns(2)
with col1: start_addr=st.text_input("Başlangıç adresi",value="Gaziantep Üniversitesi")
with col2: end_addr=st.text_input("Bitiş adresi",value="Sanko AVM")

col3,col4=st.columns(2)
with col3: btn_update=st.button("🔁 Güncelle (Veri + Öğrenme)")
with col4: btn_predict=st.button("🔮 Tahmin Et (Modeli Güncellemeden)")

# ------------------- Başlangıç -------------------
if "history" not in st.session_state: st.session_state["history"]=[]
if "reg_metric" not in st.session_state: st.session_state["reg_metric"]=metrics.MAE()
reg_model,clf_model=load_or_init_models()

# ------------------- Ana Fonksiyon -------------------
def run_once(learn=True):
    key=get_api_key()
    if not key: st.error("TomTom API anahtarı gerekli.");return
    try:
        s_lat,s_lon=geocode(start_addr,key);e_lat,e_lon=geocode(end_addr,key)
        info=route_with_traffic(s_lat,s_lon,e_lat,e_lon,key)
    except Exception as e:
        st.error(f"Hata: {e}");return
    now=datetime.now();x=build_features(now,info)
    y_time=info["travel_time_s"];y_ff=info["freeflow_time_s"] or y_time;ratio=y_time/y_ff
    y_class=label_from_delay_ratio(ratio)
    y_pred_t=reg_model.predict_one(x) or 0;y_pred_c=clf_model.predict_one(x) or "Düşük"
    st.session_state["reg_metric"].update(y_time,y_pred_t)
    if learn:
        reg_model.learn_one(x,y_time);clf_model.learn_one(x,y_class);save_models(reg_model,clf_model)
    row={"Zaman":now.strftime("%Y-%m-%d %H:%M:%S"),"Başlangıç":start_addr,"Bitiş":end_addr,
         "Mesafe (m)":info["length_m"],"Gerçek Süre (sn)":y_time,"Serbest Süre (sn)":y_ff,
         "Gecikme Oranı":round(ratio,3),"Tahmin Süre (sn)":round(y_pred_t,2),
         "Tahmin Sınıfı":y_pred_c,"Gerçek Sınıf":y_class,
         "MAE (sn)":round(st.session_state["reg_metric"].get(),2)}
    append_history(row)
    st.success("✅ Tahmin yapıldı"+(" ve model güncellendi." if learn else "."))
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Mesafe (km)",f"{info['length_m']/1000:.2f}")
    c2.metric("Gerçek Süre (dk)",f"{y_time/60:.1f}")
    c3.metric("Serbest Akış (dk)",f"{y_ff/60:.1f}")
    c4.metric("Gecikme Oranı",f"{ratio:.2f}")
    st.write("---")
    c5,c6=st.columns(2)
    with c5:
        st.subheader("Regresyon (Süre)")
        st.write(f"🔮 Tahmin (dk): **{y_pred_t/60:.1f}**")
        st.write(f"🎯 Gerçek (dk): **{y_time/60:.1f}**")
        st.caption(f"MAE (s): {st.session_state['reg_metric'].get():.1f}")
    with c6:
        st.subheader("Sınıflandırma (Yoğunluk)")
        st.write(f"🔮 Tahmin edilen: **{y_pred_c}**")
        st.write(f"🎯 Gerçek sınıf: **{y_class}**")
    st.write("---")
    st.subheader("📊 Geçmiş (Son 20)")
    st.dataframe(pd.DataFrame(st.session_state["history"][-20:]),use_container_width=True)

# ------------------- Buton İşlevleri -------------------
if btn_update: run_once(learn=True)
elif btn_predict:
    run_once(learn=False)
    # adresleri sıfırla
    st.session_state["Başlangıç adresi"]=""; st.session_state["Bitiş adresi"]=""
else:
    st.info("🔁 Yeni veri almak için 'Güncelle'ye, yalnızca tahmin için 'Tahmin Et'e bas.")
