"""Interface web : « Vraie photo ou image IA ? »

L'interface ne contient aucun modèle : elle appelle l'API avec la clé de l'utilisateur,
exactement comme le ferait n'importe quel client (curl, application mobile...).
"""

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://gateway")

st.set_page_config(page_title="Real or AI", page_icon="🔍", layout="centered")
st.title("🔍 Vraie photo ou image IA ?")

with st.sidebar:
    st.header("Connexion")
    key = st.text_input("Clé API", value=os.getenv("UI_DEFAULT_KEY", ""), type="password",
                        help="Clé fournie par l'administrateur (commence par roa_)")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"API en ligne — modèle v{health['model_version']}"
                   + (" (INT8)" if health.get("quantized") else ""))
        if health.get("canary"):
            st.info(f"Canary : v{health['canary']['version']} reçoit {health['canary']['weight']:.0%} du trafic")
    except Exception:
        st.error("API injoignable ou modèle en cours de chargement")

headers = {"X-API-Key": key}
tab_predict, tab_usage, tab_drift = st.tabs(["Analyser une image", "Mon usage", "Surveillance (admin)"])

with tab_predict:
    file = st.file_uploader("Déposez une image", type=["jpg", "jpeg", "png", "webp"])
    if file is not None:
        st.image(file, use_container_width=True)
        if not key:
            st.warning("Entrez votre clé API dans le panneau de gauche.")
        else:
            r = requests.post(f"{API_URL}/predict", headers=headers,
                              files={"file": (file.name, file.getvalue())}, timeout=30)
            if r.status_code == 200:
                res = r.json()
                prob = res["prob_ai"]
                if res["label"] == "ai":
                    st.error(f"### 🤖 Image générée par IA — {prob:.0%}")
                else:
                    st.success(f"### 📷 Vraie photo — {1 - prob:.0%}")
                st.progress(prob, text=f"Probabilité « IA » : {prob:.1%}")
                st.caption(f"Modèle v{res['model_version']} ({res['model_alias']}) — {res['latency_ms']:.0f} ms")
            else:
                st.error(f"Erreur {r.status_code} : {r.json().get('detail', r.text)}")

with tab_usage:
    if key:
        r = requests.get(f"{API_URL}/usage/me", headers=headers, timeout=10)
        if r.status_code == 200:
            u = r.json()
            c1, c2, c3 = st.columns(3)
            c1.metric("Requêtes au total", u["requests_total"])
            c2.metric("Dernières 24 h", u["requests_last_24h"])
            c3.metric("Latence moyenne", f"{u['avg_latency_ms']:.0f} ms")
            st.caption(f"Utilisateur : {u['user']} — dernière requête : {u['last_request']}")
        else:
            st.error(r.json().get("detail", r.text))
    else:
        st.info("Entrez votre clé API.")

with tab_drift:
    r = requests.get(f"{API_URL}/admin/drift", headers=headers, timeout=10) if key else None
    if r is None:
        st.info("Entrez une clé administrateur.")
    elif r.status_code != 200:
        st.warning(r.json().get("detail", r.text))
    elif not r.json():
        st.info("Pas encore de calcul de drift (il faut au moins 100 images récentes).")
    else:
        df = pd.DataFrame(r.json())
        df["ts"] = pd.to_datetime(df["ts"])
        last = df.iloc[-1]
        if last["drift"]:
            st.error(f"⚠️ DRIFT DÉTECTÉ — score {last['score']:.2f} (seuil {last['threshold']:.2f})")
        else:
            st.success(f"✅ Pas de drift — score {last['score']:.2f} (seuil {last['threshold']:.2f})")
        st.line_chart(df.set_index("ts")[["score", "threshold"]])
        st.caption("Score = capacité d'un classifieur à distinguer les images récentes des images d'entraînement "
                   "(0,5 : identiques ; 1 : totalement différentes). Fenêtre : 200 dernières images.")
