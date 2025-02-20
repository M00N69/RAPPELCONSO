import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime
from urllib.parse import urlencode
import google.generativeai as genai

# --- Configuration de la page ---
st.set_page_config(layout="wide", page_title="RappelConso", page_icon="🔍")

# --- Constantes ---
DATASET_ID = "rappelconso0"
BASE_URL = "https://data.economie.gouv.fr/api/records/1.0/search/"
START_DATE = datetime(2022, 1, 1).date()
TODAY = datetime.now().date()
CATEGORY_FILTER = "Alimentation"

# --- Clé API Gemini Pro ---
api_key = st.secrets["api_key"]
genai.configure(api_key=api_key)

generation_config = genai.GenerationConfig(
    temperature=0.2, top_p=0.4, top_k=32, max_output_tokens=256
)

system_instruction = """Vous êtes un assistant spécialisé en rappels de produits alimentaires en France. 
Répondez avec des informations factuelles en vous basant uniquement sur les données disponibles."""

# --- Chargement des données ---
@st.cache_data
def load_data():
    """Charge et traite les rappels de produits depuis l'API gouvernementale."""
    all_data = []
    offset = 0
    limit = 10000

    try:
        while True:
            params = {
                "dataset": DATASET_ID,
                "q": f'categorie_de_produit:"{CATEGORY_FILTER}"',
                "rows": limit,
                "start": offset,
            }
            url = f"{BASE_URL}?{urlencode(params)}"
            response = requests.get(url)
            response.raise_for_status()

            data = response.json().get("records", [])
            if not data:
                break

            all_data.extend(data)
            offset += limit
            if offset > 50000:
                st.warning("Trop de données chargées, les résultats sont limités.")
                break
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None

    if all_data:
        df = pd.DataFrame([rec["fields"] for rec in all_data])
        if "date_de_publication" in df.columns:
            df["date_de_publication"] = pd.to_datetime(
                df["date_de_publication"], errors="coerce"
            ).dt.date
            df = df.dropna(subset=["date_de_publication"])
            df = df[(df["date_de_publication"] >= START_DATE) & (df["date_de_publication"] <= TODAY)]
        return df
    else:
        st.error("Aucune donnée disponible.")
        return None

# --- Filtrage des données ---
def filter_data(df, subcategories, risks, search_term, date_range):
    if df is None or df.empty:
        return pd.DataFrame()

    start_date, end_date = date_range
    filtered_df = df[(df["date_de_publication"] >= start_date) & (df["date_de_publication"] <= end_date)]

    if subcategories:
        filtered_df = filtered_df[filtered_df["sous_categorie_de_produit"].isin(subcategories)]
    if risks:
        filtered_df = filtered_df[filtered_df["risques_encourus_par_le_consommateur"].isin(risks)]
    if search_term:
        filtered_df = filtered_df[
            filtered_df.apply(lambda row: search_term.lower() in str(row).lower(), axis=1)
        ]

    return filtered_df

# --- Affichage des métriques ---
def display_metrics(data):
    if data is None:
        st.warning("Données non disponibles.")
        return

    col1, col2 = st.columns([3, 1])
    col1.metric("Total de rappels", len(data))
    if col2.button("🔄 Mettre à jour"):
        st.cache_data.clear()
        st.experimental_rerun()

# --- Affichage des graphiques ---
def display_visualizations(data):
    if data is None or data.empty:
        st.warning("Pas assez de données pour les graphiques.")
        return

    recalls_per_month = data.groupby(data["date_de_publication"].astype(str)).size().reset_index(name="counts")
    fig = px.bar(recalls_per_month, x="date_de_publication", y="counts", title="Rappels par date")
    st.plotly_chart(fig, use_container_width=True)

# --- Chatbot ---
def configure_model():
    try:
        return genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            generation_config=generation_config,
            system_instruction=system_instruction,
        )
    except Exception as e:
        st.error(f"Erreur avec Gemini : {e}")
        return None

# --- Interface principale ---
def main():
    st.title("🔍 RappelConso - Dashboard & Chatbot")
    
    df = load_data()
    if df is None:
        return

    all_subcategories = df["sous_categorie_de_produit"].dropna().unique().tolist()
    all_risks = df["risques_encourus_par_le_consommateur"].dropna().unique().tolist()

    st.sidebar.title("Filtres")
    page = st.sidebar.selectbox("Page", ["Dashboard", "Visualisations", "Chatbot"])
    selected_subcategories = st.sidebar.multiselect("Sous-catégories", options=all_subcategories, default=[])
    selected_risks = st.sidebar.multiselect("Risques", options=all_risks, default=[])
    
    min_date, max_date = df["date_de_publication"].min(), df["date_de_publication"].max()
    selected_dates = st.sidebar.slider("Période", min_value=min_date, max_value=max_date, value=(min_date, max_date))
    search_term = st.sidebar.text_input("Recherche", "")

    filtered_data = filter_data(df, selected_subcategories, selected_risks, search_term, selected_dates)

    if page == "Dashboard":
        display_metrics(filtered_data)
        st.dataframe(filtered_data)
    
    elif page == "Visualisations":
        display_visualizations(filtered_data)
    
    elif page == "Chatbot":
        model = configure_model()
        if model:
            user_input = st.text_area("Posez votre question :", height=100)
            if st.button("Envoyer") and user_input.strip():
                convo = model.start_chat()
                response = convo.send_message(user_input)
                st.markdown(f"**Réponse :** {response.text}")

    st.sidebar.image("https://raw.githubusercontent.com/M00N69/RAPPELCONSO/main/logo%2004%20copie.jpg", width=150)

if __name__ == "__main__":
    main()
