import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import date
import urllib.parse

# --- Configuration de la page ---
st.set_page_config(page_title="Rappels Produits Alimentaires", layout="wide")

# --- Constantes API ---
API_URL = "https://data.economie.gouv.fr/api/records/1.0/search/"
DATASET = "rappelconso-v2-gtin-espaces"
START_DATE = date(2022, 1, 1)
ROWS_LIMIT = 10000
API_TIMEOUT_SEC = 30

# --- Fonction pour récupérer les données depuis l'API ---
@st.cache_data(show_spinner=True)
def load_data():
    """Charge les rappels alimentaires depuis l'API."""
    query_params = {
        "dataset": DATASET,
        "q": "",
        "rows": ROWS_LIMIT,
        "refine.date_publication": f">={START_DATE.strftime('%Y-%m-%d')}",
        "refine.categorie_produit": "alimentation"
    }
    query_string = urllib.parse.urlencode(query_params, safe=":=,")
    url = f"{API_URL}?{query_string}"

    with st.spinner("🔄 Chargement des rappels alimentaires..."):
        try:
            response = requests.get(url, timeout=API_TIMEOUT_SEC)
            response.raise_for_status()
            records = response.json().get("records", [])
            df = pd.DataFrame([rec["fields"] for rec in records])
            df["date_publication"] = pd.to_datetime(df["date_publication"], errors="coerce").dt.date
            df = df.sort_values(by="date_publication", ascending=False)
            return df
        except Exception as e:
            st.error(f"❌ Erreur de chargement des données : {e}")
            return pd.DataFrame()

# --- Fonction de filtrage des données ---
def filter_data(df, subcategories, risks, search_term, date_range):
    """Filtre les données selon les critères sélectionnés."""
    start_date, end_date = date_range
    filtered_df = df[(df["date_publication"] >= start_date) & (df["date_publication"] <= end_date)]

    if subcategories:
        filtered_df = filtered_df[filtered_df["sous_categorie_produit"].isin(subcategories)]
    if risks:
        filtered_df = filtered_df[filtered_df["risques_encourus"].isin(risks)]
    if search_term:
        filtered_df = filtered_df[filtered_df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]

    return filtered_df

# --- Affichage des métriques ---
def display_metrics(data):
    """Affiche les statistiques principales."""
    st.metric("📢 Nombre total de rappels", len(data))

# --- Affichage des rappels récents avec pagination ---
def display_recent_recalls(data, start_index=0, items_per_page=10):
    """Affiche les rappels avec images et pagination."""
    if not data.empty:
        st.subheader("📌 Derniers rappels d'alimentation")
        end_index = min(start_index + items_per_page, len(data))
        current_recalls = data.iloc[start_index:end_index]

        col1, col2 = st.columns(2)
        for idx, row in current_recalls.iterrows():
            with col1 if idx % 2 == 0 else col2:
                img_url = row.get("liens_vers_les_images", "").split("|")[0] if "liens_vers_les_images" in row else None
                if img_url:
                    st.image(img_url, width=120)

                st.markdown(f"""
                **🛒 {row.get('modeles_ou_references', 'N/A')}**  
                📅 **Date de publication** : {row.get('date_publication', 'N/A')}  
                🏷 **Marque** : {row.get('marque_produit', 'N/A')}  
                ⚠ **Motif du rappel** : {row.get('motif_rappel', 'N/A')}  
                🔗 [Voir l'affichette]( {row.get('lien_vers_affichette_pdf', '#')} )
                """)

        # Pagination
        col_prev, col_next = st.columns([1, 1])
        with col_prev:
            if start_index > 0 and st.button("⬅️ Précédent"):
                st.session_state.start_index = max(0, start_index - items_per_page)
        with col_next:
            if end_index < len(data) and st.button("Suivant ➡️"):
                st.session_state.start_index += items_per_page
    else:
        st.warning("⚠ Aucun rappel disponible avec ces filtres.")

# --- Affichage des visualisations ---
def display_visualizations(data):
    """Affiche les graphiques des rappels."""
    if not data.empty:
        fig_subcategories = px.pie(
            data,
            names="sous_categorie_produit",
            title="📊 Répartition par sous-catégories",
        )
        st.plotly_chart(fig_subcategories, use_container_width=True)

        data["mois"] = pd.to_datetime(data["date_publication"]).dt.strftime("%Y-%m")
        recalls_per_month = data.groupby("mois").size().reset_index(name="Nombre de rappels")

        fig_monthly = px.bar(
            recalls_per_month,
            x="mois",
            y="Nombre de rappels",
            title="📈 Évolution des rappels par mois",
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
    else:
        st.warning("⚠ Pas assez de données pour afficher les graphiques.")

# --- Interface principale ---
def main():
    st.title("📢 Rappels Produits Alimentaires")

    df = load_data()
    if df.empty:
        st.error("⚠ Aucune donnée disponible.")
        st.stop()

    all_subcategories = df["sous_categorie_produit"].dropna().unique().tolist()
    all_risks = df["risques_encourus"].dropna().unique().tolist()

    # --- Filtres dans la barre latérale ---
    with st.sidebar:
        st.header("🔍 Filtres")
        selected_subcategories = st.multiselect("Sous-catégories", options=all_subcategories, default=[])
        selected_risks = st.multiselect("Risques", options=all_risks, default=[])
        date_range = st.slider("📆 Période", min_value=df["date_publication"].min(), max_value=df["date_publication"].max(), value=(df["date_publication"].min(), df["date_publication"].max()))
        search_term = st.text_input("🔎 Recherche", "")

    filtered_data = filter_data(df, selected_subcategories, selected_risks, search_term, date_range)

    # --- Affichage du tableau de bord ---
    st.subheader("📊 Statistiques des rappels")
    display_metrics(filtered_data)
    display_visualizations(filtered_data)

    st.subheader("📜 Liste des rappels")
    display_recent_recalls(filtered_data, start_index=st.session_state.get("start_index", 0))

    # --- Téléchargement des données filtrées ---
    if not filtered_data.empty:
        csv = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(label="⬇️ Télécharger les données filtrées", data=csv, file_name="rappels_alimentation.csv", mime="text/csv")

if __name__ == "__main__":
    main()
