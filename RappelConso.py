import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, date, timedelta
import urllib.parse
import time
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from groq import Groq
import re # Pour le Markdown dans la réponse de l'IA

# Configuration de la page
st.set_page_config(
    page_title="RappelConso Insight",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS (inchangé)
st.markdown("""
    <style>
        /* --- Base & Global Styles --- */
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f0f2f6; /* Light gray background */
            color: #333;
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        /* --- Header --- */
        .header-container {
            background: linear-gradient(135deg, #0072C6 0%, #00A0E0 100%); /* Blue gradient */
            padding: 1.5rem 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .header-title {
            color: white;
            font-size: 2.5em;
            font-weight: bold;
            margin: 0;
            animation: fadeInDown 1s ease-out;
        }

        /* --- Sidebar --- */
        .css-1lcbmhc.e1fqkh3o0 { /* Sidebar main */
            background-color: #ffffff;
            border-right: 1px solid #e0e0e0;
        }
        .sidebar-logo {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 80%;
            padding: 10px 0;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .stRadio > label {
            font-size: 1.1em;
            padding: 0.5em 0.75em;
            border-radius: 5px;
            transition: background-color 0.3s ease, color 0.3s ease;
        }
        .stRadio > div[role="radiogroup"] > label:hover {
            background-color: #e6f3ff;
            color: #0072C6;
        }

        /* --- Metric Cards --- */
        .metric-card {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            margin-bottom: 1rem;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
        }
        .metric-value {
            font-size: 2.2em;
            font-weight: bold;
            color: #0072C6;
            margin-bottom: 0.3rem;
        }
        .metric-label {
            font-size: 1em;
            color: #555;
        }

        /* --- Tabs --- */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #e6f3ff;
            border-radius: 8px;
            padding: 5px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: #0072C6;
            font-weight: 500;
            border-radius: 6px;
            transition: background-color 0.3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #cce7ff;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0072C6;
            color: white !important;
            box-shadow: 0 2px 5px rgba(0, 114, 198, 0.3);
        }

        /* --- Recall Cards --- */
        .recall-card-container {
            background-color: white;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.07);
            border-left: 5px solid #00A0E0;
        }
        .recall-card-container h5 {
            color: #005A9C;
            margin-top: 0;
            margin-bottom: 0.25rem;
        }
        .recall-card-container .stImage > img {
            border-radius: 6px;
            object-fit: cover;
        }

        /* --- Buttons --- */
        .stButton > button {
            border-radius: 6px;
            padding: 0.5em 1em;
            font-weight: 500;
            transition: background-color 0.2s ease, transform 0.2s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
        }
        .stButton > button[kind="primary"] {
            background-color: #0072C6;
            color: white;
            border: none;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #005A9C;
        }
        .stButton > button[kind="secondary"] {
            background-color: #f0f2f6;
            color: #333;
            border: 1px solid #ddd;
        }
        .stButton > button[kind="secondary"]:hover {
            background-color: #e0e3e8;
        }

        /* --- Expander --- */
        .stExpander {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background-color: #fafafa;
        }
        .stExpander header {
            font-size: 1.05em;
            font-weight: 600;
            color: #005A9C;
        }

        /* --- Filter Pills --- */
        .filter-pills-container {
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .filter-pills-title {
            font-size: 0.95em;
            font-weight: 600;
            color: #555;
            margin-bottom: 0.3rem;
        }
        .filter-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        .filter-pill {
            background-color: #e6f3ff;
            color: #0072C6;
            padding: 0.3rem 0.7rem;
            border-radius: 15px;
            font-size: 0.9em;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        /* --- Chat Interface --- */
        .chat-container {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 1.5rem;
            box_shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            margin-top: 1rem;
        }
        .stChatMessage {
            border-radius: 15px !important;
            max-width: 85% !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        }
        .stChatMessage > div[data-testid="stChatMessageContent"] {
            border-radius: 15px !important;
        }

        .stChatMessage[data-testid="chatAvatarIcon-user"] + div .stChatMessageContent {
             background-color: #0072C6 !important;
             color: white !important;
             border-bottom-right-radius: 5px !important;
             border-bottom-left-radius: 15px !important;
             border-top-left-radius: 15px !important;
             border-top-right-radius: 15px !important;
        }
        .stChatMessage[data-testid="chatAvatarIcon-assistant"] + div .stChatMessageContent {
            background-color: #e9ecef !important;
            color: #333 !important;
            border-bottom-left-radius: 5px !important;
            border-bottom-right-radius: 15px !important;
            border-top-left-radius: 15px !important;
            border-top-right-radius: 15px !important;
        }
        .stChatMessage img {
            max-width: 100%;
            border-radius: 7px;
        }

        .stTextArea textarea {
            border-radius: 8px;
            border: 1px solid #ccc;
        }
        .stTextArea textarea:focus {
            border-color: #0072C6;
            box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25);
        }

        /* Suggestions d'analyses */
        .suggestion-button-container {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        .suggestion-button-container .stButton button {
            background-color: #e6f3ff;
            color: #0072C6;
            border: 1px solid #cce7ff;
            font-size: 0.9em;
            font-weight: 500;
            padding: 0.4em 0.8em;
        }
        .suggestion-button-container .stButton button:hover {
            background-color: #cce7ff;
            border-color: #00A0E0;
            transform: translateY(-1px);
        }

        /* --- Animations --- */
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .metric-card, .recall-card-container, .chart-container, .chat-container {
            animation: fadeInUp 0.5s ease-out;
        }

        /* --- Responsive Design --- */
        @media (max-width: 768px) {
            .header-title { font-size: 1.8em; }
            .metric-card { padding: 1rem; }
            .metric-value { font-size: 1.8em; }
            .metric-label { font-size: 0.9em; }
            .stTabs [data-baseweb="tab"] { font-size: 0.9em; padding: 8px 10px;}
            .main .block-container { padding-left: 0.5rem; padding-right: 0.5rem;}
            .stChatMessage { max-width: 95% !important; }
        }
    </style>
""", unsafe_allow_html=True)

# --- Constants ---
API_URL_BASE = "https://data.economie.gouv.fr/api/records/1.0/search/"
API_URL_FOR_LOAD = "https://data.economie.gouv.fr/api/records/1.0/search/?dataset=rappelconso-v2-gtin-espaces&q="

START_DATE = date(2022, 1, 1) # Date de début pour le chargement initial
API_TIMEOUT_SEC = 30
DEFAULT_ITEMS_PER_PAGE = 6
DEFAULT_RECENT_DAYS = 30
LOGO_URL = "https://raw.githubusercontent.com/M00N69/RAPPELCONSO/main/logo%2004%20copie.jpg"

# API limit constant
API_MAX_RECORDS_PER_REQUEST = 10000 # The sum of start + rows cannot exceed this


FRIENDLY_TO_API_COLUMN_MAPPING = {
    "Motif du rappel": "motif_du_rappel",
    "Risques encourus": "risques_encourus",
    "Nom de la marque": "nom_de_la_marque_du_produit",
    "Nom commercial": "nom_commercial",
    "Modèle/Référence": "modeles_ou_references",
    "Distributeurs": "distributeurs",
    "Catégorie principale": "categorie_de_produit",
    "Sous-catégorie": "sous_categorie_de_produit"
}
API_TO_FRIENDLY_COLUMN_MAPPING = {v: k for k, v in FRIENDLY_TO_API_COLUMN_MAPPING.items()}


# --- Fonctions de chargement de données (CORRIGÉE pour la limite API et les toasts) ---
@st.cache_data(show_spinner="Chargement des données RappelConso...", ttl=3600)
def load_data(api_url_base_plus_dataset, start_date_filter=START_DATE):
    """Charge les données depuis l'API en utilisant la méthode de filtrage par refine de l'URL."""
    all_records = []
    start_date_str = start_date_filter.strftime('%Y-%m-%d')
    today_str = date.today().strftime('%Y-%m-%d')

    api_url_filtered = (
        f"{api_url_base_plus_dataset}"
        f"&refine.date_publication:>={urllib.parse.quote(start_date_str)}"
        f"&refine.date_publication:<={urllib.parse.quote(today_str)}"
        f"&refine.categorie_de_produit=Alimentation"
    )

    current_start_row = 0
    rows_per_page = 1000 # Keep a reasonable page size for fetching

    while True:
        # Check the API limit BEFORE making the request
        # If current_start_row + rows_per_page exceeds the limit,
        # we need to adjust rows or break. Given the error, the simplest is to break
        # if current_start_row reaches or exceeds the limit, assuming you can't even start there.
        # Let's cap the start row *before* it reaches 10000.
        if current_start_row >= API_MAX_RECORDS_PER_REQUEST:
            # We hit the API's maximum offset limit.
            # A message will be displayed in main() based on the final number of records.
            break

        # Calculate rows for this request, ensuring start + rows <= API_MAX_RECORDS_PER_REQUEST
        # For simplicity with fixed rows_per_page, we just cap the start row as above.
        # The API will naturally return fewer rows on the last possible page.
        # So, rows_to_fetch = rows_per_page is fine here.
        rows_to_fetch = rows_per_page

        paginated_url = f"{api_url_filtered}&start={current_start_row}&rows={rows_to_fetch}"

        try:
            response = requests.get(paginated_url, timeout=API_TIMEOUT_SEC)
            response.raise_for_status()
            data = response.json()

            records = data.get('records')

            if not records:
                # No records returned means we're at the end of results for the current filters
                break

            all_records.extend([rec['fields'] for rec in records])

            # Update start for the next iteration
            current_start_row += len(records)

            # If the number of records returned is less than rows_to_fetch,
            # it means this was the last page of results.
            if len(records) < rows_to_fetch:
                break

            # Small pause to be polite to the API
            time.sleep(0.05)

        except requests.exceptions.HTTPError as http_err:
            # Error messages are acceptable outside the main conditional loop logic
            st.error(f"Erreur HTTP de l'API pendant le chargement: {http_err}")
            st.error(f"URL de la requête ayant échoué: {paginated_url}")
            try: error_detail = response.json(); st.error(f"Détails de l'erreur JSON: {error_detail}")
            except: pass
            return pd.DataFrame() # Return empty DataFrame on error
        except requests.exceptions.RequestException as e: st.error(f"Erreur de requête API pendant le chargement: {e}"); return pd.DataFrame()
        except KeyError as e: st.error(f"Erreur de structure JSON de l'API pendant le chargement: clé manquante {e}"); return pd.DataFrame()
        except requests.exceptions.Timeout: st.error("Délai d'attente dépassé lors de la requête à l'API."); return pd.DataFrame()
        except Exception as e: st.error(f"Une erreur inattendue est survenue pendant le chargement: {e}"); return pd.DataFrame()

    # End of while loop. No st.toast inside the cache function.

    if not all_records:
        # Message for empty results can be displayed in main() based on the returned df
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # --- Data Cleaning and Preparation (remains the same) ---
    if 'date_publication' in df.columns:
        df['date_publication'] = pd.to_datetime(df['date_publication'], errors='coerce').dt.date
        df = df.dropna(subset=['date_publication'])
        if not df.empty:
             df = df.sort_values(by='date_publication', ascending=False).reset_index(drop=True)
        else:
             # If all dates are invalid, return empty DataFrame to prevent further errors
             st.warning("Toutes les dates de publication sont invalides après conversion. Aucun rappel valide n'a été chargé.", icon="⚠️")
             return pd.DataFrame() # Return empty DataFrame if no valid dates
    else:
        st.warning("La colonne 'date_publication' est manquante dans les données de l'API. Le tri et certains filtres pourraient ne pas fonctionner.", icon="⚠️")
        df['date_publication'] = pd.NaT # Assign NaT if column is missing


    for api_col_name in FRIENDLY_TO_API_COLUMN_MAPPING.values():
        if api_col_name not in df.columns:
            df[api_col_name] = pd.NA
        else:
             if df[api_col_name].apply(lambda x: isinstance(x, list) and not x).any():
                  df[api_col_name] = df[api_col_name].apply(lambda x: pd.NA if isinstance(x, list) and not x else x)
             if df[api_col_name].isnull().any():
                  df[api_col_name] = df[api_col_name].replace({None: pd.NA})

    return df


# --- Fonctions de filtrage (inchangée) ---
def filter_data(data_df, selected_subcategories, selected_risks, search_term, selected_dates_tuple, selected_categories, search_column_api_name=None):
    filtered_df = data_df.copy()

    if 'categorie_de_produit' in filtered_df.columns and selected_categories:
        filtered_df = filtered_df[filtered_df['categorie_de_produit'].isin(selected_categories)]
    if 'sous_categorie_de_produit' in filtered_df.columns and selected_subcategories:
        filtered_df = filtered_df[filtered_df['sous_categorie_de_produit'].isin(selected_subcategories)]
    if 'risques_encourus' in filtered_df.columns and selected_risks:
        filtered_df = filtered_df[filtered_df['risques_encourus'].fillna('').isin(selected_risks)]

    if search_term:
        search_term_lower = search_term.lower()
        if search_column_api_name and search_column_api_name in filtered_df.columns:
            col_as_str = filtered_df[search_column_api_name].fillna('').astype(str)
            filtered_df = filtered_df[col_as_str.str.lower().str.contains(search_term_lower)]
        else:
            cols_to_search_api = [
                'nom_de_la_marque_du_produit', 'nom_commercial', 'modeles_ou_references',
                'risques_encourus', 'motif_du_rappel', 'sous_categorie_de_produit',
                'distributeurs'
            ]
            cols_to_search_api = [col for col in cols_to_search_api if col in filtered_df.columns]

            if cols_to_search_api:
                mask = filtered_df[cols_to_search_api].fillna('').astype(str).apply(
                    lambda x: x.str.lower().str.contains(search_term_lower)
                ).any(axis=1)
                filtered_df = filtered_df[mask]

    if 'date_publication' in filtered_df.columns and not filtered_df.empty and pd.notna(filtered_df['date_publication'].iloc[0]):
        start_filter_date = selected_dates_tuple[0]
        end_filter_date = selected_dates_tuple[1]

        if isinstance(start_filter_date, datetime): start_filter_date = start_filter_date.date()
        if isinstance(end_filter_date, datetime): end_filter_date = end_filter_date.date()

        if not filtered_df['date_publication'].dropna().empty and isinstance(filtered_df['date_publication'].dropna().iloc[0], date):
            try:
                filtered_df = filtered_df[
                    (filtered_df['date_publication'] >= start_filter_date) &
                    (filtered_df['date_publication'] <= end_filter_date)
                ]
            except TypeError as te:
                st.warning(f"Erreur de type lors du filtrage par date Python. Assurez-vous que la colonne 'date_publication' contient des objets date: {te}.")
            except Exception as e:
                 st.warning(f"Erreur lors du filtrage par date Python: {e}")

    return filtered_df


# --- Fonctions UI (Header, Métriques, Cartes Rappel, Pagination, Filtres Avancés) (inchangées) ---
def create_header():
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">RappelConso Insight 🔍</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("Votre assistant IA pour la surveillance et l'analyse des alertes alimentaires en France.")

def display_metrics_cards(data_df):
    if data_df.empty:
        st.info("Aucune donnée à afficher pour les métriques avec les filtres actuels.")
        return

    total_recalls = len(data_df)
    unique_subcategories = data_df['sous_categorie_de_produit'].dropna().nunique() if 'sous_categorie_de_produit' in data_df.columns else 0

    today = date.today()
    recent_days_filter = st.session_state.get('recent_days_filter', DEFAULT_RECENT_DAYS)
    days_ago = today - timedelta(days=recent_days_filter)

    recent_recalls = 0
    if 'date_publication' in data_df.columns and not data_df['date_publication'].dropna().empty:
        valid_dates = data_df['date_publication'].dropna()
        if not valid_dates.empty and isinstance(valid_dates.iloc[0], date):
             recent_recalls = len(valid_dates[valid_dates >= days_ago])

    severe_percent = 0
    if 'risques_encourus' in data_df.columns and not data_df.empty:
        grave_keywords = ['microbiologique', 'listeria', 'salmonelle', 'allerg', 'toxique', 'e. coli', 'corps étranger', 'chimique']
        search_series = data_df['risques_encourus'].fillna('').astype(str).str.lower()
        severe_risks_mask = search_series.str.contains('|'.join(grave_keywords), na=False)
        severe_risks_count = severe_risks_mask.sum()
        severe_percent = int((severe_risks_count / total_recalls) * 100) if total_recalls > 0 else 0

    cols = st.columns(4)
    metrics_data = [
        ("Total des Rappels", total_recalls),
        (f"Rappels Récents ({recent_days_filter}j)", recent_recalls),
        ("Sous-Catégories Uniques", unique_subcategories),
        ("Part de Risques Notables*", f"{severe_percent}%")
    ]

    for i, (label, value) in enumerate(metrics_data):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{value}</p>
                <p class="metric-label">{label}</p>
            </div>
            """, unsafe_allow_html=True)
    st.caption("*Risques notables incluent (non exhaustif): microbiologique, listeria, salmonelle, allergène/allergie, toxique, E. coli, corps étranger, risque chimique.")

def display_recall_card(row_data):
    with st.container():
        st.markdown('<div class="recall-card-container">', unsafe_allow_html=True)
        col_img, col_content = st.columns([1, 3])

        with col_img:
            image_url = row_data.get('liens_vers_images')
            if isinstance(image_url, list) and image_url:
                 image_url = image_url[0]
            elif isinstance(image_url, str) and image_url:
                image_url = image_url.split('|')[0]
            else:
                image_url = "https://via.placeholder.com/150/CCCCCC/FFFFFF?Text=Image+ND"
            st.image(image_url, width=130)

        with col_content:
            product_name = row_data.get('nom_commercial', row_data.get('modeles_ou_references', 'Produit non spécifié'))
            if pd.isna(product_name) or product_name == '': product_name = row_data.get('modeles_ou_references', 'Produit non spécifié')
            if pd.isna(product_name) or product_name == '': product_name = 'Produit non spécifié'

            st.markdown(f"<h5>{product_name}</h5>", unsafe_allow_html=True)

            pub_date_obj = row_data.get('date_publication')
            formatted_date = pub_date_obj.strftime('%d/%m/%Y') if isinstance(pub_date_obj, date) else 'Date inconnue'
            st.caption(f"Publié le: {formatted_date}")

            risk_text_raw = row_data.get('risques_encourus', 'Risque non spécifié')
            risk_text_lower = str(risk_text_raw).lower()

            badge_color = "grey"; badge_icon = "⚠️"
            if any(keyword in risk_text_lower for keyword in ['listeria', 'salmonelle', 'e. coli', 'danger immédiat', 'toxique']):
                badge_color = "red"; badge_icon = "☠️"
            elif any(keyword in risk_text_lower for keyword in ['allergène', 'allergie', 'microbiologique', 'corps étranger', 'chimique']):
                badge_color = "orange"; badge_icon = "🔬"
            elif risk_text_raw and risk_text_raw != 'Risque non spécifié':
                badge_color = "darkgoldenrod"; badge_icon = "❗"


            st.markdown(f"**Risque {badge_icon}:** <span style='color:{badge_color}; font-weight:bold;'>{risk_text_raw if pd.notna(risk_text_raw) else 'Non spécifié'}</span>", unsafe_allow_html=True)
            st.markdown(f"**Marque:** {row_data.get('nom_de_la_marque_du_produit', 'N/A')}")

            motif = row_data.get('motif_du_rappel')
            motif_display = 'N/A' if pd.isna(motif) else str(motif)
            if len(motif_display) > 100:
                motif_display = motif_display[:97] + "..."
            st.markdown(f"**Motif:** {motif_display}")

            distributeurs = row_data.get('distributeurs')
            distributeurs_display = 'N/A' if pd.isna(distributeurs) else str(distributeurs)
            if distributeurs_display != 'N/A':
                 if len(distributeurs_display) > 70: distributeurs_display = distributeurs_display[:67] + "..."
                 st.markdown(f"**Distributeurs:** {distributeurs_display}")

            pdf_link = row_data.get('liens_vers_la_fiche_rappel')
            if pdf_link and pd.notna(pdf_link) and pdf_link != '#':
                 st.link_button("📄 Fiche de rappel", pdf_link, type="secondary", help="Ouvrir la fiche de rappel officielle")

        st.markdown('</div>', unsafe_allow_html=True)


def display_paginated_recalls(data_df, items_per_page_setting):
    if data_df.empty:
        st.info("Aucun rappel ne correspond à vos critères de recherche.")
        st.session_state.current_page_recalls = 1
        return

    st.markdown(f"#### Affichage des rappels ({len(data_df)} résultats)")

    if 'current_page_recalls' not in st.session_state:
        st.session_state.current_page_recalls = 1

    total_pages = (len(data_df) - 1) // items_per_page_setting + 1
    if st.session_state.current_page_recalls > total_pages:
        st.session_state.current_page_recalls = max(1, total_pages)

    current_page = st.session_state.current_page_recalls

    start_idx = (current_page - 1) * items_per_page_setting
    end_idx = min(start_idx + items_per_page_setting, len(data_df))

    current_recalls_page_df = data_df.iloc[start_idx:end_idx]

    for i in range(0, len(current_recalls_page_df), 2):
        cols = st.columns(2)
        with cols[0]:
            if i < len(current_recalls_page_df):
                display_recall_card(current_recalls_page_df.iloc[i])
        with cols[1]:
            if i + 1 < len(current_recalls_page_df):
                display_recall_card(current_recalls_page_df.iloc[i + 1])

    st.markdown("---")

    if total_pages > 1:
        cols_pagination = st.columns([1, 2, 1])

        with cols_pagination[0]:
            if st.button("← Précédent", disabled=(current_page == 1), use_container_width=True, key="prev_page_btn"):
                st.session_state.current_page_recalls -= 1
                st.experimental_rerun()

        with cols_pagination[1]:
            st.markdown(f"<div style='text-align: center; margin-top: 10px;'>Page {current_page} sur {total_pages}</div>", unsafe_allow_html=True)

        with cols_pagination[2]:
            if st.button("Suivant →", disabled=(current_page == total_pages), use_container_width=True, key="next_page_btn"):
                st.session_state.current_page_recalls += 1
                st.experimental_rerun()


def create_advanced_filters(df_full_data):
    min_date_data = df_full_data['date_publication'].dropna().min() if 'date_publication' in df_full_data.columns and not df_full_data['date_publication'].dropna().empty else START_DATE
    if isinstance(min_date_data, datetime): min_date_data = min_date_data.date()
    elif not isinstance(min_date_data, date): min_date_data = START_DATE

    max_date_data = date.today()

    if 'date_filter_start' not in st.session_state or not isinstance(st.session_state.date_filter_start, date):
        st.session_state.date_filter_start = min_date_data
    if 'date_filter_end' not in st.session_state or not isinstance(st.session_state.date_filter_end, date):
        st.session_state.date_filter_end = max_date_data

    if isinstance(st.session_state.date_filter_start, datetime): st.session_state.date_filter_start = st.session_state.date_filter_start.date()
    if isinstance(st.session_state.date_filter_end, datetime): st.session_state.date_filter_end = st.session_state.date_filter_end.date()


    with st.expander("🔍 Filtres avancés et Options d'affichage", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            if 'categorie_de_produit' in df_full_data.columns:
                unique_main_categories = sorted(df_full_data['categorie_de_produit'].dropna().unique())
                selected_categories = st.multiselect(
                    "Filtrer par Catégories principales:", options=unique_main_categories,
                    default=st.session_state.get('selected_categories_filter', []),
                    key="main_cat_filter"
                )
                st.session_state.selected_categories_filter = selected_categories
            else: selected_categories = []

            if 'sous_categorie_de_produit' in df_full_data.columns:
                df_for_subcats = df_full_data[df_full_data['categorie_de_produit'].isin(selected_categories)] if selected_categories else df_full_data
                unique_subcategories = sorted(df_for_subcats['sous_categorie_de_produit'].dropna().unique())
                selected_subcategories = st.multiselect(
                    "Filtrer par Sous-catégories:", options=unique_subcategories,
                    default=st.session_state.get('selected_subcategories_filter', []),
                    key="sub_cat_filter"
                )
                st.session_state.selected_subcategories_filter = selected_subcategories
            else: selected_subcategories = []

        with col2:
            if 'risques_encourus' in df_full_data.columns:
                unique_risks = sorted(df_full_data['risques_encourus'].dropna().unique())
                display_risks = unique_risks
                if len(unique_risks) > 75:
                    top_risks_counts = df_full_data['risques_encourus'].value_counts().nlargest(75).index.tolist()
                    display_risks = sorted(top_risks_counts)
                    st.caption(f"Affichage des 75 risques les plus fréquents (sur {len(unique_risks)}).")

                selected_risks = st.multiselect(
                    "Filtrer par Types de risques:", options=display_risks,
                    default=st.session_state.get('selected_risks_filter', []),
                    key="risks_filter"
                )
                st.session_state.selected_risks_filter = selected_risks
            else: selected_risks = []

            selected_dates_tuple_local = st.date_input(
                "Filtrer par période de publication:",
                value=(st.session_state.date_filter_start, st.session_state.date_filter_end),
                min_value=min_date_data, max_value=max_date_data,
                key="date_range_picker_adv"
            )
            if len(selected_dates_tuple_local) == 2:
                if selected_dates_tuple_local[0] != st.session_state.date_filter_start or selected_dates_tuple_local[1] != st.session_state.date_filter_end:
                    st.session_state.date_filter_start, st.session_state.date_filter_end = selected_dates_tuple_local
                    st.session_state.current_page_recalls = 1
                    st.experimental_rerun()


        st.markdown("---")
        st.markdown("**Options d'affichage**")
        items_per_page_setting = st.slider("Nombre de rappels par page:", min_value=2, max_value=20,
                                 value=st.session_state.get('items_per_page_filter', DEFAULT_ITEMS_PER_PAGE), step=2,
                                 key="items_page_slider")
        if items_per_page_setting != st.session_state.get('items_per_page_filter', DEFAULT_ITEMS_PER_PAGE):
            st.session_state.items_per_page_filter = items_per_page_setting
            st.session_state.current_page_recalls = 1 # Réinitialiser la page si le nombre par page change

        recent_days_setting = st.slider("Période pour 'Rappels Récents' (jours):", min_value=7, max_value=90,
                               value=st.session_state.get('recent_days_filter', DEFAULT_RECENT_DAYS), step=1,
                               key="recent_days_slider")
        st.session_state.recent_days_filter = recent_days_setting

        if st.button("Réinitialiser filtres et options", type="secondary", use_container_width=True, key="reset_filters_btn"):
            keys_to_reset = [
                'selected_categories_filter', 'selected_subcategories_filter',
                'selected_risks_filter', 'search_term_main', 'search_column_friendly_name_select',
                'date_filter_start', 'date_filter_end',
                'items_per_page_filter', 'recent_days_filter', 'current_page_recalls'
            ]
            for key_to_del in keys_to_reset:
                if key_to_del in st.session_state: del st.session_state[key_to_del]

            min_date_data_reset = df_full_data['date_publication'].dropna().min() if 'date_publication' in df_full_data.columns and not df_full_data['date_publication'].dropna().empty else START_DATE
            if isinstance(min_date_data_reset, datetime): min_date_data_reset = min_date_data_reset.date()
            elif not isinstance(min_date_data_reset, date): min_date_data_reset = START_DATE

            st.session_state.date_filter_start = min_date_data_reset
            st.session_state.date_filter_end = date.today()
            st.session_state.items_per_page_filter = DEFAULT_ITEMS_PER_PAGE
            st.session_state.recent_days_filter = DEFAULT_RECENT_DAYS
            st.session_state.current_page_recalls = 1
            st.session_state.search_term_main = ""
            st.session_state.search_column_friendly_name_select = "Toutes les colonnes pertinentes"
            st.experimental_rerun()


    active_filters_display = []
    if st.session_state.get('search_term_main', ""):
        search_col_friendly = st.session_state.get('search_column_friendly_name_select', "Toutes les colonnes pertinentes")
        active_filters_display.append(f"Recherche: \"{st.session_state.search_term_main}\" (dans {search_col_friendly})")
    if st.session_state.get('selected_categories_filter'):
        active_filters_display.append(f"Catégories: {', '.join(st.session_state['selected_categories_filter'])}")
    if st.session_state.get('selected_subcategories_filter'):
        active_filters_display.append(f"Sous-catégories: {', '.join(st.session_state['selected_subcategories_filter'])}")
    if st.session_state.get('selected_risks_filter'):
        active_filters_display.append(f"Risques: {', '.join(st.session_state['selected_risks_filter'])}")

    current_start_date_display = st.session_state.get('date_filter_start', START_DATE)
    current_end_date_display = st.session_state.get('date_filter_end', date.today())
    min_data_date_actual = df_full_data['date_publication'].dropna().min() if 'date_publication' in df_full_data.columns and not df_full_data['date_publication'].dropna().empty else START_DATE
    if isinstance(min_data_date_actual, datetime): min_data_date_actual = min_data_date_actual.date()
    elif not isinstance(min_data_date_actual, date): min_data_date_actual = START_DATE

    if current_start_date_display != min_data_date_actual or current_end_date_display != date.today():
         active_filters_display.append(f"Période: {current_start_date_display.strftime('%d/%m/%y')} - {current_end_date_display.strftime('%d/%m/%y')}")


    if active_filters_display:
        st.markdown('<div class="filter-pills-container"><div class="filter-pills-title">Filtres actifs :</div><div class="filter-pills">' +
                    ' '.join([f'<span class="filter-pill">{f}</span>' for f in active_filters_display]) +
                    '</div></div>', unsafe_allow_html=True)

    return st.session_state.get('selected_categories_filter', []), \
           st.session_state.get('selected_subcategories_filter', []), \
           st.session_state.get('selected_risks_filter', []), \
           (st.session_state.date_filter_start, st.session_state.date_filter_end), \
           st.session_state.get('items_per_page_filter', DEFAULT_ITEMS_PER_PAGE)


# --- Fonctions de Visualisation (inchangée) ---
def create_improved_visualizations(data_df_viz):
    if data_df_viz.empty:
        st.info("Données insuffisantes pour générer des visualisations avec les filtres actuels.")
        return

    st.markdown('<div class="chart-container" style="margin-top:1rem;">', unsafe_allow_html=True)

    if 'date_publication' in data_df_viz.columns and not data_df_viz['date_publication'].dropna().empty:
        try:
            data_df_viz['date_publication_dt'] = pd.to_datetime(data_df_viz['date_publication'], errors='coerce')
            data_df_viz = data_df_viz.dropna(subset=['date_publication_dt'])
            if data_df_viz.empty:
                st.warning("Dates invalides dans les données après conversion, impossible de générer les visualisations temporelles.")
                st.markdown('</div>', unsafe_allow_html=True)
                return
        except Exception as e:
            st.warning(f"Erreur lors de la conversion des dates pour les visualisations : {e}")
            st.markdown('</div>', unsafe_allow_html=True)
            return
    else:
        st.warning("Colonne 'date_publication' manquante ou vide pour les visualisations temporelles.")
        st.markdown('</div>', unsafe_allow_html=True)
        return


    tab1, tab2, tab3 = st.tabs(["📊 Tendances Temporelles", "🍩 Répartitions par Catégorie", "☠️ Analyse des Risques"])

    with tab1:
        st.subheader("Évolution des Rappels")
        if 'date_publication_dt' in data_df_viz.columns and not data_df_viz['date_publication_dt'].empty:
            monthly_data = data_df_viz.set_index('date_publication_dt').resample('MS').size().reset_index(name='count')
            monthly_data['year_month'] = monthly_data['date_publication_dt'].dt.strftime('%Y-%m')
            monthly_data = monthly_data.sort_values('date_publication_dt')

            if not monthly_data.empty:
                fig_temporal = px.line(monthly_data, x='year_month', y='count', title="Nombre de rappels par mois", markers=True,
                                       labels={'year_month': 'Mois', 'count': 'Nombre de rappels'})
                fig_temporal.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_temporal, use_container_width=True)
            else: st.info("Pas de données mensuelles à afficher pour les tendances.")
        else: st.warning("Impossible de générer le graphique temporel.")

    with tab2:
        st.subheader("Répartition des Rappels")
        col_cat, col_subcat = st.columns(2)
        with col_cat:
            if 'categorie_de_produit' in data_df_viz.columns:
                cat_counts = data_df_viz['categorie_de_produit'].dropna().value_counts().nlargest(10)
                if not cat_counts.empty:
                    fig_cat = px.pie(values=cat_counts.values, names=cat_counts.index, title="Top 10 Catégories", hole=0.4)
                    fig_cat.update_traces(textposition='outside', textinfo='percent+label')
                    st.plotly_chart(fig_cat, use_container_width=True)
                else: st.info("Pas de données pour la répartition par catégorie.")
            else: st.warning("Colonne 'categorie_de_produit' manquante.")
        with col_subcat:
            if 'sous_categorie_de_produit' in data_df_viz.columns:
                subcat_counts = data_df_viz['sous_categorie_de_produit'].dropna().value_counts().nlargest(10)
                if not subcat_counts.empty:
                    fig_subcat = px.pie(values=subcat_counts.values, names=subcat_counts.index, title="Top 10 Sous-Catégories", hole=0.4)
                    fig_subcat.update_traces(textposition='outside', textinfo='percent+label')
                    st.plotly_chart(fig_subcat, use_container_width=True)
                else: st.info("Pas de données pour la répartition par sous-catégorie.")
            else: st.warning("Colonne 'sous_categorie_de_produit' manquante.")

        if 'nom_de_la_marque_du_produit' in data_df_viz.columns:
            valid_brands = data_df_viz['nom_de_la_marque_du_produit'].dropna()
            valid_brands = valid_brands[valid_brands != '']
            if not valid_brands.empty:
                brand_counts = valid_brands.value_counts().nlargest(10)
                if not brand_counts.empty:
                    fig_brands = px.bar(x=brand_counts.index, y=brand_counts.values,
                                        title="Top 10 Marques (nombre de rappels)",
                                        labels={'x': 'Marque', 'y': 'Nombre de rappels'},
                                        color=brand_counts.values, color_continuous_scale=px.colors.sequential.Blues_r)
                    fig_brands.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False)
                    st.plotly_chart(fig_brands, use_container_width=True)
                else: st.info("Pas assez de données de marque pour un top 10.")
            else: st.info("Pas de données de marque valides.")


    with tab3:
        st.subheader("Analyse des Risques et Motifs")
        if 'risques_encourus' in data_df_viz.columns:
            valid_risks = data_df_viz['risques_encourus'].dropna()
            valid_risks = valid_risks[valid_risks != '']
            if not valid_risks.empty:
                risk_counts = valid_risks.value_counts().nlargest(10)
                if not risk_counts.empty:
                    fig_risks = px.bar(y=risk_counts.index, x=risk_counts.values, orientation='h',
                                       title="Top 10 Risques Encourus",
                                       labels={'y': 'Risque', 'x': 'Nombre de rappels'},
                                       color=risk_counts.values, color_continuous_scale='Reds')
                    fig_risks.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False, height=max(400, len(risk_counts)*40))
                    st.plotly_chart(fig_risks, use_container_width=True)
                else: st.info("Pas assez de données de risque pour un top 10.")
            else: st.info("Pas de données de risque valides.")
        else: st.warning("Colonne 'risques_encourus' manquante.")

        if 'motif_du_rappel' in data_df_viz.columns:
            valid_motifs = data_df_viz['motif_du_rappel'].dropna()
            valid_motifs = valid_motifs[valid_motifs != '']
            if not valid_motifs.empty:
                motif_counts = valid_motifs.value_counts().nlargest(10)
                if not motif_counts.empty:
                    fig_motifs = px.bar(y=motif_counts.index, x=motif_counts.values, orientation='h',
                                        title="Top 10 Motifs de Rappel",
                                        labels={'y': 'Motif', 'x': 'Nombre de rappels'},
                                        color=motif_counts.values, color_continuous_scale='Oranges')
                    fig_motifs.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False, height=max(400, len(motif_counts)*40))
                    st.plotly_chart(fig_motifs, use_container_width=True)
                else: st.info("Pas assez de données de motif pour un top 10.")
            else: st.info("Pas de données de motif valides.")
        else: st.warning("Colonne 'motif_du_rappel' manquante.")

    st.markdown('</div>', unsafe_allow_html=True)


# --- Fonctions pour l'IA avec Groq ---
def manage_groq_api_key():
    """Gère l'input de la clé API Groq dans la sidebar et retourne si l'IA est prête."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 Assistant IA Groq")

    # L'expander reste ouvert par défaut tant qu'il n'y a pas de clé valide en session
    default_expanded = True
    if "user_groq_api_key" in st.session_state and st.session_state.user_groq_api_key and st.session_state.user_groq_api_key.startswith("gsk_"):
        default_expanded = False

    with st.sidebar.expander("Configurer l'accès à l'IA", expanded=default_expanded):
        # Initialisation de la clé en session state si elle n'existe pas
        if "user_groq_api_key" not in st.session_state:
            st.session_state.user_groq_api_key = ""

        # Champ de saisie pour la clé API
        new_key = st.text_input(
            "Votre clé API Groq:", type="password",
            value=st.session_state.user_groq_api_key, # Utilisez la valeur en session comme valeur par défaut
            help="Obtenez votre clé sur [console.groq.com](https://console.groq.com/keys). La clé est stockée temporairement en session.",
            key="groq_api_key_input_sidebar"
        )

        # Si la valeur saisie dans le champ change, mettez à jour la session state
        # Cette condition est importante pour ne pas relancer reruns en boucle
        if new_key != st.session_state.user_groq_api_key:
             st.session_state.user_groq_api_key = new_key
             # Afficher un message utilisateur lors de la saisie/modification
             if new_key:
                 if new_key.startswith("gsk_"):
                    st.success("Clé API Groq enregistrée.", icon="👍")
                    # Supprimer l'éventuelle erreur d'initialisation précédente
                    if 'groq_client_error' in st.session_state: del st.session_state.groq_client_error
                 else:
                    st.warning("Format de clé API invalide. Doit commencer par 'gsk_'.", icon="⚠️")
             else:
                 st.info("Aucune clé API Groq n'est configurée.", icon="ℹ️")
             # Déclencher un rerun pour que l'état de la clé soit mis à jour et le client Groq initialisé
             st.experimental_rerun()

        # Vérifier si une clé valide est présente en session pour activer le reste des options IA
        model_disabled = not (st.session_state.user_groq_api_key and st.session_state.user_groq_api_key.startswith("gsk_"))

        # Sélection du modèle IA
        model_options = {
            "llama3-70b-8192": "Llama 3 (70B) - Puissant", "llama3-8b-8192": "Llama 3 (8B) - Rapide",
            "mixtral-8x7b-32768": "Mixtral (8x7B) - Large contexte", "gemma-7b-it": "Gemma (7B) - Léger"
        }
        # Trouver l'index du modèle sélectionné pour maintenir la valeur par défaut correcte
        current_model = st.session_state.get('groq_model', 'llama3-8b-8192')
        model_index = list(model_options.keys()).index(current_model) if current_model in model_options else 0

        selected_model_key = st.selectbox(
            "Choisir un modèle IA:", options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=model_index,
            key="groq_model_select_sidebar",
            disabled=model_disabled # Désactivé si pas de clé valide
        )
        # Mettre à jour la session state uniquement si le sélecteur n'est pas désactivé
        if not model_disabled:
             st.session_state.groq_model = selected_model_key

        # Options avancées IA (température, tokens, contexte)
        with st.popover("Options avancées de l'IA", disabled=model_disabled): # Désactivé si pas de clé valide
            st.session_state.groq_temperature = st.slider("Température:", 0.0, 1.0, st.session_state.get('groq_temperature', 0.2), 0.1, disabled=model_disabled)
            st.session_state.groq_max_tokens = st.slider("Tokens max réponse:", 256, 4096, st.session_state.get('groq_max_tokens', 1024), 256, disabled=model_disabled)
            st.session_state.groq_max_context_recalls = st.slider("Max rappels dans contexte IA:", 5, 50, st.session_state.get('groq_max_context_recalls', 15), 1, disabled=model_disabled)


    # Afficher l'état de la connexion IA
    if st.session_state.user_groq_api_key and st.session_state.user_groq_api_key.startswith("gsk_"):
        # Tenter d'obtenir le client pour vérifier l'initialisation
        client_status = get_groq_client()
        if client_status is not None:
             st.sidebar.caption(f"🟢 IA prête ({model_options.get(st.session_state.groq_model, 'N/A')})")
             return True
        else:
             # Si get_groq_client a retourné None, il y a une erreur d'initialisation
             st.sidebar.caption(f"❌ IA non prête. Erreur client.")
             # Le détail de l'erreur est stocké dans st.session_state.groq_client_error
             if "groq_client_error" in st.session_state and st.session_state.groq_client_error:
                  st.sidebar.caption(f"Détails: {st.session_state.groq_client_error[:70]}...")
             return False # Retourner False si le client n'a pas pu être initialisé
    else:
        st.sidebar.caption("🔴 IA non configurée ou clé invalide.")
        return False


def get_groq_client():
    """Initialise et retourne le client Groq en utilisant la clé en session state."""
    api_key = st.session_state.get("user_groq_api_key")

    if api_key and isinstance(api_key, str) and api_key.startswith("gsk_"):
        try:
            # Tenter d'initialiser le client
            client = Groq(api_key=api_key)
            # Optionnel: faire un petit appel pour vérifier la clé (peut ralentir l'app)
            # client.models.list() # Exemple d'appel simple
            if 'groq_client_error' in st.session_state: del st.session_state.groq_client_error # Clear previous errors on success
            return client
        except Exception as e:
            # Capturer l'erreur d'initialisation et la stocker en session
            st.session_state.groq_client_error = str(e)
            # Ne pas afficher st.error ici pour ne pas interférer avec le cache
            return None # Retourner None si l'initialisation échoue
    else:
        # Si la clé n'est pas présente ou invalide, s'assurer qu'il n'y a pas d'erreur stockée
        if 'groq_client_error' in st.session_state: del st.session_state.groq_client_error
        return None


def prepare_context_for_ia(df_context, max_items=10):
    if df_context.empty:
        return "Aucune donnée de rappel pertinente trouvée pour cette question avec les filtres actuels."

    cols_for_ia = [
        'nom_de_la_marque_du_produit', 'nom_commercial', 'modeles_ou_references',
        'categorie_de_produit', 'sous_categorie_de_produit',
        'risques_encourus', 'motif_du_rappel', 'date_publication', 'distributeurs'
    ]
    cols_to_use = [col for col in cols_for_ia if col in df_context.columns]

    if cols_to_use:
        actual_max_items = min(max_items, len(df_context))
        context_df_sample = df_context[cols_to_use].head(actual_max_items).copy()

        for col in cols_to_use:
             context_df_sample[col] = context_df_sample[col].apply(lambda x: x if pd.notna(x) and x != '' and (not isinstance(x, list) or x) else None)


        text_context = f"Voici un échantillon de {len(context_df_sample)} rappels (sur {len(df_context)} au total avec les filtres) :\n\n"
        for _, row in context_df_sample.iterrows():
            item_desc = []
            if row.get('nom_de_la_marque_du_produit') is not None:
                item_desc.append(f"Marque: {row['nom_de_la_marque_du_produit']}")

            product_display_name = row.get('nom_commercial')
            if product_display_name is None or product_display_name == '':
                 product_display_name = row.get('modeles_ou_references')
            if product_display_name is not None and product_display_name != '':
                 item_desc.append(f"Produit: {product_display_name}")
            elif not item_desc:
                 item_desc.append("Produit non spécifié")


            if row.get('categorie_de_produit') is not None:
                item_desc.append(f"Cat: {row['categorie_de_produit']}")
            if row.get('sous_categorie_de_produit') is not None:
                item_desc.append(f"Sous-cat: {row['sous_categorie_de_produit']}")
            if row.get('risques_encourus') is not None:
                item_desc.append(f"Risque: {row['risques_encourus']}")
            if row.get('motif_du_rappel') is not None:
                motif_ctx = str(row['motif_du_rappel'])
                if len(motif_ctx) > 70: motif_ctx = motif_ctx[:67] + "..."
                item_desc.append(f"Motif: {motif_ctx}")
            if row.get('distributeurs') is not None:
                dist_ctx = str(row['distributeurs'])
                if len(dist_ctx) > 50 : dist_ctx = dist_ctx[:47] + "..."
                item_desc.append(f"Distrib: {dist_ctx}")

            if isinstance(row.get('date_publication'), date):
                date_pub_str = row['date_publication'].strftime('%d/%m/%y')
                item_desc.append(f"Date: {date_pub_str}")

            if item_desc:
                 text_context += "- " + ", ".join(item_desc) + "\n"

        return text_context
    else:
         return "Aucune colonne pertinente pour générer un contexte à partir des données disponibles."

def analyze_trends_data(df_analysis, product_type=None, risk_type=None, time_period="all"):
    if df_analysis.empty:
        return {"status": "no_data", "message": "Aucune donnée disponible pour l'analyse de tendance."}

    df_filtered = df_analysis.copy()
    if 'date_publication' in df_filtered.columns and not df_filtered['date_publication'].dropna().empty:
        try:
            df_filtered['date_publication_dt'] = pd.to_datetime(df_filtered['date_publication'], errors='coerce')
            df_filtered = df_filtered.dropna(subset=['date_publication_dt'])
            if df_filtered.empty: return {"status": "no_data", "message": "Dates invalides pour l'analyse après conversion."}
        except Exception as e:
             return {"status": "error", "message": f"Erreur lors de la conversion des dates pour l'analyse : {e}"}
    else: return {"status": "error", "message": "Colonne 'date_publication' manquante ou vide pour l'analyse de tendance."}

    analysis_title_parts = ["Évolution des rappels"]

    initial_filtered_count = len(df_filtered)
    if product_type:
        cols_product_search = ['sous_categorie_de_produit', 'nom_commercial', 'categorie_de_produit', 'nom_de_la_marque_du_produit']
        cols_product_search_exist = [col for col in cols_product_search if col in df_filtered.columns]
        if cols_product_search_exist:
            mask_product = df_filtered[cols_product_search_exist].fillna('').astype(str).apply(
                lambda x: x.str.contains(product_type, case=False, na=False)
            ).any(axis=1)
            df_filtered = df_filtered[mask_product]
            analysis_title_parts.append(f"pour '{product_type}'")
        else:
             analysis_title_parts.append(f"(filtre produit '{product_type}' non appliqué - colonnes manquantes)")


    if risk_type:
        cols_risk_search = ['risques_encourus', 'motif_du_rappel']
        cols_risk_search_exist = [col for col in cols_risk_search if col in df_filtered.columns]
        if cols_risk_search_exist:
            mask_risk = df_filtered[cols_risk_search_exist].fillna('').astype(str).apply(
                 lambda x: x.str.contains(risk_type, case=False, na=False)
            ).any(axis=1)
            df_filtered = df_filtered[mask_risk]
            analysis_title_parts.append(f"avec risque/motif '{risk_type}'")
        else:
             analysis_title_parts.append(f"(filtre risque '{risk_type}' non appliqué - colonnes manquantes)")

    if df_filtered.empty:
        return {"status": "no_data", "message": f"Aucune donnée correspondante trouvée après application des filtres spécifiques ({product_type or 'tous produits'}, {risk_type or 'tous risques'}). Total rappels filtrés initialement: {initial_filtered_count}."}

    monthly_counts = df_filtered.set_index('date_publication_dt').resample('MS').size()
    if monthly_counts.empty : return {"status": "no_data", "message": "Pas de données mensuelles à analyser après filtrage spécifique."}

    idx = pd.date_range(monthly_counts.index.min(), monthly_counts.index.max(), freq='MS')
    monthly_counts = monthly_counts.reindex(idx, fill_value=0)
    monthly_counts_display_index = monthly_counts.index.strftime('%Y-%m')
    monthly_counts_values = monthly_counts.values

    trend_stats = {"total_recalls": int(df_filtered.shape[0]), "monthly_avg": float(monthly_counts_values.mean())}
    slope = 0
    if len(monthly_counts_values) >= 2:
        X = np.arange(len(monthly_counts_values)).reshape(-1, 1)
        y = monthly_counts_values
        model = LinearRegression().fit(X, y)
        slope = float(model.coef_[0])
        trend_stats['trend_slope'] = slope
        if slope > 0.1: trend_stats['trend_direction'] = "hausse"
        elif slope < -0.1: trend_stats['trend_direction'] = "baisse"
        else: trend_stats['trend_direction'] = "stable"
    else: trend_stats['trend_direction'] = "indéterminée (données insuffisantes)"

    period_label = f"du {monthly_counts.index.min().strftime('%d/%m/%Y')} au {monthly_counts.index.max().strftime('%d/%m/%Y')}"
    analysis_title_parts.append(f"({period_label})")


    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly_counts_display_index, monthly_counts_values, marker='o', linestyle='-', label='Rappels/mois')
    if len(monthly_counts_values) >= 2 and 'trend_direction' in trend_stats and trend_stats['trend_direction'] != "indéterminée (données insuffisantes)":
        trend_line = model.predict(X)
        ax.plot(monthly_counts_display_index, trend_line, color='red', linestyle='--', label=f'Tendance ({trend_stats["trend_direction"]})')

    ax.set_title(' '.join(analysis_title_parts), fontsize=10)
    ax.set_xlabel("Mois", fontsize=8); ax.set_ylabel("Nombre de rappels", fontsize=8)
    if len(monthly_counts_display_index) > 15:
        tick_indices = np.linspace(0, len(monthly_counts_display_index) - 1, 15, dtype=int)
        ax.set_xticks(np.array(monthly_counts_display_index)[tick_indices])
    else:
        ax.set_xticks(monthly_counts_display_index)
    ax.tick_params(axis='x', rotation=45, labelsize=7); ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, linestyle=':', alpha=0.7); ax.legend(fontsize=8)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png"); buf.seek(0)
    graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    text_summary = f"Analyse de tendance ({period_label}, Rappels analysés: {trend_stats['total_recalls']}):\n"
    text_summary += f"- Moyenne mensuelle: {trend_stats['monthly_avg']:.1f} rappels.\n"
    if 'trend_direction' in trend_stats:
        text_summary += f"- Tendance générale: {trend_stats['trend_direction']}"
        if trend_stats['trend_direction'] not in ["indéterminée (données insuffisantes)", "stable"]:
             text_summary += f" (pente: {slope:.2f}).\n"
        else:
            text_summary += "\n"

    if 'sous_categorie_de_produit' in df_filtered.columns:
        top_cat = df_filtered['sous_categorie_de_produit'].dropna().value_counts().nlargest(3)
        if not top_cat.empty: text_summary += "- Top 3 sous-catégories: " + ", ".join([f"{idx} ({val})" for idx, val in top_cat.items()]) + ".\n"
    if 'risques_encourus' in df_filtered.columns:
        top_risk = df_filtered['risques_encourus'].dropna().value_counts().nlargest(3)
        if not top_risk.empty: text_summary += "- Top 3 risques: " + ", ".join([f"{idx} ({val})" for idx, val in top_risk.items()]) + ".\n"

    return {
        "status": "success", "text_summary": text_summary, "graph_base64": graph_base64,
        "monthly_data": {m:c for m,c in zip(monthly_counts_display_index.to_list(), monthly_counts_values.to_list())},
        "trend_stats": trend_stats
    }


def ask_groq_ai(client, user_query, context_data_text, trend_analysis_results=None):
    if not client:
        return "Client Groq non initialisé. Veuillez vérifier votre clé API et la configuration."

    system_prompt = f"""Tu es "RappelConso Insight Assistant", un expert IA spécialisé dans l'analyse des données de rappels de produits alimentaires en France, basé sur les données de RappelConso.
    Date actuelle: {date.today().strftime('%d/%m/%Y')}.
    Ton rôle est d'aider l'utilisateur à comprendre les données de rappels filtrées affichées dans l'application.
    Réponds aux questions de manière concise, professionnelle et en te basant STRICTEMENT sur les informations et données de contexte fournies (échantillon des rappels filtrés et résumé d'analyse de tendance si disponible).
    NE PAS INVENTER d'informations. Si les données de contexte ne te permettent pas de répondre, indique-le clairement (ex: "Je n'ai pas suffisamment d'informations dans les données fournies pour répondre précisément à cette question.").
    Utilise le Markdown pour mettre en **gras** les chiffres clés, les noms de catégories/risques importants et les points essentiels. Utilise des listes à puces si pertinent.
    Si une analyse de tendance a été effectuée et qu'un graphique est disponible (indiqué dans le contexte et les résultats passés), mentionne-le et décris brièvement ce qu'il montre en t'appuyant sur le résumé d'analyse fourni. Précise sur quelle période l'analyse porte (celle des données filtrées).
    Si la question est complètement hors sujet (ne concerne pas les rappels de produits alimentaires, la sécurité alimentaire, ou les données fournies), réponds avec une blague COURTE et pertinente sur la sécurité alimentaire ou la nourriture, puis indique que tu ne peux pas répondre à la question car elle sort de ton domaine d'expertise sur RappelConso.
    Exemple de blague : "Pourquoi le pain a-t-il appelé la police ? Parce qu'il s'est fait émietter ! 🍞 Plus sérieusement, ma connaissance se limite aux données de RappelConso."
    Sois toujours courtois et utile dans le cadre des données disponibles.
    """

    full_context_for_ai = f"Contexte des rappels de produits (échantillon des rappels *actuellement filtrés* par l'utilisateur et disponibles pour analyse):\n{context_data_text}\n\n"

    if trend_analysis_results and trend_analysis_results["status"] == "success":
        full_context_for_ai += f"Une analyse de tendance sur les données filtrées a été effectuée. Voici son résumé:\n{trend_analysis_results['text_summary']}\n"
        full_context_for_ai += "Un graphique illustrant cette tendance est disponible et sera affiché à l'utilisateur si pertinent.\n"
    elif trend_analysis_results and trend_analysis_results["status"] == "no_data":
         full_context_for_ai += f"Une tentative d'analyse de tendance a été faite, mais : {trend_analysis_results['message']}\n"
    elif trend_analysis_results and trend_analysis_results["status"] == "error":
         full_context_for_ai += f"Une tentative d'analyse de tendance a échoué : {trend_analysis_results['message']}\n"


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{full_context_for_ai}\nQuestion de l'utilisateur: \"{user_query}\"\n\nRéponse (en Markdown):"}
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages, model=st.session_state.get("groq_model", "llama3-8b-8192"),
            temperature=st.session_state.get("groq_temperature", 0.2),
            max_tokens=st.session_state.get("groq_max_tokens", 1024),
        )
        response_content = chat_completion.choices[0].message.content
        return response_content
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API Groq: {e}")
        error_message_lower = str(e).lower()
        if "authentication" in error_message_lower or "api key" in error_message_lower or "invalid api key" in error_message_lower:
            return "Erreur d'authentification avec l'API Groq. Vérifiez votre clé API dans la barre latérale."
        elif "rate limit" in error_message_lower:
             return "La limite de requêtes API Groq a été atteinte. Veuillez patienter avant de poser une autre question."
        elif "bad gateway" in error_message_lower or "service unavailable" in error_message_lower:
             return "L'API Groq rencontre actuellement des problèmes. Veuillez réessayer plus tard."
        return "Désolé, une erreur technique m'empêche de traiter votre demande IA."


# --- Main App Logic ---
def main():
    create_header()

    st.sidebar.image(LOGO_URL, use_container_width=True)
    st.sidebar.title("Navigation & Options")

    # Gestion de la clé API Groq et obtention du client
    # Cette fonction demande la clé via text_input et la stocke en session
    groq_ready = manage_groq_api_key()
    # Cette fonction récupère la clé depuis la session pour initialiser le client
    groq_client = get_groq_client() if groq_ready else None

    # Initialisation des états de session si non existants
    default_session_keys = {
        'current_page_recalls': 1, 'items_per_page_filter': DEFAULT_ITEMS_PER_PAGE,
        'recent_days_filter': DEFAULT_RECENT_DAYS, 'date_filter_start': START_DATE,
        'date_filter_end': date.today(), 'search_term_main': "",
        'search_column_friendly_name_select': "Toutes les colonnes pertinentes",
        'groq_temperature': 0.2, 'groq_max_tokens': 1024, 'groq_max_context_recalls': 15,
        'groq_model': 'llama3-8b-8192',
        'groq_chat_history': [{"role": "assistant", "content": "Bonjour ! Posez-moi une question sur les données affichées ou utilisez une suggestion."}],
        'last_processed_groq_query': '',
        'clicked_suggestion_query': None,
        'date_filter_start_init': False
    }
    for key, value in default_session_keys.items():
        if key not in st.session_state: st.session_state[key] = value


    df_alim = load_data(API_URL_FOR_LOAD, START_DATE)

    # --- Check data loading status AFTER load_data returns ---
    if df_alim.empty:
        # st.error message is already handled inside load_data if an API error occurred.
        # If df_alim is empty but no API error occurred, it means no data matched the initial date/category filters.
        if 'date_publication' in df_alim.columns and not df_alim['date_publication'].empty and pd.isna(df_alim['date_publication'].iloc[0]):
             # This case means dates were invalid and filtered out in load_data.
             # The warning for this is already in load_data.
             pass # Don't add another generic error here
        else:
             # Generic empty data message if no specific error/warning was shown by load_data
             st.info("Aucun rappel alimentaire trouvé pour les critères de chargement initiaux (date de début et catégorie Alimentation). Réessayez avec une date de début différente ou contactez l'administrateur si le problème persiste.", icon="ℹ️")
        st.stop() # Stop execution if no data is loaded

    # Check if the number of loaded records might have been capped by the API limit
    if len(df_alim) >= API_MAX_RECORDS_PER_REQUEST:
         st.warning(f"Note : Le chargement a été limité à {API_MAX_RECORDS_PER_REQUEST} rappels en raison d'une restriction de l'API externe. Les analyses et affichages se basent sur ces {API_MAX_RECORDS_PER_REQUEST} premiers rappels.", icon="⚠️")


    # S'assurer que la date de début du filtre correspond à la date min réelle des données chargées la première fois
    if not st.session_state.date_filter_start_init:
        min_data_date_actual = df_alim['date_publication'].dropna().min() if 'date_publication' in df_alim.columns and not df_alim['date_publication'].dropna().empty else START_DATE
        if isinstance(min_data_date_actual, datetime): min_data_date_actual = min_data_date_actual.date()
        elif not isinstance(min_data_date_actual, date): min_data_date_actual = START_DATE
        st.session_state.date_filter_start = min_data_date_actual
        st.session_state.date_filter_start_init = True


    cols_search = st.columns([3,2])
    with cols_search[0]:
        st.session_state.search_term_main = st.text_input(
            "Rechercher un produit, marque, risque...", value=st.session_state.get('search_term_main', ""),
            placeholder="Ex: saumon, listeria, carrefour...", key="main_search_input", label_visibility="collapsed"
        )
    with cols_search[1]:
        search_column_options_friendly = ["Toutes les colonnes pertinentes"] + list(FRIENDLY_TO_API_COLUMN_MAPPING.keys())
        current_search_col = st.session_state.get('search_column_friendly_name_select', "Toutes les colonnes pertinentes")
        try:
             default_index = search_column_options_friendly.index(current_search_col)
        except ValueError:
             default_index = 0
             st.session_state.search_column_friendly_name_select = "Toutes les colonnes pertinentes"

        st.session_state.search_column_friendly_name_select = st.selectbox(
            "Chercher dans:", search_column_options_friendly,
            index=default_index,
            key="main_search_column_select", label_visibility="collapsed"
        )

    (selected_main_categories, selected_subcategories, selected_risks,
     selected_dates_tuple_main, items_per_page_setting) = create_advanced_filters(df_alim)

    search_column_api = None
    if st.session_state.search_column_friendly_name_select != "Toutes les colonnes pertinentes":
        search_column_api = FRIENDLY_TO_API_COLUMN_MAPPING.get(st.session_state.search_column_friendly_name_select)

    current_filtered_df = filter_data(
        df_alim, selected_subcategories, selected_risks, st.session_state.search_term_main,
        selected_dates_tuple_main, selected_main_categories, search_column_api
    )

    tab_dashboard, tab_viz, tab_chatbot = st.tabs(["📊 Tableau de Bord", "📈 Visualisations", "🤖 Assistant IA"])

    with tab_dashboard:
        st.subheader("Aperçu Actuel des Rappels Alimentaires")
        display_metrics_cards(current_filtered_df)
        st.markdown("---")
        display_paginated_recalls(current_filtered_df, items_per_page_setting)

    with tab_viz:
        st.subheader("Exploration Visuelle des Données Filtrées")
        create_improved_visualizations(current_filtered_df)

    with tab_chatbot:
        st.subheader("💬 Questionner l'Assistant IA sur les Rappels Filtrés")
        st.markdown(
            "<sub>L'IA répondra en se basant sur les rappels actuellement affichés avec vos filtres. "
            "Les réponses sont des suggestions et doivent être vérifiées.</sub>",
            unsafe_allow_html=True
        )

        # Check IA status and display appropriate message
        ai_is_ready = groq_ready and (groq_client is not None)

        if not ai_is_ready:
             st.info("L'assistant IA n'est pas disponible. Veuillez configurer une clé API Groq valide dans la barre latérale.")
             # Optionnel: afficher le détail de l'erreur client si présent
             if "groq_client_error" in st.session_state and st.session_state.groq_client_error:
                  st.caption(f"Détails de l'erreur de connexion : {st.session_state.groq_client_error}")

        else: # IA is ready, display chat interface
            st.markdown("<div class='suggestion-button-container'>", unsafe_allow_html=True)
            suggestion_cols = st.columns(3)
            suggestion_queries = {
                "Tendance générale des rappels ?": {"type": "trend"},
                "Quels sont les 3 principaux risques ?": {"type": "context_only"},
                "Rappels pour 'fromage' ?": {"type": "context_specific", "product": "fromage"},
                 "Analyse des 'Listeria'": {"type": "trend", "risk": "Listeria"},
                 "Évolution des rappels 'Viande' ?": {"type": "trend", "product": "Viande"},
                 "Quels produits 'Bio' ont été rappelés ?": {"type": "context_specific", "product": "Bio"},
            }

            idx = 0
            for query_text, params in suggestion_queries.items():
                with suggestion_cols[idx % len(suggestion_cols)]:
                    if st.button(query_text, key=f"suggestion_{idx}", use_container_width=True):
                        st.session_state.clicked_suggestion_query = {"query": query_text, "params": params}
                        st.session_state.user_groq_query_input_main = query_text
                        st.experimental_rerun()
                idx += 1
            st.markdown("</div>", unsafe_allow_html=True)

            chat_display_container = st.container(height=450, border=False)
            with chat_display_container:
                for message in st.session_state.groq_chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"], unsafe_allow_html=True)
                        if message["role"] == "assistant" and "graph_base64" in message and message["graph_base64"]:
                            st.image(f"data:image/png;base64,{message['graph_base64']}")

            user_groq_query = st.chat_input(
                "Posez votre question à l'IA...",
                value=st.session_state.get('user_groq_query_input_main', ""),
                key="user_groq_query_input_main",
                disabled=not ai_is_ready # Disable if IA is not ready
            )

            query_to_process = None
            params_to_process = {}

            if st.session_state.clicked_suggestion_query:
                 query_to_process = st.session_state.clicked_suggestion_query["query"]
                 params_to_process = st.session_state.clicked_suggestion_query["params"]
                 st.session_state.clicked_suggestion_query = None
            elif user_groq_query and user_groq_query.strip() and user_groq_query != st.session_state.get('last_processed_groq_query', ''):
                 query_to_process = user_groq_query
                 query_lower = query_to_process.lower()
                 if any(k in query_lower for k in ["tendance", "évolution", "statistique", "analyse de", "combien de rappel"]):
                     params_to_process["type"] = "trend"
                     possible_products = ["fromage", "viande", "bio", "poulet", "saumon", "lait"]
                     possible_risks = ["listeria", "salmonelle", "e. coli", "allergène"]
                     for p in possible_products:
                         if p in query_lower: params_to_process["product"] = p; break
                     for r in possible_risks:
                         if r in query_lower: params_to_process["risk"] = r; break

            if query_to_process and ai_is_ready: # Only process if IA is ready
                st.session_state.groq_chat_history.append({"role": "user", "content": query_to_process})
                st.session_state.last_processed_groq_query = query_to_process
                st.session_state.user_groq_query_input_main = "" # Clear input after processing

                with chat_display_container:
                     with st.chat_message("user"): st.markdown(query_to_process)


                with st.spinner("L'assistant IA réfléchit... 🤔"):
                    context_text_for_ai = prepare_context_for_ia(
                        current_filtered_df, max_items=st.session_state.get('groq_max_context_recalls', 15)
                    )

                    trend_results = None
                    if params_to_process.get("type") == "trend":
                        trend_results = analyze_trends_data(
                            current_filtered_df,
                            product_type=params_to_process.get("product"),
                            risk_type=params_to_process.get("risk")
                        )

                    ai_response_text = ask_groq_ai(groq_client, query_to_process, context_text_for_ai, trend_results)

                assistant_message = {"role": "assistant", "content": ai_response_text}
                if trend_results and trend_results["status"] == "success" and "graph_base64" in trend_results:
                    assistant_message["graph_base64"] = trend_results["graph_base64"]

                st.session_state.groq_chat_history.append(assistant_message)

                st.experimental_rerun() # Rerun to display the new message


    st.sidebar.markdown("---")
    min_data_date_actual_display = df_alim['date_publication'].dropna().min() if 'date_publication' in df_alim.columns and not df_alim['date_publication'].dropna().empty else START_DATE
    if isinstance(min_data_date_actual_display, datetime): min_data_date_actual_display = min_data_date_actual_display.date()
    elif not isinstance(min_data_date_actual_display, date): min_data_date_actual_display = START_DATE

    st.sidebar.caption(f"Données RappelConso 'Alimentation'. {len(df_alim)} rappels chargés (depuis le {min_data_date_actual_display.strftime('%d/%m/%Y')}).")
    if st.sidebar.button("🔄 Mettre à jour les données (efface le cache)", type="primary", use_container_width=True, key="update_data_btn"):
        st.cache_data.clear()
        if 'date_filter_start_init' in st.session_state: del st.session_state.date_filter_start_init
        st.experimental_rerun()

if __name__ == "__main__":
    main()
