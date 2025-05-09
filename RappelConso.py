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
import json # Pour le débogage des réponses API

# Configuration de la page
st.set_page_config(
    page_title="RappelConso Insight",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS (Correction 2: Nouveau CSS pour un design amélioré)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

        /* --- Base & Global Styles --- */
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f5f7fa; /* Light bluish-gray background */
            color: #333;
        }

        .main .block-container {
            max-width: 1200px; /* Max width for content */
            padding: 2rem 1rem;
            margin: 0 auto; /* Center the container */
        }

        /* --- Header --- */
        .header-container {
            background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%); /* Darker blue gradient */
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15); /* Stronger shadow */
            text-align: center;
        }
        .header-title {
            color: white;
            font-size: 2.8em;
            font-weight: 700; /* Bold font */
            margin: 0;
            letter-spacing: 0.5px;
            animation: fadeInDown 1s ease-out;
        }
        .header-container p {
            color: #e0e0e0; /* Lighter white for subtitle */
            font-size: 1.2em;
            opacity: 0.9;
            margin-top: 10px;
        }

        /* --- Sidebar --- */
        .css-1lcbmhc.e1fqkh3o0 { /* Sidebar main */
            background-color: #ffffff;
            border-right: 1px solid #e0e0e0;
            box-shadow: 2px 0 5px rgba(0,0,0,0.05);
        }
        .sidebar-logo {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 90%; /* Slightly wider logo */
            padding: 15px 0;
            border-radius: 5px;
            margin-bottom: 1.5rem;
        }
        .stRadio > label {
            font-size: 1.1em;
            padding: 0.5em 0.75em;
            border-radius: 5px;
            transition: background-color 0.3s ease, color 0.3s ease;
            font-weight: 500; /* Medium font weight */
        }
         .stRadio > label:hover {
            background-color: #e1f5fe; /* Lighter blue hover */
            color: #1a5276;
        }
        .stRadio > div[role="radiogroup"] label {
             margin-bottom: 0.5rem; /* Space between radio buttons */
        }


        /* --- Metric Cards --- */
        .metric-card {
            background-color: #ffffff;
            padding: 1.8rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            margin-bottom: 1.5rem;
            border-top: 5px solid #2980b9; /* Blue border top */
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: 700; /* Bold font */
            color: #1a5276; /* Darker blue */
            margin-bottom: 0.5rem;
        }
        .metric-label {
            font-size: 1.1em;
            color: #555;
            font-weight: 500;
        }

        /* --- Tabs --- */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #e1f5fe; /* Very light blue */
            border-radius: 8px;
            padding: 5px;
            margin-bottom: 1rem;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: #1a5276; /* Darker blue for inactive tabs */
            font-weight: 500;
            border-radius: 6px;
            transition: background-color 0.3s ease, color 0.3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #b3e5fc; /* Slightly darker hover */
            color: #1a5276;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2980b9; /* Blue for active tab */
            color: white !important;
            box-shadow: 0 2px 8px rgba(41, 128, 185, 0.4);
            font-weight: 600;
        }

        /* --- Recall Cards (Correction 3: New Card Layout) --- */
        .recall-card-container {
            background-color: white;
            border-radius: 10px;
            padding: 1.2rem;
            margin-bottom: 1.5rem; /* Keep margin-bottom for spacing in columns */
            box-shadow: 0 3px 12px rgba(0,0,0,0.08);
            border-left: 5px solid #2980b9;
            transition: transform 0.2s ease;
            display: flex;
            flex-direction: column;
            height: 100%; /* Ensure cards in the same row have similar heights */
        }
        .recall-card-container:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.12);
        }
        .recall-card-image-container {
            width: 100%;
            height: 140px; /* Slightly increased height */
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            border-radius: 8px;
            margin-bottom: 0.8rem;
            background-color: #f8f9fa; /* Light background for image area */
        }
        .recall-card-image-container img {
            max-width: 100%; /* Ensure image fits */
            max-height: 100%; /* Ensure image fits */
            object-fit: contain; /* Use contain to show full image */
            border-radius: 6px;
        }
        .recall-card-content {
            flex-grow: 1; /* Allows content area to take available space */
            display: flex;
            flex-direction: column;
        }
        .recall-card-container h5 {
            color: #1a5276;
            font-size: 1.25rem;
            margin-top: 0; /* Removed top margin */
            margin-bottom: 0.6rem; /* Slightly reduced bottom margin */
            font-weight: 600;
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2; /* Limit title to 2 lines */
            -webkit-box-orient: vertical;
        }
        .recall-date {
            color: #666;
            font-size: 0.85rem;
            margin-bottom: 0.8rem;
        }
        .risk-badge {
            display: inline-block;
            padding: 0.3rem 0.7rem;
            border-radius: 18px; /* More rounded pills */
            font-size: 0.9rem;
            font-weight: 600; /* Bolder text */
            margin-bottom: 0.8rem;
            text-transform: capitalize; /* Capitalize first letter */
        }
        .risk-high {
            background-color: #ffebee;
            color: #c62828; /* Darker red */
            border: 1px solid #ef9a9a;
        }
        .risk-medium {
            background-color: #fff3e0;
            color: #ef6c00; /* Darker orange */
             border: 1px solid #ffcc80;
        }
        .risk-low { /* Used for 'Risque non spécifié' or less severe */
            background-color: #e8f5e9;
            color: #2e7d32; /* Darker green */
            border: 1px solid #a5d6a7;
        }
         .risk-default { /* For any other explicit risk not high/medium */
            background-color: #e3f2fd;
            color: #1565c0; /* Darker blue */
             border: 1px solid #90caf9;
        }

        .recall-info-item {
            margin-bottom: 0.6rem; /* Space between info items */
            font-size: 0.95rem;
        }
        .recall-info-label {
            font-weight: 600;
            color: #555;
            margin-right: 0.4rem; /* Increased space */
        }

        .recall-card-footer {
            margin-top: auto; /* Pushes footer to the bottom */
            padding-top: 1rem; /* Increased padding */
            border-top: 1px solid #eee;
        }
        .recall-card-footer .stButton button {
             background-color: #e1f5fe !important;
             color: #1a5276 !important;
             border-color: #b3e5fc !important;
             font-weight: 500;
        }
         .recall-card-footer .stButton button:hover {
            background-color: #b3e5fc !important;
            border-color: #81d4fa !important;
             color: #1a5276 !important;
        }


        /* --- Buttons --- */
        .stButton > button {
            border-radius: 6px;
            padding: 0.5em 1em;
            font-weight: 500;
            transition: background-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
             box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .stButton > button[kind="primary"] {
            background-color: #2980b9;
            color: white;
            border: none;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #1a5276;
        }
        .stButton > button[kind="secondary"] {
            background-color: #f0f2f6;
            color: #333;
            border: 1px solid #ddd;
        }
        .stButton > button[kind="secondary"]:hover {
            background-color: #e0e3e8;
            color: #000;
        }

        /* --- Expander --- */
        .stExpander {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            background-color: #fafafa;
            margin-bottom: 1rem; /* Add space below expander */
        }
        .stExpander header {
            font-size: 1.05em;
            font-weight: 600;
            color: #1a5276;
            padding: 0.75rem 1rem; /* Padding inside header */
        }
         .stExpander div[data-baseweb="button"] { /* Target expander header button */
            padding: 0 !important; /* Remove default button padding */
         }
         .stExpander div[data-baseweb="button"] > div:first-child { /* Target the text/icon div */
             padding: 0.75rem 1rem !important; /* Apply padding here */
         }


        /* --- Filter Pills --- */
        .filter-pills-container {
            margin-top: 1.5rem; /* More space above pills */
            margin-bottom: 1.5rem; /* More space below pills */
            padding: 1rem;
            background-color: #eef; /* Light blue background for filter summary */
            border-left: 4px solid #2980b9;
            border-radius: 8px;
        }
        .filter-pills-title {
            font-size: 1em;
            font-weight: 600;
            color: #1a5276;
            margin-bottom: 0.6rem;
        }
        .filter-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem; /* Increased gap */
        }
        .filter-pill {
            background-color: #b3e5fc; /* Light blue pill background */
            color: #1a5276;
            padding: 0.4rem 0.8rem; /* Increased padding */
            border-radius: 20px; /* More rounded pills */
            font-size: 0.9em;
            font-weight: 500;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
             border: 1px solid #81d4fa;
        }

        /* --- Chat Interface --- */
        .chat-container {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            margin-top: 1rem;
        }
         /* Adjust Streamlit's internal chat message styles */
        .stChatMessage {
            margin-bottom: 1rem !important; /* Space between messages */
            padding: 0.75rem 1rem !important; /* Padding inside message bubble */
            border-radius: 15px !important;
            max-width: 85% !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
            position: relative; /* Needed for triangle */
        }

        /* Styles for user messages */
        .stChatMessage[data-testid="chatAvatarIcon-user"] {
            margin-left: auto; /* Align user message to the right */
            margin-right: 0;
        }
        .stChatMessage[data-testid="chatAvatarIcon-user"] + div .stChatMessageContent {
            background-color: #2980b9 !important; /* Blue background */
            color: white !important;
            border-bottom-right-radius: 5px !important; /* Pointy corner */
            border-bottom-left-radius: 15px !important;
            border-top-left-radius: 15px !important;
            border-top-right-radius: 15px !important;
        }
        /* Styles for assistant messages */
         .stChatMessage[data-testid="chatAvatarIcon-assistant"] {
            margin-right: auto; /* Align assistant message to the left */
            margin-left: 0;
        }
        .stChatMessage[data-testid="chatAvatarIcon-assistant"] + div .stChatMessageContent {
            background-color: #e9ecef !important; /* Light gray background */
            color: #333 !important;
            border-bottom-left-radius: 5px !important; /* Pointy corner */
            border-bottom-right-radius: 15px !important;
            border-top-left-radius: 15px !important;
            border-top-right-radius: 15px !important;
        }

        .stChatMessage p { /* Ensure text within chat messages is styled */
            font-size: 1em !important;
            line-height: 1.5 !important;
            margin: 0 !important; /* Remove default paragraph margin */
        }
         .stChatMessage img { /* Images within chat messages */
            max-width: 100%;
            border-radius: 7px;
            margin-top: 10px;
        }

        .stTextArea textarea {
            border-radius: 8px;
            border: 1px solid #ccc;
            padding: 10px;
        }
        .stTextArea textarea:focus {
            border-color: #2980b9;
            box-shadow: 0 0 0 0.2rem rgba(41, 128, 185, 0.25);
        }

        /* Suggestions d'analyses */
        .suggestion-button-container {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        .suggestion-button-container .stButton button {
            background-color: #e1f5fe; /* Light blue */
            color: #1a5276; /* Darker blue */
            border: 1px solid #b3e5fc; /* Medium blue border */
            font-size: 0.9em;
            font-weight: 500;
            padding: 0.4em 0.8em;
        }
        .suggestion-button-container .stButton button:hover {
            background-color: #b3e5fc;
            border-color: #81d4fa;
            transform: translateY(-1px);
        }

        /* --- Debug Panel --- */
        .debug-panel {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 0.9em;
        }
        .debug-title {
            color: #1a5276;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .debug-section {
            margin: 5px 0;
            padding: 5px;
            background-color: #ffffff;
            border-left: 3px solid #1a5276;
            word-break: break-all; /* Prevent overflow */
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
         /* Apply animation to main content blocks */
        .main .block-container > div > div {
             animation: fadeInUp 0.5s ease-out;
        }


        /* --- Responsive Design --- */
        @media (max-width: 768px) {
            .header-title { font-size: 2em; }
            .header-container p { font-size: 1em; }
            .metric-card { padding: 1.2rem; margin-bottom: 1rem; }
            .metric-value { font-size: 2em; }
            .metric-label { font-size: 1em; }
            .stTabs [data-baseweb="tab"] { font-size: 0.9em; padding: 8px 10px;}
            .main .block-container { padding-left: 0.5rem; padding-right: 0.5rem; }
            .stChatMessage { max-width: 95% !important; padding: 0.6rem 0.8rem !important; } /* Less padding on small screens */
             .recall-card-container { padding: 1rem; margin-bottom: 1rem; }
             .recall-card-image-container { height: 100px; }
             .recall-card-container h5 { font-size: 1.1rem; margin-bottom: 0.5rem; }
             .risk-badge { font-size: 0.8rem; padding: 0.2rem 0.5rem; margin-bottom: 0.6rem; }
             .recall-date, .recall-info-item { font-size: 0.9rem; margin-bottom: 0.4rem;}
             .recall-card-footer { padding-top: 0.6rem; }
        }
    </style>
""", unsafe_allow_html=True)

# --- Constants ---
# URL de base de l'API v2
API_BASE_URL = "https://data.economie.gouv.fr/api/v2/catalog/datasets/rappelconso-v2-gtin-espaces/records"
START_DATE = date(2022, 1, 1)
API_TIMEOUT_SEC = 30
DEFAULT_ITEMS_PER_PAGE = 6
DEFAULT_RECENT_DAYS = 30
LOGO_URL = "https://raw.githubusercontent.com/M00N69/RAPPELCONSO/main/logo%2004%20copie.jpg"

# Mappings des noms de colonnes entre l'interface et l'API (gardé pour compatibilité)
FRIENDLY_TO_API_COLUMN_MAPPING = {
    "Motif du rappel": "motif_du_rappel",
    "Risques encourus": "risques_encourus",
    "Nom de la marque": "nom_de_la_marque_du_produit",
    "Nom commercial": "nom_commercial", # Note: API v2 uses 'libelle' sometimes for this
    "Modèle/Référence": "modeles_ou_references",
    "Distributeurs": "distributeurs",
    "Catégorie principale": "categorie_de_produit", # Note: API v2 uses 'categorie_produit'
    "Sous-catégorie": "sous_categorie_de_produit" # Note: API v2 uses 'sous_categorie_produit'
}
API_TO_FRIENDLY_COLUMN_MAPPING = {v: k for k, v in FRIENDLY_TO_API_COLUMN_MAPPING.items()}

# Map des nouveaux noms de colonnes API v2 vers les noms utilisés dans le code v1 (pour compatibilité interne)
API_V2_TO_INTERNAL_COLUMN_MAPPING = {
    "reference_fiche": "reference_fiche", # Nouveau dans V2
    "date_debut_commercialisation": "date_debut_commercialisation",
    "date_date_fin_commercialisation": "date_fin_commercialisation", # Correction: mapping vers ancien nom
    "temperature_conservation": "temperature_conservation",
    "marque_salubrite": "marque_de_salubrite", # Correction: mapping vers ancien nom
    "informations_complementaires": "informations_complementaires",
    "liens_vers_les_images": "liens_vers_images",
    "lien_vers_la_liste_des_produits": "lien_vers_liste_des_produits",
    "lien_vers_la_liste_des_distributeurs": "lien_vers_liste_des_distributeurs",
    "lien_vers_affichette_pdf": "lien_vers_affichette_pdf",
    "lien_vers_la_fiche_rappel": "lien_vers_la_fiche_rappel",
    "date_publication": "date_publication",
    "categorie_produit": "categorie_de_produit", # Nouveau nom V2 -> Ancien nom V1
    "sous_categorie_produit": "sous_categorie_de_produit", # Nouveau nom V2 -> Ancien nom V1
    "marque_produit": "nom_de_la_marque_du_produit", # Nouveau nom V2 -> Ancien nom V1
    "motif_rappel": "motif_du_rappel", # Nouveau nom V2 -> Ancien nom V1
    "zone_geographique_de_vente": "zone_geographique_de_vente",
    "modeles_ou_references": "modeles_ou_references", # Nom inchangé
    "identification_produits": "identification_produit", # Nouveau nom V2 -> Ancien nom V1
    "conditionnements": "conditionnements", # Nom inchangé
    "risques_encourus": "risques_encourus", # Nom inchangé
    "conduites_a_tenir_par_le_consommateur": "conduites_a_tenir", # Nouveau nom V2 -> Ancien nom V1
    "numero_contact": "numero_contact", # Nom inchangé
    "modalites_de_compensation": "modalites_compensation", # Nouveau nom V2 -> Ancien nom V1
    "libelle": "libelle", # Nouveau dans V2, peut servir de fallback pour nom_commercial
}

# --- Fonctions utilitaires / débogage ---
def debug_log(message, data=None):
    """Fonction pour afficher des logs de débogage"""
    if not st.session_state.get('debug_mode', False):
        return

    with st.expander(f"DEBUG: {message}", expanded=False):
        st.write(message)
        if data is not None:
            if isinstance(data, pd.DataFrame):
                st.write(f"Shape: {data.shape}")
                st.write(f"Columns: {data.columns.tolist()}")
                st.write("Sample data:")
                st.dataframe(data.head(3))
            elif isinstance(data, dict) or isinstance(data, list):
                try:
                    st.json(data)
                except TypeError:
                    st.write(data) # Fallback for non-serializable data
            else:
                st.write(data)

def debug_dataframe(df, section_name=""):
    """Affiche des informations de débogage sur un DataFrame."""
    if not st.session_state.get('debug_mode', False):
        return

    with st.expander(f"DEBUG DataFrame: {section_name}", expanded=False):
        st.write(f"Nombre de lignes: {len(df)}")
        st.write(f"Colonnes: {df.columns.tolist()}")

        # Afficher des exemples de lignes
        if not df.empty:
            st.write("Exemple de données:")
            st.dataframe(df.head(3))

            # Vérifier les types de colonnes
            for col in ['date_publication', 'categorie_de_produit', 'risques_encourus', 'liens_vers_images']:
                if col in df.columns and not df[col].empty:
                     sample_value = df[col].iloc[0]
                     st.write(f"Type de '{col}': {type(sample_value)}")
                     st.write(f"Exemple de '{col}': {sample_value}")
        else:
             st.write("DataFrame est vide.")


# --- Fonctions de chargement de données ---
@st.cache_data(show_spinner="Chargement des données RappelConso...", ttl=3600)
def load_data(start_date_filter=START_DATE):
    """Charge les données depuis l'API v2 de RappelConso."""
    all_records = []
    start_date_str = start_date_filter.strftime('%Y-%m-%d')
    today_str = date.today().strftime('%Y-%m-%d')

    # Construction des filtres avec la nouvelle syntaxe API v2
    # Note: Using 'categorie_produit' from V2 API
    where_clause = f"categorie_produit='Alimentation' AND date_publication >= '{start_date_str}' AND date_publication <= '{today_str}'"

    debug_log(f"Clause WHERE: {where_clause}")

    current_start_row = 0
    rows_per_page = 100  # Pagination interne
    total_hits_estimate = 0

    try:
        # Première requête pour obtenir le nombre total (optimisation, peut être imprécis mais donne une idée)
        # Note: limit=0 might not work for count on all APIs, but let's keep the attempt.
        # A more reliable way might be limit=1&rows=0 if the API supports it, or just fetching the first page.
        # Let's assume limit=0 works for total_count
        params = {
            'where': where_clause,
            'limit': 0
        }
        response = requests.get(API_BASE_URL, params=params, timeout=API_TIMEOUT_SEC)
        response.raise_for_status()
        data = response.json()
        total_hits_estimate = data.get('total_count', 0)

        debug_log(f"Total estimé de rappels: {total_hits_estimate}")

        if total_hits_estimate == 0:
            st.warning("Aucun rappel 'Alimentation' trouvé pour les critères spécifiés.")
            return pd.DataFrame()

        # Récupération des données par lots
        # Limit the total number of records fetched to avoid excessive memory usage/long load times
        max_fetch_limit = 5000 # Arbitrary limit, adjust if needed
        actual_fetch_limit = min(total_hits_estimate, max_fetch_limit)

        while current_start_row < actual_fetch_limit:
            params = {
                'where': where_clause,
                'limit': rows_per_page,
                'offset': current_start_row,
                'timezone': 'Europe/Paris' # Specify timezone
            }

            response = requests.get(API_BASE_URL, params=params, timeout=API_TIMEOUT_SEC)
            response.raise_for_status()
            data = response.json()

            # Extraction des records avec la nouvelle structure API v2
            records = data.get('records', [])

            if not records:
                break # No more records

            # Adapter à la nouvelle structure de réponse
            for rec in records:
                fields = rec.get('record', {}).get('fields', {})
                if fields:
                    all_records.append(fields)

            current_start_row += len(records)
            # Update total_hits_estimate if the first page yields more precise info
            if current_start_row == len(records) and data.get('total_count') is not None:
                 total_hits_estimate = data['total_count']
                 actual_fetch_limit = min(total_hits_estimate, max_fetch_limit)


            if total_hits_estimate > 0:
                # Show progress based on actual fetched records vs estimated total or limit
                progress_percent = min(100, int((current_start_row / actual_fetch_limit) * 100))
                st.toast(f"Chargement des rappels en cours... {current_start_row}/{actual_fetch_limit} ({progress_percent}%)", icon="⏳")

            time.sleep(0.05)  # Pause pour éviter de surcharger l'API

        if current_start_row >= max_fetch_limit and total_hits_estimate > max_fetch_limit:
             st.warning(f"Limite de {max_fetch_limit} rappels atteinte. Seuls les {max_fetch_limit} premiers rappels 'Alimentation' les plus récents sont chargés.")


    except requests.exceptions.HTTPError as http_err:
        st.error(f"Erreur HTTP de l'API: {http_err}")
        try:
            error_detail = response.json()
            st.error(f"Détails de l'erreur JSON: {error_detail}")
        except:
            st.error(f"Contenu brut de l'erreur: {response.text}")
        return pd.DataFrame()
    except requests.exceptions.Timeout:
         st.error("La requête API a dépassé le temps imparti.")
         return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return pd.DataFrame()

    if not all_records:
        st.warning("Aucun rappel 'Alimentation' trouvé pour les critères spécifiés.")
        return pd.DataFrame()

    # Création du DataFrame
    df = pd.DataFrame(all_records)

    debug_log("DataFrame créé à partir des données API (avant renommage)", df.head(3))
    debug_log("Colonnes DataFrame avant renommage", df.columns.tolist())


    # Renommage des colonnes pour correspondre à l'ancienne structure et à la logique interne
    # Only rename if the new V2 column exists in the fetched data
    rename_cols = {k: v for k, v in API_V2_TO_INTERNAL_COLUMN_MAPPING.items() if k in df.columns}
    df = df.rename(columns=rename_cols)

    debug_log("DataFrame après renommage des colonnes V2->V1", df.head(3))
    debug_log("Colonnes DataFrame après renommage", df.columns.tolist())


    # Conversion et nettoyage des dates
    if 'date_publication' in df.columns:
        # Convertir en datetime, coercer les erreurs, puis extraire la date
        df['date_publication'] = pd.to_datetime(df['date_publication'], errors='coerce').dt.date
        # Supprimer les lignes où la conversion de date a échoué
        df = df.dropna(subset=['date_publication'])
        df = df.sort_values(by='date_publication', ascending=False).reset_index(drop=True)
    else:
        st.warning("Colonne 'date_publication' manquante après chargement des données.")
        df['date_publication'] = pd.NA # Ensure column exists even if empty


    # Assurer que toutes les colonnes nécessaires pour l'application existent
    # Utiliser les noms internes (correspondant aux anciens noms V1)
    required_internal_columns = [
        "categorie_de_produit", "sous_categorie_de_produit", "nom_de_la_marque_du_produit",
        "nom_commercial", "modeles_ou_references", "distributeurs", "risques_encourus",
        "motif_du_rappel", "date_publication", "liens_vers_images", "lien_vers_la_fiche_rappel",
        "libelle" # Keep 'libelle' as it can be useful (e.g., for product name fallback)
    ]
    for col in required_internal_columns:
        if col not in df.columns:
            debug_log(f"Colonne '{col}' manquante dans les données, ajoutée avec des valeurs NA.")
            df[col] = pd.NA

    debug_dataframe(df, "DataFrame final après prétraitement")

    return df

def filter_data(data_df, selected_subcategories, selected_risks, search_term, selected_dates_tuple, selected_categories, search_column_api_name=None):
    """Filtre les données selon les critères spécifiés."""
    if data_df.empty:
        return pd.DataFrame()

    filtered_df = data_df.copy()

    debug_log(f"Filtrage - données initiales: {len(filtered_df)} lignes")

    # Filtrage par catégorie
    if selected_categories and 'categorie_de_produit' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['categorie_de_produit'].isin(selected_categories)]
        debug_log(f"Après filtre catégories ({selected_categories}): {len(filtered_df)} lignes")
    elif not selected_categories and 'categorie_de_produit' in filtered_df.columns:
        # If 'Alimentation' is not explicitly selected by the user in the multiselect,
        # but we loaded only 'Alimentation', this filter is effectively always active for 'Alimentation'.
        # If we loaded ALL categories and filter here, we'd need to handle the default case.
        # Since load_data filters by 'Alimentation', this multiselect acts as a sub-filter of 'Alimentation'.
        # If the user selects nothing, it means "all subcategories within the loaded categories".
        pass # No filter applied if selected_categories is empty

    # Filtrage par sous-catégorie
    if selected_subcategories and 'sous_categorie_de_produit' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['sous_categorie_de_produit'].isin(selected_subcategories)]
        debug_log(f"Après filtre sous-catégories ({selected_subcategories}): {len(filtered_df)} lignes")

    # Filtrage par risque
    if selected_risks and 'risques_encourus' in filtered_df.columns:
         # Use pd.notna and then isin
        filtered_df = filtered_df[filtered_df['risques_encourus'].apply(lambda x: pd.notna(x) and x in selected_risks)]
        debug_log(f"Après filtre risques ({selected_risks}): {len(filtered_df)} lignes")


    # Filtrage par texte
    if search_term:
        search_term_lower = search_term.lower()
        # Determine the internal column name from the friendly name mapping
        search_column_internal_name = None
        if search_column_api_name and search_column_api_name in filtered_df.columns: # search_column_api_name is already the internal name here
             search_column_internal_name = search_column_api_name

        if search_column_internal_name:
            # Recherche dans une colonne spécifique, gérer les NA et convertir en string
            if search_column_internal_name in filtered_df.columns:
                 col_series = filtered_df[search_column_internal_name]
                 # Ensure comparison is safe for NA values and different types
                 mask = col_series.apply(lambda x: pd.notna(x) and str(x).lower().find(search_term_lower) != -1)
                 filtered_df = filtered_df[mask]
                 debug_log(f"Après recherche dans {search_column_internal_name}='{search_term}': {len(filtered_df)} lignes")
            else:
                 debug_log(f"Colonne de recherche spécifique '{search_column_internal_name}' non trouvée.", None) # Should not happen if logic is correct
                 # Fallback to searching all columns if specific column is missing

        if not search_column_internal_name or search_column_internal_name not in filtered_df.columns:
            # Recherche dans plusieurs colonnes pertinentes
            # Use internal column names
            cols_to_search_internal = [
                'nom_de_la_marque_du_produit', 'nom_commercial', 'modeles_ou_references', 'libelle',
                'risques_encourus', 'motif_du_rappel', 'sous_categorie_de_produit',
                'distributeurs'
            ]
            cols_to_search_internal = [col for col in cols_to_search_internal if col in filtered_df.columns]

            if cols_to_search_internal:
                 # Apply search to each column, handle NA, convert to string
                mask = filtered_df[cols_to_search_internal].apply(
                    lambda series: series.apply(lambda x: pd.notna(x) and str(x).lower().find(search_term_lower) != -1)
                ).any(axis=1)
                filtered_df = filtered_df[mask]
                debug_log(f"Après recherche dans plusieurs colonnes '{search_term}': {len(filtered_df)} lignes")
            else:
                 debug_log("Aucune colonne pertinente trouvée pour la recherche textuelle.", None)


    # Filtrage par date
    if 'date_publication' in filtered_df.columns and not filtered_df.empty:
        start_filter_date = selected_dates_tuple[0]
        end_filter_date = selected_dates_tuple[1]

        # Ensure dates are date objects for comparison
        if isinstance(start_filter_date, datetime):
            start_filter_date = start_filter_date.date()
        if isinstance(end_filter_date, datetime):
            end_filter_date = end_filter_date.date()

        debug_log(f"Filtrage par date: {start_filter_date} à {end_filter_date}")

        try:
            # Ensure the column is date objects or pd.NaT before filtering
            # This conversion already happened in load_data, but double-check type
            if not pd.api.types.is_datetime64_any_dtype(filtered_df['date_publication']) and not all(isinstance(d, date) or pd.isna(d) for d in filtered_df['date_publication']):
                 debug_log("Re-converting date_publication for filtering", None)
                 filtered_df['date_publication'] = pd.to_datetime(filtered_df['date_publication'], errors='coerce').dt.date

            # Filter by date, correctly handling potential NA values in date_publication
            mask_date = filtered_df['date_publication'].apply(
                lambda x: pd.notna(x) and x >= start_filter_date and x <= end_filter_date
            )
            filtered_df = filtered_df[mask_date]
            debug_log(f"Après filtre date: {len(filtered_df)} lignes")
        except Exception as e:
            st.warning(f"Erreur lors du filtrage par date: {e}")
            debug_log(f"Erreur détaillée filtre date: {e}", e)
            # If date filtering fails, return the DataFrame without date filtering
            # filtered_df = data_df # Or just return the state before this filter
            # Decided to just return the partially filtered df and show the warning


    debug_dataframe(filtered_df, "DataFrame après tous les filtres")

    return filtered_df


# --- Fonctions UI (Header, Métriques, Cartes Rappel, Pagination, Filtres Avancés) ---
# Correction 5: Modification de l'en-tête
def create_header():
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">RappelConso Insight 🔍</h1>
        <p style="color: white; font-size: 1.2em; opacity: 0.9; margin-top: 10px;">
            Votre assistant IA pour la surveillance et l'analyse des alertes alimentaires en France
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_metrics_cards(data_df):
    if data_df.empty:
        st.info("Aucune donnée à afficher pour les métriques avec les filtres actuels.")
        return

    total_recalls = len(data_df)
    unique_subcategories = data_df['sous_categorie_de_produit'].nunique() if 'sous_categorie_de_produit' in data_df.columns else 0

    today = date.today()
    recent_days_filter = st.session_state.get('recent_days_filter', DEFAULT_RECENT_DAYS)
    days_ago = today - timedelta(days=recent_days_filter)

    recent_recalls = 0
    # Calcul des métriques récentes de manière robuste
    if 'date_publication' in data_df.columns and not data_df['date_publication'].dropna().empty:
        # Assurer que la colonne est de type date ou NaT pour la comparaison, gérer les NA
        temp_dates_metric = data_df['date_publication']
        recent_recalls = len(data_df[temp_dates_metric.apply(lambda x: pd.notna(x) and isinstance(x, date) and x >= days_ago)])


    severe_percent = 0
    if 'risques_encourus' in data_df.columns and not data_df.empty:
        grave_keywords = ['microbiologique', 'listeria', 'salmonelle', 'allerg', 'toxique', 'e. coli', 'corps étranger', 'chimique', 'physique', 'contamination'] # Added common risks
        # Use .str accessor safely after handling NA
        if pd.api.types.is_object_dtype(data_df['risques_encourus']):
            # Fill NA with empty string before converting to lower and searching
            search_series = data_df['risques_encourus'].fillna('').astype(str).str.lower()
            severe_risks_mask = search_series.str.contains('|'.join(grave_keywords), na=False)
            severe_risks_count = severe_risks_mask.sum()
            severe_percent = int((severe_risks_count / total_recalls) * 100) if total_recalls > 0 else 0
        else:
             debug_log("Colonne 'risques_encourus' n'est pas de type objet, risque non analysé pour les métriques.", None)


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
    st.caption("*Risques notables incluent (non exhaustif): microbiologique, listeria, salmonelle, allergène/allergie, toxique, E. coli, corps étranger, risque chimique/physique, contamination.")

# Correction 3: Fonction d'affichage des cartes de rappel améliorée
def display_recall_card(row_data):
    with st.container():
        st.markdown('<div class="recall-card-container">', unsafe_allow_html=True)

        # Image du produit
        st.markdown('<div class="recall-card-image-container">', unsafe_allow_html=True)
        image_url = None
        # Ensure 'liens_vers_images' exists and is not NA
        if 'liens_vers_images' in row_data and pd.notna(row_data['liens_vers_images']):
            image_links = str(row_data['liens_vers_images'])
            # Check if the string is not empty after conversion
            if image_links:
                 # Split by '|' and take the first link, then strip whitespace
                if '|' in image_links:
                    image_url = image_links.split('|')[0].strip()
                else:
                    image_url = image_links.strip() # Just strip if no '|'

        # Check if the URL is valid before attempting to display
        if image_url and isinstance(image_url, str) and image_url.startswith('http'):
            try:
                # Use st.image directly inside the container, the CSS handles size and fit
                st.image(image_url, caption=row_data.get('nom_commercial', 'Image produit'))
            except Exception as e:
                # Handle cases where the URL is bad or image loading fails
                debug_log(f"Erreur chargement image pour URL: {image_url} - {e}", row_data)
                st.image("https://via.placeholder.com/300/CCCCCC/666666?Text=Image+Non+Disponible")
        else:
            # Display placeholder if no valid URL is found
            st.image("https://via.placeholder.com/300/CCCCCC/666666?Text=Image+Non+Disponible")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="recall-card-content">', unsafe_allow_html=True)

        # Nom du produit - Use libelle as primary fallback if available
        product_name = row_data.get('nom_commercial',
                           row_data.get('libelle', # Added libelle as a common fallback in V2
                           row_data.get('modeles_ou_references',
                           'Produit non spécifié')))
        st.markdown(f"<h5>{product_name}</h5>", unsafe_allow_html=True)

        # Date de publication
        pub_date_obj = row_data.get('date_publication')
        # Safely format date, check for pd.NaT as well
        if pd.notna(pub_date_obj) and isinstance(pub_date_obj, date):
            formatted_date = pub_date_obj.strftime('%d/%m/%Y')
        else:
            formatted_date = "Date inconnue"

        st.markdown(f'<div class="recall-date">Publié le: {formatted_date}</div>', unsafe_allow_html=True)

        # Badge de risque
        # Ensure 'risques_encourus' exists and is not NA before processing
        risk_text_raw = 'Risque non spécifié'
        if 'risques_encourus' in row_data and pd.notna(row_data['risques_encourus']):
             risk_text_raw = str(row_data['risques_encourus']) # Ensure it's a string

        risk_text_lower = risk_text_raw.lower()

        badge_class = "risk-default" # Default class for explicit risks not matching high/medium
        badge_icon = "⚠️" # Default icon

        if any(keyword in risk_text_lower for keyword in ['listeria', 'salmonelle', 'e. coli', 'danger immédiat', 'toxique', 'botulisme']): # Added botulisme
            badge_class = "risk-high"
            badge_icon = "☠️"
        elif any(keyword in risk_text_lower for keyword in ['allergène', 'allergie', 'microbiologique', 'corps étranger', 'chimique', 'physique', 'contamination']): # Added physique, contamination
            badge_class = "risk-medium"
            badge_icon = "🔬"
        elif risk_text_raw == 'Risque non spécifié': # Specific class for unspecified
             badge_class = "risk-low" # Use low for unspecified or very minor risks

        st.markdown(f'<div class="risk-badge {badge_class}">{badge_icon} {risk_text_raw}</div>', unsafe_allow_html=True)

        # Informations supplémentaires (handle potential NA values)
        marque = row_data.get('nom_de_la_marque_du_produit')
        if pd.notna(marque):
            st.markdown('<div class="recall-info-item"><span class="recall-info-label">Marque:</span> ' +
                       f"{str(marque)}</div>", unsafe_allow_html=True)

        motif = row_data.get('motif_du_rappel')
        if pd.notna(motif):
            motif_str = str(motif)
            if len(motif_str) > 100:
                motif_str = motif_str[:97] + "..."
            st.markdown('<div class="recall-info-item"><span class="recall-info-label">Motif:</span> ' +
                       f"{motif_str}</div>", unsafe_allow_html=True)

        # Distributeurs (if available and not NA)
        distributeurs = row_data.get('distributeurs')
        if pd.notna(distributeurs) and str(distributeurs).strip(): # Also check if string is not empty
            distributeurs_str = str(distributeurs)
            if len(distributeurs_str) > 70:
                distributeurs_str = distributeurs_str[:67] + "..."
            st.markdown('<div class="recall-info-item"><span class="recall-info-label">Distributeurs:</span> ' +
                       f"{distributeurs_str}</div>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Footer avec bouton
        st.markdown('<div class="recall-card-footer">', unsafe_allow_html=True)
        pdf_link = row_data.get('lien_vers_la_fiche_rappel')
        # Check if link exists and is not NA before creating the button
        if pdf_link and pd.notna(pdf_link) and str(pdf_link).strip() and str(pdf_link) != '#':
             st.link_button("📄 Fiche de rappel complète", str(pdf_link), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# Correction 4: Fonction de pagination améliorée avec disposition en grille
def display_paginated_recalls(data_df, items_per_page_setting):
    if data_df.empty:
        st.info("Aucun rappel ne correspond à vos critères de recherche.")
        return

    st.markdown(f"#### Affichage des rappels ({len(data_df)} résultats)")

    if 'current_page_recalls' not in st.session_state:
        st.session_state.current_page_recalls = 1

    total_pages = (len(data_df) + items_per_page_setting - 1) // items_per_page_setting # Correct ceiling division
    # Ensure current page is not out of bounds after filtering
    if st.session_state.current_page_recalls > total_pages and total_pages > 0:
        st.session_state.current_page_recalls = total_pages
    elif total_pages == 0:
         st.session_state.current_page_recalls = 1

    current_page = st.session_state.current_page_recalls

    start_idx = (current_page - 1) * items_per_page_setting
    end_idx = min(start_idx + items_per_page_setting, len(data_df))

    # Check if start_idx is beyond the data length
    if start_idx >= len(data_df) and len(data_df) > 0:
        # This can happen if user filters drastically on a high page number
        st.session_state.current_page_recalls = 1
        st.experimental_rerun() # Rerun to display the first page of filtered data
        return # Exit current execution path

    current_recalls_page_df = data_df.iloc[start_idx:end_idx].reset_index(drop=True) # Reset index for iloc

    debug_log(f"Pagination: page {current_page}/{total_pages}, affichage lignes {start_idx} à {end_idx-1}")

    # Utilisation de st.columns avec n colonnes (2 ou 3 selon la largeur ou préférence)
    num_columns = 3 # Set number of columns for the grid
    if items_per_page_setting < num_columns: # Adjust if items per page is less than desired columns
        num_columns = items_per_page_setting
    if len(current_recalls_page_df) < num_columns: # Adjust if current page has fewer items than columns
         num_columns = len(current_recalls_page_df)

    # Ensure num_columns is at least 1 if there are items
    if num_columns == 0 and len(current_recalls_page_df) > 0: num_columns = 1


    if num_columns > 0:
        # Calculate rows needed
        rows = (len(current_recalls_page_df) + num_columns - 1) // num_columns # Correct ceiling division

        for row_idx in range(rows):
            columns = st.columns(num_columns)
            for col_idx in range(num_columns):
                item_idx = row_idx * num_columns + col_idx
                if item_idx < len(current_recalls_page_df):
                    with columns[col_idx]:
                        # Pass the row data (a Series) to the display function
                        display_recall_card(current_recalls_page_df.iloc[item_idx])

    st.markdown("---")

    if total_pages > 1:
        cols_pagination = st.columns([1, 2, 1])

        with cols_pagination[0]:
            if st.button("← Précédent", disabled=(current_page == 1), use_container_width=True, key="prev_page_btn"):
                st.session_state.current_page_recalls -= 1
                st.rerun()

        with cols_pagination[1]:
            st.markdown(f"<div style='text-align: center; margin-top: 10px;'>Page {current_page} sur {total_pages}</div>", unsafe_allow_html=True)

        with cols_pagination[2]:
            if st.button("Suivant →", disabled=(current_page == total_pages or total_pages == 0), use_container_width=True, key="next_page_btn"): # Disable if total_pages is 0
                st.session_state.current_page_recalls += 1
                st.rerun()


def create_advanced_filters(df_full_data):
    # Use pd.NaT for invalid/missing dates to allow proper min/max calculation
    # Calculate min date from actual data, fallback to START_DATE
    min_date_data = df_full_data['date_publication'].dropna().min() if not df_full_data['date_publication'].dropna().empty else START_DATE
    max_date_data = date.today()

    debug_log(f"Date min dans les données: {min_date_data}, Date max: {max_date_data}")

    # Initialize or ensure date filter session states are correct types (date objects)
    if 'date_filter_start' not in st.session_state or not isinstance(st.session_state.date_filter_start, date):
        st.session_state.date_filter_start = min_date_data
    # Ensure the start date in session state is not *after* the current max data date or today
    if st.session_state.date_filter_start > max_date_data:
         st.session_state.date_filter_start = min_date_data # Reset if inconsistency

    if 'date_filter_end' not in st.session_state or not isinstance(st.session_state.date_filter_end, date):
        st.session_state.date_filter_end = max_date_data
    # Ensure the end date in session state is not *before* the current min data date
    if st.session_state.date_filter_end < min_date_data:
         st.session_state.date_filter_end = max_date_data # Reset if inconsistency


    # S'assurer que ce sont des objets date et non datetime
    if isinstance(st.session_state.date_filter_start, datetime): st.session_state.date_filter_start = st.session_state.date_filter_start.date()
    if isinstance(st.session_state.date_filter_end, datetime): st.session_state.date_filter_end = st.session_state.date_filter_end.date()

    with st.expander("🔍 Filtres avancés et Options d'affichage", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # Ensure column exists before trying to get unique values
            if 'categorie_de_produit' in df_full_data.columns:
                # Use pd.notna to handle potential NA values correctly
                unique_main_categories = sorted(df_full_data['categorie_de_produit'].dropna().unique())
                selected_categories = st.multiselect(
                    "Filtrer par Catégories principales:", options=unique_main_categories,
                    default=st.session_state.get('selected_categories_filter', []),
                    key="main_cat_filter"
                )
                st.session_state.selected_categories_filter = selected_categories
            else:
                st.warning("Colonne 'categorie_de_produit' manquante")
                selected_categories = [] # Ensure selected_categories is defined

            if 'sous_categorie_de_produit' in df_full_data.columns:
                # Filter the DataFrame used for populating subcategory options based on main category selection
                df_for_subcats = df_full_data
                if selected_categories:
                    df_for_subcats = df_full_data[df_full_data['categorie_de_produit'].isin(selected_categories)]
                # Use pd.notna for unique values
                unique_subcategories = sorted(df_for_subcats['sous_categorie_de_produit'].dropna().unique())
                selected_subcategories = st.multiselect(
                    "Filtrer par Sous-catégories:", options=unique_subcategories,
                    default=st.session_state.get('selected_subcategories_filter', []),
                    key="sub_cat_filter"
                )
                st.session_state.selected_subcategories_filter = selected_subcategories
            else:
                st.warning("Colonne 'sous_categorie_de_produit' manquante")
                selected_subcategories = [] # Ensure selected_subcategories is defined


        with col2:
            if 'risques_encourus' in df_full_data.columns:
                # Use pd.notna for unique values
                unique_risks = sorted(df_full_data['risques_encourus'].dropna().unique())
                display_risks = unique_risks
                # Limit options if too many
                if len(unique_risks) > 75:
                    # Use pd.notna with value_counts
                    top_risks_counts = df_full_data['risques_encourus'].dropna().value_counts().nlargest(75).index.tolist()
                    display_risks = sorted(top_risks_counts)
                    st.caption(f"Affichage des 75 risques les plus fréquents (sur {len(unique_risks)}).")

                selected_risks = st.multiselect(
                    "Filtrer par Types de risques:", options=display_risks,
                    default=st.session_state.get('selected_risks_filter', []),
                    key="risks_filter"
                )
                st.session_state.selected_risks_filter = selected_risks
            else:
                st.warning("Colonne 'risques_encourus' manquante")
                selected_risks = [] # Ensure selected_risks is defined

            # Utiliser les dates de session comme valeur par défaut pour le widget
            selected_dates_tuple_local = st.date_input(
                "Filtrer par période de publication:",
                # Ensure the tuple contains date objects or NaT, and that the min/max values are correct types
                value=(st.session_state.date_filter_start, st.session_state.date_filter_end),
                min_value=min_date_data, max_value=max_date_data,
                key="date_range_picker_adv"
            )

            # The date_input can return a tuple of None if dates are cleared, or a single date if only one is picked.
            # Handle these cases gracefully.
            if selected_dates_tuple_local is not None and len(selected_dates_tuple_local) == 2:
                start_date_widget, end_date_widget = selected_dates_tuple_local
                # Update session only if the widget values are valid dates and different from current state
                if (pd.notna(start_date_widget) and pd.notna(end_date_widget) and
                   (start_date_widget != st.session_state.date_filter_start or end_date_widget != st.session_state.date_filter_end)):
                    st.session_state.date_filter_start = start_date_widget
                    st.session_state.date_filter_end = end_date_widget
                    # Réinitialiser la page si la date change
                    st.session_state.current_page_recalls = 1
            # else: Handle potential tuple of None or single date if date_input allows it (Streamlit usually enforces tuple of 2)


        st.markdown("---")
        st.markdown("**Options d'affichage**")
        items_per_page_setting = st.slider("Nombre de rappels par page:", min_value=2, max_value=30, # Increased max for grid view
                                 value=st.session_state.get('items_per_page_filter', DEFAULT_ITEMS_PER_PAGE), step=2,
                                 key="items_page_slider")
        if items_per_page_setting != st.session_state.get('items_per_page_filter', DEFAULT_ITEMS_PER_PAGE):
            st.session_state.items_per_page_filter = items_per_page_setting
            st.session_state.current_page_recalls = 1 # Reset page on items per page change

        recent_days_setting = st.slider("Période pour 'Rappels Récents' (jours):", min_value=7, max_value=180, # Increased max period
                               value=st.session_state.get('recent_days_filter', DEFAULT_RECENT_DAYS), step=1,
                               key="recent_days_slider")
        st.session_state.recent_days_filter = recent_days_setting

        if st.button("Réinitialiser filtres et options", type="secondary", use_container_width=True, key="reset_filters_btn"):
            # Recalculate min_date_data on reset
            min_date_data_reset = df_full_data['date_publication'].dropna().min() if not df_full_data['date_publication'].dropna().empty else START_DATE

            keys_to_reset = [
                'selected_categories_filter', 'selected_subcategories_filter',
                'selected_risks_filter', 'search_term_main', 'search_column_friendly_name_select',
                'date_filter_start', 'date_filter_end',
                'items_per_page_filter', 'recent_days_filter', 'current_page_recalls',
                'user_groq_api_key', 'groq_chat_history' # Also reset Groq key and chat history
            ]
            for key_to_del in keys_to_reset:
                if key_to_del in st.session_state:
                    del st.session_state[key_to_del]

            # Set default values explicitly after deletion
            st.session_state.date_filter_start = min_date_data_reset
            st.session_state.date_filter_end = date.today()
            st.session_state.items_per_page_filter = DEFAULT_ITEMS_PER_PAGE
            st.session_state.recent_days_filter = DEFAULT_RECENT_DAYS
            st.session_state.current_page_recalls = 1
            st.session_state.search_term_main = ""
            st.session_state.search_column_friendly_name_select = "Toutes les colonnes pertinentes"
            # Initialize chat history default message
            st.session_state.groq_chat_history = [{"role": "assistant", "content": "Bonjour ! Posez-moi une question sur les données affichées ou utilisez une suggestion."}]

            st.experimental_rerun()

    # Display active filters
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

    # Display date filter if it's not the full range of the loaded data
    current_start_date_display = st.session_state.get('date_filter_start', START_DATE)
    current_end_date_display = st.session_state.get('date_filter_end', date.today())
    # Recalculate full data date range to check against
    full_data_min_date = df_full_data['date_publication'].dropna().min() if not df_full_data['date_publication'].dropna().empty else START_DATE
    full_data_max_date = df_full_data['date_publication'].dropna().max() if not df_full_data['date_publication'].dropna().empty else date.today() # Use actual max data date

    if (current_start_date_display > full_data_min_date) or (current_end_date_display < full_data_max_date):
        # Only show date filter pill if it's narrower than the full loaded data range
        active_filters_display.append(f"Période: {current_start_date_display.strftime('%d/%m/%y')} - {current_end_date_display.strftime('%d/%m/%y')}")


    if active_filters_display:
        st.markdown('<div class="filter-pills-container"><div class="filter-pills-title">Filtres actifs :</div><div class="filter-pills">' +
                    ' '.join([f'<span class="filter-pill">{f}</span>' for f in active_filters_display]) +
                    '</div></div>', unsafe_allow_html=True)

    # Return the current session state filter values
    return st.session_state.get('selected_categories_filter', []), \
           st.session_state.get('selected_subcategories_filter', []), \
           st.session_state.get('selected_risks_filter', []), \
           (st.session_state.date_filter_start, st.session_state.date_filter_end), \
           st.session_state.get('items_per_page_filter', DEFAULT_ITEMS_PER_PAGE)


# --- Fonctions de Visualisation (Onglet Visualisations) ---
def create_improved_visualizations(data_df_viz):
    if data_df_viz.empty:
        st.info("Données insuffisantes pour générer des visualisations avec les filtres actuels.")
        return

    # Ensure required columns for plotting exist before proceeding
    required_cols_viz = ['date_publication', 'categorie_de_produit', 'sous_categorie_de_produit', 'nom_de_la_marque_du_produit', 'risques_encourus', 'motif_du_rappel']
    for col in required_cols_viz:
        if col not in data_df_viz.columns:
             st.warning(f"Colonne '{col}' manquante, certaines visualisations peuvent ne pas s'afficher.")


    st.markdown('<div class="chart-container" style="margin-top:1rem;">', unsafe_allow_html=True)

    # Prétraitement unifié des dates pour toutes les visualisations
    # Ensure 'date_publication' exists and is convertible to datetime
    if 'date_publication' in data_df_viz.columns:
        data_df_viz = data_df_viz.copy() # Avoid modifying original cached df
        # Convert to datetime objects, coercing errors
        data_df_viz['date_publication_dt'] = pd.to_datetime(data_df_viz['date_publication'], errors='coerce')
        # Drop rows where date conversion failed
        data_df_viz = data_df_viz.dropna(subset=['date_publication_dt'])

        if data_df_viz.empty:
            st.warning("Dates invalides dans les données filtrées, impossible de générer les visualisations temporelles.")
            st.markdown('</div>', unsafe_allow_html=True)
            return
    else:
        st.warning("Colonne 'date_publication' manquante pour les visualisations temporelles.")
        st.markdown('</div>', unsafe_allow_html=True)
        return


    debug_log("Données pour visualisations (après prétraitement date)", data_df_viz.head(3))

    tab1, tab2, tab3 = st.tabs(["📊 Tendances Temporelles", "🍩 Répartitions par Catégorie", "☠️ Analyse des Risques"])

    with tab1:
        st.subheader("Évolution des Rappels")
        if 'date_publication_dt' in data_df_viz.columns and not data_df_viz['date_publication_dt'].empty:
            # Resample by month and count, then create year-month string for plotting
            monthly_data = data_df_viz.set_index('date_publication_dt').resample('M').size().reset_index(name='count')
            monthly_data['year_month'] = monthly_data['date_publication_dt'].dt.strftime('%Y-%m')
            monthly_data = monthly_data.sort_values('year_month') # Ensure chronological order

            if not monthly_data.empty:
                fig_temporal = px.line(monthly_data, x='year_month', y='count', title="Nombre de rappels par mois", markers=True,
                                       labels={'year_month': 'Mois', 'count': 'Nombre de rappels'})
                fig_temporal.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_temporal, use_container_width=True)
            else:
                 st.info("Pas de données mensuelles à afficher pour les tendances.")
        else:
             st.warning("Impossible de générer le graphique temporel.")


    with tab2:
        st.subheader("Répartition des Rappels")
        col_cat, col_subcat = st.columns(2)
        with col_cat:
            if 'categorie_de_produit' in data_df_viz.columns:
                 # Count values, drop NA, take top 10
                cat_counts = data_df_viz['categorie_de_produit'].dropna().value_counts().nlargest(10)
                if not cat_counts.empty:
                    fig_cat = px.pie(values=cat_counts.values, names=cat_counts.index, title="Top 10 Catégories", hole=0.4)
                    fig_cat.update_traces(textposition='inside', textinfo='percent+label', insidetextfont=dict(color="white"), marker=dict(line=dict(color='#000000', width=1))) # Improved text position and outline
                    st.plotly_chart(fig_cat, use_container_width=True)
                else: st.info("Pas de données pour la répartition par catégorie.")
            else: st.warning("Colonne 'categorie_de_produit' manquante.")

        with col_subcat:
            if 'sous_categorie_de_produit' in data_df_viz.columns:
                 # Count values, drop NA, take top 10
                subcat_counts = data_df_viz['sous_categorie_de_produit'].dropna().value_counts().nlargest(10)
                if not subcat_counts.empty:
                    fig_subcat = px.pie(values=subcat_counts.values, names=subcat_counts.index, title="Top 10 Sous-Catégories", hole=0.4)
                    fig_subcat.update_traces(textposition='inside', textinfo='percent+label', insidetextfont=dict(color="white"), marker=dict(line=dict(color='#000000', width=1))) # Improved text position and outline
                    st.plotly_chart(fig_subcat, use_container_width=True)
                else: st.info("Pas de données pour la répartition par sous-catégorie.")
            else: st.warning("Colonne 'sous_categorie_de_produit' manquante.")

        # Brand chart below the pie charts
        if 'nom_de_la_marque_du_produit' in data_df_viz.columns:
            # Exclure les valeurs manquantes ou vides avant de compter
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
        col_risk, col_motif = st.columns(2)

        with col_risk:
            if 'risques_encourus' in data_df_viz.columns:
                # Exclure les valeurs manquantes ou vides
                valid_risks = data_df_viz['risques_encourus'].dropna()
                valid_risks = valid_risks[valid_risks != '']
                if not valid_risks.empty:
                    risk_counts = valid_risks.value_counts().nlargest(10)
                    if not risk_counts.empty:
                        fig_risks = px.bar(y=risk_counts.index, x=risk_counts.values, orientation='h',
                                           title="Top 10 Risques Encourus",
                                           labels={'y': 'Risque', 'x': 'Nombre de rappels'},
                                           color=risk_counts.values, color_continuous_scale=px.colors.sequential.Reds_r)
                        fig_risks.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False, height=max(400, len(risk_counts)*40))
                        st.plotly_chart(fig_risks, use_container_width=True)
                    else: st.info("Pas assez de données de risque pour un top 10.")
                else: st.info("Pas de données de risque valides.")
            else: st.warning("Colonne 'risques_encourus' manquante.")

        with col_motif:
            if 'motif_du_rappel' in data_df_viz.columns:
                valid_motifs = data_df_viz['motif_du_rappel'].dropna()
                valid_motifs = valid_motifs[valid_motifs != '']
                if not valid_motifs.empty:
                    motif_counts = valid_motifs.value_counts().nlargest(10)
                    if not motif_counts.empty:
                        fig_motifs = px.bar(y=motif_counts.index, x=motif_counts.values, orientation='h',
                                            title="Top 10 Motifs de Rappel",
                                            labels={'y': 'Motif', 'x': 'Nombre de rappels'},
                                            color=motif_counts.values, color_continuous_scale=px.colors.sequential.Oranges_r)
                        fig_motifs.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False, height=max(400, len(motif_counts)*40))
                        st.plotly_chart(fig_motifs, use_container_width=True)
                    else: st.info("Pas assez de données de motif pour un top 10.")
                else: st.info("Pas de données de motif valides.")
            else: st.warning("Colonne 'motif_du_rappel' manquante.")

    st.markdown('</div>', unsafe_allow_html=True)


# --- Fonctions pour l'IA avec Groq ---
def manage_groq_api_key():
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 Assistant IA Groq")

    default_expanded = True
    if "user_groq_api_key" in st.session_state and st.session_state.user_groq_api_key and st.session_state.user_groq_api_key.startswith("gsk_"):
        default_expanded = False

    with st.sidebar.expander("Configurer l'accès à l'IA", expanded=default_expanded):
        if "user_groq_api_key" not in st.session_state:
            st.session_state.user_groq_api_key = ""

        new_key = st.text_input(
            "Votre clé API Groq:", type="password",
            value=st.session_state.user_groq_api_key,
            help="Obtenez votre clé sur [console.groq.com](https://console.groq.com/keys). La clé est stockée temporairement.",
            key="groq_api_key_input_sidebar"
        )
        # Only update session state if the input value actually changed
        if new_key != st.session_state.user_groq_api_key:
             st.session_state.user_groq_api_key = new_key
             # Provide feedback immediately upon key change
             if new_key and new_key.startswith("gsk_"):
                 st.success("Clé API Groq enregistrée.", icon="👍")
             elif new_key: # Invalid format
                 st.warning("Format de clé API invalide.", icon="⚠️")
             # No rerun here, let it happen naturally or by user action

        model_options = {
            "llama3-70b-8192": "Llama 3 (70B) - Puissant",
            "llama3-8b-8192": "Llama 3 (8B) - Rapide",
            "mixtral-8x7b-32768": "Mixtral (8x7B) - Large contexte",
            "gemma-7b-it": "Gemma (7B) - Léger"
        }
        # Set default model if not in session state
        if 'groq_model' not in st.session_state:
             st.session_state.groq_model = list(model_options.keys())[0] # Default to first option

        selected_model_key = st.selectbox(
            "Choisir un modèle IA:", options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=list(model_options.keys()).index(st.session_state.groq_model), # Set default from session state
            key="groq_model_select_sidebar"
        )
        # Only update session state if the selected value changed
        if selected_model_key != st.session_state.groq_model:
             st.session_state.groq_model = selected_model_key
             # No rerun here

        with st.popover("Options avancées de l'IA"):
            # Set default advanced options if not in session state
            if 'groq_temperature' not in st.session_state: st.session_state.groq_temperature = 0.2
            if 'groq_max_tokens' not in st.session_state: st.session_state.groq_max_tokens = 1024
            if 'groq_max_context_recalls' not in st.session_state: st.session_state.groq_max_context_recalls = 15

            st.session_state.groq_temperature = st.slider("Température:", 0.0, 1.0, st.session_state.groq_temperature, 0.1)
            st.session_state.groq_max_tokens = st.slider("Tokens max réponse:", 256, 4096, st.session_state.groq_max_tokens, 256)
            st.session_state.groq_max_context_recalls = st.slider("Max rappels dans contexte IA:", 5, 50, st.session_state.groq_max_context_recalls, 1)

    if st.session_state.user_groq_api_key and st.session_state.user_groq_api_key.startswith("gsk_"):
        st.sidebar.caption(f"🟢 IA prête ({model_options.get(st.session_state.groq_model, 'N/A')})")
        return True
    else:
        st.sidebar.caption("🔴 IA non configurée ou clé invalide.")
        return False

def get_groq_client():
    if "user_groq_api_key" in st.session_state and st.session_state.user_groq_api_key and st.session_state.user_groq_api_key.startswith("gsk_"):
        try:
            return Groq(api_key=st.session_state.user_groq_api_key)
        except Exception as e:
            # Display error only once or in debug mode
            debug_log(f"Erreur d'initialisation du client Groq: {e}", e)
            # Removed st.error here to avoid flooding the UI, rely on the sidebar caption
            return None
    return None

def prepare_context_for_ia(df_context, max_items=10):
    """Prépare les données de contexte pour l'IA Groq de manière robuste."""
    if df_context.empty:
        return "Aucune donnée de rappel pertinente trouvée pour cette question avec les filtres actuels."

    cols_for_ia = [
        'nom_de_la_marque_du_produit', 'nom_commercial', 'modeles_ou_references',
        'categorie_de_produit', 'sous_categorie_de_produit', 'libelle', # Included libelle
        'risques_encourus', 'motif_du_rappel', 'date_publication', 'distributeurs',
        'reference_fiche' # Include reference_fiche if exists
    ]

    # Vérifier quelles colonnes sont présentes dans le DataFrame
    cols_to_use = [col for col in cols_for_ia if col in df_context.columns]

    if not cols_to_use:
        return "Structure de données incompatible pour l'analyse IA."

    actual_max_items = min(max_items, len(df_context))
    # Ensure the sample DataFrame contains only the columns we intend to use
    context_df_sample = df_context[cols_to_use].head(actual_max_items)

    text_context = f"Voici un échantillon de **{len(context_df_sample)} rappels** (sur **{len(df_context)}** au total avec les filtres actuels) :\n\n"

    # Generate text description for each recall in the sample
    for index, row in context_df_sample.iterrows():
        item_desc = []

        # Construction plus robuste avec vérification des NaN pour chaque colonne
        for col in cols_to_use:
            value = row.get(col) # Use .get for safety if column wasn't added by fallback
            if pd.notna(value):
                # Formater la date si c'est une date
                if col == 'date_publication' and isinstance(value, date):
                    value_str = value.strftime('%d/%m/%y')
                else:
                    value_str = str(value).strip() # Convert to string and strip whitespace

                if value_str: # Only add if the string value is not empty
                    # Mapping des noms de colonnes pour l'affichage dans le contexte IA
                    display_name = {
                        'nom_de_la_marque_du_produit': 'Marque',
                        'nom_commercial': 'Produit',
                        'modeles_ou_references': 'Ref',
                        'categorie_de_produit': 'Cat',
                        'sous_categorie_de_produit': 'Sous-cat',
                        'risques_encourus': 'Risque',
                        'motif_du_rappel': 'Motif',
                        'date_publication': 'Date',
                        'distributeurs': 'Distrib',
                        'libelle': 'Libelle',
                        'reference_fiche': 'Ref Fiche'
                    }.get(col, col)

                    # Truncate long descriptions for conciseness
                    if len(value_str) > 70 and col in ['motif_du_rappel', 'distributeurs', 'modeles_ou_references']:
                         value_str = value_str[:67] + "..."

                    item_desc.append(f"**{display_name}**: {value_str}")

        if item_desc: # Only add a line if there's content for the item
            text_context += "- " + ", ".join(item_desc) + "\n"

    if len(context_df_sample) < len(df_context):
         text_context += f"\n... et {len(df_context) - len(context_df_sample)} rappels supplémentaires correspondent aux filtres mais ne sont pas inclus dans cet échantillon.\n"

    debug_log("Contexte préparé pour l'IA", text_context)
    return text_context


def analyze_trends_data(df_analysis, product_type=None, risk_type=None, time_period="monthly"): # Changed default time_period to "monthly" as that's what's plotted
    """Analyse les tendances dans les données de rappel."""
    if df_analysis.empty:
        return {"status": "no_data", "message": "Aucune donnée disponible pour l'analyse de tendance."}

    df_filtered = df_analysis.copy()

    # S'assurer que la colonne date_publication existe et conversion uniforme des dates
    if 'date_publication' in df_filtered.columns:
        # Convertir en datetime, coercer les erreurs, puis extraire la date si pas déjà fait ou type incorrect
        if not pd.api.types.is_datetime64_any_dtype(df_filtered['date_publication']): # Check if it's not already datetime
             df_filtered['date_publication_dt'] = pd.to_datetime(df_filtered['date_publication'], errors='coerce')
        else:
             df_filtered['date_publication_dt'] = df_filtered['date_publication'] # Use existing datetime column

        df_filtered = df_filtered.dropna(subset=['date_publication_dt'])
        if df_filtered.empty:
            return {"status": "no_data", "message": "Dates invalides pour l'analyse."}
    else:
        return {"status": "error", "message": "Colonne 'date_publication' manquante."}

    # Determine the actual analysis period from the filtered data
    min_analysis_date = df_filtered['date_publication_dt'].min()
    max_analysis_date = df_filtered['date_publication_dt'].max()
    period_label = f"du {min_analysis_date.strftime('%d/%m/%Y')} au {max_analysis_date.strftime('%d/%m/%Y')}"

    analysis_title_parts = ["Évolution des rappels"]

    # Filtrage des données si des critères spécifiques sont fournis pour l'analyse de tendance
    # Note: This filters WITHIN the already filtered data from the main app UI
    initial_count = len(df_filtered)
    if product_type:
        # Search in multiple relevant columns
        product_cols = ['sous_categorie_de_produit', 'nom_commercial', 'categorie_de_produit', 'libelle', 'modeles_ou_references']
        product_cols_present = [col for col in product_cols if col in df_filtered.columns]
        if product_cols_present:
            mask_product = df_filtered[product_cols_present].apply(
                lambda series: series.astype(str).str.contains(product_type, case=False, na=False)
            ).any(axis=1)
            df_filtered = df_filtered[mask_product]
            analysis_title_parts.append(f"pour '{product_type}'")
        else:
             debug_log(f"Aucune colonne pertinente pour filtrer par produit '{product_type}'.", None)


    if risk_type:
        risk_cols = ['risques_encourus', 'motif_du_rappel']
        risk_cols_present = [col for col in risk_cols if col in df_filtered.columns]
        if risk_cols_present:
            mask_risk = df_filtered[risk_cols_present].apply(
                 lambda series: series.astype(str).str.contains(risk_type, case=False, na=False)
            ).any(axis=1)
            df_filtered = df_filtered[mask_risk]
            analysis_title_parts.append(f"avec risque/motif '{risk_type}'")
        else:
             debug_log(f"Aucune colonne pertinente pour filtrer par risque '{risk_type}'.", None)


    if df_filtered.empty:
        # Add more specific message if additional filtering was applied
        if len(df_filtered) < initial_count:
             return {"status": "no_data", "message": f"Aucune donnée pour les filtres spécifiques ({product_type or 'tous produits'}, {risk_type or 'tous risques'}) sur la période actuelle."}
        else:
             return {"status": "no_data", "message": "Aucune donnée filtrée pour l'analyse de tendance."}


    # Regroupement par mois pour l'analyse temporelle
    # Resample on the datetime column, size(), reindex to fill missing months
    monthly_counts = df_filtered.set_index('date_publication_dt').resample('MS').size() # Resample on Month Start
    if monthly_counts.empty:
         return {"status": "no_data", "message": "Pas de données mensuelles à analyser après filtrage spécifique."}

    # Reindex to ensure all months in the period are present, fill missing with 0
    all_months_in_period = pd.date_range(start=monthly_counts.index.min(), end=monthly_counts.index.max(), freq='MS')
    monthly_counts = monthly_counts.reindex(all_months_in_period, fill_value=0)

    # Prepare index for plotting (string format YYYY-MM)
    monthly_counts_display_index = monthly_counts.index.strftime('%Y-%m')

    # Calculer les statistiques pour l'analyse
    total_recalls_analyzed = int(df_filtered.shape[0])
    monthly_avg = float(monthly_counts.mean()) if not monthly_counts.empty else 0

    trend_stats = {"total_recalls": total_recalls_analyzed, "monthly_avg": monthly_avg}

    # Analyse de tendance (régression linéaire)
    slope = 0
    # Need at least 2 data points (months) for regression
    if len(monthly_counts) >= 2:
        # Use numerical representation of months (e.g., 0, 1, 2...) as input for regression
        X = np.arange(len(monthly_counts)).reshape(-1, 1)
        y = monthly_counts.values.astype(float) # Ensure y is float for regression
        # Add a small constant if y is all zeros to avoid singular matrix if model changes
        if np.all(y == y[0]): y = y + 1e-6
        try:
            model = LinearRegression().fit(X, y)
            slope = float(model.coef_[0])
            trend_stats['trend_slope'] = slope
            # Define threshold for 'hausse'/'baisse' (e.g., change of more than 0.1 recalls/month)
            if slope > 0.1: trend_stats['trend_direction'] = "hausse"
            elif slope < -0.1: trend_stats['trend_direction'] = "baisse"
            else: trend_stats['trend_direction'] = "stable"
        except Exception as e:
            debug_log(f"Erreur lors de la régression linéaire: {e}", e)
            trend_stats['trend_direction'] = "indéterminée (erreur de calcul)"

    else:
        trend_stats['trend_direction'] = "indéterminée (données insuffisantes)"


    # Création du graphique matplotlib
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly_counts_display_index, monthly_counts.values, marker='o', linestyle='-', label='Rappels/mois')

    # Ajouter la ligne de tendance si suffisamment de données et calcul réussi
    if len(monthly_counts) >= 2 and 'trend_slope' in trend_stats:
        trend_line = model.predict(X)
        ax.plot(monthly_counts_display_index, trend_line, color='red', linestyle='--', label=f'Tendance ({trend_stats.get("trend_direction", "indéterminée")})')

    # Mise en forme du graphique
    ax.set_title(' '.join(analysis_title_parts), fontsize=12) # Increased title size
    ax.set_xlabel("Mois", fontsize=10); ax.set_ylabel("Nombre de rappels", fontsize=10) # Increased label sizes
    ax.tick_params(axis='x', rotation=45, labelsize=8); ax.tick_params(axis='y', labelsize=8) # Increased tick label sizes
    ax.grid(True, linestyle=':', alpha=0.6); # Slightly less opaque grid
    ax.legend(fontsize=9) # Increased legend size
    plt.tight_layout() # Adjust layout to prevent labels overlapping

    # Conversion du graphique en image base64 pour l'affichage
    buf = io.BytesIO()
    plt.savefig(buf, format="png"); buf.seek(0)
    graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) # Close the plot figure to free memory

    # Rédaction du résumé textuel de l'analyse
    text_summary = f"Analyse de tendance ({period_label}):\n"
    text_summary += f"- **Total rappels analysés** : **{trend_stats['total_recalls']}**\n"
    text_summary += f"- **Moyenne mensuelle** : **{trend_stats['monthly_avg']:.1f}** rappels\n"
    if 'trend_direction' in trend_stats:
        text_summary += f"- **Tendance générale** : {trend_stats['trend_direction']}"
        if trend_stats.get('trend_direction') not in ["indéterminée (données insuffisantes)", "stable", "indéterminée (erreur de calcul)"]:
             text_summary += f" (pente: {slope:.2f})\n"
        else:
            text_summary += "\n"

    # Ajouter des informations supplémentaires sur les sous-catégories et les risques (basé sur df_filtered)
    if 'sous_categorie_de_produit' in df_filtered.columns:
        # Use dropna before value_counts
        top_cat = df_filtered['sous_categorie_de_produit'].dropna().value_counts().nlargest(3)
        if not top_cat.empty:
            text_summary += "- **Top 3 sous-catégories** : " + ", ".join([f"{idx} (**{val}**)" for idx, val in top_cat.items()]) + "\n"

    if 'risques_encourus' in df_filtered.columns:
        # Use dropna before value_counts
        top_risk = df_filtered['risques_encourus'].dropna().value_counts().nlargest(3)
        if not top_risk.empty:
            text_summary += "- **Top 3 risques** : " + ", ".join([f"{idx} (**{val}**)" for idx, val in top_risk.items()]) + "\n"

    # Include product/risk filters in the summary if they were applied
    if product_type: text_summary += f"- Analyse spécifique pour le produit : '{product_type}'\n"
    if risk_type: text_summary += f"- Analyse spécifique pour le risque : '{risk_type}'\n"


    return {
        "status": "success", "text_summary": text_summary, "graph_base64": graph_base64,
        "monthly_data": {str(k): int(v) for k,v in monthly_counts_display_index.to_dict().items()} if not monthly_counts.empty else {}, # Ensure serializable keys
        "trend_stats": trend_stats
    }


def ask_groq_ai(client, user_query, context_data_text, trend_analysis_results=None):
    """Interroge l'API Groq avec le contexte des données et les résultats d'analyse."""
    if not client:
        return "Client Groq non initialisé. Vérifiez votre clé API dans la barre latérale."

    debug_log("Préparation requête Groq", {
        "taille du contexte": len(context_data_text),
        "analyse de tendance": trend_analysis_results.get("status", "aucune") if trend_analysis_results else "aucune",
        "modèle": st.session_state.get("groq_model", "default"),
        "temperature": st.session_state.get("groq_temperature", 0.2),
        "max_tokens": st.session_state.get("groq_max_tokens", 1024)
    })

    system_prompt = f"""Tu es "RappelConso Insight Assistant", un expert IA spécialisé dans l'analyse des données de rappels de produits alimentaires en France, basé sur les données de RappelConso.
    Date actuelle: {date.today().strftime('%d/%m/%Y')}.
    Tu réponds aux questions de l'utilisateur de manière concise, professionnelle et en te basant STRICTEMENT sur les informations et données de contexte fournies, qui correspondent aux rappels actuellement affichés et filtrés dans l'application.
    NE PAS INVENTER d'informations ou faire de suppositions sur des données non fournies (ex: données historiques complètes au-delà de la période chargée).
    Si les données de contexte ne te permettent pas de répondre, indique-le clairement (ex: "Je n'ai pas suffisamment d'informations dans les données fournies pour répondre précisément à cette question.").
    Utilise le Markdown pour mettre en **gras** les chiffres clés et les points importants.
    Si une analyse de tendance a été effectuée (et fournie dans le contexte `trend_analysis_results`) et qu'un graphique est disponible, mentionne-le et décris brièvement ce qu'il montre en t'appuyant sur le résumé d'analyse fourni. Si l'analyse de tendance a échoué ou n'a pas de données, mentionne pourquoi selon le statut.
    Si la question est hors sujet (ne concerne pas les rappels de produits alimentaires, la sécurité alimentaire liée aux rappels, ou les données fournies), réponds avec une blague COURTE et pertinente sur la sécurité alimentaire ou la nourriture, puis indique que tu ne peux pas répondre à la question car elle sort de ton domaine d'expertise RappelConso.
    Exemple de blague : "Pourquoi le fromage n'aime-t-il pas se regarder dans le miroir ? Parce qu'il fond ! Plus sérieusement, ma spécialité, ce sont les rappels de produits."
    Sois toujours courtois.
    """

    full_context_for_ai = f"Contexte des rappels de produits (échantillon et filtres actuels) :\n{context_data_text}\n\n"

    # Add trend analysis results to the context if available
    if trend_analysis_results:
        full_context_for_ai += f"Résultats de l'analyse de tendance :\n"
        full_context_for_ai += f"Statut : {trend_analysis_results.get('status', 'non spécifié')}\n"
        if trend_analysis_results.get("status") == "success":
             full_context_for_ai += f"Résumé : {trend_analysis_results.get('text_summary', 'Résumé non disponible')}\n"
             if trend_analysis_results.get("graph_base64"):
                  full_context_for_ai += "Un graphique illustrant cette tendance est disponible et sera affiché à l'utilisateur.\n"
        elif trend_analysis_results.get("status") == "no_data":
             full_context_for_ai += f"Message : {trend_analysis_results.get('message', 'Données insuffisantes pour la tendance')}\n"
        elif trend_analysis_results.get("status") == "error":
             full_context_for_ai += f"Message : {trend_analysis_results.get('message', 'Erreur lors du calcul de la tendance')}\n"
        full_context_for_ai += "\n" # Add space after trend results


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{full_context_for_ai}Question de l'utilisateur: \"{user_query}\""}
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=st.session_state.get("groq_model", "llama3-8b-8192"),
            temperature=st.session_state.get("groq_temperature", 0.2),
            max_tokens=st.session_state.get("groq_max_tokens", 1024),
        )
        response_content = chat_completion.choices[0].message.content

        # Apply bold formatting to numbers AFTER getting the response
        # Use a more specific regex to avoid bolding numbers in links, dates etc unless desired
        # This regex aims to bold numbers that are likely counts or values.
        # It avoids numbers part of URLs, dates (dd/mm/yyyy), and percentages.
        # It also avoids numbers immediately followed by letters (like references).
        # Refined regex: Look for numbers not preceded by / or :, not followed by % or letters/dashes, possibly with comma/dot for decimals.
        response_content = re.sub(r'(?<![/:-\.])\b(\d+([,.]\d+)?)\b(?!\s*%|[a-zA-Z-])', r'**\1**', response_content)


        return response_content
    except Exception as e:
        debug_log(f"Erreur API Groq: {str(e)}", e)
        error_message = str(e).lower()
        if "authentication" in error_message or "api key" in error_message or "invalid api key" in error_message:
            return "Erreur d'authentification avec l'API Groq. Veuillez vérifier votre clé API dans la barre latérale."
        elif "rate limit" in error_message:
             return "Désolé, la limite de requêtes Groq est atteinte. Veuillez réessayer plus tard."
        elif "model" in error_message and "not found" in error_message:
             return f"Le modèle IA sélectionné ({st.session_state.get('groq_model', 'N/A')}) n'est pas disponible. Veuillez en choisir un autre dans la barre latérale."
        return f"Désolé, une erreur technique m'empêche de traiter votre demande: {str(e)}"


# --- Main App Logic ---
def main():
    # Initialisation du mode débogage
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False

    # Configuration Groq (avant le chargement pour avoir la clé disponible si nécessaire)
    groq_ready = manage_groq_api_key()
    groq_client = get_groq_client() if groq_ready else None

    create_header()

    st.sidebar.image(LOGO_URL, use_container_width=True, output_format='PNG') # Ensure correct format for logo
    st.sidebar.title("Navigation & Options")

    # Toggle pour activer/désactiver le mode debug
    st.sidebar.markdown("---")
    debug_expander = st.sidebar.expander("Options avancées de l'application", expanded=False): # Renamed expander
        st.session_state.debug_mode = st.checkbox("Mode débogage", value=st.session_state.debug_mode, help="Affiche des informations techniques pour le diagnostic des problèmes.")
        if st.session_state.debug_mode:
            st.info("Mode débogage activé. Des informations supplémentaires seront affichées en dessous des sections pertinentes.")
    st.sidebar.markdown("---") # Separator below advanced options

    # Initialisation des états de session par défaut
    default_session_keys = {
        'current_page_recalls': 1, 'items_per_page_filter': DEFAULT_ITEMS_PER_PAGE,
        'recent_days_filter': DEFAULT_RECENT_DAYS, 'date_filter_start': START_DATE,
        'date_filter_end': date.today(), 'search_term_main': "",
        'search_column_friendly_name_select': "Toutes les colonnes pertinentes",
        'groq_chat_history': [{"role": "assistant", "content": "Bonjour ! Posez-moi une question sur les données affichées ou utilisez une suggestion."}],
        'clicked_suggestion_query': None # State to handle suggestion button clicks
    }
    for key, value in default_session_keys.items():
        if key not in st.session_state: st.session_state[key] = value

    # Chargement des données depuis la nouvelle API v2
    with st.spinner("Chargement des données RappelConso (Alimentation)..."):
        df_alim = load_data(st.session_state.date_filter_start) # Use session start date for loading


    if df_alim.empty:
        st.error("Aucune donnée de rappel alimentaire n'a pu être chargée. Vérifiez les messages d'erreur ci-dessus ou réessayez.")
        # Display debug info even if data loading failed
        if st.session_state.debug_mode:
             st.write("Mode débogage: Aucune donnée chargée.")
        st.stop()

    # Mise à jour de la date de début de session si les données chargées sont plus anciennes
    if 'date_filter_start_init' not in st.session_state:
        min_data_date_actual = df_alim['date_publication'].dropna().min() if not df_alim['date_publication'].dropna().empty else START_DATE
        # Only update session state if the actual min date from data is earlier than the current session state start date
        if isinstance(min_data_date_actual, date) and min_data_date_actual < st.session_state.date_filter_start:
             st.session_state.date_filter_start = min_data_date_actual
             debug_log(f"Date de début du filtre mise à jour pour correspondre aux données chargées: {st.session_state.date_filter_start}", None)
        st.session_state.date_filter_start_init = True # Mark as initialized


    # Barre de recherche principale
    cols_search = st.columns([3,2])
    with cols_search[0]:
        # Add label for accessibility, but hide it visually
        st.session_state.search_term_main = st.text_input(
            "Rechercher un produit, marque, risque...", value=st.session_state.get('search_term_main', ""),
            placeholder="Ex: saumon, listeria, carrefour...", key="main_search_input", label_visibility="hidden" # Label hidden
        )
    with cols_search[1]:
        # Ensure column options match internal names used in filter_data
        search_column_options_friendly = ["Toutes les colonnes pertinentes"] + [k for k in FRIENDLY_TO_API_COLUMN_MAPPING.keys()]
        # Map friendly name back to internal name for filter_data
        selected_friendly_name = st.selectbox(
            "Chercher dans:", search_column_options_friendly,
            index=search_column_options_friendly.index(st.session_state.get('search_column_friendly_name_select', "Toutes les colonnes pertinentes")),
            key="main_search_column_select", label_visibility="hidden" # Label hidden
        )
        st.session_state.search_column_friendly_name_select = selected_friendly_name # Store friendly name in session state


    # Filtres avancés
    (selected_main_categories, selected_subcategories, selected_risks,
     selected_dates_tuple_main, items_per_page_setting) = create_advanced_filters(df_alim)

    # Déterminer la colonne de recherche interne pour filter_data
    search_column_internal = None
    if st.session_state.search_column_friendly_name_select != "Toutes les colonnes pertinentes":
        # Find the internal name corresponding to the selected friendly name
        search_column_internal = FRIENDLY_TO_API_COLUMN_MAPPING.get(st.session_state.search_column_friendly_name_select)


    # Filtrer les données selon les critères
    current_filtered_df = filter_data(
        df_alim, selected_subcategories, selected_risks, st.session_state.search_term_main,
        selected_dates_tuple_main, selected_main_categories, search_column_internal # Pass internal column name
    )

    # Affichage des onglets
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
            "<sub>L'IA répondra en se basant sur les rappels actuellement affichés. "
            "Les réponses sont des suggestions et doivent être vérifiées.</sub>",
            unsafe_allow_html=True
        )

        if not groq_ready:
            st.warning("Veuillez configurer votre clé API Groq dans la barre latérale pour utiliser l'assistant.", icon="⚠️")
        else:
            # Boutons de suggestions
            st.markdown("<div class='suggestion-button-container'>", unsafe_allow_html=True)
            suggestion_cols = st.columns(3)
            suggestion_queries = {
                "Tendance générale des rappels ?": {"type": "trend"},
                "Quels sont les 3 principaux risques ?": {"type": "context_only"},
                "Rappels pour 'fromage' ?": {"type": "context_specific", "product": "fromage"},
                 "Analyse des 'Listeria'": {"type": "trend", "risk": "Listeria"},
                 "Évolution des rappels 'Viande' ?": {"type": "trend", "product": "Viande"},
                 "Quels produits 'Bio' ont été rappelés ?": {"type": "context_specific", "product": "Bio"},
                 "Quels sont les principaux motifs de rappel ?": {"type": "context_only"}, # Added
                 "Combien de rappels par marque ?": {"type": "context_only"}, # Added
                 "Quel est le distributeur le plus concerné ?": {"type": "context_only"}, # Added
            }
            # Reset clicked_suggestion_query at the start of the iteration if it was processed in the previous run
            if 'clicked_suggestion_query_processed' in st.session_state:
                 del st.session_state.clicked_suggestion_query_processed

            idx = 0
            for query_text, params in suggestion_queries.items():
                with suggestion_cols[idx % len(suggestion_cols)]:
                    # Use a unique key for each button
                    button_key = f"suggestion_button_{idx}"
                    if st.button(query_text, key=button_key, use_container_width=True):
                        # Set the clicked query in session state
                        st.session_state.clicked_suggestion_query = query_text
                        # Mark that a suggestion was clicked to potentially trigger rerun
                        st.session_state.suggestion_clicked_flag = True
                        # Rerun immediately on suggestion click to process the query
                        st.rerun()
                idx += 1
            st.markdown("</div>", unsafe_allow_html=True)

            # Check if a suggestion flag was set and process it
            query_to_process = None
            if st.session_state.get('suggestion_clicked_flag'):
                query_to_process = st.session_state.clicked_suggestion_query
                # Reset the flag and the clicked query state after getting it
                del st.session_state.suggestion_clicked_flag
                # No need to del st.session_state.clicked_suggestion_query yet, it's used below

            # If no suggestion was clicked, check the chat input
            elif st.session_state.get('user_groq_query_input_main'):
                 query_to_process = st.session_state.user_groq_query_input_main
                 # Clear the input after getting the query
                 st.session_state.user_groq_query_input_main = "" # Clear the widget state


            # Historique de chat display container
            chat_display_container = st.container(height=450, border=False)
            with chat_display_container:
                # Display messages from history
                for message in st.session_state.groq_chat_history:
                    with st.chat_message(message["role"]):
                         st.markdown(message["content"], unsafe_allow_html=True)
                         # Display image if present in the message payload (from trend analysis)
                         if message["role"] == "assistant" and "graph_base64" in message and message["graph_base64"]:
                             st.image(f"data:image/png;base64,{message['graph_base64']}", caption="Tendance RappelConso")


            # Input utilisateur
            user_groq_query = st.chat_input("Posez votre question à l'IA...", key="user_groq_query_input_main", disabled=not groq_ready,
                                           value=st.session_state.get('user_groq_query_input_main_value', '')) # Use a helper value for persistence

            # If the chat input was used and is not empty, trigger processing
            if user_groq_query and not st.session_state.get('suggestion_clicked_flag'):
                 query_to_process = user_groq_query
                 # Update the helper value so input clears on rerun
                 st.session_state.user_groq_query_input_main_value = ''
                 st.rerun() # Trigger rerun to add user message and get AI response


            # Process the determined query
            if query_to_process:
                # Add user message to history and display it immediately
                st.session_state.groq_chat_history.append({"role": "user", "content": query_to_process})
                # Rerun needed to show the user message in the chat window before the spinner
                st.rerun()

            # If a rerun happened and the last message is the user's query, process it
            if st.session_state.groq_chat_history and st.session_state.groq_chat_history[-1]["role"] == "user":
                current_query = st.session_state.groq_chat_history[-1]["content"]
                with st.spinner("L'assistant IA réfléchit... 🤔"):
                    # --- IA Processing Logic ---
                    # Determine if trend analysis is needed based on query text or suggestion type
                    trend_analysis_needed = False
                    trend_product, trend_risk = None, None

                    # Check if the query originated from a suggestion button
                    originating_suggestion = next((params for q, params in suggestion_queries.items() if q == current_query), None)

                    if originating_suggestion:
                         if originating_suggestion.get("type") == "trend": trend_analysis_needed = True
                         trend_product = originating_suggestion.get("product")
                         trend_risk = originating_suggestion.get("risk")
                    # If not from a suggestion, analyze query text for keywords
                    elif any(k in current_query.lower() for k in ["tendance", "évolution", "statistique", "analyse de", "combien de rappel", "fréquence", "pic de", "augmentation", "diminution"]): # Expanded keywords
                        trend_analysis_needed = True
                        query_lower = current_query.lower()
                        # Attempt to extract potential product/risk from the query for focused trend analysis
                        possible_products = ["fromage", "viande", "bio", "poulet", "saumon", "lait", "produit laitier", "alimentaire", "poisson", "plat préparé", "fruit", "légume", "épicerie"] # Expanded list
                        possible_risks = ["listeria", "salmonelle", "e. coli", "allergène", "microbiologique", "corps étranger", "chimique", "physique"] # Expanded list
                        for p in possible_products:
                            if p in query_lower: trend_product = p; break
                        for r in possible_risks:
                            if r in query_lower: trend_risk = r; break
                        debug_log(f"Analyse de query pour tendance: query='{current_query}', trend_product='{trend_product}', trend_risk='{trend_risk}'", None)


                    trend_results = None
                    if trend_analysis_needed:
                        # Pass the CURRENTLY FILTERED data for trend analysis
                        trend_results = analyze_trends_data(current_filtered_df, product_type=trend_product, risk_type=trend_risk)
                        debug_log("Résultats analyse de tendance", trend_results)

                    # Prepare context data sample based on CURRENTLY FILTERED data
                    context_text_for_ai = prepare_context_for_ia(
                        current_filtered_df, max_items=st.session_state.get('groq_max_context_recalls', 15)
                    )

                    # Call the AI
                    ai_response_text = ask_groq_ai(groq_client, current_query, context_text_for_ai, trend_results)

                # --- End IA Processing Logic ---

                # Prepare the assistant message payload
                assistant_message = {"role": "assistant", "content": ai_response_text}
                # Add graph data to the message if trend analysis was successful and produced a graph
                if trend_results and trend_results.get("status") == "success" and trend_results.get("graph_base64"):
                     assistant_message["graph_base64"] = trend_results["graph_base64"]

                # Append assistant response to history
                st.session_state.groq_chat_history.append(assistant_message)
                # Trigger a rerun to display the new assistant message and graph
                st.rerun()


    # Informations et mise à jour des données (Sidebar)
    st.sidebar.markdown("---")
    # Display the actual date range of currently loaded data
    actual_min_data_date = df_alim['date_publication'].dropna().min() if not df_alim['date_publication'].dropna().empty else START_DATE
    actual_max_data_date = df_alim['date_publication'].dropna().max() if not df_alim['date_publication'].dropna().empty else date.today()
    st.sidebar.caption(f"Données RappelConso 'Alimentation'. **{len(df_alim)}** rappels chargés du {actual_min_data_date.strftime('%d/%m/%Y')} au {actual_max_data_date.strftime('%d/%m/%Y')}.")

    if st.sidebar.button("🔄 Mettre à jour les données", type="primary", use_container_width=True, key="update_data_btn"):
        st.cache_data.clear() # Clear the cache for load_data
        # Reset relevant session state keys that depend on loaded data
        keys_to_reset_on_update = [
             'date_filter_start_init', # Force re-initialization of start date
             'date_filter_start', 'date_filter_end', # Reset date filters to full new range
             'selected_categories_filter', 'selected_subcategories_filter', 'selected_risks_filter', # Clear category/risk filters
             'current_page_recalls' # Reset pagination
             # Keep search term and options, items per page, recent days filter
        ]
        for key_to_del in keys_to_reset_on_update:
             if key_to_del in st.session_state:
                 del st.session_state[key_to_del]

        # Rerun the app to reload data and re-initialize filters/state
        st.experimental_rerun()

    st.sidebar.markdown("---")
    st.sidebar.info("Projet open source par M00N69. Code disponible sur [GitHub](https://github.com/M00N69/RAPPELCONSO)") # Link to GitHub


if __name__ == "__main__":
    main()
