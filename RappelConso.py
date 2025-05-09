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

# CSS
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
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
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
            color: #0072C6;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .debug-section {
            margin: 5px 0;
            padding: 5px;
            background-color: #ffffff;
            border-left: 3px solid #0072C6;
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
# URL de base de l'API v2
API_BASE_URL = "https://data.economie.gouv.fr/api/v2/catalog/datasets/rappelconso-v2-gtin-espaces/records"
START_DATE = date(2022, 1, 1)
API_TIMEOUT_SEC = 30
DEFAULT_ITEMS_PER_PAGE = 6
DEFAULT_RECENT_DAYS = 30
LOGO_URL = "https://raw.githubusercontent.com/M00N69/RAPPELCONSO/main/logo%2004%20copie.jpg"

# Mappings des noms de colonnes entre l'interface et l'API
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

# Map des nouveaux noms de colonnes API v2 vers les anciens noms
API_V2_TO_V1_COLUMN_MAPPING = {
    "categorie_produit": "categorie_de_produit",
    "sous_categorie_produit": "sous_categorie_de_produit",
    "marque_produit": "nom_de_la_marque_du_produit",
    "motif_rappel": "motif_du_rappel",
    "zone_geographique_de_vente": "zone_geographique_de_vente",
    "modeles_ou_references": "modeles_ou_references",
    "identification_produits": "identification_produit",
    "conditionnements": "conditionnements",
    "date_debut_commercialisation": "date_debut_commercialisation",
    "date_date_fin_commercialisation": "date_fin_commercialisation",
    "temperature_conservation": "temperature_conservation", 
    "marque_salubrite": "marque_de_salubrite",
    "informations_complementaires": "informations_complementaires",
    "distributeurs": "distributeurs",
    "risques_encourus": "risques_encourus",
    "conduites_a_tenir_par_le_consommateur": "conduites_a_tenir",
    "numero_contact": "numero_contact",
    "modalites_de_compensation": "modalites_compensation",
    "liens_vers_les_images": "liens_vers_images",
    "lien_vers_la_liste_des_produits": "lien_vers_liste_des_produits",
    "lien_vers_la_liste_des_distributeurs": "lien_vers_liste_des_distributeurs",
    "lien_vers_affichette_pdf": "lien_vers_affichette_pdf",
    "lien_vers_la_fiche_rappel": "lien_vers_la_fiche_rappel",
    "date_publication": "date_publication",
    "libelle": "nom_commercial" # Nouveau libellé correspond généralement au nom commercial
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
                st.json(data)
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
            if 'date_publication' in df.columns:
                date_sample = df['date_publication'].iloc[0] if not df['date_publication'].empty else None
                st.write(f"Type de date_publication: {type(date_sample)}")
                st.write(f"Exemple de date: {date_sample}")

# --- Fonctions de chargement de données ---
@st.cache_data(show_spinner="Chargement des données RappelConso...", ttl=3600)
def load_data(start_date_filter=START_DATE):
    """Charge les données depuis l'API v2 de RappelConso."""
    all_records = []
    start_date_str = start_date_filter.strftime('%Y-%m-%d')
    today_str = date.today().strftime('%Y-%m-%d')
    
    # Construction des filtres avec la nouvelle syntaxe API v2
    where_clause = f"categorie_produit='Alimentation' AND date_publication >= '{start_date_str}' AND date_publication <= '{today_str}'"
    
    debug_log(f"Clause WHERE: {where_clause}")
    
    current_start_row = 0
    rows_per_page = 100  # Pagination interne
    total_hits_estimate = 0
    
    try:
        # Première requête pour obtenir le nombre total
        params = {
            'where': where_clause,
            'limit': 0  # Juste pour obtenir le total
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
        while current_start_row < total_hits_estimate:
            params = {
                'where': where_clause,
                'limit': rows_per_page,
                'offset': current_start_row
            }
            
            response = requests.get(API_BASE_URL, params=params, timeout=API_TIMEOUT_SEC)
            response.raise_for_status()
            data = response.json()
            
            # Extraction des records avec la nouvelle structure API v2
            records = data.get('records', [])
            
            if not records:
                break
                
            # Adapter à la nouvelle structure de réponse
            for rec in records:
                fields = rec.get('record', {}).get('fields', {})
                if fields:
                    all_records.append(fields)
            
            current_start_row += len(records)
            if total_hits_estimate > 0:
                st.toast(f"{len(all_records)}/{total_hits_estimate} rappels chargés...", icon="⏳")
            
            time.sleep(0.05)  # Pause pour éviter de surcharger l'API
            
    except requests.exceptions.HTTPError as http_err:
        st.error(f"Erreur HTTP de l'API: {http_err}")
        try:
            error_detail = response.json()
            st.error(f"Détails de l'erreur JSON: {error_detail}")
        except:
            st.error(f"Contenu brut de l'erreur: {response.text}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return pd.DataFrame()
    
    if not all_records:
        st.warning("Aucun rappel 'Alimentation' trouvé pour les critères spécifiés.")
        return pd.DataFrame()
    
    # Création du DataFrame
    df = pd.DataFrame(all_records)
    
    debug_log("DataFrame créé à partir des données API", df)
    
    # Renommage des colonnes pour correspondre à l'ancienne structure
    rename_cols = {k: v for k, v in API_V2_TO_V1_COLUMN_MAPPING.items() if k in df.columns}
    df = df.rename(columns=rename_cols)
    
    # Conversion et nettoyage des dates
    if 'date_publication' in df.columns:
        df['date_publication'] = pd.to_datetime(df['date_publication'], errors='coerce').dt.date
        df = df.dropna(subset=['date_publication'])
        df = df.sort_values(by='date_publication', ascending=False).reset_index(drop=True)
    
    # Ajout des colonnes manquantes
    for api_col_name in FRIENDLY_TO_API_COLUMN_MAPPING.values():
        if api_col_name not in df.columns:
            df[api_col_name] = pd.NA
    
    debug_log("DataFrame après prétraitement", df)
    
    return df

def filter_data(data_df, selected_subcategories, selected_risks, search_term, selected_dates_tuple, selected_categories, search_column_api_name=None):
    """Filtre les données selon les critères spécifiés."""
    if data_df.empty:
        return pd.DataFrame()
        
    filtered_df = data_df.copy()
    
    debug_log(f"Filtrage - données initiales: {len(filtered_df)} lignes")
    
    # Filtrage par catégorie
    if selected_categories:
        filtered_df = filtered_df[filtered_df['categorie_de_produit'].isin(selected_categories)]
        debug_log(f"Après filtre catégories ({selected_categories}): {len(filtered_df)} lignes")
    
    # Filtrage par sous-catégorie
    if selected_subcategories:
        filtered_df = filtered_df[filtered_df['sous_categorie_de_produit'].isin(selected_subcategories)]
        debug_log(f"Après filtre sous-catégories ({selected_subcategories}): {len(filtered_df)} lignes")
    
    # Filtrage par risque
    if selected_risks:
        if 'risques_encourus' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['risques_encourus'].fillna('').isin(selected_risks)]
            debug_log(f"Après filtre risques ({selected_risks}): {len(filtered_df)} lignes")
    
    # Filtrage par texte
    if search_term:
        search_term_lower = search_term.lower()
        if search_column_api_name and search_column_api_name in filtered_df.columns:
            # Recherche dans une colonne spécifique
            col_as_str = filtered_df[search_column_api_name].fillna('').astype(str)
            filtered_df = filtered_df[col_as_str.str.lower().str.contains(search_term_lower)]
            debug_log(f"Après recherche dans {search_column_api_name}='{search_term}': {len(filtered_df)} lignes")
        else:
            # Recherche dans plusieurs colonnes
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
                debug_log(f"Après recherche dans plusieurs colonnes '{search_term}': {len(filtered_df)} lignes")
    
    # Filtrage par date 
    if 'date_publication' in filtered_df.columns and not filtered_df.empty:
        start_filter_date = selected_dates_tuple[0]
        end_filter_date = selected_dates_tuple[1]
        
        if isinstance(start_filter_date, datetime): 
            start_filter_date = start_filter_date.date()
        if isinstance(end_filter_date, datetime): 
            end_filter_date = end_filter_date.date()
        
        debug_log(f"Filtrage par date: {start_filter_date} à {end_filter_date}")
            
        try:
            # Conversion uniforme en objet date si nécessaire
            if not pd.api.types.is_object_dtype(filtered_df['date_publication']):
                filtered_df['date_publication'] = pd.to_datetime(filtered_df['date_publication'], errors='coerce').dt.date
            
            # Filtre par date
            filtered_df = filtered_df[
                (filtered_df['date_publication'] >= start_filter_date) &
                (filtered_df['date_publication'] <= end_filter_date)
            ]
            debug_log(f"Après filtre date: {len(filtered_df)} lignes")
        except Exception as e:
            st.warning(f"Erreur lors du filtrage par date: {e}")
    
    return filtered_df


# --- Fonctions UI (Header, Métriques, Cartes Rappel, Pagination, Filtres Avancés) ---
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
    unique_subcategories = data_df['sous_categorie_de_produit'].nunique() if 'sous_categorie_de_produit' in data_df.columns else 0

    today = date.today()
    recent_days_filter = st.session_state.get('recent_days_filter', DEFAULT_RECENT_DAYS)
    days_ago = today - timedelta(days=recent_days_filter)
    
    recent_recalls = 0
    # Vérification plus robuste pour le calcul des métriques
    if 'date_publication' in data_df.columns and not data_df['date_publication'].dropna().empty:
        # S'assurer que la colonne est de type date pour la comparaison
        if not all(isinstance(d, date) for d in data_df['date_publication'].dropna()):
            temp_dates_metric = pd.to_datetime(data_df['date_publication'], errors='coerce').dt.date
        else:
            temp_dates_metric = data_df['date_publication']
        
        recent_recalls = len(data_df[~temp_dates_metric.isna() & (temp_dates_metric >= days_ago)])

    severe_percent = 0
    if 'risques_encourus' in data_df.columns and not data_df.empty:
        grave_keywords = ['microbiologique', 'listeria', 'salmonelle', 'allerg', 'toxique', 'e. coli', 'corps étranger', 'chimique']
        search_series = data_df['risques_encourus'].astype(str).str.lower()
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
            if isinstance(image_url, str) and image_url:
                image_url = image_url.split('|')[0] 
            else:
                image_url = "https://via.placeholder.com/150/CCCCCC/FFFFFF?Text=Image+ND"
            st.image(image_url, width=130)

        with col_content:
            # Utiliser nom_commercial s'il existe, sinon modeles_ou_references, sinon libelle
            product_name = row_data.get('nom_commercial', 
                           row_data.get('modeles_ou_references',
                           row_data.get('libelle', 'Produit non spécifié')))
            st.markdown(f"<h5>{product_name}</h5>", unsafe_allow_html=True)

            pub_date_obj = row_data.get('date_publication')
            formatted_date = pub_date_obj.strftime('%d/%m/%Y') if isinstance(pub_date_obj, date) else str(pub_date_obj)
            st.caption(f"Publié le: {formatted_date}")

            risk_text_raw = row_data.get('risques_encourus', 'Risque non spécifié')
            risk_text_lower = str(risk_text_raw).lower() 

            badge_color = "grey"; badge_icon = "⚠️"
            if any(keyword in risk_text_lower for keyword in ['listeria', 'salmonelle', 'e. coli', 'danger immédiat', 'toxique']):
                badge_color = "red"; badge_icon = "☠️"
            elif any(keyword in risk_text_lower for keyword in ['allergène', 'allergie', 'microbiologique', 'corps étranger', 'chimique']):
                badge_color = "orange"; badge_icon = "🔬"
            elif risk_text_raw != 'Risque non spécifié':
                badge_color = "darkgoldenrod"; badge_icon = "❗"

            st.markdown(f"**Risque {badge_icon}:** <span style='color:{badge_color}; font-weight:bold;'>{risk_text_raw}</span>", unsafe_allow_html=True)
            st.markdown(f"**Marque:** {row_data.get('nom_de_la_marque_du_produit', 'N/A')}")
            
            motif = row_data.get('motif_du_rappel', 'N/A')
            if len(str(motif)) > 100: 
                motif = str(motif)[:97] + "..."
            st.markdown(f"**Motif:** {motif}")

            distributeurs = row_data.get('distributeurs', 'N/A')
            if pd.notna(distributeurs) and distributeurs:
                 if len(str(distributeurs)) > 70: distributeurs = str(distributeurs)[:67] + "..."
                 st.markdown(f"**Distributeurs:** {distributeurs}")

            pdf_link = row_data.get('lien_vers_la_fiche_rappel', '#')
            if pdf_link and pdf_link != '#':
                 st.link_button("📄 Fiche de rappel", pdf_link, type="secondary", help="Ouvrir la fiche de rappel officielle")
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_paginated_recalls(data_df, items_per_page_setting):
    if data_df.empty:
        st.info("Aucun rappel ne correspond à vos critères de recherche.")
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

    debug_log(f"Pagination: page {current_page}/{total_pages}, affichage lignes {start_idx} à {end_idx-1}")

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
                st.rerun()
        
        with cols_pagination[1]: 
            st.markdown(f"<div style='text-align: center; margin-top: 10px;'>Page {current_page} sur {total_pages}</div>", unsafe_allow_html=True)

        with cols_pagination[2]: 
            if st.button("Suivant →", disabled=(current_page == total_pages), use_container_width=True, key="next_page_btn"):
                st.session_state.current_page_recalls += 1
                st.rerun()

def create_advanced_filters(df_full_data):
    # Utilisation de pd.NaT pour les dates non valides/manquantes
    min_date_data = df_full_data['date_publication'].dropna().min() if not df_full_data['date_publication'].dropna().empty else START_DATE
    max_date_data = date.today()

    debug_log(f"Date min dans les données: {min_date_data}, Date max: {max_date_data}")

    # Vérifier et corriger les types de date dans session_state au début
    if 'date_filter_start' not in st.session_state or not isinstance(st.session_state.date_filter_start, date):
        st.session_state.date_filter_start = min_date_data
    if 'date_filter_end' not in st.session_state or not isinstance(st.session_state.date_filter_end, date):
        st.session_state.date_filter_end = max_date_data
    
    # S'assurer que ce sont des objets date et non datetime
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
            else: 
                st.warning("Colonne 'categorie_de_produit' manquante")
                selected_categories = []

            if 'sous_categorie_de_produit' in df_full_data.columns:
                df_for_subcats = df_full_data[df_full_data['categorie_de_produit'].isin(selected_categories)] if selected_categories else df_full_data
                unique_subcategories = sorted(df_for_subcats['sous_categorie_de_produit'].dropna().unique())
                selected_subcategories = st.multiselect(
                    "Filtrer par Sous-catégories:", options=unique_subcategories,
                    default=st.session_state.get('selected_subcategories_filter', []),
                    key="sub_cat_filter"
                )
                st.session_state.selected_subcategories_filter = selected_subcategories
            else: 
                st.warning("Colonne 'sous_categorie_de_produit' manquante")
                selected_subcategories = []

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
            else: 
                st.warning("Colonne 'risques_encourus' manquante")
                selected_risks = []
            
            # Utiliser les dates de session comme valeur par défaut pour le widget
            selected_dates_tuple_local = st.date_input(
                "Filtrer par période de publication:",
                value=(st.session_state.date_filter_start, st.session_state.date_filter_end),
                min_value=min_date_data, max_value=max_date_data,
                key="date_range_picker_adv"
            )
            if len(selected_dates_tuple_local) == 2:
                 # Mettre à jour la session seulement si les dates du widget changent
                if selected_dates_tuple_local[0] != st.session_state.date_filter_start or selected_dates_tuple_local[1] != st.session_state.date_filter_end:
                    st.session_state.date_filter_start, st.session_state.date_filter_end = selected_dates_tuple_local
                    # Réinitialiser la page si la date change
                    st.session_state.current_page_recalls = 1


        st.markdown("---")
        st.markdown("**Options d'affichage**")
        items_per_page_setting = st.slider("Nombre de rappels par page:", min_value=2, max_value=20, 
                                 value=st.session_state.get('items_per_page_filter', DEFAULT_ITEMS_PER_PAGE), step=2,
                                 key="items_page_slider")
        if items_per_page_setting != st.session_state.get('items_per_page_filter', DEFAULT_ITEMS_PER_PAGE):
            st.session_state.items_per_page_filter = items_per_page_setting
            st.session_state.current_page_recalls = 1

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
                if key_to_del in st.session_state: 
                    del st.session_state[key_to_del]
            
            # Recalculer min_date_data au cas où df_full_data a été mis à jour
            min_date_data_reset = df_full_data['date_publication'].dropna().min() if not df_full_data['date_publication'].dropna().empty else START_DATE
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
    
    current_start_date_display = st.session_state.date_filter_start
    current_end_date_display = st.session_state.date_filter_end
    if current_start_date_display != min_date_data or current_end_date_display != max_date_data:
        active_filters_display.append(f"Période: {current_start_date_display.strftime('%d/%m/%y')} - {current_end_date_display.strftime('%d/%m/%y')}")

    if active_filters_display:
        st.markdown('<div class="filter-pills-container"><div class="filter-pills-title">Filtres actifs :</div><div class="filter-pills">' + 
                    ' '.join([f'<span class="filter-pill">{f}</span>' for f in active_filters_display]) + 
                    '</div></div>', unsafe_allow_html=True)

    # Retourner les dates de session actuelles
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

    st.markdown('<div class="chart-container" style="margin-top:1rem;">', unsafe_allow_html=True)
    
    # Prétraitement unifié des dates pour toutes les visualisations
    if 'date_publication' in data_df_viz.columns and not data_df_viz['date_publication'].dropna().empty:
        # Assurer que les dates sont du bon type
        if not all(isinstance(d, date) for d in data_df_viz['date_publication'].dropna()):
            data_df_viz = data_df_viz.copy()
            data_df_viz['date_publication_dt'] = pd.to_datetime(data_df_viz['date_publication'], errors='coerce')
        else:
            data_df_viz = data_df_viz.copy()
            data_df_viz['date_publication_dt'] = pd.to_datetime(data_df_viz['date_publication'])
        
        data_df_viz = data_df_viz.dropna(subset=['date_publication_dt'])
        if data_df_viz.empty:
            st.warning("Dates invalides dans les données, impossible de générer les visualisations temporelles.")
            st.markdown('</div>', unsafe_allow_html=True)
            return
    else:
        st.warning("Colonne 'date_publication' manquante ou vide pour les visualisations temporelles.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    debug_log("Données pour visualisations", data_df_viz.head(3))

    tab1, tab2, tab3 = st.tabs(["📊 Tendances Temporelles", "🍩 Répartitions par Catégorie", "☠️ Analyse des Risques"])

    with tab1:
        st.subheader("Évolution des Rappels")
        if 'date_publication_dt' in data_df_viz.columns and not data_df_viz['date_publication_dt'].empty:
            monthly_data = data_df_viz.set_index('date_publication_dt').resample('M').size().reset_index(name='count')
            monthly_data['year_month'] = monthly_data['date_publication_dt'].dt.strftime('%Y-%m')
            monthly_data = monthly_data.sort_values('year_month')

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
                cat_counts = data_df_viz['categorie_de_produit'].value_counts().nlargest(10)
                if not cat_counts.empty:
                    fig_cat = px.pie(values=cat_counts.values, names=cat_counts.index, title="Top 10 Catégories", hole=0.4)
                    fig_cat.update_traces(textposition='outside', textinfo='percent+label')
                    st.plotly_chart(fig_cat, use_container_width=True)
                else: st.info("Pas de données pour la répartition par catégorie.")
            else: st.warning("Colonne 'categorie_de_produit' manquante.")
        with col_subcat:
            if 'sous_categorie_de_produit' in data_df_viz.columns:
                subcat_counts = data_df_viz['sous_categorie_de_produit'].value_counts().nlargest(10)
                if not subcat_counts.empty:
                    fig_subcat = px.pie(values=subcat_counts.values, names=subcat_counts.index, title="Top 10 Sous-Catégories", hole=0.4)
                    fig_subcat.update_traces(textposition='outside', textinfo='percent+label')
                    st.plotly_chart(fig_subcat, use_container_width=True)
                else: st.info("Pas de données pour la répartition par sous-catégorie.")
            else: st.warning("Colonne 'sous_categorie_de_produit' manquante.")
        
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
        if new_key != st.session_state.user_groq_api_key:
             st.session_state.user_groq_api_key = new_key
             if new_key and new_key.startswith("gsk_"): st.success("Clé API Groq enregistrée.", icon="👍")
             elif new_key: st.warning("Format de clé API invalide.", icon="⚠️")
             st.rerun() 

        model_options = {
            "llama3-70b-8192": "Llama 3 (70B) - Puissant", "llama3-8b-8192": "Llama 3 (8B) - Rapide",
            "mixtral-8x7b-32768": "Mixtral (8x7B) - Large contexte", "gemma-7b-it": "Gemma (7B) - Léger"
        }
        selected_model_key = st.selectbox(
            "Choisir un modèle IA:", options=list(model_options.keys()),
            format_func=lambda x: model_options[x], index=0, 
            key="groq_model_select_sidebar"
        )
        st.session_state.groq_model = selected_model_key
        
        with st.popover("Options avancées de l'IA"):
            st.session_state.groq_temperature = st.slider("Température:", 0.0, 1.0, st.session_state.get('groq_temperature', 0.2), 0.1)
            st.session_state.groq_max_tokens = st.slider("Tokens max réponse:", 256, 4096, st.session_state.get('groq_max_tokens', 1024), 256)
            st.session_state.groq_max_context_recalls = st.slider("Max rappels dans contexte IA:", 5, 50, st.session_state.get('groq_max_context_recalls', 15), 1)

    if st.session_state.user_groq_api_key and st.session_state.user_groq_api_key.startswith("gsk_"):
        st.sidebar.caption(f"🟢 IA prête ({model_options.get(st.session_state.groq_model, 'N/A')})")
        return True
    else:
        st.sidebar.caption("🔴 IA non configurée ou clé invalide.")
        return False

def get_groq_client():
    if "user_groq_api_key" in st.session_state and st.session_state.user_groq_api_key:
        try:
            return Groq(api_key=st.session_state.user_groq_api_key)
        except Exception as e:
            st.error(f"Erreur d'initialisation du client Groq: {e}. Vérifiez votre clé.")
            return None
    return None

def prepare_context_for_ia(df_context, max_items=10):
    """Prépare les données de contexte pour l'IA Groq de manière robuste."""
    if df_context.empty:
        return "Aucune donnée de rappel pertinente trouvée pour cette question avec les filtres actuels."

    # Version améliorée avec plus de robustesse
    cols_for_ia = [
        'nom_de_la_marque_du_produit', 'nom_commercial', 'modeles_ou_references', 
        'categorie_de_produit', 'sous_categorie_de_produit', 
        'risques_encourus', 'motif_du_rappel', 'date_publication', 'distributeurs'
    ]
    
    # Vérifier quelles colonnes sont présentes
    cols_to_use = [col for col in cols_for_ia if col in df_context.columns]
    
    if not cols_to_use:
        return "Structure de données incompatible pour l'analyse IA."
    
    actual_max_items = min(max_items, len(df_context))
    context_df_sample = df_context[cols_to_use].head(actual_max_items)
    
    text_context = f"Voici un échantillon de {len(context_df_sample)} rappels (sur {len(df_context)} au total avec les filtres) :\n\n"
    
    for _, row in context_df_sample.iterrows():
        item_desc = []
        
        # Construction plus robuste avec vérification des NaN
        for col in cols_to_use:
            if pd.notna(row.get(col)):
                # Formater la date si c'est une date
                if col == 'date_publication' and isinstance(row[col], date):
                    value = row[col].strftime('%d/%m/%y')
                else:
                    value = str(row[col])
                    if len(value) > 70 and col in ['motif_du_rappel', 'distributeurs']:
                        value = value[:67] + "..."
                
                # Mapping des noms de colonnes pour l'affichage
                display_name = {
                    'nom_de_la_marque_du_produit': 'Marque',
                    'nom_commercial': 'Produit',
                    'modeles_ou_references': 'Ref',
                    'categorie_de_produit': 'Cat',
                    'sous_categorie_de_produit': 'Sous-cat',
                    'risques_encourus': 'Risque',
                    'motif_du_rappel': 'Motif',
                    'date_publication': 'Date',
                    'distributeurs': 'Distrib'
                }.get(col, col)
                
                item_desc.append(f"{display_name}: {value}")
        
        text_context += "- " + ", ".join(item_desc) + "\n"
            
    return text_context

def analyze_trends_data(df_analysis, product_type=None, risk_type=None, time_period="12m"):
    """Analyse les tendances dans les données de rappel."""
    if df_analysis.empty:
        return {"status": "no_data", "message": "Aucune donnée disponible pour l'analyse de tendance."}

    df_filtered = df_analysis.copy()
    
    # S'assurer que la colonne date_publication existe et conversion uniforme des dates
    if 'date_publication' in df_filtered.columns:
        # Vérifier et convertir les dates si nécessaire
        if not all(isinstance(d, date) for d in df_filtered['date_publication'].dropna()):
            df_filtered['date_publication_dt'] = pd.to_datetime(df_filtered['date_publication'], errors='coerce')
        else:
            df_filtered['date_publication_dt'] = pd.to_datetime(df_filtered['date_publication'])
        
        df_filtered = df_filtered.dropna(subset=['date_publication_dt'])
        if df_filtered.empty: 
            return {"status": "no_data", "message": "Dates invalides pour l'analyse."}
    else: 
        return {"status": "error", "message": "Colonne 'date_publication' manquante."}
    
    period_label = f"du {df_filtered['date_publication_dt'].min().strftime('%d/%m/%Y')} au {df_filtered['date_publication_dt'].max().strftime('%d/%m/%Y')}"
    analysis_title_parts = ["Évolution des rappels"]

    # Filtrage des données si des critères spécifiques sont fournis
    if product_type:
        mask_product = (
            df_filtered['sous_categorie_de_produit'].astype(str).str.contains(product_type, case=False, na=False) |
            df_filtered['nom_commercial'].astype(str).str.contains(product_type, case=False, na=False) |
            df_filtered['categorie_de_produit'].astype(str).str.contains(product_type, case=False, na=False)
        )
        df_filtered = df_filtered[mask_product]
        analysis_title_parts.append(f"pour '{product_type}'")
    if risk_type:
        mask_risk = (
            df_filtered['risques_encourus'].astype(str).str.contains(risk_type, case=False, na=False) |
            df_filtered['motif_du_rappel'].astype(str).str.contains(risk_type, case=False, na=False)
        )
        df_filtered = df_filtered[mask_risk]
        analysis_title_parts.append(f"avec risque/motif '{risk_type}'")
    
    if df_filtered.empty:
        return {"status": "no_data", "message": f"Aucune donnée pour les filtres spécifiques ({product_type or 'tous produits'}, {risk_type or 'tous risques'}) sur la période."}

    # Regroupement par mois pour l'analyse temporelle
    monthly_counts = df_filtered.set_index('date_publication_dt').resample('M').size()
    if monthly_counts.empty: 
        return {"status": "no_data", "message": "Pas de données mensuelles à analyser après filtrage."}
    
    # Compléter les mois manquants avec des zéros pour une visualisation cohérente
    idx = pd.date_range(monthly_counts.index.min(), monthly_counts.index.max(), freq='MS')
    monthly_counts = monthly_counts.reindex(idx, fill_value=0)
    monthly_counts_display = monthly_counts.copy()
    monthly_counts_display.index = monthly_counts_display.index.strftime('%Y-%m')

    # Calculer les statistiques pour l'analyse
    trend_stats = {"total_recalls": int(df_filtered.shape[0]), "monthly_avg": float(monthly_counts.mean())}
    
    # Analyse de tendance (régression linéaire)
    slope = 0
    if len(monthly_counts) >= 2:
        X = np.arange(len(monthly_counts)).reshape(-1, 1)
        y = monthly_counts.values
        model = LinearRegression().fit(X, y)
        slope = float(model.coef_[0])
        trend_stats['trend_slope'] = slope
        if slope > 0.1: trend_stats['trend_direction'] = "hausse"
        elif slope < -0.1: trend_stats['trend_direction'] = "baisse"
        else: trend_stats['trend_direction'] = "stable"
    else: 
        trend_stats['trend_direction'] = "indéterminée (données insuffisantes)"

    # Création du graphique
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly_counts_display.index, monthly_counts.values, marker='o', linestyle='-', label='Rappels/mois')
    
    # Ajouter la ligne de tendance si suffisamment de données
    if len(monthly_counts) >= 2 and 'trend_direction' in trend_stats and trend_stats['trend_direction'] != "indéterminée (données insuffisantes)":
        trend_line = model.predict(X)
        ax.plot(monthly_counts_display.index, trend_line, color='red', linestyle='--', label=f'Tendance ({trend_stats["trend_direction"]})')
    
    # Mise en forme du graphique
    ax.set_title(' '.join(analysis_title_parts), fontsize=10)
    ax.set_xlabel("Mois", fontsize=8); ax.set_ylabel("Nombre de rappels", fontsize=8)
    ax.tick_params(axis='x', rotation=45, labelsize=7); ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, linestyle=':', alpha=0.7); ax.legend(fontsize=8)
    plt.tight_layout()
    
    # Conversion du graphique en image base64 pour l'affichage
    buf = io.BytesIO()
    plt.savefig(buf, format="png"); buf.seek(0)
    graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    # Rédaction du résumé textuel de l'analyse
    text_summary = f"Analyse de tendance ({period_label}):\n"
    text_summary += f"- **Total rappels analysés**: {trend_stats['total_recalls']}\n"
    text_summary += f"- **Moyenne mensuelle**: {trend_stats['monthly_avg']:.1f} rappels\n"
    if 'trend_direction' in trend_stats:
        text_summary += f"- **Tendance générale**: {trend_stats['trend_direction']}"
        if trend_stats['trend_direction'] not in ["indéterminée (données insuffisantes)", "stable"]:
             text_summary += f" (pente: {slope:.2f})\n"
        else:
            text_summary += "\n"
    
    # Ajouter des informations supplémentaires sur les sous-catégories et les risques
    if 'sous_categorie_de_produit' in df_filtered.columns:
        top_cat = df_filtered['sous_categorie_de_produit'].dropna().value_counts().nlargest(3)
        if not top_cat.empty: text_summary += "- **Top 3 sous-catégories:** " + ", ".join([f"{idx} ({val})" for idx, val in top_cat.items()]) + "\n"
    if 'risques_encourus' in df_filtered.columns:
        top_risk = df_filtered['risques_encourus'].dropna().value_counts().nlargest(3)
        if not top_risk.empty: text_summary += "- **Top 3 risques:** " + ", ".join([f"{idx} ({val})" for idx, val in top_risk.items()]) + "\n"

    return {
        "status": "success", "text_summary": text_summary, "graph_base64": graph_base64,
        "monthly_data": {str(k): int(v) for k,v in monthly_counts_display.to_dict().items()},
        "trend_stats": trend_stats
    }


def ask_groq_ai(client, user_query, context_data_text, trend_analysis_results=None):
    """Interroge l'API Groq avec le contexte des données et les résultats d'analyse."""
    if not client: 
        return "Client Groq non initialisé. Vérifiez votre clé API."

    debug_log("Préparation requête Groq", {
        "taille du contexte": len(context_data_text),
        "analyse de tendance": trend_analysis_results["status"] if trend_analysis_results else "aucune"
    })

    system_prompt = f"""Tu es "RappelConso Insight Assistant", un expert IA spécialisé dans l'analyse des données de rappels de produits alimentaires en France, basé sur les données de RappelConso.
    Date actuelle: {date.today().strftime('%d/%m/%Y')}.
    Ta mission est de répondre aux questions de l'utilisateur de manière concise, professionnelle et en te basant STRICTEMENT sur les informations et données de contexte fournies.
    NE PAS INVENTER d'informations. Si les données ne te permettent pas de répondre, indique-le clairement (ex: "Je n'ai pas cette information dans les données fournies.").
    Utilise le Markdown pour mettre en **gras** les chiffres clés et les points importants.
    Si une analyse de tendance a été effectuée et qu'un graphique est disponible (indiqué dans le contexte), mentionne-le et décris brièvement ce qu'il montre en t'appuyant sur le résumé d'analyse fourni.
    Si la question est hors sujet (ne concerne pas les rappels de produits alimentaires, la sécurité alimentaire, ou les données fournies), réponds avec une blague COURTE et pertinente sur la sécurité alimentaire ou la nourriture, puis indique que tu ne peux pas répondre à la question.
    Exemple de blague : "Pourquoi le steak haché a-t-il rompu avec la salade ? Parce qu'elle était trop vinaigrette ! Plus sérieusement, cette question sort de mon domaine d'expertise sur RappelConso."
    Sois toujours courtois.
    """
    
    full_context_for_ai = f"Contexte des rappels de produits (échantillon et filtres actuels):\n{context_data_text}\n\n"

    if trend_analysis_results and trend_analysis_results["status"] == "success":
        full_context_for_ai += f"Une analyse de tendance a été effectuée. Voici son résumé:\n{trend_analysis_results['text_summary']}\n"
        full_context_for_ai += "Un graphique illustrant cette tendance est disponible et sera affiché à l'utilisateur si pertinent.\n"

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
        
        # Mettre en gras certains chiffres pour une meilleure lisibilité
        response_content = re.sub(r'(\b\d{1,3}(?:[,.]\d{1,2})?\b)(?!%|[-\w])', r'**\1**', response_content) 
        return response_content
    except Exception as e:
        debug_log(f"Erreur Groq: {str(e)}")
        if "authentication" in str(e).lower() or "api key" in str(e).lower():
            return "Erreur d'authentification avec l'API Groq. Vérifiez votre clé API."
        return f"Désolé, une erreur technique m'empêche de traiter votre demande: {str(e)}"


# --- Main App Logic ---
def main():
    # Initialisation du mode débogage
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    
    create_header()
    
    st.sidebar.image(LOGO_URL, use_container_width=True)
    st.sidebar.title("Navigation & Options")
    
    # Toggle pour activer/désactiver le mode debug
    st.sidebar.markdown("---")
    debug_expander = st.sidebar.expander("Options avancées", expanded=False)
    with debug_expander:
        st.session_state.debug_mode = st.checkbox("Mode débogage", value=st.session_state.debug_mode)
        if st.session_state.debug_mode:
            st.info("Mode débogage activé. Des informations supplémentaires seront affichées.")
    
    # Configuration Groq
    groq_ready = manage_groq_api_key()
    groq_client = get_groq_client() if groq_ready else None

    # Initialisation des états de session
    default_session_keys = {
        'current_page_recalls': 1, 'items_per_page_filter': DEFAULT_ITEMS_PER_PAGE,
        'recent_days_filter': DEFAULT_RECENT_DAYS, 'date_filter_start': START_DATE,
        'date_filter_end': date.today(), 'search_term_main': "",
        'search_column_friendly_name_select': "Toutes les colonnes pertinentes"
    }
    for key, value in default_session_keys.items():
        if key not in st.session_state: st.session_state[key] = value

    # Chargement des données depuis la nouvelle API v2
    with st.spinner("Chargement des données RappelConso..."):
        df_alim = load_data(START_DATE)
    
    if df_alim.empty:
        st.error("Aucune donnée de rappel alimentaire n'a pu être chargée. Vérifiez les messages d'erreur ci-dessus ou réessayez.")
        st.stop()
    
    # Mise à jour de la date de début
    if 'date_filter_start_init' not in st.session_state:
        min_data_date_actual = df_alim['date_publication'].dropna().min() if not df_alim['date_publication'].dropna().empty else START_DATE
        if isinstance(min_data_date_actual, date): st.session_state.date_filter_start = min_data_date_actual
        st.session_state.date_filter_start_init = True

    # Barre de recherche principale
    cols_search = st.columns([3,2])
    with cols_search[0]:
        st.session_state.search_term_main = st.text_input(
            "Rechercher un produit, marque, risque...", value=st.session_state.get('search_term_main', ""),
            placeholder="Ex: saumon, listeria, carrefour...", key="main_search_input", label_visibility="collapsed"
        )
    with cols_search[1]:
        search_column_options_friendly = ["Toutes les colonnes pertinentes"] + list(FRIENDLY_TO_API_COLUMN_MAPPING.keys())
        st.session_state.search_column_friendly_name_select = st.selectbox(
            "Chercher dans:", search_column_options_friendly, 
            index=search_column_options_friendly.index(st.session_state.get('search_column_friendly_name_select', "Toutes les colonnes pertinentes")),
            key="main_search_column_select", label_visibility="collapsed"
        )

    # Filtres avancés
    (selected_main_categories, selected_subcategories, selected_risks, 
     selected_dates_tuple_main, items_per_page_setting) = create_advanced_filters(df_alim)

    # Déterminer la colonne de recherche
    search_column_api = None
    if st.session_state.search_column_friendly_name_select != "Toutes les colonnes pertinentes":
        search_column_api = FRIENDLY_TO_API_COLUMN_MAPPING.get(st.session_state.search_column_friendly_name_select)

    # Filtrer les données selon les critères
    current_filtered_df = filter_data(
        df_alim, selected_subcategories, selected_risks, st.session_state.search_term_main,
        selected_dates_tuple_main, selected_main_categories, search_column_api
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
            st.warning("Veuillez configurer votre clé API Groq dans la barre latérale.", icon="⚠️")
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
            }
            if 'clicked_suggestion_query' not in st.session_state: 
                st.session_state.clicked_suggestion_query = None
            
            idx = 0
            for query_text, params in suggestion_queries.items():
                with suggestion_cols[idx % len(suggestion_cols)]:
                    if st.button(query_text, key=f"suggestion_{idx}", use_container_width=True):
                        st.session_state.clicked_suggestion_query = query_text
                idx += 1
            st.markdown("</div>", unsafe_allow_html=True)

            # Historique de chat
            if "groq_chat_history" not in st.session_state:
                st.session_state.groq_chat_history = [{"role": "assistant", "content": "Bonjour ! Posez-moi une question sur les données affichées ou utilisez une suggestion."}]

            # Affichage des messages existants
            chat_display_container = st.container(height=450, border=False)
            with chat_display_container:
                for message in st.session_state.groq_chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"], unsafe_allow_html=True)
                        if message["role"] == "assistant" and "graph_base64" in message and message["graph_base64"]:
                            st.image(f"data:image/png;base64,{message['graph_base64']}")
            
            # Input utilisateur
            user_groq_query = st.chat_input("Posez votre question à l'IA...", key="user_groq_query_input_main", disabled=not groq_ready)
            
            # Traitement des requêtes (suggestion ou input utilisateur)
            query_to_process = None
            if st.session_state.clicked_suggestion_query:
                query_to_process = st.session_state.clicked_suggestion_query
                st.session_state.clicked_suggestion_query = None 
            elif user_groq_query:
                query_to_process = user_groq_query

            if query_to_process:
                st.session_state.groq_chat_history.append({"role": "user", "content": query_to_process})
                with chat_display_container: 
                     with st.chat_message("user"): st.markdown(query_to_process)

                with st.spinner("L'assistant IA réfléchit... 🤔"):
                    # Préparation du contexte
                    context_text_for_ai = prepare_context_for_ia(
                        current_filtered_df, max_items=st.session_state.get('groq_max_context_recalls', 15)
                    )
                    
                    trend_analysis_needed = False
                    trend_product, trend_risk = None, None

                    # Déterminer si une analyse de tendance est nécessaire
                    if query_to_process in suggestion_queries: 
                        params = suggestion_queries[query_to_process]
                        if params.get("type") == "trend": trend_analysis_needed = True
                        trend_product = params.get("product")
                        trend_risk = params.get("risk")
                    elif any(k in query_to_process.lower() for k in ["tendance", "évolution", "statistique", "analyse de", "combien de rappel"]):
                        trend_analysis_needed = True
                        query_lower = query_to_process.lower()
                        possible_products = ["fromage", "viande", "bio", "poulet", "saumon", "lait"] 
                        possible_risks = ["listeria", "salmonelle", "e. coli", "allergène"] 
                        for p in possible_products:
                            if p in query_lower: trend_product = p; break
                        for r in possible_risks:
                            if r in query_lower: trend_risk = r; break
                    
                    trend_results = None
                    if trend_analysis_needed:
                        trend_results = analyze_trends_data(current_filtered_df, product_type=trend_product, risk_type=trend_risk)
                    
                    # Appel à l'IA avec le contexte préparé
                    ai_response_text = ask_groq_ai(groq_client, query_to_process, context_text_for_ai, trend_results)
                
                # Préparation de la réponse avec graphique si disponible
                assistant_message = {"role": "assistant", "content": ai_response_text}
                if trend_results and trend_results["status"] == "success" and "graph_base64" in trend_results:
                    assistant_message["graph_base64"] = trend_results["graph_base64"]
                
                st.session_state.groq_chat_history.append(assistant_message)
                st.rerun()

    # Informations et mise à jour des données
    st.sidebar.markdown("---")
    start_date_display = st.session_state.get('date_filter_start', START_DATE)
    st.sidebar.caption(f"Données RappelConso 'Alimentation'. {len(df_alim)} rappels depuis {start_date_display.strftime('%d/%m/%Y')}.")
    if st.sidebar.button("🔄 Mettre à jour les données", type="primary", use_container_width=True, key="update_data_btn"):
        st.cache_data.clear() 
        if 'date_filter_start_init' in st.session_state: del st.session_state.date_filter_start_init
        st.experimental_rerun()

if __name__ == "__main__":
    main()
