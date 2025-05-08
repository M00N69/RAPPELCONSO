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

# CSS (Repris de votre code original, potentiellement avec de légères ajustements pour cohérence)
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
# NOTE: API_URL_BASE is just the base path. The dataset and query part are added in load_data.
API_URL_BASE = "https://data.economie.gouv.fr/api/records/1.0/search/" 
# The full URL structure needed by load_data is: API_URL_BASE + ?dataset=...&q=
API_URL_FOR_LOAD = "https://data.economie.gouv.fr/api/records/1.0/search/?dataset=rappelconso-v2-gtin-espaces&q="

START_DATE = date(2022, 1, 1) # Date de début pour le chargement initial
API_TIMEOUT_SEC = 30
DEFAULT_ITEMS_PER_PAGE = 6
DEFAULT_RECENT_DAYS = 30
LOGO_URL = "https://raw.githubusercontent.com/M00N69/RAPPELCONSO/main/logo%2004%20copie.jpg"


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


# --- Fonctions de chargement de données (CORRIGÉE) ---
@st.cache_data(show_spinner="Chargement des données RappelConso...", ttl=3600)
def load_data(api_url_base_plus_dataset, start_date_filter=START_DATE): 
    """Charge les données depuis l'API en utilisant la méthode de filtrage par refine de l'URL."""
    all_records = []
    start_date_str = start_date_filter.strftime('%Y-%m-%d')
    today_str = date.today().strftime('%Y-%m-%d')

    # --- Construction de l'URL corrigée ---
    # Ajouter les filtres refine directement à l'URL de base
    # L'opérateur (>=, <=) doit être après le champ, la valeur (date) est encodée
    api_url_filtered = (
        f"{api_url_base_plus_dataset}" # Ex: "...?dataset=rappelconso-v2-gtin-espaces&q="
        f"&refine.date_publication:>={urllib.parse.quote(start_date_str)}" # Correct: Opérateur direct, seule la date est encodée
        f"&refine.date_publication:<={urllib.parse.quote(today_str)}"   # Correct: Opérateur direct, seule la date est encodée
        f"&refine.categorie_de_produit=Alimentation" # Filtre pour n'obtenir que l'alimentation
    )
    # -----------------------------------------------------------

    current_start_row = 0
    total_hits_estimate = 0
    rows_per_page = 1000 # Pagination interne

    while True:
        # Ajouter la pagination (&start=... &rows=...) à l'URL déjà filtrée
        paginated_url = f"{api_url_filtered}&start={current_start_row}&rows={rows_per_page}"

        try:
            response = requests.get(paginated_url, timeout=API_TIMEOUT_SEC)
            response.raise_for_status()
            data = response.json()

            if current_start_row == 0:
                total_hits_estimate = data.get('nhits', 0)

            records = data.get('records')

            if records:
                all_records.extend([rec['fields'] for rec in records])
                if total_hits_estimate > 0:
                    # Affichage plus discret du chargement
                    if current_start_row == 0:
                        st.toast(f"Début du chargement...", icon="⏳")
                    else:
                        st.toast(f"{len(all_records)} / ~{total_hits_estimate} rappels chargés...", icon="⏳")

                current_start_row += len(records)
                # Condition de sortie améliorée : soit on a atteint le total estimé, soit la dernière page est incomplète
                # Ajout d'une limite de sécurité pour éviter des boucles infinies si nhits est incorrect
                if (total_hits_estimate > 0 and len(all_records) >= total_hits_estimate) or (total_hits_estimate > 0 and len(records) < rows_per_page) or (total_hits_estimate == 0 and len(records) < rows_per_page and current_start_row > 0) or current_start_row > 500000:
                     if total_hits_estimate > 0 and len(all_records) < total_hits_estimate:
                         st.warning(f"Chargement incomplet : {len(all_records)} / {total_hits_estimate} chargés. Une erreur a pu se produire ou la pagination est terminée.", icon="⚠️")
                     else:
                         st.toast(f"Chargement terminé. {len(all_records)} rappels chargés.", icon="✅")
                     break
            else:
                # Aucun enregistrement dans la réponse, fin de la pagination
                st.toast(f"Chargement terminé. {len(all_records)} rappels chargés.", icon="✅")
                break

            time.sleep(0.05) # Petite pause pour ne pas surcharger l'API

        except requests.exceptions.HTTPError as http_err:
            st.error(f"Erreur HTTP de l'API: {http_err}")
            st.error(f"URL de la requête ayant échoué: {paginated_url}")
            try:
                error_detail = response.json()
                st.error(f"Détails de l'erreur JSON: {error_detail}")
            except requests.exceptions.JSONDecodeError:
                 st.error(f"Contenu brut de l'erreur: {response.text}")
            return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur de requête API: {e}")
            return pd.DataFrame()
        except KeyError as e:
            st.error(f"Erreur de structure JSON de l'API: clé manquante {e}")
            return pd.DataFrame()
        except requests.exceptions.Timeout:
            st.error("Délai d'attente dépassé lors de la requête à l'API.")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Une erreur inattendue est survenue pendant le chargement: {e}")
            return pd.DataFrame()


    if not all_records:
        st.warning("Aucun rappel 'Alimentation' trouvé pour les critères spécifiés ou pendant le chargement.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    if 'date_publication' in df.columns:
        # Utiliser errors='coerce' est bien, mais vérifier si des valeurs non-NaT existent après
        df['date_publication'] = pd.to_datetime(df['date_publication'], errors='coerce').dt.date
        df = df.dropna(subset=['date_publication'])
        if not df.empty:
             df = df.sort_values(by='date_publication', ascending=False).reset_index(drop=True)
        else:
             st.warning("Toutes les dates de publication sont invalides après conversion. Le tri et les filtres de date peuvent être affectés.")
             # Retourner un DataFrame vide si aucune date valide n'est trouvée pour éviter les erreurs plus loin
             return pd.DataFrame()
    else:
        st.warning("La colonne 'date_publication' est manquante dans les données de l'API. Le tri et certains filtres pourraient ne pas fonctionner.")
        # Créer la colonne avec des valeurs nulles si elle n'existe pas
        df['date_publication'] = pd.NaT # Assigne des valeurs NaT (Not a Time)


    # Assurer l'existence des colonnes attendues même si elles sont vides
    # Remplacer les valeurs nulles (None) par pd.NA/pd.NA pour une meilleure compatibilité pandas
    for api_col_name in FRIENDLY_TO_API_COLUMN_MAPPING.values():
        if api_col_name not in df.columns:
            df[api_col_name] = pd.NA
        # Remplacer les listes vides [] par pd.NA car elles peuvent causer des erreurs
        if df[api_col_name].apply(lambda x: isinstance(x, list) and not x).any():
             df[api_col_name] = df[api_col_name].apply(lambda x: pd.NA if isinstance(x, list) and not x else x)
        # Remplacer les None par pd.NA
        if df[api_col_name].isnull().any():
             df[api_col_name] = df[api_col_name].replace({None: pd.NA})

    return df


# --- Fonctions de filtrage ( inchangée) ---
def filter_data(data_df, selected_subcategories, selected_risks, search_term, selected_dates_tuple, selected_categories, search_column_api_name=None):
    # Le filtrage par date via l'API est maintenant géré dans load_data.
    # On garde le filtrage Python pour les autres critères.
    filtered_df = data_df.copy()

    # Assurer que les colonnes existent avant de filtrer
    if 'categorie_de_produit' in filtered_df.columns and selected_categories:
        filtered_df = filtered_df[filtered_df['categorie_de_produit'].isin(selected_categories)]
    if 'sous_categorie_de_produit' in filtered_df.columns and selected_subcategories:
        filtered_df = filtered_df[filtered_df['sous_categorie_de_produit'].isin(selected_subcategories)]
    if 'risques_encourus' in filtered_df.columns and selected_risks:
        # Utiliser .fillna('') pour gérer les NaN dans la colonne 'risques_encourus'
        filtered_df = filtered_df[filtered_df['risques_encourus'].fillna('').isin(selected_risks)]

    if search_term:
        search_term_lower = search_term.lower()
        if search_column_api_name and search_column_api_name in filtered_df.columns:
            # Utiliser .fillna('') et .astype(str) pour éviter les erreurs sur les NaN
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
                # Appliquer la recherche sur chaque colonne et combiner avec 'any'
                mask = filtered_df[cols_to_search_api].fillna('').astype(str).apply(
                    lambda x: x.str.lower().str.contains(search_term_lower)
                ).any(axis=1)
                filtered_df = filtered_df[mask]

    # Filtrage par date en Python (devient un filtre supplémentaire ou redondant, mais ne cause pas d'erreur)
    # On peut le garder comme sécurité si le filtre API n'est pas parfait ou pour les cas où la date n'est pas au bon format dans la donnée API
    if 'date_publication' in filtered_df.columns and not filtered_df.empty and pd.notna(filtered_df['date_publication'].iloc[0]):
        start_filter_date = selected_dates_tuple[0]
        end_filter_date = selected_dates_tuple[1]
        
        # Assurer que les dates sont des objets date pour la comparaison
        if isinstance(start_filter_date, datetime): start_filter_date = start_filter_date.date()
        if isinstance(end_filter_date, datetime): end_filter_date = end_filter_date.date()

        # Assurer que la colonne date_publication contient bien des objets date non NaT
        # Le to_datetime(errors='coerce').dt.date + dropna dans load_data devrait garantir ça
        # Mais on ajoute une vérification supplémentaire ici au cas où
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


# --- Fonctions UI (Header, Métriques, Cartes Rappel, Pagination, Filtres Avancés) ---
# Ces fonctions restent identiques à la version précédente
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
    # Utiliser .dropna() avant .nunique() pour exclure les NaN du comptage
    unique_subcategories = data_df['sous_categorie_de_produit'].dropna().nunique() if 'sous_categorie_de_produit' in data_df.columns else 0

    today = date.today()
    recent_days_filter = st.session_state.get('recent_days_filter', DEFAULT_RECENT_DAYS)
    days_ago = today - timedelta(days=recent_days_filter)

    recent_recalls = 0
    # Vérification plus robuste pour le calcul des métriques
    if 'date_publication' in data_df.columns and not data_df['date_publication'].dropna().empty:
        # S'assurer que la colonne est de type date pour la comparaison et gérer les NaN
        valid_dates = data_df['date_publication'].dropna()
        if not valid_dates.empty and isinstance(valid_dates.iloc[0], date):
             recent_recalls = len(valid_dates[valid_dates >= days_ago])

    severe_percent = 0
    if 'risques_encourus' in data_df.columns and not data_df.empty:
        grave_keywords = ['microbiologique', 'listeria', 'salmonelle', 'allerg', 'toxique', 'e. coli', 'corps étranger', 'chimique']
        # Utiliser .fillna('').astype(str).str.lower() pour gérer les NaN et les types non string
        search_series = data_df['risques_encourus'].fillna('').astype(str).str.lower()
        severe_risks_mask = search_series.str.contains('|'.join(grave_keywords), na=False) # na=False ici gère les NaN si fillna échoue
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
            # Gérer le cas où liens_vers_images est une liste ou contient des NaN/None
            if isinstance(image_url, list) and image_url:
                 image_url = image_url[0]
            elif isinstance(image_url, str) and image_url:
                image_url = image_url.split('|')[0] # Gérer le format string séparé par '|'
            else:
                image_url = "https://via.placeholder.com/150/CCCCCC/FFFFFF?Text=Image+ND" # Image de remplacement
            st.image(image_url, width=130)

        with col_content:
            # Utiliser .get() et fournir une valeur par défaut, et gérer les NaN/None
            product_name = row_data.get('nom_commercial', row_data.get('modeles_ou_references', 'Produit non spécifié'))
            if pd.isna(product_name) or product_name == '': product_name = row_data.get('modeles_ou_references', 'Produit non spécifié')
            if pd.isna(product_name) or product_name == '': product_name = 'Produit non spécifié'

            st.markdown(f"<h5>{product_name}</h5>", unsafe_allow_html=True)

            pub_date_obj = row_data.get('date_publication')
            # Vérifier si pub_date_obj est un objet date avant de formater
            formatted_date = pub_date_obj.strftime('%d/%m/%Y') if isinstance(pub_date_obj, date) else 'Date inconnue'
            st.caption(f"Publié le: {formatted_date}")

            risk_text_raw = row_data.get('risques_encourus', 'Risque non spécifié')
            # Assurer que risk_text_raw est une chaîne pour .lower()
            risk_text_lower = str(risk_text_raw).lower()

            badge_color = "grey"; badge_icon = "⚠️"
            if any(keyword in risk_text_lower for keyword in ['listeria', 'salmonelle', 'e. coli', 'danger immédiat', 'toxique']):
                badge_color = "red"; badge_icon = "☠️"
            elif any(keyword in risk_text_lower for keyword in ['allergène', 'allergie', 'microbiologique', 'corps étranger', 'chimique']):
                badge_color = "orange"; badge_icon = "🔬"
            elif risk_text_raw and risk_text_raw != 'Risque non spécifié': # Si ce n'est pas vide/N/A mais pas une alerte majeure
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
        # Réinitialiser la page si les données deviennent vides
        st.session_state.current_page_recalls = 1
        return

    st.markdown(f"#### Affichage des rappels ({len(data_df)} résultats)")

    if 'current_page_recalls' not in st.session_state:
        st.session_state.current_page_recalls = 1

    total_pages = (len(data_df) - 1) // items_per_page_setting + 1
    # S'assurer que la page actuelle ne dépasse pas le total
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
                st.rerun()

        with cols_pagination[1]:
            st.markdown(f"<div style='text-align: center; margin-top: 10px;'>Page {current_page} sur {total_pages}</div>", unsafe_allow_html=True)

        with cols_pagination[2]:
            if st.button("Suivant →", disabled=(current_page == total_pages), use_container_width=True, key="next_page_btn"):
                st.session_state.current_page_recalls += 1
                st.rerun()


def create_advanced_filters(df_full_data):
    # Utilisation de pd.NaT pour les dates non valides/manquantes
    # S'assurer que min_date_data est un objet date et pas un NaT si df_full_data est vide ou n'a pas de dates valides
    min_date_data = df_full_data['date_publication'].dropna().min() if 'date_publication' in df_full_data.columns and not df_full_data['date_publication'].dropna().empty else START_DATE
    # Assurer que min_date_data est bien un date.date si possible
    if isinstance(min_date_data, datetime): min_date_data = min_date_data.date()
    elif not isinstance(min_date_data, date): min_date_data = START_DATE # Fallback si format inattendu

    max_date_data = date.today()

    # Vérifier et corriger les types de date dans session_state au début
    if 'date_filter_start' not in st.session_state or not isinstance(st.session_state.date_filter_start, date):
        st.session_state.date_filter_start = min_date_data
    if 'date_filter_end' not in st.session_state or not isinstance(st.session_state.date_filter_end, date):
        st.session_state.date_filter_end = max_date_data

    # S'assurer que ce sont des objets date et non datetime si l'API renvoie datetime parfois
    if isinstance(st.session_state.date_filter_start, datetime): st.session_state.date_filter_start = st.session_state.date_filter_start.date()
    if isinstance(st.session_state.date_filter_end, datetime): st.session_state.date_filter_end = st.session_state.date_filter_end.date()


    with st.expander("🔍 Filtres avancés et Options d'affichage", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            if 'categorie_de_produit' in df_full_data.columns:
                # Utiliser .dropna() pour exclure les NaN des options
                unique_main_categories = sorted(df_full_data['categorie_de_produit'].dropna().unique())
                selected_categories = st.multiselect(
                    "Filtrer par Catégories principales:", options=unique_main_categories,
                    default=st.session_state.get('selected_categories_filter', []),
                    key="main_cat_filter"
                )
                st.session_state.selected_categories_filter = selected_categories
            else: selected_categories = []

            if 'sous_categorie_de_produit' in df_full_data.columns:
                # Filtrer le DataFrame avant de prendre les sous-catégories si des catégories principales sont sélectionnées
                df_for_subcats = df_full_data[df_full_data['categorie_de_produit'].isin(selected_categories)] if selected_categories else df_full_data
                # Utiliser .dropna() pour exclure les NaN des options
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
                # Utiliser .dropna() pour exclure les NaN des options
                unique_risks = sorted(df_full_data['risques_encourus'].dropna().unique())
                display_risks = unique_risks
                if len(unique_risks) > 75:
                    # Prendre les 75 risques les plus fréquents s'il y en a trop
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

            # Utiliser les dates de session comme valeur par défaut pour le widget
            selected_dates_tuple_local = st.date_input(
                "Filtrer par période de publication:",
                value=(st.session_state.date_filter_start, st.session_state.date_filter_end),
                min_value=min_date_data, max_value=max_date_data,
                key="date_range_picker_adv"
            )
            # Le date_input retourne toujours un tuple de 2 dates si la sélection est valide
            if len(selected_dates_tuple_local) == 2:
                 # Mettre à jour la session seulement si les dates du widget changent
                if selected_dates_tuple_local[0] != st.session_state.date_filter_start or selected_dates_tuple_local[1] != st.session_state.date_filter_end:
                    st.session_state.date_filter_start, st.session_state.date_filter_end = selected_dates_tuple_local
                    # Réinitialiser la page des rappels affichée si la date change
                    st.session_state.current_page_recalls = 1
                    st.rerun() # Rerun pour appliquer les nouveaux filtres

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

            # Recalculer min_date_data au cas où df_full_data a été mis à jour depuis le dernier chargement
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


    # Afficher les filtres actifs
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
    # Ne pas afficher le filtre de date s'il correspond à la plage complète des données chargées
    min_data_date_actual = df_full_data['date_publication'].dropna().min() if 'date_publication' in df_full_data.columns and not df_full_data['date_publication'].dropna().empty else START_DATE
    if isinstance(min_data_date_actual, datetime): min_data_date_actual = min_data_date_actual.date()
    elif not isinstance(min_data_date_actual, date): min_data_date_actual = START_DATE

    if current_start_date_display != min_data_date_actual or current_end_date_display != date.today():
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


# --- Fonctions de Visualisation (Onglet Visualisations) (légères améliorations de robustesse) ---
def create_improved_visualizations(data_df_viz):
    if data_df_viz.empty:
        st.info("Données insuffisantes pour générer des visualisations avec les filtres actuels.")
        return

    st.markdown('<div class="chart-container" style="margin-top:1rem;">', unsafe_allow_html=True)

    # Assurer que la colonne de date est bien en format datetime pour le resample
    if 'date_publication' in data_df_viz.columns and not data_df_viz['date_publication'].dropna().empty:
        try:
            # Convertir en datetime si ce n'est pas déjà le cas (par sécurité, même si load_data le fait normalement)
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
            # Grouper par mois et compter
            monthly_data = data_df_viz.set_index('date_publication_dt').resample('MS').size().reset_index(name='count') # 'MS' pour début du mois
            monthly_data['year_month'] = monthly_data['date_publication_dt'].dt.strftime('%Y-%m')
            # Assurer que les mois sont triés correctement
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
                # Utiliser .dropna() pour exclure les NaN du comptage
                cat_counts = data_df_viz['categorie_de_produit'].dropna().value_counts().nlargest(10)
                if not cat_counts.empty:
                    fig_cat = px.pie(values=cat_counts.values, names=cat_counts.index, title="Top 10 Catégories", hole=0.4)
                    fig_cat.update_traces(textposition='outside', textinfo='percent+label')
                    st.plotly_chart(fig_cat, use_container_width=True)
                else: st.info("Pas de données pour la répartition par catégorie.")
            else: st.warning("Colonne 'categorie_de_produit' manquante.")
        with col_subcat:
            if 'sous_categorie_de_produit' in data_df_viz.columns:
                 # Utiliser .dropna() pour exclure les NaN du comptage
                subcat_counts = data_df_viz['sous_categorie_de_produit'].dropna().value_counts().nlargest(10)
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
                                       color=risk_counts.values, color_continuous_scale='Reds') # Scale mis à jour
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
                                        color=motif_counts.values, color_continuous_scale='Oranges') # Scale mis à jour
                    fig_motifs.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False, height=max(400, len(motif_counts)*40))
                    st.plotly_chart(fig_motifs, use_container_width=True)
                else: st.info("Pas assez de données de motif pour un top 10.")
            else: st.info("Pas de données de motif valides.")
        else: st.warning("Colonne 'motif_du_rappel' manquante.")

    st.markdown('</div>', unsafe_allow_html=True)


# --- Fonctions pour l'IA avec Groq (inchangée) ---
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
             # Afficher un message plus utile lors de l'entrée/modification de la clé
             if new_key:
                 if new_key.startswith("gsk_"):
                    st.success("Clé API Groq enregistrée.", icon="👍")
                 else:
                    st.warning("Format de clé API invalide. Doit commencer par 'gsk_'.", icon="⚠️")
             else:
                 st.info("Aucune clé API Groq n'est configurée.", icon="ℹ️")
             st.rerun() # Rerun pour mettre à jour l'état de la connexion et les options

        model_options = {
            "llama3-70b-8192": "Llama 3 (70B) - Puissant", "llama3-8b-8192": "Llama 3 (8B) - Rapide",
            "mixtral-8x7b-32768": "Mixtral (8x7B) - Large contexte", "gemma-7b-it": "Gemma (7B) - Léger"
        }
        # Sélection du modèle n'est possible que si une clé valide est présente
        model_disabled = not (st.session_state.user_groq_api_key and st.session_state.user_groq_api_key.startswith("gsk_"))
        selected_model_key = st.selectbox(
            "Choisir un modèle IA:", options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=list(model_options.keys()).index(st.session_state.get('groq_model', 'llama3-8b-8192')) if st.session_state.get('groq_model', 'llama3-8b-8192') in model_options else 0, # Maintien de la sélection ou défaut
            key="groq_model_select_sidebar", disabled=model_disabled
        )
        # Mettre à jour la session state uniquement si le sélecteur n'est pas désactivé
        if not model_disabled:
             st.session_state.groq_model = selected_model_key

        with st.popover("Options avancées de l'IA", disabled=model_disabled):
            st.session_state.groq_temperature = st.slider("Température:", 0.0, 1.0, st.session_state.get('groq_temperature', 0.2), 0.1, disabled=model_disabled)
            st.session_state.groq_max_tokens = st.slider("Tokens max réponse:", 256, 4096, st.session_state.get('groq_max_tokens', 1024), 256, disabled=model_disabled)
            st.session_state.groq_max_context_recalls = st.slider("Max rappels dans contexte IA:", 5, 50, st.session_state.get('groq_max_context_recalls', 15), 1, disabled=model_disabled)

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
            st.error(f"Erreur d'initialisation du client Groq: {e}. Vérifiez votre clé.")
            # Stocker l'erreur d'initialisation dans la session state pour éviter de spammer les logs
            st.session_state.groq_client_error = str(e)
            return None
    # Retourner None si pas de clé valide ou erreur d'initialisation précédente
    return None


def prepare_context_for_ia(df_context, max_items=10):
    if df_context.empty:
        return "Aucune donnée de rappel pertinente trouvée pour cette question avec les filtres actuels."

    cols_for_ia = [
        'nom_de_la_marque_du_produit', 'nom_commercial', 'modeles_ou_references',
        'categorie_de_produit', 'sous_categorie_de_produit',
        'risques_encourus', 'motif_du_rappel', 'date_publication', 'distributeurs'
    ]
    # Filtrer les colonnes pour n'utiliser que celles qui existent dans le DataFrame
    cols_to_use = [col for col in cols_for_ia if col in df_context.columns]

    if cols_to_use:
        actual_max_items = min(max_items, len(df_context))
        # Utiliser .copy() pour éviter SettingWithCopyWarning potentiel
        context_df_sample = df_context[cols_to_use].head(actual_max_items).copy()

        # Remplacer les NaN/None par des chaînes vides ou N/A avant de construire le texte
        for col in cols_to_use:
             context_df_sample[col] = context_df_sample[col].apply(lambda x: x if pd.notna(x) and x != '' else None)

        text_context = f"Voici un échantillon de {len(context_df_sample)} rappels (sur {len(df_context)} au total avec les filtres) :\n\n"
        for _, row in context_df_sample.iterrows():
            item_desc = []
            # Vérifier si la valeur est non nulle avant d'ajouter
            if row.get('nom_de_la_marque_du_produit') is not None:
                item_desc.append(f"Marque: {row['nom_de_la_marque_du_produit']}")

            product_display_name = row.get('nom_commercial')
            if product_display_name is None or product_display_name == '':
                 product_display_name = row.get('modeles_ou_references')
            if product_display_name is not None and product_display_name != '':
                 item_desc.append(f"Produit: {product_display_name}")
            elif not item_desc: # Si ni marque ni produit/modèle n'est dispo
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

            if isinstance(row.get('date_publication'), date): # Vérifier explicitement le type date
                date_pub_str = row['date_publication'].strftime('%d/%m/%y')
                item_desc.append(f"Date: {date_pub_str}")

            # Ajouter la description de l'item uniquement si elle n'est pas vide
            if item_desc:
                 text_context += "- " + ", ".join(item_desc) + "\n"

        return text_context
    else:
         return "Aucune colonne pertinente pour générer un contexte à partir des données disponibles."


def analyze_trends_data(df_analysis, product_type=None, risk_type=None, time_period="all"):
    if df_analysis.empty:
        return {"status": "no_data", "message": "Aucune donnée disponible pour l'analyse de tendance."}

    df_filtered = df_analysis.copy()
    # Assurer que la colonne de date est prête pour l'analyse
    if 'date_publication' in df_filtered.columns and not df_filtered['date_publication'].dropna().empty:
        try:
            df_filtered['date_publication_dt'] = pd.to_datetime(df_filtered['date_publication'], errors='coerce')
            df_filtered = df_filtered.dropna(subset=['date_publication_dt'])
            if df_filtered.empty: return {"status": "no_data", "message": "Dates invalides pour l'analyse après conversion."}
        except Exception as e:
             return {"status": "error", "message": f"Erreur lors de la conversion des dates pour l'analyse : {e}"}
    else: return {"status": "error", "message": "Colonne 'date_publication' manquante ou vide pour l'analyse de tendance."}

    analysis_title_parts = ["Évolution des rappels"]

    # Appliquer les filtres spécifiques (product/risk) si demandés
    initial_filtered_count = len(df_filtered)
    if product_type:
        # Rechercher dans plusieurs colonnes pertinentes pour le produit
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
        # Rechercher dans plusieurs colonnes pertinentes pour le risque
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

    # Analyse des données temporelles
    monthly_counts = df_filtered.set_index('date_publication_dt').resample('MS').size() # 'MS' pour début du mois
    if monthly_counts.empty : return {"status": "no_data", "message": "Pas de données mensuelles à analyser après filtrage spécifique."}

    # Remplir les mois manquants avec 0 pour la tendance
    idx = pd.date_range(monthly_counts.index.min(), monthly_counts.index.max(), freq='MS')
    monthly_counts = monthly_counts.reindex(idx, fill_value=0)
    # Créer l'index pour l'affichage (format 'YYYY-MM')
    monthly_counts_display_index = monthly_counts.index.strftime('%Y-%m')
    monthly_counts_values = monthly_counts.values

    # Statistiques de tendance
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

    # Période analysée pour le résumé
    period_label = f"du {monthly_counts.index.min().strftime('%d/%m/%Y')} au {monthly_counts.index.max().strftime('%d/%m/%Y')}"
    analysis_title_parts.append(f"({period_label})")


    # Générer le graphique matplotlib
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(monthly_counts_display_index, monthly_counts_values, marker='o', linestyle='-', label='Rappels/mois')
    if len(monthly_counts_values) >= 2 and 'trend_direction' in trend_stats and trend_stats['trend_direction'] != "indéterminée (données insuffisantes)":
        trend_line = model.predict(X)
        ax.plot(monthly_counts_display_index, trend_line, color='red', linestyle='--', label=f'Tendance ({trend_stats["trend_direction"]})')

    ax.set_title(' '.join(analysis_title_parts), fontsize=10)
    ax.set_xlabel("Mois", fontsize=8); ax.set_ylabel("Nombre de rappels", fontsize=8)
    # Ajuster les ticks X si trop nombreux
    if len(monthly_counts_display_index) > 15:
        tick_indices = np.linspace(0, len(monthly_counts_display_index) - 1, 15, dtype=int)
        ax.set_xticks(np.array(monthly_counts_display_index)[tick_indices])
    else:
        ax.set_xticks(monthly_counts_display_index)
    ax.tick_params(axis='x', rotation=45, labelsize=7); ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, linestyle=':', alpha=0.7); ax.legend(fontsize=8)
    plt.tight_layout()

    # Sauvegarder le graphique en mémoire
    buf = io.BytesIO()
    plt.savefig(buf, format="png"); buf.seek(0)
    graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) # Fermer la figure pour libérer la mémoire

    # Résumé textuel pour l'IA
    text_summary = f"Analyse de tendance ({period_label}, Rappels analysés: {trend_stats['total_recalls']}):\n"
    text_summary += f"- Moyenne mensuelle: {trend_stats['monthly_avg']:.1f} rappels.\n"
    if 'trend_direction' in trend_stats:
        text_summary += f"- Tendance générale: {trend_stats['trend_direction']}"
        if trend_stats['trend_direction'] not in ["indéterminée (données insuffisantes)", "stable"]:
             text_summary += f" (pente: {slope:.2f}).\n"
        else:
            text_summary += "\n"

    # Top 3 des sous-catégories et risques dans les données filtrées pour l'analyse
    if 'sous_categorie_de_produit' in df_filtered.columns:
        top_cat = df_filtered['sous_categorie_de_produit'].dropna().value_counts().nlargest(3)
        if not top_cat.empty: text_summary += "- Top 3 sous-catégories: " + ", ".join([f"{idx} ({val})" for idx, val in top_cat.items()]) + ".\n"
    if 'risques_encourus' in df_filtered.columns:
        top_risk = df_filtered['risques_encourus'].dropna().value_counts().nlargest(3)
        if not top_risk.empty: text_summary += "- Top 3 risques: " + ", ".join([f"{idx} ({val})" for idx, val in top_risk.items()]) + ".\n"

    return {
        "status": "success", "text_summary": text_summary, "graph_base64": graph_base64,
        "monthly_data": {str(k): int(v) for k,v in monthly_counts_display_index.to_list() for v in monthly_counts_values.to_list()}, # Adapter si besoin, mais le texte suffit pour l'IA
        "trend_stats": trend_stats
    }


def ask_groq_ai(client, user_query, context_data_text, trend_analysis_results=None):
    # Vérifier si le client Groq est valide
    if not client:
        # L'erreur d'initialisation est déjà affichée par get_groq_client/manage_groq_api_key
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

    # Construire le contexte complet pour l'IA
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
        {"role": "user", "content": f"{full_context_for_ai}\nQuestion de l'utilisateur: \"{user_query}\"\n\nRéponse (en Markdown):"} # Ajout d'une instruction pour répondre en Markdown
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages, model=st.session_state.get("groq_model", "llama3-8b-8192"),
            temperature=st.session_state.get("groq_temperature", 0.2),
            max_tokens=st.session_state.get("groq_max_tokens", 1024),
        )
        response_content = chat_completion.choices[0].message.content
        # Application basique de gras sur les chiffres (hors pourcentages et chiffres dans des mots)
        # Le modèle devrait déjà le faire avec le prompt, mais c'est une sécurité
        # response_content = re.sub(r'(\b\d{1,3}(?:[,.]\d+)?\b)(?!%|[-\w])', r'**\1**', response_content)
        return response_content
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API Groq: {e}")
        # Afficher des messages spécifiques pour les erreurs courantes
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
    groq_ready = manage_groq_api_key()
    groq_client = get_groq_client() if groq_ready else None

    # Initialisation des états de session si non existants
    default_session_keys = {
        'current_page_recalls': 1, 'items_per_page_filter': DEFAULT_ITEMS_PER_PAGE,
        'recent_days_filter': DEFAULT_RECENT_DAYS, 'date_filter_start': START_DATE,
        'date_filter_end': date.today(), 'search_term_main': "",
        'search_column_friendly_name_select': "Toutes les colonnes pertinentes",
        'groq_temperature': 0.2, 'groq_max_tokens': 1024, 'groq_max_context_recalls': 15,
        'groq_model': 'llama3-8b-8192', # Modèle par défaut
        'groq_chat_history': [{"role": "assistant", "content": "Bonjour ! Posez-moi une question sur les données affichées ou utilisez une suggestion."}]
    }
    for key, value in default_session_keys.items():
        if key not in st.session_state: st.session_state[key] = value

    # Chargement des données
    df_alim = load_data(API_URL_FOR_LOAD, START_DATE)

    # Vérifier si le chargement a réussi et si le DataFrame n'est pas vide
    if df_alim.empty:
        # L'erreur détaillée est déjà affichée dans load_data
        st.error("Aucune donnée de rappel alimentaire n'a pu être chargée ou trouvée pour la période. Vérifiez les messages d'erreur ci-dessus ou réessayez/modifiez la date de début de chargement.")
        # Option pour tenter de recharger avec une date de début différente si l'erreur persiste?
        # st.sidebar.date_input("Tenter de charger depuis:", value=START_DATE, key="reload_start_date") # Exemple
        # if st.sidebar.button("Recharger avec cette date", key="btn_reload_date"): st.cache_data.clear(); st.experimental_rerun()
        st.stop() # Arrêter l'exécution si pas de données

    # S'assurer que la date de début du filtre correspond à la date min réelle des données chargées la première fois
    if 'date_filter_start_init' not in st.session_state or not st.session_state.date_filter_start_init:
        min_data_date_actual = df_alim['date_publication'].dropna().min() if 'date_publication' in df_alim.columns and not df_alim['date_publication'].dropna().empty else START_DATE
        if isinstance(min_data_date_actual, date): st.session_state.date_filter_start = min_data_date_actual
        elif isinstance(min_data_date_actual, datetime): st.session_state.date_filter_start = min_data_date_actual.date()
        else: st.session_state.date_filter_start = START_DATE # Fallback
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
        # Trouver l'index correct pour maintenir la sélection
        current_search_col = st.session_state.get('search_column_friendly_name_select', "Toutes les colonnes pertinentes")
        try:
             default_index = search_column_options_friendly.index(current_search_col)
        except ValueError:
             default_index = 0 # Revenir au défaut si la colonne précédente n'existe plus
             st.session_state.search_column_friendly_name_select = "Toutes les colonnes pertinentes"

        st.session_state.search_column_friendly_name_select = st.selectbox(
            "Chercher dans:", search_column_options_friendly,
            index=default_index,
            key="main_search_column_select", label_visibility="collapsed"
        )

    # Filtres avancés
    # create_advanced_filters gère aussi la mise à jour des dates dans session_state
    (selected_main_categories, selected_subcategories, selected_risks,
     selected_dates_tuple_main, items_per_page_setting) = create_advanced_filters(df_alim)

    search_column_api = None
    if st.session_state.search_column_friendly_name_select != "Toutes les colonnes pertinentes":
        search_column_api = FRIENDLY_TO_API_COLUMN_MAPPING.get(st.session_state.search_column_friendly_name_select)

    # Filtrage des données selon tous les critères sélectionnés
    current_filtered_df = filter_data(
        df_alim, selected_subcategories, selected_risks, st.session_state.search_term_main,
        selected_dates_tuple_main, selected_main_categories, search_column_api
    )

    # Onglets principaux
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

        if not groq_ready:
            st.warning("Veuillez configurer votre clé API Groq dans la barre latérale pour utiliser l'assistant IA.", icon="⚠️")
            # Optionnel: Afficher un message si une erreur d'initialisation Groq est stockée
            if "groq_client_error" in st.session_state and st.session_state.groq_client_error:
                 st.error(f"Dernière erreur lors de l'initialisation de l'IA: {st.session_state.groq_client_error}")
        else:
            st.markdown("<div class='suggestion-button-container'>", unsafe_allow_html=True)
            suggestion_cols = st.columns(3)
            # Définir les suggestions avec les types d'analyse souhaités
            suggestion_queries = {
                "Tendance générale des rappels ?": {"type": "trend"},
                "Quels sont les 3 principaux risques ?": {"type": "context_only"},
                "Rappels pour 'fromage' ?": {"type": "context_specific", "product": "fromage"},
                 "Analyse des 'Listeria'": {"type": "trend", "risk": "Listeria"},
                 "Évolution des rappels 'Viande' ?": {"type": "trend", "product": "Viande"},
                 "Quels produits 'Bio' ont été rappelés ?": {"type": "context_specific", "product": "Bio"},
            }
            # Gérer l'état du bouton de suggestion cliqué
            if 'clicked_suggestion_query' not in st.session_state: st.session_state.clicked_suggestion_query = None

            idx = 0
            for query_text, params in suggestion_queries.items():
                with suggestion_cols[idx % len(suggestion_cols)]:
                    if st.button(query_text, key=f"suggestion_{idx}", use_container_width=True):
                        st.session_state.clicked_suggestion_query = {"query": query_text, "params": params}
                        st.session_state.user_groq_query_input_main = query_text # Pré-remplir l'input chat
                        # Trigger un rerun pour traiter la suggestion
                        st.rerun()
                idx += 1
            st.markdown("</div>", unsafe_allow_html=True)

            # Affichage de l'historique du chat
            chat_display_container = st.container(height=450, border=False)
            with chat_display_container:
                # Afficher l'historique du chat stocké en session state
                for message in st.session_state.groq_chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"], unsafe_allow_html=True)
                        # Si le message assistant contient un graphique, l'afficher
                        if message["role"] == "assistant" and "graph_base64" in message and message["graph_base64"]:
                            st.image(f"data:image/png;base64,{message['graph_base64']}")

            # Input utilisateur pour le chat
            user_groq_query = st.chat_input(
                "Posez votre question à l'IA...",
                value=st.session_state.get('user_groq_query_input_main', ""), # Pré-rempli par la suggestion
                key="user_groq_query_input_main",
                disabled=not groq_ready or groq_client is None # Désactiver si IA non prête ou client non initialisé
            )

            # Logique de traitement de la query (soit input utilisateur, soit suggestion cliquée)
            query_to_process = None
            params_to_process = {}

            # Si une suggestion a été cliquée (et n'a pas encore été traitée)
            if st.session_state.clicked_suggestion_query:
                 query_to_process = st.session_state.clicked_suggestion_query["query"]
                 params_to_process = st.session_state.clicked_suggestion_query["params"]
                 # Effacer la suggestion cliquée après l'avoir récupérée
                 st.session_state.clicked_suggestion_query = None
            # Si l'utilisateur a soumis via l'input chat (et ce n'était pas juste la suggestion pré-remplie)
            elif user_groq_query and user_groq_query != st.session_state.get('last_processed_groq_query', ''):
                 query_to_process = user_groq_query
                 # Tenter de déterminer si c'est une demande de tendance basée sur le texte
                 query_lower = query_to_process.lower()
                 if any(k in query_lower for k in ["tendance", "évolution", "statistique", "analyse de", "combien de rappel"]):
                     params_to_process["type"] = "trend"
                     # Tentative d'extraction simple de produit/risque du texte libre
                     possible_products = ["fromage", "viande", "bio", "poulet", "saumon", "lait"]
                     possible_risks = ["listeria", "salmonelle", "e. coli", "allergène"]
                     for p in possible_products:
                         if p in query_lower: params_to_process["product"] = p; break # Prendre le premier trouvé
                     for r in possible_risks:
                         if r in query_lower: params_to_process["risk"] = r; break # Prendre le premier trouvé


            # Si une query est à processer
            if query_to_process:
                # Ajouter la question de l'utilisateur à l'historique immédiatement
                st.session_state.groq_chat_history.append({"role": "user", "content": query_to_process})
                st.session_state.last_processed_groq_query = query_to_process # Stocker la dernière query traitée pour éviter doublon input

                # Afficher la nouvelle question dans l'UI
                with chat_display_container:
                     with st.chat_message("user"): st.markdown(query_to_process)


                with st.spinner("L'assistant IA réfléchit... 🤔"):
                    # Préparer le contexte textuel des rappels filtrés
                    context_text_for_ai = prepare_context_for_ia(
                        current_filtered_df, max_items=st.session_state.get('groq_max_context_recalls', 15)
                    )

                    # Effectuer l'analyse de tendance si le type est "trend"
                    trend_results = None
                    if params_to_process.get("type") == "trend":
                        trend_results = analyze_trends_data(
                            current_filtered_df,
                            product_type=params_to_process.get("product"),
                            risk_type=params_to_process.get("risk")
                        )

                    # Appeler l'IA avec la question, le contexte des données et les résultats de tendance
                    ai_response_text = ask_groq_ai(groq_client, query_to_process, context_text_for_ai, trend_results)

                # Préparer le message de l'assistant pour l'historique (incluant le graphique si pertinent)
                assistant_message = {"role": "assistant", "content": ai_response_text}
                # Ajouter le graphique si l'analyse de tendance a réussi et a produit un graphique
                if trend_results and trend_results["status"] == "success" and "graph_base64" in trend_results:
                    assistant_message["graph_base64"] = trend_results["graph_base64"]

                # Ajouter le message de l'assistant à l'historique
                st.session_state.groq_chat_history.append(assistant_message)

                # Rerun pour afficher la réponse de l'IA (y compris l'image)
                st.experimental_rerun()


    # Section pied de page ou info de données
    st.sidebar.markdown("---")
    # Obtenir la date min réelle des données chargées pour l'affichage
    min_data_date_actual_display = df_alim['date_publication'].dropna().min() if 'date_publication' in df_alim.columns and not df_alim['date_publication'].dropna().empty else START_DATE
    if isinstance(min_data_date_actual_display, datetime): min_data_date_actual_display = min_data_date_actual_display.date()
    elif not isinstance(min_data_date_actual_display, date): min_data_date_actual_display = START_DATE

    st.sidebar.caption(f"Données RappelConso 'Alimentation'. {len(df_alim)} rappels chargés depuis le {min_data_date_actual_display.strftime('%d/%m/%Y')}.")
    if st.sidebar.button("🔄 Mettre à jour les données (efface le cache)", type="primary", use_container_width=True, key="update_data_btn"):
        # Effacer le cache des données pour forcer un nouveau chargement
        st.cache_data.clear()
        # Réinitialiser l'indicateur de date de début pour le filtre
        if 'date_filter_start_init' in st.session_state: del st.session_state.date_filter_start_init
        # Optionnel: Réinitialiser les filtres à leurs valeurs par défaut après rechargement complet
        # keys_to_reset_on_reload = ['selected_categories_filter', 'selected_subcategories_filter', 'selected_risks_filter', 'search_term_main', 'search_column_friendly_name_select', 'date_filter_start', 'date_filter_end', 'items_per_page_filter', 'recent_days_filter', 'current_page_recalls', 'groq_chat_history']
        # for key_to_del in keys_to_reset_on_reload:
        #     if key_to_del in st.session_state: del st.session_state[key_to_del]
        st.experimental_rerun() # Relancer l'application pour charger les nouvelles données

if __name__ == "__main__":
    main()
