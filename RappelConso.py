import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
import io
import base64
from datetime import datetime, date, timedelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # Kept as in original list, though not used
import re
import logging
import json
from typing import Dict, List, Optional, Any, Generator, Tuple

# Configuration de la page Streamlit
st.set_page_config(
    page_title="RappelConso Insight",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONSTANTES ---
BASE_API_URL = "https://data.economie.gouv.fr/api/v2/catalog/datasets/rappelconso-v2-gtin-espaces/records"
START_DATE = date(2022, 1, 1)  # Date de début par défaut pour le chargement API
API_TIMEOUT = 30  # Timeout pour les requêtes API (secondes)
DEFAULT_ITEMS_PER_PAGE = 6  # Nombre d'éléments par page
DEFAULT_RECENT_DAYS = 30  # Période par défaut pour "Récents"
LOGO_URL = "https://raw.githubusercontent.com/M00N69/RAPPELCONSO/main/logo%2004%20copie.jpg"

# Mapping des noms de champs (API v2 -> structure normalisée)
FIELD_MAPPING = {
    "categorie_produit": "categorie_de_produit",
    "sous_categorie_produit": "sous_categorie_de_produit",
    "marque_produit": "nom_de_la_marque_du_produit",
    "modeles_ou_references": "modeles_ou_references",
    "motif_rappel": "motif_du_rappel",
    "risques_encourus": "risques_encourus",
    "distributeurs": "distributeurs",
    "liens_vers_les_images": "liens_vers_images",
    "lien_vers_la_fiche_rappel": "lien_vers_la_fiche_rappel",
    "date_publication": "date_publication",
    "libelle": "nom_commercial" # 'libelle' might be an alternative for 'nom_commercial'
}

# Mapping des noms de champs pour l'interface utilisateur (pour le contexte AI par exemple)
UI_FIELD_NAMES = {
    "categorie_de_produit": "Catégorie principale",
    "sous_categorie_de_produit": "Sous-catégorie",
    "nom_de_la_marque_du_produit": "Marque",
    "nom_commercial": "Nom commercial",
    "modeles_ou_references": "Modèle/Référence",
    "motif_du_rappel": "Motif du rappel",
    "risques_encourus": "Risques encourus",
    "distributeurs": "Distributeurs",
    "date_publication": "Date de publication"
}


# --- CSS POUR L'INTERFACE ---
st.markdown("""
<style>
    /* --- Design général --- */
    body {
        font-family: 'Roboto', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f5f7fa;
        color: #333;
    }

    .main .block-container {
        max-width: 1200px;
        padding: 2rem 1rem;
        margin: 0 auto;
    }

    /* --- Header --- */
    .header-container {
        background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .header-title {
        color: white;
        font-size: 2.8em;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .header-subtitle {
        color: white;
        opacity: 0.9;
        font-size: 1.2em;
        margin-top: 10px;
    }

    /* --- Metric Cards --- */
    .metric-card {
        background-color: #ffffff;
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        text-align: center;
        transition: transform 0.3s ease, box_shadow 0.3s ease;
        margin-bottom: 1.5rem;
        border-top: 5px solid #2980b9;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        color: #2980b9;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 1.1em;
        color: #555;
        font-weight: 500;
    }

    /* --- Recall Cards --- */
    .recall-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        border-left: 5px solid #2980b9;
        transition: transform 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .recall-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.12);
    }
    .recall-card h3 {
        color: #1a5276;
        font-size: 1.25rem;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    .recall-image {
        width: 100%;
        height: 160px;
        object-fit: contain;
        border-radius: 6px;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
    }
    .recall-content {
        flex-grow: 1;
    }
    .recall-footer {
        margin-top: auto;
        padding-top: 0.8rem;
        border-top: 1px solid #eee;
    }

    /* --- Risk badges --- */
    .risk-badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 0.8rem;
    }
    .risk-high {
        background-color: #ffebee;
        color: #d32f2f;
    }
    .risk-medium {
        background-color: #fff8e1;
        color: #ff8f00;
    }
    .risk-low {
        background-color: #e8f5e9;
        color: #388e3c;
    }
    .recall-date {
        color: #666;
        font-size: 0.85rem;
        margin-bottom: 0.8rem;
    }
    .info-item {
        margin-bottom: 0.5rem;
    }
    .info-label {
        font-weight: 600;
        color: #555;
        margin-right: 0.3rem;
    }

    /* --- Filters --- */
    .filter-group {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #e1e8ed;
    }
    .filter-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        color: #2c3e50;
    }
    .active-filters {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    .filter-tag {
        background-color: #e1f5fe;
        color: #0288d1;
        padding: 0.25rem 0.6rem;
        border-radius: 16px;
        font-size: 0.8rem;
        display: inline-flex;
        align-items: center;
    }
    .filter-tag button {
        background: none;
        border: none;
        color: #0288d1;
        margin-left: 0.25rem;
        cursor: pointer;
    }

    /* --- Pagination --- */
    .pagination {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
    }
    .page-btn {
        background-color: #fff;
        border: 1px solid #ddd;
        padding: 0.5rem 1rem;
        margin: 0 0.25rem;
        border-radius: 4px;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .page-btn:hover {
        background-color: #f5f5f5;
    }
    .page-btn.active {
        background-color: #2980b9;
        color: white;
        border-color: #2980b9;
    }
    .page-info {
        display: flex;
        align-items: center;
        margin: 0 1rem;
    }

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2980b9;
        color: white;
    }

    /* --- AI Assistant --- */
    .chat-container {
        border-radius: 10px;
        margin-top: 1rem;
        overflow: hidden;
    }
    .chat-message {
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 10px;
    }
    .chat-message.user {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .chat-message.assistant {
        background-color: #f5f5f5;
        border-left: 4px solid #9e9e9e;
    }
    .chat-input {
        display: flex;
        margin-top: 1rem;
    }
    .chat-input input {
        flex-grow: 1;
        padding: 0.5rem;
        border: 1px solid #ddd;
        border-radius: 4px;
        margin-right: 0.5rem;
    }
    .chat-input button {
        background-color: #2196f3;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        cursor: pointer;
    }

    /* --- Charts --- */
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }

    /* --- Debug --- */
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

    /* --- Responsive --- */
    @media (max-width: 768px) {
        .header-title { font-size: 1.8em; }
        .metric-card { padding: 1rem; }
        .metric-value { font-size: 1.8em; }
    }
</style>
""", unsafe_allow_html=True)

# --- CLASSES ET FONCTIONS GÉNÉRALES ---

class RappelConsoAPI:
    """Module d'accès à l'API RappelConso v2"""

    @staticmethod
    def build_query(
        category: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        additional_filters: Optional[Dict[str, str]] = None
    ) -> str:
        """Construit une clause WHERE pour l'API v2"""

        filters = []

        # API v2 uses 'categorie_produit' not 'categorie_de_produit' in query
        if category:
            filters.append(f"categorie_produit='{category}'")

        if start_date:
            filters.append(f"date_publication >= '{start_date}'")

        if end_date:
                filters.append(f"date_publication <= '{end_date}'")

        if additional_filters:
            for field, value in additional_filters.items():
                # Ensure correct API field names are used in the query
                api_field = next((api_k for api_k, norm_v in FIELD_MAPPING.items() if norm_v == field), field)
                filters.append(f"{api_field}='{value}'")

        return " AND ".join(filters)

    @classmethod
    def fetch_recalls(
        cls,
        where_clause: str,
        page_size: int = 1000, # Increased page size for efficiency
        max_records: Optional[int] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Récupère les rappels par lots avec pagination
        Renvoie un générateur pour économiser la mémoire
        """

        offset = 0
        total_fetched = 0
        total_count = None # Not used for loop termination with API v2, but good for logging

        logger.info(f"Starting API fetch with WHERE clause: {where_clause}")

        while True:
            try:
                params = {
                    "where": where_clause,
                    "limit": page_size,
                    "offset": offset,
                    "order_by": "date_publication DESC" # Order by date descending
                }

                logger.debug(f"Fetching data batch: limit={page_size}, offset={offset}")

                response = requests.get(
                    BASE_API_URL,
                    params=params,
                    timeout=API_TIMEOUT
                )
                response.raise_for_status()
                data = response.json()

                # V2 API might not have total_count easily available on first call params
                # We rely on receiving empty records to stop
                records = data.get("records", [])
                if not records:
                    logger.info("No more records to fetch.")
                    break

                # Transformation des données au format attendu
                for record in records:
                    fields = record.get("record", {}).get("fields", {})
                    if fields:
                        # Normalisation des champs
                        normalized_fields = cls._normalize_fields(fields)
                        yield normalized_fields

                        total_fetched += 1
                        if max_records and total_fetched >= max_records:
                            logger.info(f"Reached max_records limit: {max_records}")
                            return

                offset += len(records)

                # Éviter de surcharger l'API, surtout si la taille de page est petite
                if len(records) == page_size: # Only sleep if we got a full page, suggesting more data
                     time.sleep(0.1)
                else: # If less than page_size records, this was the last page
                    logger.info(f"Last page fetched (less than {page_size} records).")
                    break


            except requests.exceptions.RequestException as e:
                logger.error(f"API error during fetch (offset {offset}): {str(e)}")
                if response.status_code == 429:  # Too Many Requests
                    logger.warning("Rate limit hit (429), waiting 5 seconds before retry...")
                    time.sleep(5)  # Wait before retrying
                    continue # Retry the same offset
                else:
                    st.error(f"Erreur lors de la récupération des données de l'API RappelConso: {e}")
                    st.warning("Veuillez réessayer plus tard.")
                    st.stop() # Stop the Streamlit app execution on critical error
            except Exception as e:
                logger.error(f"An unexpected error occurred during fetch (offset {offset}): {str(e)}")
                st.error(f"Une erreur inattendue s'est produite: {e}")
                st.stop()


    @staticmethod
    def _normalize_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise les champs des données de l'API V2"""

        normalized = {}

        # Apply mapping for known fields, keep others if present
        for api_key, norm_key in FIELD_MAPPING.items():
             if api_key in fields:
                 normalized[norm_key] = fields[api_key]

        # Handle date conversion
        if "date_publication" in normalized and isinstance(normalized["date_publication"], str):
            try:
                # Assuming ISO format like 'YYYY-MM-DDTHH:MM:SS+00:00'
                normalized["date_publication"] = datetime.fromisoformat(normalized["date_publication"].replace("Z", "+00:00")).date()
            except ValueError:
                logger.warning(f"Could not parse date_publication: {normalized['date_publication']}")
                normalized["date_publication"] = pd.NaT # Use pandas NaT for invalid dates


        # Ensure essential columns exist, even if empty
        essential_columns = [
            "categorie_de_produit", "sous_categorie_de_produit",
            "nom_de_la_marque_du_produit", "motif_du_rappel",
            "risques_encourus", "date_publication", "nom_commercial"
        ]

        for col in essential_columns:
            if col not in normalized:
                normalized[col] = pd.NA # Use pandas NA for missing values

        return normalized

    @classmethod
    @st.cache_data(ttl=3600, show_spinner="Chargement et traitement des données RappelConso...") # Cache for 1 hour
    def load_to_dataframe(
        cls,
        where_clause: str,
        max_records: Optional[int] = None
    ) -> pd.DataFrame:
        """Charge les données dans un DataFrame pandas avec caching"""

        logger.info(f"Loading data to DataFrame with where: {where_clause}")

        records = list(cls.fetch_recalls(where_clause, max_records=max_records))

        if not records:
            logger.warning("No records found matching the criteria")
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Ensure date column is datetime type for filtering/plotting
        if "date_publication" in df.columns:
             # Convert to datetime objects first, coercing errors, then extract date
             df["date_publication"] = pd.to_datetime(df["date_publication"], errors="coerce").dt.date

        # Remove rows where essential columns are entirely missing (optional, but good practice)
        # df.dropna(subset=essential_columns, how="all", inplace=True)

        # Sort by date of publication (most recent first)
        if "date_publication" in df.columns and not df["date_publication"].empty:
            df = df.sort_values("date_publication", ascending=False)

        logger.info(f"Loaded {len(df)} records into DataFrame.")

        return df


class GroqAssistant:
    """Module d'intégration avec l'API Groq"""

    def __init__(self):
        self.api_key = None
        self.is_authenticated = False
        self.Groq = None # Will be imported dynamically

    def set_api_key(self, api_key: str) -> bool:
        """Configure la clé API et teste l'authentification"""
        self.api_key = api_key
        self.is_authenticated = False

        if not api_key or not api_key.startswith("gsk_"):
             logger.warning("Invalid API key format provided")
             return False

        # Attempt dynamic import
        try:
            from groq import Groq
            self.Groq = Groq
        except ImportError:
            logger.error("Package 'groq' is not installed. Please install it (`pip install groq`) to use the AI assistant.")
            st.error("Le package 'groq' n'est pas installé. L'assistant IA ne peut pas fonctionner.")
            self.Groq = None
            return False # Return False because dependency is missing

        try:
            # Test simple d'authentification
            client = self.Groq(api_key=self.api_key)
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": "Test connection"}],
                model="llama3-8b-8192", # Use a smaller model for quick test
                max_tokens=10
            )
            # If the call succeeds, authentication is likely okay
            self.is_authenticated = True
            logger.info("Groq authentication successful")
            return True

        except Exception as e:
            logger.error(f"Groq authentication error: {str(e)}")
            self.is_authenticated = False
            return False

    def is_ready(self) -> bool:
        """Vérifie si le service est prêt à être utilisé"""
        return self.is_authenticated and self.api_key is not None and self.Groq is not None

    def query_assistant(
        self,
        user_query: str,
        context_data: pd.DataFrame,
        model_name: str, # Model name now required
        temperature: float = 0.2,
        max_tokens: int = 1024,
        max_context_items: int = 15
    ) -> Dict[str, Any]:
        """Interroge l'assistant Groq avec le contexte des données"""
        if not self.is_ready():
            return {
                "success": False,
                "error": "API Groq non configurée ou authentification échouée."
            }
        if context_data.empty:
             return {
                 "success": True,
                 "response": "Je ne peux pas analyser les rappels car aucune donnée ne correspond à vos filtres.",
                 "metrics": {}
             }


        try:
            # Préparation du contexte à partir des données
            context_text = self._prepare_context(context_data, max_items=max_context_items)

            # Construction du prompt système
            system_prompt = self._build_system_prompt()

            # Construction du message utilisateur
            user_message = f"Contexte des rappels de produits (basé sur {len(context_data)} rappels filtrés, {min(len(context_data), max_context_items)} exemples donnés):\n---\n{context_text}\n---\n\nQuestion de l'utilisateur: {user_query}\n\nRéponse:"

            # Client Groq
            client = self.Groq(api_key=self.api_key)

            # Appel à l'API
            start_time = time.time()

            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )

            elapsed_time = time.time() - start_time

            # Extraction de la réponse
            response_content = response.choices[0].message.content

            return {
                "success": True,
                "response": response_content,
                "metrics": {
                    "response_time": elapsed_time,
                    "model": model_name,
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                }
            }

        except Exception as e:
            logger.error(f"Groq query error: {str(e)}")
            return {
                "success": False,
                "error": f"Une erreur s'est produite lors de la communication avec l'assistant IA: {e}"
            }

    def _prepare_context(self, df: pd.DataFrame, max_items: int = 15) -> str:
        """Prépare le contexte pour l'IA à partir d'un DataFrame"""

        if df.empty:
            return "Aucune donnée de rappel disponible."

        # Sélection des colonnes pertinentes et disponibles
        cols_for_context = [
            "date_publication",
            "categorie_de_produit", "sous_categorie_de_produit",
            "nom_de_la_marque_du_produit", "nom_commercial",
            "modeles_ou_references",
            "risques_encourus", "motif_du_rappel",
            "distributeurs"
        ]

        available_cols = [col for col in cols_for_context if col in df.columns]

        # Limiter le nombre d'éléments pour le contexte AI
        sample_df = df[available_cols].head(max_items).copy() # Use .copy() to avoid SettingWithCopyWarning

        # Ensure date column is in date format for display
        if "date_publication" in sample_df.columns:
             sample_df["date_publication"] = sample_df["date_publication"].apply(
                 lambda x: x.strftime("%d/%m/%Y") if isinstance(x, date) else "Date inconnue"
             )

        # Construction du contexte textuel
        context_lines = []
        context_lines.append(f"Voici {len(sample_df)} exemples de rappels (sur {len(df)} rappels filtrés au total):")

        for i, row in sample_df.iterrows():
            context_lines.append(f"\nRappel #{i+1}:")

            for col in available_cols:
                value = row.get(col)
                if pd.notna(value) and value != "":
                    str_value = str(value)
                    # Tronquer les valeurs trop longues pour ne pas dépasser la limite de tokens
                    if len(str_value) > 150:
                        str_value = str_value[:147] + "..."

                    # Affichage plus convivial des noms de colonnes
                    display_name = UI_FIELD_NAMES.get(col, col)
                    context_lines.append(f"- {display_name}: {str_value}")

        return "\n".join(context_lines)

    def _build_system_prompt(self) -> str:
        """Construit le prompt système pour l'assistant"""

        return """Tu es "RappelConso Insight Assistant", un expert IA spécialisé dans l'analyse
        des données de rappels de produits alimentaires en France.
        
        Ton objectif est de répondre aux questions de l'utilisateur en te basant
        strictement sur le "Contexte des rappels de produits" qui t'est fourni.

        CONSIGNES IMPORTANTES:
        1. Base tes réponses UNIQUEMENT sur les données présentes dans le contexte fourni.
        2. Si l'information demandée n'est pas trouvable ou déductible des données du contexte, indique clairement que tu ne peux pas répondre avec les données disponibles. Ne spécule pas.
        3. Ne fais pas de recherches externes. Tes connaissances se limitent au contexte fourni.
        4. Sois concis, factuel et professionnel dans ton langage.
        5. Structure ta réponse en utilisant le format Markdown (listes, gras, etc.) pour la clarté.
        6. Si l'utilisateur pose une question sur des tendances ou des agrégations (ex: "Quels sont les risques les plus fréquents ?"), réfère-toi au contexte fourni et indique que ton analyse se base uniquement sur cet échantillon/ces données filtrées. Ne prétends pas analyser la totalité des données RappelConso si le contexte ne couvre qu'une partie.
        7. N'invente pas d'informations sur les produits, risques ou marques si elles ne sont pas dans le contexte.
        8. Utilise les noms de champs tels que définis dans le contexte (ex: "Catégorie principale", "Risques encourus").
        9. Pour les questions générales ou des demandes d'explication sur RappelConso non liées aux données spécifiques, réponds brièvement mais indique que ton rôle principal est d'analyser les données fournies.

        Exemples d'analyses basées sur le contexte:
        - Identifier les risques les plus cités dans le contexte.
        - Mentionner les marques qui apparaissent plusieurs fois.
        - Citer les motifs ou distributeurs présents.
        - Observer les dates de publication dans l'échantillon.

        Limites de l'analyse:
        - Tu ne peux pas donner de statistiques précises sur l'ensemble des rappels en France, seulement sur les données fournies.
        - Tu ne peux pas prédire l'avenir.
        - Tu ne peux pas donner d'avis médical ou juridique.

        Commence toujours par répondre directement à la question en te basant sur les faits du contexte.
        """


# --- FONCTIONS D'INTERFACE UTILISATEUR ---

def create_header():
    """Affiche l'en-tête de l'application"""
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">RappelConso Insight 🔍</h1>
        <p class="header-subtitle">
            L'outil professionnel d'analyse des alertes alimentaires en France
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_metrics(df_data):
    """Affiche les métriques principales"""
    if df_data.empty:
        st.info("Aucune donnée disponible pour afficher les métriques.")
        return

    # Calcul des métriques
    total_recalls = len(df_data)

    # Nombre de sous-catégories uniques (prendre en compte les valeurs manquantes)
    unique_subcategories = df_data["sous_categorie_de_produit"].nunique() if "sous_categorie_de_produit" in df_data.columns else 0

    # Rappels récents (période configurable)
    recent_days = st.session_state.get("recent_days_filter", DEFAULT_RECENT_DAYS)
    today = date.today()
    cutoff_date = today - timedelta(days=recent_days)

    recent_recalls = 0
    if "date_publication" in df_data.columns:
        # Ensure date_publication is date objects and filter valid dates
        valid_dates = df_data["date_publication"].dropna()
        recent_recalls = sum(valid_dates >= cutoff_date)


    # Calcul du pourcentage de risques graves (prendre en compte les valeurs manquantes)
    grave_keywords = ["microbiologique", "listeria", "salmonelle", "allerg", "toxique", "e. coli", "corps étranger", "chimique"]
    severe_percent = 0
    if "risques_encourus" in df_data.columns:
        # Convert to string and handle potential NaNs before lowercasing
        risk_series = df_data["risques_encourus"].astype(str).str.lower().fillna("")
        severe_risks = risk_series.str.contains("|".join(grave_keywords), na=False).sum()
        severe_percent = int((severe_risks / total_recalls * 100) if total_recalls > 0 else 0)

    # Affichage des métriques
    cols = st.columns(4)

    metrics = [
        ("Total des Rappels Filtrés", f"{total_recalls}"),
        (f"Rappels Récents ({recent_days}j)", f"{recent_recalls}"),
        ("Sous-Catégories Uniques", f"{unique_subcategories}"),
        ("Risques Notables", f"{severe_percent}%")
    ]

    for i, (label, value) in enumerate(metrics):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # Note explicative sur les risques notables
    st.caption(
        "*Risques notables: microbiologique, listeria, salmonelle, allergène, "
        "toxique, E. coli, corps étranger, risque chimique (selon le contenu des 'risques encourus')"
    )


def display_recall_card(row_data):
    """Affiche une carte de rappel"""

    # Récupération des données avec fallbacks
    product_name = (
        row_data.get("nom_commercial") or
        row_data.get("modeles_ou_references") or
        "Produit non spécifié" # Fallback if both commercial name and model are missing
    )

    brand = row_data.get("nom_de_la_marque_du_produit", "Marque non spécifiée")

    pub_date = row_data.get("date_publication")
    if isinstance(pub_date, date):
        formatted_date = pub_date.strftime("%d/%m/%Y")
    else:
        formatted_date = "Date inconnue" # Handle pd.NaT or other non-date types

    risk_text = str(row_data.get("risques_encourus", "Risque non spécifié")).strip()
    risk_text_lower = risk_text.lower()

    # Détermination du niveau de risque basé sur les mots-clés
    if any(keyword in risk_text_lower for keyword in ["listeria", "salmonelle", "e. coli", "danger immédiat", "toxique"]):
        risk_class = "risk-high"
        risk_icon = "☠️"
    elif any(keyword in risk_text_lower for keyword in ["allergène", "allergie", "microbiologique", "corps étranger", "chimique"]):
        risk_class = "risk-medium"
        risk_icon = "🔬"
    else:
        risk_class = "risk-low"
        risk_icon = "⚠️"

    # Récupération du motif et troncation
    motif = str(row_data.get("motif_du_rappel", "Motif non spécifié")).strip()
    if len(motif) > 120: # Allow a bit more space for motif
        motif = motif[:117] + "..."

     # Récupération des distributeurs et troncation
    distributeurs = str(row_data.get("distributeurs", "Non spécifié")).strip()
    if len(distributeurs) > 80: # Allow a bit more space for distributeurs
        distributeurs = distributeurs[:77] + "..."

    # Image du produit
    image_url = None
    if "liens_vers_images" in row_data and pd.notna(row_data["liens_vers_images"]):
        image_links_raw = str(row_data["liens_vers_images"]).strip()
        if image_links_raw:
            # Split by '|' and take the first valid URL
            image_urls = [url.strip() for url in image_links_raw.split("|")]
            for url in image_urls:
                 if url.startswith("http"): # Simple check for a valid URL format
                      image_url = url
                      break # Use the first valid URL found


    # URL de la fiche de rappel
    pdf_link = row_data.get("lien_vers_la_fiche_rappel")
    # Ensure pdf_link is a valid string before using
    pdf_link = str(pdf_link).strip() if pd.notna(pdf_link) else ""

    # Construction du HTML pour la carte
    html = f"""
    <div class="recall-card">
        <div style="text-align: center; margin-bottom: 15px;">
    """

    # Affichage de l'image avec fallback
    if image_url:
        # Using data-src for lazy loading potential improvement, but not standard in raw HTML in markdown
        # Stick to standard img tag for simplicity in Streamlit markdown
        html += f'<img src="{image_url}" class="recall-image" onerror="this.onerror=null; this.src=\'https://via.placeholder.com/400x300/f0f0f0/666666?text=Image+non+disponible\';" alt="Image produit" />'
    else:
        html += '<img src="https://via.placeholder.com/400x300/f0f0f0/666666?text=Image+non+disponible" class="recall-image" alt="Image non disponible" />'

    html += f"""
        </div>
        <div class="recall-content">
            <h3>{product_name}</h3>
            <div class="recall-date">Publié le: {formatted_date}</div>
            <div class="risk-badge {risk_class}">{risk_icon} {risk_text if risk_text != 'Risque non spécifié' else 'Risque non spécifié'}</div>
            <div class="info-item"><span class="info-label">Marque:</span> {brand if brand != 'Marque non spécifiée' else 'Non spécifiée'}</div>
            <div class="info-item"><span class="info-label">Motif:</span> {motif if motif != 'Motif non spécifié' else 'Non spécifié'}</div>
            <div class="info-item"><span class="info-label">Distributeurs:</span> {distributeurs if distributeurs != 'Non spécifié' else 'Non spécifiés'}</div>
        </div>
    """

    # Bouton vers la fiche PDF uniquement si disponible et valide
    if pdf_link and pdf_link.startswith("http"):
        html += f"""
        <div class="recall-footer">
            <a href="{pdf_link}" target="_blank" style="display: inline-block; background-color: #2980b9; color: white; padding: 8px 16px; border-radius: 4px; text-decoration: none; font-size: 14px; width: 100%; text-align: center;">📄 Fiche de rappel complète</a>
        </div>
        """
    else:
         html += f"""
         <div class="recall-footer">
             <span style="display: inline-block; background-color: #f0f0f0; color: #666; padding: 8px 16px; border-radius: 4px; text-decoration: none; font-size: 14px; width: 100%; text-align: center;">Fiche de rappel non disponible</span>
         </div>
         """

    html += "</div>"

    # Afficher la carte
    st.markdown(html, unsafe_allow_html=True)


def display_paginated_recalls(data_df, items_per_page_setting):
    """Affiche les rappels avec pagination"""
    if data_df.empty:
        st.info("Aucun rappel ne correspond à vos critères de recherche.")
        return

    st.markdown(f"#### Rappels correspondant aux filtres ({len(data_df)} trouvés)")

    # Initialisation de la page courante
    if "current_page_recalls" not in st.session_state:
        st.session_state.current_page_recalls = 1

    # Calcul du nombre total de pages
    total_pages = (len(data_df) - 1) // items_per_page_setting + 1

    # S'assurer que la page courante est valide après filtrage
    if st.session_state.current_page_recalls > total_pages:
        st.session_state.current_page_recalls = max(1, total_pages)
    # Also reset if current page becomes 0 due to filters removing all items before current page
    if st.session_state.current_page_recalls == 0 and total_pages > 0:
        st.session_state.current_page_recalls = 1


    current_page = st.session_state.current_page_recalls

    # Sélection des rappels pour la page courante
    start_idx = (current_page - 1) * items_per_page_setting
    end_idx = min(start_idx + items_per_page_setting, len(data_df))
    current_recalls_page_df = data_df.iloc[start_idx:end_idx]

    # Affichage en grille à 3 colonnes
    num_columns = 3

    # Calculate rows needed for the current page's items
    items_on_page = len(current_recalls_page_df)
    rows = (items_on_page + num_columns - 1) // num_columns if items_on_page > 0 else 0


    # Affichage des cartes en grille
    for row_idx in range(rows):
        cols = st.columns(num_columns)

        for col_idx in range(num_columns):
            item_idx_on_page = row_idx * num_columns + col_idx

            if item_idx_on_page < items_on_page:
                # Get the actual index in the filtered_df
                actual_df_index = current_recalls_page_df.iloc[item_idx_on_page].name # Use .name to get the original index

                with cols[col_idx]:
                    # Pass the row data using .loc[] with the actual index
                    display_recall_card(data_df.loc[actual_df_index])


    # Pagination controls below the grid
    if total_pages > 1:
        # Use a form to prevent reruns on every button click until submit
        # This might be overly complex for simple pagination, direct st.button is fine.
        # Using columns for layout
        pagination_cols = st.columns([1, 2, 1])

        with pagination_cols[0]:
            if current_page > 1:
                if st.button("← Précédent", use_container_width=True, key="prev_page_btn"):
                    st.session_state.current_page_recalls -= 1
                    st.rerun()

        with pagination_cols[1]:
            st.markdown(f"""
            <div style="text-align: center; margin-top: 10px;">
                Page {current_page} sur {total_pages}
            </div>
            """, unsafe_allow_html=True)

        with pagination_cols[2]:
            if current_page < total_pages:
                if st.button("Suivant →", use_container_width=True, key="next_page_btn"):
                    st.session_state.current_page_recalls += 1
                    st.rerun()


def create_advanced_filters(df_full_data):
    """Créer des filtres avancés pour les données"""

    # Déterminer la plage de dates disponible dans les données chargées
    min_data_date = START_DATE # Default to API start date
    max_data_date = date.today() # Default to today

    if "date_publication" in df_full_data.columns and not df_full_data["date_publication"].dropna().empty:
        valid_dates = df_full_data["date_publication"].dropna()
        if not valid_dates.empty:
             min_data_date = min(valid_dates)
             max_data_date = max(valid_dates)

    # Ensure min_data_date is not after max_data_date if data is weird
    if min_data_date > max_data_date:
        min_data_date = max_data_date


    # Initialiser les filtres dans l'état de session
    # Use the data's min/max date as the default range, but respect user changes
    if "date_filter_start" not in st.session_state:
        st.session_state.date_filter_start = min_data_date
    if "date_filter_end" not in st.session_state:
        st.session_state.date_filter_end = max_data_date

    # S'assurer que les dates sont des objets date et non datetime
    # This step might be redundant if load_to_dataframe already ensures date objects, but safe.
    for date_key in ["date_filter_start", "date_filter_end"]:
        if isinstance(st.session_state.get(date_key), datetime):
            st.session_state[date_key] = st.session_state[date_key].date()
        # If it's NaT or None, default it
        if not isinstance(st.session_state.get(date_key), date):
             if date_key == "date_filter_start":
                 st.session_state[date_key] = min_data_date
             else: # date_filter_end
                 st.session_state[date_key] = max_data_date


    # Créer l'expander pour les filtres avancés
    with st.expander("🔍 Filtres avancés", expanded=False):
        col1, col2 = st.columns(2)

        # Filtres de catégorie
        with col1:
            selected_categories = []
            if "categorie_de_produit" in df_full_data.columns:
                unique_categories = sorted(df_full_data["categorie_de_produit"].dropna().unique())
                selected_categories = st.multiselect(
                    "Catégories principales:",
                    options=unique_categories,
                    default=st.session_state.get("selected_categories_filter", []),
                    key="categories_filter",
                    placeholder="Choisir des catégories..."
                )
                # Update session state only if different from default multiselect behavior (or just assign directly)
                # Direct assignment is simpler and safe
                st.session_state.selected_categories_filter = selected_categories


            # Filtrer les sous-catégories en fonction des catégories sélectionnées
            selected_subcategories = []
            if "sous_categorie_de_produit" in df_full_data.columns:
                # If categories are selected, filter subcategories based on those categories
                if selected_categories:
                    subcats_df = df_full_data[df_full_data["categorie_de_produit"].isin(selected_categories)]
                else:
                    subcats_df = df_full_data # If no categories selected, show all subcategories

                unique_subcategories = sorted(subcats_df["sous_categorie_de_produit"].dropna().unique())
                selected_subcategories = st.multiselect(
                    "Sous-catégories:",
                    options=unique_subcategories,
                    default=st.session_state.get("selected_subcategories_filter", []),
                    key="subcategories_filter",
                    placeholder="Choisir des sous-catégories..."
                )
                st.session_state.selected_subcategories_filter = selected_subcategories


        # Filtres de risques et date
        with col2:
            selected_risks = []
            if "risques_encourus" in df_full_data.columns:
                # Count risks, handle NaNs, take top N
                risk_counts = df_full_data["risques_encourus"].fillna("Non spécifié").value_counts()
                # Filter out "Non spécifié" from options if desired, or limit total options
                unique_risks_options = sorted(risk_counts[risk_counts.index != "Non spécifié"].index.tolist() + (["Non spécifié"] if "Non spécifié" in risk_counts.index else [])) # Keep Non spécifié at the end if present
                # Limit to top N risks if list is too long
                if len(unique_risks_options) > 100: # Arbitrary limit to keep dropdown manageable
                     unique_risks_options = sorted(risk_counts.nlargest(100).index.tolist())

                selected_risks = st.multiselect(
                    "Types de risques:",
                    options=unique_risks_options,
                    default=st.session_state.get("selected_risks_filter", []),
                    key="risks_filter",
                    placeholder="Choisir des risques..."
                )
                st.session_state.selected_risks_filter = selected_risks

            # Filtre par période de publication
            # Use the actual data min/max date for the date input range constraints
            selected_dates_tuple = st.date_input(
                "Période de publication:",
                value=(st.session_state.date_filter_start, st.session_state.date_filter_end),
                min_value=min_data_date,
                max_value=max_data_date,
                key="date_range_filter"
            )

            # Handle the case where the user selects a single date, result is a single date object
            if isinstance(selected_dates_tuple, tuple) and len(selected_dates_tuple) == 2:
                # Ensure start is before end
                start_date_input, end_date_input = selected_dates_tuple
                if start_date_input > end_date_input:
                     st.warning("La date de début ne peut pas être postérieure à la date de fin. Ajustement automatique.")
                     st.session_state.date_filter_start = end_date_input
                     st.session_state.date_filter_end = start_date_input
                else:
                    st.session_state.date_filter_start = start_date_input
                    st.session_state.date_filter_end = end_date_input
            elif isinstance(selected_dates_tuple, date):
                # Handle single date selection - treat it as start and end date
                 st.session_state.date_filter_start = selected_dates_tuple
                 st.session_state.date_filter_end = selected_dates_tuple


        # Options d'affichage
        st.markdown("---")
        st.markdown("**Options d'affichage**")

        col_display1, col_display2 = st.columns(2)

        with col_display1:
            items_per_page = st.slider(
                "Nombre de rappels par page:",
                min_value=3,
                max_value=30, # Increased max for flexibility
                value=st.session_state.get("items_per_page_filter", DEFAULT_ITEMS_PER_PAGE),
                step=3,
                key="items_per_page_slider"
            )
            st.session_state.items_per_page_filter = items_per_page

        with col_display2:
            recent_days = st.slider(
                "Période pour 'Rappels Récents' (jours):",
                min_value=7,
                max_value=180, # Increased max for flexibility
                value=st.session_state.get("recent_days_filter", DEFAULT_RECENT_DAYS),
                step=1,
                key="recent_days_slider"
            )
            st.session_state.recent_days_filter = recent_days

        # Bouton de réinitialisation
        if st.button("Réinitialiser tous les filtres", type="secondary", use_container_width=True, key="reset_filters_btn"):
            # Reset specific filter session state keys
            for key in [
                "selected_categories_filter", "selected_subcategories_filter",
                "selected_risks_filter", "search_term"
            ]:
                if key in st.session_state:
                    del st.session_state[key]

            # Reset date filter to the full range of currently loaded data
            st.session_state.date_filter_start = min_data_date
            st.session_state.date_filter_end = max_data_date

            # Reset pagination and display options to defaults
            st.session_state.current_page_recalls = 1
            st.session_state.items_per_page_filter = DEFAULT_ITEMS_PER_PAGE
            st.session_state.recent_days_filter = DEFAULT_RECENT_DAYS

            st.rerun() # Rerun to apply changes and filters


    # Affichage des filtres actifs
    active_filters = []

    if st.session_state.get("search_term"):
        active_filters.append(f"Recherche: \"{st.session_state.search_term}\"")

    if selected_categories:
        # Truncate list for display if too long
        cat_display = ", ".join(selected_categories[:3])
        if len(selected_categories) > 3:
            cat_display += f", +{len(selected_categories)-3} autres"
        active_filters.append(f"Catégories: {cat_display}")

    if selected_subcategories:
        # Truncate list for display if too long
        subcat_display = ", ".join(selected_subcategories[:3])
        if len(selected_subcategories) > 3:
             subcat_display += f", +{len(selected_subcategories)-3} autres"
        active_filters.append(f"Sous-catégories: {subcat_display}")

    if selected_risks:
        # Truncate list for display if too long
        risk_display = ", ".join(selected_risks[:3])
        if len(selected_risks) > 3:
             risk_display += f", +{len(selected_risks)-3} autres"
        active_filters.append(f"Risques: {risk_display}")


    # Check if dates are different from the currently loaded data's full range
    current_filter_start = st.session_state.get("date_filter_start")
    current_filter_end = st.session_state.get("date_filter_end")

    if isinstance(current_filter_start, date) and isinstance(current_filter_end, date):
         # Check if the filter range is smaller than the data range
         if current_filter_start > min_data_date or current_filter_end < max_data_date:
            start_str = current_filter_start.strftime("%d/%m/%Y")
            end_str = current_filter_end.strftime("%d/%m/%Y")
            active_filters.append(f"Période: {start_str} - {end_str}")
    elif active_filters: # If dates are invalid but other filters exist, still show the bar
         active_filters.append("Période: Date(s) invalide(s)")


    if active_filters:
        st.markdown("""
        <div class="active-filters">
            <strong>Filtres actifs:</strong>
        """, unsafe_allow_html=True)

        for f in active_filters:
            st.markdown(f"""
            <span class="filter-tag">{f}</span>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='font-style: italic;'>Aucun filtre actif (affichant toutes les données chargées)</p>", unsafe_allow_html=True)


    # Return the filter values
    return (
        selected_categories,
        selected_subcategories,
        selected_risks,
        (st.session_state.date_filter_start, st.session_state.date_filter_end),
        st.session_state.items_per_page_filter
    )


def create_visualizations(data_df):
    """Créer des visualisations interactives à partir des données"""
    if data_df.empty:
        st.info("Données insuffisantes ou inexistantes pour créer des visualisations. Veuillez ajuster vos filtres.")
        return

    # Préparation des données pour les visualisations (date)
    df_vis = data_df.copy() # Work on a copy

    if "date_publication" in df_vis.columns:
        # Ensure date column is datetime type for resampling/plotting
        # Use errors='coerce' to turn invalid dates into NaT
        df_vis["date_publication_dt"] = pd.to_datetime(df_vis["date_publication"], errors="coerce")
        # Drop rows where date conversion failed
        df_vis.dropna(subset=["date_publication_dt"], inplace=True)

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Tendances Temporelles", "🍩 Répartition par Catégorie/Marque", "⚠️ Analyse des Risques"])

    # Tab 1: Temporal Trends
    with tab1:
        st.subheader("Évolution des Rappels dans le Temps")

        if "date_publication_dt" in df_vis.columns and not df_vis["date_publication_dt"].empty:
            # Grouping by month
            # Ensure all months in the range are included, even if no recalls
            min_date_data = df_vis["date_publication_dt"].min().to_period("M").to_timestamp()
            max_date_data = df_vis["date_publication_dt"].max().to_period("M").to_timestamp() + pd.DateOffset(months=1) - pd.Timedelta(days=1)

            # Create a full date range index
            date_range = pd.date_range(start=min_date_data, end=max_date_data, freq="MS")

            # Group and count
            monthly_counts = df_vis.groupby(pd.Grouper(key="date_publication_dt", freq="M")).size()
            # Reindex to include all months in range and fill missing with 0
            monthly_data = monthly_counts.reindex(date_range, fill_value=0).reset_index()
            monthly_data.columns = ["date", "count"]
            # Format date for display if needed, but Plotly handles datetime well
            monthly_data["month_year"] = monthly_data["date"].dt.strftime("%Y-%m")


            # Create the chart
            fig_trend = px.line(
                monthly_data,
                x="date",
                y="count",
                title="Nombre de rappels par mois",
                labels={"date": "Mois", "count": "Nombre de rappels"},
                markers=True
            )

            fig_trend.update_layout(
                xaxis_title="Mois",
                yaxis_title="Nombre de rappels",
                hovermode="x unified",
                xaxis_tickangle=-45,
                # Add range slider? Can be too much for small charts
                # xaxis=dict(rangeselector=dict(buttons=list([
                #     dict(count=1, label="1m", step="month", stepmode="backward"),
                #     dict(count=6, label="6m", step="month", stepmode="backward"),
                #     dict(count=1, label="YTD", step="year", stepmode="todate"),
                #     dict(count=1, label="1y", step="year", stepmode="backward"),
                #     dict(step="all")
                # ])), rangeslider=dict(visible=True), type="date")
            )

            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("Données temporelles insuffisantes pour cette visualisation.")

    # Tab 2: Distribution by Category/Brand
    with tab2:
        st.subheader("Répartition par Catégorie et Marque")

        col1, col2 = st.columns(2)

        with col1:
            if "sous_categorie_de_produit" in df_vis.columns and not df_vis["sous_categorie_de_produit"].dropna().empty:
                # Top 10 subcategories
                subcat_counts = df_vis["sous_categorie_de_produit"].value_counts().nlargest(10)

                if not subcat_counts.empty:
                    fig_subcat = px.pie(
                        names=subcat_counts.index,
                        values=subcat_counts.values,
                        title="Top 10 Sous-Catégories",
                        hole=0.4
                    )
                    fig_subcat.update_traces(textposition="inside", textinfo="percent+label", insidetextfont=dict(color="white"), sort=False)
                    fig_subcat.update_layout(legend_title_text="Sous-catégorie")
                    st.plotly_chart(fig_subcat, use_container_width=True)
                else:
                    st.info("Pas assez de données pour la visualisation des sous-catégories.")
            else:
                 st.info("Aucune donnée de sous-catégorie pour cette visualisation.")


        with col2:
            if "nom_de_la_marque_du_produit" in df_vis.columns and not df_vis["nom_de_la_marque_du_produit"].dropna().empty:
                # Top 10 brands
                brand_counts = df_vis["nom_de_la_marque_du_produit"].value_counts().nlargest(10)

                if not brand_counts.empty:
                    fig_brands = px.bar(
                        x=brand_counts.index,
                        y=brand_counts.values,
                        title="Top 10 Marques",
                        labels={"x": "Marque", "y": "Nombre de rappels"},
                        color=brand_counts.values,
                        color_continuous_scale="Blues"
                    )
                    fig_brands.update_layout(xaxis_tickangle=-45, showlegend=False)
                    st.plotly_chart(fig_brands, use_container_width=True)
                else:
                    st.info("Pas assez de données pour la visualisation des marques.")
            else:
                 st.info("Aucune donnée de marque pour cette visualisation.")

    # Tab 3: Risk Analysis
    with tab3:
        st.subheader("Analyse des Types de Risques")

        if "risques_encourus" in df_vis.columns and not df_vis["risques_encourus"].dropna().empty:
            # Top 15 risks (increased from 10 for more detail)
            risk_counts = df_vis["risques_encourus"].value_counts().nlargest(15)

            if not risk_counts.empty:
                fig_risks = px.bar(
                    y=risk_counts.index,
                    x=risk_counts.values,
                    orientation="h",
                    title="Types de Risques les plus fréquents",
                    labels={"y": "Type de risque", "x": "Nombre de rappels"},
                    color=risk_counts.values,
                    color_continuous_scale="Reds"
                )
                fig_risks.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
                st.plotly_chart(fig_risks, use_container_width=True)

                # Evolution of main risks over time
                if "date_publication_dt" in df_vis.columns and not df_vis["date_publication_dt"].empty:
                    st.subheader("Évolution des Principaux Risques")

                    # Select the top 5 most common risks (excluding "Non spécifié")
                    top_risks_for_trend = df_vis["risques_encourus"].value_counts()
                    top_risks_for_trend = top_risks_for_trend[top_risks_for_trend.index != "Non spécifié"].nlargest(5).index.tolist()

                    if top_risks_for_trend:
                         # Filter data for these risks
                        top_risks_df = df_vis[df_vis["risques_encourus"].isin(top_risks_for_trend)].copy()

                        if not top_risks_df.empty:
                            # Group by month and risk
                            # Ensure all months in the range for *these risks* are included
                            min_date_trend = top_risks_df["date_publication_dt"].min().to_period("M").to_timestamp()
                            max_date_trend = top_risks_df["date_publication_dt"].max().to_period("M").to_timestamp() + pd.DateOffset(months=1) - pd.Timedelta(days=1)
                            date_range_trend = pd.date_range(start=min_date_trend, end=max_date_trend, freq="MS")

                            risk_evolution = top_risks_df.groupby([
                                pd.Grouper(key="date_publication_dt", freq="M"),
                                "risques_encourus"
                            ]).size().reset_index()
                            risk_evolution.columns = ["date", "risque", "count"]

                            # Create a MultiIndex from full date range and top risks list
                            full_index = pd.MultiIndex.from_product([date_range_trend, top_risks_for_trend], names=["date", "risque"])

                            # Reindex the evolution data to fill missing month/risk combinations with 0
                            risk_evolution = risk_evolution.set_index(["date", "risque"]).reindex(full_index, fill_value=0).reset_index()


                            # Create the evolution chart
                            fig_risk_evol = px.line(
                                risk_evolution,
                                x="date",
                                y="count",
                                color="risque",
                                title="Évolution mensuelle des Principaux Risques",
                                labels={"date": "Mois", "count": "Nombre de rappels", "risque": "Type de risque"},
                                markers=True
                            )
                            fig_risk_evol.update_layout(
                                xaxis_title="Mois",
                                yaxis_title="Nombre de rappels",
                                hovermode="x unified",
                                xaxis_tickangle=-45,
                                legend_title_text="Risque"
                            )
                            st.plotly_chart(fig_risk_evol, use_container_width=True)
                        else:
                             st.info("Aucun rappel pour les principaux risques sélectionnés dans la période filtrée.")
                    else:
                         st.info("Moins de 5 types de risques distincts trouvés pour afficher la tendance.")

            else:
                st.info("Pas assez de données pour l'analyse des risques.")
        else:
            st.warning("Données sur les risques insuffisantes ou inexistantes pour cette visualisation.")


def setup_groq_assistant():
    """Configure l'assistant IA Groq"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 Configuration de l'Assistant IA")

    # Vérifier si l'instance de GroqAssistant existe déjà
    if "groq_assistant" not in st.session_state:
        st.session_state.groq_assistant = GroqAssistant()

    # Check if groq package is installed first
    if st.session_state.groq_assistant.Groq is None:
         st.sidebar.error("Assistant IA désactivé: Le package 'groq' n'est pas installé.")
         return False # Cannot proceed if package is missing


    # État d'expansion par défaut du panneau: expanded si la clé n'est pas encore valide
    default_expanded = not st.session_state.groq_assistant.is_ready()

    with st.sidebar.expander("Configurer votre clé API Groq", expanded=default_expanded):
        # Récupérer la clé actuelle depuis l'état de session
        current_key = st.session_state.get("groq_api_key", "")

        # Champ de saisie de la clé
        new_key = st.text_input(
            "Votre clé API Groq:",
            type="password",
            value=current_key,
            help=(
                "Obtenez votre clé sur [console.groq.com](https://console.groq.com/keys). "
                "La clé est stockée uniquement dans votre session."
            ),
            key="groq_api_key_input"
        )

        # If the input field value changes OR the state's key changes (e.g., from default)
        # This logic ensures set_api_key is called only when the key string potentially changes.
        if new_key != current_key or (current_key and not st.session_state.groq_assistant.is_authenticated):
            st.session_state["groq_api_key"] = new_key # Store the new key in session state immediately
            if new_key:
                with st.spinner("Validation de la clé API..."):
                    # Test the new key
                    is_valid = st.session_state.groq_assistant.set_api_key(new_key)

                    if is_valid:
                        st.success("✅ Clé API valide et authentification réussie.")
                    else:
                        st.error(
                            "❌ La clé API semble invalide ou l'authentification a échoué. "
                            "Vérifiez votre clé et réessayez."
                        )
            else:
                # Key was cleared
                st.session_state.groq_assistant.is_authenticated = False # Ensure state reflects no key
                st.info("Entrez votre clé API Groq pour activer l'assistant.")


        # Options supplémentaires si la clé est configurée ET valide
        if st.session_state.groq_assistant.is_ready():
            model_options = {
                "llama3-70b-8192": "Llama 3 (70B) - Puissant",
                "llama3-8b-8192": "Llama 3 (8B) - Rapide",
                "mixtral-8x7b-32768": "Mixtral (8x7B) - Large contexte",
                "gemma-7b-it": "Gemma (7B) - Léger"
            }

            # Get default model, ensuring it's in options or default to a safe one
            default_model = st.session_state.get("groq_model", "llama3-8b-8192")
            if default_model not in model_options:
                 default_model = "llama3-8b-8192" # Fallback

            selected_model = st.selectbox(
                "Choisir un modèle:",
                options=list(model_options.keys()),
                format_func=lambda x: model_options.get(x, x), # Handle potential missing key
                index=list(model_options.keys()).index(default_model), # Set index based on default
                key="groq_model_select"
            )
            st.session_state["groq_model"] = selected_model # Store selected model

            with st.popover("Options avancées"):
                st.slider(
                    "Température (Créativité):",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get("groq_temperature", 0.2),
                    step=0.05, # More granular step
                    key="groq_temperature_slider", # Use a different key to avoid conflict if needed
                    help="Une valeur plus élevée rend la réponse plus aléatoire/créative."
                )
                st.session_state["groq_temperature"] = st.session_state["groq_temperature_slider"] # Store in state


                st.slider(
                    "Tokens max en réponse:",
                    min_value=256,
                    max_value=2048, # Max tokens often limited by model context or API
                    value=st.session_state.get("groq_max_tokens", 1024),
                    step=128, # Smaller step
                    key="groq_max_tokens_slider",
                     help="Limite la longueur de la réponse de l'IA."
                )
                st.session_state["groq_max_tokens"] = st.session_state["groq_max_tokens_slider"] # Store in state


                st.slider(
                    "Éléments de contexte (exemples):",
                    min_value=5,
                    max_value=30, # Limit max context items to keep prompt size manageable
                    value=st.session_state.get("groq_context_items", 15),
                    step=1,
                    key="groq_context_items_slider",
                    help="Nombre de rappels récents inclus dans le contexte de l'IA."
                )
                st.session_state["groq_context_items"] = st.session_state["groq_context_items_slider"] # Store in state


    # Display current status of the configuration
    if st.session_state.groq_assistant.is_ready():
        selected_model_name_display = model_options.get(
            st.session_state.get("groq_model", "llama3-8b-8192"),
            "Modèle inconnu"
        )
        st.sidebar.caption(f"🟢 Assistant IA prêt • Modèle: {selected_model_name_display}")
        return True
    else:
        st.sidebar.caption("🔴 Assistant IA non configuré ou erreur de connexion")
        return False


def display_ai_assistant(df_data):
    """Affiche l'interface d'assistant IA"""
    st.subheader("💬 Assistant IA - Analyse des Rappels")

    st.markdown(
        "<p style='color: #666; font-style: italic; font-size: 0.9em;'>"
        "Posez vos questions sur les rappels actuellement affichés dans les résultats. "
        "L'assistant IA analysera ces données et vous fournira des insights pertinents. "
        "**Attention:** Son analyse se limite aux données visibles avec vos filtres."
        "</p>",
        unsafe_allow_html=True
    )

    # Check if the assistant is configured and ready
    # The setup_groq_assistant function should be called BEFORE this function,
    # and its return value or session state should be checked.
    # We re-check the state here defensively.
    if not st.session_state.get("groq_assistant") or not st.session_state.groq_assistant.is_ready():
        st.warning(
            "⚠️ L'assistant IA n'est pas configuré ou prêt. Veuillez configurer votre clé API Groq dans la barre latérale.",
            icon="⚠️"
        )
        return

    # Initialiser l'historique de chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Bonjour ! Je suis l'assistant RappelConso Insight. Je peux analyser les rappels actuellement affichés. Comment puis-je vous aider ?"
            }
        ]

    # Propositions de questions
    st.markdown("**Questions suggérées:**")

    # Define suggestions dynamically based on available data/filters
    suggestions = ["Quels sont les principaux risques parmi ces rappels ?"]
    if "nom_de_la_marque_du_produit" in df_data.columns and not df_data["nom_de_la_marque_du_produit"].dropna().empty:
        suggestions.append("Quelles marques apparaissent le plus souvent ?")
    if "sous_categorie_de_produit" in df_data.columns and not df_data["sous_categorie_de_produit"].dropna().empty:
         suggestions.append("Quelles sous-catégories sont les plus représentées ?")
    if "motif_du_rappel" in df_data.columns and not df_data["motif_du_rappel"].dropna().empty:
         suggestions.append("Quels sont les motifs de rappel les plus fréquents ?")
    if "date_publication" in df_data.columns and not df_data["date_publication"].dropna().empty:
         suggestions.append("Y a-t-il une tendance temporelle notable ?")

    # Add some generic ones if specific columns are missing
    if len(suggestions) < 3:
         suggestions.extend([
             "Donne-moi un résumé des rappels listés.",
             "Trouve les rappels liés aux allergènes dans cette liste."
         ])
         suggestions = list(set(suggestions)) # Remove duplicates

    suggestion_cols = st.columns(3)

    # Display suggestions using buttons
    for i, suggestion in enumerate(suggestions):
        with suggestion_cols[i % 3]: # Arrange buttons in 3 columns
            # Use a form to prevent rerunning on every button click, but still submit
            # A simple button might be fine too, depends on desired behavior
            # Let's stick to simple button and rerun for immediate feedback
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                # When a suggestion button is clicked, update the query input and trigger processing
                st.session_state.query = suggestion # Update the query text input value
                # Trigger the processing function directly
                process_query(df_data, suggestion)
                # Rerun to update chat history display and clear input field
                st.rerun()


    # Afficher l'historique des messages
    st.markdown("---")

    # Display messages from chat history
    # Using a placeholder for the chat container
    chat_placeholder = st.empty()

    with chat_placeholder.container():
        for i, message in enumerate(st.session_state.chat_history):
            role_class = "user" if message["role"] == "user" else "assistant"

            st.markdown(f"""
            <div class="chat-message {role_class}">
                <strong>{"Vous" if role_class == "user" else "Assistant IA"}:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)


    # Chat input area at the bottom
    st.markdown("---") # Separator before input

    # Input form for new query to prevent reruns while typing
    with st.form(key="chat_form", clear_on_submit=True):
        query = st.text_input(
            "Posez votre question sur les rappels affichés:",
            key="query_input_text", # Key for the text input widget itself
            placeholder="Ex: Quels sont les rappels les plus fréquents par catégorie ?",
            label_visibility="collapsed" # Hide the default label
        )
        send_button = st.form_submit_button("Envoyer")

    # Process the query when the send button is clicked and query is not empty
    if send_button and query:
        # Add the user question to history
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Process the question and get the response
        process_query(df_data, query)

        # Rerun to update the chat history display (handled by the form's clear_on_submit)
        # st.rerun() # Rerun is automatically handled by form submission with clear_on_submit=True


def process_query(df_data, query):
    """Traite une question utilisateur avec l'assistant IA"""
    # Ensure groq_assistant and its readiness are checked before calling this
    assistant_instance = st.session_state.get("groq_assistant")
    if not assistant_instance or not assistant_instance.is_ready():
        response = "Assistant IA non configuré ou non prêt."
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        logger.error(response)
        return # Exit if not ready

    if df_data.empty:
        response = "Je ne peux pas répondre car aucune donnée de rappel n'est disponible avec les filtres actuels. Veuillez ajuster vos filtres."
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        logger.warning("AI query attempted with empty DataFrame.")
        return

    # Retrieve assistant parameters from session state
    model = st.session_state.get("groq_model", "llama3-8b-8192") # Default if not set
    temperature = st.session_state.get("groq_temperature", 0.2)
    max_tokens = st.session_state.get("groq_max_tokens", 1024)
    max_context_items = st.session_state.get("groq_context_items", 15)

    logger.info(f"Processing AI query: '{query[:50]}...'")
    logger.info(f"AI params: model={model}, temp={temperature}, max_tokens={max_tokens}, context_items={max_context_items}")

    # The spinner should ideally be shown near the chat area, not globally
    # Since this is called by a button/form submission that triggers a rerun,
    # a global spinner might be okay, but placing it near the chat is better UX.
    # Streamlit's `with st.spinner(...)` is good for this.
    # It needs to be *in* the Streamlit script execution flow.
    # We can show a message in the chat area itself while processing.

    # Placeholder for assistant thinking message
    thinking_message_placeholder = st.empty()
    thinking_message_placeholder.markdown("🤖 L'assistant réfléchit...", unsafe_allow_html=True)


    try:
        # Call the assistant's query method
        result = assistant_instance.query_assistant(
            query,
            df_data, # Pass the filtered DataFrame
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_context_items=max_context_items
        )

        # Clear the thinking message
        thinking_message_placeholder.empty()

        if result["success"]:
            response = result["response"]
            logger.info(f"AI response received (success). Metrics: {result.get('metrics', {})}")
        else:
            response = f"Désolé, je n'ai pas pu traiter votre question. Erreur: {result.get('error', 'Inconnue')}"
            logger.error(f"AI query failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        # Catch any errors during the process_query logic itself
        thinking_message_placeholder.empty()
        response = f"Une erreur interne s'est produite lors du traitement de la question: {e}"
        logger.error(f"Internal error during process_query: {e}")


    # Add the assistant's response to history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Note: st.rerun() is typically called after modifying session state
    # to refresh the UI. If process_query is called by a button/form submit,
    # the rerun might happen automatically, but an explicit rerun() here
    # guarantees the UI update after the response is added. However, be mindful
    # of rerun loops. If called from a form submit with clear_on_submit=True,
    # the rerun happens after the function returns. If called from a simple button,
    # rerun() is necessary. Since we are using a form submit button,
    # the explicit rerun() might not be strictly needed right after this function finishes,
    # but if this function were ever called elsewhere (like a suggestion button),
    # it would be. For consistency and clarity, let's assume a rerun is needed
    # *if* this function doesn't inherently cause one (like from a simple button).
    # Given the display_ai_assistant calls process_query and *then* reruns if a suggestion button is clicked,
    # and the form submission inherently reruns, an explicit rerun here might be redundant but harmless.
    # Let's omit the explicit rerun() here as it's likely handled by the caller context.


def debug_mode_ui():
    """Interface pour le mode débogage"""
    if not st.session_state.get("debug_mode", False):
        return

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Mode Débogage")

    debug_options = st.sidebar.expander("Options de débogage", expanded=False)
    with debug_options:
        # Afficher les informations sur l'état de session
        if st.checkbox("Afficher l'état de session", value=False, key="show_session_state_debug"):
            # Convert items to string representation for display
            st.json({k: (str(v) if not isinstance(v, (dict, list, pd.DataFrame)) else "...") for k, v in st.session_state.items()})
            # Optionally display dataframes separately if needed
            if st.checkbox("Afficher DataFrame (si chargé)", value=False, key="show_dataframe_debug"):
                 if "data_df" in st.session_state and not st.session_state.data_df.empty:
                      st.write(st.session_state.data_df.head())
                      st.write(f"DataFrame shape: {st.session_state.data_df.shape}")
                 else:
                      st.info("DataFrame not loaded or empty.")


        # Simulation d'erreurs pour tester la robustesse
        if st.checkbox("Simuler une erreur API", value=st.session_state.get("simulate_api_error", False), key="simulate_api_error_checkbox"):
            st.session_state.simulate_api_error = st.session_state.simulate_api_error_checkbox
            if st.session_state.simulate_api_error:
                 # This simulation approach isn't ideal as it just displays a message
                 # A better way would be to raise an exception within the API fetch function
                 # based on this state variable. But for simple UI test, this is okay.
                 st.error("La simulation d'erreur API est active. Le prochain appel API *pourrait* échouer.")
            else:
                 st.info("La simulation d'erreur API est inactive.")


        # Vider le cache Streamlit
        if st.button("Vider le cache Streamlit", key="clear_streamlit_cache_btn"):
            st.cache_data.clear()
            st.success("Cache vidé!")
            # Clear data from session state as well, as load_to_dataframe is cached
            if "data_df" in st.session_state:
                del st.session_state.data_df
            st.info("Données en session effacées. Rechargement nécessaire.")
            st.rerun() # Rerun after clearing cache and data


    # Button to toggle debug mode
    # This button is in the sidebar, handled at the start of main()
    pass # The toggle checkbox is handled at the beginning of main()


def export_data(data_df):
    """Exporte les données au format Excel"""
    if data_df.empty:
        st.warning("Aucune donnée filtrée à exporter")
        return

    # Create an in-memory buffer for the Excel file
    buffer = io.BytesIO()

    try:
        # Use ExcelWriter with the buffer
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Prepare DataFrame for export: ensure dates are formatted, handle NaNs
            export_df = data_df.copy()

            # Convert date objects to string for consistent Excel formatting, handling NaT/NaN
            if "date_publication" in export_df.columns:
                 export_df["date_publication"] = export_df["date_publication"].apply(
                     lambda x: x.strftime("%Y-%m-%d") if isinstance(x, date) else ""
                 )

            # Fill potential other NaNs with empty strings for cleaner export
            export_df = export_df.fillna("")

            # Map column names to user-friendly names for export
            export_df.rename(columns=UI_FIELD_NAMES, inplace=True)

            # Write the DataFrame to Excel
            export_df.to_excel(writer, sheet_name="Rappels_Filtrés", index=False)

            # Optional: Auto-adjust column widths in the Excel file
            try:
                worksheet = writer.sheets["Rappels_Filtrés"]
                for i, col in enumerate(export_df.columns):
                    # Calculate max length of data in the column or header length
                    max_len = max(export_df[col].astype(str).str.len().max(), len(str(col))) + 2 # Add padding
                    # Limit max width to avoid excessively wide columns
                    max_len = min(max_len, 50)
                    worksheet.set_column(i, i, max_len)
            except Exception as e:
                 logger.warning(f"Could not auto-adjust column widths in Excel: {e}")


    except Exception as e:
        st.error(f"Une erreur s'est produite lors de la création du fichier Excel: {e}")
        logger.error(f"Excel export error: {e}")
        return # Exit function if Excel creation fails


    # Get the binary data from the buffer
    buffer.seek(0)

    # Generate a filename with the current date
    filename = f"rappelconso_export_{date.today().strftime('%Y-%m-%d')}.xlsx"

    # Offer the download button
    st.download_button(
        label="📥 Exporter les données filtrées (Excel)",
        data=buffer,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", # Standard MIME type for .xlsx
        key="download_button",
        help=f"Télécharger les {len(data_df)} rappels actuellement affichés au format Excel"
    )


# --- MAIN APPLICATION FUNCTION ---
def main():
    # Initialisation du mode débogage (check URL parameters or session state)
    # Let's check URL params first for easy activation, then fall back to session state
    query_params = st.query_params
    debug_from_url = query_params.get("debug", "false").lower() == "true"

    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = debug_from_url
    # If URL param is set, override session state
    elif debug_from_url:
        st.session_state.debug_mode = True

    # Toggle checkbox in sidebar controls the session state
    debug_toggle_sidebar = st.sidebar.checkbox(
        "Mode débogage",
        value=st.session_state.debug_mode,
        key="debug_mode_sidebar_checkbox" # Unique key for the widget
    )
    # Update session state if sidebar checkbox is changed
    if debug_toggle_sidebar != st.session_state.debug_mode:
         st.session_state.debug_mode = debug_toggle_sidebar
         # If turning debug mode off, clear the URL parameter
         if not st.session_state.debug_mode:
              query_params.pop("debug", None)
              st.query_params(query_params)
         # Rerun to apply debug mode changes (e.g. show/hide debug options)
         st.rerun()


    if st.session_state.debug_mode:
        debug_mode_ui()


    # Configuration de la barre latérale pour les paramètres de recherche globaux
    st.sidebar.image(LOGO_URL, use_container_width=True)
    st.sidebar.title("RappelConso Insight")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Paramètres de Chargement")

    # Date de début de la période pour le CHARGEMENT API (non le filtre)
    # This date determines *which* records are fetched initially.
    # The date *filter* then operates on these loaded records.
    default_load_start_date = st.session_state.get("load_start_date", START_DATE)
    if not isinstance(default_load_start_date, date):
        default_load_start_date = START_DATE # Ensure it's a date object

    load_start_date_input = st.sidebar.date_input(
        "Charger les rappels depuis:",
        value=default_load_start_date,
        min_value=date(2022, 1, 1), # API V2 data starts from 2022-01-01
        max_value=date.today(),
        key="load_start_date_input",
        help="Cette date détermine le volume initial de données chargées depuis l'API. "
             "Un filtre de date plus précis est disponible dans les filtres avancés."
    )

    # If the load start date changes, clear cached data and trigger reload
    if load_start_date_input != st.session_state.get("load_start_date"):
        st.session_state.load_start_date = load_start_date_input
        # Clear cached data whenever the load date changes
        st.cache_data.clear()
        if "data_df" in st.session_state:
            del st.session_state.data_df
        # Reset filter dates as well, as the data range has changed
        if "date_filter_start" in st.session_state:
            del st.session_state.date_filter_start
        if "date_filter_end" in st.session_state:
             del st.session_state.date_filter_end
        # Reset pagination
        st.session_state.current_page_recalls = 1
        st.rerun() # Rerun to trigger data reload

    # Note: The API only allows filtering by 'date_publication'.
    # We load all 'Alimentation' data since the selected load_start_date
    # and then apply further filters (including date range) in pandas.
    # This is more efficient than making multiple API calls for different filter combinations.
    # The API query for initial load is fixed: 'Alimentation' since `load_start_date`.

    # Chargement des données
    df_alim = pd.DataFrame() # Initialize an empty DataFrame
    try:
        # Construct the base API query for initial load (only category and start date)
        # End date for API load is always today for the initial fetch
        api_start_date_str = st.session_state.load_start_date.strftime("%Y-%m-%d")
        api_end_date_str = date.today().strftime("%Y-%m-%d") # Load up to today's data

        # API query for initial load
        base_where_clause = RappelConsoAPI.build_query(
            category="Alimentation",
            start_date=api_start_date_str,
            end_date=api_end_date_str # This end date is for the *initial load*
        )

        if st.session_state.debug_mode:
             st.sidebar.markdown(f"**Clause WHERE (Chargement API):**\n```\n{base_where_clause}\n```")

        # Load data using the cached function
        # The cache key will be the base_where_clause, so data is reloaded
        # only if the loading parameters change.
        df_alim = RappelConsoAPI.load_to_dataframe(base_where_clause)

        if df_alim.empty:
            st.warning("Aucun rappel 'Alimentation' trouvé depuis la date sélectionnée.")
            # Still allow filters/AI display, they will show no data message
            filtered_df = pd.DataFrame() # Ensure filtered_df is empty if df_alim is
            items_per_page = DEFAULT_ITEMS_PER_PAGE # Use default if no data

            # Need to call filter setup to get filter widgets displayed,
            # even if no data is loaded. Pass the empty df_alim.
            selected_categories, selected_subcategories, selected_risks, selected_dates, items_per_page = create_advanced_filters(df_alim)


        else:
            # Data loaded successfully, proceed with filtering
            st.sidebar.success(f"✅ {len(df_alim)} rappels 'Alimentation' chargés depuis le {st.session_state.load_start_date.strftime('%d/%m/%Y')}.", icon="✅")

            # Barre de recherche
            # Search term input should be placed near the filters it affects
            st.markdown("#### Recherche Rapide")
            search_term = st.text_input(
                "Rechercher par mot-clé (produit, marque, risque...):",
                value=st.session_state.get("search_term", ""),
                placeholder="Ex: fromage, listeria, carrefour...",
                key="search_input"
            )
            # Update session state search term if input changes
            if search_term != st.session_state.get("search_term_state_check", ""):
                 st.session_state.search_term = search_term
                 st.session_state.search_term_state_check = search_term # Helper to detect input change
                 st.session_state.current_page_recalls = 1 # Reset pagination on search
                 st.rerun() # Rerun to apply search filter

            # Advanced filters - get selected values and items_per_page
            selected_categories, selected_subcategories, selected_risks, selected_dates, items_per_page = create_advanced_filters(df_alim)

            # Apply the filters to the loaded DataFrame
            filtered_df = df_alim.copy()

            # 1. Filtre par recherche textuelle
            if st.session_state.get("search_term"):
                search_term = st.session_state.search_term.lower()
                # Columns to search within
                search_columns = [
                    "nom_de_la_marque_du_produit", "nom_commercial",
                    "modeles_ou_references", "risques_encourus",
                    "motif_du_rappel", "sous_categorie_de_produit", "distributeurs"
                ]
                # Filter to only include columns that actually exist in the DataFrame
                existing_search_columns = [col for col in search_columns if col in filtered_df.columns]

                if existing_search_columns:
                    # Create a mask by checking if the search term is in any of the specified columns (as strings)
                    # Fillna('') is important to avoid errors on NaN values
                    mask = filtered_df[existing_search_columns].fillna("").astype(str).apply(
                        lambda x: x.str.lower().str.contains(search_term, na=False) # na=False ensures NaNs don't match
                    ).any(axis=1) # True if term found in *any* of the columns for that row

                    filtered_df = filtered_df[mask]
                else:
                    logger.warning("Search columns do not exist in the DataFrame.")


            # 2. Filtre par catégorie
            if selected_categories:
                # Ensure the column exists before filtering
                if "categorie_de_produit" in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df["categorie_de_produit"].isin(selected_categories)]
                else:
                    logger.warning("'categorie_de_produit' column not found for filtering.")
                    filtered_df = pd.DataFrame() # No data matches if category filter cannot be applied


            # 3. Filtre par sous-catégorie
            if selected_subcategories:
                 if "sous_categorie_de_produit" in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df["sous_categorie_de_produit"].isin(selected_subcategories)]
                 else:
                    logger.warning("'sous_categorie_de_produit' column not found for filtering.")
                    filtered_df = pd.DataFrame() # No data matches if subcategory filter cannot be applied


            # 4. Filtre par risque
            if selected_risks:
                 if "risques_encourus" in filtered_df.columns:
                     # Use .fillna('') to handle potential NaN values in the risks column
                    filtered_df = filtered_df[filtered_df["risques_encourus"].fillna("").isin(selected_risks)]
                 else:
                    logger.warning("'risques_encourus' column not found for filtering.")
                    filtered_df = pd.DataFrame() # No data matches if risk filter cannot be applied


            # 5. Filtre par date de publication
            if "date_publication" in filtered_df.columns:
                start_date_filter, end_date_filter = selected_dates

                # Ensure the column is date objects and filter
                # This conversion should already happen in load_to_dataframe, but double-check
                if not all(isinstance(d, date) or pd.isna(d) for d in filtered_df["date_publication"]):
                    filtered_df["date_publication"] = pd.to_datetime(filtered_df["date_publication"], errors="coerce").dt.date


                # Filter out NaT values before applying date comparison
                date_series = filtered_df["date_publication"].dropna()

                if not date_series.empty:
                     mask = (date_series >= start_date_filter) & (date_series <= end_date_filter)
                     # Reindex mask to align with filtered_df before applying
                     filtered_df = filtered_df[mask.reindex(filtered_df.index, fill_value=False)]
                else:
                     # If no valid dates after previous filters, the date filter results in empty df
                     filtered_df = pd.DataFrame()
            else:
                 logger.warning("'date_publication' column not found for filtering.")
                 # If no date column, date filter cannot be applied. Keep current filtered_df.


        # --- Display filtered data and insights ---

        # Tabs for different views
        tab_dashboard, tab_viz, tab_ai = st.tabs(["📊 Tableau de Bord", "📈 Visualisations", "🤖 Assistant IA"])

        # Tab 1: Dashboard
        with tab_dashboard:
            # Display metrics based on FILTERED data
            display_metrics(filtered_df)

            # Export option - placed below metrics
            export_col1, export_col2 = st.columns([3, 1])
            with export_col2:
                 export_data(filtered_df) # Pass the filtered DataFrame

            st.markdown("---")
            # Display paginated recalls based on FILTERED data
            display_paginated_recalls(filtered_df, items_per_page)


        # Tab 2: Visualizations
        with tab_viz:
            create_visualizations(filtered_df) # Pass the filtered DataFrame


        # Tab 3: AI Assistant
        with tab_ai:
            # Setup Groq Assistant (happens in sidebar config, check readiness)
            groq_ready = setup_groq_assistant() # Ensure this is called to update state

            # Display the AI assistant interface if ready
            if groq_ready:
                display_ai_assistant(filtered_df) # Pass the filtered DataFrame for context
            else:
                # setup_groq_assistant already displays warnings/info in sidebar
                # No need to repeat here, display_ai_assistant handles the state check.
                pass


    except requests.exceptions.ConnectionError:
         st.error("Erreur de connexion à l'API RappelConso. Veuillez vérifier votre connexion internet ou réessayer plus tard.")
         logger.exception("Connection Error during API call.")
    except requests.exceptions.Timeout:
         st.error("La requête à l'API RappelConso a dépassé le délai d'attente.")
         logger.exception("Timeout Error during API call.")
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la communication avec l'API RappelConso: {e}")
        logger.exception("Request Error during API call.")
        if st.session_state.debug_mode:
             st.warning("Détails de l'erreur ci-dessous en mode débogage.")
             st.exception(e)
    except Exception as e:
        # Catch any other unexpected errors during data loading, filtering, or display
        st.error(f"Une erreur inattendue s'est produite: {str(e)}")
        logger.exception("An unexpected error occurred in main execution flow.")
        if st.session_state.debug_mode:
            st.warning("Détails de l'erreur ci-dessous en mode débogage.")
            st.exception(e)
        else:
            st.info(
                "Une erreur s'est produite lors du traitement. "
                "Essayez de vider le cache (via le mode débogage) ou de rafraîchir la page."
            )

        # Option to clear cache/data if an error occurs
        if st.button("Vider le cache et recharger", key="retry_after_error_btn"):
            st.cache_data.clear()
            if "data_df" in st.session_state:
                del st.session_state.data_df
            # Also reset filter state to default load dates
            if "date_filter_start" in st.session_state:
                 del st.session_state.date_filter_start
            if "date_filter_end" in st.session_state:
                 del st.session_state.date_filter_end
            st.session_state.current_page_recalls = 1
            st.rerun()


# Lancement de l'application
if __name__ == "__main__":
    main()
