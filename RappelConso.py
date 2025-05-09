import streamlit as st
import pandas as pd
import requests
from datetime import datetime, date, timedelta, timezone
import time
import plotly.express as px
import numpy as np
import math

# --- Configuration de la page ---
st.set_page_config(
    page_title="RappelConso Insight",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS simplifié pour le style de base ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Roboto', sans-serif;
    }

    .header {
        background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .header h1 {
        font-size: 2.5em;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .header p {
        font-size: 1.1em;
        opacity: 0.9;
    }

    .card {
        background-color: white;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease-in-out;
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    .metric {
        text-align: center;
        padding: 1rem;
        background-color: #e9ecef; /* Light grey background */
        border-radius: 8px;
        border-top: 3px solid #2980b9;
        margin-bottom: 1rem; /* Added margin */
    }
    .metric-value {
        font-size: 2.2em;
        font-weight: bold;
        color: #2980b9;
    }
    .metric div {
        font-size: 1em;
        color: #333; /* Darker text for label */
    }

    .recall-card {
        border-left: 4px solid #2980b9;
        padding-left: 0.8rem;
        height: 100%; /* Fill the column height */
    }
    
    .risk-high {
        color: #e74c3c; /* Red */
        font-weight: bold;
    }
    .risk-medium {
        color: #f39c12; /* Orange */
        font-weight: bold;
    }
    .risk-low {
        color: #27ae60; /* Green */
        font-weight: bold;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f0f0;
        border-radius: 4px 4px 0 0;
        gap: 10px;
        padding: 0px 20px;
    }

    .stTabs [data-baseweb="tab"] svg {
        color: #2980b9;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2980b9;
        color: white;
        border-top: 3px solid #1a5276;
    }
    .stTabs [aria-selected="true"] svg {
        color: white;
    }

    .footer {
        margin-top: 3rem; /* Increased margin */
        padding-top: 1.5rem;
        border-top: 1px solid #ccc;
        text-align: center;
        color: #7f8c8d;
        font-size: 0.9em; /* Slightly larger font */
    }

    /* Adjust image container within card */
    .recall-card img {
         margin-top: 0px !important; /* Remove top margin added previously */
    }
     .recall-card div[data-testid="stImage"] {
        margin-top: 0px !important; /* Ensure no margin from streamlit image wrapper */
        margin-bottom: 10px; /* Add some space below the image */
    }

    /* Style for download button */
    .stDownloadButton > button {
        background-color: #27ae60; /* Green */
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1em;
    }
     .stDownloadButton > button:hover {
        background-color: #229954; /* Darker green */
        color: white;
    }


</style>
""", unsafe_allow_html=True)

# --- Mode debug ---
DEBUG = False # Set to True to enable debug logs

def debug_log(message, data=None):
    """Affiche des informations de débogage"""
    if DEBUG:
        st.sidebar.markdown(f"**DEBUG:** {message}")
        if data is not None:
             # Use a more compact representation for debug data in sidebar
            if isinstance(data, (dict, list)):
                 st.sidebar.json(data) # Use json for dict/list
            elif isinstance(data, pd.DataFrame):
                st.sidebar.dataframe(data.head()) # Use dataframe for pandas
            else:
                 st.sidebar.write(data) # Default write for other types


# --- Liste prédéfinie de catégories ---
# Fixée à "alimentation" pour répondre à la demande.
CATEGORIES = ["alimentation"]


# --- Fonction de chargement des données avec cache ---
@st.cache_data(ttl=3600, show_spinner=False) # Cache pendant 1 heure, cache key dépend des paramètres
def load_rappel_data(start_date: date = None, category: str = "alimentation", max_records: int = 1000):
    """
    Charge les données de l'API RappelConso.
    Applique le filtre de date et de catégorie à l'API.
    Force la catégorie à 'alimentation'.
    """
    api_url = "https://data.economie.gouv.fr/api/v2/catalog/datasets/rappelconso-v2-gtin-espaces/records"
    
    # On force la catégorie à 'alimentation' ici pour répondre à la demande spécifique.
    actual_category_to_load = "alimentation" 

    params = {
        "limit": 100, # Limite par page, max 100 dans l'API
        "offset": 0
    }
    
    # Ajouter un filtre de catégorie
    if actual_category_to_load:
        params["refine.categorie_produit"] = actual_category_to_load
    # Note: Pas de 'else' pour charger toutes les catégories car la catégorie est forcée.

    # Ajouter un filtre de date de publication si spécifié
    if start_date:
         start_date_str = start_date.strftime("%Y-%m-%d")
         params["refine.date_publication"] = f">='{start_date_str}'"
    
    all_records = []
    total_count = 0
    
    # Indicateur de progression
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Première requête pour estimer le nombre total
        initial_params = params.copy()
        initial_params["limit"] = 1
        
        status_text.text(f"Connexion à l'API RappelConso (catégorie: '{actual_category_to_load}')...")
        response = requests.get(api_url, params=initial_params, timeout=30)
        response.raise_for_status() # Lève une exception pour les erreurs HTTP
        
        data = response.json()
        total_count = data.get("total_count", 0)
        debug_log(f"Estimation Total Count API (cat: {actual_category_to_load}, date: >={start_date_str}): {total_count}", data)
        
        if total_count == 0:
            status_text.warning(f"Aucun rappel trouvé avec les filtres initiaux (catégorie: '{actual_category_to_load}').")
            progress_bar.progress(1.0)
            return pd.DataFrame()

        estimated_fetch_count = min(total_count, max_records)
        status_text.text(f"Chargement des données ({estimated_fetch_count} maximum)...")
        
        offset = 0
        # Ajuster la condition de boucle pour s'assurer de ne pas dépasser max_records
        while offset < total_count and len(all_records) < max_records:
            params["offset"] = offset
            params["limit"] = min(100, max_records - len(all_records))

            if params["limit"] <= 0:
                 break

            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            
            page_data = response.json()
            records = page_data.get("records", [])
            
            if not records:
                break # Plus de données à charger
            
            # Extraction des champs
            for record in records:
                if "record" in record and "fields" in record["record"]:
                    all_records.append(record["record"]["fields"])
            
            offset += len(records)
            # Mettre à jour la progression basée sur le nombre réel d'enregistrements collectés par rapport à max_records
            progress_bar.progress(min(1.0, len(all_records) / max_records))
            
            time.sleep(0.05)
            
            if len(all_records) >= max_records:
                 status_text.info(f"Limite de {max_records} enregistrements atteinte.")
                 break

    except requests.exceptions.Timeout:
        status_text.error("Erreur: Délai d'attente dépassé lors de la connexion à l'API.")
        progress_bar.empty()
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        status_text.error(f"Erreur API: Impossible de charger les données. {e}")
        debug_log("API Error details", e)
        progress_bar.empty()
        return pd.DataFrame()
    except Exception as e:
        status_text.error(f"Une erreur inattendue est survenue lors du chargement: {e}")
        debug_log("Loading Error details", e)
        progress_bar.empty()
        return pd.DataFrame()

    finally:
        if len(all_records) > 0 and progress_bar:
             progress_bar.progress(1.0)
        if status_text:
             status_text.empty() # Nettoyer le texte de statut
        if progress_bar:
             progress_bar.empty() # Nettoyer la barre de progression
    
    if not all_records:
        st.warning(f"Aucun rappel trouvé avec les filtres spécifiés (catégorie: '{actual_category_to_load}').")
        return pd.DataFrame()
    
    # Créer le DataFrame
    df = pd.DataFrame(all_records)
    
    # Débogage
    debug_log("DataFrame brut après chargement", df.head())
    
    # Standardiser les noms de colonnes pour l'interface et traiter les dates
    column_mapping = {
        "categorie_produit": "categorie",
        "sous_categorie_produit": "sous_categorie",
        "marque_produit": "marque",
        "modeles_ou_references": "modele",
        "motif_rappel": "motif",
        "risques_encourus": "risques",
        "distributeurs": "distributeurs",
        "liens_vers_les_images": "image_urls_raw", # Renommé pour clarté
        "lien_vers_la_fiche_rappel": "fiche_url",
        "date_publication": "date_raw",  # Renommé pour clarté
        "libelle": "nom",
        "id": "id",
        "zone_geographique_de_vente": "zone_vente",
        "conduites_a_tenir_par_le_consommateur": "conduites_a_tenir"
    }
    
    # Appliquer le renommage des colonnes existantes
    df.rename(columns=column_mapping, inplace=True)
    
    # Convertir la colonne date_raw en datetime et extraire la date pour affichage
    if "date_raw" in df.columns:
        # Convertir en datetime, coercer les erreurs en NaT (Not a Time) et s'assurer qu'ils sont en UTC
        df['date'] = pd.to_datetime(df['date_raw'], errors='coerce', utc=True)
        # Extraire la partie date pour l'affichage (ignorer le timezeone pour la conversion str)
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    else:
        df['date'] = pd.NaT # Add dummy datetime col if date_raw is missing
        df['date_str'] = "Date manquante"

    # Extraire la première URL d'image si plusieurs sont présentes
    if "image_urls_raw" in df.columns:
         df['image_url'] = df['image_urls_raw'].apply(lambda x: str(x).split('|')[0].strip() if pd.notna(x) and str(x).strip() else None)
    else:
         df['image_url'] = None # Add dummy image col

    # Trier par date (plus récent en premier) - Gérer les NaT
    if "date" in df.columns:
        df = df.sort_values("date", ascending=False, na_position='last').reset_index(drop=True)

    # Assurer que seule la catégorie 'alimentation' est présente si c'est la demande explicite
    # Ce filtre est redondant si le filtre API fonctionne, mais ajoute une sécurité.
    # Vérifier si la colonne 'categorie' existe avant de filtrer
    if 'categorie' in df.columns:
        df = df[df['categorie'] == 'alimentation'].reset_index(drop=True)
    else:
        st.warning("La colonne 'categorie' est absente des données. Le filtrage 'alimentation' ne peut pas être appliqué après chargement.")


    # Débogage final
    debug_log("DataFrame traité et prêt", df.head())
    debug_log("Colonnes après traitement", list(df.columns))
    debug_log("Types de données", df.dtypes)
    if 'categorie' in df.columns:
        debug_log("Value counts 'categorie' (après filtre)", df['categorie'].value_counts())
    if 'sous_categorie' in df.columns:
        debug_log("Value counts 'sous_categorie'", df['sous_categorie'].value_counts().head())
    
    return df

# Fonction pour déterminer la classe de risque
def get_risk_class(risques):
    """Détermine la classe CSS en fonction des risques mentionnés."""
    if isinstance(risques, str):
        risques_lower = risques.lower()
        if any(kw in risques_lower for kw in ["listeria", "salmonelle", "e. coli", "toxique", "dangereux", "mortel", "blessures graves", "contamination bactérienne"]):
            return "risk-high"
        elif any(kw in risques_lower for kw in ["allergène", "microbiologique", "corps étranger", "chimique", "blessures", "intoxication", "dépassement de seuils", "non conforme", "additifs"]):
            return "risk-medium"
    return "risk-low" # Default or for less severe risks like odor/taste or minor defects

# Fonction pour afficher une carte de rappel (utilisée DANS une colonne)
def display_recall_card(row):
    """Affiche une carte de rappel dans le contexte d'une colonne."""
    
    # Extraire les données
    nom = row.get("nom", row.get("modele", "Produit non spécifié"))
    marque = row.get("marque", "Marque non spécifiée")
    date_str = row.get("date_str", "Date non spécifiée")
    categorie = row.get("categorie", "Catégorie non spécifiée")
    sous_categorie = row.get("sous_categorie", "Sous-catégorie non spécifiée")
    motif = row.get("motif", "Non spécifié")
    risques = row.get("risques", "Non spécifié")
    conduites = row.get("conduites_a_tenir", "Non spécifié")
    distributeurs = row.get("distributeurs", "Non spécifié")
    zone_vente = row.get("zone_vente", "Non spécifiée")
    image_url = row.get("image_url")
    fiche_url = row.get("fiche_url", "#")
    
    # Déterminer la classe de risque
    risk_class = get_risk_class(risques)
    
    # Afficher la carte
    # Utiliser st.container() à l'intérieur de la colonne pour regrouper les éléments de la carte
    with st.container():
        st.markdown(f"""<div class="card recall-card">""", unsafe_allow_html=True)
        
        st.markdown(f"<h4>{nom}</h4>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Marque:</strong> {marque}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Date de publication:</strong> {date_str}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Catégorie:</strong> {categorie} > {sous_categorie}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Risques:</strong> <span class='{risk_class}'>{risques}</span></p>", unsafe_allow_html=True)
        
        # Afficher l'image si disponible, centrée dans la colonne
        if image_url:
            try:
                # Utiliser st.image directement, les ajustements CSS devraient aider au centrage/taille
                 st.image(image_url, width=100)
            except Exception as e:
                debug_log(f"Error loading image {image_url}", e)
                st.info("Image non disponible")
        
        # Expander pour plus de détails
        with st.expander("Voir plus de détails"):
            st.markdown(f"<p><strong>Motif du rappel:</strong> {motif}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Distributeurs:</strong> {distributeurs}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Zone de vente:</strong> {zone_vente}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Conduites à tenir par le consommateur:</strong> {conduites}</p>", unsafe_allow_html=True)

            if fiche_url and fiche_url != "#":
                st.link_button("Voir la fiche complète sur RappelConso", fiche_url, type="secondary")

        st.markdown(f"</div>", unsafe_allow_html=True)


# Helper function for pagination controls
def display_pagination_controls(current_page_state_key, total_items, items_per_page):
    """Affiche les boutons de pagination et le numéro de page."""
    # items_per_page est le nombre de cartes par page (donc 2*items_per_page_per_col si affiché en 2 colonnes)
    # Dans ce cas, items_per_page est déjà le total par page.
    total_pages = math.ceil(total_items / items_per_page) if total_items > 0 else 1

    # S'assurer que la page actuelle ne dépasse pas le total de pages (peut arriver après filtrage ou recherche)
    if st.session_state[current_page_state_key] > total_pages:
         st.session_state[current_page_state_key] = total_pages
    if st.session_state[current_page_state_key] < 1:
         st.session_state[current_page_state_key] = 1


    col_prev, col_info, col_next = st.columns([1, 3, 1])

    with col_prev:
        if st.session_state[current_page_state_key] > 1:
            # Ajout d'une clé dynamique pour éviter les conflits si plusieurs paginations sur la même page (non le cas ici, mais bonne pratique)
            if st.button("← Précédent", key=f"btn_prev_{current_page_state_key}"):
                st.session_state[current_page_state_key] -= 1
                st.rerun()

    with col_info:
        st.markdown(f"<div style='text-align:center;'>Page {st.session_state[current_page_state_key]} sur {total_pages}</div>", unsafe_allow_html=True)

    with col_next:
        if st.session_state[current_page_state_key] < total_pages:
             # Ajout d'une clé dynamique
            if st.button("Suivant →", key=f"btn_next_{current_page_state_key}"):
                st.session_state[current_page_state_key] += 1
                st.rerun()

# --- Fonction principale ---
def main():
    # Initialiser les états de session nécessaires avant toute utilisation
    if "rappel_data" not in st.session_state:
        st.session_state.rappel_data = None
    if "load_params" not in st.session_state:
        st.session_state.load_params = None
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    if "search_current_page" not in st.session_state:
        st.session_state.search_current_page = 1
    if "selected_subcategories" not in st.session_state:
        st.session_state.selected_subcategories = ["Toutes"]
    if "selected_brands" not in st.session_state:
         st.session_state.selected_brands = ["Toutes"]
    if "selected_risks" not in st.session_state:
         st.session_state.selected_risks = ["Tous"]
    if "quick_search_input" not in st.session_state:
         st.session_state.quick_search_input = ""
    if "items_per_page_main" not in st.session_state:
        st.session_state.items_per_page_main = 10 # Default 10 items total per page (5 pairs)
    if "items_per_page_search" not in st.session_state:
        st.session_state.items_per_page_search = 10 # Default 10 items total per page (5 pairs)


    # En-tête
    st.markdown("""
    <div class="header">
        <h1>RappelConso Insight</h1>
        <p>Analyse des rappels de produits en France</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Sidebar pour les filtres de chargement ---
    st.sidebar.title("Paramètres de chargement")
    
    # Sélection de la catégorie (fixe à alimentation selon la demande)
    st.sidebar.write("Catégorie chargée: **alimentation**")
    selected_category_loading = "alimentation" # <--- Fixe la catégorie pour le chargement

    # Date de début
    start_date = st.sidebar.date_input(
        "Rappels depuis la date:",
        value=date.today() - timedelta(days=180), # Par défaut, 6 mois avant aujourd'hui
        max_value=date.today()
    )
    
    # Nombre max de rappels
    max_records = st.sidebar.slider(
        "Nombre max d'enregistrements à charger:", 
        min_value=100,
        max_value=10000, # Augmenté la limite max
        value=2000,
        step=100
    )
    
    # Bouton pour charger les données
    load_button = st.sidebar.button("Charger/Actualiser les données", type="primary")

    # Déterminer si les données doivent être chargées (au premier lancement ou si les paramètres de chargement ont changé)
    current_load_params = (selected_category_loading, start_date, max_records)
    
    # Charger les données si le bouton est cliqué OU si c'est la première exécution ET qu'aucun paramètre n'est enregistré
    if load_button or (st.session_state.rappel_data is None and st.session_state.load_params is None):
        # Afficher un message pendant le chargement
        status_message = st.sidebar.info("Chargement des données en cours...")
        st.session_state.rappel_data = load_rappel_data(
            start_date=start_date,
            category=selected_category_loading, # Utilise la catégorie sélectionnée/fixée
            max_records=max_records
        )
        st.session_state.load_params = current_load_params # Sauvegarder les paramètres utilisés

        # Réinitialiser les filtres d'analyse post-chargement et pagination après un nouveau chargement
        st.session_state.selected_subcategories = ["Toutes"]
        st.session_state.selected_brands = ["Toutes"]
        st.session_state.selected_risks = ["Tous"]
        st.session_state.current_page = 1 # Reset main list pagination
        st.session_state.search_current_page = 1 # Reset search results pagination
        # Ne pas réinitialiser l'input de recherche, l'utilisateur pourrait vouloir relancer la même recherche
        # st.session_state.quick_search_input = "" # Clear search input

        status_message.empty() # Clear the loading message (if it was there)
        # Use st.toast or st.success for feedback after loading
        if st.session_state.rappel_data is not None and not st.session_state.rappel_data.empty:
            st.toast("Données chargées avec succès !", icon="✅")
        else:
             st.toast("Aucune donnée chargée avec les paramètres spécifiés.", icon="⚠️")

        st.rerun() # Rerun pour appliquer le chargement et afficher les données

    # Vérifier si les données sont chargées
    if st.session_state.rappel_data is None or st.session_state.rappel_data.empty:
        if st.session_state.load_params is not None: # Only show message if a load was attempted
             st.info("Aucun rappel ne correspond aux filtres de chargement sélectionnés, ou veuillez charger les données en cliquant sur le bouton dans la barre latérale.")
        return
    
    df = st.session_state.rappel_data.copy() # Utiliser une copie pour les filtrages post-chargement

    # --- Filtres post-chargement (dans la barre latérale) ---
    st.sidebar.markdown("---")
    st.sidebar.title("Filtres d'analyse")

    # Filtre Sous-catégorie
    # Utiliser un set pour les options pour éviter les doublons même si dropna est appelé
    all_subcategories = ["Toutes"] + sorted(set(df["sous_categorie"].dropna().tolist()))
    selected_subcategories = st.sidebar.multiselect(
        "Filtrer par sous-catégorie:",
        options=all_subcategories,
        default=st.session_state.selected_subcategories,
        key="sidebar_subcat_filter" # Clé unique
    )
    st.session_state.selected_subcategories = selected_subcategories # Sauvegarder la sélection


    # Filtre Marque
    all_brands = ["Toutes"] + sorted(set(df["marque"].dropna().tolist()))
    selected_brands = st.sidebar.multiselect(
        "Filtrer par marque:",
        options=all_brands,
        default=st.session_state.selected_brands,
        key="sidebar_brand_filter" # Clé unique
    )
    st.session_state.selected_brands = selected_brands # Sauvegarder la sélection


    # Filtre Risque
    all_risks = ["Tous"] + sorted(set(df["risques"].dropna().tolist()))
    selected_risks = st.sidebar.multiselect(
        "Filtrer par risques encourus:",
        options=all_risks,
        default=st.session_state.selected_risks,
        key="sidebar_risk_filter" # Clé unique
    )
    st.session_state.selected_risks = selected_risks # Sauvegarder la sélection

    
    # Appliquer les filtres post-chargement pour obtenir le DataFrame affiché
    df_filtered = df.copy()

    if "Toutes" not in selected_subcategories:
        df_filtered = df_filtered[df_filtered["sous_categorie"].isin(selected_subcategories)]

    if "Toutes" not in selected_brands:
        df_filtered = df_filtered[df_filtered["marque"].isin(selected_brands)]

    if "Tous" not in selected_risks:
         df_filtered = df_filtered[df_filtered["risques"].isin(selected_risks)]

    # Vérifier si le DataFrame filtré est vide (important pour la suite)
    if df_filtered.empty:
        st.warning("Aucun rappel ne correspond aux filtres d'analyse sélectionnés.")
        # Afficher les métriques basées sur le df *chargé* quand même pour information
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.markdown(f"""<div class="metric"><div class="metric-value">{len(df)}</div><div>Rappels chargés</div></div>""", unsafe_allow_html=True)
        with col2: st.markdown(f"""<div class="metric"><div class="metric-value">0</div><div>Rappels affichés</div></div>""", unsafe_allow_html=True)
        with col3: st.markdown(f"""<div class="metric"><div class="metric-value">0</div><div>Rappels (30 derniers jours)</div></div>""", unsafe_allow_html=True)
        with col4: st.markdown(f"""<div class="metric"><div class="metric-value">0</div><div>Marques uniques</div></div>""", unsafe_allow_html=True)
        
        # Afficher les onglets mais avec des messages d'absence de données à l'intérieur
        tab1, tab2, tab3 = st.tabs(["📋 Liste des rappels", "📊 Visualisations", "🔍 Recherche rapide"])
        with tab1: st.info("Aucun rappel à afficher avec les filtres sélectionnés.")
        with tab2: st.info("Aucune donnée à visualiser avec les filtres sélectionnés.")
        with tab3: st.info("Aucune donnée à rechercher avec les filtres sélectionnés.")
        
        return # Arrêter l'exécution de main() ici si df_filtered est vide


    # --- Afficher quelques métriques ---
    st.subheader("Vue d'ensemble")
    col1, col2, col3, col4 = st.columns(4)
    
    # Nombre total de rappels chargés (basé sur le chargement API initial)
    with col1:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-value">{len(df)}</div>
            <div>Rappels chargés</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Nombre de rappels filtrés actuellement affichés (basé sur df_filtered)
    with col2:
         st.markdown(f"""
         <div class="metric">
             <div class="metric-value">{len(df_filtered)}</div>
             <div>Rappels affichés</div>
         </div>
         """, unsafe_allow_html=True)

    # Rappels récents (calculés sur les données filtrées)
    # Utiliser la colonne 'date' qui est datetime
    today = datetime.now(tz=timezone.utc) # Comparer avec un datetime UTC
    # Filtrer les dates non valides (NaT) avant la comparaison
    recent_recalls = df_filtered[df_filtered['date'].notna() & (df_filtered['date'] >= (today - timedelta(days=30)))]
    recent_count = len(recent_recalls)
    
    with col3:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-value">{recent_count}</div>
            <div>Rappels (30 derniers jours)</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Nombre de marques uniques dans les données filtrées
    if "marque" in df_filtered.columns:
        # Ajouter dropna() pour compter uniquement les marques non nulles
        brand_count_filtered = df_filtered["marque"].dropna().nunique()
        with col4:
            st.markdown(f"""
            <div class="metric">
                <div class="metric-value">{brand_count_filtered}</div>
                <div>Marques uniques (filtrées)</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        with col4:
             st.markdown(f"""
             <div class="metric">
                 <div class="metric-value">N/A</div>
                 <div>Marques uniques</div>
             </div>
             """, unsafe_allow_html=True)

    # --- Bouton de téléchargement pour les données filtrées ---
    if not df_filtered.empty:
         st.markdown("---") # Séparateur visuel
         st.download_button(
             label="Télécharger les données filtrées (CSV)",
             data=df_filtered.to_csv(index=False).encode('utf-8'),
             file_name=f'rappelconso_alimentation_depuis_{start_date.strftime("%Y-%m-%d")}_filtered.csv',
             mime='text/csv',
             key="download_filtered_csv" # Clé unique
         )
         st.markdown("---") # Séparateur visuel


    # --- Afficher les onglets ---
    tab1, tab2, tab3 = st.tabs(["📋 Liste des rappels", "📊 Visualisations", "🔍 Recherche rapide"])
    
    with tab1:
        # Pagination des rappels filtrés
        st.subheader("Liste des rappels filtrés")
        
        # Slider pour le nombre d'items par page (affecte la pagination principale)
        items_per_page_main = st.select_slider(
            "Rappels par page:",
            options=[5, 10, 20, 50],
            value=st.session_state.items_per_page_main, # Utiliser l'état de session
            key="slider_items_per_page_main"
        )
        st.session_state.items_per_page_main = items_per_page_main

        # Réinitialiser la page si les filtres changent ou si le nombre d'items par page change
        current_pagination_state_main = hash((tuple(selected_subcategories), tuple(selected_brands), tuple(selected_risks), items_per_page_main))
        if "pagination_state_main" not in st.session_state or st.session_state.pagination_state_main != current_pagination_state_main:
             st.session_state.current_page = 1
             st.session_state.pagination_state_main = current_pagination_state_main

        total_items_main = len(df_filtered)
        
        # Afficher les rappels de la page actuelle en 2 colonnes
        start_idx_main = (st.session_state.current_page - 1) * items_per_page_main
        end_idx_main = min(start_idx_main + items_per_page_main, total_items_main)

        if total_items_main > 0:
            # Afficher en 2 colonnes
            # On itère sur les indices de start_idx à end_idx
            indices_to_display = range(start_idx_main, end_idx_main)
            for i in range(0, len(indices_to_display), 2):
                col1, col2 = st.columns(2)
                
                # Afficher le premier produit de la paire
                item_index_1 = indices_to_display[i]
                with col1:
                    display_recall_card(df_filtered.iloc[item_index_1])
                
                # Afficher le deuxième produit si il existe
                if i + 1 < len(indices_to_display): # Vérifier si le deuxième produit est dans la tranche actuelle d'indices
                    item_index_2 = indices_to_display[i+1]
                    with col2:
                         display_recall_card(df_filtered.iloc[item_index_2])
                # else:
                #    with col2:
                #         st.empty() # Streamlit gère les colonnes vides automatiquement

            # Afficher les contrôles de pagination pour la liste principale
            display_pagination_controls("current_page", total_items_main, items_per_page_main)

        else:
            st.info("Aucun rappel à afficher avec les filtres actuels.")

    with tab2:
        # Visualisations
        st.subheader("Visualisations des données filtrées")
        
        # Évolution temporelle des rappels
        st.write("### Évolution des rappels par mois")
        
        # Filtrer les lignes avec date NaT avant de grouper
        df_filtered_valid_dates = df_filtered.dropna(subset=['date'])

        if "date" in df_filtered_valid_dates.columns and not df_filtered_valid_dates.empty:
            try:
                # Grouper par mois en utilisant la colonne datetime
                monthly_counts = df_filtered_valid_dates.groupby(df_filtered_valid_dates['date'].dt.to_period('M')).size().reset_index(name="count")
                monthly_counts["month_str"] = monthly_counts["date"].astype(str)
                
                # Trier chronologiquement les mois
                monthly_counts = monthly_counts.sort_values("month_str")

                if not monthly_counts.empty:
                    fig_time = px.line(
                        monthly_counts,
                        x="month_str",
                        y="count",
                        title="Nombre de rappels par mois",
                        labels={"month_str": "Mois", "count": "Nombre de rappels"},
                        markers=True
                    )
                    fig_time.update_layout(xaxis_title="Mois", yaxis_title="Nombre de rappels", hovermode="x unified") # Améliorer le survol
                    st.plotly_chart(fig_time, use_container_width=True)
                else:
                    st.warning("Pas assez de données temporelles valides pour créer un graphique.")
            except Exception as e:
                st.error(f"Erreur lors de la création du graphique temporel: {str(e)}")
                debug_log("Time series chart error", e)
        else:
            st.info("La colonne de date n'est pas disponible ou contient des valeurs manquantes dans les données filtrées.")


        # Distribution par sous-catégorie
        if "sous_categorie" in df_filtered.columns:
            st.write("### Répartition par sous-catégorie")
            
            valid_subcats = df_filtered["sous_categorie"].dropna()
            
            if not valid_subcats.empty:
                num_top_subcats = st.slider("Nombre de sous-catégories à afficher:", 5, 30, 10, key="subcat_slider")
                top_subcats = valid_subcats.value_counts().nlargest(num_top_subcats)
                
                fig_subcat = px.pie(
                    values=top_subcats.values,
                    names=top_subcats.index,
                    title=f"Top {num_top_subcats} des sous-catégories",
                    hole=0.3 # Add donut chart style
                )
                fig_subcat.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_subcat, use_container_width=True)
            else:
                st.info("Aucune donnée de sous-catégorie disponible dans les données filtrées.")
        
        # Distribution par type de risque
        if "risques" in df_filtered.columns:
            st.write("### Répartition par type de risque")
            
            valid_risks = df_filtered["risques"].dropna()
            
            if not valid_risks.empty:
                risk_counts = valid_risks.value_counts().reset_index()
                risk_counts.columns = ["risk", "count"]

                risk_counts['risk_level'] = risk_counts['risk'].apply(get_risk_class)

                risk_counts['risk_level_label'] = risk_counts['risk_level'].map({
                     'risk-high': 'Risque Élevé',
                     'risk-medium': 'Risque Moyen',
                     'risk-low': 'Risque Faible'
                })
                
                risk_level_order = ['Risque Élevé', 'Risque Moyen', 'Risque Faible']
                risk_counts['risk_level_label'] = pd.Categorical(risk_counts['risk_level_label'], categories=risk_level_order, ordered=True)
                risk_counts = risk_counts.sort_values(['risk_level_label', 'count'], ascending=[True, False])

                fig_risks = px.bar(
                    risk_counts,
                    x="risk",
                    y="count",
                    title="Répartition par type de risque",
                    labels={"risk": "Type de risque", "count": "Nombre de rappels", "risk_level_label": "Niveau de risque"},
                    color="risk_level_label", # Colorer par niveau de risque
                    color_discrete_map={
                         'Risque Élevé': '#e74c3c',
                         'Risque Moyen': '#f39c12',
                         'Risque Faible': '#27ae60'
                    }
                )
                
                fig_risks.update_layout(xaxis_tickangle=-45)
                
                st.plotly_chart(fig_risks, use_container_width=True)
            else:
                st.info("Aucune donnée de risque disponible dans les données filtrées.")

        # Distribution par marque (Top N)
        if "marque" in df_filtered.columns:
             st.write("### Marques les plus fréquentes")
             valid_brands = df_filtered["marque"].dropna()
             if not valid_brands.empty:
                 num_top_brands = st.slider("Nombre de marques à afficher:", 5, 30, 10, key="top_brands_slider")
                 top_brands = valid_brands.value_counts().nlargest(num_top_brands)
                 fig_brands = px.bar(
                     x=top_brands.index,
                     y=top_brands.values,
                     title=f"Top {num_top_brands} des marques",
                     labels={"x": "Marque", "y": "Nombre de rappels"}
                 )
                 fig_brands.update_layout(xaxis_tickangle=-45)
                 st.plotly_chart(fig_brands, use_container_width=True)
             else:
                 st.info("Aucune donnée de marque disponible dans les données filtrées.")


    with tab3:
        # Recherche rapide dans les données filtrées
        st.subheader("Recherche rapide dans les rappels affichés (champ: Motif)")
        
        # Utiliser session state pour maintenir la valeur de l'input de recherche
        search_term = st.text_input(
            "Entrez un terme pour rechercher dans le Motif du rappel:",
            placeholder="Ex: listéria, salmonelle, corps étranger",
            key="quick_search_input" # Clé unique
        )

        # Appliquer la recherche si un terme est saisi
        search_results_df = pd.DataFrame() # Initialiser un dataframe vide
        if search_term:
            search_term_lower = search_term.lower()
            
            # Colonnes sur lesquelles effectuer la recherche rapide : UNIQUEMENT LE MOTIF
            search_cols = ["motif"]
            search_cols_existing = [col for col in search_cols if col in df_filtered.columns]
            
            if search_cols_existing:
                 # Créer une colonne texte combinée pour la recherche (ici, juste le motif)
                 df_filtered_with_search = df_filtered.copy()
                 # Concaténer seulement les colonnes existantes et gérées (ici, juste 'motif')
                 df_filtered_with_search['search_text'] = df_filtered_with_search[search_cols_existing].astype(str).fillna('').agg(' '.join, axis=1).str.lower()

                 # Appliquer le filtre de recherche
                 search_results_df = df_filtered_with_search[df_filtered_with_search['search_text'].str.contains(search_term_lower, na=False)].copy() # Utiliser .copy() ici aussi

                 # La colonne temporaire 'search_text' n'est pas nécessaire dans le df_filtered original

            st.markdown(f"**{len(search_results_df)}** résultats trouvés pour '{search_term}' dans le Motif.")

        if search_term and not search_results_df.empty:
            # --- Pagination pour les résultats de recherche ---
            st.markdown("---") # Séparateur visuel
            st.write("Navigation dans les résultats de recherche:")

            # Slider pour le nombre d'items par page (spécifique à la recherche)
            items_per_page_search = st.select_slider(
                "Résultats par page:",
                options=[5, 10, 20, 50],
                value=st.session_state.items_per_page_search, # Utiliser l'état de session
                key="slider_items_per_page_search"
            )
            st.session_state.items_per_page_search = items_per_page_search

            # Réinitialiser la page de recherche si le terme de recherche ou items_per_page_search change
            current_pagination_state_search = hash((search_term_lower, items_per_page_search))
            # Ne réinitialiser que si le terme de recherche change *et* qu'il n'est pas vide
            if (st.session_state.get("last_search_term", "") != search_term_lower and search_term_lower != "") or \
               ("pagination_state_search" not in st.session_state or st.session_state.pagination_state_search != current_pagination_state_search):
                 st.session_state.search_current_page = 1
                 st.session_state.pagination_state_search = current_pagination_state_search
            st.session_state.last_search_term = search_term_lower # Mettre à jour le dernier terme recherché


            total_items_search = len(search_results_df)

            # Afficher les rappels de la page actuelle en 2 colonnes
            start_idx_search = (st.session_state.search_current_page - 1) * items_per_page_search
            end_idx_search = min(start_idx_search + items_per_page_search, total_items_search)

            # Afficher en 2 colonnes
            indices_to_display_search = range(start_idx_search, end_idx_search)
            for i in range(0, len(indices_to_display_search), 2):
                col1, col2 = st.columns(2)
                
                # Afficher le premier produit de la paire
                item_index_1 = indices_to_display_search[i]
                with col1:
                    display_recall_card(search_results_df.iloc[item_index_1])
                
                # Afficher le deuxième produit si il existe
                if i + 1 < len(indices_to_display_search): # Vérifier si le deuxième produit est dans la tranche actuelle d'indices
                    item_index_2 = indices_to_display_search[i+1]
                    with col2:
                         display_recall_card(search_results_df.iloc[item_index_2])
                # else:
                #     with col2:
                #          st.empty() # Streamlit gère les colonnes vides automatiquement

            # Afficher les contrôles de pagination pour la recherche
            display_pagination_controls("search_current_page", total_items_search, items_per_page_search)

        elif search_term:
             st.warning(f"Aucun résultat trouvé pour '{search_term}' dans le Motif.")
        else:
            st.info("Entrez un terme dans la barre de recherche pour afficher les résultats (recherche limitée au Motif).")


    # Footer
    st.markdown("""
    <div class="footer">
        RappelConso Insight - Application basée sur les données de <a href="https://www.rappelconso.gouv.fr/" target="_blank">RappelConso.gouv.fr</a>. Données fournies par data.economie.gouv.fr
    </div>
    """, unsafe_allow_html=True)


# Exécuter l'application
if __name__ == "__main__":
    main()
