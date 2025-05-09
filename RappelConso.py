import streamlit as st
import pandas as pd
import requests
from datetime import datetime, date, timedelta, timezone # Ajout de timezone ici
import time
import plotly.express as px
import numpy as np

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
</style>
""", unsafe_allow_html=True)

# --- Mode debug ---
DEBUG = False # Set to True to enable debug logs

def debug_log(message, data=None):
    """Affiche des informations de débogage"""
    if DEBUG:
        st.sidebar.markdown(f"**DEBUG:** {message}")
        if data is not None:
             st.sidebar.json(data) # Simpler output in sidebar for debug

# --- Liste prédéfinie de catégories (peut être étendue) ---
# La catégorie 'alimentation' est généralement la plus pertinente pour les rappels de sécurité
# Charger toutes les catégories via l'API est complexe et peut être lent/coûteux.
# On se limite donc à une sélection ou on garde 'alimentation' comme filtre API principal.
# Pour permettre le filtre API, on utilise une liste fixe.
CATEGORIES = ["", "alimentation", "vêtements et accessoires", "maison-habitat", "jouets", "hygiène-beauté", "véhicules", "autres"]
# L'option vide "" permettra de charger toutes les catégories (si l'API le supporte en omettant le paramètre, sinon il faut l'adapter)
# Le code actuel utilise refine.categorie_produit. Si "" est sélectionné, on n'envoie pas ce paramètre.

# --- Fonction de chargement des données avec cache ---
@st.cache_data(ttl=3600, show_spinner=False) # Cache pendant 1 heure, cache key dépend des paramètres
def load_rappel_data(start_date: date = None, category: str = "", max_records: int = 1000):
    """
    Charge les données de l'API RappelConso.
    Applique le filtre de date et de catégorie à l'API si possible.
    """
    api_url = "https://data.economie.gouv.fr/api/v2/catalog/datasets/rappelconso-v2-gtin-espaces/records"
    
    params = {
        "limit": 100, # Limite par page, max 100 dans l'API
        "offset": 0
    }
    
    # Ajouter un filtre de catégorie si spécifié
    if category and category != "":
        params["refine.categorie_produit"] = category
    
    # Ajouter un filtre de date de publication si spécifié
    # L'API V2 supporte refine.date_publication. On utilise le format '>=' pour une date de début.
    # Le format datetime de l'API est ISO 8601, YYYY-MM-DDTHH:MM:SS+00:00.
    # Filtrer juste par date 'YYYY-MM-DD' semble fonctionner en pratique pour le début de la journée.
    if start_date:
         start_date_str = start_date.strftime("%Y-%m-%d")
         params["refine.date_publication"] = f">='{start_date_str}'"
    
    all_records = []
    total_count = 0
    
    # Indicateur de progression
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Première requête pour estimer le nombre total (peut être inexact avec des filtres complexes)
        # On ne demande que 1 record pour être rapide et obtenir total_count
        initial_params = params.copy()
        initial_params["limit"] = 1
        
        status_text.text("Connexion à l'API et estimation du total...")
        response = requests.get(api_url, params=initial_params, timeout=30)
        response.raise_for_status() # Lève une exception pour les erreurs HTTP
        
        data = response.json()
        total_count = data.get("total_count", 0)
        debug_log(f"Estimation Total Count API: {total_count}", data)
        
        if total_count == 0:
            status_text.warning(f"Aucun rappel trouvé avec les filtres initiaux.")
            progress_bar.progress(1.0)
            return pd.DataFrame()

        estimated_fetch_count = min(total_count, max_records)
        status_text.text(f"Chargement des données RappelConso ({estimated_fetch_count} maximum)...")
        
        offset = 0
        # Ajuster la condition de boucle pour s'assurer de ne pas dépasser max_records
        while offset < total_count and len(all_records) < max_records:
            params["offset"] = offset
            
            # Assurez-vous de ne pas demander plus que max_records dans la dernière page
            params["limit"] = min(100, max_records - len(all_records))

            # Si la limite devient 0 ou moins, c'est qu'on a déjà atteint max_records
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
            
            # Petite pause pour éviter de surcharger l'API
            time.sleep(0.05)
            
            # Vérifier si la limite max_records est atteinte après avoir ajouté les records
            if len(all_records) >= max_records:
                 status_text.info(f"Limite de {max_records} enregistrements atteinte.")
                 break

    except requests.exceptions.Timeout:
        status_text.error("Erreur: Délai d'attente dépassé lors de la connexion à l'API.")
        progress_bar.empty()
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        status_text.error(f"Erreur API: Impossible de charger les données. {e}")
        progress_bar.empty()
        return pd.DataFrame()
    except Exception as e:
        status_text.error(f"Une erreur inattendue est survenue lors du chargement: {e}")
        progress_bar.empty()
        return pd.DataFrame()

    finally:
        # Assurez-vous que la progression atteint 100% même si on s'arrête avant total_count à cause de max_records
        if len(all_records) > 0:
             progress_bar.progress(1.0)
        status_text.empty() # Nettoyer le texte de statut
        progress_bar.empty() # Nettoyer la barre de progression
    
    if not all_records:
        st.warning("Aucun rappel trouvé avec les filtres spécifiés.")
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
    
    # Débogage final
    debug_log("DataFrame traité et prêt", df.head())
    debug_log("Colonnes après traitement", list(df.columns))
    debug_log("Types de données", df.dtypes)
    debug_log("Value counts 'categorie'", df['categorie'].value_counts().head())
    debug_log("Value counts 'sous_categorie'", df['sous_categorie'].value_counts().head())
    
    return df

# Fonction pour déterminer la classe de risque
def get_risk_class(risques):
    """Détermine la classe CSS en fonction des risques mentionnés."""
    if isinstance(risques, str):
        risques_lower = risques.lower()
        if any(kw in risques_lower for kw in ["listeria", "salmonelle", "e. coli", "toxique", "dangereux", "mortel", "blessures graves"]):
            return "risk-high"
        elif any(kw in risques_lower for kw in ["allergène", "microbiologique", "corps étranger", "chimique", "blessures", "intoxication", "dépassement de seuils"]):
            return "risk-medium"
    return "risk-low" # Default or for less severe risks like odor/taste or minor defects

# Fonction pour afficher une carte de rappel
def display_recall_card(row):
    """Affiche une carte de rappel avec plus de détails dans un expander."""
    
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
    st.markdown(f"""<div class="card recall-card">""", unsafe_allow_html=True)
    
    # Colonnes pour titre/date/risque et image
    col_text, col_img = st.columns([3, 1])

    with col_text:
        st.markdown(f"<h4>{nom}</h4>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Marque:</strong> {marque}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Date de publication:</strong> {date_str}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Catégorie:</strong> {categorie} > {sous_categorie}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Risques:</strong> <span class='{risk_class}'>{risques}</span></p>", unsafe_allow_html=True)
    
    with col_img:
        if image_url:
            try:
                # Ajouter une petite marge en haut de l'image si elle est présente
                st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
                st.image(image_url, width=100)
                st.markdown("</div>", unsafe_allow_html=True)
            except:
                st.info("Image non disponible")
        else:
            st.empty() # Keep space if no image

    # Expander pour plus de détails
    with st.expander("Voir plus de détails"):
        st.markdown(f"<p><strong>Motif du rappel:</strong> {motif}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Distributeurs:</strong> {distributeurs}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Zone de vente:</strong> {zone_vente}</p>", unsafe_allow_html=True)
        st.markdown(f"<p><strong>Conduites à tenir par le consommateur:</strong> {conduites}</p>", unsafe_allow_html=True)

        if fiche_url and fiche_url != "#":
            st.link_button("Voir la fiche complète sur RappelConso", fiche_url, type="secondary")

    st.markdown(f"</div>", unsafe_allow_html=True)

# --- Fonction principale ---
def main():
    # En-tête
    st.markdown("""
    <div class="header">
        <h1>RappelConso Insight</h1>
        <p>Analyse des rappels de produits en France</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Sidebar pour les filtres de chargement ---
    st.sidebar.title("Paramètres de chargement")
    
    # Sélection de la catégorie
    selected_category = st.sidebar.selectbox(
        "Catégorie:",
        options=CATEGORIES,
        index=CATEGORIES.index("alimentation") # Par défaut sur alimentation
    )

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
    # Utiliser un état de session pour gérer le chargement
    if "rappel_data" not in st.session_state:
        st.session_state.rappel_data = None
        st.session_state.load_params = None

    load_button = st.sidebar.button("Charger/Actualiser les données", type="primary")

    # Déterminer si les données doivent être chargées (au premier lancement ou si les paramètres de chargement ont changé)
    current_load_params = (selected_category, start_date, max_records)
    
    # Charger les données si le bouton est cliqué OU si c'est la première exécution ET qu'aucun paramètre n'est enregistré
    if load_button or (st.session_state.rappel_data is None and st.session_state.load_params is None):
        st.session_state.rappel_data = load_rappel_data(
            start_date=start_date,
            category=selected_category,
            max_records=max_records
        )
        st.session_state.load_params = current_load_params # Sauvegarder les paramètres utilisés
        # Réinitialiser les filtres d'analyse post-chargement à "Toutes" après un nouveau chargement
        st.session_state.selected_subcategories = ["Toutes"]
        st.session_state.selected_brands = ["Toutes"]
        st.session_state.selected_risks = ["Tous"]
        st.session_state.current_page = 1 # Reset pagination on new load
        # Rerun pour appliquer le chargement et afficher les données
        st.rerun()


    # Vérifier si les données sont chargées
    if st.session_state.rappel_data is None or st.session_state.rappel_data.empty:
        if st.session_state.load_params is not None: # Only show message if a load was attempted
             st.info("Veuillez charger les données en cliquant sur le bouton dans la barre latérale, ou aucun rappel ne correspond aux filtres sélectionnés.")
        return
    
    df = st.session_state.rappel_data.copy() # Utiliser une copie pour les filtrages post-chargement

    # --- Filtres post-chargement (dans la barre latérale) ---
    st.sidebar.markdown("---")
    st.sidebar.title("Filtres d'analyse")

    # Filtre Sous-catégorie
    all_subcategories = ["Toutes"] + sorted(df["sous_categorie"].dropna().unique().tolist())
    # Utiliser st.session_state pour maintenir la sélection entre les runs
    if "selected_subcategories" not in st.session_state:
        st.session_state.selected_subcategories = ["Toutes"]

    selected_subcategories = st.sidebar.multiselect(
        "Filtrer par sous-catégorie:",
        options=all_subcategories,
        default=st.session_state.selected_subcategories # Utiliser la valeur par défaut de la session
    )
    st.session_state.selected_subcategories = selected_subcategories # Sauvegarder la sélection


    # Filtre Marque
    all_brands = ["Toutes"] + sorted(df["marque"].dropna().unique().tolist())
    if "selected_brands" not in st.session_state:
         st.session_state.selected_brands = ["Toutes"]

    selected_brands = st.sidebar.multiselect(
        "Filtrer par marque:",
        options=all_brands,
        default=st.session_state.selected_brands # Utiliser la valeur par défaut de la session
    )
    st.session_state.selected_brands = selected_brands # Sauvegarder la sélection


    # Filtre Risque
    all_risks = ["Tous"] + sorted(df["risques"].dropna().unique().tolist())
    if "selected_risks" not in st.session_state:
         st.session_state.selected_risks = ["Tous"]

    selected_risks = st.sidebar.multiselect(
        "Filtrer par risques encourus:",
        options=all_risks,
        default=st.session_state.selected_risks # Utiliser la valeur par défaut de la session
    )
    st.session_state.selected_risks = selected_risks # Sauvegarder la sélection

    
    # Appliquer les filtres post-chargement
    df_filtered = df.copy()

    if "Toutes" not in selected_subcategories:
        df_filtered = df_filtered[df_filtered["sous_categorie"].isin(selected_subcategories)]

    if "Toutes" not in selected_brands:
        df_filtered = df_filtered[df_filtered["marque"].isin(selected_brands)]

    if "Tous" not in selected_risks:
         df_filtered = df_filtered[df_filtered["risques"].isin(selected_risks)]

    # Vérifier si le DataFrame filtré est vide
    if df_filtered.empty:
        st.warning("Aucun rappel ne correspond aux filtres d'analyse sélectionnés.")
        # Afficher les métriques basées sur le df *chargé* quand même ? Ou les masquer ?
        # On peut afficher 0 pour les métriques basées sur df_filtered
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.markdown(f"""<div class="metric"><div class="metric-value">{len(df)}</div><div>Rappels chargés</div></div>""", unsafe_allow_html=True)
        with col2: st.markdown(f"""<div class="metric"><div class="metric-value">0</div><div>Rappels affichés</div></div>""", unsafe_allow_html=True)
        with col3: st.markdown(f"""<div class="metric"><div class="metric-value">0</div><div>Rappels (30 derniers jours)</div></div>""", unsafe_allow_html=True)
        with col4: st.markdown(f"""<div class="metric"><div class="metric-value">0</div><div>Marques uniques</div></div>""", unsafe_allow_html=True)
        return

    # --- Afficher quelques métriques ---
    st.subheader("Vue d'ensemble")
    col1, col2, col3, col4 = st.columns(4)
    
    # Nombre total de rappels chargés
    with col1:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-value">{len(df)}</div>
            <div>Rappels chargés</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Nombre de rappels filtrés actuellement affichés
    with col2:
         st.markdown(f"""
         <div class="metric">
             <div class="metric-value">{len(df_filtered)}</div>
             <div>Rappels affichés</div>
         </div>
         """, unsafe_allow_html=True)

    # Rappels récents (calculés sur les données filtrées)
    # Utiliser la colonne 'date' qui est datetime
    # CORRECTION ICI : Utiliser datetime.now(tz=timezone.utc)
    today = datetime.now(tz=timezone.utc) # Comparer avec un datetime UTC
    recent_recalls = df_filtered[df_filtered['date'].dropna() >= (today - timedelta(days=30))] # Ajouter dropna() pour éviter les erreurs avec NaT
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


    # --- Afficher les onglets ---
    tab1, tab2, tab3 = st.tabs(["📋 Liste des rappels", "📊 Visualisations", "🔍 Recherche rapide"])
    
    with tab1:
        # Pagination des rappels filtrés
        st.subheader("Liste des rappels filtrés")
        
        items_per_page = st.select_slider(
            "Rappels par page:",
            options=[5, 10, 20, 50],
            value=10
        )
        
        # Réinitialiser la page si les filtres changent ou si le nombre d'items par page change
        # On utilise un hash simple des filtres et items_per_page
        current_pagination_state = hash((tuple(selected_subcategories), tuple(selected_brands), tuple(selected_risks), items_per_page))
        if "pagination_state" not in st.session_state or st.session_state.pagination_state != current_pagination_state:
             st.session_state.current_page = 1
             st.session_state.pagination_state = current_pagination_state

        total_items = len(df_filtered)
        total_pages = (total_items - 1) // items_per_page + 1 if total_items > 0 else 1
        
        # S'assurer que la page actuelle est valide
        st.session_state.current_page = max(1, min(st.session_state.current_page, total_pages))
        
        # Afficher les rappels de la page actuelle
        start_idx = (st.session_state.current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)

        if total_items > 0:
            # Afficher les rappels de la page actuelle
            for i in range(start_idx, end_idx):
                display_recall_card(df_filtered.iloc[i])
            
            # Pagination controls
            col_prev, col_info, col_next = st.columns([1, 3, 1])
            
            with col_prev:
                # Utiliser un key unique pour le bouton si nécessaire dans certains scénarios
                if st.session_state.current_page > 1:
                    if st.button("← Précédent", key="btn_prev_page"):
                        st.session_state.current_page -= 1
                        st.rerun() # Rerun pour actualiser la page affichée
        
            with col_info:
                st.markdown(f"<div style='text-align:center;'>Page {st.session_state.current_page} sur {total_pages}</div>", unsafe_allow_html=True)
            
            with col_next:
                # Utiliser un key unique
                if st.session_state.current_page < total_pages:
                    if st.button("Suivant →", key="btn_next_page"):
                        st.session_state.current_page += 1
                        st.rerun() # Rerun pour actualiser la page affichée
        else:
            st.info("Aucun rappel à afficher avec la pagination et les filtres actuels.")

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
                # Utiliser .dt.to_period('M') pour grouper par mois calendaire
                monthly_counts = df_filtered_valid_dates.groupby(df_filtered_valid_dates['date'].dt.to_period('M')).size().reset_index(name="count")
                # Convertir la période en string pour l'affichage sur l'axe
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
        else:
            st.info("La colonne de date n'est pas disponible ou contient des valeurs manquantes dans les données filtrées.")


        # Distribution par sous-catégorie
        if "sous_categorie" in df_filtered.columns:
            st.write("### Répartition par sous-catégorie")
            
            # Filtrer les valeurs NA
            valid_subcats = df_filtered["sous_categorie"].dropna()
            
            if not valid_subcats.empty:
                # Afficher les top N sous-catégories (modifiable par l'utilisateur ?)
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
            
            # Filtrer les valeurs NA
            valid_risks = df_filtered["risques"].dropna()
            
            if not valid_risks.empty:
                # Compter les occurrences des risques
                risk_counts = valid_risks.value_counts().reset_index()
                risk_counts.columns = ["risk", "count"]

                # Tenter de classer les risques
                risk_counts['risk_level'] = risk_counts['risk'].apply(get_risk_class)

                # Mapper les classes CSS aux libellés
                risk_counts['risk_level_label'] = risk_counts['risk_level'].map({
                     'risk-high': 'Risque Élevé',
                     'risk-medium': 'Risque Moyen',
                     'risk-low': 'Risque Faible'
                })
                
                # Définir l'ordre des niveaux de risque pour le tri
                risk_level_order = ['Risque Élevé', 'Risque Moyen', 'Risque Faible']
                risk_counts['risk_level_label'] = pd.Categorical(risk_counts['risk_level_label'], categories=risk_level_order, ordered=True)
                # Trier d'abord par niveau de risque, puis par nombre de rappels
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
        st.subheader("Recherche rapide dans les rappels affichés")
        
        search_term = st.text_input("Entrez un terme pour rechercher:", placeholder="Ex: listeria, saumon, Leclerc", key="quick_search_input")
        
        if search_term:
            # Recherche insensible à la casse dans plusieurs colonnes pertinentes
            search_term_lower = search_term.lower()
            
            search_cols = ["nom", "marque", "motif", "risques", "sous_categorie", "distributeurs", "zone_vente"]
            search_cols_existing = [col for col in search_cols if col in df_filtered.columns]
            
            # Créer une colonne texte combinée pour la recherche
            # Utiliser str() pour gérer les types mixtes et fillna pour éviter les NaNs
            # Assurez-vous que toutes les colonnes sélectionnées existent dans df_filtered
            df_filtered['search_text'] = df_filtered[search_cols_existing].astype(str).fillna('').agg(' '.join, axis=1).str.lower()

            # Appliquer le filtre de recherche
            search_results_df = df_filtered[df_filtered['search_text'].str.contains(search_term_lower, na=False)]
            
            st.markdown(f"**{len(search_results_df)}** résultats trouvés pour '{search_term}' dans les données filtrées.")

            if not search_results_df.empty:
                # Afficher les résultats (on peut ajouter de la pagination ici aussi si nécessaire)
                # Limiter l'affichage si beaucoup de résultats, ou ajouter pagination
                num_search_results = len(search_results_df)
                max_display_search = 50 # Limiter à 50 résultats dans la recherche rapide pour la lisibilité

                if num_search_results > max_display_search:
                     st.info(f"Affichage des {max_display_search} premiers résultats sur {num_search_results} trouvés.")

                for i in range(min(num_search_results, max_display_search)):
                     display_recall_card(search_results_df.iloc[i])

                # Optionnel: ajouter pagination pour la recherche si > max_display_search

            else:
                st.warning(f"Aucun résultat trouvé pour '{search_term}' dans les données filtrées.")
        else:
            st.info("Entrez un terme dans la barre de recherche pour afficher les résultats.")

    # Footer
    st.markdown("""
    <div class="footer">
        RappelConso Insight - Application basée sur les données de <a href="https://www.rappelconso.gouv.fr/" target="_blank">RappelConso.gouv.fr</a>. Données fournies par data.economie.gouv.fr
    </div>
    """, unsafe_allow_html=True)

# Exécuter l'application
if __name__ == "__main__":
    main()
