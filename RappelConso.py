import streamlit as st
import pandas as pd
import requests
from datetime import datetime, date, timedelta, timezone
import time
import plotly.express as px
import numpy as np
import math
from groq import Groq # Importez la librairie Groq
import json # Pour une gestion potentielle plus fine du JSON de l'échantillon

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

     /* Style for AI analysis button */
     .stButton button[data-testid^="stButton"] { # Use startswith for potential multiple buttons
         /* Example styling */
         background-color: #3498db; /* Blue */
         color: white;
     }
      .stButton button[data-testid^="stButton"]:hover {
         background-color: #2980b9; /* Darker blue */
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
                 try:
                      st.sidebar.json(data) # Use json for dict/list
                 except Exception as e:
                      st.sidebar.write(f"Cannot display data as JSON: {e}")
                      st.sidebar.write(data) # Fallback to writing raw data
            elif isinstance(data, pd.DataFrame):
                st.sidebar.dataframe(data.head()) # Use dataframe for pandas
            else:
                 st.sidebar.write(data) # Default write for other types


# --- Liste prédéfinie de catégories ---
# Fixée à "alimentation" pour répondre à la demande.
CATEGORIES = ["alimentation"]

# --- Modèles Groq disponibles (liste mise à jour) ---
# Modèles avec grande fenêtre de contexte si disponibles
GROQ_MODELS = ["gemma2-9b-it", "llama-3.3-70b-versatile", "llama3-70b-8192", "deepseek-r1-distill-llama-70b", "llama3-8b-8192"] # Updated list

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
        "conduites_a_tenir_par_le_consommateur": "conduites_a_tenir",
        "date_debut_commercialisation": "date_debut_commercialisation",
        "date_date_fin_commercialisation": "date_fin_commercialisation",
        "temperature_conservation": "temperature_conservation",
        "marque_salubrite": "marque_salubrite",
        "informations_complementaires": "informations_complementaires",
        "description_complementaire_risque": "description_complementaire_risque",
        "numero_contact": "numero_contact",
        "modalites_de_compensation": "modalites_de_compensation",
        "date_de_fin_de_la_procedure_de_rappel": "date_fin_procedure"
    }
    
    # Appliquer le renommage des colonnes existantes
    df.rename(columns=column_mapping, inplace=True)
    
    # Convertir la colonne date_raw en datetime et extraire la date pour affichage
    if "date_raw" in df.columns:
        # Convertir en datetime, coercer les erreurs en NaT (Not a Time) et s'assurer qu'ils sont en UTC
        df['date'] = pd.to_datetime(df['date_raw'], errors='coerce', utc=True)
        # Extraire la partie date pour l'affichage (ignorer le timezeone pour le conversion str)
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

    # --- CORRECTION MAJEURE : Supprimer les doublons basés sur l'ID ---
    # L'ID de la fiche ("id" ou "numero_fiche") devrait être unique.
    # Si l'API retourne des doublons pour une raison quelconque, on les supprime ici.
    # Utiliser l'ID comme critère de déduplication.
    initial_rows = len(df)
    if 'id' in df.columns:
         # IMPORTANT: reset_index() est nécessaire après drop_duplicates si vous comptez utiliser .iloc[] par la suite
         df.drop_duplicates(subset=['id'], keep='first', inplace=True)
         df.reset_index(drop=True, inplace=True) # Reset index after dropping rows
         if len(df) < initial_rows:
              st.sidebar.warning(f"Supprimé {initial_rows - len(df)} doublons basés sur l'ID.")
    else:
         st.warning("La colonne 'id' est absente des données. Impossible de supprimer les doublons basés sur l'ID.")
         # Si 'id' est absent, tenter une déduplication sur un sous-ensemble de colonnes clés
         key_cols_for_dedup = ['nom', 'marque', 'date_str', 'motif', 'risques']
         key_cols_existing = [col for col in key_cols_for_dedup if col in df.columns]
         if len(key_cols_existing) > 0:
              df.drop_duplicates(subset=key_cols_existing, keep='first', inplace=True)
              df.reset_index(drop=True, inplace=True) # Reset index after dropping rows
              if len(df) < initial_rows:
                   st.sidebar.warning(f"Supprimé {initial_rows - len(df)} doublons basés sur un ensemble de colonnes clés.")


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
    # Utiliser .get pour éviter les KeyError si une colonne est manquante après chargement/filtrage
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
        st.markdown(f("<p><strong>Date de publication:</strong> {}</p>").format(date_str), unsafe_allow_html=True) 
        # CORRECTION : Syntaxe correcte des f-strings, corrigée ici
        st.markdown(f"<p><strong>Catégorie:</strong> {categorie} > {sous_categorie}</p>", unsafe_allow_html=True) 
        st.markdown(f"<p><strong>Risques:</strong> <span class='{risk_class}'>{risques}</span></p>", unsafe_allow_html=True)
        
        # Afficher l'image si disponible, centrée dans la colonne
        if image_url:
            try:
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
    # items_per_page est le nombre de cartes par page
    total_pages = math.ceil(total_items / items_per_page) if total_items > 0 else 1

    # S'assurer que la page actuelle ne dépasse pas le total de pages (peut arriver après filtrage ou recherche)
    current_page = st.session_state.get(current_page_state_key, 1)
    if current_page > total_pages:
         st.session_state[current_page_state_key] = total_pages
    if current_page < 1:
         st.session_state[current_page_state_key] = 1

    # Récupérer la valeur corrigée pour l'affichage et les boutons
    current_page = st.session_state.get(current_page_state_key, 1)


    col_prev, col_info, col_next = st.columns([1, 3, 1])

    with col_prev:
        if current_page > 1:
            # Clé dynamique basée sur le nom de l'état de page pour éviter les conflits
            if st.button("← Précédent", key=f"btn_prev_{current_page_state_key}"):
                st.session_state[current_page_state_key] = current_page - 1
                st.rerun()

    with col_info:
        st.markdown(f"<div style='text-align:center;'>Page {current_page} sur {total_pages}</div>", unsafe_allow_html=True)

    with col_next:
        if current_page < total_pages:
            # Clé dynamique basée sur le nom de l'état de page
            if st.button("Suivant →", key=f"btn_next_{current_page_state_key}"):
                st.session_state[current_page_state_key] = current_page + 1
                st.rerun()

# Fonction pour interagir avec l'API Groq
def get_groq_response(api_key, model, prompt):
    """Envoie un prompt à l'API Groq et retourne la réponse."""
    if not api_key:
        return "Erreur : Clé API Groq non fournie."
    if not model:
        return "Erreur : Modèle Groq non sélectionné."
    if not prompt:
        return "Erreur : Aucune question posée."

    # Nouveau System Prompt plus détaillé pour mieux guider l'IA sur l'analyse de l'échantillon JSON
    # Augmentation de la clarté sur l'analyse de l'échantillon pour les détails spécifiques
    system_prompt = """
    Vous êtes un expert analyste en sécurité alimentaire et en rappels de produits en France, spécialisé dans l'interprétation des données de RappelConso.
    Votre tâche est de répondre aux questions de l'utilisateur en vous basant *strictement* sur les données de rappels qui vous sont fournies dans le contexte.
    Les données sont un ensemble filtré de rappels de produits alimentaires. Le contexte inclut un résumé statistique général et un échantillon de rappels individuels en format JSON.
    
    Consignes importantes :
    1.  **Basez-vous UNIQUEMENT** sur les données fournies dans le contexte (Résumé des données et Échantillon de rappels JSON). N'utilisez pas de connaissances externes.
    2.  Le **Résumé des données** vous donne des statistiques générales (top motifs, top risques, etc.). C'est utile pour les tendances générales.
    3.  L'**Échantillon de rappels (JSON)** liste les détails *spécifiques* (nom, marque, sous-catégorie, motif) de chaque rappel de l'échantillon. Ce sont les données *brutes* que vous devez parcourir et analyser pour trouver des informations précises sur des cas particuliers ou des liens entre champs.
    4.  **POUR LES QUESTIONS SPECIFIQUES** (ex: quels produits sont associés à un risque particulier comme "bris de verre", quels distributeurs sont listés pour un motif donné, ou pour obtenir des détails sur des rappels précis), **vous DEVEZ IMPÉRATIVEMENT ANALYSER attentivement l'Échantillon de rappels (JSON)**. Parcourez les éléments de cet échantillon (chaque objet JSON dans la liste) et examinez attentivement les champs comme 'nom', 'marque', 'motif', 'sous_categorie', etc. Recherchez les termes clés de la question (ex: "verre", "métal", "listeria") dans ces champs.
    5.  Si vous trouvez des correspondances dans l'échantillon JSON, **liste les produits spécifiques trouvés dans l'échantillon** qui correspondent à la demande. Indiquez leur nom, marque, sous-catégorie et motif pour chaque produit trouvé. Si plusieurs produits sont trouvés, listez-les clairement.
    6.  Si une question ne peut pas être répondue avec les données fournies (parce que l'information n'apparaît pas du tout dans le résumé ET n'apparaît pas dans l'échantillon JSON), dites-le clairement (ex: "Je ne dispose pas d'informations suffisantes dans les données fournies pour répondre précisément à cette question. L'information [mentionner le type d'information recherchée, ex: "sur les rappels liés au verre"] n'apparaît pas dans l'échantillon de rappels disponible pour l'analyse.").
    7.  Structurez votre réponse de manière claire, en utilisant des tirets, des listes ou des paragraphes courts. Commencez par répondre directement à la question si possible, puis ajoutez des précisions ou des limitations basées sur les données.
    8.  Soyez concis et allez droit au but.
    9.  Ne mentionnez pas directement le format JSON ou les "indices" de l'échantillon dans votre réponse à l'utilisateur. Présentez les informations de manière naturelle, comme si vous aviez lu les fiches de rappel.
    10. Les colonnes importantes pour l'analyse sont : 'nom', 'marque', 'sous_categorie', 'motif'. Concentrez votre analyse sur celles-ci dans l'échantillon JSON. Les autres colonnes du DataFrame ('risques', 'date_str', 'zone_vente', 'distributeurs', 'conduites_a_tenir') sont incluses dans le résumé mais pas dans l'échantillon JSON pour réduire la taille.
    """

    try:
        client = Groq(api_key=api_key)

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt # Utilise le nouveau system prompt
                },
                {
                    "role": "user",
                    "content": prompt, # Le prompt inclut toujours le contexte des données + la question utilisateur
                }
            ],
            model=model,
            temperature=0.1, # Température basse pour des réponses factuelles
            max_tokens=3000, # Augmente légèrement les tokens max pour une réponse plus complète si nécessaire et compte tenu de la plus grande fenêtre de contexte de certains modèles
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        debug_log("Erreur API Groq", e)
        # Cacher la clé API dans le message d'erreur public
        error_message = str(e).replace(api_key, "[Votre Clé API]")
        return f"Une erreur est survenue lors de l'appel à l'API Groq : {error_message}"

# Fonction pour préparer le contexte des données filtrées pour l'IA
def prepare_data_context(df_filtered):
    """
    Prépare un résumé textuel et un échantillon de données brutes
    des données filtrées pour le modèle IA.
    """
    if df_filtered.empty:
        return "Aucune donnée de rappel n'est disponible pour l'analyse."

    total_count = len(df_filtered)
    date_min = df_filtered['date_str'].min() if 'date_str' in df_filtered.columns and not df_filtered['date_str'].isna().all() else "N/A"
    date_max = df_filtered['date_str'].max() if 'date_str' in df_filtered.columns and not df_filtered['date_str'].isna().all() else "N/A"

    context_summary = f"Analyse basée sur {total_count} rappels de produits alimentaires publiés entre le {date_min} et le {date_max}. "

    # Ajouter les informations des top N pour les colonnes pertinentes s'ils existent et ne sont pas vides après filtrage
    if 'sous_categorie' in df_filtered.columns and not df_filtered['sous_categorie'].dropna().empty:
        top_subcats = df_filtered['sous_categorie'].value_counts().nlargest(5).to_dict()
        context_summary += f"\nRésumé des sous-catégories les plus fréquentes : {top_subcats}. " # Clarifié "Résumé"

    if 'motif' in df_filtered.columns and not df_filtered['motif'].dropna().empty:
        top_motifs = df_filtered['motif'].value_counts().nlargest(5).to_dict()
        context_summary += f"\nRésumé des motifs de rappel les plus fréquents : {top_motifs}. " # Clarifié "Résumé"

    if 'risques' in df_filtered.columns and not df_filtered['risques'].dropna().empty:
        top_risks = df_filtered['risques'].value_counts().nlargest(5).to_dict()
        context_summary += f"\nRésumé des risques les plus fréquents : {top_risks}. " # Clarifié "Résumé"
        
    if 'marque' in df_filtered.columns and not df_filtered['marque'].dropna().empty:
        top_brands = df_filtered['marque'].value_counts().nlargest(5).to_dict()
        context_summary += f"\nRésumé des marques avec le plus de rappels : {top_brands}. " # Clarifié "Résumé"

    # Inclure un échantillon des données brutes (limiter le nombre de lignes et les colonnes)
    # Sélectionner les colonnes spécifiées pour l'échantillon JSON
    relevant_cols_for_sample = ['nom', 'marque', 'sous_categorie', 'motif'] # Colonnes limitées comme demandé

    # Filtrer pour garder uniquement les colonnes qui existent dans df_filtered
    relevant_cols_existing_in_filtered = [col for col in relevant_cols_for_sample if col in df_filtered.columns]

    # Taille de l'échantillon JSON augmentée pour les modèles avec grande fenêtre de contexte
    # Attention : même avec une grande fenêtre, 200 rappels peuvent être volumineux. Réduire si les erreurs 413 persistent.
    sample_size = min(len(df_filtered), 200) # Maintien l'échantillon à 200

    # Utiliser head(sample_size) pour les plus récents et sélectionner les colonnes pertinentes pour l'échantillon
    df_sample = df_filtered[relevant_cols_existing_in_filtered].head(sample_size) 

    context_sample = ""
    if not df_sample.empty:
        # Convertir explicitement l'échantillon entier en chaîne de caractères pour éviter les erreurs de sérialisation
        df_sample_str = df_sample.astype(str) 
        
        context_sample = f"\n\nÉchantillon de {len(df_sample_str)} rappels (JSON) :\n" # Indique la taille de l'échantillon
        # Convertir l'échantillon en string JSON.
        try:
            list_of_dicts = df_sample_str.to_dict(orient='records')
            # Utilisation de json.dumps pour plus de contrôle et de robustesse
            context_sample += json.dumps(list_of_dicts, indent=2, ensure_ascii=False)

        except Exception as e:
             debug_log("Error converting sample to JSON after astype(str)", e)
             context_sample = f"\nÉchantillon de rappels (JSON) : Échantillon non disponible en raison d'une erreur de formatage ({e}).\n" 


    full_context = context_summary + context_sample

    return full_context

# --- Fonction principale ---
def main():
    # Initialiser les états de session nécessaires avant toute utilisation
    # Cette initialisation en début de main() est la plus fiable dans Streamlit
    if "rappel_data" not in st.session_state: st.session_state.rappel_data = None
    if "load_params" not in st.session_state: st.session_state.load_params = None
    if "current_page" not in st.session_state: st.session_state.current_page = 1
    if "search_current_page" not in st.session_state: st.session_state.search_current_page = 1
    if "selected_subcategories" not in st.session_state: st.session_state.selected_subcategories = ["Toutes"]
    if "selected_brands" not in st.session_state: st.session_state.selected_brands = ["Toutes"]
    if "selected_risks" not in st.session_state: st.session_state.selected_risks = ["Tous"]
    if "quick_search_input" not in st.session_state: st.session_state.quick_search_input = ""
    # Correction ici : utiliser une clé différente pour l'état de la valeur du widget
    if "last_search_term_widget_value" not in st.session_state:
         st.session_state.last_search_term_widget_value = ""
    if "items_per_page_main" not in st.session_state: st.session_state.items_per_page_main = 10
    if "items_per_page_search" not in st.session_state: st.session_state.items_per_page_search = 10
    if "groq_api_key" not in st.session_state: st.session_state.groq_api_key = ""
    # Correction ici : s'assurer que l'état groq_model est un modèle valide dès l'initialisation
    if "groq_model" not in st.session_state or (st.session_state.groq_model is not None and st.session_state.groq_model not in GROQ_MODELS):
        st.session_state.groq_model = GROQ_MODELS[0] if GROQ_MODELS else None # Utiliser le premier modèle comme défaut si la liste n'est pas vide
    if "ai_question" not in st.session_state: st.session_state.ai_question = ""
    if "ai_response" not in st.session_state: st.session_state.ai_response = ""
    if "last_ai_question" not in st.session_state: st.session_state.last_ai_question = ""


    # En-tête (identique)
    st.markdown("""
    <div class="header">
        <h1>RappelConso Insight</h1>
        <p>Analyse des rappels de produits en France</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Sidebar pour les paramètres ---
    st.sidebar.title("Paramètres")

    st.sidebar.subheader("Chargement des données")
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
        st.session_state.last_search_term_widget_value = "" # Reset search term state for pagination check
        st.session_state.quick_search_input = "" # Clear search input widget via session state
        # Réinitialiser la réponse IA et la question
        st.session_state.ai_response = ""
        st.session_state.last_ai_question = ""


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
    st.sidebar.subheader("Filtres d'analyse")

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
        # Utiliser isin et une liste pour gérer les NaN correctement si besoin, mais dropna() sur les options devrait suffire
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
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Liste des rappels", "📊 Visualisations", "🔍 Recherche rapide", "🤖 Analyse IA"])
        with tab1: st.info("Aucun rappel à afficher avec les filtres sélectionnés.")
        with tab2: st.info("Aucune donnée à visualiser avec les filtres sélectionnés.")
        with tab3: st.info("Aucune donnée à rechercher avec les filtres sélectionnés.")
        with tab4: st.info("Aucune donnée disponible pour l'analyse IA avec les filtres sélectionnés.")

        # Réinitialiser la réponse IA si le filtre devient vide
        st.session_state.ai_response = ""
        st.session_state.last_ai_question = ""

        return # Arrêter l'exécution de main() ici si df_filtered est vide


    # --- Afficher quelques métriques ---
    st.subheader("Vue d'overview") # Changed title to avoid confusion with "Analyse IA"
    col1, col2, col3, col4 = st.columns(4)
    
    # Nombre total de rappels chargés (based on the initial API load)
    with col1:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-value">{len(df)}</div>
            <div>Rappels chargés</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Nombre de rappels filtrés actuellement affichés (based on df_filtered)
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
         # Correction : Générer le CSV à la demande dans le lambda pour s'assurer qu'il est à jour
         @st.cache_data(ttl=60) # Cache le CSV pour 60s pour éviter de le regénérer à chaque rerun minor
         def convert_df_to_csv(df):
             # Utiliser .copy() ici avant to_csv pour éviter les SettingWithCopyWarning potentielles
             # si df_filtered_copy était modifiée avant le to_csv
             return df.to_csv(index=False).encode('utf-8')

         csv_data = convert_df_to_csv(df_filtered)

         st.download_button(
             label="Télécharger les données filtrées (CSV)",
             data=csv_data,
             file_name=f'rappelconso_alimentation_depuis_{start_date.strftime("%Y-%m-%d")}_filtered.csv',
             mime='text/csv',
             key="download_filtered_csv" # Clé unique
         )
         st.markdown("---") # Séparateur visuel


    # --- Section Analyse IA dans la Sidebar ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Analyse IA")
    st.sidebar.info("Analysez les données filtrées à l'aide de l'IA Groq.")

    # Input Clé API Groq
    groq_api_key = st.sidebar.text_input(
        "Votre clé API Groq :",
        type="password",
        value=st.session_state.groq_api_key,
        key="groq_api_key_input", # Clé unique
        help="Entrez votre clé API Groq (vous pouvez l'obtenir sur console.groq.com). La clé n'est pas enregistrée durablement."
    )
    st.session_state.groq_api_key = groq_api_key # Sauvegarder dans l'état de session

    # Sélection du modèle Groq
    groq_model = st.sidebar.selectbox(
        "Choisissez le modèle :",
        options=GROQ_MODELS,
        index=GROQ_MODELS.index(st.session_state.groq_model) if st.session_state.groq_model in GROQ_MODELS else 0,
        key="groq_model_select", # Clé unique
        help="Sélectionnez le modèle d'IA Groq à utiliser. Les modèles avec grande fenêtre de contexte (ex: llama-3.3-70b-versatile) sont meilleurs pour analyser l'échantillon JSON."
    )
    st.session_state.groq_model = groq_model # Sauvegarder dans l'état de session

    # Champ de texte pour la question de l'utilisateur
    ai_question = st.sidebar.text_area(
        "Posez une question sur les données filtrées :",
        value=st.session_state.ai_question,
        key="ai_question_input", # Clé unique
        height=150,
        placeholder="Ex: Quels sont les principaux produits qui ont eu un rappel pour présence de verre ? Y a-t-il des tendances récentes ?",
        help="L'IA analysera les données actuellement affichées (filtrées)."
    )
    st.session_state.ai_question = ai_question # Sauvegarder dans l'état de session

    # Bouton pour déclencher l'analyse
    analyze_button = st.sidebar.button("Analyser avec l'IA", type="secondary", key="analyze_ai_button")

    # --- Logic to trigger AI analysis ---
    if analyze_button:
         if not groq_api_key:
             st.sidebar.error("Veuillez entrer votre clé API Groq.")
         elif not groq_model:
             st.sidebar.error("Veuillez sélectionner un modèle Groq.")
         elif not ai_question:
             st.sidebar.warning("Veuillez poser une question.")
         # La vérification de df_filtered.empty est faite plus haut dans main()
         # et le return arrête l'exécution si les données filtrées sont vides.
         else:
             # Préparer le prompt pour l'IA
             data_context = prepare_data_context(df_filtered)
             # Le prompt inclut le system prompt (dans get_groq_response) et ce contenu utilisateur
             full_prompt_content = f"Données contextuelles sur les rappels :\n{data_context}\n\nQuestion de l'utilisateur :\n{ai_question}\n\nRéponse :"

             # Appeler l'API Groq avec un spinner
             with st.spinner("L'IA analyse les données..."):
                 st.session_state.ai_response = get_groq_response(groq_api_key, groq_model, full_prompt_content)
                 st.session_state.last_ai_question = ai_question # Sauvegarder la question posée

             st.toast("Analyse IA terminée !", icon="🤖")
             # Rerun pour mettre à jour l'affichage principal avec la réponse
             st.rerun()


    # --- Afficher les onglets (avec le nouvel onglet IA) ---
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Liste des rappels", "📊 Visualisations", "🔍 Recherche rapide", "🤖 Analyse IA"])
    
    # Stocker l'onglet actif si vous voulez qu'il reste sélectionné
    # current_active_tab = st.session_state.get("current_tab", "Liste des rappels")
    # Assigner le contenu aux onglets...

    with tab1:
        # ... (le contenu de l'onglet Liste des rappels) ...
        st.subheader("Liste des rappels filtrés")
        
        items_per_page_main = st.select_slider(
            "Rappels par page:",
            options=[5, 10, 20, 50],
            value=st.session_state.items_per_page_main, # Utiliser l'état de session
            key="slider_items_per_page_main"
        )
        st.session_state.items_per_page_main = items_per_page_main

        # Recalculer le hash de pagination si les filtres ou le nombre d'items par page changent
        current_pagination_state_main = hash((
            tuple(st.session_state.selected_subcategories),
            tuple(st.session_state.selected_brands),
            tuple(st.session_state.selected_risks),
            items_per_page_main
        ))
        
        # Comparer le hash actuel avec le dernier hash enregistré pour la pagination principale
        # Utiliser .get() avec un défaut pour éviter les erreurs au tout premier run
        if st.session_state.get("pagination_state_main") != current_pagination_state_main:
             st.session_state.current_page = 1 # Réinitialiser à la page 1 si les filtres/items par page changent
             st.session_state.pagination_state_main = current_pagination_state_main # Mettre à jour le hash enregistré

        total_items_main = len(df_filtered)
        
        # Afficher les rappels de la page actuelle en 2 colonnes
        start_idx_main = (st.session_state.current_page - 1) * items_per_page_main
        end_idx_main = min(start_idx_main + items_per_page_main, total_items_main)

        if total_items_main > 0:
            # CORRECTION: Obtenir la sous-section du DataFrame pour la page actuelle
            # Utiliser .iloc pour la sélection par position, puis reset_index pour avoir des indices 0, 1, 2... sur la page
            # .copy() est une bonne pratique ici pour éviter les SettingWithCopyWarning potentielles
            page_df_main = df_filtered.iloc[start_idx_main:end_idx_main].copy().reset_index(drop=True) 
            
            # Itérer sur les indices locaux de cette sous-section (0, 1, 2...) par pas de 2
            # len(page_df_main) donne le nombre d'éléments sur la page actuelle
            for i in range(0, len(page_df_main), 2):
                col1, col2 = st.columns(2)
                
                # Afficher le premier produit de la paire en utilisant l'index local 'i' de page_df_main
                with col1:
                    # Grâce à reset_index(), l'index 'i' est valide si i < len(page_df_main)
                    display_recall_card(page_df_main.iloc[i])
                
                # Afficher le deuxième produit si il existe dans la sous-section
                if i + 1 < len(page_df_main):
                    with col2:
                         display_recall_card(page_df_main.iloc[i+1])
                # else:
                #    avec col2: st.empty() # Streamlit gère les colonnes vides automatiquement

            display_pagination_controls("current_page", total_items_main, items_per_page_main)

        # else: (message déjà géré par le bloc 'if df_filtered.empty' plus haut)


    with tab2:
        # Contenu de l'onglet Visualisations (identique, utilise les fonctions modifiées)
        st.subheader("Visualisations des données filtrées")
        
        st.write("### Évolution des rappels par mois")
        df_filtered_valid_dates = df_filtered.dropna(subset=['date'])
        if "date" in df_filtered_valid_dates.columns and not df_filtered_valid_dates.empty:
            try:
                # Grouper par mois en utilisant la colonne datetime.dt.to_period('M')
                # et convertir en string pour l'axe Plotly.
                monthly_counts = df_filtered_valid_dates.groupby(df_filtered_valid_dates['date'].dt.to_period('M')).size().reset_index(name="count")
                monthly_counts["month_str"] = monthly_counts["date"].astype(str)
                monthly_counts = monthly_counts.sort_values("month_str")
                if not monthly_counts.empty:
                    fig_time = px.line(monthly_counts, x="month_str", y="count", title="Nombre de rappels par mois", labels={"month_str": "Mois", "count": "Nombre de rappels"}, markers=True)
                    fig_time.update_layout(xaxis_title="Mois", yaxis_title="Nombre de rappels", hovermode="x unified")
                    st.plotly_chart(fig_time, use_container_width=True)
                else: st.warning("Pas assez de données temporelles valides pour créer un graphique.")
            except Exception as e: st.error(f"Erreur lors de la création du graphique temporel: {str(e)}"); debug_log("Time series chart error", e)
        else: st.info("La colonne de date n'est pas disponible ou contient des valeurs manquantes dans les données filtrées.")

        if "sous_categorie" in df_filtered.columns:
            st.write("### Répartition par sous-catégorie")
            valid_subcats = df_filtered["sous_categorie"].dropna()
            if not valid_subcats.empty:
                num_top_subcats = st.slider("Nombre de sous-catégories à afficher:", 5, 30, 10, key="subcat_slider")
                top_subcats = valid_subcats.value_counts().nlargest(num_top_subcats)
                fig_subcat = px.pie(values=top_subcats.values, names=top_subcats.index, title=f"Top {num_top_subcats} des sous-catégories", hole=0.3)
                fig_subcat.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_subcat, use_container_width=True)
            else: st.info("Aucune donnée de sous-catégorie disponible dans les données filtrées.")

        if "risques" in df_filtered.columns:
            st.write("### Répartition par type de risque")
            valid_risks = df_filtered["risques"].dropna()
            if not valid_risks.empty:
                risk_counts = valid_risks.value_counts().reset_index()
                risk_counts.columns = ["risk", "count"]
                risk_counts['risk_level'] = risk_counts['risk'].apply(get_risk_class)
                risk_counts['risk_level_label'] = risk_counts['risk_level'].map({'risk-high': 'Risque Élevé', 'risk-medium': 'Risque Moyen', 'risk-low': 'Risque Faible'})
                risk_level_order = ['Risque Élevé', 'Risque Moyen', 'Risque Faible']
                risk_counts['risk_level_label'] = pd.Categorical(risk_counts['risk_level_label'], categories=risk_level_order, ordered=True)
                risk_counts = risk_counts.sort_values(['risk_level_label', 'count'], ascending=[True, False])
                fig_risks = px.bar(risk_counts, x="risk", y="count", title="Répartition par type de risque", labels={"risk": "Type de risque", "count": "Nombre de rappels", "risk_level_label": "Niveau de risque"}, color="risk_level_label", color_discrete_map={'Risque Élevé': '#e74c3c', 'Risque Moyen': '#f39c12', 'Risque Faible': '#27ae60'})
                fig_risks.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_risks, use_container_width=True)
            else: st.info("Aucune donnée de risque disponible dans les données filtrées.")

        if "marque" in df_filtered.columns:
             st.write("### Marques les plus fréquentes")
             valid_brands = df_filtered["marque"].dropna()
             if not valid_brands.empty:
                 num_top_brands = st.slider("Nombre de marques à afficher:", 5, 30, 10, key="top_brands_slider")
                 top_brands = valid_brands.value_counts().nlargest(num_top_brands)
                 fig_brands = px.bar(x=top_brands.index, y=top_brands.values, title=f"Top {num_top_brands} des marques", labels={"x": "Marque", "y": "Nombre de rappels"})
                 fig_brands.update_layout(xaxis_tickangle=-45)
                 st.plotly_chart(fig_brands, use_container_width=True)
             else: st.info("Aucune donnée de marque disponible dans les données filtrées.")

    with tab3:
        st.subheader("Recherche rapide dans les rappels affichés (champ: Motif)")
        
        # L'input de texte lui-même met à jour st.session_state.quick_search_input
        search_term = st.text_input(
            "Entrez un terme pour rechercher dans le Motif du rappel:",
            placeholder="Ex: listéria, salmonelle, corps étranger",
            value=st.session_state.quick_search_input, # Utilisez la valeur de l'état pour initialiser le widget
            key="quick_search_input" # Cette clé gère l'état automatiquement
        )
        # CORRECTION: Ligne redondante supprimée

        search_results_df = pd.DataFrame()
        # Appliquer la recherche si search_term n'est PAS vide (pour éviter de tout afficher par défaut)
        if search_term:
            search_term_lower = search_term.lower()
            search_cols = ["motif"] # Recherche UNIQUEMENT sur le motif
            search_cols_existing = [col for col in search_cols if col in df_filtered.columns]
            
            if search_cols_existing:
                 df_filtered_with_search = df_filtered.copy()
                 # S'assurer que 'motif' est bien traité comme string avant de mettre en minuscule
                 # On accède directement à la colonne 'motif' car search_cols_existing ne contient que 'motif'
                 # S'assurer que la colonne 'motif' existe avant d'essayer d'y accéder
                 if 'motif' in df_filtered_with_search.columns:
                      df_filtered_with_search['search_text'] = df_filtered_with_search['motif'].astype(str).str.lower()
                 else:
                      # Si 'motif' n'existe pas (très improbable si load_rappel_data fonctionne), la recherche ne trouvera rien
                      df_filtered_with_search['search_text'] = "" 


                 search_results_df = df_filtered_with_search[df_filtered_with_search['search_text'].str.contains(search_term_lower, na=False)].copy()

            st.markdown(f"**{len(search_results_df)}** résultats trouvés pour '{search_term}' dans le Motif.")

        # Afficher les résultats SEULEMENT si search_term n'est PAS vide ET qu'il y a des résultats
        if search_term and not search_results_df.empty:
            st.markdown("---")
            st.write("Navigation dans les résultats de recherche:")

            items_per_page_search = st.select_slider("Résultats par page:", options=[5, 10, 20, 50], value=st.session_state.items_per_page_search, key="slider_items_per_page_search")
            st.session_state.items_per_page_search = items_per_page_search

            # Hacher la valeur actuelle du widget search_term
            current_pagination_state_search = hash((st.session_state.quick_search_input.lower() if st.session_state.quick_search_input else "", items_per_page_search))
            
            # Réinitialiser seulement si le terme de recherche (du widget) change (et il n'est pas vide), OU si la pagination_state change
            if (st.session_state.get("last_search_term_widget_value", "") != (st.session_state.quick_search_input.lower() if st.session_state.quick_search_input else "") and st.session_state.quick_search_input != "") or \
               ("pagination_state_search" not in st.session_state or st.session_state.pagination_state_search != current_pagination_state_search):
                 st.session_state.search_current_page = 1
                 st.session_state.pagination_state_search = current_pagination_state_search
            # Mettre à jour le dernier terme du widget stocké pour la prochaine comparaison
            st.session_state.last_search_term_widget_value = st.session_state.quick_search_input.lower() if st.session_state.quick_search_input else ""


            total_items_search = len(search_results_df)
            start_idx_search = (st.session_state.search_current_page - 1) * items_per_page_search
            end_idx_search = min(start_idx_search + items_per_page_search, total_items_search)

            # CORRECTION: Obtenir la sous-section du DataFrame pour la page actuelle de recherche
            # Utiliser .iloc pour la sélection par position, puis reset_index pour avoir des indices 0, 1, 2... sur la page
            # .copy() est une bonne pratique ici pour éviter les SettingWithCopyWarning potentielles
            page_df_search = search_results_df.iloc[start_idx_search:end_idx_search].copy().reset_index(drop=True) 

            # Afficher en 2 colonnes en itérant sur les indices locaux de cette sous-section
            for i in range(0, len(page_df_search), 2):
                col1, col2 = st.columns(2)
                # Afficher les produits en utilisant l'index local 'i' de page_df_search
                with col1:
                     if i < len(page_df_search): # Double check index validity
                        display_recall_card(page_df_search.iloc[i])
                if i + 1 < len(page_df_search):
                    with col2:
                         display_recall_card(page_df_search.iloc[i+1])

            display_pagination_controls("search_current_page", total_items_search, items_per_page_search)

        # Afficher les messages d'info/warning seulement si search_term est vide ou s'il y a une recherche sans résultat
        elif not search_term:
             st.info("Entrez un terme dans la barre de recherche pour afficher les résultats (recherche limitée au Motif).")
        # Le cas else (search_term non vide mais search_results_df vide) est déjà géré par le message "X résultats trouvés pour..."

    with tab4:
        st.subheader("🤖 Analyse des données filtrées par l'IA Groq")
        # Afficher la réponse IA stockée
        if st.session_state.ai_response:
            st.write("#### Question posée :")
            # Afficher la dernière question posée
            st.markdown(st.session_state.get("last_ai_question", "N/A")) 
            st.write("#### Réponse de l'IA :")
            # Afficher la réponse de l'IA formatée
            st.markdown(st.session_state.ai_response) 
        else:
            st.info("Aucune analyse IA n'a été effectuée. Entrez votre clé API et votre question dans la section 'Analyse IA' de la barre latérale et cliquez sur 'Analyser avec l'IA'.")


    # Footer (identique)
    st.markdown("""
    <div class="footer">
        RappelConso Insight - Application basée sur les données de <a href="https://www.rappelconso.gouv.fr/" target="_blank">RappelConso.gouv.fr</a>. Données fournies par data.economie.gouv.fr
    </div>
    """, unsafe_allow_html=True)


# Exécuter l'application
if __name__ == "__main__":
    # Initialisation des états de session (redondant avec main mais sécurise)
    if "rappel_data" not in st.session_state: st.session_state.rappel_data = None
    if "load_params" not in st.session_state: st.session_state.load_params = None
    if "current_page" not in st.session_state: st.session_state.current_page = 1
    if "search_current_page" not in st.session_state: st.session_state.search_current_page = 1
    if "selected_subcategories" not in st.session_state: st.session_state.selected_subcategories = ["Toutes"]
    if "selected_brands" not in st.session_state: st.session_state.selected_brands = ["Toutes"]
    if "selected_risks" not in st.session_state: st.session_state.selected_risks = ["Tous"]
    if "quick_search_input" not in st.session_state: st.session_state.quick_search_input = ""
    # Correction ici : utiliser une clé différente pour l'état de la valeur du widget
    if "last_search_term_widget_value" not in st.session_state:
         st.session_state.last_search_term_widget_value = ""
    if "items_per_page_main" not in st.session_state: st.session_state.items_per_page_main = 10
    if "items_per_page_search" not in st.session_state: st.session_state.items_per_page_search = 10
    if "groq_api_key" not in st.session_state: st.session_state.groq_api_key = ""
    # Correction ici : s'assurer que l'état groq_model est un modèle valide dès l'initialisation
    if "groq_model" not in st.session_state or (st.session_state.groq_model is not None and st.session_state.groq_model not in GROQ_MODELS):
        st.session_state.groq_model = GROQ_MODELS[0] if GROQ_MODELS else None # Utiliser le premier modèle comme défaut si la liste n'est pas vide
    if "ai_question" not in st.session_state: st.session_state.ai_question = ""
    if "ai_response" not in st.session_state: st.session_state.ai_response = ""
    if "last_ai_question" not in st.session_state: st.session_state.last_ai_question = ""

    main()
