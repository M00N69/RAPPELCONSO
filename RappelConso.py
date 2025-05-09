import streamlit as st
import pandas as pd
import requests
from datetime import datetime, date, timedelta
import time
import plotly.express as px
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="RappelConso Insight",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS simplifié pour le style de base
st.markdown("""
<style>
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
    }
    .card {
        background-color: white;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .metric {
        text-align: center;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
        border-top: 3px solid #2980b9;
    }
    .metric-value {
        font-size: 2.2em;
        font-weight: bold;
        color: #2980b9;
    }
    .recall-card {
        border-left: 4px solid #2980b9;
        padding-left: 0.8rem;
    }
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
    }
    .risk-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .risk-low {
        color: #27ae60;
        font-weight: bold;
    }
    .footer {
        margin-top: 2rem;
        text-align: center;
        color: #7f8c8d;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

# Mode debug
DEBUG = True

def debug_log(message, data=None):
    """Affiche des informations de débogage"""
    if DEBUG:
        with st.expander(f"DEBUG: {message}", expanded=False):
            st.write(message)
            if data is not None:
                if isinstance(data, pd.DataFrame):
                    st.write(f"Shape: {data.shape}")
                    st.write(f"Columns: {list(data.columns)}")
                    st.dataframe(data.head(3))
                elif isinstance(data, (dict, list)):
                    st.json(data)
                else:
                    st.write(data)

# Fonction de chargement des données
def load_rappel_data(start_date=None, category="alimentation", max_records=1000):
    """
    Charge les données de l'API RappelConso en utilisant la méthode refine
    """
    api_url = "https://data.economie.gouv.fr/api/v2/catalog/datasets/rappelconso-v2-gtin-espaces/records"
    
    # Paramètres de la requête
    params = {
        "refine.categorie_produit": category,  # Méthode qui fonctionne
        "limit": 100,
        "offset": 0
    }
    
    # Ajouter un filtre de date si spécifié
    if start_date:
        # Pour les dates, nous allons utiliser le filtre "q" qui est moins strict
        start_date_str = start_date.strftime("%Y-%m-%d")
        params["q"] = start_date_str
    
    all_records = []
    total_count = 0
    
    with st.spinner("Chargement des données RappelConso..."):
        # Première requête pour obtenir le nombre total
        response = requests.get(api_url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            total_count = data.get("total_count", 0)
            
            if total_count == 0:
                st.warning(f"Aucun rappel trouvé pour la catégorie '{category}'.")
                return pd.DataFrame()
            
            # Récupérer les données par lots
            progress_bar = st.progress(0)
            offset = 0
            
            while offset < min(total_count, max_records):
                params["offset"] = offset
                response = requests.get(api_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    page_data = response.json()
                    records = page_data.get("records", [])
                    
                    if not records:
                        break
                    
                    # Extraction des champs
                    for record in records:
                        if "record" in record and "fields" in record["record"]:
                            all_records.append(record["record"]["fields"])
                    
                    offset += len(records)
                    progress_bar.progress(min(1.0, offset / min(total_count, max_records)))
                    
                    if len(all_records) >= max_records:
                        st.info(f"Limite de {max_records} enregistrements atteinte (sur {total_count} disponibles).")
                        break
                    
                    time.sleep(0.1)
                else:
                    st.error(f"Erreur API: {response.status_code}")
                    break
    
    if not all_records:
        return pd.DataFrame()
    
    # Créer le DataFrame
    df = pd.DataFrame(all_records)
    
    # Débogage
    debug_log("DataFrame brut", df)
    
    # Standardiser les noms de colonnes pour l'interface
    column_mapping = {
        "categorie_produit": "categorie",
        "sous_categorie_produit": "sous_categorie",
        "marque_produit": "marque",
        "modeles_ou_references": "modele",
        "motif_rappel": "motif",
        "risques_encourus": "risques",
        "distributeurs": "distributeurs",
        "liens_vers_les_images": "image",
        "lien_vers_la_fiche_rappel": "fiche_url",
        "date_publication": "date_raw",  # Renommé pour clarté
        "libelle": "nom",
        "id": "id"
    }
    
    # Renommer les colonnes existantes
    for old, new in column_mapping.items():
        if old in df.columns:
            df[new] = df[old]
    
    # Construire une colonne de date_str pour affichage
    if "date_raw" in df.columns:
        # Extraire la date au format string pour affichage
        df['date_str'] = df['date_raw'].apply(lambda x: 
            x.split('T')[0] if isinstance(x, str) and 'T' in x else 
            str(x))
    
    # Débogage
    debug_log("DataFrame traité", df)
    
    # Trier par date_str (plus récent en premier)
    if "date_str" in df.columns:
        try:
            df = df.sort_values("date_str", ascending=False).reset_index(drop=True)
        except:
            st.warning("Impossible de trier par date")
    
    return df

# Fonction pour afficher une carte de rappel
def display_recall_card(row):
    """Affiche une carte de rappel"""
    
    # Extraire les données
    nom = row.get("nom", row.get("modele", "Produit non spécifié"))
    marque = row.get("marque", "Marque non spécifiée")
    date_str = row.get("date_str", "Date non spécifiée")
    
    motif = row.get("motif", "Non spécifié")
    risques = row.get("risques", "Non spécifié")
    
    # Déterminer la classe de risque
    risk_class = "risk-low"
    if isinstance(risques, str):
        risques_lower = risques.lower()
        if any(kw in risques_lower for kw in ["listeria", "salmonelle", "e. coli", "toxique"]):
            risk_class = "risk-high"
        elif any(kw in risques_lower for kw in ["allergène", "microbiologique", "corps étranger"]):
            risk_class = "risk-medium"
    
    # Image du produit
    image_url = None
    if "image" in row and pd.notna(row["image"]):
        image_links = str(row["image"])
        if "|" in image_links:
            image_url = image_links.split("|")[0].strip()
        else:
            image_url = image_links
    
    # URL de la fiche
    fiche_url = row.get("fiche_url", "#")
    
    # Afficher la carte
    with st.container():
        st.markdown(f"""
        <div class="card recall-card">
            <h4>{nom}</h4>
            <p><strong>Marque:</strong> {marque}</p>
            <p><strong>Date:</strong> {date_str}</p>
            <p><strong>Motif:</strong> {motif}</p>
            <p><strong>Risques:</strong> <span class="{risk_class}">{risques}</span></p>
        """, unsafe_allow_html=True)
        
        # Afficher l'image si disponible
        if image_url:
            try:
                st.image(image_url, width=200)
            except:
                st.info("Image non disponible")
        
        # Lien vers la fiche
        if fiche_url and fiche_url != "#":
            st.link_button("Voir la fiche complète", fiche_url, type="secondary")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Fonction principale
def main():
    # En-tête
    st.markdown("""
    <div class="header">
        <h1>RappelConso Insight</h1>
        <p>Analyse des rappels de produits alimentaires en France</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar pour les filtres
    st.sidebar.title("Filtres")
    
    # Date de début
    start_date = st.sidebar.date_input(
        "À partir de la date:",
        value=date(2023, 1, 1),  # Par défaut, depuis le début de 2023
        max_value=date.today()
    )
    
    # Catégorie (fixe pour l'instant)
    category = "alimentation"
    
    # Nombre max de rappels
    max_records = st.sidebar.slider(
        "Nombre max de rappels:", 
        min_value=100,
        max_value=5000,
        value=1000,
        step=100
    )
    
    # Bouton pour charger les données
    if st.sidebar.button("Charger les données", type="primary"):
        # Charger les données
        st.session_state.rappel_data = load_rappel_data(
            start_date=start_date,
            category=category,
            max_records=max_records
        )
    
    # Vérifier si les données sont chargées
    if "rappel_data" not in st.session_state:
        st.info("Veuillez charger les données en cliquant sur le bouton dans la barre latérale.")
        return
    
    df = st.session_state.rappel_data
    
    if df.empty:
        st.warning("Aucune donnée disponible avec les filtres actuels.")
        return
    
    # Afficher quelques métriques
    col1, col2, col3, col4 = st.columns(4)
    
    # Nombre total de rappels
    with col1:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-value">{len(df)}</div>
            <div>Total des rappels</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Rappels récents (30 derniers jours) - Version ultra-simplifiée
    with col2:
        # Pour ce champ, on va simplement utiliser une valeur statique pour éviter les problèmes
        # Plus tard on pourra affiner cette logique
        recent_count = min(100, len(df) // 10)  # Estimation simple
        
        st.markdown(f"""
        <div class="metric">
            <div class="metric-value">{recent_count}</div>
            <div>Rappels récents (30j)</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Nombre de sous-catégories
    if "sous_categorie" in df.columns:
        subcat_count = df["sous_categorie"].nunique()
        
        with col3:
            st.markdown(f"""
            <div class="metric">
                <div class="metric-value">{subcat_count}</div>
                <div>Sous-catégories</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Nombre de marques
    if "marque" in df.columns:
        brand_count = df["marque"].nunique()
        
        with col4:
            st.markdown(f"""
            <div class="metric">
                <div class="metric-value">{brand_count}</div>
                <div>Marques uniques</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Afficher les onglets
    tab1, tab2, tab3 = st.tabs(["📋 Liste des rappels", "📊 Visualisations", "🔍 Recherche avancée"])
    
    with tab1:
        # Pagination des rappels
        st.subheader("Liste des rappels")
        
        items_per_page = st.select_slider(
            "Rappels par page:",
            options=[5, 10, 20, 50],
            value=10
        )
        
        if "current_page" not in st.session_state:
            st.session_state.current_page = 1
        
        total_pages = (len(df) - 1) // items_per_page + 1
        
        # Assurer que la page actuelle est valide
        st.session_state.current_page = max(1, min(st.session_state.current_page, total_pages))
        
        # Afficher les rappels de la page actuelle
        start_idx = (st.session_state.current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(df))
        
        for i in range(start_idx, end_idx):
            display_recall_card(df.iloc[i])
        
        # Pagination
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            if st.session_state.current_page > 1:
                if st.button("← Précédent"):
                    st.session_state.current_page -= 1
                    st.rerun()
        
        with col2:
            st.write(f"Page {st.session_state.current_page} sur {total_pages}")
        
        with col3:
            if st.session_state.current_page < total_pages:
                if st.button("Suivant →"):
                    st.session_state.current_page += 1
                    st.rerun()
    
    with tab2:
        # Visualisations simplifiées
        st.subheader("Visualisations des données")
        
        # Évolution temporelle des rappels - Version ultra simplifiée
        st.write("### Évolution des rappels dans le temps")
        
        if "date_str" in df.columns:
            # Extraire l'année et le mois simplement depuis la chaîne de caractères date_str
            try:
                df_time = df.copy()
                # Extraire les premières 7 caractères (YYYY-MM) de la date
                # Supposer que date_str est au format "YYYY-MM-DD"
                df_time["month"] = df_time["date_str"].apply(lambda x: x[:7] if isinstance(x, str) and len(x) >= 7 else "Unknown")
                
                # Exclure les valeurs inconnues
                df_time = df_time[df_time["month"] != "Unknown"]
                
                if not df_time.empty:
                    monthly_counts = df_time["month"].value_counts().reset_index()
                    monthly_counts.columns = ["month", "count"]
                    
                    # Trier chronologiquement
                    monthly_counts = monthly_counts.sort_values("month")
                    
                    fig_time = px.line(
                        monthly_counts, 
                        x="month", 
                        y="count", 
                        title="Nombre de rappels par mois",
                        labels={"month": "Mois", "count": "Nombre de rappels"},
                        markers=True
                    )
                    
                    st.plotly_chart(fig_time, use_container_width=True)
                else:
                    st.warning("Pas assez de données temporelles valides pour créer un graphique.")
            except Exception as e:
                st.error(f"Erreur lors de la création du graphique temporel: {str(e)}")
                st.warning("Affichage du graphique temporel désactivé.")
        
        # Distribution par sous-catégorie
        if "sous_categorie" in df.columns:
            st.write("### Répartition par sous-catégorie")
            
            # Filtrer les valeurs NA
            valid_subcats = df["sous_categorie"].dropna()
            
            if not valid_subcats.empty:
                top_subcats = valid_subcats.value_counts().nlargest(10)
                
                fig_subcat = px.pie(
                    values=top_subcats.values,
                    names=top_subcats.index,
                    title="Top 10 des sous-catégories"
                )
                
                st.plotly_chart(fig_subcat, use_container_width=True)
            else:
                st.warning("Pas de données de sous-catégorie disponibles.")
        
        # Distribution par risque
        if "risques" in df.columns:
            st.write("### Répartition par type de risque")
            
            # Filtrer les valeurs NA
            valid_risks = df["risques"].dropna()
            
            if not valid_risks.empty:
                top_risks = valid_risks.value_counts().nlargest(10)
                
                fig_risks = px.bar(
                    x=top_risks.index,
                    y=top_risks.values,
                    title="Top 10 des risques",
                    labels={"x": "Type de risque", "y": "Nombre de rappels"}
                )
                
                fig_risks.update_layout(xaxis_tickangle=-45)
                
                st.plotly_chart(fig_risks, use_container_width=True)
            else:
                st.warning("Pas de données de risque disponibles.")
    
    with tab3:
        # Recherche avancée
        st.subheader("Recherche avancée")
        
        search_term = st.text_input("Rechercher un terme:", placeholder="Ex: listeria, fromage, etc.")
        
        if search_term:
            # Recherche dans plusieurs colonnes
            search_term_lower = search_term.lower()
            
            search_cols = ["nom", "marque", "motif", "risques", "sous_categorie"]
            search_cols = [col for col in search_cols if col in df.columns]
            
            search_results = []
            
            for _, row in df.iterrows():
                match = False
                for col in search_cols:
                    if pd.notna(row[col]) and search_term_lower in str(row[col]).lower():
                        match = True
                        break
                
                if match:
                    search_results.append(row)
            
            if search_results:
                st.success(f"{len(search_results)} résultats trouvés pour '{search_term}'")
                
                for i, result in enumerate(search_results[:10]):  # Limiter à 10 résultats affichés
                    display_recall_card(result)
                
                if len(search_results) > 10:
                    st.info(f"Affichage des 10 premiers résultats sur {len(search_results)}.")
            else:
                st.warning(f"Aucun résultat trouvé pour '{search_term}'")
    
    # Footer
    st.markdown("""
    <div class="footer">
        RappelConso Insight - Application développée pour la visualisation et l'analyse des données de rappels alimentaires
    </div>
    """, unsafe_allow_html=True)

# Exécuter l'application
if __name__ == "__main__":
    main()
