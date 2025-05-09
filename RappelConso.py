import streamlit as st
import pandas as pd
import requests
from datetime import datetime, date, timedelta
import time

# Fonction de débogage
def debug_log(message, data=None):
    if st.session_state.get("debug_mode", False):
        st.sidebar.markdown(f"**DEBUG:** {message}")
        if data is not None:
            st.sidebar.write(data)

# Test de connexion à l'API
def test_api_connection():
    try:
        test_url = "https://data.economie.gouv.fr/api/v2/catalog/datasets/rappelconso-v2-gtin-espaces/records?limit=1"
        debug_log("Test de connexion API avec URL minimale", test_url)
        
        response = requests.get(test_url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "records" in data and len(data["records"]) > 0:
            st.sidebar.success("✅ Connexion à l'API réussie!")
            return True
        else:
            st.sidebar.warning("⚠️ Connexion API OK mais aucune donnée reçue")
            return True
    except Exception as e:
        st.sidebar.error(f"❌ Erreur de connexion API: {str(e)}")
        return False

# Fonction principale de chargement des données
def load_rappelconso_data(category_filter="Alimentation"):
    """Charge les données de l'API RappelConso pour une catégorie spécifique"""
    
    if not test_api_connection():
        st.error("Impossible de se connecter à l'API RappelConso.")
        return pd.DataFrame()
    
    try:
        # URL de base de l'API
        api_url = "https://data.economie.gouv.fr/api/v2/catalog/datasets/rappelconso-v2-gtin-espaces/records"
        
        # Dates pour le filtrage
        start_date = st.session_state.get("load_start_date", date(2022, 1, 1))
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date = date.today()
        
        # Construction de la requête avec filtre précis de catégorie
        query_params = {
            "where": f'categorie_produit="{category_filter}"',  # Filtre exact
            "limit": 100,
            "offset": 0
        }
        
        debug_log("Requête API avec filtre de catégorie", query_params)
        
        # Requête initiale pour obtenir le nombre total
        with st.spinner("Test de récupération initiale de données..."):
            response = requests.get(api_url, params=query_params, timeout=30)
            response.raise_for_status()
            initial_data = response.json()
        
        # Vérifier le nombre total
        total_count = initial_data.get("total_count", 0)
        debug_log(f"Requête de test réussie. Total enregistrements '{category_filter}': {total_count}")
        
        if total_count == 0:
            st.warning(f"Aucun rappel trouvé pour la catégorie '{category_filter}'.")
            return pd.DataFrame()
        
        # Récupérer toutes les données par pages
        all_records = []
        total_fetched = 0
        page_size = 100
        max_records = 10000  # Augmenté pour avoir plus de données
        
        # Barre de progression
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        offset = 0
        while offset < min(total_count, max_records):
            query_params["offset"] = offset
            query_params["limit"] = page_size
            
            try:
                progress_text.text(f"Chargement des rappels {offset+1}-{min(offset+page_size, total_count)} sur {total_count}...")
                response = requests.get(api_url, params=query_params, timeout=30)
                response.raise_for_status()
                
                page_data = response.json()
                page_records = page_data.get("records", [])
                
                if not page_records:
                    debug_log(f"Pas de records à l'offset {offset}", None)
                    break
                
                # Extraction des champs
                for record in page_records:
                    if "record" in record and "fields" in record["record"]:
                        all_records.append(record["record"]["fields"])
                
                total_fetched += len(page_records)
                progress_bar.progress(min(1.0, total_fetched / min(total_count, max_records)))
                
                offset += len(page_records)
                time.sleep(0.1)
                
                if total_fetched >= max_records:
                    debug_log(f"Limite de {max_records} enregistrements atteinte")
                    break
                
            except Exception as e:
                debug_log(f"Erreur lors du chargement de la page à l'offset {offset}", str(e))
                break
        
        # Nettoyer les éléments UI temporaires
        progress_text.empty()
        progress_bar.empty()
        
        # Vérifier les résultats
        if not all_records:
            st.error("Aucune donnée extraite des réponses de l'API")
            return pd.DataFrame()
        
        # Créer le dataframe
        df = pd.DataFrame(all_records)
        debug_log(f"DataFrame créé avec {len(df)} lignes", df.columns.tolist())
        
        # Normaliser les noms de colonnes
        column_mapping = {
            "categorie_produit": "categorie_de_produit",
            "sous_categorie_produit": "sous_categorie_de_produit",
            "marque_produit": "nom_de_la_marque_du_produit",
            "motif_rappel": "motif_du_rappel",
            "liens_vers_les_images": "liens_vers_images",
            "lien_vers_la_fiche_rappel": "lien_vers_la_fiche_rappel",
        }
        
        # Appliquer le mapping des colonnes existantes
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
                if old_col != new_col:
                    df = df.drop(columns=[old_col])
        
        # Gestion des dates corrigée
        if "date_publication" in df.columns:
            try:
                debug_log("Traitement des dates...", None)
                # Convertir en datetime sans timezone
                df["date_publication"] = pd.to_datetime(df["date_publication"], errors="coerce").dt.tz_localize(None)
                
                # Filtrer par date
                start_date_dt = pd.Timestamp(start_date).tz_localize(None)
                end_date_dt = pd.Timestamp(end_date).tz_localize(None)
                
                # Appliquer le filtre de date
                df = df[(df["date_publication"] >= start_date_dt) & 
                         (df["date_publication"] <= end_date_dt)]
                
                # Convertir en date (sans heure)
                df["date_publication"] = df["date_publication"].dt.date
                
                # Tri par date
                df = df.sort_values("date_publication", ascending=False)
                
                debug_log("Traitement des dates réussi", None)
            except Exception as date_error:
                debug_log("Erreur lors du traitement des dates", str(date_error))
                import traceback
                debug_log("Traceback", traceback.format_exc())
        
        # Afficher des statistiques
        st.success(f"✅ {len(df)} rappels de catégorie '{category_filter}' chargés avec succès!")
        
        # Sauvegarder dans la session state
        st.session_state.rappels_df = df
        
        return df
        
    except requests.exceptions.HTTPError as http_err:
        st.error(f"Erreur HTTP: {http_err}")
        if hasattr(http_err, 'response'):
            debug_log("Détails de l'erreur HTTP", {
                "status_code": http_err.response.status_code,
                "content": http_err.response.text[:500] + "..." if len(http_err.response.text) > 500 else http_err.response.text
            })
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        import traceback
        debug_log("Traceback", traceback.format_exc())
        return pd.DataFrame()

# Fonction pour afficher les statistiques de base
def show_basic_stats(df):
    """Affiche des statistiques de base sur les données chargées"""
    st.header("📊 Aperçu des données")
    
    # Onglets pour différentes vues
    tab1, tab2, tab3 = st.tabs(["Aperçu", "Statistiques", "Distribution"])
    
    with tab1:
        st.write(f"**Nombre total de rappels:** {len(df)}")
        st.dataframe(df.head(10), use_container_width=True)
    
    with tab2:
        st.subheader("Statistiques par catégorie")
        
        if "sous_categorie_de_produit" in df.columns:
            subcat_counts = df["sous_categorie_de_produit"].value_counts().head(10)
            st.write("**Top 10 des sous-catégories:**")
            st.bar_chart(subcat_counts)
        
        if "risques_encourus" in df.columns:
            risk_counts = df["risques_encourus"].value_counts().head(10)
            st.write("**Top 10 des risques encourus:**")
            st.bar_chart(risk_counts)
    
    with tab3:
        st.subheader("Distribution temporelle")
        
        if "date_publication" in df.columns:
            # Convertir en datetime si nécessaire
            if not pd.api.types.is_datetime64_dtype(df["date_publication"]):
                date_df = df.copy()
                date_df["date_month"] = pd.to_datetime(df["date_publication"]).dt.to_period("M")
            else:
                date_df = df.copy()
                date_df["date_month"] = df["date_publication"].dt.to_period("M")
            
            # Compter par mois
            monthly_counts = date_df["date_month"].value_counts().sort_index()
            monthly_counts.index = monthly_counts.index.astype(str)
            
            st.write("**Nombre de rappels par mois:**")
            st.line_chart(monthly_counts)

# Interface utilisateur
st.title("Analyse des Rappels Produits Alimentation")

# Activer le mode debug
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = True

# Paramètres de chargement
st.header("🔍 Paramètres")

col1, col2 = st.columns(2)

with col1:
    # Date de début
    default_date = date(2022, 1, 1)
    start_date = st.date_input("Charger les rappels depuis:", value=default_date)
    st.session_state.load_start_date = start_date

with col2:
    # Catégorie (pour l'instant fixée à Alimentation)
    category = st.selectbox(
        "Catégorie de produits:",
        options=["Alimentation"],
        index=0
    )

# Bouton pour charger les données
if st.button("🔄 Charger les données", type="primary"):
    df = load_rappelconso_data(category_filter=category)
    
    if not df.empty:
        show_basic_stats(df)
    else:
        st.error("Aucune donnée n'a pu être chargée. Consultez les messages d'erreur.")

# Afficher les données déjà chargées si elles existent
elif "rappels_df" in st.session_state:
    df = st.session_state.rappels_df
    st.info(f"Utilisation des données déjà chargées ({len(df)} rappels)")
    show_basic_stats(df)

# Boutons d'action supplémentaires
col1, col2 = st.columns(2)
with col1:
    if st.button("❌ Effacer les données"):
        if "rappels_df" in st.session_state:
            del st.session_state.rappels_df
        st.cache_data.clear()
        st.success("Données effacées et cache vidé!")
        st.rerun()

with col2:
    if "rappels_df" in st.session_state and not st.session_state.rappels_df.empty:
        # Convertir en Excel pour téléchargement
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            st.session_state.rappels_df.to_excel(writer, sheet_name="Rappels", index=False)
        buffer.seek(0)
        
        st.download_button(
            label="💾 Télécharger les données (Excel)",
            data=buffer,
            file_name=f"rappels_alimentation_{date.today().strftime('%Y-%m-%d')}.xlsx",
            mime="application/vnd.ms-excel"
        )
