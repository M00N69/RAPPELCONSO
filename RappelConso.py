import streamlit as st
import pandas as pd
import requests
from datetime import datetime, date, timedelta
import time

# Fonction de débogage simplifiée
def debug_log(message, data=None):
    if st.session_state.get("debug_mode", False):
        st.sidebar.markdown(f"**DEBUG:** {message}")
        if data is not None:
            st.sidebar.write(data)

# Fonction pour tester directement l'API
def test_api_connection():
    """Teste une connexion simple à l'API"""
    try:
        # URL plus simple pour le test
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

# Fonction améliorée pour charger les données
def load_rappelconso_data():
    """Charge les données de l'API RappelConso avec une gestion d'erreurs améliorée"""
    
    # Vérifier d'abord que l'API est accessible
    if not test_api_connection():
        st.error("Impossible de se connecter à l'API RappelConso. Vérifiez votre connexion internet.")
        return pd.DataFrame()
    
    try:
        # URL de base de l'API
        api_url = "https://data.economie.gouv.fr/api/v2/catalog/datasets/rappelconso-v2-gtin-espaces/records"
        
        # Dates pour le filtrage
        start_date = st.session_state.get("load_start_date", date(2022, 1, 1))
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date = date.today()
        
        # Construction de la requête (MODIFIÉE pour résoudre les problèmes)
        # 1. Utiliser des guillemets doubles pour les chaînes
        # 2. Format de date ISO sans guillemets
        # 3. Retirer l'ordre de tri pour simplifier
        query_params = {
            # Solution 1: Utiliser le format q= avec une syntaxe plus simple
            "q": "Alimentation",  # Recherche simple par terme
            
            # Date range using q filter (technique alternative)
            # "refine.date_publication": f">={start_date.strftime('%Y-%m-%d')}",
            
            # Limit and offset still needed
            "limit": 100,
            "offset": 0
        }
        
        debug_log("Requête API avec paramètres simplifiés", query_params)
        
        # Effectuer la requête
        with st.spinner("Test de récupération initiale de données..."):
            response = requests.get(api_url, params=query_params, timeout=30)
            response.raise_for_status()
            initial_data = response.json()
        
        # Vérifier si nous avons des résultats
        total_count = initial_data.get("total_count", 0)
        debug_log(f"Requête de test réussie. Total enregistrements: {total_count}")
        
        if total_count == 0:
            st.warning("Aucun rappel trouvé avec la requête de test.")
            return pd.DataFrame()
        
        # Si la requête simple fonctionne, nous pouvons récupérer plus de données
        all_records = []
        total_fetched = 0
        page_size = 100
        max_records = 5000  # Limiter le nombre total pour éviter des problèmes de mémoire
        
        # Initialiser la barre de progression
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        offset = 0
        while offset < min(total_count, max_records):
            query_params["offset"] = offset
            query_params["limit"] = page_size
            
            try:
                progress_text.text(f"Chargement des données {offset+1}-{min(offset+page_size, total_count)} sur {total_count}...")
                response = requests.get(api_url, params=query_params, timeout=30)
                response.raise_for_status()
                
                page_data = response.json()
                page_records = page_data.get("records", [])
                
                if not page_records:
                    debug_log(f"Pas de records à l'offset {offset}", None)
                    break
                
                # Extraire les données des records
                for record in page_records:
                    if "record" in record and "fields" in record["record"]:
                        all_records.append(record["record"]["fields"])
                    else:
                        debug_log(f"Structure record inattendue à l'index {len(all_records)}", record)
                
                total_fetched += len(page_records)
                progress_bar.progress(min(1.0, total_fetched / min(total_count, max_records)))
                
                offset += len(page_records)
                time.sleep(0.1)  # Pause courte pour éviter de surcharger l'API
                
                if total_fetched >= max_records:
                    debug_log(f"Limite de {max_records} enregistrements atteinte")
                    break
                
            except Exception as e:
                debug_log(f"Erreur lors du chargement de la page à l'offset {offset}", str(e))
                break
        
        # Effacer la barre de progression
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
            "date_publication": "date_publication"
        }
        
        # Appliquer le mapping des colonnes existantes
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
                if old_col != new_col:  # Éviter de supprimer une colonne qui vient d'être créée
                    df = df.drop(columns=[old_col])
        
        # Vérifier si la colonne date_publication existe
        if "date_publication" in df.columns:
            try:
                # Convertir en datetime et filtrer par date
                df["date_publication"] = pd.to_datetime(df["date_publication"], errors="coerce")
                
                # Filtrer pour la plage de dates, si nécessaire
                start_date_dt = pd.Timestamp(start_date)
                end_date_dt = pd.Timestamp(end_date)
                df = df[(df["date_publication"] >= start_date_dt) & 
                         (df["date_publication"] <= end_date_dt)]
                
                # Convertir en date (sans heure)
                df["date_publication"] = df["date_publication"].dt.date
                
                # Tri par date
                df = df.sort_values("date_publication", ascending=False)
            except Exception as date_error:
                debug_log("Erreur lors du traitement des dates", str(date_error))
        
        # Afficher des statistiques de base
        st.success(f"✅ {len(df)} rappels chargés avec succès!")
        
        return df
        
    except requests.exceptions.HTTPError as http_err:
        st.error(f"Erreur HTTP: {http_err}")
        if hasattr(http_err, 'response'):
            debug_log("Détails de l'erreur HTTP", {
                "status_code": http_err.response.status_code,
                "headers": dict(http_err.response.headers),
                "content": http_err.response.text[:500] + "..." if len(http_err.response.text) > 500 else http_err.response.text
            })
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        return pd.DataFrame()

# Interface utilisateur simplifiée pour tester
st.title("Test de chargement API RappelConso")

# Activer le mode debug
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = True

# Paramètres de chargement
st.header("🔍 Paramètres de Chargement")

# Date de début
default_date = date(2022, 1, 1)
start_date = st.date_input("Charger les rappels depuis:", value=default_date)
st.session_state.load_start_date = start_date

# Bouton pour charger les données
if st.button("🔄 Charger les données", type="primary"):
    df = load_rappelconso_data()
    
    if not df.empty:
        st.header("📊 Aperçu des données")
        st.write(f"Nombre total de rappels: {len(df)}")
        st.dataframe(df.head(10))
        
        # Afficher quelques statistiques
        if "categorie_de_produit" in df.columns:
            st.header("📈 Statistiques rapides")
            st.write("Top 5 des sous-catégories:")
            if "sous_categorie_de_produit" in df.columns:
                st.write(df["sous_categorie_de_produit"].value_counts().head(5))
    else:
        st.error("Aucune donnée n'a pu être chargée. Consultez les messages d'erreur.")

# Afficher la documentation de l'API
with st.expander("Documentation API RappelConso", expanded=False):
    st.markdown("""
    ## Informations sur l'API RappelConso
    
    L'API RappelConso est disponible via data.economie.gouv.fr et permet d'accéder aux données des rappels de produits.
    
    ### Points d'accès
    
    - **API Explorer**: [https://data.economie.gouv.fr/explore/dataset/rappelconso-v2-gtin-espaces/api/](https://data.economie.gouv.fr/explore/dataset/rappelconso-v2-gtin-espaces/api/)
    - **API Records**: [https://data.economie.gouv.fr/api/v2/catalog/datasets/rappelconso-v2-gtin-espaces/records](https://data.economie.gouv.fr/api/v2/catalog/datasets/rappelconso-v2-gtin-espaces/records)
    
    ### Filtrage des données
    
    L'API permet plusieurs méthodes de filtrage:
    
    1. **Paramètre `q`**: Recherche textuelle générale
    2. **Paramètre `where`**: Filtre avec expressions SQL
    3. **Paramètre `refine`**: Filtrage par valeur exacte
    
    ### Exemples
    
    ```
    # Recherche simple
    ?q=Alimentation
    
    # Filtre avec where
    ?where=categorie_produit="Alimentation"
    
    # Filtre avec refine
    ?refine.categorie_produit=Alimentation
    ```
    """)

# Afficher les options de débogage
if st.session_state.debug_mode:
    with st.expander("Options de débogage avancées", expanded=False):
        if st.button("🧪 Test simple connexion API"):
            test_api_connection()
        
        if st.button("❌ Effacer le cache"):
            st.cache_data.clear()
            st.success("Cache effacé!")
