import streamlit as st
import pandas as pd
import requests
from datetime import datetime, date, timedelta
import time
import urllib.parse

# Configuration
st.set_page_config(page_title="RappelConso API Explorer", layout="wide")

# Fonction de débogage
def debug_log(message, data=None):
    if st.session_state.get("debug_mode", False):
        st.sidebar.markdown(f"**DEBUG:** {message}")
        if data is not None:
            st.sidebar.write(data)

# Fonction pour explorer les valeurs de catégories disponibles
def explore_categories():
    """Explore les valeurs de catégories disponibles dans l'API"""
    try:
        api_url = "https://data.economie.gouv.fr/api/v2/catalog/datasets/rappelconso-v2-gtin-espaces/records"
        
        # Requête simple sans filtre
        params = {
            "limit": 1000,  # On prend un échantillon assez grand
            "select": "categorie_produit"  # On ne récupère que la colonne catégorie
        }
        
        with st.spinner("Exploration des catégories..."):
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        
        # Extraction des valeurs uniques de catégorie
        categories = set()
        records = data.get("records", [])
        
        for record in records:
            if "record" in record and "fields" in record["record"]:
                fields = record["record"]["fields"]
                if "categorie_produit" in fields:
                    categories.add(fields["categorie_produit"])
        
        return sorted(list(categories))
    
    except Exception as e:
        st.error(f"Erreur lors de l'exploration des catégories: {str(e)}")
        return []

# Fonction pour tester différentes approches de filtrage
def test_category_filters(category):
    """Teste différentes méthodes de filtrage par catégorie"""
    api_url = "https://data.economie.gouv.fr/api/v2/catalog/datasets/rappelconso-v2-gtin-espaces/records"
    
    # Liste des approches à tester
    filter_approaches = [
        # Approche 1: where avec guillemets doubles
        {"where": f'categorie_produit="{category}"', "limit": 5},
        
        # Approche 2: where avec guillemets simples
        {"where": f"categorie_produit='{category}'", "limit": 5},
        
        # Approche 3: q simple
        {"q": category, "limit": 5},
        
        # Approche 4: refine
        {"refine.categorie_produit": category, "limit": 5},
        
        # Approche 5: where avec like
        {"where": f"categorie_produit like '%{category}%'", "limit": 5},
        
        # Approche 6: where sans guillemets
        {"where": f"categorie_produit={category}", "limit": 5}
    ]
    
    results = []
    
    for approach in filter_approaches:
        try:
            response = requests.get(api_url, params=approach, timeout=10)
            
            # Même si la requête échoue, on récupère les informations
            status = "✅" if response.status_code == 200 else "❌"
            count = response.json().get("total_count", 0) if response.status_code == 200 else 0
            
            results.append({
                "approach": approach,
                "status_code": response.status_code,
                "status": status,
                "count": count,
                "url": response.url
            })
        
        except Exception as e:
            results.append({
                "approach": approach,
                "status_code": "ERROR",
                "status": "❌",
                "count": 0,
                "url": str(e)
            })
    
    return results

# Fonction pour charger les données avec la méthode qui fonctionne
def load_data_with_working_filter(category, filter_method, start_date):
    """Charge les données en utilisant la méthode de filtrage qui fonctionne"""
    api_url = "https://data.economie.gouv.fr/api/v2/catalog/datasets/rappelconso-v2-gtin-espaces/records"
    
    # Adapter les paramètres selon la méthode qui fonctionne
    if filter_method == "where_double_quotes":
        params = {"where": f'categorie_produit="{category}"', "limit": 100}
    elif filter_method == "where_single_quotes":
        params = {"where": f"categorie_produit='{category}'", "limit": 100}
    elif filter_method == "q":
        params = {"q": category, "limit": 100}
    elif filter_method == "refine":
        params = {"refine.categorie_produit": category, "limit": 100}
    elif filter_method == "where_like":
        params = {"where": f"categorie_produit like '%{category}%'", "limit": 100}
    elif filter_method == "where_no_quotes":
        params = {"where": f"categorie_produit={category}", "limit": 100}
    else:
        # Par défaut, utiliser refine qui est généralement fiable
        params = {"refine.categorie_produit": category, "limit": 100}
    
    try:
        # Requête initiale pour obtenir le nombre total
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        total_count = data.get("total_count", 0)
        if total_count == 0:
            st.warning(f"Aucun rappel trouvé pour la catégorie '{category}' avec la méthode {filter_method}.")
            return pd.DataFrame()
        
        # Récupérer toutes les données par pages
        all_records = []
        offset = 0
        page_size = 100
        max_records = 5000
        
        with st.progress(0) as progress_bar:
            while offset < min(total_count, max_records):
                params["offset"] = offset
                params["limit"] = page_size
                
                response = requests.get(api_url, params=params, timeout=30)
                response.raise_for_status()
                
                page_data = response.json()
                page_records = page_data.get("records", [])
                
                if not page_records:
                    break
                
                # Extraction des champs
                for record in page_records:
                    if "record" in record and "fields" in record["record"]:
                        all_records.append(record["record"]["fields"])
                
                offset += len(page_records)
                progress_bar.progress(min(1.0, offset / min(total_count, max_records)))
                
                if len(all_records) >= max_records:
                    break
                
                time.sleep(0.1)
        
        # Créer le dataframe
        if not all_records:
            st.warning(f"Aucune donnée extraite des réponses de l'API pour la catégorie '{category}'.")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_records)
        
        # Filtrer par date si nécessaire
        if "date_publication" in df.columns and start_date:
            # Convertir en datetime sans timezone
            df["date_publication"] = pd.to_datetime(df["date_publication"], errors="coerce")
            
            # Filtrer
            if not pd.isna(df["date_publication"]).all():
                start_dt = pd.to_datetime(start_date)
                df = df[df["date_publication"] >= start_dt]
        
        return df
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

# Interface utilisateur
st.title("🔍 RappelConso API Explorer")
st.write("Cet outil vous aide à explorer l'API RappelConso et trouver la méthode de filtrage qui fonctionne.")

# Activer le mode debug
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = True

# Section 1: Exploration des catégories disponibles
st.header("1️⃣ Explorer les catégories disponibles")

if st.button("🔎 Découvrir les catégories disponibles"):
    with st.spinner("Récupération des catégories..."):
        categories = explore_categories()
    
    if categories:
        st.success(f"✅ {len(categories)} catégories trouvées!")
        st.write("**Catégories disponibles:**")
        st.write(categories)
        
        # Sauvegarder les catégories dans la session
        st.session_state.available_categories = categories
    else:
        st.error("Aucune catégorie trouvée ou erreur lors de la récupération.")

# Section 2: Tester les méthodes de filtrage
st.header("2️⃣ Tester les méthodes de filtrage")

# Sélectionner une catégorie
category_options = st.session_state.get("available_categories", ["alimentation", "Alimentation"])
selected_category = st.selectbox("Choisir une catégorie à tester:", options=category_options)

if st.button("🧪 Tester les méthodes de filtrage"):
    with st.spinner("Test des différentes méthodes de filtrage..."):
        test_results = test_category_filters(selected_category)
    
    st.subheader("Résultats des tests")
    
    # Trouver la meilleure méthode (celle qui renvoie des résultats)
    working_methods = [r for r in test_results if r["count"] > 0]
    
    if working_methods:
        # Sauvegarder la méthode qui fonctionne
        st.session_state.working_filter_method = working_methods[0]["approach"]
        st.success(f"✅ {len(working_methods)} méthodes fonctionnent!")
    else:
        st.error("❌ Aucune méthode de filtrage n'a fonctionné!")
    
    # Afficher les résultats dans un tableau
    for i, result in enumerate(test_results):
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            st.write(f"{result['status']} Méthode {i+1}")
        
        with col2:
            st.code(str(result["approach"]))
        
        with col3:
            st.write(f"Résultats: {result['count']}")
        
        # Afficher l'URL complète en petit
        st.caption(f"URL: {result['url']}")
        st.divider()

# Section 3: Charger les données avec la méthode qui fonctionne
st.header("3️⃣ Charger les données")

col1, col2 = st.columns(2)

with col1:
    load_category = st.selectbox(
        "Catégorie à charger:",
        options=category_options,
        index=0
    )

with col2:
    start_date = st.date_input("Charger depuis:", value=date(2022, 1, 1))

# Liste des méthodes de filtrage
filter_methods = [
    "where_double_quotes",
    "where_single_quotes",
    "q",
    "refine",
    "where_like",
    "where_no_quotes"
]

# Sélectionner la méthode
selected_method = st.selectbox(
    "Méthode de filtrage:",
    options=filter_methods,
    index=filter_methods.index("refine")  # refine est souvent fiable
)

if st.button("📥 Charger les données", type="primary"):
    with st.spinner("Chargement des données en cours..."):
        df = load_data_with_working_filter(load_category, selected_method, start_date)
    
    if not df.empty:
        st.success(f"✅ {len(df)} rappels chargés avec succès!")
        
        # Afficher les données
        st.subheader("Aperçu des données")
        st.dataframe(df.head(10))
        
        # Sauvegarder dans la session
        st.session_state.loaded_data = df
        
        # Option de téléchargement
        import io
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Rappels", index=False)
        
        buffer.seek(0)
        
        st.download_button(
            label="💾 Télécharger les données (Excel)",
            data=buffer,
            file_name=f"rappelconso_{load_category}_{date.today().strftime('%Y-%m-%d')}.xlsx",
            mime="application/vnd.ms-excel"
        )
    else:
        st.error("Aucune donnée n'a pu être chargée.")
