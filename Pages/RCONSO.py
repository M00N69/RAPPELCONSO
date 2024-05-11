import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# Data loading and cleaning function
@st.cache(allow_output_mutation=True)
def load_data():
    url = "https://data.economie.gouv.fr/api/records/1.0/search/?dataset=rappelconso0&q=categorie_de_produit:Alimentation&rows=10000"
    response = requests.get(url)
    data = response.json()
    records = [rec['fields'] for rec in data['records']]
    df = pd.DataFrame(records)
    df['date_de_publication'] = pd.to_datetime(df['date_de_publication'], errors='coerce')
    df['date_de_fin_de_la_procedure_de_rappel'] = pd.to_datetime(df['date_de_fin_de_la_procedure_de_rappel'], errors='coerce')
    df = df[df['date_de_publication'] >= '2021-04-01']  # Filtering out older dates
    return df

df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation et Filtres")
page = st.sidebar.selectbox("Choisir une page", ["Accueil", "Visualisation", "Détails"])

if not df.empty:
    selected_year = st.sidebar.selectbox('Sélectionner l\'année', options=sorted(df['date_de_publication'].dt.year.unique()))
    filtered_data = df[df['date_de_publication'].dt.year == selected_year]
    min_date, max_date = filtered_data['date_de_publication'].min(), filtered_data['date_de_publication'].max()
    selected_dates = st.sidebar.slider("Sélectionner la plage de dates", 
                                       min_value=min_date.to_pydatetime(), 
                                       max_value=max_date.to_pydatetime(), 
                                       value=(min_date.to_pydatetime(), max_date.to_pydatetime()))
    filtered_data = filtered_data[(filtered_data['date_de_publication'] >= selected_dates[0]) & 
                                  (filtered_data['date_de_publication'] <= selected_dates[1])]

# Accueil page
if page == "Accueil":
    st.title("Accueil - Dashboard des Rappels de Produits")
    st.write("Ce tableau de bord présente uniquement les produits de la catégorie 'Alimentation'.")
    
    # Displaying metrics and filters for Accueil page
    if not filtered_data.empty:
        selected_subcategories = st.multiselect("Sous-catégorie de produit", options=filtered_data['sous_categorie_de_produit'].unique())
        selected_risks = st.multiselect("Risques encourus par le consommateur", options=filtered_data['risques_encourus_par_le_consommateur'].unique())
        if selected_subcategories:
            filtered_data = filtered_data[filtered_data['sous_categorie_de_produit'].isin(selected_subcategories)]
        if selected_risks:
            filtered_data = filtered_data[filtered_data['risques_encourus_par_le_consommateur'].isin(selected_risks)]

        # Display active and total recalls
        active_recalls = filtered_data[filtered_data['date_de_fin_de_la_procedure_de_rappel'] > datetime.now()]
        st.metric("Nombre de rappels dans la période sélectionnée", len(filtered_data))
        st.metric("Rappels actifs", len(active_recalls))

        # Displaying last 10 recalls
        recent_recalls = filtered_data.nlargest(10, 'date_de_publication')
        for index, row in recent_recalls.iterrows():
            st.image(row['liens_vers_les_images'], caption=f"{row['date_de_publication'].strftime('%d/%m/%Y')} - {row['noms_des_modeles_ou_references']} ({row['nom_de_la_marque_du_produit']})", width=300)
            st.markdown(f"[Document]({row['lien_vers_affichette_pdf']})", unsafe_allow_html=True)

# Additional pages can be set up similarly
