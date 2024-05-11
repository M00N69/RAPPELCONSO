import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime

@st.cache(allow_output_mutation=True)
def load_data():
    url = "https://data.economie.gouv.fr/api/records/1.0/search/?dataset=rappelconso0&q=&rows=10000"
    response = requests.get(url)
    data = response.json()
    records = [rec['fields'] for rec in data['records']]
    df = pd.DataFrame(records)
    df['date_de_publication'] = pd.to_datetime(df['date_de_publication'], errors='coerce')
    df['date_de_fin_de_la_procedure_de_rappel'] = pd.to_datetime(df['date_de_fin_de_la_procedure_de_rappel'], errors='coerce')
    df = df[df['date_de_publication'] >= '2021-04-01']  # Filter out dates before April 2021
    return df

df = load_data()

# Setup sidebar for navigation and filters
st.sidebar.title("Navigation et Filtres")
page = st.sidebar.selectbox("Choisir une page", ["Accueil", "Visualisation", "Détails"])
selected_year = st.sidebar.selectbox('Sélectionner l\'année', options=sorted(df['date_de_publication'].dt.year.unique()))

# Filter data by year
filtered_data = df[df['date_de_publication'].dt.year == selected_year]
min_date, max_date = filtered_data['date_de_publication'].min(), filtered_data['date_de_publication'].max()
selected_dates = st.sidebar.slider("Sélectionner la plage de dates", 
                                   min_value=min_date.to_pydatetime(), 
                                   max_value=max_date.to_pydatetime(), 
                                   value=(min_date.to_pydatetime(), max_date.to_pydatetime()))
filtered_data = filtered_data[(filtered_data['date_de_publication'] >= selected_dates[0]) & 
                              (filtered_data['date_de_publication'] <= selected_dates[1])]

# Pages setup
if page == "Accueil":
    st.title("Accueil - Dashboard des Rappels de Produits")
    st.write("Ce tableau de bord présente uniquement les produits de la catégorie 'Alimentation'.")

    # Page-specific filters after title
    selected_subcategories = st.multiselect("Sous-catégorie de produit", options=filtered_data['sous_categorie_de_produit'].unique())
    selected_risks = st.multiselect("Risques encourus par le consommateur", options=filtered_data['risques_encourus_par_le_consommateur'].unique())
    if selected_subcategories:
        filtered_data = filtered_data[filtered_data['sous_categorie_de_produit'].isin(selected_subcategories)]
    if selected_risks:
        filtered_data = filtered_data[filtered_data['risques_encourus_par_le_consommateur'].isin(selected_risks)]

    # Display metrics
    active_recalls = filtered_data[filtered_data['date_de_fin_de_la_procedure_de_rappel'] > datetime.now()]
    st.metric("Nombre de rappels dans la période sélectionnée", len(filtered_data))
    st.metric("Rappels actifs", len(active_recalls))
    active_recalls = filtered_data[filtered_data['date_de_fin_de_la_procedure_de_rappel'] >= datetime.now()]
    st.write("Nombre de rappels actifs:", len(active_recalls))
    # Display the last 10 recalls
    df['date_de_publication'] = df['date_de_publication'].dt.strftime('%d/%m/%Y')  # Format date
    df = df.nlargest(10, 'date_de_publication')
    df['lien_vers_affichette_pdf'] = df['lien_vers_affichette_pdf'].apply(lambda x: f"[📄]({x})")
    st.dataframe(df[['liens_vers_les_images', 'date_de_publication', 'noms_des_modeles_ou_references', 'nom_de_la_marque_du_produit', 'lien_vers_affichette_pdf']])

    # Alternative for displaying images if direct embedding is problematic
    st.write("Cliquez sur les liens pour voir les images des produits rappelés:")
    for index, row in df.iterrows():
        st.markdown(f"[![Produit]({row['liens_vers_les_images']})]({row['liens_vers_les_images']})", unsafe_allow_html=True)
        
elif page == "Visualisation":
    st.title("Visualisation des Rappels de Produits")
    risques_pie = px.pie(filtered_data, names='risques_encourus_par_le_consommateur', title='Risques encourus par les consommateurs')
    nature_juridique_pie = px.pie(filtered_data, names='nature_juridique_du_rappel', title='Nature juridique du rappel')
    st.plotly_chart(risques_pie, use_container_width=True)
    st.plotly_chart(nature_juridique_pie, use_container_width=True)

elif page == "Détails":
    st.title("Détails des Rappels de Produits")
    st.dataframe(filtered_data)
    st.download_button("Télécharger les données", filtered_data.to_csv().encode('utf-8'), file_name='details_rappels.csv', mime='text/csv')

# Ensure all visualizations and UI elements handle empty data gracefully
