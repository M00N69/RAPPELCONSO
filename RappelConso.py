import streamlit as st
import pandas as pd
import requests
from datetime import datetime

@st.cache(allow_output_mutation=True)
def load_data():
    url = "https://data.economie.gouv.fr/api/records/1.0/search/?dataset=rappelconso0&q=&rows=10000"
    response = requests.get(url)
    data = response.json()
    records = [rec['fields'] for rec in data['records']]
    df = pd.DataFrame(records)
    
    # Ensure all dates are strings or null; this might help identify data issues
    print("Sample date before conversion:", df['date_de_publication'].iloc[0])  # Debugging output

    # Convert and handle non-convertible values
    df['date_de_publication'] = pd.to_datetime(df['date_de_publication'], errors='coerce')

    # Drop rows where dates could not be converted
    df = df.dropna(subset=['date_de_publication'])

    # Filter out dates before April 2021
    df = df[df['date_de_publication'] >= '2021-04-01']

    return df

df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation et Filtres")
if not df.empty:
    try:
        years = sorted(df['date_de_publication'].dt.year.unique())
        selected_year = st.sidebar.selectbox('Sélectionner l\'année', options=years)

        # Filter data by selected year
        filtered_data = df[df['date_de_publication'].dt.year == selected_year]

        # Setup date range slider
        if not filtered_data.empty:
            min_date, max_date = filtered_data['date_de_publication'].min(), filtered_data['date_de_publication'].max()
            selected_dates = st.sidebar.slider("Sélectionner la plage de dates",
                                               min_value=min_date.to_pydatetime(), 
                                               max_value=max_date.to_pydatetime(), 
                                               value=(min_date.to_pydatetime(), max_date.to_pydatetime()))
            filtered_data = filtered_data[(filtered_data['date_de_publication'] >= selected_dates[0]) & 
                                          (filtered_data['date_de_publication'] <= selected_dates[1])]
    except Exception as e:
        st.sidebar.write("Error in processing date data:", e)

page = st.sidebar.selectbox("Choisir une page", ["Accueil", "Visualisation", "Détails"])

if page == "Accueil":
    st.title("Accueil - Dashboard des Rappels de Produits")
    st.write("Ce tableau de bord présente uniquement les produits de la catégorie 'Alimentation'.")

    # Display metrics and filters specific to the Accueil page
    active_recalls = filtered_data[filtered_data['date_de_fin_de_la_procedure_de_rappel'] > datetime.now()]
    st.metric("Nombre de rappels dans la période sélectionnée", len(filtered_data))
    st.metric("Rappels actifs", len(active_recalls))

    # Filters on the Accueil page
    selected_subcategories = st.multiselect("Sous-catégorie de produit", options=filtered_data['sous_categorie_de_produit'].unique())
    selected_risks = st.multiselect("Risques encourus par le consommateur", options=filtered_data['risques_encourus_par_le_consommateur'].unique())
    if selected_subcategories:
        filtered_data = filtered_data[filtered_data['sous_categorie_de_produit'].isin(selected_subcategories)]
    if selected_risks:
        filtered_data = filtered_data[filtered_data['risques_encourus_par_le_consommateur'].isin(selected_risks)]

    # Display the last 10 recalls
    display_data = filtered_data.nlargest(10, 'date_de_publication')
    display_data['date_de_publication'] = display_data['date_de_publication'].dt.strftime('%d/%m/%Y')  # Format date
    display_data['lien_vers_affichette_pdf'] = display_data['lien_vers_affichette_pdf'].apply(lambda x: f"[📄]({x})")
    st.dataframe(display_data[['liens_vers_les_images', 'date_de_publication', 'noms_des_modeles_ou_references', 'nom_de_la_marque_du_produit', 'lien_vers_affichette_pdf']])

    # Alternative for displaying images if direct embedding is problematic
    st.write("Cliquez sur les liens pour voir les images des produits rappelés:")
    for index, row in display_data.iterrows():
        st.image(row['liens_vers_les_images'], caption=f"Produit: {row['noms_des_modeles_ou_references']}", width=300)
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
