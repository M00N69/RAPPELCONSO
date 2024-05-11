import streamlit as st
import pandas as pd
import plotly.express as px  # Make sure this line is added
import requests
from datetime import datetime

# Setting the page to wide mode
st.set_page_config(layout="wide", initial_sidebar_state="expanded")


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
elif page == "Visualisation":
    st.title("Visualisation des Rappels de Produits")
    st.write("Cette page permet d'explorer les différents aspects des rappels de produits à travers des graphiques interactifs.")

    # Ensure data is available
    if not filtered_data.empty:
        # Filter data for significant categories
        value_counts = filtered_data['sous_categorie_de_produit'].value_counts(normalize=True) * 100
        significant_categories = value_counts[value_counts >= 2]
        filtered_categories_data = filtered_data[filtered_data['sous_categorie_de_produit'].isin(significant_categories.index)]

        # Check if filtered_categories_data is not empty
        if not filtered_categories_data.empty:
            col1, col2 = st.columns([2, 1])  # Adjust the ratio as needed for visual balance

            with col1:
                # Pie Chart for Product Sub-Categories
                fig_products = px.pie(filtered_categories_data, names='sous_categorie_de_produit',
                                      title='Produits',
                                      color_discrete_sequence=px.colors.sequential.RdBu,
                                      width=800,  # Specify width
                                      height=600)  # Specify height
                st.plotly_chart(fig_products, use_container_width=False)
            
            # Assume filtered_legal_data is also properly defined as above for filtered_categories_data
            with col2:
                fig_legal = px.pie(filtered_legal_data, names='nature_juridique_du_rappel',
                                   title='Nature juridique des rappels',
                                   color_discrete_sequence=px.colors.sequential.RdBu,
                                   width=400,  # Specify width
                                   height=300)  # Specify height
                st.plotly_chart(fig_legal, use_container_width=False)

        else:
            st.error("No significant product categories found. Adjust your filters or data processing.")

    else:
        st.error("Aucune donnée disponible pour les visualisations basées sur les filtres sélectionnés.")

        # Bar Chart for Monthly Recalls
        filtered_data['month'] = filtered_data['date_de_publication'].dt.strftime('%Y-%m')
        recalls_per_month = filtered_data.groupby('month').size().reset_index(name='count')
        fig_monthly_recalls = px.bar(recalls_per_month, x='month', y='count',
                                     labels={'month': 'Mois', 'count': 'Nombre de rappels'},
                                     title='Nombre de rappels par mois')
        st.plotly_chart(fig_monthly_recalls, use_container_width=True)
else:
        st.error("Aucune donnée disponible pour les visualisations basées sur les filtres sélectionnés.")

elif page == "Détails":
    st.title("Détails des Rappels de Produits")
    st.write("Consultez ici un tableau détaillé des rappels de produits, incluant toutes les informations disponibles.")

    if not filtered_data.empty:
        # Displaying the full DataFrame
        st.dataframe(filtered_data)

        # Download button for the filtered dataset
        csv = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger les données filtrées",
            data=csv,
            file_name='details_rappels.csv',
            mime='text/csv'
        )
    else:
        st.error("Aucune donnée à afficher. Veuillez ajuster vos filtres ou choisir une autre année.")

