import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# Function to load and normalize data from the API
@st.cache(allow_output_mutation=True)
def load_data(url):
    try:
        response = requests.get(url)
        data = response.json()
        # Normalizing data based on the JSON structure - adapt as needed based on actual data structure
        df = pd.json_normalize(data, record_path=['records'], meta=[['fields', 'categorie_de_produit'], 
                                                                   ['fields', 'sous_categorie_de_produit'], 
                                                                   ['fields', 'nom_de_la_marque_du_produit'], 
                                                                   ['fields', 'informations_complementaires_publiques'],
                                                                   ['fields', 'description']])
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

# URL for the API endpoint
url = "https://data.economie.gouv.fr/api/records/1.0/search/?dataset=rappelconso0&q=&rows=100"

# Loading data using the defined function
df = load_data(url)

# Streamlit user interface for the app
st.title('Visualisation des Rappels de Produits')

# Filter options based on data
if not df.empty:
    categories = df['fields.categorie_de_produit'].unique().tolist()
    selected_category = st.selectbox("Choisissez une catégorie", categories)

    # Filters based on selected category
    subcategories = df[df['fields.categorie_de_produit'] == selected_category]['fields.sous_categorie_de_produit'].unique().tolist()
    selected_subcategory = st.selectbox("Choisissez une sous-catégorie", subcategories)

    brands = df['fields.nom_de_la_marque_du_produit'].unique().tolist()
    selected_brand = st.multiselect("Sélectionnez la marque", brands)

    # Filtering data based on selections
    filtered_data = df[
        (df['fields.categorie_de_produit'] == selected_category) &
        (df['fields.sous_categorie_de_produit'] == selected_subcategory) &
        (df['fields.nom_de_la_marque_du_produit'].isin(selected_brand))
    ]

    # Visualization - Bar chart of recalls by sub-category
    st.subheader('Nombre de Rappels par Sous-catégorie')
    fig = px.bar(filtered_data, x='fields.sous_categorie_de_produit', title='Répartition des rappels par sous-catégorie')
    st.plotly_chart(fig)

    # Interactive data table
    st.subheader('Détails des Rappels')
    st.dataframe(filtered_data)

    # Download button for filtered data
    st.download_button(
        "Télécharger les données filtrées",
        filtered_data.to_csv().encode('utf-8'),
        "rappels.csv",
        "text/csv",
        key='download-csv'
    )

    # Detailed information with an expander
    st.subheader('Informations Complémentaires sur les Rappels')
    for idx, row in filtered_data.iterrows():
        with st.expander(f"Rappel {idx + 1} - {row['fields.nom_du_produit']}"):
            st.write(f"**Catégorie:** {row['fields.categorie_de_produit']}")
            st.write(f"**Sous-catégorie:** {row['fields.sous_categorie_de_produit']}")
            st.write(f"**Marque:** {row['fields.nom_de_la_marque_du_produit']}")
            st.write(f"**Description:** {row['fields.description']}")
else:
    st.write("No data available.")
