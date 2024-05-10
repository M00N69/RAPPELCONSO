import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# Function to load and correctly handle data from the API
@st.cache(allow_output_mutation=True)
def load_data(url):
    try:
        response = requests.get(url)
        data = response.json()
        # Properly normalize the data according to its structure
        # We need to find the right path to the list of records
        if 'records' in data:
            # Assuming each record's relevant data is nested under 'fields'
            records = [record['fields'] for record in data['records'] if 'fields' in record]
            df = pd.DataFrame(records)
            return df
        else:
            st.error("JSON structure does not contain 'records'")
            return pd.DataFrame()
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
    categories = df['categorie_de_produit'].unique().tolist()
    selected_category = st.selectbox("Choisissez une catégorie", categories)

    # Filters based on selected category
    subcategories = df[df['categorie_de_produit'] == selected_category]['sous_categorie_de_produit'].unique().tolist()
    selected_subcategory = st.selectbox("Choisissez une sous-catégorie", subcategories)

    brands = df['nom_de_la_marque_du_produit'].unique().tolist()
    selected_brand = st.multiselect("Sélectionnez la marque", brands)

    # Filtering data based on selections
    filtered_data = df[
        (df['categorie_de_produit'] == selected_category) &
        (df['sous_categorie_de_produit'] == selected_subcategory) &
        (df['nom_de_la_marque_du_produit'].isin(selected_brand))
    ]

    # Visualization - Bar chart of recalls by sub-category
    st.subheader('Nombre de Rappels par Sous-catégorie')
    fig = px.bar(filtered_data, x='sous_categorie_de_produit', title='Répartition des rappels par sous-catégorie')
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
        with st.expander(f"Rappel {idx + 1} - {row['nom_du_produit']}"):
            st.write(f"**Catégorie:** {row['categorie_de_produit']}")
            st.write(f"**Sous-catégorie:** {row['sous_categorie_de_produit']}")
            st.write(f"**Marque:** {row['nom_de_la_marque_du_produit']}")
            st.write(f"**Description:** {row.get('description', 'No description provided')}")
else:
    st.write("No data available. Please check the API or data extraction method.")
