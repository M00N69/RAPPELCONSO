import streamlit as st
import pandas as pd
import requests
import plotly.express as px

@st.cache(allow_output_mutation=True)
def load_all_data(base_url, page_size=1000):
    initial_response = requests.get(f"{base_url}&rows=0")
    total_records = initial_response.json()['nhits']
    
    num_pages = -(-total_records // page_size)  # Calculating the number of pages needed
    
    all_records = []
    for i in range(num_pages):
        start = i * page_size
        response = requests.get(f"{base_url}&start={start}&rows={page_size}")
        all_records.extend(response.json()['records'])
    
    # Extracting fields from records and creating DataFrame
    df = pd.DataFrame([record['fields'] for record in all_records])
    df['date_de_publication'] = pd.to_datetime(df['date_de_publication'])
    return df

# Base URL for API call
base_url = "https://data.economie.gouv.fr/api/records/1.0/search/?dataset=rappelconso0&q=categorie_de_produit:Alimentation"
df = load_all_data(base_url)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Home", "Visualization", "Details"])

if page == "Home":
    st.title("Welcome to the Rappels de Produits App")
    st.write("This application allows you to explore comprehensive product recall data across various categories. Use the sidebar to navigate through the app.")

elif page == "Visualization":
    st.title('Visualisation des Rappels de Produits')
    if not df.empty:
        categories = df['categorie_de_produit'].unique().tolist()
        selected_category = st.selectbox("Choose a category:", categories)
        
        subcategories = df[df['categorie_de_produit'] == selected_category]['sous_categorie_de_produit'].unique().tolist()
        selected_subcategory = st.selectbox("Choose a sub-category:", subcategories)
        
        brands = df['nom_de_la_marque_du_produit'].unique().tolist()
        selected_brand = st.multiselect("Select a brand:", brands)

        filtered_data = df[
            (df['categorie_de_produit'] == selected_category) &
            (df['sous_categorie_de_produit'] == selected_subcategory) &
            (df['nom_de_la_marque_du_produit'].isin(selected_brand))
        ]

        if filtered_data.empty:
            st.write("No data available for the selected filters.")
        else:
            st.subheader('Number of Recalls by Sub-category')
            fig = px.bar(filtered_data, x='sous_categorie_de_produit', title='Recalls by Sub-category')
            st.plotly_chart(fig)
    else:
        st.write("No data available. Please check the API or data extraction method.")

elif page == "Details":
    st.title('Details of the Rappels')
    if not df.empty:
        st.dataframe(df)
    else:
        st.write("No detailed data available to display.")
