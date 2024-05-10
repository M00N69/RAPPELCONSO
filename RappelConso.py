import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# Function to load data
@st.cache(allow_output_mutation=True)
def load_data(url):
    try:
        response = requests.get(url)
        data = response.json()
        if 'records' in data:
            records = [record['fields'] for record in data['records'] if 'fields' in record]
            df = pd.DataFrame(records)
            return df
        else:
            st.error("JSON structure does not contain 'records'")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

url = "https://data.economie.gouv.fr/api/records/1.0/search/?dataset=rappelconso0&q=&rows=100"
df = load_data(url)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Home", "Visualization", "Details"])

if page == "Home":
    st.title("Welcome to the Rappels de Produits App")
    st.write("This application allows you to explore product recall data. Please use the sidebar to navigate between different pages.")

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
