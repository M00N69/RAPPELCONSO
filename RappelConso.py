import streamlit as st
import pandas as pd
import requests
import plotly.express as px

@st.cache(allow_output_mutation=True)
def load_all_data(base_url, page_size=1000):
    initial_response = requests.get(f"{base_url}&rows=0")
    total_records = initial_response.json()['nhits']
    num_pages = -(-total_records // page_size)
    
    all_records = []
    for i in range(num_pages):
        start = i * page_size
        response = requests.get(f"{base_url}&start={start}&rows={page_size}")
        all_records.extend(response.json()['records'])
    
    df = pd.DataFrame([record['fields'] for record in all_records])
    df['date_de_publication'] = pd.to_datetime(df['date_de_publication'])
    return df

base_url = "https://data.economie.gouv.fr/api/records/1.0/search/?dataset=rappelconso0&q=categorie_de_produit:Alimentation"
df = load_all_data(base_url)

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Home", "Visualization", "Details"])

if page == "Visualization":
    st.title('Visualisation des Rappels de Produits')

    if not df.empty:
        categories = df['categorie_de_produit'].unique().tolist()
        selected_category = st.selectbox("Choose a category:", categories)
        
        subcategories = df[df['categorie_de_produit'] == selected_category]['sous_categorie_de_produit'].unique().tolist()
        selected_subcategory = st.selectbox("Choose a sub-category:", subcategories)

        # Date filter
        start_date, end_date = st.select_slider(
            "Select a date range:",
            options=pd.to_datetime(df['date_de_publication']).sort_values().unique(),
            value=(pd.to_datetime(df['date_de_publication']).min(), pd.to_datetime(df['date_de_publication']).max())
        )
        
        filtered_data = df[
            (df['categorie_de_produit'] == selected_category) &
            (df['sous_categorie_de_produit'] == selected_subcategory) &
            (df['date_de_publication'] >= start_date) &
            (df['date_de_publication'] <= end_date)
        ]

        if filtered_data.empty:
            st.write("No data available for the selected filters.")
        else:
            # Pie chart for Risks
            risk_fig = px.pie(filtered_data, names='risques_encourus_par_le_consommateur', title='Risks Incurred by Consumers')
            st.plotly_chart(risk_fig)

            # Pie chart for Legal Nature
            legal_fig = px.pie(filtered_data, names='nature_juridique_du_rappel', title='Legal Nature of Recall')
            st.plotly_chart(legal_fig)
    else:
        st.write("No data available. Please check the API or data extraction method.")
