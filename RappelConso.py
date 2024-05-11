import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime

# Function to load data
@st.cache(allow_output_mutation=True)
def load_data():
    url = "https://data.economie.gouv.fr/api/records/1.0/search/?dataset=rappelconso0&q=&rows=10000"
    response = requests.get(url)
    data = response.json()
    records = [rec['fields'] for rec in data['records']]
    df = pd.DataFrame(records)
    df['date_de_publication'] = pd.to_datetime(df['date_de_publication'], errors='coerce')
    df['date_de_fin_de_la_procedure_de_rappel'] = pd.to_datetime(df['date_de_fin_de_la_procedure_de_rappel'], errors='coerce')
    return df

df = load_data()

# Sidebar for navigation and filters
st.sidebar.title("Navigation et Filtres")
page = st.sidebar.selectbox("Choisir une page", ["Accueil", "Visualisation", "Détails"])
selected_year = st.sidebar.selectbox('Sélectionner l\'année', options=sorted(df['date_de_publication'].dt.year.unique()))

# Filtering by year
df = df[df['date_de_publication'].dt.year == selected_year]
min_date, max_date = df['date_de_publication'].min(), df['date_de_publication'].max()
selected_dates = st.sidebar.slider("Sélectionner la plage de dates", min_value=min_date, max_value=max_date, value=(min_date, max_date))
df = df[(df['date_de_publication'] >= selected_dates[0]) & (df['date_de_publication'] <= selected_dates[1])]

# Filter by sous_categorie_de_produit and risques_encourus_par_le_consommateur
selected_subcategories = st.sidebar.multiselect("Sous-catégorie de produit", options=df['sous_categorie_de_produit'].unique())
selected_risks = st.sidebar.multiselect("Risques encourus par le consommateur", options=df['risques_encourus_par_le_consommateur'].unique())
if selected_subcategories:
    df = df[df['sous_categorie_de_produit'].isin(selected_subcategories)]
if selected_risks:
    df = df[df['risques_encourus_par_le_consommateur'].isin(selected_risks)]

if page == "Accueil":
    st.title("Accueil")
    st.write("Bienvenue sur le tableau de bord des rappels de produits. Ce tableau ne comprend que les produits de la catégorie 'Alimentation'.")
    active_recalls = df[df['date_de_fin_de_la_procedure_de_rappel'] >= datetime.now()]
    st.metric("Nombre de rappels dans la période sélectionnée", len(df))
    st.metric("Rappels actifs", len(active_recalls))
    recent_recalls = df.nlargest(10, 'date_de_publication')[['liens_vers_les_images', 'date_de_publication', 'noms_des_modeles_ou_references', 'nom_de_la_marque_du_produit', 'lien_vers_affichette_pdf']]
    st.dataframe(recent_recalls)

elif page == "Visualisation":
    st.title("Visualisation des Rappels de Produits")
    col1, col2 = st.columns(2)
    with col1:
        pie_risks = px.pie(df, names='risques_encourus_par_le_consommateur', title='Risques encourus')
        st.plotly_chart(pie_risks)
    with col2:
        pie_legal = px.pie(df, names='nature_juridique_du_rappel', title='Nature juridique du rappel')
        st.plotly_chart(pie_legal)
    # Bar chart of recalls per month
    df['month'] = df['date_de_publication'].dt.strftime('%Y-%m')
    recall_counts = df.groupby('month').size()
    active_counts = active_recalls.groupby('month').size()
    bar_chart = px.bar(recall_counts, title="Nombre de rappels par mois")
    bar_chart.add_scatter(x=active_counts.index, y=active_counts, mode='lines', name='Rappels actifs')
    st.plotly_chart(bar_chart)

elif page == "Détails":
    st.title("Détails des Rappels de Produits")
    st.dataframe(df)
    st.download_button("Télécharger les données", df.to_csv().encode('utf-8'), file_name='details_rappels.csv', mime='text/csv')

# Ensure all visualizations and UI elements handle empty data gracefully
