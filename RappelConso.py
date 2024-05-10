import streamlit as st
import pandas as pd
import plotly.express as px

# Fonction pour charger les données
@st.cache
def load_data(url):
    df = pd.read_json(url)
    return df

# Interface utilisateur
st.title('Visualisation des Rappels de Produits')

# Chargement des données
url = "https://data.economie.gouv.fr/api/records/1.0/search/?dataset=rappelconso0&q=&rows=100"
df = load_data(url)

# Création des filtres dynamiques basés sur les données
categories = df['fields_categorie_de_produit'].unique().tolist()
selected_category = st.selectbox("Choisissez une catégorie", categories)

# Filtres avancés basés sur la catégorie sélectionnée
subcategories = df[df['fields_categorie_de_produit'] == selected_category]['fields_sous_categorie_de_produit'].unique().tolist()
selected_subcategory = st.selectbox("Choisissez une sous-catégorie", subcategories)

brands = df['fields_nom_de_la_marque_du_produit'].unique().tolist()
selected_brand = st.multiselect("Sélectionnez la marque", brands)

# Filtrage des données en fonction des sélections
filtered_data = df[
    (df['fields_categorie_de_produit'] == selected_category) &
    (df['fields_sous_categorie_de_produit'] == selected_subcategory) &
    (df['fields_nom_de_la_marque_du_produit'].isin(selected_brand))
]

# Visualisation - Graphiques de répartition
st.subheader('Nombre de Rappels par Sous-catégorie')
fig = px.bar(filtered_data, x='fields_sous_categorie_de_produit', title='Répartition des rappels par sous-catégorie')
st.plotly_chart(fig)

# Tableau de données interactif
st.subheader('Détails des Rappels')
st.dataframe(filtered_data)

# Téléchargement des données
st.download_button(
    "Télécharger les données filtrées",
    filtered_data.to_csv().encode('utf-8'),
    "rappels.csv",
    "text/csv",
    key='download-csv'
)

# Informations détaillées avec expander
st.subheader('Informations Complémentaires sur les Rappels')
for idx, row in filtered_data.iterrows():
    with st.expander(f"Rappel {idx + 1} - {row['fields_nom_du_produit']}"):
        st.write(f"**Catégorie:** {row['fields_categorie_de_produit']}")
        st.write(f"**Sous-catégorie:** {row['fields_sous_categorie_de_produit']}")
        st.write(f"**Marque:** {row['fields_nom_de_la_marque_du_produit']}")
        st.write(f"**Description:** {row['fields_informations_complementaires_publiques']}")
