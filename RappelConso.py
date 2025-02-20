import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime
import google.generativeai as genai

# Configuration de la page 
st.set_page_config(layout="wide")

# Custom CSS for styling
st.markdown("""
     
""", unsafe_allow_html=True)

# --- Constants ---
DATASET_ID = "rappelconso0"
EXPORT_URL = f"https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/{DATASET_ID}/exports/csv"

# --- Gemini Pro API Settings ---
api_key = st.secrets["api_key"]
genai.configure(api_key=api_key)

generation_config = genai.GenerationConfig(
    temperature=0.2,
    top_p=0.4,
    top_k=32,
    max_output_tokens=256,
)

system_instruction = """Vous êtes un chatbot utile et informatif qui répond aux questions concernant les rappels de produits alimentaires en France, en utilisant la base de données RappelConso.
Concentrez-vous sur la fourniture d'informations concernant les dates de rappel, les produits, les marques, les risques et les catégories.
Évitez de faire des déclarations subjectives ou de donner des opinions. Basez vos réponses strictement sur les données fournies.
Vos réponses doivent être aussi claires et précises que possible, pour éclairer les utilisateurs sur les rappels en cours ou passés."""

# --- Helper Functions ---

@st.cache_data
def load_data():
    """Loads and preprocesses the recall data using the export endpoint."""
    response = requests.get(EXPORT_URL)
    if response.status_code == 200:
        try:
            export_link = response.json().get('url')
            if export_link:
                csv_response = requests.get(export_link)
                csv_data = csv_response.content.decode('utf-8-sig')  # Use utf-8-sig to handle BOM
                df = pd.read_csv(pd.compat.StringIO(csv_data))
                df['date_de_publication'] = pd.to_datetime(df['date_de_publication'], errors='coerce').dt.date
                df = df.dropna(subset=['date_de_publication'])
                return df
            else:
                st.error("Le lien d'exportation n'a pas été trouvé dans la réponse.")
        except ValueError as e:
            st.error(f"Erreur lors du traitement de la réponse JSON : {e}")
        except Exception as e:
            st.error(f"Erreur inattendue lors du traitement de la réponse : {e}")
    else:
        st.error(f"Erreur lors de l'exportation du dataset : {response.status_code}")
    return None

def filter_data(df, subcategories, risks, search_term, date_range):
    """Filters the data based on user selections and search term."""
    start_date, end_date = date_range

    # Filter by date range
    filtered_df = df[(df['date_de_publication'] >= start_date) & (df['date_de_publication'] <= end_date)]

    # Filter by subcategories
    if subcategories:
        filtered_df = filtered_df[filtered_df['sous_categorie_de_produit'].isin(subcategories)]

    # Filter by risks
    if risks:
        filtered_df = filtered_df[filtered_df['risques_encourus_par_le_consommateur'].isin(risks)]

    # Filter by search term
    if search_term:
        filtered_df = filtered_df[filtered_df.apply(
            lambda row: any(search_term.lower() in str(val).lower() for val in row),
            axis=1
        )]

    return filtered_df

def display_metrics(data):
    """Displays the total number of recalls."""
    st.markdown("## Métriques")
    st.metric("Nombre Total de Rappels", len(data))

def display_recent_recalls(data, start_index):
    """Displays the recent recalls with pagination."""
    items_per_page = 10
    recent_recalls = data.sort_values(by='date_de_publication', ascending=False)
    end_index = start_index + items_per_page
    current_recalls = recent_recalls.iloc[start_index:end_index]

    if not current_recalls.empty:
        st.markdown("## Rappels Récents")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if start_index > 0:
                if st.button("Précédent", key="prev"):
                    st.session_state.start_index -= items_per_page
        with col3:
            if end_index < len(recent_recalls):
                if st.button("Suivant", key="next"):
                    st.session_state.start_index += items_per_page

        # Create two columns for displaying recall items
        col1, col2 = st.columns(2)
        for idx, row in current_recalls.iterrows():
            with col1 if idx % 2 == 0 else col2:
                st.markdown(f"""
                **{row['noms_des_modeles_ou_references']}**  
                **Date de Publication:** {row['date_de_publication'].strftime('%d/%m/%Y')}  
                **Marque:** {row['nom_de_la_marque_du_produit']}  
                **Motif du Rappel:** {row['motif_du_rappel']}  
                [Voir l'affichette]({row['url_affichette']})
                """, unsafe_allow_html=True)
    else:
        st.error("Aucune donnée disponible pour l'affichage des rappels.")

def display_visualizations(data):
    """Creates and displays the visualizations."""
    if not data.empty:
        value_counts = data['sous_categorie_de_produit'].value_counts(normalize=True) * 100
        significant_categories = value_counts[value_counts >= 2]
        filtered_categories_data = data[data['sous_categorie_de_produit'].isin(significant_categories.index)]

        legal_counts = data['nature_juridique_du_rappel'].value_counts(normalize=True) * 100
        significant_legal = legal_counts[legal_counts >= 2]
        filtered_legal_data = data[data['nature_juridique_du_rappel'].isin(significant_legal.index)]

        if not filtered_categories_data.empty and not filtered_legal_data.empty:
            col1, col2 = st.columns(2)

            with col1:
                fig_products = px.pie(filtered_categories_data,
                                      names='sous_categorie_de_produit',
                                      title='Sous-catégories',
                                      color_discrete_sequence=px.colors.sequential.RdBu,
                                      width=600,
                                      height=400)
                st.plotly_chart(fig_products, use_container_width=True)

            with col2:
                fig_legal = px.pie(filtered_legal_data,
                                   names='nature_juridique_du_rappel',
                                   title='Décision de rappel',
                                   color_discrete_sequence=px.colors.sequential.RdBu,
                                   width=600,
                                   height=400)
                st.plotly_chart(fig_legal, use_container_width=True)

            # Add a bar chart showing the number of recalls per month
            data['month'] = pd.to_datetime(data['date_de_publication']).dt.strftime('%Y-%m')
            recalls_per_month = data.groupby('month').size().reset_index(name='counts')
            fig_monthly_recalls = px.bar(recalls_per_month,
                                         x='month', y='counts',
                                         labels={'month': 'Mois', 'counts': 'Nombre de rappels'},
                                         title='Nombre de rappels par mois',
                                         width=1200, height=400)
            st.plotly_chart(fig_monthly_recalls, use_container_width=True)
        else:
            st.error("Données insuffisantes pour un ou plusieurs graphiques.")
    else:
        st.error("Aucune donnée disponible pour les visualisations basées sur les filtres sélectionnés.")

def display_top_charts(data):
    """Displays top 5 subcategories and risks charts."""
    st.markdown(' ', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        top_subcategories = data['sous_categorie_de_produit'].value_counts().head(5)
        fig_top_subcategories = px.bar(x=top_subcategories.index,
                                       y=top_subcategories.values,
                                       labels={'x': 'Sous-catégories', 'y': 'Nombre de rappels'},
                                       title='Top 5 des sous-catégories')
        st.plotly_chart(fig_top_subcategories, use_container_width=True)

    with col2:
        top_risks = data['risques_encourus_par_le_consommateur'].value_counts().head(5)
        fig_top_risks = px.bar(x=top_risks.index,
                              y=top_risks.values,
                              labels={'x': 'Risques', 'y': 'Nombre de rappels'},
                              title='Top 5 des risques')
        st.plotly_chart(fig_top_risks, use_container_width=True)

    st.markdown(" ", unsafe_allow_html=True)

def get_relevant_data_as_text(user_question, data):
    """Extracts and formats relevant data from the DataFrame as text."""
    keywords = user_question.lower().split()
    selected_rows = data[data.apply(
        lambda row: any(keyword in str(val).lower() for keyword in keywords for val in row),
        axis=1
    )].head(3)  # Limit to 3 rows

    context = "Informations pertinentes à partir de la base de données RappelConso:\n"
    for index, row in selected_rows.iterrows():
        context += f"- Date de Publication: {row['date_de_publication'].strftime('%d/%m/%Y')}\n"
        context += f"- Nom du Produit: {row.get('noms_des_modeles_ou_references', 'N/A')}\n"
        context += f"- Marque: {row.get('nom_de_la_marque_du_produit', 'N/A')}\n"
        context += f"- Risques: {row.get('risques_encourus_par_le_consommateur', 'N/A')}\n"
        context += f"- Catégorie: {row.get('sous_categorie_de_produit', 'N/A')}\n"
        context += "\n"
    return context

def configure_model():
    """Creates and configures a GenerativeModel instance."""
    return genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        system_instruction=system_instruction,
    )

def detect_language(text):
    french_keywords = ["quels", "quelle", "comment", "pourquoi", "où", "qui", "quand", "le", "la", "les", "un", "une", "des"]
    if any(keyword in text.lower() for keyword in french_keywords):
        return "fr"
    return "en"

def main():
    st.title("RappelConso - Chatbot & Dashboard")

    # Initialize session state for pagination
    if 'start_index' not in st.session_state:
        st.session_state.start_index = 0

    # Load data
    df = load_data()

    # Check if data was loaded successfully
    if df is not None:
        # Extract unique values for subcategories and risks
        all_subcategories = df['sous_categorie_de_produit'].unique().tolist()
        all_risks = df['risques_encourus_par_le_consommateur'].unique().tolist()

        # --- Sidebar ---
        st.sidebar.title("Navigation & Filtres")
        page = st.sidebar.selectbox("Choisir Page", ["Page principale", "Visualisation", "Détails", "Chatbot"])

        with st.sidebar.expander("Filtres avancés", expanded=False):
            # Sub-category and risks filters (none selected by default)
            selected_subcategories = st.multiselect("Sous-catégories", options=all_subcategories, default=[])
            selected_risks = st.multiselect("Risques", options=all_risks, default=[])

            # Date range filter
            min_date = df['date_de_publication'].min()
            max_date = df['date_de_publication'].max()
            selected_dates = st.slider("Sélectionnez la période",
                                       min_value=min_date, max_value=max_date,
                                       value=(min_date, max_date))

        # --- Search Bar ---
        search_term = st.text_input("Recherche (Nom produit, Marque, etc.)", "")

        # --- Instructions Expander ---
        with st.expander("Instructions d'utilisation"):
            st.markdown("""
            ### Instructions d'utilisation

            - **Filtres Avancés** : Utilisez les filtres pour affiner votre recherche par sous-catégories, risques et périodes de temps.
            - **Nombre Total de Rappels** : Un indicateur du nombre total de rappels correspondant aux critères sélectionnés.
            - **Graphiques Top 5** : Deux graphiques affichent les 5 sous-catégories de produits les plus rappelées et les 5 principaux risques.
            - **Liste des Derniers Rappels** : Une liste paginée des rappels les plus récents, incluant le nom du produit, la date de rappel, la marque, le motif du rappel, et un lien pour voir l'affichette du rappel.
            - **Chatbot** : Posez vos questions concernant les rappels de produits et obtenez des réponses basées sur les données les plus récentes.
            """)

        # --- Page Content ---
        filtered_data = filter_data(df, selected_subcategories, selected_risks, search_term, selected_dates)

        if page == "Page principale":
            display_metrics(filtered_data)
            display_top_charts(filtered_data)  # Display top 5 charts for categories and risks
            display_recent_recalls(filtered_data, start_index=st.session_state.start_index)

        elif page == "Visualisation":
            st.header("Visualisations des rappels de produits")
            st.write("Cette page vous permet d'explorer différents aspects des rappels de produits à travers des graphiques interactifs.")
            display_visualizations(filtered_data)

        elif page == "Détails":
            st.header("Détails des rappels de produits")
            st.write("Consultez un tableau détaillé des rappels de produits ici, incluant toutes les informations disponibles.")

            if not filtered_data.empty:
                st.dataframe(filtered_data)
                csv = filtered_data.to_csv(index=False).encode('utf-8')
                st.download_button(label="Télécharger les données filtrées",
                                     data=csv,
                                     file_name='details_rappels.csv',
                                     mime='text/csv')
            else:
                st.error("Aucune donnée à afficher. Veuillez ajuster vos filtres ou choisir une autre année.")

        elif page == "Chatbot":
            st.header("Posez vos questions sur les rappels de produits")

            model = configure_model()  # Créez l'instance du modèle

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            user_input = st.text_area("Votre question:", height=150)

            if st.button("Envoyer"):
                if user_input.strip() == "":
                    st.warning("Veuillez entrer une question valide.")
                else:
                    with st.spinner('Gemini Pro réfléchit...'):
                        try:
                            # Détecter la langue de l'entrée utilisateur
                            language = detect_language(user_input)

                            # Extraire les données pertinentes des rappels filtrés
                            relevant_data = get_relevant_data_as_text(user_input, filtered_data)

                            # Créer un contexte structuré pour le modèle
                            context = (
                                "Informations sur les rappels filtrés :\n\n" +
                                relevant_data +
                                "\n\nQuestion de l'utilisateur : " + user_input
                            )

                            # Démarrer une session de chat ou continuer la session existante
                            convo = model.start_chat(history=st.session_state.chat_history)

                            # Envoyer le contexte structuré et la question
                            response = convo.send_message(context)

                            # Mettre à jour l'historique du chat
                            st.session_state.chat_history.append({"role": "user", "parts": [user_input]})
                            st.session_state.chat_history.append({"role": "assistant", "parts": [response.text]})

                            # Afficher l'historique du chat avec une mise en forme améliorée
                            for message in st.session_state.chat_history:
                                role = message["role"]
                                content = message["parts"][0]
                                if role == "user":
                                    st.markdown(f"**Vous :** {content}")
                                else:
                                    st.markdown(f"**Assistant :** {content}")
                        except Exception as e:
                            st.error(f"Une erreur s'est produite: {e}")

        # --- Logo and Link in Sidebar ---
        st.sidebar.markdown(
            f"""
            <img src="https://raw.githubusercontent.com/M00N69/RAPPELCONSO/main/logo%2004%20copie.jpg" alt="Logo" style="width:200px;">
            """, unsafe_allow_html=True
        )
    else:
        st.error("Erreur lors du chargement des données. Veuillez réessayer.")

if __name__ == "__main__":
    main()
