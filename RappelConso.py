import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime, date
import google.generativeai as genai
import urllib.parse  # Import for URL encoding

# Configuration de la page
st.set_page_config(layout="wide")

# --- Constants ---
API_URL = "https://data.economie.gouv.fr/api/records/1.0/search/?dataset=rappelconso-v2-gtin-espaces&q=&rows=10000"
START_DATE = date(2022, 1, 1)  # Define the start date for filtering
API_PAGE_SIZE = 10000  # Define page size for API requests
API_TIMEOUT_SEC = 30  # Timeout for API requests

# --- Gemini Pro API Settings ---
try:
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
except KeyError:
    st.error("Clé API Gemini Pro manquante. Veuillez configurer la clé 'api_key' dans les secrets Streamlit.")
    genai = None

# --- Helper Functions ---

@st.cache_data(show_spinner=True)  # Show spinner during data loading
def load_data(url, start_date=START_DATE, category=None):
    """Loads and preprocesses the recall data from API with date filtering from START_DATE onwards."""
    all_records = []
    offset = 0
    fetched_count = 0  # Track fetched records to prevent infinite loop if total_count is missing/wrong

    start_date_str = start_date.strftime('%Y-%m-%d')  # Format date for API query
    today_str = date.today().strftime('%Y-%m-%d')

    # Construct base URL with date filter to load data from START_DATE to today
    base_url_with_date_filter = f"{url}&refine.date_publication:>={urllib.parse.quote(start_date_str)}&refine.date_publication:<={urllib.parse.quote(today_str)}"
    
    if category:
        base_url_with_date_filter += f"&refine.categorie_produit={urllib.parse.quote(category)}"
    
    while True:
        response = requests.get(f"{base_url_with_date_filter}&offset={offset}&rows={API_PAGE_SIZE}", timeout=API_TIMEOUT_SEC)
        if response.status_code != 200:
            st.error(f"Failed to fetch data: {response.status_code} - {response.text}")
            break
        
        data = response.json()
        records = data.get("records", [])
        if not records:
            break
        
        all_records.extend(records)
        offset += API_PAGE_SIZE
        fetched_count += len(records)
        
        # Check if we have fetched all records based on the total count
        total_count = data.get("nhits", 0)
        if fetched_count >= total_count:
            break
    
    # Process the records into a DataFrame
    processed_records = [
        {
            "date_publication": record["fields"].get("date_publication"),
            "marque_produit": record["fields"].get("marque_produit"),
            "modeles_ou_references": record["fields"].get("modeles_ou_references"),
            "motif_rappel": record["fields"].get("motif_rappel"),
            "risques_encourus": record["fields"].get("risques_encourus"),
            "sous_categorie_produit": record["fields"].get("sous_categorie_produit"),
            "nature_juridique_rappel": record["fields"].get("nature_juridique_rappel"),
            "categorie_produit": record["fields"].get("categorie_produit"),
        }
        for record in all_records
    ]
    
    df = pd.DataFrame(processed_records)
    
    # Convert date_publication to datetime if it exists
    if 'date_publication' in df.columns:
        df['date_publication'] = pd.to_datetime(df['date_publication'], errors='coerce')
    
    # Check for NaN values after conversion
    if df['date_publication'].isna().any():
        st.warning("Certaines dates de publication ne sont pas valides et ont été converties en NaN.")
    
    return df

def filter_data(data, subcategories, risks, search_term, date_range):
    """Filters the data based on the given criteria."""
    filtered_df = data.copy()
    
    if subcategories:
        filtered_df = filtered_df[filtered_df['sous_categorie_produit'].isin(subcategories)]
    
    if risks:
        filtered_df = filtered_df[filtered_df['risques_encourus'].isin(risks)]
    
    if search_term:
        filtered_df = filtered_df[filtered_df.apply(
            lambda row: any(search_term.lower() in str(val).lower() for val in row),
            axis=1
        )]
    
    if date_range:
        start_date, end_date = date_range
        # Ensure date_publication is datetime
        if 'date_publication' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['date_publication']):
            filtered_df = filtered_df[(filtered_df['date_publication'] >= start_date) & (filtered_df['date_publication'] <= end_date)]
        else:
            st.error("La colonne date_publication n'est pas au format datetime.")
    
    return filtered_df

def display_metrics(data):
    """Displays the metrics section."""
    st.subheader("Nombre Total de Rappels")
    st.metric("Total Rappels", len(data))
    st.markdown(" ", unsafe_allow_html=True)

def display_recent_recalls(data, start_index=0, items_per_page=10):
    """Displays recent recalls with pagination."""
    if not data.empty:
        total_items = len(data)
        end_index = start_index + items_per_page
        current_recalls = data.iloc[start_index:end_index]

        # Pagination controls
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.empty()  # Placeholder for symmetry
        with col2:
            if start_index > 0:
                if st.button("Précédent", key="prev"):
                    st.session_state.start_index -= items_per_page
        with col3:
            if end_index < len(data):
                if st.button("Suivant", key="next"):
                    st.session_state.start_index += items_per_page

        # Two columns for displaying recall items
        col1, col2 = st.columns(2)
        for idx, row in current_recalls.iterrows():
            with col1 if idx % 2 == 0 else col2:
                st.markdown(f"""
                **{row['modeles_ou_references'] if 'modeles_ou_references' in row else 'N/A'}**
                **Date de publication:** {row['date_publication'].strftime('%d/%m/%Y') if isinstance(row['date_publication'], date) else 'N/A'}
                **Marque:** {row['marque_produit'] if 'marque_produit' in row else 'N/A'}
                **Motif du rappel:** {row['motif_rappel'] if 'motif_rappel' in row else 'N/A'}
                [Voir l'affichette](http://example.com/{row['modeles_ou_references']})
                """, unsafe_allow_html=True)
    else:
        st.error("Aucune donnée disponible pour l'affichage des rappels.")

def display_visualizations(data):
    """Creates and displays the visualizations."""
    if not data.empty:
        value_counts = data['sous_categorie_produit'].value_counts(normalize=True) * 100
        significant_categories = value_counts[value_counts >= 2]
        filtered_categories_data = data[data['sous_categorie_produit'].isin(significant_categories.index)]

        legal_counts = data['nature_juridique_rappel'].value_counts(normalize=True) * 100
        significant_legal = legal_counts[legal_counts >= 2]
        filtered_legal_data = data[data['nature_juridique_rappel'].isin(significant_legal.index)]

        if not filtered_categories_data.empty and not filtered_legal_data.empty:
            col1, col2 = st.columns(2)

            with col1:
                fig_products = px.pie(filtered_categories_data,
                                      names='sous_categorie_produit',
                                      title='Répartition par Sous-catégories (Top)',
                                      color_discrete_sequence=px.colors.sequential.RdBu,
                                      width=600,
                                      height=400)
                st.plotly_chart(fig_products, use_container_width=True)

            with col2:
                fig_legal = px.pie(filtered_legal_data,
                                   names='nature_juridique_rappel',
                                   title='Répartition par Type de Décision (Top)',
                                   color_discrete_sequence=px.colors.sequential.RdBu,
                                   width=600,
                                   height=400)
                st.plotly_chart(fig_legal, use_container_width=True)

            # Bar chart for monthly recalls
            data['month'] = pd.to_datetime(data['date_publication']).dt.strftime('%Y-%m')
            recalls_per_month = data.groupby('month').size().reset_index(name='counts')
            fig_monthly_recalls = px.bar(recalls_per_month,
                                          x='month', y='counts',
                                          labels={'month': 'Mois', 'counts': 'Nombre de rappels'},
                                          title='Nombre de rappels par mois',
                                          width=1200, height=400)
            st.plotly_chart(fig_monthly_recalls, use_container_width=True)
        else:
            st.warning("Données insuffisantes pour afficher certains graphiques après filtrage.")  # Changed to warning
    else:
        st.error("Aucune donnée disponible pour les visualisations avec les filtres actuels.")

def display_top_charts(data):
    """Displays top 5 subcategories and risks charts."""
    st.markdown(' ', unsafe_allow_html=True)

    if data.empty:  # Early return if data is empty
        st.warning("Aucune donnée disponible pour afficher les graphiques Top 5.")
        return

    col1, col2 = st.columns(2)

    with col1:
        top_subcategories_df = data['sous_categorie_produit'].value_counts().head(5).reset_index()  # Create DataFrame
        top_subcategories_df.columns = ['Sous_categorie', 'Nombre_de_rappels']  # Rename columns for clarity
        if not top_subcategories_df.empty:
            fig_top_subcategories = px.bar(top_subcategories_df,
                                           x='Sous_categorie',  # Use column name from DataFrame
                                           y='Nombre_de_rappels',  # Use column name from DataFrame
                                           labels={'Sous_categorie': 'Sous-catégories', 'Nombre_de_rappels': 'Nombre de rappels'},
                                           title='Top 5 des Sous-catégories les plus Rappelées')
            st.plotly_chart(fig_top_subcategories, use_container_width=True)
        else:
            st.info("Pas assez de données de sous-catégories pour afficher le graphique Top 5.")

    with col2:
        top_risks_df = data['risques_encourus'].value_counts().head(5).reset_index()  # Create DataFrame
        top_risks_df.columns = ['Risque', 'Nombre_de_rappels']  # Rename columns for clarity
        if not top_risks_df.empty:
            fig_top_risks = px.bar(top_risks_df,
                                   x='Risque',  # Use column name from DataFrame
                                   y='Nombre_de_rappels',  # Use column name from DataFrame
                                   labels={'Risque': 'Risques', 'Nombre_de_rappels': 'Nombre de rappels'},
                                   title='Top 5 des Risques les plus Fréquents')
            st.plotly_chart(fig_top_risks, use_container_width=True)
        else:
            st.info("Pas assez de données de risques pour afficher le graphique Top 5.")

    st.markdown(" ", unsafe_allow_html=True)

def get_relevant_data_as_text(user_question, data):
    """Extracts and formats relevant data from the DataFrame as text."""
    keywords = user_question.lower().split()
    selected_rows = data[data.apply(
        lambda row: any(keyword in str(val).lower() for keyword in keywords for val in row),
        axis=1
    )].head(3)  # Limit to 3 rows

    context = "Informations pertinentes de la base de données RappelConso:\n"
    for index, row in selected_rows.iterrows():
        context += f"- Date de Publication: {row['date_publication'].strftime('%d/%m/%Y') if isinstance(row['date_publication'], date) else 'N/A'}\n"
        context += f"- Nom du Produit: {row.get('modeles_ou_references', 'N/A')}\n"
        context += f"- Marque: {row.get('marque_produit', 'N/A')}\n"
        context += f"- Risques: {row.get('risques_encourus', 'N/A')}\n"
        context += f"- Catégorie: {row.get('sous_categorie_produit', 'N/A')}\n"
        context += "\n"
    return context

def configure_model():
    """Creates and configures a GenerativeModel instance."""
    return genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config=generation_config,
        system_instruction=system_instruction,
    )

def detect_language(text):
    french_keywords = ["quels", "quelle", "comment", "pourquoi", "où", "qui", "quand", "le", "la", "les", "un", "une", "des"]
    if any(keyword in text.lower() for keyword in french_keywords):
        return "fr"
    return "en"

def main():
    st.title("RappelConso - Chatbot & Tableau de Bord")

    # Initialize session state for pagination
    if 'start_index' not in st.session_state:
        st.session_state.start_index = 0

    # Load data using API filtering for date, starting from 2022-01-01, and category "alimentation"
    df = load_data(API_URL, START_DATE, category="alimentation")

    if df.empty:  # Stop if initial data load fails - HANDLE EMPTY DATAFRAME HERE
        st.error("Impossible de charger les données de rappels depuis l'API. Veuillez vérifier la console pour plus de détails. L'API RappelConso est peut-être inaccessible ou ne retourne pas de données pour la période spécifiée.")
        st.stop()  # Stop further execution

    # Extract unique values for subcategories and risks
    if not df.empty:
        all_subcategories = df['sous_categorie_produit'].unique().tolist()
        all_risks = df['risques_encourus'].unique().tolist()
    else:
        all_subcategories = []
        all_risks = []

    # --- Sidebar ---
    st.sidebar.title("Navigation & Filtres")
    page = st.sidebar.selectbox("Choisir Page", ["Page principale", "Visualisation", "Details", "Chatbot"])

    with st.sidebar.expander("Filtres avancés", expanded=False):
        # Sub-category and risks filters (none selected by default)
        selected_subcategories = st.multiselect("Sous-catégories", options=all_subcategories, default=[])
        selected_risks = st.multiselect("Risques", options=all_risks, default=[])

        # Date range filter - RE-ADDED SLIDER HERE
        min_date = START_DATE  # Minimum date is fixed to START_DATE
        max_date = date.today()  # Maximum date is today
        default_dates = (START_DATE, max_date)  # Default range from START_DATE to today
        selected_dates = st.slider("Sélectionnez la période",
                                   min_value=min_date, max_value=max_date,
                                   value=default_dates)  # Set default value

    # --- Search Bar ---
    search_term = st.text_input("Recherche (Nom produit, Marque, etc.)", "")

    # --- Instructions Expander ---
    with st.expander("Instructions d'utilisation"):
        st.markdown("""
        ### Instructions d'utilisation

        - **Filtres Avancés** : Utilisez les filtres pour affiner votre recherche par sous-catégories, risques et période de temps. La période par défaut est fixée du 01/01/2022 à aujourd'hui.
        - **Nombre Total de Rappels** : Un indicateur du nombre total de rappels correspondant aux critères sélectionnés.
        - **Graphiques Top 5** : Deux graphiques affichent les 5 sous-catégories de produits les plus rappelées et les 5 principaux risques.
        - **Liste des Derniers Rappels** : Une liste paginée des rappels les plus récents, incluant le nom du produit, la date de rappel, la marque, le motif du rappel, et un lien pour voir l'affichette du rappel.
        - **Chatbot** : Posez vos questions concernant les rappels de produits et obtenez des réponses basées sur les données les plus récentes.
        - **Mettre à jour les données**: Cliquez sur le bouton pour recharger les données les plus récentes depuis la source.
        """)

    # --- Page Content ---
    filtered_data = filter_data(df, selected_subcategories, selected_risks, search_term, selected_dates)

    if page == "Page principale":
        display_metrics(filtered_data)
        display_top_charts(filtered_data)
        display_recent_recalls(filtered_data, start_index=st.session_state.start_index)

    elif page == "Visualisation":
        st.header("Visualisations des rappels de produits")
        st.write("Explorez les tendances et répartitions des rappels de produits alimentaires à travers ces graphiques interactifs.")
        display_visualizations(filtered_data)

    elif page == "Details":
        st.header("Détails des rappels de produits")
        st.write("Consultez et téléchargez un tableau détaillé des rappels de produits, filtré selon vos préférences.")

        if not filtered_data.empty:
            st.dataframe(filtered_data)
            csv = filtered_data.to_csv(index=False).encode('utf-8')
            st.download_button(label="Télécharger les données filtrées",
                               data=csv,
                               file_name='details_rappels.csv',
                               mime='text/csv')
        else:
            st.info("Aucune donnée à afficher avec les filtres sélectionnés. Ajustez vos filtres pour voir les détails.")  # Changed to info

    elif page == "Chatbot":
        st.header("Chatbot RappelConso")
        st.write("Posez vos questions sur les rappels de produits alimentaires en France.")

        model = configure_model()

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_area("Votre question:", height=150)

        if st.button("Envoyer", key="chat_button"):  # Added key to button
            if user_input.strip() == "":
                st.warning("Veuillez entrer une question valide.")
            else:
                with st.spinner('Réflexion du Chatbot...'):
                    try:
                        language = detect_language(user_input)
                        relevant_data = get_relevant_data_as_text(user_input, filtered_data)

                        context = (
                            "Informations contextuelles sur les rappels de produits filtrés :\n\n" +
                            relevant_data +
                            "\n\nQuestion de l'utilisateur : " + user_input
                        )

                        convo = model.start_chat(history=st.session_state.chat_history)
                        response = convo.send_message(context)

                        st.session_state.chat_history.append({"role": "user", "parts": [user_input]})
                        st.session_state.chat_history.append({"role": "assistant", "parts": [response.text]})

                        for message in st.session_state.chat_history:
                            role = message["role"]
                            content = message["parts"][0]
                            if role == "user":
                                st.markdown(f"**Vous :** {content}")
                            else:
                                st.markdown(f"**Assistant :** {content}")
                    except Exception as e:
                        st.error(f"Erreur du Chatbot: {e}")

    # --- Logo and Link in Sidebar ---
    st.sidebar.markdown(
        f"""
        ![Logo](https://raw.githubusercontent.com/M00N69/RAPPELCONSO/main/logo%2004%20copie.jpg)
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    
