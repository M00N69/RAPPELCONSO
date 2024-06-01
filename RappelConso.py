import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime
from dateutil.parser import parse
import google.generativeai as genai

# --- Constants ---
DATA_URL = "https://data.economie.gouv.fr/api/records/1.0/search/?dataset=rappelconso0&q=categorie_de_produit:Alimentation&rows=10000"
START_DATE = pd.Timestamp('2021-04-01')
DATE_FORMAT = '%A %d %B %Y'
RELEVANT_COLUMNS = [
    'noms_des_modeles_ou_references',
    'nom_de_la_marque_du_produit',
    'risques_encourus_par_le_consommateur',
    'sous_categorie_de_produit'
]  # Columns potentially relevant for chatbot context

# --- Gemini Pro API Settings ---
api_key = st.secrets["api_key"]
genai.configure(api_key=api_key)

# --- Gemini Configuration ---
generation_config = genai.GenerationConfig(
    temperature=0.7,
    top_p=0.4,
    top_k=32,
    max_output_tokens=256,
)

# System Instruction
system_instruction = """You are a helpful and informative chatbot that answers questions about food product recalls in France, using the RappelConso database. 
Focus on providing information about recall dates, products, brands, risks, and categories. 
Avoid making subjective statements or offering opinions. Base your responses strictly on the data provided."""

# --- Helper Functions ---

def safe_parse_date(date_str, fmt=DATE_FORMAT):
    """Parses a date string, trying a specific format first and then falling back to dateutil."""
    try:
        return pd.to_datetime(date_str, format=fmt, errors='coerce')
    except ValueError:
        return parse(date_str, dayfirst=True, yearfirst=False)

@st.cache_data
def load_data(url=DATA_URL):
    """Loads and preprocesses the recall data."""
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame([rec['fields'] for rec in data['records']])
    df['date_de_publication'] = pd.to_datetime(df['date_de_publication'], errors='coerce')
    df['date_de_fin_de_la_procedure_de_rappel'] = df['date_de_fin_de_la_procedure_de_rappel'].apply(safe_parse_date)
    df = df[df['date_de_publication'] >= START_DATE]
    return df

def filter_data(df, year, date_range, subcategories, risks, search_term):
    """Filters the data based on user selections and search term."""
    filtered_df = df[df['date_de_publication'].dt.year == year]
    filtered_df = filtered_df[(filtered_df['date_de_publication'] >= date_range[0]) & (filtered_df['date_de_publication'] <= date_range[1])]

    if subcategories:
        filtered_df = filtered_df[filtered_df['sous_categorie_de_produit'].isin(subcategories)]
    if risks:
        filtered_df = filtered_df[filtered_df['risques_encourus_par_le_consommateur'].isin(risks)]

    if search_term:
        filtered_df = filtered_df[filtered_df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]

    return filtered_df

def display_metrics(data):
    """Displays key metrics about the recalls."""
    active_recalls = data[data['date_de_fin_de_la_procedure_de_rappel'] > datetime.now()]
    st.metric("Rappels dans la période sélectionnée", len(data))
    st.metric("Rappels actifs", len(active_recalls))

def display_recent_recalls(data, num_columns=5):
    """Displays recent recalls in a grid format."""
    if not data.empty:
        st.subheader("Derniers Rappels")
        recent_recalls = data.nlargest(10, 'date_de_publication')
        num_rows = (len(recent_recalls) + num_columns - 1) // num_columns

        for i in range(num_rows):
            cols = st.columns(num_columns)
            for col, idx in zip(cols, range(i * num_columns, min((i + 1) * num_columns, len(recent_recalls)))):
                if idx < len(recent_recalls):
                    row = recent_recalls.iloc[idx]
                    col.image(row['liens_vers_les_images'],
                              caption=f"{row['date_de_publication'].strftime('%d/%m/%Y')} - {row['noms_des_modeles_ou_references']} ({row['nom_de_la_marque_du_produit']})",
                              width=120)
                    col.markdown(f"[AFFICHETTE]({row['lien_vers_affichette_pdf']})", unsafe_allow_html=True)
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
            col1, col2 = st.columns([2, 1])

            with col1:
                fig_products = px.pie(filtered_categories_data,
                                      names='sous_categorie_de_produit',
                                      title='Produits',
                                      color_discrete_sequence=px.colors.sequential.RdBu,
                                      width=800,
                                      height=600)
                st.plotly_chart(fig_products, use_container_width=False)

            with col2:
                fig_legal = px.pie(filtered_legal_data,
                                   names='nature_juridique_du_rappel',
                                   title='Nature juridique des rappels',
                                   color_discrete_sequence=px.colors.sequential.RdBu,
                                   width=800,
                                   height=600)
                st.plotly_chart(fig_legal, use_container_width=False)

            data['month'] = data['date_de_publication'].dt.strftime('%Y-%m')
            recalls_per_month = data.groupby('month').size().reset_index(name='counts')
            fig_monthly_recalls = px.bar(recalls_per_month,
                                         x='month', y='counts',
                                         labels={'month': 'Mois', 'counts': 'Nombre de rappels'},
                                         title='Nombre de rappels par mois')
            st.plotly_chart(fig_monthly_recalls, use_container_width=True)
        else:
            st.error("Données insuffisantes pour un ou plusieurs graphiques.")
    else:
        st.error("Aucune donnée disponible pour les visualisations basées sur les filtres sélectionnés.")

def get_relevant_data_as_text(user_question, data):
    """Extracts and formats relevant data from the DataFrame as text."""
    keywords = user_question.lower().split()
    selected_rows = data[data[RELEVANT_COLUMNS].apply(
        lambda row: any(keyword in str(val).lower() for keyword in keywords for val in row),
        axis=1
    )].head(3) # Limit to 3 rows

    context = "Relevant information from the RappelConso database:\n"
    for index, row in selected_rows.iterrows():
        for col in RELEVANT_COLUMNS:
            context += f"- {col}: {str(row[col])}\n" 
    context += "\n"
    return context

def configure_model():
    """Creates and configures a GenerativeModel instance."""
    return genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        system_instruction=system_instruction,
    )

def main():
    st.title("RappelConso - Chatbot & Dashboard")

    # Load data
    df = load_data()

    # --- Sidebar ---
    st.sidebar.title("Navigation et Filtres")
    page = st.sidebar.selectbox("Choisir une page", ["Accueil", "Visualisation", "Détails", "Chatbot"])

    # Year filter (set default to current year)
    current_year = datetime.now().year
    selected_year = st.sidebar.selectbox('Sélectionner l\'année', options=sorted(df['date_de_publication'].dt.year.unique()), index=len(sorted(df['date_de_publication'].dt.year.unique()))-1)

    # Date range slider
    filtered_data = df[df['date_de_publication'].dt.year == selected_year]
    min_date, max_date = filtered_data['date_de_publication'].min(), filtered_data['date_de_publication'].max()
    selected_dates = st.sidebar.slider("Sélectionner la plage de dates",
                                       min_value=min_date.to_pydatetime(),
                                       max_value=max_date.to_pydatetime(),
                                       value=(min_date.to_pydatetime(), max_date.to_pydatetime()))

    # Sub-category and risks filters (all options selected by default, but not shown)
    all_subcategories = df['sous_categorie_de_produit'].unique().tolist()
    all_risks = df['risques_encourus_par_le_consommateur'].unique().tolist()

    with st.sidebar.expander("Filtres avancés", expanded=False):
        selected_subcategories = st.multiselect("Sous-catégories", options=all_subcategories, default=all_subcategories)
        selected_risks = st.multiselect("Risques", options=all_risks, default=all_risks)

    # --- Search Bar ---
    search_term = st.text_input("Rechercher (Nom du produit, marque, etc.)", "")

    # --- Page Content ---
    filtered_data = filter_data(df, selected_year, selected_dates, selected_subcategories, selected_risks, search_term)

    if page == "Accueil":
        st.title("Accueil - Dashboard des Rappels de Produits")
        st.write("Ce tableau de bord présente uniquement les produits de la catégorie 'Alimentation'.")

        display_metrics(filtered_data)
        display_recent_recalls(filtered_data)

    elif page == "Visualisation":
        st.title("Visualisation des Rappels de Produits")
        st.write("Cette page permet d'explorer les différents aspects des rappels de produits à travers des graphiques interactifs.")
        display_visualizations(filtered_data)

    elif page == "Détails":
        st.title("Détails des Rappels de Produits")
        st.write("Consultez ici un tableau détaillé des rappels de produits, incluant toutes les informations disponibles.")

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
        st.title("Posez vos questions sur les rappels de produits")

        model = configure_model()  # Create the model instance

        # Store chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_area("Votre question:", height=150)
        if st.button("Envoyer"):
            if user_input:
                with st.spinner('Gemini Pro réfléchit...'):
                    try:
                        # Detect the language of the input
                        language = 'fr' if user_input.lower().strip().startswith(('quels', 'quelle', 'comment', 'pourquoi', 'où', 'qui', 'quand')) else 'en'

                        # Set default period to the current year
                        current_year = datetime.now().year
                        filtered_data = df[df['date_de_publication'].dt.year == current_year]

                        relevant_data = get_relevant_data_as_text(user_input, filtered_data)

                        # Start a chat session or continue the existing one
                        convo = model.start_chat(
                            history=st.session_state.chat_history
                        )

                        # Send relevant data as context in the message
                        message = relevant_data + "\n\nQuestion: " + user_input
                        response = convo.send_message(message)
                        # Update chat history
                        st.session_state.chat_history.append({"role": "user", "parts": [user_input]})
                        st.session_state.chat_history.append({"role": "assistant", "parts": [response.text]})

                        st.write(response.text)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Veuillez saisir une question.")

if __name__ == "__main__":
    main()

