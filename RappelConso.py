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
    <style>
        /* Container for each recall item */
        .recall-container {
            background-color: #f9f9f9;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 15px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
        }
        
        /* Image styling */
        .recall-image {
            width: 120px;
            height: auto;
            border-radius: 10px;
            margin-right: 20px;
        }
        
        /* Text styling within the recall container */
        .recall-content {
            flex-grow: 1;
        }

        .recall-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .recall-date {
            color: #555;
            font-size: 0.9em;
            margin-bottom: 10px;
        }

        .recall-description {
            font-size: 1em;
            color: #333;
        }
        
        /* Pagination buttons */
        .pagination-container {
            display: flex;
            justify-content: space-between;
            margin-top: 30px;
        }

        .stButton>button {
            background-color: #0044cc;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
        }

        .stButton>button:hover {
            background-color: #0033aa;
        }
    </style>
""", unsafe_allow_html=True)

# --- Constants ---
DATA_URL = "https://data.economie.gouv.fr/api/records/1.0/search/?dataset=rappelconso0&q=categorie_de_produit:Alimentation&rows=10000"

# --- Gemini Pro API Settings ---
api_key = st.secrets["api_key"]
genai.configure(api_key=api_key)

# --- Gemini Configuration ---
generation_config = genai.GenerationConfig(
    temperature=0.2,
    top_p=0.4,
    top_k=32,
    max_output_tokens=256,
)

# System Instruction
system_instruction = """Vous êtes un chatbot utile et informatif qui répond aux questions concernant les rappels de produits alimentaires en France, en utilisant la base de données RappelConso. 
Concentrez-vous sur la fourniture d'informations concernant les dates de rappel, les produits, les marques, les risques et les catégories. 
Évitez de faire des déclarations subjectives ou de donner des opinions. Basez vos réponses strictement sur les données fournies. 
Vos réponses doivent être aussi claires et précises que possible, pour éclairer les utilisateurs sur les rappels en cours ou passés."""

# --- Helper Functions ---

@st.cache_data
def load_data(url=DATA_URL):
    """Loads and preprocesses the recall data."""
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame([rec['fields'] for rec in data['records']])

    # Convert date_de_publication to datetime using pd.to_datetime()
    df['date_de_publication'] = pd.to_datetime(df['date_de_publication'], errors='coerce')

    # Handle rows with invalid dates
    df = df.dropna(subset=['date_de_publication'])  # Remove rows with invalid dates

    return df

def filter_data(df, subcategories, risks, search_term, date_range):
    """Filters the data based on user selections, date range, and search term."""
    filtered_df = df

    if subcategories:
        filtered_df = filtered_df[filtered_df['sous_categorie_de_produit'].isin(subcategories)]
    if risks:
        filtered_df = filtered_df[filtered_df['risques_encourus_par_le_consommateur'].isin(risks)]
    
    # Filter by date range
    filtered_df = filtered_df[(filtered_df['date_de_publication'] >= date_range[0]) & (filtered_df['date_de_publication'] <= date_range[1])]

    if search_term:
        filtered_df = filtered_df[filtered_df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]

    return filtered_df

def display_metrics(data):
    """Displays key metrics about the recalls."""
    st.metric("Total Recalls", len(data))

def display_recent_recalls(data, start_index=0, items_per_page=10):
    """Displays recent recalls in a visually appealing format with pagination, arranged in two columns."""
    if not data.empty:
        st.subheader("Derniers Rappels")
        recent_recalls = data.sort_values(by='date_de_publication', ascending=False)  # Sort all recalls by date
        end_index = min(start_index + items_per_page, len(recent_recalls))
        current_recalls = recent_recalls.iloc[start_index:end_index]

        # Pagination controls on a single line with buttons on the left and right
        col1, col2, col3 = st.columns([1, 2, 1])
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
                    <div class="recall-container">
                        <img src="{row['liens_vers_les_images']}" class="recall-image" alt="Product Image">
                        <div class="recall-content">
                            <div class="recall-title">{row['noms_des_modeles_ou_references']}</div>
                            <div class="recall-date">{row['date_de_publication'].strftime('%d/%m/%Y')}</div>
                            <div class="recall-description">{row['nom_de_la_marque_du_produit']}</div>
                            <a href="{row['lien_vers_affichette_pdf']}" target="_blank">Voir l'affichette</a>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.error("Aucune donnée disponible pour l'affichage des rappels.")

def display_visualizations(data):
    """Creates and displays the visualizations."""
    if not data.empty:
        # Top 5 subcategories and top 5 risks visualizations
        col1, col2 = st.columns(2)

        # Top 5 subcategories
        top_subcategories = data['sous_categorie_de_produit'].value_counts().nlargest(5)
        with col1:
            fig_subcategories = px.bar(top_subcategories, x=top_subcategories.index, y=top_subcategories.values, title="Top 5 Sous-catégories")
            st.plotly_chart(fig_subcategories, use_container_width=True)

        # Top 5 risks
        top_risks = data['risques_encourus_par_le_consommateur'].value_counts().nlargest(5)
        with col2:
            fig_risks = px.bar(top_risks, x=top_risks.index, y=top_risks.values, title="Top 5 Risques")
            st.plotly_chart(fig_risks, use_container_width=True)

        # Other visualizations
        data['month'] = pd.to_datetime(data['date_de_publication']).dt.strftime('%Y-%m')
        recalls_per_month = data.groupby('month').size().reset_index(name='counts')
        fig_monthly_recalls = px.bar(recalls_per_month,
                                     x='month', y='counts',
                                     labels={'month': 'Month', 'counts': 'Number of Recalls'},
                                     title='Number of Recalls per Month')
        st.plotly_chart(fig_monthly_recalls, use_container_width=True)
    else:
        st.error("No data available for visualizations based on the selected filters.")

def get_relevant_data_as_text(user_question, data):
    """Extracts and formats relevant data from the DataFrame as text."""
    keywords = user_question.lower().split()
    selected_rows = data[data.apply(
        lambda row: any(keyword in str(val).lower() for keyword in keywords for val in row),
        axis=1
    )].head(3)  # Limit to 3 rows

    context = "Relevant information from the RappelConso database:\n"
    for index, row in selected_rows.iterrows():
        context += f"- Date of Publication: {row['date_de_publication'].strftime('%d/%m/%Y')}\n"
        context += f"- Product Name: {row.get('noms_des_modeles_ou_references', 'N/A')}\n"
        context += f"- Brand: {row.get('nom_de_la_marque_du_produit', 'N/A')}\n"
        context += f"- Risks: {row.get('risques_encourus_par_le_consommateur', 'N/A')}\n"
        context += f"- Category: {row.get('sous_categorie_de_produit', 'N/A')}\n"
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

    # Ensure 'date_de_publication' is converted to datetime
    df['date_de_publication'] = pd.to_datetime(df['date_de_publication'], errors='coerce')

    # Extract unique values for subcategories and risks
    all_subcategories = df['sous_categorie_de_produit'].unique().tolist()
    all_risks = df['risques_encourus_par_le_consommateur'].unique().tolist()

    # --- Sidebar ---
    st.sidebar.title("Navigation and Filters")
    page = st.sidebar.selectbox("Choose a Page", ["Page principale", "Visualisation", "Details", "Chatbot"])

    with st.sidebar.expander("Advanced Filters", expanded=False):
        # Sub-category and risks filters (none selected by default)
        selected_subcategories = st.multiselect("Souscategories", options=all_subcategories, default=[])
        selected_risks = st.multiselect("Risques", options=all_risks, default=[])

        # Date filter slider
        min_date = df['date_de_publication'].min().date()  # Convert to date to avoid time inconsistencies
        max_date = df['date_de_publication'].max().date()  # Convert to date to avoid time inconsistencies

        selected_dates = st.slider("Sélectionnez la période",
                                   min_value=min_date,
                                   max_value=max_date,
                                   value=(min_date, max_date))

    # Filter data based on user selections
    st.write(f"Subcategories selected: {selected_subcategories}")
    st.write(f"Risks selected: {selected_risks}")
    st.write(f"Date range selected: {selected_dates}")
    st.write(f"Search term: {search_term}")

    filtered_data = filter_data(df, selected_subcategories, selected_risks, search_term)

    # --- Page Content ---
    if page == "Page principale":
        st.header("Principal -  Dashboard RAPPELCONSO")
        st.write("This dashboard only presents products in the 'Alimentation' category.")

        display_metrics(filtered_data)
        display_recent_recalls(filtered_data, start_index=st.session_state.start_index)

    elif page == "Visualisation":
        st.header("Product Recall Visualizations")
        st.write("This page allows you to explore different aspects of product recalls through interactive charts.")
        display_visualizations(filtered_data)

    elif page == "Details":
        st.header("Product Recall Details")
        st.write("Consult a detailed table of product recalls here, including all available information.")

        if not filtered_data.empty:
            st.dataframe(filtered_data)
            csv = filtered_data.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Filtered Data",
                               data=csv,
                               file_name='details_rappels.csv',
                               mime='text/csv')
        else:
            st.error("No data to display. Please adjust your filters or choose a different year.")

    elif page == "Chatbot":
        st.header("Ask Your Questions About Product Recalls")

        model = configure_model()  # Create the model instance

        # Store chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_area("Your Question:", height=150)
        if st.button("Send"):
            if user_input:
                with st.spinner('Gemini Pro is thinking...'):
                    try:
                        # Detect the language of the input
                        language = detect_language(user_input)

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
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()



