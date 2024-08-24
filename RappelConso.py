import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime
import google.generativeai as genai

# Custom CSS for styling
st.markdown("""
    <style>
        .main { 
            font-family: "Arial", sans-serif; 
            color: #ffffff;  /* Brighter text color */
        }
        h1, h2, h3, h4, h5, h6 {
            color: #1E90FF; /* Bright blue color for headers */
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
        .stTextInput>div>div>input {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            color: #ffffff; /* Brighter text color for input */
        }
        .stTextInput>div>div>input:focus {
            border-color: #0044cc;
        }
        .stMetric {
            color: #ffffff;  /* Brighter text color for metrics */
        }
        .stMetricLabel {
            color: #ffffff;  /* Brighter text color for metric labels */
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

@st.cache_data
def load_data(url=DATA_URL):
    """Loads and preprocesses the recall data."""
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame([rec['fields'] for rec in data['records']])

    # Convert date_de_publication to datetime using pd.to_datetime()
    df['date_de_publication'] = pd.to_datetime(df['date_de_publication'], errors='coerce')

    # Handle rows with invalid dates
    st.write("Rows with invalid 'date_de_publication':", df[df['date_de_publication'].isna()])
    df = df.dropna(subset=['date_de_publication'])  # Remove rows with invalid dates

    return df

def filter_data(df, subcategories, risks, search_term):
    """Filters the data based on user selections and search term."""
    filtered_df = df

    if subcategories:
        filtered_df = filtered_df[filtered_df['sous_categorie_de_produit'].isin(subcategories)]
    if risks:
        filtered_df = filtered_df[filtered_df['risques_encourus_par_le_consommateur'].isin(risks)]

    if search_term:
        filtered_df = filtered_df[filtered_df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]

    return filtered_df

def display_metrics(data):
    """Displays key metrics about the recalls."""
    st.metric("Total Recalls", len(data))

def display_recent_recalls(data, start_index=0, num_columns=5, items_per_page=10):
    """Displays recent recalls in a grid format with pagination."""
    if not data.empty:
        st.subheader("Latest Recalls")
        recent_recalls = data.nlargest(100, 'date_de_publication')  # Get the 100 most recent recalls
        num_items = len(recent_recalls)
        num_rows = (items_per_page + num_columns - 1) // num_columns

        end_index = min(start_index + items_per_page, num_items)
        current_recalls = recent_recalls.iloc[start_index:end_index]

        for i in range(num_rows):
            cols = st.columns(num_columns)
            for col, idx in zip(cols, range(i * num_columns, min((i + 1) * num_columns, len(current_recalls)))):
                if idx < len(current_recalls):
                    row = current_recalls.iloc[idx]
                    col.image(row['liens_vers_les_images'],
                              caption=f"{row['date_de_publication'].strftime('%d/%m/%Y')} - {row['noms_des_modeles_ou_references']} ({row['nom_de_la_marque_du_produit']})",
                              width=120)
                    col.markdown(f"[AFFICHETTE]({row['lien_vers_affichette_pdf']})", unsafe_allow_html=True)

        # Pagination controls
        if start_index > 0:
            if st.button("Previous"):
                st.session_state.start_index -= items_per_page

        if end_index < num_items:
            if st.button("Next"):
                st.session_state.start_index += items_per_page

    else:
        st.error("No data available for displaying recalls.")

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
                                      title='Products',
                                      color_discrete_sequence=px.colors.sequential.RdBu,
                                      width=800,
                                      height=600)
                st.plotly_chart(fig_products, use_container_width=False)

            with col2:
                fig_legal = px.pie(filtered_legal_data,
                                   names='nature_juridique_du_rappel',
                                   title='Legal Nature of Recalls',
                                   color_discrete_sequence=px.colors.sequential.RdBu,
                                   width=800,
                                   height=600)
                st.plotly_chart(fig_legal, use_container_width=False)

            data['month'] = pd.to_datetime(data['date_de_publication']).dt.strftime('%Y-%m')
            recalls_per_month = data.groupby('month').size().reset_index(name='counts')
            fig_monthly_recalls = px.bar(recalls_per_month,
                                         x='month', y='counts',
                                         labels={'month': 'Month', 'counts': 'Number of Recalls'},
                                         title='Number of Recalls per Month')
            st.plotly_chart(fig_monthly_recalls, use_container_width=True)
        else:
            st.error("Insufficient data for one or more charts.")
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

    # --- Sidebar ---
    st.sidebar.title("Navigation and Filters")
    page = st.sidebar.selectbox("Choose a Page", ["Home", "Visualization", "Details", "Chatbot"])

    # Sub-category and risks filters (all options selected by default, but not shown)
    all_subcategories = df['sous_categorie_de_produit'].unique().tolist()
    all_risks = df['risques_encourus_par_le_consommateur'].unique().tolist()

    with st.sidebar.expander("Advanced Filters", expanded=False):
        selected_subcategories = st.multiselect("Subcategories", options=all_subcategories, default=all_subcategories)
        selected_risks = st.multiselect("Risks", options=all_risks, default=all_risks)

    # --- Search Bar ---
    search_term = st.text_input("Search (Product Name, Brand, etc.)", "")

    # --- Page Content ---
    filtered_data = filter_data(df, selected_subcategories, selected_risks, search_term)

    if page == "Home":
        st.header("Home - Product Recall Dashboard")
        st.write("This dashboard only presents products in the 'Alimentation' category.")

        display_metrics(filtered_data)
        display_recent_recalls(filtered_data, start_index=st.session_state.start_index)

    elif page == "Visualization":
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
