import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, date, timedelta
import google.generativeai as genai
import urllib.parse
import time

# Configuration de la page avec favicon et titre personnalisé
st.set_page_config(
    page_title="RappelConso - Sécurité Alimentaire",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS amélioré avec animations, thème cohérent et responsive design
st.markdown("""
    <style>
        /* (CSS omitted for brevity) */
    </style>
""", unsafe_allow_html=True)

# --- Constants ---
API_URL = "https://data.economie.gouv.fr/api/records/1.0/search/?dataset=rappelconso-v2-gtin-espaces&q=&rows=10000"
START_DATE = date(2022, 1, 1)
API_PAGE_SIZE = 10000
API_TIMEOUT_SEC = 30

# Mapping des colonnes
COLUMN_MAPPING = {
    "motif": "motif_rappel",
    "description complémentaire": "description_complementaire_risque",
    "risque": "risques_encourus"
}

# --- Fonctions de chargement de données ---
@st.cache_data(show_spinner=True)
def load_data(url, start_date=START_DATE):
    """Loads and preprocesses the recall data from API with date filtering from START_DATE onwards."""
    all_records = []

    start_date_str = start_date.strftime('%Y-%m-%d')
    today_str = date.today().strftime('%Y-%m-%d')

    # Construct base URL with date filter
    base_url_with_date_filter = f"{url}&refine.date_publication:>={urllib.parse.quote(start_date_str)}&refine.date_publication:<={urllib.parse.quote(today_str)}&refine.categorie_de_produit=Alimentation"

    with st.spinner("Chargement des données (depuis 2022)..."):
        request_url = base_url_with_date_filter
        try:
            response = requests.get(request_url, timeout=API_TIMEOUT_SEC)
            response.raise_for_status()
            data = response.json()
            records = data.get('records')

            if records:
                all_records.extend([rec['fields'] for rec in records])
                print(f"Fetched {len(records)} records.")
            else:
                print("No records from API.")

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur de requête API: {e}")
            print(f"API Request Error: {e}")
            return pd.DataFrame()
        except KeyError as e:
            st.error(f"Erreur de structure JSON de l'API: clé manquante {e}")
            print(f"JSON Structure Error: Missing key {e}")
            return pd.DataFrame()
        except requests.exceptions.Timeout:
            st.error(f"Délai d'attente dépassé lors de la requête à l'API.")
            print("API Timeout Error")
            return pd.DataFrame()

    if not all_records:
        print("No records loaded in total.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    print(f"DataFrame created with {len(df)} rows.")

    # Convert date_de_publication to datetime objects
    df['date_publication'] = pd.to_datetime(df['date_publication'], errors='coerce').dt.date

    df = df.dropna(subset=['date_publication'])
    df = df.sort_values(by='date_publication', ascending=False)

    return df

def filter_data(data, selected_subcategories, selected_risks, search_term, selected_dates, selected_categories, search_column=None):
    """Filters the DataFrame based on the given criteria."""
    filtered_df = data.copy()

    # Filter by subcategories
    if selected_subcategories:
        filtered_df = filtered_df[filtered_df['sous_categorie_produit'].isin(selected_subcategories)]

    # Filter by risks
    if selected_risks:
        filtered_df = filtered_df[filtered_df['risques_encourus'].isin(selected_risks)]

    # Filter by categories
    if selected_categories:
        filtered_df = filtered_df[filtered_df['categorie_produit'].isin(selected_categories)]

    # Filter by search term in a specific column
    if search_term:
        if search_column and search_column != "Toutes les colonnes":
            # Utiliser le mapping pour obtenir le nom réel de la colonne
            real_column_name = COLUMN_MAPPING.get(search_column)

            if real_column_name and real_column_name in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[real_column_name].astype(str).str.contains(search_term, case=False, na=False)]
            else:
                st.warning(f"La recherche a été effectuée dans toutes les colonnes.")
                filtered_df = filtered_df[filtered_df.apply(
                    lambda row: any(search_term.lower() in str(val).lower() for val in row),
                    axis=1
                )]
        else:
            # Recherche dans toutes les colonnes
            filtered_df = filtered_df[filtered_df.apply(
                lambda row: any(search_term.lower() in str(val).lower() for val in row),
                axis=1
            )]

    # Filter by date range
    filtered_df = filtered_df[(filtered_df['date_publication'] >= selected_dates[0]) & (filtered_df['date_publication'] <= selected_dates[1])]

    return filtered_df

def clear_cache():
    st.cache_data.clear()

# --- Fonctions d'interface utilisateur améliorées ---

def create_header():
    """Crée un header moderne avec titre."""
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">RappelConso - Surveillance des Alertes Alimentaires</h1>
    </div>
    """, unsafe_allow_html=True)

def create_search_bar(placeholder="Rechercher par nom, marque, référence..."):
    """Crée une barre de recherche moderne avec icône."""
    return st.text_input("", placeholder=placeholder, label_visibility="collapsed")

def display_metrics_cards(data):
    """Affiche des cartes de métriques plus attrayantes avec icônes et animations."""
    if data.empty:
        return

    # Calculer les métriques
    total_recalls = len(data)
    unique_categories = data['sous_categorie_produit'].nunique()

    # Calculer les rappels récents (30 derniers jours)
    today = date.today()
    thirty_days_ago = today - timedelta(days=30)
    recent_recalls = len(data[data['date_publication'] >= thirty_days_ago])

    # Calculer le pourcentage de risques graves
    grave_keywords = ['microbiologique', 'listeria', 'salmonelle', 'allergie', 'allergène', 'toxique']
    severe_risks = data[data['risques_encourus'].str.lower().str.contains('|'.join(grave_keywords), na=False)]
    severe_percent = int((len(severe_risks) / total_recalls) * 100) if total_recalls > 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-value">{}</p>
            <p class="metric-label">Total des Rappels</p>
        </div>
        """.format(total_recalls), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-value">{}</p>
            <p class="metric-label">Rappels Récents (30j)</p>
        </div>
        """.format(recent_recalls), unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-value">{}</p>
            <p class="metric-label">Catégories Uniques</p>
        </div>
        """.format(unique_categories), unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-value">{}%</p>
            <p class="metric-label">Risques Graves</p>
        </div>
        """.format(severe_percent), unsafe_allow_html=True)

def create_tabs(tabs):
    """Crée des onglets personnalisés plus attrayants visuellement."""
    st.markdown('<div class="custom-tabs">', unsafe_allow_html=True)

    for i, tab in enumerate(tabs):
        active_class = "active" if i == 0 else ""
        st.markdown(f'<div class="tab-item {active_class}" onclick="selectTab({i})">{tab}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Comme Streamlit ne prend pas en charge le JavaScript natif, on utilise un widget Streamlit
    return st.radio("", tabs, horizontal=True, label_visibility="collapsed")

def display_recall_card(row):
    """Affiche un rappel dans une carte moderne avec style personnalisé."""
    col_img, col_content = st.columns([1, 3])

    with col_img:
        # Image du produit
        image_url = row.get('liens_vers_les_images', '').split('|')[0] if 'liens_vers_les_images' in row and row['liens_vers_les_images'] else "https://via.placeholder.com/120"
        st.image(image_url, width=120)

    with col_content:
        # Informations du produit
        st.markdown(f"#### {row.get('modeles_ou_references', 'Produit non spécifié')}")

        # Date de publication
        formatted_date = row['date_publication'].strftime('%d/%m/%Y') if isinstance(row['date_publication'], date) else 'N/A'
        st.markdown(f"📅 **{formatted_date}**")

        # Badge de risque coloré selon la gravité
        risk_text = str(row.get('risques_encourus', '')).lower()
        if any(keyword in risk_text for keyword in ['listeria', 'salmonelle', 'toxique', 'grave']):
            st.markdown(f"🔴 **{row.get('risques_encourus', 'Risque non spécifié')}**")
        elif any(keyword in risk_text for keyword in ['allergie', 'allergène', 'microbiologique']):
            st.markdown(f"🟠 **{row.get('risques_encourus', 'Risque non spécifié')}**")
        else:
            st.markdown(f"🟡 **{row.get('risques_encourus', 'Risque non spécifié')}**")

        # Marque et motif
        st.markdown(f"**Marque:** {row.get('marque_produit', 'N/A')}")
        st.markdown(f"**Motif:** {row.get('motif_rappel', 'N/A')}")

        # Bouton pour l'affichette
        pdf_link = row.get('lien_vers_affichette_pdf', '#')
        st.markdown(f"[📄 Voir l'affichette]({pdf_link})")

    # Ligne de séparation entre les rappels
    st.markdown("---")

def display_recent_recalls_improved(data, start_index=0, items_per_page=6):
    """Affiche les rappels récents avec une présentation améliorée et pagination."""
    if data.empty:
        st.info("Aucun rappel ne correspond à vos critères de recherche.")
        return

    # Sous-titre avec compteur
    st.markdown(f"### Derniers Rappels ({len(data)} résultats)")

    # Pagination améliorée
    end_index = min(start_index + items_per_page, len(data))
    current_recalls = data.iloc[start_index:end_index]

    # Afficher les cartes de rappel en grille 2x3
    for i in range(0, len(current_recalls), 2):
        col1, col2 = st.columns(2)

        with col1:
            if i < len(current_recalls):
                display_recall_card(current_recalls.iloc[i])

        with col2:
            if i + 1 < len(current_recalls):
                display_recall_card(current_recalls.iloc[i + 1])

    # Contrôles de pagination améliorés
    st.markdown('<div class="pagination-container">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if start_index > 0:
            if st.button("← Précédent", key="prev"):
                st.session_state.start_index = max(0, start_index - items_per_page)
                st.rerun()

    with col2:
        st.markdown(f'<div class="pagination-info">Affichage {start_index + 1}-{end_index} sur {len(data)}</div>', unsafe_allow_html=True)

    with col3:
        if end_index < len(data):
            if st.button("Suivant →", key="next"):
                st.session_state.start_index = start_index + items_per_page
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

def create_advanced_filters(df):
    """Crée des filtres avancés plus interactifs et visuels."""
    with st.expander("Filtres avancés", expanded=False):
        # Filtres à deux colonnes pour une meilleure utilisation de l'espace
        col1, col2 = st.columns(2)

        with col1:
            # Filtres de catégories avec comptage
            all_categories = df['categorie_produit'].value_counts().reset_index()
            all_categories.columns = ['categorie', 'count']
            categories_options = [f"{row['categorie']} ({row['count']})" for _, row in all_categories.iterrows()]

            selected_categories = st.multiselect(
                "Catégories de produits",
                options=categories_options,
                default=[]
            )

            # Extraction des catégories sélectionnées sans les compteurs
            selected_categories_clean = [cat.split(" (")[0] for cat in selected_categories]

            # Filtre des sous-catégories en fonction des catégories sélectionnées
            filtered_df_for_subcats = df if not selected_categories_clean else df[df['categorie_produit'].isin(selected_categories_clean)]
            all_subcategories = filtered_df_for_subcats['sous_categorie_produit'].value_counts().reset_index()
            all_subcategories.columns = ['sous_categorie', 'count']
            subcategories_options = [f"{row['sous_categorie']} ({row['count']})" for _, row in all_subcategories.iterrows()]

            selected_subcategories = st.multiselect(
                "Sous-catégories",
                options=subcategories_options,
                default=[]
            )

            # Extraction des sous-catégories sélectionnées sans les compteurs
            selected_subcategories_clean = [subcat.split(" (")[0] for subcat in selected_subcategories]

        with col2:
            # Filtres de risques avec comptage
            all_risks = df['risques_encourus'].value_counts().reset_index()
            all_risks.columns = ['risque', 'count']
            risks_options = [f"{row['risque']} ({row['count']})" for _, row in all_risks.iterrows()]

            selected_risks = st.multiselect(
                "Types de risques",
                options=risks_options,
                default=[]
            )

            # Extraction des risques sélectionnés sans les compteurs
            selected_risks_clean = [risk.split(" (")[0] for risk in selected_risks]

            # Filtre de date avec sélecteur plus convivial
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input("Du", START_DATE)
            with col_date2:
                end_date = st.date_input("Au", date.today())

        # Bouton de réinitialisation des filtres
        if st.button("Réinitialiser les filtres", type="secondary"):
            st.session_state.clear()
            st.experimental_rerun()

    # Afficher les filtres actifs sous forme de badges
    active_filters = []

    if selected_categories_clean:
        active_filters.extend([f"Catégorie: {cat}" for cat in selected_categories_clean])

    if selected_subcategories_clean:
        active_filters.extend([f"Sous-catégorie: {subcat}" for subcat in selected_subcategories_clean])

    if selected_risks_clean:
        active_filters.extend([f"Risque: {risk}" for risk in selected_risks_clean])

    if start_date != START_DATE or end_date != date.today():
        active_filters.append(f"Période: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}")

    if active_filters:
        st.markdown('<div class="filter-pills">', unsafe_allow_html=True)
        for filter_text in active_filters:
            st.markdown(f'<div class="filter-pill">{filter_text}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    return selected_categories_clean, selected_subcategories_clean, selected_risks_clean, (start_date, end_date)

def create_improved_visualizations(data):
    """Crée des visualisations plus informatives et visuellement attrayantes."""
    if data.empty:
        st.info("Données insuffisantes pour générer des visualisations.")
        return

    # Onglets pour les différentes visualisations
    viz_tab = create_tabs(["Tendances temporelles", "Répartition par catégorie", "Types de risques", "Cartographie"])

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)

    if viz_tab == "Tendances temporelles":
        # Analyse des tendances temporelles
        data['year_month'] = pd.to_datetime(data['date_publication']).dt.strftime('%Y-%m')
        monthly_data = data.groupby('year_month').size().reset_index(name='count')

        # Calculer la tendance
        x = list(range(len(monthly_data)))

        # Créer un graphique combiné avec barre et ligne de tendance
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Ajouter les barres pour les comptages mensuels
        fig.add_trace(
            go.Bar(
                x=monthly_data['year_month'],
                y=monthly_data['count'],
                name="Rappels mensuels",
                marker_color='rgba(30, 136, 229, 0.8)'
            )
        )

        # Calculer la moyenne mobile sur 3 mois
        if len(monthly_data) >= 3:
            monthly_data['moving_avg'] = monthly_data['count'].rolling(window=3).mean()

            # Ajouter la ligne de tendance
            fig.add_trace(
                go.Scatter(
                    x=monthly_data['year_month'],
                    y=monthly_data['moving_avg'],
                    mode='lines',
                    name="Tendance (moy. mobile 3 mois)",
                    line=dict(color='rgba(255, 87, 34, 0.8)', width=3)
                ),
                secondary_y=False
            )

        # Mise en page du graphique
        fig.update_layout(
            title="Évolution des rappels de produits au fil du temps",
            xaxis_title="Mois",
            yaxis_title="Nombre de rappels",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            template="plotly_white",
            height=500
        )

        # Optimisation pour mobile
        fig.update_layout(
            margin=dict(l=10, r=10, t=50, b=30),
            autosize=True
        )

        st.plotly_chart(fig, use_container_width=True)

    elif viz_tab == "Répartition par catégorie":
        # Répartition par catégorie avec graphiques en anneau
        col1, col2 = st.columns(2)

        with col1:
            # Graphique des sous-catégories
            top_subcategories = data['sous_categorie_produit'].value_counts().head(8)

            # Ajout d'une catégorie "Autres" pour le reste
            if len(data['sous_categorie_produit'].unique()) > 8:
                other_count = data['sous_categorie_produit'].value_counts().iloc[8:].sum()
                top_subcategories = pd.concat([top_subcategories, pd.Series({"Autres": other_count})])

            fig_subcats = px.pie(
                names=top_subcategories.index,
                values=top_subcategories.values,
                title="Répartition par Sous-catégories",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Bold
            )

            fig_subcats.update_traces(textposition='outside', textinfo='percent+label')
            fig_subcats.update_layout(
                annotations=[dict(text='Sous-catégories', showarrow=False)],
                showlegend=False
            )

            st.plotly_chart(fig_subcats, use_container_width=True)

        with col2:
            # Graphique des catégories principales
            top_categories = data['categorie_produit'].value_counts().head(6)

            # Ajout d'une catégorie "Autres" pour le reste
            if len(data['categorie_produit'].unique()) > 6:
                other_count = data['categorie_produit'].value_counts().iloc[6:].sum()
                top_categories = pd.concat([top_categories, pd.Series({"Autres": other_count})])

            fig_cats = px.pie(
                names=top_categories.index,
                values=top_categories.values,
                title="Répartition par Catégories",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )

            fig_cats.update_traces(textposition='outside', textinfo='percent+label')
            fig_cats.update_layout(
                annotations=[dict(text='Catégories', showarrow=False)],
                showlegend=False
            )

            st.plotly_chart(fig_cats, use_container_width=True)

        # Histogramme horizontal des 10 produits les plus rappelés
        top_products = data['marque_produit'].value_counts().head(10).sort_values(ascending=True)

        fig_products = go.Figure(go.Bar(
            x=top_products.values,
            y=top_products.index,
            orientation='h',
            marker=dict(
                color='rgba(38, 166, 154, 0.8)',
                line=dict(color='rgba(38, 166, 154, 1.0)', width=2)
            )
        ))

        fig_products.update_layout(
            title="Top 10 des Marques avec le Plus de Rappels",
            xaxis_title="Nombre de rappels",
            yaxis_title="Marque",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig_products, use_container_width=True)

    elif viz_tab == "Types de risques":
        # Analyse des risques
        col1, col2 = st.columns(2)

        with col1:
            # Graphique des types de risques
            top_risks = data['risques_encourus'].value_counts().head(8)

            fig_risks = px.bar(
                x=top_risks.index,
                y=top_risks.values,
                title="Principaux Types de Risques",
                color=top_risks.values,
                color_continuous_scale='Reds',
                labels={'x': 'Type de risque', 'y': 'Nombre de rappels', 'color': 'Nombre'}
            )

            fig_risks.update_layout(xaxis_tickangle=-45, height=450)
            st.plotly_chart(fig_risks, use_container_width=True)

        with col2:
            # Heatmap des risques par catégorie
            risk_category = pd.crosstab(data['risques_encourus'], data['categorie_produit'])
            risk_category = risk_category.loc[risk_category.sum(axis=1).sort_values(ascending=False).head(6).index]
            risk_category = risk_category[risk_category.sum().sort_values(ascending=False).head(6).index]

            fig_heatmap = px.imshow(
                risk_category,
                labels=dict(x="Catégorie de produit", y="Type de risque", color="Nombre de rappels"),
                title="Corrélation entre Risques et Catégories",
                color_continuous_scale='YlOrRd'
            )

            fig_heatmap.update_layout(height=450)
            st.plotly_chart(fig_heatmap, use_container_width=True)

        # Graphique de l'évolution des principaux risques au fil du temps
        top_5_risks = data['risques_encourus'].value_counts().head(5).index.tolist()
        data_risks_over_time = data[data['risques_encourus'].isin(top_5_risks)].copy()
        data_risks_over_time['year_quarter'] = pd.to_datetime(data_risks_over_time['date_publication']).dt.to_period('Q').astype(str)

        risk_time = pd.crosstab(data_risks_over_time['year_quarter'], data_risks_over_time['risques_encourus'])

        fig_risk_time = px.line(
            risk_time,
            x=risk_time.index,
            y=risk_time.columns,
            title="Évolution des Principaux Risques au Fil du Temps",
            labels={'x': 'Trimestre', 'y': 'Nombre de rappels', 'variable': 'Type de risque'}
        )

        fig_risk_time.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig_risk_time, use_container_width=True)

    elif viz_tab == "Cartographie":
        st.info("La cartographie de rappels par région sera disponible prochainement.")

    st.markdown('</div>', unsafe_allow_html=True)

def create_improved_chatbot():
    """Crée une interface de chatbot plus engageante et intuitive."""
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown('<div class="chat-messages" id="chat-messages">', unsafe_allow_html=True)

    # Afficher l'historique des messages
    if "chat_history" in st.session_state:
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["parts"][0]

            if role == "user":
                st.markdown(f'<div class="message user">{content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="message bot">{content}</div>', unsafe_allow_html=True)
    else:
        # Message de bienvenue du chatbot
        st.markdown("""
        <div class="message bot">
            Bonjour 👋 Je suis l'assistant RappelConso. Je peux vous aider à trouver des informations
            sur les rappels de produits alimentaires en France. Que souhaitez-vous savoir ?
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Zone de saisie du message
    st.markdown('<div class="chat-input">', unsafe_allow_html=True)

    # Comme Streamlit ne supporte pas directement l'UI personnalisée, on utilise les widgets Streamlit
    user_input = st.text_area("Votre question:",
                             placeholder="Exemple : Quels sont les rappels récents de fromage ?",
                             height=100,
                             key="chat_input",
                             label_visibility="collapsed")

    col1, col2 = st.columns([4, 1])

    with col2:
        send_button = st.button("Envoyer", key="send_button", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    return user_input, send_button

def create_loading_animation():
    """Crée une animation de chargement personnalisée."""
    st.markdown("""
    <div class="loading">
        <div class="progress-bar">
            <div class="progress-bar-fill" id="progress-fill"></div>
        </div>
        <p id="loading-text">Chargement des données en cours...</p>
    </div>
    """, unsafe_allow_html=True)

    # Simuler une progression (dans un vrai code, cela serait lié à la progression réelle)
    for i in range(10):
        time.sleep(0.1)
        st.markdown(f"""
        <script>
            document.getElementById('progress-fill').style.width = '{(i+1)*10}%';
            document.getElementById('loading-text').innerText = 'Chargement des données en cours... {(i+1)*10}%';
        </script>
        """, unsafe_allow_html=True)

def create_onboarding_tips():
    """Affiche des conseils d'utilisation pour les nouveaux utilisateurs."""
    if "onboarding_done" not in st.session_state:
        st.session_state.onboarding_done = False

    if not st.session_state.onboarding_done:
        with st.expander("💡 Conseils d'utilisation (cliquez pour développer)", expanded=True):
            st.markdown("""
            ### Bienvenue sur RappelConso !

            Voici quelques conseils pour utiliser efficacement cette application :

            1. **Recherche rapide** : Utilisez la barre de recherche pour trouver rapidement des produits par marque, nom ou référence.
            2. **Filtres avancés** : Affinez votre recherche avec les filtres accessibles dans le menu latéral.
            3. **Visualisations** : Explorez les tendances et statistiques dans l'onglet Visualisation.
            4. **Chatbot** : Posez des questions en langage naturel comme "Quels sont les rappels de fromage ce mois-ci ?"
            5. **Notifications** : Activez les notifications pour être alerté des nouveaux rappels qui correspondent à vos critères.

            [En savoir plus sur RappelConso](https://rappel.conso.gouv.fr/)
            """)

            if st.button("J'ai compris", key="onboarding_button"):
                st.session_state.onboarding_done = True
                st.rerun()

def main():
    # Configuration de l'état de la session pour la pagination et les filtres
    if 'start_index' not in st.session_state:
        st.session_state.start_index = 0

    # Création du header moderne
    create_header()

    # Onboarding pour les nouveaux utilisateurs
    create_onboarding_tips()

    # --- Sidebar améliorée ---
    st.sidebar.markdown("""
    <img src="https://raw.githubusercontent.com/M00N69/RAPPELCONSO/main/logo%2004%20copie.jpg" class="sidebar-logo">
    """, unsafe_allow_html=True)

    st.sidebar.title("Navigation")

    # Tabs de navigation plus moderne
    page = st.sidebar.radio(
        "Sélectionner une page",
        ["Tableau de bord", "Recherche avancée", "Visualisations", "Détails", "Chatbot"],
        format_func=lambda x: f"📊 {x}" if x == "Tableau de bord" else
                      f"🔍 {x}" if x == "Recherche avancée" else
                      f"📈 {x}" if x == "Visualisations" else
                      f"📋 {x}" if x == "Détails" else
                      f"🤖 {x}"
    )

    # Bouton de mise à jour des données
    if st.sidebar.button("🔄 Mettre à jour les données", type="primary"):
        st.cache_data.clear()
        st.session_state["restart_key"] = st.session_state.get("restart_key", 0) + 1
        st.sidebar.success("Données mises à jour avec succès!")

    # Chargement des données avec animation
    with st.spinner("Chargement des données..."):
        # Simuler un délai de chargement pour démonstration
        # time.sleep(1)

        try:
            # Charge les données (fonction à implémenter comme dans le code original)
            # Pour la démonstration, on simule un chargement réussi
            df = load_data(API_URL, START_DATE)

            if df.empty:
                st.error("Impossible de charger les données. Veuillez réessayer plus tard.")
                st.stop()
        except Exception as e:
            st.error(f"Erreur lors du chargement des données: {e}")
            st.stop()

    # Barre de recherche moderne commune à toutes les pages
    search_term = create_search_bar("Rechercher un produit, une marque, un risque...")

    # --- Contenu principal selon la page sélectionnée ---
    if page == "Tableau de bord":
        # Sélecteur de colonne pour la recherche
        search_column = st.selectbox("Choisissez la colonne à rechercher",
                                     options=list(COLUMN_MAPPING.keys()),
                                     index=0)

        # Filtrer les données selon le terme de recherche
        filtered_data = filter_data(df, [], [], search_term, (START_DATE, date.today()), [], search_column if search_column != "Toutes les colonnes" else None)

        # Afficher les métriques améliorées
        display_metrics_cards(filtered_data)

        # Afficher les derniers rappels avec une présentation améliorée
        display_recent_recalls_improved(filtered_data, st.session_state.start_index)

        # Afficher une sélection des visualisations les plus pertinentes
        st.subheader("Aperçu des tendances")
        create_improved_visualizations(filtered_data)

    elif page == "Recherche avancée":
        st.subheader("Recherche avancée de rappels")

        # Filtres avancés améliorés
        categories, subcategories, risks, dates = create_advanced_filters(df)

        # Filtrer les données
        filtered_data = filter_data(df, subcategories, risks, search_term, dates, categories)

        # Afficher les résultats de recherche
        st.write(f"**{len(filtered_data)}** rappels correspondent à vos critères.")

        # Afficher les résultats avec la présentation améliorée
        display_recent_recalls_improved(filtered_data, st.session_state.start_index, items_per_page=10)

    elif page == "Visualisations":
        st.subheader("Analyse des tendances et statistiques")

        # Filtres simplifiés pour les visualisations
        categories, subcategories, risks, dates = create_advanced_filters(df)

        # Filtrer les données
        filtered_data = filter_data(df, subcategories, risks, search_term, dates, categories)

        # Afficher les visualisations améliorées
        create_improved_visualizations(filtered_data)

    elif page == "Détails":
        st.subheader("Détails des rappels")

        # Filtres avancés
        categories, subcategories, risks, dates = create_advanced_filters(df)

        # Filtrer les données
        filtered_data = filter_data(df, subcategories, risks, search_term, dates, categories)

        if not filtered_data.empty:
            # Afficher les données sous forme de tableau interactif
            st.dataframe(
                filtered_data,
                column_config={
                    "date_publication": st.column_config.DateColumn("Date de publication"),
                    "marque_produit": st.column_config.TextColumn("Marque"),
                    "modeles_ou_references": st.column_config.TextColumn("Modèle/Référence"),
                    "risques_encourus": st.column_config.TextColumn("Risques"),
                    "sous_categorie_produit": st.column_config.TextColumn("Sous-catégorie"),
                },
                use_container_width=True,
                hide_index=True
            )

            # Bouton de téléchargement
            csv = filtered_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les données (CSV)",
                data=csv,
                file_name='rappelconso_export.csv',
                mime='text/csv'
            )
        else:
            st.info("Aucune donnée à afficher avec les filtres sélectionnés.")

    elif page == "Chatbot":
        st.subheader("Assistant RappelConso")

        # Créer l'interface du chatbot améliorée
        user_input, send_button = create_improved_chatbot()

        # Traitement des messages du chatbot
        if send_button and user_input.strip():
            # Configurer l'API Gemini si disponible
            try:
                api_key = st.secrets["api_key"]
                genai.configure(api_key=api_key)

                # Configuration du modèle
                model = configure_model()

                # Obtenir le contexte pertinent de la base de données
                filtered_data = filter_data(df, [], [], "", (START_DATE, date.today()), [])
                relevant_data = get_relevant_data_as_text(user_input, filtered_data)

                # Construire le message
                context = (
                    "Informations contextuelles sur les rappels de produits :\n\n" +
                    relevant_data +
                    "\n\nQuestion de l'utilisateur : " + user_input
                )

                # Initialiser l'historique du chat si nécessaire
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []

                # Ajouter la question de l'utilisateur à l'historique
                st.session_state.chat_history.append({"role": "user", "parts": [user_input]})

                # Obtenir la réponse du modèle avec une animation de chargement
                with st.spinner("Réflexion en cours..."):
                    convo = model.start_chat(history=[])
                    response = convo.send_message(context)

                    # Ajouter la réponse à l'historique
                    st.session_state.chat_history.append({"role": "assistant", "parts": [response.text]})

                # Rafraîchir l'UI
                st.rerun()

            except Exception as e:
                st.error(f"Erreur lors de la communication avec l'API: {e}")

# Lancement de l'application
if __name__ == "__main__":
    main()
