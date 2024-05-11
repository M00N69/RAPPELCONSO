import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime

# Cache the data loading function to avoid reloading on every rerun
@st.cache(allow_output_mutation=True)
def load_data():
    url = "https://data.economie.gouv.fr/api/records/1.0/search/?dataset=rappelconso0&q=&rows=10000"
    response = requests.get(url)
    data = response.json()
    records = [rec['fields'] for rec in data['records']]
    df = pd.DataFrame(records)
    df['date_de_publication'] = pd.to_datetime(df['date_de_publication'])
    return df

df = load_data()

# Sidebar for Year and Date Range Filter
if not df.empty:
    df['year'] = df['date_de_publication'].dt.year
    years = df['year'].unique()
    selected_year = st.sidebar.selectbox('Select Year', options=sorted(years))

    filtered_data = df[df['year'] == selected_year]
    if not filtered_data.empty:
        min_date, max_date = filtered_data['date_de_publication'].min(), filtered_data['date_de_publication'].max()
        selected_dates = st.sidebar.slider("Select date range:", min_value=min_date, max_value=max_date, value=(min_date, max_date))
        filtered_data = filtered_data[(filtered_data['date_de_publication'] >= selected_dates[0]) & (filtered_data['date_de_publication'] <= selected_dates[1])]
    else:
        st.sidebar.write("No data available for the selected year.")
else:
    st.sidebar.write("No data available.")

# Main Page Content
st.title('Visualisation des Rappels de Produits')

# Filtering by Risk Types
if not filtered_data.empty:
    risk_types = st.multiselect('Select risk types', options=filtered_data['risques_encourus_par_le_consommateur'].unique())
    if risk_types:
        filtered_data = filtered_data[filtered_data['risques_encourus_par_le_consommateur'].isin(risk_types)]

    # Displaying pie charts
    col1, col2 = st.columns(2)
    with col1:
        risk_fig = px.pie(filtered_data, names='risques_encourus_par_le_consommateur', title='Risks Incurred by Consumers')
        st.plotly_chart(risk_fig, use_container_width=True)

    with col2:
        legal_fig = px.pie(filtered_data, names='nature_juridique_du_rappel', title='Legal Nature of Recall')
        st.plotly_chart(legal_fig, use_container_width=True)

    # Exporting data to Excel
    st.download_button(
        label="Download Filtered Data as Excel",
        data=filtered_data.to_csv().encode('utf-8'),
        file_name='filtered_data.csv',
        mime='text/csv'
    )

    # Displaying data frame in an expander
    with st.expander("See Detailed Data"):
        st.dataframe(filtered_data)

    # Display total number of recalls
    st.write(f"Total Recalls: {len(filtered_data)}")

else:
    st.write("Adjust the filters to view data.")

