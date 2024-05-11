import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime

@st.cache(allow_output_mutation=True)
def load_data():
    # Placeholder for your actual data loading logic
    url = "https://data.economie.gouv.fr/api/records/1.0/search/?dataset=rappelconso0&q="
    response = requests.get(url + "&rows=1000")
    data = response.json()
    records = [rec['fields'] for rec in data['records']]
    df = pd.DataFrame(records)
    df['date_de_publication'] = pd.to_datetime(df['date_de_publication'])
    return df

df = load_data()

# Sidebar for Filters
st.sidebar.title("Filters")
selected_year = st.sidebar.selectbox('Select Year', options=pd.to_datetime(df['date_de_publication']).dt.year.unique())

filtered_data = df[df['date_de_publication'].dt.year == selected_year]

# Date range selector
min_date, max_date = filtered_data['date_de_publication'].min(), filtered_data['date_de_publication'].max()
selected_dates = st.sidebar.slider("Select date range", min_value=min_date, max_value=max_date, value=(min_date, max_date))
filtered_data = filtered_data[(filtered_data['date_de_publication'] >= selected_dates[0]) & (filtered_data['date_de_publication'] <= selected_dates[1])]

# Risk type filtering
risk_types = st.sidebar.multiselect('Select risk types', options=filtered_data['risques_encourus_par_le_consommateur'].unique())
if risk_types:
    filtered_data = filtered_data[filtered_data['risques_encourus_par_le_consommateur'].isin(risk_types)]

# Main Page
st.title('Visualisation des Rappels de Produits')

# Displaying pie charts with fixed layout
col1, col2 = st.columns(2)
with col1:
    risk_fig = px.pie(filtered_data, names='risques_encourus_par_le_consommateur', title='Risks Incurred by Consumers')
    st.plotly_chart(risk_fig, use_container_width=True)

with col2:
    legal_fig = px.pie(filtered_data, names='nature_juridique_du_rappel', title='Legal Nature of Recall')
    st.plotly_chart(legal_fig, use_container_width=True)

# Interactive dashboards
with st.expander("See Detailed Data"):
    st.dataframe(filtered_data)

# Exporting data
@st.cache
def convert_df_to_excel(df):
    return df.to_excel(pd.ExcelWriter('filtered_data.xlsx'), index=False)

if not filtered_data.empty:
    st.download_button(
        label="Download Excel",
        data=convert_df_to_excel(filtered_data),
        file_name='filtered_data.xlsx',
        mime='application/vnd.ms-excel'
    )

# Display total number of recalls
st.sidebar.write(f"Total Recalls: {len(filtered_data)}")

