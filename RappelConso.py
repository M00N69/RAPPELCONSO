import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime

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

if not df.empty:
    df['year'] = df['date_de_publication'].dt.year
    selected_year = st.sidebar.selectbox('Select Year', options=sorted(df['year'].unique()))

    filtered_data = df[df['year'] == selected_year]
    if not filtered_data.empty:
        min_date = filtered_data['date_de_publication'].min()
        max_date = filtered_data['date_de_publication'].max()
        selected_dates = st.sidebar.slider(
            "Select a date range:", 
            min_value=min_date.to_pydatetime(), 
            max_value=max_date.to_pydatetime(), 
            value=(min_date.to_pydatetime(), max_date.to_pydatetime())
        )
        filtered_data = filtered_data[
            (filtered_data['date_de_publication'] >= selected_dates[0]) & 
            (filtered_data['date_de_publication'] <= selected_dates[1])
        ]

        # Layout and styling adjustments for pie charts
        col1, col2 = st.columns(2)
        with col1:
            risk_fig = px.pie(filtered_data, names='risques_encourus_par_le_consommateur', title='Risks Incurred by Consumers')
            risk_fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=40, l=0, r=0, b=0)
            )
            st.plotly_chart(risk_fig, use_container_width=True)

        with col2:
            legal_fig = px.pie(filtered_data, names='nature_juridique_du_rappel', title='Legal Nature of Recall')
            legal_fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=40, l=0, r=0, b=0)
            )
            st.plotly_chart(legal_fig, use_container_width=True)
    else:
        st.write("No data available for the selected year and date range.")
else:
    st.write("No data available.")
