import streamlit as st
import pandas as pd
from datetime import datetime

# Sample JSON data provided
data = {
    "nhits": 8996,
    "parameters": {
        "dataset": "rappelconso0",
        "q": "categorie_de_produit:Alimentation",
        "rows": 10000,
        "start": 0,
        "format": "json",
        "timezone": "UTC"
    },
    "records": [
        {
            "datasetid": "rappelconso0",
            "recordid": "71902465aec74c5261f9e5df78c6ee6310ecf7e2",
            "fields": {
                "conduites_a_tenir_par_le_consommateur": "Ne plus consommer",
                "libelle": "Bagels Sésame Système U",
                "rappelguid": "F57D7BC2-1373-4000-B1C3-34028C8DBFAD",
                "nature_juridique_du_rappel": "Volontaire (sans arrêté préfectoral)",
                "reference_fiche": "2021-03-0004",
                "motif_du_rappel": "Présence d’un produit chimique, l’oxyde d’éthylène, à une teneur supérieure à la limite maximum réglementaire",
                "distributeurs": "Système U",
                "date_de_publication": "2021-03-26T15:44:43+00:00",
                "sous_categorie_de_produit": "Céréales et produits de boulangerie",
                "categorie_de_produit": "Alimentation"
            }
        },
        {
            "datasetid": "rappelconso0",
            "recordid": "7795e774899cbd002195e593b8ee5c4d70cbdcef",
            "fields": {
                "conduites_a_tenir_par_le_consommateur": "Ne plus consommer Ne plus utiliser le produit Rapporter le produit au point de vente Contacter le service consommateur",
                "motif_du_rappel": "présence de LISTERIA MONOCYTOGENES",
                "distributeurs": "E. LECLERC",
                "date_de_publication": "2021-04-15T12:30:00+00:00",
                "sous_categorie_de_produit": "Lait et produits laitiers",
                "categorie_de_produit": "Alimentation"
            }
        }
    ]
}

# --- Simplified START_DATE declaration ---
START_DATE = datetime(2021, 4, 1).date()  # Use only the date part

@st.cache_data
def load_data():
    """Loads and preprocesses the recall data from the provided JSON."""
    # Load data into DataFrame
    df = pd.DataFrame([record['fields'] for record in data['records']])

    # Ensure 'date_de_publication' is converted to datetime and extract only the date part
    df['date_de_publication'] = pd.to_datetime(df['date_de_publication'], errors='coerce').dt.date

    # Debugging: Show the data types in the DataFrame
    st.write("Data types after conversion:", df.dtypes)

    # Debugging: Ensure that all dates are valid
    if df['date_de_publication'].isna().any():
        st.write("Rows with invalid 'date_de_publication':", df[df['date_de_publication'].isna()])
        st.error("Some dates could not be parsed and were removed.")
        df = df.dropna(subset=['date_de_publication'])

    # Filter data to only include records on or after START_DATE
    try:
        df = df[df['date_de_publication'] >= START_DATE]
    except TypeError as e:
        st.error(f"TypeError encountered during date comparison: {e}")
        st.write("Unique values in 'date_de_publication':", df['date_de_publication'].unique())
        return pd.DataFrame()  # Return an empty DataFrame to prevent further errors

    return df

def main():
    st.title("RappelConso - Dashboard")

    # Load data
    df = load_data()

    if df.empty:
        st.error("No valid data available after filtering.")
    else:
        # Display the filtered DataFrame
        st.write("Filtered DataFrame", df)

if __name__ == "__main__":
    main()
