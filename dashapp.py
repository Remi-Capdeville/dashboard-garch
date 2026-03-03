import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from arch import arch_model

# 1. Configuration de la page
st.set_page_config(page_title="GARCH Dashboard - Rémi", layout="wide")

# Style CSS pour un rendu épuré
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True) # <-- Corrigé ici : html au lieu de index

st.title("📈 Analyse de Volatilité GARCH(1,1)")
st.markdown("---")

# 2. Barre latérale
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Symbole (Ticker)", "^GSPC")
start_date = st.sidebar.date_input("Date de début", pd.to_datetime("2010-01-01"))

# 3. Chargement des données
@st.cache_data
def load_data(t, s):
    try:
        df = yf.download(t, start=s)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        return df.dropna()
    except Exception:
        return None

data = load_data(ticker, start_date)

# 4. Calcul et Affichage
if data is not None and len(data) > 50:
    returns_pct = data['log_return'] * 100
    
    with st.spinner('Calcul du modèle...'):
        model = arch_model(returns_pct, vol='GARCH', p=1, q=1, dist='t')
        res = model.fit(disp='off')

    # Graphique Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=res.conditional_volatility / 100,
        mode='lines',
        name='Volatilité GARCH',
        line=dict(color='#2E86AB', width=2)
    ))

    fig.update_layout(
        template='plotly_dark',
        title=f"Volatilité Conditionnelle : {ticker}",
        xaxis_title="Date",
        yaxis_title="Volatilité",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Résumé statistique
    with st.expander("Voir les détails du modèle"):
        st.text(res.summary())
else:
    st.warning("Veuillez entrer un Ticker valide ou augmenter la plage de dates.")
