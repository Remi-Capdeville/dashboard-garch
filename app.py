import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from arch import arch_model

# Configuration de la page
st.set_page_config(page_title="GARCH Dashboard", layout="wide")

st.title("📈 Analyse de Volatilité GARCH(1,1)")
st.sidebar.header("Paramètres")

# 1. Récupération des données
ticker = st.sidebar.text_input("Symbole (Ticker)", "^GSPC")
start_date = st.sidebar.date_input("Date de début", pd.to_datetime("2010-01-01"))

@st.cache_data # Pour éviter de recharger à chaque clic
def load_data(t, s):
    df = yf.download(t, start=s)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel("Ticker", axis=1)
    df['log_return'] = np.log(df['Close']/df['Close'].shift(1))
    return df.dropna()

data = load_data(ticker, start_date)

# 2. Modèle GARCH
st.subheader(f"Modèle GARCH(1,1) sur {ticker}")
returns_pct = data['log_return'] * 100

with st.spinner('Calcul du modèle...'):
    model = arch_model(returns_pct, vol='GARCH', p=1, q=1, dist='t')
    res = model.fit(disp='off')

# 3. Graphique Plotly (Ton code adapté)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=data.index, 
    y=res.conditional_volatility / 100,
    mode='lines',
    name='Volatilité Filtrée (GARCH)',
    line=dict(color='#2E86AB', width=2)
))

fig.update_layout(
    template='plotly_dark',
    title="Volatilité Conditionnelle Estimée",
    xaxis_title="Date",
    yaxis_title="Volatilité",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# 4. Métriques
col1, col2, col3 = st.columns(3)
col1.metric("Persistance (α + β)", f"{res.params['alpha[1]'] + res.params['beta[1]']:.3f}")
col2.metric("Log-Likelihood", f"{res.loglikelihood:.2f}")
col3.metric("AIC", f"{res.aic:.2f}")

st.write("### Résumé du modèle")
st.text(res.summary())
