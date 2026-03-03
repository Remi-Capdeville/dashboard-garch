import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from arch import arch_model

# 1. Configuration de la page
st.set_page_config(page_title="GARCH Dashboard - Rémi", layout="wide")

# Style CSS pour cacher le menu Streamlit et faire plus "App pro"
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_index=True)

st.title("📈 Analyse de Volatilité GARCH(1,1)")
st.markdown("---")

# 2. Barre latérale pour les paramètres
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Symbole (Ticker)", "^GSPC")
start_date = st.sidebar.date_input("Date de début", pd.to_datetime("2010-01-01"))

# 3. Fonction de chargement des données sécurisée
@st.cache_data
def load_data(t, s):
    try:
        df = yf.download(t, start=s)
        if df.empty:
            return None
        
        # Gestion des colonnes si yfinance renvoie un MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        return df.dropna()
    except Exception:
        return None

data = load_data(ticker, start_date)

# 4. Vérification et Affichage
if data is not None and len(data) > 50:
    # --- Calcul du modèle GARCH ---
    returns_pct = data['log_return'] * 100
    
    with st.spinner('Calcul du modèle GARCH en cours...'):
        # Modèle avec distribution de Student pour capturer les queues épaisses
        model = arch_model(returns_pct, vol='GARCH', p=1, q=1, dist='t')
        res = model.fit(disp='off')

    # --- Graphique Plotly ---
    fig = go.Figure()

    # Volatilité réalisée (rendements absolus)
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['log_return'].abs(),
        mode='lines',
        name='Volatilité Réalisée (|r|)',
        line=dict(color='rgba(241, 143, 1, 0.2)', width=1),
        fill='tozeroy'
    ))

    # Volatilité conditionnelle GARCH
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=res.conditional_volatility / 100,
        mode='lines',
        name='Volatilité GARCH(1,1)',
        line=dict(color='#2E86AB', width=2.5)
    ))

    fig.update_layout(
        template='plotly_dark',
        title=f"Analyse de la volatilité pour {ticker}",
        xaxis_title="Date",
        yaxis_title="Volatilité",
        height=600,
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Métriques ---
    col1, col2, col3, col4 = st.columns(4)
    persistance = res.params['alpha[1]'] + res.params['beta[1]']
    
    col1.metric("Persistance (α+β)", f"{persistance:.3f}")
    col2.metric("Moyenne (μ)", f"{res.params['mu']:.4f}")
    col3.metric("AIC", f"{int(res.aic)}")
    col4.metric("Observations", len(data))

    with st.expander("Voir les détails statistiques du modèle"):
        st.text(res.summary())

else:
    st.error(f"❌ Erreur : Impossible de charger les données pour '{ticker}'.")
    st.info("Vérifiez que le ticker est correct (ex: AAPL, BTC-USD, ^FCHI) et que la date n'est pas trop récente.")
