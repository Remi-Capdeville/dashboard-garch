# requirements:
# pip install dash plotly yfinance arch numpy pandas dash-bootstrap-components statsmodels scipy

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import yfinance as yf
import scipy as sp

from datetime import date, timedelta
from functools import lru_cache

# Dash & Plotly
from dash import Dash, html, dcc, Input, Output, State, no_update, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Modeling
from arch import arch_model
from statsmodels.tsa.stattools import acf
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# ---------------------------
# Configuration
# ---------------------------
TRADING_DAYS_PER_YEAR = 252
DEFAULT_TICKER = "SPY"
DEFAULT_HORIZON = 21
DEFAULT_SIMS = 5000
MIN_OBS_FOR_FIT = 300
CONF_INTERVAL = 0.90

US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

# ---------------------------
# Thème et Design
# ---------------------------
THEME_COLORS = {
    'primary': '#6C63FF',
    'secondary': '#FF6B6B',
    'accent': '#4ECDC4',
    'warning': '#FFD93D',
    'success': '#95E1D3',
    'dark': '#1A1A2E',
    'light': '#F5F5F5',
    'gradient1': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'gradient2': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    'gradient3': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    'gradient4': 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
}

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

.main-container {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 25px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    padding: 2rem;
    margin: 2rem auto;
    animation: slideUp 0.5s ease-out;
}

@keyframes slideUp {
    from {
        transform: translateY(30px);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

.header-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.5rem;
    animation: fadeIn 0.8s ease-out;
}

.header-subtitle {
    color: #666;
    text-align: center;
    font-size: 1.1rem;
    margin-bottom: 2rem;
    animation: fadeIn 1s ease-out;
}

@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

.control-card {
    background: white;
    border: none;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    transition: all 0.3s ease;
    position: relative;
}

.control-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(108, 99, 255, 0.2);
}

.input-modern {
    border: 2px solid #e0e0e0;
    border-radius: 10px;
    padding: 10px 15px;
    font-size: 1rem;
    transition: all 0.3s ease;
    background: white;
    width: 100%;
}

.input-modern:focus {
    border-color: #6C63FF;
    box-shadow: 0 0 0 3px rgba(108, 99, 255, 0.1);
    outline: none;
}

/* Style spécifique pour les inputs de date */
input[type="date"] {
    cursor: pointer;
    position: relative;
    color: #333;
    font-family: 'Inter', sans-serif;
}

input[type="date"]::-webkit-calendar-picker-indicator {
    cursor: pointer;
    position: absolute;
    right: 10px;
    width: 20px;
    height: 20px;
    opacity: 0.6;
}

input[type="date"]:hover::-webkit-calendar-picker-indicator {
    opacity: 1;
}

.date-input-label {
    color: #8a8a8a;
    font-size: 0.85rem;
    margin-bottom: 0.3rem;
    font-weight: 500;
}

.btn-primary-gradient {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 10px;
    padding: 12px 30px;
    color: white;
    font-weight: 600;
    font-size: 1.1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3);
}

.btn-primary-gradient:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(108, 99, 255, 0.4);
}

.btn-primary-gradient:active {
    transform: translateY(0);
}

.stats-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    border: none;
    border-radius: 15px;
    padding: 1.5rem;
    color: white;
    box-shadow: 0 10px 30px rgba(240, 87, 108, 0.3);
    margin-bottom: 2rem;
    animation: slideIn 0.6s ease-out;
}

@keyframes slideIn {
    from {
        transform: translateX(-20px);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

.stats-title {
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 1rem;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
}

.stats-text {
    font-size: 0.95rem;
    opacity: 0.95;
    line-height: 1.6;
}

.tab-custom {
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    color: #666 !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.tab-custom:hover {
    color: #6C63FF !important;
}

.tab-custom.tab--selected {
    color: #6C63FF !important;
    border-bottom: 3px solid #6C63FF !important;
}

.alert-custom {
    border-radius: 10px;
    border: none;
    animation: shake 0.5s;
}

@keyframes shake {
    0%, 100% { transform: translateX(0); }
    10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
    20%, 40%, 60%, 80% { transform: translateX(5px); }
}

.loading-custom {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 200px;
}

.input-group-text-modern {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    font-weight: 500;
}

/* Amélioration des graphiques */
.js-plotly-plot {
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
}

/* Style pour la section date */
.date-section {
    background: rgba(108, 99, 255, 0.05);
    border-radius: 10px;
    padding: 0.5rem;
    border: 1px solid rgba(108, 99, 255, 0.1);
}
"""

# ---------------------------
# Data & Modeling Functions (UNCHANGED)
# ---------------------------

@lru_cache(maxsize=32)
def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:

    print(ticker, start, end)

    df = yf.download(ticker, start=start.strip(), end=end.strip(), auto_adjust=True, progress=False)
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"Aucune donnée pour le ticker '{ticker}' sur la période spécifiée.")
    return df[["Close"]].dropna()

def compute_log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff().dropna()

def fit_garch_t(returns_log: pd.Series):
    if len(returns_log) < MIN_OBS_FOR_FIT:
        raise ValueError(f"Série trop courte (< {MIN_OBS_FOR_FIT} observations) pour un fit fiable.")

    r_pct = (returns_log * 100.0).astype(float)
    am = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(disp="off", show_warning=False)
    if res.convergence_flag != 0:
        print(f"Attention : l'optimiseur n'a pas convergé (code: {res.convergence_flag}).")
    return res

def simulate_garch_intervals(res, r: pd.Series, h: int, n_sims: int = 5000, seed: int = 42):
    p = res.params
    w  = float(p.get("omega"))
    a  = float(p.get("alpha[1]", p.get("alpha", np.nan)))
    b  = float(p.get("beta[1]",  p.get("beta",  np.nan)))
    mu = float(p.get("mu", p.get("Const", 0.0)))
    nu = p.get("nu", None)

    # Dernier état (unités en % car série entrée en %)
    sigma_last_pct = float(res.conditional_volatility.iloc[-1])
    sig2_last_pct  = sigma_last_pct ** 2
    # Résidu standardisé observé (variance 1 par construction sous arch)
    z_last = float(res.std_resid.iloc[-1])

    rng = np.random.default_rng(seed)
    if nu is not None and np.isfinite(nu) and nu > 2.0:
        # Tirages t puis re-scale pour variance 1
        eps = rng.standard_t(df=float(nu), size=(n_sims, h)).astype(float)
        eps *= np.sqrt((float(nu) - 2.0) / float(nu))
    else:
        eps = rng.normal(size=(n_sims, h)).astype(float)

    sig2 = np.empty((n_sims, h), dtype=float)
    rsim = np.empty((n_sims, h), dtype=float)

    # Pas 0 : conditionnel à z_last observé
    sig2[:, 0] = w + a * (sig2_last_pct * (z_last ** 2)) + b * sig2_last_pct
    sig0 = np.sqrt(np.maximum(sig2[:, 0], 0.0))
    rsim[:, 0] = mu + sig0 * eps[:, 0]

    # Pas suivants : utiliser les chocs simulés eps (variance 1)
    for t in range(1, h):
        sig2[:, t] = w + a * (sig2[:, t-1] * (eps[:, t-1] ** 2)) + b * sig2[:, t-1]
        sig_t = np.sqrt(np.maximum(sig2[:, t], 0.0))
        rsim[:, t] = mu + sig_t * eps[:, t]

    # Convertit en décimal (les sigmas sont en %)
    vol_paths_dec = np.sqrt(np.maximum(sig2, 0.0)) / 100.0

    lower_q = (1.0 - CONF_INTERVAL) / 2.0
    upper_q = 1.0 - lower_q

    q_low  = np.quantile(vol_paths_dec, lower_q, axis=0)
    q_med  = np.quantile(vol_paths_dec, 0.50,    axis=0)
    q_high = np.quantile(vol_paths_dec, upper_q, axis=0)

    return {"q_low": q_low, "q_median": q_med, "q_high": q_high}


def to_annualized_percent(vol_daily_dec: np.ndarray) -> np.ndarray:
    return (vol_daily_dec * np.sqrt(TRADING_DAYS_PER_YEAR)) * 100.0

def make_forecast_index(last_dt: pd.Timestamp, h: int) -> pd.DatetimeIndex:
    start_next = last_dt + US_BUSINESS_DAY
    return pd.date_range(start=start_next, periods=h, freq=US_BUSINESS_DAY)

# ---------------------------
# Enhanced Plotting Functions
# ---------------------------

def create_main_vol_figure(vol_hist, idx_fc, vol_q_low, vol_q_med, vol_q_high, ticker):
    fig = go.Figure()
    
    # Configuration du thème moderne
    fig.add_trace(go.Scatter(
        x=idx_fc, y=vol_q_high, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    
    fig.add_trace(go.Scatter(
        x=idx_fc, y=vol_q_low, mode="lines",
        fill="tonexty", 
        line=dict(width=0),
        fillcolor="rgba(102, 126, 234, 0.15)",
        name=f"Intervalle de Confiance {CONF_INTERVAL*100:.0f}%",
        hoverlabel=dict(bgcolor="rgba(102, 126, 234, 0.8)", font_color="white")
    ))
    
    fig.add_trace(go.Scatter(
        x=vol_hist.index, y=vol_hist.values, 
        mode="lines",
        name="Volatilité Conditionnelle",
        line=dict(color="#764ba2", width=2.5),
        hoverlabel=dict(bgcolor="#764ba2", font_color="white")
    ))
    
    fig.add_trace(go.Scatter(
        x=idx_fc, y=vol_q_med,
        mode="lines+markers",
        name="Prévision (Médiane MC)",
        line=dict(color="#FF6B6B", width=2.5, dash="dot"),
        marker=dict(size=4, color="#FF6B6B"),
        hoverlabel=dict(bgcolor="#FF6B6B", font_color="white")
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>Prévision de Volatilité pour {ticker}</b>",
            font=dict(size=22, color="#2c3e50")
        ),
        paper_bgcolor="rgba(255, 255, 255, 0)",
        plot_bgcolor="rgba(250, 250, 252, 0.5)",
        margin=dict(l=60, r=30, t=80, b=60),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#e0e0e0",
            borderwidth=1,
            font=dict(size=11)
        ),
        xaxis=dict(
            title="Date",
            title_font=dict(size=14, color="#666"),
            gridcolor="rgba(200, 200, 200, 0.3)",
            showline=True,
            linewidth=1,
            linecolor="rgba(200, 200, 200, 0.5)",
            zeroline=False
        ),
        yaxis=dict(
            title="Volatilité Annualisée (%)",
            title_font=dict(size=14, color="#666"),
            gridcolor="rgba(200, 200, 200, 0.3)",
            showline=True,
            linewidth=1,
            linecolor="rgba(200, 200, 200, 0.5)",
            zeroline=False
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter"
        ),
        transition=dict(duration=500),
        autosize=True,
        height=500
    )
    
    return fig

def create_diagnostic_figures(res):
    std_resid = res.std_resid.dropna()

    # --- QQ-plot : t standardisée (variance 1) ---
    nu = float(res.params['nu'])
    t_dist = sp.stats.t(df=nu, scale=np.sqrt((nu - 2.0) / nu))
    osm, osr = sp.stats.probplot(std_resid, dist=t_dist, fit=False)

    qq_fig = go.Figure()
    qq_fig.add_trace(go.Scatter(
        x=osm, y=osr,
        mode='markers',
        name='Quantiles Observés',
        marker=dict(
            size=8,
            color=osr,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Valeur",
                thickness=15,
                len=0.7,
                x=1.02
            ),
            line=dict(width=0.5, color='white')
        ),
        hoverlabel=dict(bgcolor="rgba(102, 126, 234, 0.8)", font_color="white")
    ))
    
    line_x = np.array([np.min(osm), np.max(osm)])
    qq_fig.add_trace(go.Scatter(
        x=line_x, y=line_x,
        mode='lines',
        line=dict(color='#FF6B6B', width=2, dash='dash'),
        name='Référence Théorique'
    ))
    
    qq_fig.update_layout(
        title=dict(
            text="<b>QQ-Plot des Résidus</b>",
            font=dict(size=18, color="#2c3e50")
        ),
        xaxis_title="Quantiles Théoriques (Student-t)",
        yaxis_title="Quantiles Empiriques",
        paper_bgcolor="rgba(255, 255, 255, 0)",
        plot_bgcolor="rgba(250, 250, 252, 0.5)",
        height=400,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#e0e0e0",
            borderwidth=1
        )
    )

    # ACF avec style moderne
    acf_vals = acf(std_resid**2, nlags=30, fft=True)
    lags = np.arange(len(acf_vals))
    conf_level = 1.96 / np.sqrt(len(std_resid))

    acf_fig = go.Figure()
    
    colors = ['#FF6B6B' if abs(val) > conf_level else '#667eea' for val in acf_vals[1:]]
    
    acf_fig.add_trace(go.Bar(
        x=lags[1:],
        y=acf_vals[1:],
        name='Autocorrélation',
        marker=dict(
            color=colors,
            line=dict(width=0)
        ),
        hoverlabel=dict(bgcolor="rgba(102, 126, 234, 0.8)", font_color="white")
    ))
    
    acf_fig.add_shape(
        type="line",
        x0=0.5, y0=conf_level,
        x1=len(lags)-0.5, y1=conf_level,
        line=dict(color="#4ECDC4", dash="dash", width=2),
    )
    acf_fig.add_shape(
        type="line",
        x0=0.5, y0=-conf_level,
        x1=len(lags)-0.5, y1=-conf_level,
        line=dict(color="#4ECDC4", dash="dash", width=2),
    )
    
    acf_fig.update_layout(
        title=dict(
            text="<b>ACF des Résidus au Carré</b>",
            font=dict(size=18, color="#2c3e50")
        ),
        xaxis_title="Lag",
        yaxis_title="Autocorrélation",
        paper_bgcolor="rgba(255, 255, 255, 0)",
        plot_bgcolor="rgba(250, 250, 252, 0.5)",
        height=400,
        showlegend=True
    )

    return qq_fig, acf_fig

# ---------------------------
# Dash App Layout with Fixed Date Inputs
# ---------------------------

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://use.fontawesome.com/releases/v6.0.0/css/all.css"
    ],
)
app.title = "GARCH Volatility Dashboard"

# Injection du CSS personnalisé
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>''' + CUSTOM_CSS + '''</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Controls avec inputs date HTML5 natifs (pas de conflits CSS)
controls = html.Div([
    dbc.Row([
        dbc.Col([
            html.Label("📈 Ticker", className="fw-bold mb-2", style={'color': '#666'}),
            dcc.Input(
                id="ticker",
                value=DEFAULT_TICKER,
                debounce=True,
                className="input-modern"
            )
        ], width=12, md=3, className="mb-3"),
        
        dbc.Col([
            html.Label("📅 Période d'Analyse", className="fw-bold mb-2", style={'color': '#666'}),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label("Date de début:", className="date-input-label"),
                        dbc.Input(
                            id="start-date",
                            type="date",
                            value=(date.today() - timedelta(days=365*5)).isoformat(),
                            min="1980-01-01",
                            max=date.today().isoformat(),
                            className="input-modern"
                        )
                    ], width=6),
                    dbc.Col([
                        html.Label("Date de fin:", className="date-input-label"),
                        dbc.Input(
                            id="end-date",
                            type="date",
                            value=date.today().isoformat(),
                            min="1980-01-01",
                            max=date.today().isoformat(),
                            className="input-modern"
                        )
                    ], width=6),
                ])
            ], className="date-section")
        ], width=12, md=4, className="mb-3"),
        
        dbc.Col([
            html.Label("🎯 Horizon (jours)", className="fw-bold mb-2", style={'color': '#666'}),
            dcc.Input(
                id="horizon",
                type="number",
                value=DEFAULT_HORIZON,
                min=1,
                max=252,
                step=1,
                className="input-modern"
            )
        ], width=12, md=2, className="mb-3"),
        
        dbc.Col([
            html.Label("🎲 Simulations", className="fw-bold mb-2", style={'color': '#666'}),
            dcc.Input(
                id="nsims",
                type="number",
                value=DEFAULT_SIMS,
                min=500,
                max=20000,
                step=500,
                className="input-modern"
            )
        ], width=12, md=3, className="mb-3"),
    ])
], className="control-card")

app.layout = html.Div([
    dbc.Container([
        html.Div([
            html.H1("GARCH Volatility Forecast", className="header-title"),
            html.P("Analyse avancée de la volatilité avec modélisation GARCH(1,1) et intervalles Monte Carlo", 
                   className="header-subtitle"),
            
            controls,
            
            dbc.Row([
                dbc.Col([
                    html.Button(
                        [html.I(className="fas fa-chart-line me-2"), "Lancer l'Analyse"],
                        id="run",
                        n_clicks=0,
                        className="btn-primary-gradient w-100"
                    )
                ], width=12)
            ], className="mb-4"),
            
            html.Hr(style={'borderColor': '#e0e0e0', 'opacity': '0.3'}),
            
            dcc.Loading(
                id="loading-container",
                type="dot",
                color="#6C63FF",
                children=[
                    dbc.Alert(
                        id="error-alert",
                        is_open=False,
                        duration=10000,
                        color="danger",
                        className="alert-custom"
                    ),
                    html.Div(id="model-stats"),
                    
                    dcc.Tabs(
                        id="tabs",
                        className="mb-3",
                        children=[
                            dcc.Tab(
                                label='Prévision de Volatilité',
                                className="tab-custom",
                                selected_className="tab--selected",
                                children=[
                                    html.Div([
                                        dcc.Graph(id="vol-graph", config={'displayModeBar': False})
                                    ], style={'marginTop': '20px'})
                                ]
                            ),
                            dcc.Tab(
                                label='Diagnostics du Modèle',
                                className="tab-custom",
                                selected_className="tab--selected",
                                children=[
                                    html.Div([
                                        dbc.Row([
                                            dbc.Col([
                                                dcc.Graph(id="qq-plot", config={'displayModeBar': False})
                                            ], width=12, lg=6),
                                            dbc.Col([
                                                dcc.Graph(id="acf-plot", config={'displayModeBar': False})
                                            ], width=12, lg=6),
                                        ])
                                    ], style={'marginTop': '20px'})
                                ]
                            ),
                        ],
                    ),
                ],
            ),
        ], className="main-container")
    ], fluid=True)
], style={'minHeight': '100vh', 'background': THEME_COLORS['gradient1']})

# ---------------------------
# Callback (Updated with new date inputs)
# ---------------------------

@app.callback(
    [
        Output("vol-graph", "figure"),
        Output("qq-plot", "figure"),
        Output("acf-plot", "figure"),
        Output("model-stats", "children"),
        Output("error-alert", "children"),
        Output("error-alert", "is_open"),
    ],
    Input("run", "n_clicks"),
    [
        State("ticker", "value"),
        State("start-date", "value"),  # Changé
        State("end-date", "value"),     # Changé
        State("horizon", "value"),
        State("nsims", "value")
    ],
    prevent_initial_call=True,
)
def run_pipeline(n_clicks, ticker, start_date, end_date, h, n_sims):
    try:
        # 1) Data
        ticker_upper = ticker.strip().upper()
        df = fetch_prices(ticker_upper, start_date, end_date)
        r = compute_log_returns(df["Close"])

        # 2) Fit
        res = fit_garch_t(r)

        # 3) Hist vol
        sig_hist_dec = res.conditional_volatility.reindex(r.index) / 100.0
        vol_hist = pd.Series(to_annualized_percent(sig_hist_dec.values), index=r.index, name="Hist Ann Vol %")

        # 4) Forecast index
        idx_fc = make_forecast_index(r.index[-1], h)

        # 5) Monte Carlo
        q_dec = simulate_garch_intervals(res, r=r, h=int(h), n_sims=int(n_sims))
        vol_q_low  = to_annualized_percent(q_dec["q_low"])
        vol_q_med  = to_annualized_percent(q_dec["q_median"])
        vol_q_high = to_annualized_percent(q_dec["q_high"])

        # 6) Figures
        main_fig = create_main_vol_figure(vol_hist, idx_fc, vol_q_low, vol_q_med, vol_q_high, ticker_upper)
        qq_fig, acf_fig = create_diagnostic_figures(res)

        # 7) Stats avec style moderne
        params = res.params
        alpha = float(params['alpha[1]'])
        beta  = float(params['beta[1]'])
        persistence = alpha + beta
        if persistence >= 1.0:
            print(f"Attention: IGARCH (alpha+beta={persistence:.4f} ≥ 1). "
                "La variance inconditionnelle n'existe pas ; interpréter les prévisions avec prudence.")
        if (params['omega'] <= 0) or (alpha < 0) or (beta < 0):
            print("Attention: paramètres hors domaine (ω>0, α≥0, β≥0).")
        
        stats_card = html.Div([
            html.Div([
                html.H5([
                    html.I(className="fas fa-chart-area me-2"),
                    "Résumé du Modèle GARCH(1,1)"
                ], className="stats-title"),
                
                html.Div([
                    html.Div([
                        html.Span("📊 Période: ", className="fw-bold"),
                        html.Span(f"{r.index[0].date()} → {r.index[-1].date()} ({len(r)} observations)")
                    ], className="stats-text mb-2"),
                    
                    html.Div([
                        html.Span("🎯 Distribution: ", className="fw-bold"),
                        html.Span(f"Student-t avec ν = {params['nu']:.2f} degrés de liberté")
                    ], className="stats-text mb-2"),
                    
                    html.Div([
                        html.Span("⚡ Persistance (α+β): ", className="fw-bold"),
                        html.Span(f"{persistence:.4f}"),
                        html.Span(" | ", style={'opacity': '0.6'}),
                        html.Span(f"ω = {params['omega']:.3e} | α = {alpha:.3f} | β = {beta:.3f}")
                    ], className="stats-text mb-2"),
                    
                    html.Div([
                        html.Span("📈 Performance: ", className="fw-bold"),
                        html.Span(f"Log-L = {res.loglikelihood:.2f} | AIC = {res.aic:.2f} | BIC = {res.bic:.2f}")
                    ], className="stats-text"),
                ])
            ], className="stats-card")
        ])

        return main_fig, qq_fig, acf_fig, stats_card, None, False

    except Exception as e:
        err = f"❌ Erreur : {type(e).__name__} - {e}"
        print(err)
        empty_fig = go.Figure().update_layout(
            title="Erreur lors du calcul",
            paper_bgcolor="rgba(255, 255, 255, 0)",
            plot_bgcolor="rgba(250, 250, 252, 0.5)"
        )
        return empty_fig, empty_fig, empty_fig, None, err, True


if __name__ == "__main__":
    app.run(debug=True, port=8056)
