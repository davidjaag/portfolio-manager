import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, date
import json
import warnings
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
warnings.filterwarnings('ignore')

# ── Global Plotly 3D-style template ──────────────────────────────────────────
import plotly.io as pio
pio.templates["egade"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="white",
        plot_bgcolor="#fafbfe",
        font=dict(family="DM Sans, sans-serif", color="#1a2332", size=12),
        colorway=["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#06b6d4","#84cc16","#f43f5e"],
        xaxis=dict(
            gridcolor="rgba(13,27,42,0.06)", gridwidth=1,
            linecolor="rgba(13,27,42,0.12)", tickcolor="rgba(13,27,42,0.12)",
            showgrid=True, zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(13,27,42,0.06)", gridwidth=1,
            linecolor="rgba(13,27,42,0.12)", tickcolor="rgba(13,27,42,0.12)",
            showgrid=True, zeroline=False,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(13,27,42,0.1)", borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=16, r=16, t=48, b=16),
    )
)
pio.templates.default = "egade"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS + JS 3D effects ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&display=swap');
:root{
    --navy:#0d1b2a;--navy2:#1a3a5c;--gold:#c9a227;--gold2:#e8c547;
    --slate:#2d4a6b;--ice:#e8f0f8;--fog:#f0f4f9;
    --text:#1a2332;--muted:#6b7c93;
    --green:#059669;--red:#ef4444;--amber:#d97706;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--text);}
h1,h2,h3{font-family:'DM Serif Display',serif;}

/* ── BACKGROUND dot grid ── */
.main,[data-testid="stAppViewContainer"]>.main{
    background-color:#edf2f8 !important;
    background-image:radial-gradient(rgba(13,27,42,0.065) 1px,transparent 1px) !important;
    background-size:26px 26px !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0d1b2a 0%,#112238 100%) !important;
    border-right:1px solid rgba(201,162,39,0.2) !important;
    box-shadow:6px 0 40px rgba(0,0,0,0.35) !important;
}
[data-testid="stSidebar"] *{color:#dce8f5 !important;}
[data-testid="stSidebar"] input{background:#fff !important;color:#1a2332 !important;-webkit-text-fill-color:#1a2332 !important;border:1px solid rgba(201,162,39,0.4) !important;border-radius:8px !important;}
[data-testid="stSidebar"] [data-baseweb="input"] input{color:#1a2332 !important;-webkit-text-fill-color:#1a2332 !important;}
[data-testid="stSidebar"] label{color:rgba(201,162,39,0.9) !important;font-size:11px !important;font-weight:600 !important;letter-spacing:.05em !important;}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] small{color:#9ab0c8 !important;}
[data-testid="stSidebar"] button[kind="secondary"]{background:rgba(201,162,39,0.15) !important;color:#c9a227 !important;border:1px solid rgba(201,162,39,0.3) !important;}
[data-testid="stSidebar"] hr{border-color:rgba(201,162,39,0.2) !important;}

/* ── SIDEBAR portfolio buttons — invisible overlay ── */
[data-testid="stSidebar"] [data-testid="stButton"] button{
    background:transparent !important;border:none !important;
    box-shadow:none !important;color:transparent !important;
    -webkit-text-fill-color:transparent !important;
    font-size:0 !important;height:44px !important;
    margin-top:-44px !important;padding:0 !important;
    cursor:pointer !important;width:100% !important;
    display:block !important;position:relative !important;z-index:10 !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover,
[data-testid="stSidebar"] [data-testid="stButton"] button:focus{
    background:transparent !important;border:none !important;
    box-shadow:none !important;outline:none !important;
}

/* ── KPI CARDS: pronounced 3D ── */
.kpi-card{
    background:#fff;
    border-radius:16px;
    padding:22px 26px;
    border-left:4px solid var(--gold);
    /* Multi-layer shadow = 3D depth illusion */
    box-shadow:
        0 1px 0 rgba(255,255,255,0.9) inset,
        0 -3px 0 rgba(0,0,0,0.09) inset,
        0 4px 0 0 rgba(201,162,39,0.25),
        0 8px 0 0 rgba(201,162,39,0.08),
        0 16px 40px rgba(13,27,42,0.14),
        0 4px 10px rgba(13,27,42,0.08);
    transition:transform .3s cubic-bezier(.22,.68,0,1.2), box-shadow .3s ease;
    position:relative; overflow:hidden;
    cursor:default;
}
.kpi-card:hover{
    transform:translateY(-8px) scale(1.015) !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.9) inset,
        0 -3px 0 rgba(0,0,0,0.09) inset,
        0 4px 0 0 rgba(201,162,39,0.3),
        0 8px 0 0 rgba(201,162,39,0.1),
        0 36px 72px rgba(13,27,42,0.2),
        0 8px 20px rgba(13,27,42,0.12) !important;
}
.kpi-card::before{content:'';position:absolute;top:-20px;right:-20px;width:100px;height:100px;border-radius:50%;background:radial-gradient(circle,rgba(201,162,39,0.1) 0%,transparent 70%);}
.kpi-label{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:6px;}
.kpi-value{font-family:'DM Serif Display',serif;font-size:28px;color:var(--navy2);line-height:1;}
.kpi-value.positive{color:var(--green);}.kpi-value.negative{color:var(--red);}
.kpi-sub{font-size:11px;color:var(--muted);margin-top:4px;}

/* ── SECTION TITLE ── */
.section-title{font-family:'DM Serif Display',serif;font-size:22px;color:var(--navy2);border-bottom:2px solid var(--gold);padding-bottom:8px;margin:24px 0 16px 0;}

/* ── BANNER ── */
.app-banner{
    background:linear-gradient(135deg,#0d1b2a 0%,#1a3a5c 55%,#2d4a6b 100%);
    border-radius:20px;padding:26px 34px;margin-bottom:28px;
    display:flex;align-items:center;justify-content:space-between;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.07) inset,
        0 -2px 0 rgba(0,0,0,0.4) inset,
        0 20px 60px rgba(13,27,42,0.35),
        0 6px 16px rgba(13,27,42,0.22);
    position:relative;overflow:hidden;
}
.app-banner::before{content:'';position:absolute;top:-80%;right:-5%;width:500px;height:500px;border-radius:50%;background:radial-gradient(circle,rgba(201,162,39,0.09) 0%,transparent 60%);pointer-events:none;}
.banner-title{font-family:'DM Serif Display',serif;font-size:28px;color:#fff;margin:0;}
.banner-sub{color:var(--gold2);font-size:13px;margin-top:4px;}
.banner-date{color:rgba(232,240,248,0.55);font-size:12px;}

/* ── PLOTLY CHART WRAPPERS — 3D lift via filter ── */
/* This targets the actual rendered iframe/div Streamlit uses */
div[data-testid="stPlotlyChart"]{
    filter: drop-shadow(0 4px 12px rgba(13,27,42,0.12)) drop-shadow(0 1px 3px rgba(13,27,42,0.07)) !important;
    transition: filter 0.4s ease, transform 0.4s ease !important;
    transform: translateY(0px) !important;
    border-radius: 16px !important;
    display: block !important;
}
div[data-testid="stPlotlyChart"]:hover{
    filter: drop-shadow(0 16px 40px rgba(13,27,42,0.2)) drop-shadow(0 4px 10px rgba(13,27,42,0.12)) !important;
    transform: translateY(-6px) !important;
}

/* Target the inner div too */
div[data-testid="stPlotlyChart"] > div{
    border-radius:16px !important;
    overflow:hidden !important;
}

/* ── ELEMENT CONTAINERS — subtle depth ── */
div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"]{
    gap: 16px !important;
}

/* ── TABLE ── */
.styled-table{width:100%;border-collapse:collapse;font-size:13px;}
.styled-table th{background:var(--navy2);color:#fff;padding:11px 14px;text-align:center;font-weight:600;font-size:11px;letter-spacing:.05em;text-transform:uppercase;}
.styled-table td{padding:9px 14px;text-align:center;border-bottom:1px solid #e8eef5;}
.styled-table tr:hover td{background:var(--ice);}
.styled-table tr:nth-child(even) td{background:#f8fafd;}

/* ── ORDER CARDS ── */
.order-card{border-radius:12px;padding:16px 20px;margin:8px 0;display:flex;align-items:center;gap:16px;}
.order-buy{background:linear-gradient(135deg,#d5f5e3,#e8f8f0);border-left:4px solid #059669;}
.order-sell{background:linear-gradient(135deg,#fadbd8,#fde8e6);border-left:4px solid #ef4444;}
.order-hold{background:linear-gradient(135deg,#e8f0f8,#eef4fb);border-left:4px solid #2d4a6b;}

/* ── BADGES ── */
.badge-ok{background:#d1fae5;color:#065f46;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;}
.badge-warn{background:#fef3c7;color:#92400e;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;}
.badge-danger{background:#fee2e2;color:#991b1b;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;}

/* ── LOGIN ── */
.user-pill{display:flex;align-items:center;gap:12px;padding:14px 16px;border-radius:12px;border:2px solid var(--ice);margin-bottom:10px;background:var(--fog);transition:border .2s,background .2s,transform .15s;}
.user-pill:hover{border-color:var(--gold);background:white;transform:translateX(3px);}
.user-avatar{width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:700;color:white;flex-shrink:0;}


/* ── CHART 3D WRAPPER ── */
.chart3d {
    background: #ffffff;
    border-radius: 18px;
    padding: 4px;
    margin-bottom: 12px;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.95) inset,
        0 -4px 0 rgba(0,0,0,0.08) inset,
        0 6px 0 rgba(13,27,42,0.05),
        0 20px 56px rgba(13,27,42,0.13),
        0 5px 14px rgba(13,27,42,0.08);
    transition: transform 0.35s cubic-bezier(.22,.68,0,1.2), box-shadow 0.35s ease;
    will-change: transform;
}
.chart3d:hover {
    transform: translateY(-8px) scale(1.005);
    box-shadow:
        0 1px 0 rgba(255,255,255,0.95) inset,
        0 -4px 0 rgba(0,0,0,0.08) inset,
        0 6px 0 rgba(13,27,42,0.07),
        0 40px 80px rgba(13,27,42,0.2),
        0 10px 28px rgba(13,27,42,0.12);
}
/* ── ANIMATIONS ── */
@keyframes fadeInUp{from{opacity:0;transform:translateY(18px);}to{opacity:1;transform:translateY(0);}}
@keyframes fadeIn{from{opacity:0;}to{opacity:1;}}
.anim-fadeinup{animation:fadeInUp .55s cubic-bezier(.22,.68,0,1.2) both;}
.anim-fadein{animation:fadeIn .45s ease both;}
.anim-delay-1{animation-delay:.07s;}.anim-delay-2{animation-delay:.14s;}
.anim-delay-3{animation-delay:.21s;}.anim-delay-4{animation-delay:.28s;}
.page-wrap{animation:fadeInUp .4s ease both;}
</style>

<script>
/* Inject 3D effects directly into Streamlit's DOM */
(function(){
  var css = document.createElement('style');
  css.innerHTML = [
    'div[data-testid="stPlotlyChart"]{',
    '  filter:drop-shadow(0 4px 14px rgba(13,27,42,0.13)) drop-shadow(0 1px 4px rgba(13,27,42,0.07)) !important;',
    '  transition:filter 0.4s ease,transform 0.4s cubic-bezier(.22,.68,0,1.2) !important;',
    '  transform:translateY(0) !important;',
    '  border-radius:16px !important;',
    '}',
    'div[data-testid="stPlotlyChart"]:hover{',
    '  filter:drop-shadow(0 18px 48px rgba(13,27,42,0.21)) drop-shadow(0 5px 14px rgba(13,27,42,0.13)) !important;',
    '  transform:translateY(-7px) !important;',
    '}',
  ].join('');
  document.head.appendChild(css);
})();
</script>
""", unsafe_allow_html=True)


# ── Multi-user config ─────────────────────────────────────────────────────────
# Cada usuario tiene: nombre, PIN (4 dígitos), avatar emoji, color,
# sheet_id propio, y lista de portafolios a los que tiene acceso.
# "portfolios": None  →  acceso a todos
# "portfolios": ["Tangencia"]  →  solo ese portafolio
USERS = {  # USERS_START
    "david": {
        "name":       "David Jassan",
        "pin":        "1234",
        "avatar":     "DJ",
        "color":      "#1a3a5c",
        "sheet_id":   "1eApNRcJSnqYYkUxK2uDWqUOwXh6lNkoOZ-zvzVSoKFw",
        "portfolios": None,         # None = acceso a todos
        "role":       "admin",
    },
    "ana": {
        "name":       "Ana García",
        "pin":        "2222",
        "avatar":     "AG",
        "color":      "#1a7a4a",
        "sheet_id":   "",
        "portfolios": ["Conservador", "Familiar"],
        "role":       "investor",
    },
    "carlos": {
        "name":       "Carlos López",
        "pin":        "3333",
        "avatar":     "CL",
        "color":      "#c0392b",
        "sheet_id":   "",
        "portfolios": ["Agresivo"],
        "role":       "investor",
    },
    "maria": {
        "name":       "María Torres",
        "pin":        "4444",
        "avatar":     "MT",
        "color":      "#c9a227",
        "sheet_id":   "",
        "portfolios": ["Tangencia", "Conservador"],
        "role":       "investor",
    },
}  # USERS_END

# ── Session state init ────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user"  not in st.session_state:
    st.session_state.current_user  = None
if "pin_attempt"   not in st.session_state:
    st.session_state.pin_attempt   = ""
if "selected_user_login" not in st.session_state:
    st.session_state.selected_user_login = None
if "login_error"   not in st.session_state:
    st.session_state.login_error   = False


# ── Google Sheets connection ──────────────────────────────────────────────────
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

@st.cache_resource
def get_gspread_client():
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        return gspread.authorize(creds)
    except Exception as e:
        return None

@st.cache_data(ttl=30)
def load_sheet(sheet_id, tab_name):
    client = get_gspread_client()
    if client is None:
        return None
    try:
        sh = client.open_by_key(sheet_id)
        ws = sh.worksheet(tab_name)
        data = ws.get_all_records()
        return pd.DataFrame(data) if data else pd.DataFrame()
    except:
        return None

def save_row(sheet_id, tab_name, row_data):
    client = get_gspread_client()
    if client is None:
        return False
    try:
        sh = client.open_by_key(sheet_id)
        ws = sh.worksheet(tab_name)
        ws.append_row(row_data)
        return True
    except:
        return False

# ── Demo data (used when Google Sheets not configured) ────────────────────────

# ── Markowitz optimizer — calcula pesos óptimos desde retornos históricos ─────
# Universo completo de activos analizados (con justificación)
UNIVERSE_FULL = {
    # ticker: (nombre, sector, beta, alpha, razon_inclusion, fue_seleccionado, portafolios)
    "NVDA":   ("NVIDIA Corp",          "Semiconductores · IA",          1.83, +0.38,
               "Mayor alpha del universo (+0.38). Líder indiscutible en GPUs para IA. "
               "Retorno anualizado 5a: ~180%. Alta volatilidad compensada por Sharpe >1.5.",
               True,  ["Tangencia", "Agresivo"]),
    "CAT":    ("Caterpillar Inc",       "Maquinaria Industrial",         1.07, +0.08,
               "Activo de valor con beta neutro (~1.07). Correlación baja con tech. "
               "Ciclo de infraestructura global y reshoring de manufactura USA lo impulsan.",
               True,  ["Tangencia", "Conservador", "Familiar"]),
    "NEM":    ("Newmont Corp",          "Minería de Oro",                0.88, +0.05,
               "Cobertura natural contra inflación y risk-off. Beta defensivo (0.88). "
               "Correlación negativa con renta variable en crisis. Ancla de estabilidad.",
               True,  ["Tangencia", "Conservador", "Familiar"]),
    "AAPL":   ("Apple Inc",             "Tecnología Consumer",           1.08, +0.12,
               "Liquidez máxima, dividendo creciente, recompras agresivas. "
               "Complementa NVDA reduciendo volatilidad tech sin perder alpha.",
               True,  ["Tangencia", "Conservador"]),
    "BATS":   ("British Am Tobacco",    "Tabaco · Defensivo",            0.62, -0.02,
               "Beta más bajo del universo (0.62). Dividendo ~9% anual. "
               "Activo anticíclico: sube en correcciones. Ideal para portafolios defensivos.",
               True,  ["Conservador", "Familiar"]),
    "TSLA":   ("Tesla Inc",             "Vehículos Eléctricos",          1.65, +0.15,
               "Alto beta (1.65) con alpha positivo. Exposición a transición energética. "
               "Solo viable en portafolio Agresivo por volatilidad extrema (~65% anual).",
               True,  ["Agresivo"]),
    "ASTS":   ("AST SpaceMobile",       "Telecom Satelital",             1.43, -0.18,
               "Alpha negativo actual pero tesis de crecimiento exponencial (5G satelital). "
               "Seleccionado en Agresivo como posición especulativa con upside asimétrico.",
               True,  ["Agresivo"]),
    "NOW":    ("ServiceNow Inc",        "SaaS Enterprise",               1.27, +0.22,
               "Alpha sólido (+0.22) en SaaS enterprise. Crecimiento de ingresos >20% YoY. "
               "Menor volatilidad que NVDA con retorno comparable ajustado por riesgo.",
               True,  ["Agresivo"]),
    "JPM":    ("JPMorgan Chase",        "Banca · Financiero",            1.13, +0.06,
               "Mejor banco global por calidad de activos y ROE. Beta moderado (1.13). "
               "Beneficiario de tasas altas. Complementa metales preciosos en Familiar.",
               True,  ["Familiar"]),
    # Activos analizados pero NO seleccionados
    "AMZN":  ("Amazon.com Inc",         "E-Commerce · Cloud",           1.28, +0.09,
               "Alta correlación con NVDA y NOW (r>0.75). Agregar los tres reduciría "
               "diversificación. NOW ofrece Sharpe superior en segmento SaaS puro.",
               False, []),
    "MSFT":  ("Microsoft Corp",         "Software · Cloud",              0.92, +0.11,
               "Excelente activo por sí solo (beta bajo, alpha positivo). Descartado porque "
               "AAPL ya cubre el segmento tech defensivo con mayor eficiencia de Sharpe.",
               False, []),
    "GLD":   ("SPDR Gold ETF",          "Oro · ETF",                     0.82, +0.01,
               "Cobertura de oro pero con alpha casi cero (0.01 vs 0.05 de NEM). "
               "NEM ofrece misma cobertura más upside operativo en minería.",
               False, []),
    "XOM":   ("ExxonMobil Corp",        "Energía · Petróleo",            0.95, -0.03,
               "Alpha negativo (-0.03). Sector energía tradicional con riesgo regulatorio "
               "creciente. Sustituido por CAT que ofrece beta similar con alpha positivo.",
               False, []),
    "META":  ("Meta Platforms",         "Redes Sociales · IA",           1.52, +0.14,
               "Alpha positivo pero alta correlación con NVDA (r~0.68). Agregar META "
               "incrementaría riesgo idiosincrático tech sin mejorar Sharpe del portafolio.",
               False, []),
}

@st.cache_data(ttl=3600)
def load_price_history_from_sheets(sheet_id, portfolio_name):
    """
    Lee TODAS las filas de Precios_<portfolio_name> (datos Bloomberg limpios).
    Calcula retornos mensuales internamente: r_t = P_t/P_{t-1} - 1
    Devuelve pd.DataFrame con columnas = tickers, filas = retornos mensuales.
    No requiere ninguna hoja nueva — usa exactamente el tab Precios_* que ya existe.
    Acepta nombres Bloomberg (NVDA US Equity) o tickers cortos (NVDA).
    """
    client = get_gspread_client()
    if client is None:
        return None
    try:
        sh = client.open_by_key(sheet_id)
        # Mismo tab que ya usa load_prices_from_sheets para el precio spot
        for tab in [f"Precios_{portfolio_name}", "Precios"]:
            try:
                ws   = sh.worksheet(tab)
                data = ws.get_all_values()
                if len(data) < 3:   # necesitamos al menos 2 filas de precios
                    continue
                headers = data[0]
                rows    = data[1:]
                # Construir DataFrame de precios
                price_rows = []
                for row in rows:
                    vals = {}
                    for j, col in enumerate(headers):
                        if col.lower() in ("fecha", "date", ""):
                            continue
                        tkr = BLOOMBERG_TO_TICKER.get(col) or col.strip().upper()
                        try:
                            v = float(str(row[j]).replace(",", "."))
                            if v > 0:
                                vals[tkr] = v
                        except:
                            pass
                    if vals:
                        price_rows.append(vals)
                if len(price_rows) < 3:
                    continue
                prices_df = pd.DataFrame(price_rows).dropna()
                returns_df = np.log(prices_df / prices_df.shift(1)).dropna()
                if len(returns_df) >= 12:
                    return returns_df
            except:
                continue
        return None
    except:
        return None

def generate_demo_returns(tickers, n_months=60):
    """
    Fallback: retornos sintéticos cuando no hay Sheet conectado.
    Usa betas y alphas calibrados para mantener estructura realista.
    """
    np.random.seed(42)
    _betas  = {"NVDA":1.83,"CAT":1.07,"NEM":0.88,"AAPL":1.08,"BATS":0.62,
               "TSLA":1.65,"ASTS":1.43,"NOW":1.27,"JPM":1.13,"9988HK":1.42}
    _alphas = {"NVDA":0.38/12,"CAT":0.08/12,"NEM":0.05/12,"AAPL":0.12/12,
               "BATS":-0.02/12,"TSLA":0.15/12,"ASTS":-0.18/12,"NOW":0.22/12,
               "JPM":0.06/12,"9988HK":-0.09/12}
    mkt = np.random.normal(0.008, 0.04, n_months)
    return pd.DataFrame({
        t: _alphas.get(t,0) + _betas.get(t,1.0)*mkt + np.random.normal(0,0.03,n_months)
        for t in tickers
    })

def markowitz_optimize(returns_df, rf=0.0264/12):
    """
    Optimización de Markowitz via scipy.optimize.minimize.
    Returns dict with keys:
      tangencia  → pesos máximo Sharpe
      min_var    → pesos mínima varianza
      frontier   → lista de (vol, ret, weights) en la frontera
      cov_matrix → matriz de covarianza (annualizada)
      corr_matrix→ matriz de correlación
      exp_rets   → retornos esperados anualizados por activo
      sharpe_tangencia → Sharpe del portafolio tangencia
    """
    from scipy.optimize import minimize

    tickers = list(returns_df.columns)
    n       = len(tickers)
    mu      = returns_df.mean().values          # retorno medio mensual
    cov     = returns_df.cov().values            # covarianza mensual
    cov_ann = cov * 12
    mu_ann  = mu  * 12
    rf_ann  = rf  * 12

    # ── Portafolio Tangencia (máximo Sharpe) ──────────────────────────────────
    def neg_sharpe(w):
        ret  = np.dot(w, mu_ann)
        vol  = np.sqrt(w @ cov_ann @ w)
        return -(ret - rf_ann) / vol if vol > 1e-9 else 0

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds      = [(0, 1)] * n
    w0          = np.ones(n) / n

    res_tang = minimize(neg_sharpe, w0, method="SLSQP",
                        bounds=bounds, constraints=constraints,
                        options={"ftol":1e-10, "maxiter":1000})
    w_tang = res_tang.x
    w_tang = np.clip(w_tang, 0, 1)
    w_tang /= w_tang.sum()

    # ── Mínima Varianza ───────────────────────────────────────────────────────
    def port_var(w):
        return w @ cov_ann @ w

    res_mv = minimize(port_var, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol":1e-10, "maxiter":1000})
    w_mv = np.clip(res_mv.x, 0, 1)
    w_mv /= w_mv.sum()

    # ── Frontera eficiente (100 puntos) ──────────────────────────────────────
    ret_min = float(np.dot(w_mv, mu_ann))
    ret_max = float(np.max(mu_ann)) * 0.98
    frontier = []
    for target_r in np.linspace(ret_min, ret_max, 80):
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, r=target_r: np.dot(w, mu_ann) - r},
        ]
        res = minimize(port_var, w0, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"ftol":1e-9, "maxiter":500})
        if res.success:
            wf = np.clip(res.x, 0, 1)
            wf /= wf.sum()
            vol_f = float(np.sqrt(wf @ cov_ann @ wf))
            frontier.append((vol_f*100, float(np.dot(wf, mu_ann))*100,
                             {t: float(wf[i]) for i,t in enumerate(tickers)}))

    # ── Sharpe tangencia ──────────────────────────────────────────────────────
    ret_tang = float(np.dot(w_tang, mu_ann))
    vol_tang = float(np.sqrt(w_tang @ cov_ann @ w_tang))
    sr_tang  = (ret_tang - rf_ann) / vol_tang if vol_tang > 1e-9 else 0

    # ── Correlación ──────────────────────────────────────────────────────────
    std_v    = np.sqrt(np.diag(cov_ann))
    corr_ann = cov_ann / np.outer(std_v, std_v)
    np.fill_diagonal(corr_ann, 1.0)

    return {
        "tickers":          tickers,
        "tangencia":        {t: float(w_tang[i]) for i,t in enumerate(tickers)},
        "min_var":          {t: float(w_mv[i])   for i,t in enumerate(tickers)},
        "frontier":         frontier,
        "cov_matrix":       cov_ann,
        "corr_matrix":      corr_ann,
        "exp_rets":         {t: float(mu_ann[i]) for i,t in enumerate(tickers)},
        "sharpe_tangencia": sr_tang,
        "vol_tang":         vol_tang,
        "ret_tang":         ret_tang,
        "vol_minvar":       float(np.sqrt(w_mv @ cov_ann @ w_mv)),
        "ret_minvar":       float(np.dot(w_mv, mu_ann)),
    }

def get_demo_portfolios():
    """
    Cada portafolio define su UNIVERSO completo de activos (los que Bloomberg entregó).
    El optimizador Markowitz corre sobre ese universo y determina los pesos.
    Algunos activos recibirán peso 0 — el modelo los descarta automáticamente.
    'target' son los pesos MATLAB como semilla inicial; se sobreescriben en runtime.
    """
    return {
        "Tangencia": {
            "description": "Máximo Sharpe Ratio",
            "color": "#f59e0b",
            "rf": 0.0264,
            # Universo Bloomberg portafolio 1
            "universe": ["NVDA", "CAT", "NEM", "AAPL"],
            # Pesos MATLAB como referencia inicial (sobreescritos por optimizer)
            "target":   {"NVDA": 0.50, "CAT": 0.24, "NEM": 0.24, "AAPL": 0.02},
        },
        "Conservador": {
            "description": "Portafolio Tangencia",
            "color": "#10b981",
            "rf": 0.0264,
            # Universo Bloomberg portafolio 2 (11 activos — MATLAB entregó 4 con peso>0)
            "universe": ["AAPL", "ASTS", "JPM", "CIEN", "CAT", "NOW",
                         "NVDA", "NEM", "TSLA", "EXPE", "SPX"],
            # Pesos MATLAB resultado portafolio 2
            "target":   {"NVDA": 0.4999, "CAT": 0.2407, "NEM": 0.2466, "AAPL": 0.0128},
        },
        "Agresivo": {
            "description": "Alto Crecimiento",
            "color": "#ef4444",
            "rf": 0.0264,
            # Universo Bloomberg portafolio 3 (actualizar con tus activos reales)
            "universe": ["NVDA", "ASTS", "TSLA", "NOW"],
            "target":   {"NVDA": 0.60, "ASTS": 0.20, "TSLA": 0.15, "NOW": 0.05},
        },
        "Familiar": {
            "description": "Preservación de Capital",
            "color": "#2d4a6b",
            "rf": 0.0264,
            # Universo Bloomberg portafolio 4 (actualizar con tus activos reales)
            "universe": ["NEM", "BATS", "JPM", "CAT"],
            "target":   {"NEM": 0.35, "BATS": 0.35, "JPM": 0.20, "CAT": 0.10},
        },
    }

def get_user_portfolios(username):
    """Return list of portfolio names accessible to this user."""
    user    = USERS.get(username, {})
    allowed = user.get("portfolios")
    all_ports = list(get_demo_portfolios().keys())
    return all_ports if allowed is None else [p for p in allowed if p in all_ports]

def get_demo_operations(portfolio_name):
    # Tangencia: NVDA 50%, CAT 24%, NEM 24%, AAPL 2%  — ~$50k MXN inicial
    # Conservador: NEM 40%, BATS 30%, CAT 20%, AAPL 10%
    # Agresivo: NVDA 60%, ASTS 20%, TSLA 15%, NOW 5%
    # Familiar: NEM 35%, BATS 35%, JPM 20%, CAT 10%
    ops = {
        "Tangencia": [
            {"Fecha": "2025-01-15", "Ticker": "NVDA", "Tipo": "Compra", "Cantidad": 9.0,  "Precio_USD": 135.20, "Comision_USD": 0.50, "TC_MXN": 20.60},
            {"Fecha": "2025-01-15", "Ticker": "CAT",  "Tipo": "Compra", "Cantidad": 3.0,  "Precio_USD": 358.40, "Comision_USD": 0.50, "TC_MXN": 20.60},
            {"Fecha": "2025-01-15", "Ticker": "NEM",  "Tipo": "Compra", "Cantidad": 27.0, "Precio_USD": 44.80,  "Comision_USD": 0.50, "TC_MXN": 20.60},
            {"Fecha": "2025-01-15", "Ticker": "AAPL", "Tipo": "Compra", "Cantidad": 1.0,  "Precio_USD": 229.00, "Comision_USD": 0.50, "TC_MXN": 20.60},
            {"Fecha": "2025-06-01", "Ticker": "NVDA", "Tipo": "Compra", "Cantidad": 3.0,  "Precio_USD": 131.50, "Comision_USD": 0.50, "TC_MXN": 19.80},
            {"Fecha": "2025-06-01", "Ticker": "NEM",  "Tipo": "Compra", "Cantidad": 8.0,  "Precio_USD": 52.30,  "Comision_USD": 0.50, "TC_MXN": 19.80},
            {"Fecha": "2025-09-01", "Ticker": "CAT",  "Tipo": "Compra", "Cantidad": 1.0,  "Precio_USD": 390.20, "Comision_USD": 0.50, "TC_MXN": 20.10},
            {"Fecha": "2025-12-01", "Ticker": "NVDA", "Tipo": "Venta",  "Cantidad": 2.0,  "Precio_USD": 138.00, "Comision_USD": 0.50, "TC_MXN": 20.40},
        ],
        "Conservador": [],
        "Agresivo": [
            {"Fecha": "2025-01-20", "Ticker": "NVDA", "Tipo": "Compra", "Cantidad": 12.0, "Precio_USD": 136.00, "Comision_USD": 0.50, "TC_MXN": 20.15},
            {"Fecha": "2025-01-20", "Ticker": "TSLA", "Tipo": "Compra", "Cantidad": 4.0,  "Precio_USD": 380.00, "Comision_USD": 0.50, "TC_MXN": 20.15},
            {"Fecha": "2025-01-20", "Ticker": "ASTS", "Tipo": "Compra", "Cantidad": 30.0, "Precio_USD": 15.50,  "Comision_USD": 0.50, "TC_MXN": 20.15},
            {"Fecha": "2025-01-20", "Ticker": "NOW",  "Tipo": "Compra", "Cantidad": 1.0,  "Precio_USD": 950.00, "Comision_USD": 0.50, "TC_MXN": 20.15},
            {"Fecha": "2025-07-01", "Ticker": "NVDA", "Tipo": "Compra", "Cantidad": 4.0,  "Precio_USD": 128.00, "Comision_USD": 0.50, "TC_MXN": 19.70},
            {"Fecha": "2025-10-01", "Ticker": "TSLA", "Tipo": "Venta",  "Cantidad": 2.0,  "Precio_USD": 420.00, "Comision_USD": 0.50, "TC_MXN": 20.20},
        ],
        "Familiar": [
            {"Fecha": "2025-01-10", "Ticker": "NEM",  "Tipo": "Compra", "Cantidad": 30.0, "Precio_USD": 41.00,  "Comision_USD": 0.50, "TC_MXN": 20.05},
            {"Fecha": "2025-01-10", "Ticker": "BATS", "Tipo": "Compra", "Cantidad": 25.0, "Precio_USD": 34.80,  "Comision_USD": 0.50, "TC_MXN": 20.05},
            {"Fecha": "2025-01-10", "Ticker": "JPM",  "Tipo": "Compra", "Cantidad": 4.0,  "Precio_USD": 230.00, "Comision_USD": 0.50, "TC_MXN": 20.05},
            {"Fecha": "2025-01-10", "Ticker": "CAT",  "Tipo": "Compra", "Cantidad": 2.0,  "Precio_USD": 345.00, "Comision_USD": 0.50, "TC_MXN": 20.05},
            {"Fecha": "2025-05-01", "Ticker": "NEM",  "Tipo": "Compra", "Cantidad": 10.0, "Precio_USD": 50.20,  "Comision_USD": 0.50, "TC_MXN": 20.00},
            {"Fecha": "2025-05-01", "Ticker": "BATS", "Tipo": "Compra", "Cantidad": 8.0,  "Precio_USD": 35.90,  "Comision_USD": 0.50, "TC_MXN": 20.00},
            {"Fecha": "2025-11-01", "Ticker": "JPM",  "Tipo": "Compra", "Cantidad": 1.0,  "Precio_USD": 245.00, "Comision_USD": 0.50, "TC_MXN": 20.30},
        ]
    }
    return pd.DataFrame(ops.get(portfolio_name, []))

def get_demo_prices():
    return {
        "NVDA": 138.50, "CAT": 345.20, "NEM": 44.10, "AAPL": 228.30,
        "BATS": 36.80, "TSLA": 395.00, "ASTS": 18.50, "NOW": 985.00,
        "JPM": 238.00, "9988HK": 95.00, "SPX": 5720.00
    }

# ── Live quotes from Yahoo Finance ──────────────────────────────────────────
# Ticker map: app ticker → Yahoo Finance symbol
YF_TICKER_MAP = {
    "NVDA":   "NVDA",
    "CAT":    "CAT",
    "NEM":    "NEM",
    "AAPL":   "AAPL",
    "BATS":   "BATS.L",
    "TSLA":   "TSLA",
    "ASTS":   "ASTS",
    "NOW":    "NOW",
    "JPM":    "JPM",
    "9988HK": "9988.HK",
    "SPX":    "^GSPC",
}

@st.cache_data(ttl=300)  # refresca cada 5 minutos
def get_live_quotes(tickers):
    """Returns dict ticker → price, change_pct, week52 range, hist_7d, chg_7d_pct"""
    if not YF_AVAILABLE:
        return {}
    results = {}
    for tkr in tickers:
        yf_sym = YF_TICKER_MAP.get(tkr, tkr)
        try:
            ticker_obj = yf.Ticker(yf_sym)
            info       = ticker_obj.fast_info
            price      = getattr(info, "last_price",    None)
            open_p     = getattr(info, "open",           None)
            prev_close = getattr(info, "previous_close", None)
            day_high   = getattr(info, "day_high",       None)
            day_low    = getattr(info, "day_low",        None)
            w52_high   = getattr(info, "year_high",      None)
            w52_low    = getattr(info, "year_low",       None)
            chg_pct    = (price / prev_close - 1) * 100 if price and prev_close else None
            ma52w      = ((w52_high or 0) + (w52_low or 0)) / 2 if w52_high and w52_low else None
            vs_ma52w   = (price / ma52w - 1) * 100 if price and ma52w else None
            hist_7d = []; chg_7d_pct = None
            try:
                hist = ticker_obj.history(period="10d", interval="1d")
                if not hist.empty and "Close" in hist.columns:
                    closes = hist["Close"].dropna().tolist()[-7:]
                    hist_7d = closes
                    if len(closes) >= 2:
                        chg_7d_pct = (closes[-1] / closes[0] - 1) * 100
            except:
                pass
            results[tkr] = {
                "price": price, "open": open_p, "prev_close": prev_close,
                "day_high": day_high, "day_low": day_low, "change_pct": chg_pct,
                "week52_high": w52_high, "week52_low": w52_low,
                "ma52w": ma52w, "vs_ma52w": vs_ma52w,
                "hist_7d": hist_7d, "chg_7d_pct": chg_7d_pct,
            }
        except:
            results[tkr] = {}
    return results

# Mapeo de nombres de columna Bloomberg → ticker corto
BLOOMBERG_TO_TICKER = {
    "AAPL US Equity": "AAPL",
    "ASTS US Equity": "ASTS",
    "JPM US Equity":  "JPM",
    "9988 HK Equity": "9988HK",
    "CAT US Equity":  "CAT",
    "NOW US Equity":  "NOW",
    "NVDA US Equity": "NVDA",
    "NEM US Equity":  "NEM",
    "TSLA US Equity": "TSLA",
    "BATS LN Equity": "BATS",
    "CIEN US Equity": "CIEN",
    "EXPE US Equity": "EXPE",
    "EXPE US":        "EXPE",   # alternate Bloomberg format
    "SPX Index":      "SPX",
    "SPX":            "SPX",
}

@st.cache_data(ttl=300)
def load_prices_from_sheets(sheet_id, portfolio_name=None):
    """
    Lee la última fila de Precios_<portfolio_name> (ej. Precios_Tangencia).
    Si no existe ese tab, intenta el tab genérico 'Precios' como fallback.
    Devuelve dict ticker→precio o None.
    Las columnas pueden ser tickers cortos (NVDA, CAT) o nombres Bloomberg
    (NVDA US Equity) — se acepta ambos via BLOOMBERG_TO_TICKER.
    """
    client = get_gspread_client()
    if client is None:
        return None
    try:
        sh = client.open_by_key(sheet_id)
        # Determinar qué tab leer
        tab_names_to_try = []
        if portfolio_name:
            tab_names_to_try.append(f"Precios_{portfolio_name}")
        tab_names_to_try.append("Precios")  # fallback genérico

        ws = None
        used_tab = None
        for tab in tab_names_to_try:
            try:
                ws = sh.worksheet(tab)
                used_tab = tab
                break
            except:
                continue
        if ws is None:
            return None

        data = ws.get_all_values()
        if len(data) < 2:
            return None

        headers  = data[0]
        last_row = data[-1]   # fila más reciente = precio actual
        prices = {}
        for col, val in zip(headers, last_row):
            if not val or col.lower() in ("fecha", "date", ""):
                continue
            # Acepta nombre Bloomberg O ticker corto directamente
            ticker = BLOOMBERG_TO_TICKER.get(col) or col.strip().upper()
            try:
                prices[ticker] = float(str(val).replace(",", "."))
            except:
                pass
        return prices if prices else None
    except Exception as e:
        return None

def load_asset_data_from_sheets(sheet_id):
    """
    Lee tab 'Activos' del Sheet.
    Columnas requeridas: Ticker | PE_actual | PE_promedio | Beta | Alpha | Descripcion
    Devuelve dict: { "NVDA": (pe_cur, pe_avg, beta, alpha, desc), ... }
    Si la columna no existe o el valor está vacío, usa None.
    """
    client = get_gspread_client()
    if client is None:
        return None
    try:
        sh  = client.open_by_key(sheet_id)
        ws  = sh.worksheet("Activos")
        rows = ws.get_all_records()
        if not rows:
            return None
        result = {}
        for row in rows:
            tkr = str(row.get("Ticker", "")).strip().upper()
            if not tkr:
                continue
            def _f(key):
                v = row.get(key, "")
                if v == "" or v is None:
                    return None
                try:
                    return float(str(v).replace(",", "."))
                except:
                    return None
            result[tkr] = (
                _f("PE_actual"),
                _f("PE_promedio"),
                _f("Beta")   if _f("Beta")  is not None else 1.0,
                _f("Alpha")  if _f("Alpha") is not None else 0.0,
                str(row.get("Descripcion", "")).strip(),
            )
        return result if result else None
    except Exception as e:
        return None





def write_users_to_file(users_dict):
    """Reescribe el bloque USERS entre marcadores en app.py."""
    import os, re
    app_path = os.path.abspath(__file__)
    with open(app_path, 'r', encoding='utf-8') as f:
        src = f.read()
    nl = '\n'
    parts = ['USERS = {  # USERS_START' + nl]
    for key, ud in users_dict.items():
        ports = ud.get('portfolios')
        ports_repr = 'None' if ports is None else repr(ports)
        parts.append('    ' + repr(key) + ': {' + nl)
        parts.append('        "name":       ' + repr(ud.get('name','')) + ',' + nl)
        parts.append('        "pin":        ' + repr(ud.get('pin','0000')) + ',' + nl)
        parts.append('        "avatar":     ' + repr(ud.get('avatar','??')) + ',' + nl)
        parts.append('        "color":      ' + repr(ud.get('color','#1a3a5c')) + ',' + nl)
        parts.append('        "sheet_id":   "",' + nl)
        parts.append('        "portfolios": ' + ports_repr + ',' + nl)
        parts.append('        "role":       ' + repr(ud.get('role','investor')) + ',' + nl)
        parts.append('    },' + nl)
    parts.append('}  # USERS_END' + nl)
    new_block = ''.join(parts)
    pattern = r'USERS = \{  # USERS_START\n.*?\}  # USERS_END\n'
    if re.search(pattern, src, re.DOTALL):
        new_src = re.sub(pattern, new_block, src, flags=re.DOTALL)
        with open(app_path, 'w', encoding='utf-8') as f:
            f.write(new_src)
        return True, str(len(users_dict)) + ' usuarios guardados en app.py'
    else:
        return False, 'Marcadores USERS_START/USERS_END no encontrados en app.py'



def get_demo_history():
    months = ["Ene 25","Feb 25","Mar 25","Abr 25","May 25","Jun 25",
              "Jul 25","Ago 25","Sep 25","Oct 25","Nov 25","Dic 25"]
    vals   = [100000, 103200, 105800, 102400, 108900, 111200,
              115600, 112300, 118700, 122100, 119800, 125400]
    return pd.DataFrame({"Mes": months, "Valor": vals})

def calc_history_from_prices(price_history_df, ops_df, tc):
    """
    Builds portfolio value history from Bloomberg price series + operations.
    price_history_df: DataFrame with columns = tickers, rows = dates (all historical prices)
    ops_df: operations DataFrame with Fecha, Ticker, Tipo, Cantidad, TC_MXN
    Returns DataFrame {"Mes": label, "Valor": MXN value} per period.
    This gives the REAL Sharpe, volatility and drawdown for each portfolio.
    """
    if price_history_df is None or price_history_df.empty or ops_df.empty:
        return None
    try:
        price_df = price_history_df.copy()
        # Parse ops dates
        ops = ops_df.copy()
        ops["Fecha"] = pd.to_datetime(ops["Fecha"], errors="coerce")
        ops = ops.dropna(subset=["Fecha"])

        history_rows = []
        for i, row in price_df.iterrows():
            row_date = pd.Timestamp(f"2020-01-01") + pd.DateOffset(months=i)
            # Accumulate shares held up to this period
            ops_to_date = ops[ops["Fecha"] <= row_date]
            if ops_to_date.empty:
                continue
            portfolio_value = 0.0
            for tkr in price_df.columns:
                tkr_ops   = ops_to_date[ops_to_date["Ticker"] == tkr]
                compras   = tkr_ops[tkr_ops["Tipo"] == "Compra"]["Cantidad"].sum()
                ventas    = tkr_ops[tkr_ops["Tipo"] == "Venta"]["Cantidad"].sum()
                qty       = compras - ventas
                price_now = row.get(tkr, 0)
                if qty > 0 and price_now > 0:
                    portfolio_value += qty * price_now
            if portfolio_value > 0:
                history_rows.append({
                    "Mes":   row_date.strftime("%b %y"),
                    "Valor": portfolio_value
                })
        if len(history_rows) >= 3:
            return pd.DataFrame(history_rows)
        return None
    except Exception:
        return None

# ── Portfolio calculations ────────────────────────────────────────────────────
def calc_positions(ops_df, prices, tc, target_tickers=None):
    """Calculate positions. If target_tickers provided, include all even with no ops."""
    # Start with tickers that have operations
    op_tickers = list(ops_df["Ticker"].unique()) if not ops_df.empty else []
    # Add target tickers with zero position if not already present
    all_tickers = list(op_tickers)
    if target_tickers:
        for t in target_tickers:
            if t not in all_tickers:
                all_tickers.append(t)
    if not all_tickers:
        return pd.DataFrame()

    positions = []
    tickers = all_tickers
    
    for tkr in tickers:
        if not ops_df.empty and tkr in ops_df["Ticker"].values:
            tkr_ops = ops_df[ops_df["Ticker"] == tkr]
            compras = tkr_ops[tkr_ops["Tipo"] == "Compra"]
            ventas  = tkr_ops[tkr_ops["Tipo"] == "Venta"]
            qty_compra = compras["Cantidad"].sum()
            qty_venta  = ventas["Cantidad"].sum() if not ventas.empty else 0
            qty_total  = qty_compra - qty_venta
            if qty_compra > 0:
                costo_total = (compras["Cantidad"] * compras["Precio_USD"]).sum()
                costo_prom  = costo_total / qty_compra
            else:
                costo_prom = 0
        else:
            qty_total  = 0.0
            costo_prom = 0.0
        
        precio_actual = prices.get(tkr, 0)
        valor_usd     = qty_total * precio_actual
        valor_mxn     = valor_usd * tc
        ganancia_usd  = (precio_actual - costo_prom) * qty_total
        ganancia_pct  = (precio_actual / costo_prom - 1) if costo_prom > 0 else 0
        
        positions.append({
            "Ticker":       tkr,
            "Cantidad":     qty_total,
            "Precio Actual": precio_actual,
            "Costo Prom":   costo_prom,
            "Valor USD":    valor_usd,
            "Valor MXN":    valor_mxn,
            "Ganancia USD": ganancia_usd,
            "Ganancia %":   ganancia_pct,
        })
    
    return pd.DataFrame(positions)

def calc_alerts(positions_df, target_weights, tc, monthly_contrib,
                threshold_reduce=0.65, threshold_rebal=0.05):
    if positions_df.empty:
        return pd.DataFrame()
    total_mxn = positions_df["Valor MXN"].sum()
    df = positions_df.copy()
    if total_mxn == 0:
        df["Peso Actual"]=0.0;df["Peso Objetivo"]=df["Ticker"].map(target_weights).fillna(0.0)
        df["Desviación"]=-df["Peso Objetivo"];df["Alerta"]="OK"
        df["MXN Diferencia"]=0.0;df["Compra Sugerida MXN"]=0.0
        df["Acciones a Comprar"]=0.0;df["Monto a Vender MXN"]=0.0;df["Acciones a Vender"]=0.0
        return df
    df["Peso Actual"]   = df["Valor MXN"] / total_mxn
    df["Peso Objetivo"] = df["Ticker"].map(target_weights).fillna(0)
    df["Desviación"]    = df["Peso Actual"] - df["Peso Objetivo"]

    def get_alert(row):
        if row["Desviación"] > threshold_rebal:
            return "VENDER"
        elif row["Desviación"] < -threshold_rebal:
            return "COMPRAR"
        return "OK"
    df["Alerta"] = df.apply(get_alert, axis=1)

    # Exact MXN to reach target weight (positive = buy, negative = sell)
    df["MXN Diferencia"]      = df.apply(
        lambda r: r["Peso Objetivo"] * total_mxn - r["Valor MXN"], axis=1)
    df["Compra Sugerida MXN"] = df["MXN Diferencia"].clip(lower=0)
    df["Acciones a Comprar"]  = df.apply(
        lambda r: r["Compra Sugerida MXN"] / (r["Precio Actual"] * tc)
        if r["Precio Actual"] > 0 else 0, axis=1)
    df["Monto a Vender MXN"]  = df["MXN Diferencia"].clip(upper=0).abs()
    df["Acciones a Vender"]   = df.apply(
        lambda r: r["Monto a Vender MXN"] / (r["Precio Actual"] * tc)
        if r["Precio Actual"] > 0 else 0, axis=1)
    return df

def calc_risk_metrics_bloomberg(log_ret_df, weights, rf=0.0264):
    try:
        cols=[t for t in weights if t in log_ret_df.columns]
        if not cols: return {}
        w=np.array([weights[t] for t in cols]);w=w/w.sum()
        R=log_ret_df[cols].dropna().values
        if len(R)<12: return {}
        mu_ann=R.mean(axis=0)*12;cov_ann=np.cov(R.T,ddof=1)*12
        ret_p=float(w@mu_ann);vol_p=float(np.sqrt(w@cov_ann@w))
        sharpe=(ret_p-rf)/vol_p if vol_p>0 else 0
        pr=R@w;cum=np.exp(np.cumsum(pr));mx=np.maximum.accumulate(cum)
        max_dd=float(np.min((cum-mx)/mx));var_95=float(np.percentile(pr,5))
        mask=pr<var_95;cvar_95=float(pr[mask].mean()) if mask.any() else var_95
        roll_v=float(pd.Series(pr).rolling(6).std().iloc[-1]*np.sqrt(12)) if len(pr)>=6 else None
        return {"vol_annual":vol_p,"ret_annual":ret_p,"sharpe":sharpe,"max_dd":max_dd,
                "total_return":float(cum[-1]-1),"var_95":var_95,"cvar_95":cvar_95,
                "rolling_vol":roll_v,"n_months":len(pr)}
    except: return {}


def calc_risk_metrics(history_df, rf=0.0264):
    if history_df.empty or len(history_df) < 3:
        return {}
    
    vals = history_df["Valor"].values.astype(float)
    returns = np.diff(vals) / vals[:-1]
    
    if len(returns) < 2:
        return {}
    
    vol_annual   = np.std(returns) * np.sqrt(12)
    ret_annual   = np.mean(returns) * 12
    sharpe       = (ret_annual - rf) / vol_annual if vol_annual > 0 else 0
    max_dd       = np.min([(vals[i] - np.max(vals[:i+1])) / np.max(vals[:i+1]) 
                           for i in range(1, len(vals))])
    total_return = (vals[-1] / vals[0] - 1)
    
    # VaR 95%
    var_95  = np.percentile(returns, 5)
    cvar_95 = np.mean(returns[returns < var_95]) if any(returns < var_95) else var_95
    
    # Correlation rolling (need at least 6 points)
    rolling_corr = None
    if len(returns) >= 6:
        rolling_corr = pd.Series(returns).rolling(6).std().iloc[-1] * np.sqrt(12)
    
    return {
        "vol_annual":   vol_annual,
        "ret_annual":   ret_annual,
        "sharpe":       sharpe,
        "max_dd":       max_dd,
        "total_return": total_return,
        "var_95":       var_95,
        "cvar_95":      cvar_95,
        "rolling_vol":  rolling_corr,
        "n_months":     len(returns),
    }

def sharpe_label(sharpe):
    if sharpe > 1.5: return "⭐⭐⭐ Muy Fuerte", "#1a7a4a"
    if sharpe > 1.0: return "⭐⭐ Sólido",       "#2d7a4a"
    if sharpe > 0.5: return "⭐ Moderado",        "#d4820a"
    return "⚠ Débil",                              "#c0392b"

# ── Login screen ─────────────────────────────────────────────────────────────
if not st.session_state.authenticated:

    st.markdown("""
    <style>
    .stApp,[data-testid="stAppViewContainer"]{
        background:radial-gradient(ellipse at 30% 40%,#112240 0%,#0a1622 45%,#05080f 100%) !important;
        background-image:none !important;
    }
    .main{background:transparent !important;background-image:none !important;}
    [data-testid="stSidebar"]{display:none !important;}
    @keyframes floatOrb{0%,100%{transform:translateY(0) scale(1);opacity:.22;}50%{transform:translateY(-32px) scale(1.07);opacity:.32;}}
    .orb{position:fixed;border-radius:50%;pointer-events:none;z-index:0;animation:floatOrb 10s ease-in-out infinite;}
    .orb1{width:560px;height:560px;left:-140px;top:-140px;background:radial-gradient(circle,rgba(37,99,235,.28) 0%,transparent 68%);animation-delay:0s;}
    .orb2{width:400px;height:400px;right:-100px;bottom:0%;background:radial-gradient(circle,rgba(201,162,39,.2) 0%,transparent 65%);animation-delay:4s;}
    .orb3{width:250px;height:250px;left:38%;top:52%;background:radial-gradient(circle,rgba(139,92,246,.16) 0%,transparent 65%);animation-delay:7s;}
    @keyframes riseGlass{from{opacity:0;transform:translateY(40px) scale(.96);}to{opacity:1;transform:translateY(0) scale(1);}}
    /* Solid white card — fully readable against dark bg */
    .gpanel{
        background:#ffffff;
        border-radius:24px;
        box-shadow:0 32px 80px rgba(0,0,0,.55), 0 8px 24px rgba(0,0,0,.3),
                   0 1px 0 rgba(255,255,255,0.9) inset;
        animation:riseGlass 1s cubic-bezier(.22,.68,0,1.1) both;
        padding:36px 40px; position:relative; overflow:hidden; z-index:1;
        border-top: 4px solid #c9a227;
    }
    .gpanel::before{content:"";position:absolute;top:-40px;right:-40px;width:160px;height:160px;border-radius:50%;background:radial-gradient(circle,rgba(201,162,39,.08) 0%,transparent 70%);pointer-events:none;}
    @keyframes rowIn{from{opacity:0;transform:translateX(-14px);}to{opacity:1;transform:translateX(0);}}
    [data-testid="stAppViewContainer"] [data-testid="stButton"] > div > button{
        background:#f4f7fb !important;
        border:1.5px solid #e2e8f0 !important;
        border-radius:16px !important; color:#1a2332 !important;
        font-size:14px !important; font-weight:600 !important;
        text-align:left !important; height:auto !important;
        min-height:72px !important; padding:16px 20px !important;
        margin-bottom:12px !important; white-space:pre-line !important;
        line-height:1.55 !important;
        transition:background .22s,border-color .22s,transform .18s,box-shadow .18s !important;
        box-shadow:0 2px 8px rgba(13,27,42,0.06) !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stButton"] > div > button:hover{
        background:#ffffff !important;
        border-color:#c9a227 !important;
        transform:translateX(5px) !important;
        box-shadow:0 4px 20px rgba(201,162,39,.25), -3px 0 0 #c9a227 !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stButton"] > div > button p{
        color:#1a2332 !important;font-size:14px !important;
        font-weight:600 !important;margin:0 !important;white-space:pre-line !important;
    }
    [data-testid="stAppViewContainer"] .stTextInput input{
        background:#f4f7fb !important;
        border:1.5px solid #d1dae6 !important;
        border-radius:14px !important;color:#1a2332 !important;
        -webkit-text-fill-color:#1a2332 !important;
        font-size:26px !important;letter-spacing:16px !important;
        text-align:center !important;padding:18px !important;
    }
    [data-testid="stAppViewContainer"] .stTextInput input:focus{
        border-color:#c9a227 !important;
        box-shadow:0 0 0 3px rgba(201,162,39,.15) !important;outline:none !important;
    }
    [data-testid="stAppViewContainer"] .stTextInput label{display:none !important;}
    </style>
    <div class="orb orb1"></div><div class="orb orb2"></div><div class="orb orb3"></div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.3, 1])
    with col:
        logo_img = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyODAgODAiIHdpZHRoPSIyODAiIGhlaWdodD0iODAiPjxyZWN0IHg9IjAiIHk9IjAiIHdpZHRoPSI2IiBoZWlnaHQ9IjgwIiBmaWxsPSIjYzgxMDJlIi8+PHRleHQgeD0iMTgiIHk9IjUyIiBmb250LWZhbWlseT0iR2VvcmdpYSxzZXJpZiIgZm9udC1zaXplPSIzOCIgZm9udC13ZWlnaHQ9IjcwMCIgbGV0dGVyLXNwYWNpbmc9IjMiIGZpbGw9IiMxYTIzMzIiPkVHQURFPC90ZXh0Pjx0ZXh0IHg9IjE5IiB5PSI2OCIgZm9udC1mYW1pbHk9IkFyaWFsLHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTEiIGZvbnQtd2VpZ2h0PSI0MDAiIGxldHRlci1zcGFjaW5nPSIyLjUiIGZpbGw9IiM2YjdjOTMiPkJVU0lORVNTIFNDSE9PTDwvdGV4dD48dGV4dCB4PSIxOSIgeT0iMjAiIGZvbnQtZmFtaWx5PSJBcmlhbCxzYW5zLXNlcmlmIiBmb250LXNpemU9IjkiIGZvbnQtd2VpZ2h0PSI0MDAiIGxldHRlci1zcGFjaW5nPSIxLjUiIGZpbGw9IiNjODEwMmUiPlRFQyBERSBNT05URVJSRVk8L3RleHQ+PC9zdmc+"
        st.markdown(
            f'<div style="text-align:center;padding:52px 0 28px;position:relative;z-index:1;">'
            f'<div style="display:inline-block;background:rgba(255,255,255,0.97);'
            f'border-radius:18px;padding:22px 38px;'
            f'box-shadow:0 24px 64px rgba(0,0,0,.5),0 0 0 1px rgba(255,255,255,.1);">'
            f'<img src="{logo_img}" style="width:190px;display:block;"></div>'
            f'<div style="margin-top:16px;font-size:11px;color:rgba(201,162,39,.8);'
            f'letter-spacing:.18em;font-weight:700;text-transform:uppercase;">'
            f'Mercados Financieros · Portfolio Manager</div></div>',
            unsafe_allow_html=True
        )

        if st.session_state.selected_user_login is None:
            st.markdown("""
            <div class="gpanel">
                <div style="font-family:'DM Serif Display',serif;font-size:26px;
                            color:#0d1b2a;margin-bottom:6px;">Bienvenido</div>
                <div style="font-size:13px;color:#6b7c93;
                            margin-bottom:28px;letter-spacing:.02em;">
                    Selecciona tu perfil para continuar
                </div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            for ukey, udata in USERS.items():
                port_list = get_user_portfolios(ukey)
                port_str  = "  ·  ".join(port_list)
                label = udata['avatar'] + '  ' + udata['name'] + '\n' + port_str
                if st.button(label, key=f"user_btn_{ukey}", use_container_width=True):
                    st.session_state.selected_user_login = ukey
                    st.session_state.login_error = False
                    st.rerun()

        else:
            ukey  = st.session_state.selected_user_login
            udata = USERS[ukey]
            port_list = get_user_portfolios(ukey)
            uc = udata["color"]; ua = udata["avatar"]; un = udata["name"]
            port_joined = " &nbsp;·&nbsp; ".join(port_list)
            st.markdown(
                "<div class='gpanel'>"
                "<div style='display:flex;align-items:center;gap:18px;"
                "padding-bottom:22px;margin-bottom:22px;"
                "border-bottom:1px solid #e8eef5;'>"
                f"<div style='width:64px;height:64px;border-radius:50%;"
                f"background:{uc};display:flex;align-items:center;"
                f"justify-content:center;font-size:30px;flex-shrink:0;"
                f"box-shadow:0 6px 28px {uc}66;'>{ua}</div>"
                f"<div><div style='font-weight:700;font-size:20px;color:white;"
                f"font-family:DM Serif Display,serif;'>{un}</div>"
                f"<div style='font-size:12px;color:#6b7c93;margin-top:5px;'>"
                f"{port_joined}</div></div></div>"
                "<div style='font-size:11px;font-weight:700;color:#c9a227;"
                "letter-spacing:.12em;text-transform:uppercase;margin-bottom:14px;'>"
                "PIN de acceso</div>"
                "</div>",
                unsafe_allow_html=True
            )
            pin_input = st.text_input(
                "PIN", type="password", max_chars=4,
                placeholder="● ● ● ●", label_visibility="collapsed", key="pin_field"
            )
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            c1, c2 = st.columns([1, 1.4])
            with c1:
                if st.button("← Volver", key="back_btn", use_container_width=True):
                    st.session_state.selected_user_login = None
                    st.session_state.login_error = False
                    st.rerun()
            with c2:
                if st.button("Entrar →", type="primary", key="enter_btn", use_container_width=True):
                    if pin_input == udata["pin"]:
                        st.session_state.authenticated = True
                        st.session_state.current_user  = ukey
                        st.session_state.login_error   = False
                        st.rerun()
                    else:
                        st.session_state.login_error = True
            if st.session_state.login_error:
                st.markdown(
                    "<div style='margin-top:12px;background:rgba(239,68,68,.15);"
                    "border:1px solid rgba(239,68,68,.35);border-radius:12px;"
                    "padding:12px 18px;color:#fca5a5;font-size:13px;text-align:center;'>"
                    "⚠ PIN incorrecto. Intenta de nuevo.</div>",
                    unsafe_allow_html=True
                )

    st.stop()


# ── Authenticated: get current user ──────────────────────────────────────────
current_user = st.session_state.current_user
user_data    = USERS[current_user]
user_ports   = get_user_portfolios(current_user)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── EGADE Logo ────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='padding:16px 12px 8px; border-bottom:1px solid rgba(201,162,39,0.2); margin-bottom:12px;'>"
        "<img src='data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyODAgODAiIHdpZHRoPSIyODAiIGhlaWdodD0iODAiPgogIDwhLS0gUmVkIGFjY2VudCBiYXIgLS0+CiAgPHJlY3QgeD0iMCIgeT0iMCIgd2lkdGg9IjYiIGhlaWdodD0iODAiIGZpbGw9IiNjODEwMmUiLz4KICA8IS0tIEVHQURFIHdvcmRtYXJrIC0tPgogIDx0ZXh0IHg9IjE4IiB5PSI1MiIgZm9udC1mYW1pbHk9Ikdlb3JnaWEsc2VyaWYiIGZvbnQtc2l6ZT0iMzgiIGZvbnQtd2VpZ2h0PSI3MDAiCiAgICAgICAgbGV0dGVyLXNwYWNpbmc9IjMiIGZpbGw9IndoaXRlIj5FR0FERTwvdGV4dD4KICA8IS0tIEJ1c2luZXNzIFNjaG9vbCBzdWJ0aXRsZSAtLT4KICA8dGV4dCB4PSIxOSIgeT0iNjgiIGZvbnQtZmFtaWx5PSJBcmlhbCxzYW5zLXNlcmlmIiBmb250LXNpemU9IjExIiBmb250LXdlaWdodD0iNDAwIgogICAgICAgIGxldHRlci1zcGFjaW5nPSIyLjUiIGZpbGw9InJnYmEoMjU1LDI1NSwyNTUsMC43KSI+QlVTSU5FU1MgU0NIT09MPC90ZXh0PgogIDwhLS0gVGVjIGRlIE1vbnRlcnJleSB0YWcgLS0+CiAgPHRleHQgeD0iMTkiIHk9IjIwIiBmb250LWZhbWlseT0iQXJpYWwsc2Fucy1zZXJpZiIgZm9udC1zaXplPSI5IiBmb250LXdlaWdodD0iNDAwIgogICAgICAgIGxldHRlci1zcGFjaW5nPSIxLjUiIGZpbGw9IiNjODEwMmUiPlRFQyBERSBNT05URVJSRVk8L3RleHQ+Cjwvc3ZnPg=='  style='width:180px; display:block;'>"
        "</div>",
        unsafe_allow_html=True
    )
    # ── User header ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='padding:16px 0 12px'>
        <div style='font-family:"DM Serif Display",serif; font-size:22px; color:#e8f0f8;'>
            Portfolio<br><span style='color:#c9a227;'>Manager</span>
        </div>
        <div style='font-size:11px; color:#6b7c93; margin-top:4px;'>Sistema de Gestión v1.0</div>
        <div style='margin-top:14px; display:flex; align-items:center; gap:10px;
                    background:rgba(255,255,255,0.06); border-radius:10px; padding:10px 12px;'>
            <div style='width:34px; height:34px; border-radius:50%; background:{user_data["color"]};
                        display:flex; align-items:center; justify-content:center;
                        font-weight:700; font-size:13px; color:white; flex-shrink:0;'>
                {user_data["avatar"]}
            </div>
            <div>
                <div style='font-size:13px; font-weight:600; color:#e8f0f8;'>{user_data["name"]}</div>
                <div style='font-size:10px; color:#6b7c93;'>{user_data["role"].capitalize()}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⬡ Cerrar sesión", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.current_user  = None
        st.session_state.selected_user_login = None
        st.rerun()

    st.divider()

    # Google Sheets config — use per-user sheet_id if configured
    sheet_id   = user_data.get("sheet_id", "")
    use_sheets = bool(sheet_id) and get_gspread_client() is not None
    with st.expander("⚙ Configuración Google Sheets", expanded=False):
        if use_sheets:
            st.success("✓ Conectado a Google Sheets")
        elif sheet_id:
            st.warning("⚠ Credenciales no configuradas")
        else:
            st.info("Sin Sheet ID — usando datos demo")
    
    st.divider()
    
    # Portfolio selector
    portfolios = get_demo_portfolios()
    port_names = [p for p in list(portfolios.keys()) if p in user_ports]
    
    st.markdown("<div style='font-size:11px; font-weight:700; letter-spacing:0.1em; color:#c9a227; text-transform:uppercase; margin-bottom:10px;'>Portafolio Activo</div>", unsafe_allow_html=True)

    if "selected_port" not in st.session_state or st.session_state.selected_port not in port_names:
        st.session_state.selected_port = port_names[0]

    # Hide button text/border via CSS — button becomes invisible click area
    st.markdown("""
    <style>
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        position: relative !important;
        margin-top: -46px !important;
        height: 46px !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: transparent !important;
        font-size: 0px !important;
        cursor: pointer !important;
        z-index: 999 !important;
        width: 100% !important;
        display: block !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover,
    [data-testid="stSidebar"] [data-testid="stButton"] button:focus {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    for idx_p, pn in enumerate(port_names):
        pcolor = portfolios[pn]["color"]
        pdesc  = portfolios[pn]["description"]
        is_sel = st.session_state.selected_port == pn
        bg     = pcolor if is_sel else "rgba(255,255,255,0.08)"
        glow   = f"0 0 0 2px {pcolor}, 0 4px 20px {pcolor}66" if is_sel else f"0 0 0 1.5px {pcolor}88"
        fw     = "700" if is_sel else "400"
        # Pure HTML card — click handled by st.button directly below, pulled up via CSS
        st.markdown(
            f"<div style='background:{bg};border-radius:11px;padding:10px 13px;"
            f"margin-bottom:2px;box-shadow:{glow};pointer-events:none;'>"
            f"<div style='font-size:13px;font-weight:{fw};color:white;line-height:1.3;'>{pn}</div>"
            f"<div style='font-size:10px;color:rgba(255,255,255,0.6);margin-top:2px;'>{pdesc}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        if st.button("​", key=f"portbtn_{pn}", use_container_width=True):
            st.session_state.selected_port = pn
            st.rerun()

    selected_port = st.session_state.selected_port

    port_data  = portfolios[selected_port]
    port_color = port_data["color"]
    # ── Pesos óptimos: Bloomberg real → Markowitz → target_wts ──────────────────
    # SIN Sheet: usa pesos MATLAB hardcodeados — sin optimizer, sin datos sintéticos
    # CON Sheet: lee Precios_<port>, calcula retornos reales, corre Markowitz tangencia
    _mkz        = None
    _mkz_ok     = False
    _ret_source = "MATLAB (estático)"

    _ret_df_raw = None
    if use_sheets and sheet_id:
        _ret_df_raw = load_price_history_from_sheets(sheet_id, selected_port)
        if _ret_df_raw is not None and len(_ret_df_raw) >= 12:
            try:
                _ret_df    = _ret_df_raw.dropna(axis=1)
                _mkz       = markowitz_optimize(_ret_df)
                _wts_raw   = _mkz["tangencia"]
                target_wts = {t: w for t, w in _wts_raw.items() if w >= 0.001}
                if not target_wts:
                    target_wts = port_data["target"]
                _mkz_ok    = True
                _ret_source = f"Bloomberg · {len(_ret_df_raw)} meses"
            except Exception:
                target_wts = port_data["target"]
        else:
            target_wts = port_data["target"]
    else:
        target_wts = port_data["target"]

    st.session_state["mkz"]        = _mkz
    st.session_state["mkz_ok"]     = _mkz_ok
    st.session_state["ret_source"] = _ret_source
    st.session_state["target_wts"] = target_wts

    _src_color = "#1a7a4a" if "Bloomberg" in _ret_source else "#2d4a6b"
    st.markdown(
        f"<div style='background:{_src_color}18;border:1px solid {_src_color}44;"
        f"border-radius:8px;padding:6px 10px;font-size:10px;color:{_src_color};"
        f"margin-top:6px;'>⚙ Pesos: {_ret_source}</div>",
        unsafe_allow_html=True)

    
    # Global params
    st.markdown("<div style='font-size:11px; font-weight:700; letter-spacing:0.1em; color:#c9a227; text-transform:uppercase; margin-bottom:8px;'>Nuevo Registro a Libro</div>", unsafe_allow_html=True)
    
    tc_actual     = st.number_input("TC MXN/USD", value=20.50, step=0.01, format="%.2f")
    monthly_add   = st.number_input("Aportación mensual (MXN)", value=2000.0, step=100.0, format="%.0f")
    thresh_reduce = st.slider("Umbral REDUCIR", 0.55, 0.80, 0.65, 0.01, format="%d%%")
    thresh_rebal  = st.slider("Umbral REBALANCEAR", 0.03, 0.15, 0.05, 0.01, format="%d%%")
    
    st.divider()
    
    # Nav
    st.markdown("<div style='font-size:11px; font-weight:700; letter-spacing:0.1em; color:#c9a227; text-transform:uppercase; margin-bottom:8px;'>Navegación</div>", unsafe_allow_html=True)
    
    _is_admin = user_data.get("role") == "admin"
    if _is_admin:
        _pages = ["📊 Dashboard", "📋 Operaciones", "⚡ Análisis de Riesgo", "👥 Usuarios", "🔭 QuickView"]
    else:
        _pages = ["📊 Mi Portafolio", "📋 Registrar Operación", "📈 Cómo va mi inversión"]
    page = st.radio("Navegación", _pages, label_visibility="collapsed")

# ── Load data ─────────────────────────────────────────────────────────────────
# Operaciones: desde Google Sheets o demo
if use_sheets and sheet_id:
    ops_raw = load_sheet(sheet_id, f"Ops_{selected_port}")
    if ops_raw is not None and not ops_raw.empty:
        ops_df = ops_raw
        ops_source = "Google Sheets"
    else:
        ops_df = get_demo_operations(selected_port)
        ops_source = "Demo (tab vacío)"
else:
    ops_df = get_demo_operations(selected_port)
    ops_source = "Demo"

# Asegurar tipos numéricos correctos
if not ops_df.empty:
    for col in ["Cantidad", "Precio_USD", "Comision_USD", "TC_MXN"]:
        if col in ops_df.columns:
            ops_df[col] = pd.to_numeric(ops_df[col], errors="coerce").fillna(0)

# ── History: build from Bloomberg price series + ops if available ────────────
_price_hist_raw = None
if use_sheets and sheet_id:
    _price_hist_raw = load_price_history_from_sheets(sheet_id, selected_port)

if _price_hist_raw is not None and not ops_df.empty:
    _history_real = calc_history_from_prices(_price_hist_raw, ops_df, tc_actual)
    history = _history_real if _history_real is not None else get_demo_history()
else:
    history = get_demo_history()

# Precios: desde tab Precios_<portafolio> (ej. Precios_Tangencia),
# con fallback a tab genérico 'Precios', y finalmente a demo.
prices = None
prices_tab_used = None
if use_sheets and sheet_id:
    prices = load_prices_from_sheets(sheet_id, portfolio_name=selected_port)
    if prices:
        # Detect which tab was actually used
        prices_tab_used = f"Precios_{selected_port}"

if prices is None:
    prices = get_demo_prices()
    prices_source = "Demo"
else:
    prices_source = f"Bloomberg · {prices_tab_used} ({len(prices)} activos)"

# ── Banner ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-banner">
    <div>
        <div class="banner-title">{selected_port}</div>
        <div class="banner-sub">{port_data['description']}</div>
        <div style="font-size:11px; color:rgba(201,162,39,0.8); margin-top:2px;">Precios: {prices_source}</div>
        <div class="banner-date">Actualizado: {datetime.now().strftime('%d %b %Y, %H:%M')}</div>
    </div>
    <div style='text-align:right; color:rgba(232,240,248,0.6); font-size:13px;'>
        <div style='font-size:36px; font-family:"DM Serif Display",serif; color:{port_color};'>
            {len(target_wts)} activos
        </div>
        <div>en portafolio</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Compute positions & alerts ────────────────────────────────────────────────
# Only show positions for tickers the optimizer selected (target_wts)
# Operations on tickers not in target_wts are still counted in PnL but not shown
# as separate rows — avoids showing stale/discarded universe tickers
_active_tickers = list(target_wts.keys())
positions_df = calc_positions(ops_df, prices, tc_actual, target_tickers=_active_tickers)
alerts_df    = calc_alerts(positions_df, target_wts, tc_actual, monthly_add,
                           thresh_reduce, thresh_rebal)
_bbg_rm = {}
if _ret_df_raw is not None and target_wts:
    _bbg_rm = calc_risk_metrics_bloomberg(_ret_df_raw, target_wts, rf=port_data.get("rf", 0.0264))
risk_metrics = _bbg_rm if _bbg_rm else calc_risk_metrics(history)
total_mxn    = alerts_df["Valor MXN"].sum() if not alerts_df.empty else 0
total_usd    = alerts_df["Valor USD"].sum() if not alerts_df.empty else 0

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page in ("📊 Dashboard", "📊 Mi Portafolio"):

    _is_admin = user_data.get("role") == "admin"

    # ══════════════════════════════════════════════════════════════════════════
    # INVESTOR VIEW — clean, action-first
    # ══════════════════════════════════════════════════════════════════════════
    if not _is_admin:

        # ── Determine overall portfolio status ────────────────────────────────
        pending = []
        if not alerts_df.empty:
            pending = alerts_df[alerts_df["Alerta"] != "OK"]

        has_actions = len(pending) > 0
        total_ganancia = alerts_df["Ganancia USD"].sum() if not alerts_df.empty else 0
        gan_pct = (total_ganancia / (total_usd - total_ganancia) * 100
                   if (total_usd - total_ganancia) > 0 else 0)

        # ── Status banner ─────────────────────────────────────────────────────
        INVESTOR_PROFILES = {
            "Tangencia":   {"label":"BALANCEADO",   "icon":"⚖️",  "color":"#c9a227",
                            "tolerancia":"Media","horizonte":"3–7 años"},
            "Conservador": {"label":"CONSERVADOR",  "icon":"🛡️",  "color":"#1a7a4a",
                            "tolerancia":"Baja", "horizonte":"1–4 años"},
            "Agresivo":    {"label":"AGRESIVO",     "icon":"🚀",  "color":"#c0392b",
                            "tolerancia":"Alta", "horizonte":"5–10 años"},
            "Familiar":    {"label":"PATRIMONIAL",  "icon":"🏛️",  "color":"#2d4a6b",
                            "tolerancia":"Muy baja","horizonte":"10+ años"},
        }
        prof = INVESTOR_PROFILES.get(selected_port, INVESTOR_PROFILES["Tangencia"])

        if has_actions:
            n_buy  = len(alerts_df[alerts_df["Alerta"] == "COMPRAR"])
            n_sell = len(alerts_df[alerts_df["Alerta"] == "VENDER"])
            parts = []
            if n_buy:  parts.append(f"{n_buy} acción{'es' if n_buy>1 else ''} a comprar")
            if n_sell: parts.append(f"{n_sell} acción{'es' if n_sell>1 else ''} a vender")
            status_msg = "Tu portafolio necesita ajuste — " + " · ".join(parts)
            status_bg  = "linear-gradient(135deg,#fff8e1 0%,#fff3cd 100%)"
            status_border = "#c9a227"
            status_icon   = "⚠️"
        else:
            status_msg    = "Tu portafolio está en equilibrio — no hay acciones pendientes"
            status_bg     = "linear-gradient(135deg,#edfaf3 0%,#d5f5e3 100%)"
            status_border = "#1a7a4a"
            status_icon   = "✅"

        st.markdown(f"""
        <div style='background:{status_bg};border-left:5px solid {status_border};
                    border-radius:12px;padding:18px 24px;margin-bottom:20px;
                    box-shadow:0 2px 12px rgba(0,0,0,0.07);'>
            <div style='font-size:20px;font-family:"DM Serif Display",serif;
                        color:#1a2332;'>{status_icon} {status_msg}</div>
            <div style='font-size:12px;color:#6b7c93;margin-top:4px;'>
                {prof["icon"]} Perfil {prof["label"]} &nbsp;·&nbsp;
                Horizonte {prof["horizonte"]} &nbsp;·&nbsp;
                Tolerancia al riesgo: {prof["tolerancia"]}
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Perfil del Inversionista ───────────────────────────────────────────
        INVESTOR_PROFILES_FULL = {
            "Tangencia": {
                "label":"BALANCEADO","icon":"⚖️","color":"#c9a227",
                "bg":"linear-gradient(135deg,#fffbee 0%,#fef3cc 100%)","border":"#c9a227",
                "objetivo":"Maximizar el Sharpe Ratio — mejor retorno por unidad de riesgo",
                "horizonte":"Mediano–largo plazo · 3–7 años","horizonte_years":5,
                "tolerancia":"Media","tolerancia_pct":50,
                "volatilidad_range":(20,28),"retorno_obj":(30,35),"sharpe_obj":1.2,
                "perfil":"Inversionista disciplinado que busca eficiencia. Diversifica entre crecimiento y valor. Rebalancea activamente cuando hay desviaciones.",
                "stats":[("Volatilidad esperada","~20–28%"),("Retorno objetivo","~30–35%"),("Sharpe objetivo","> 1.2"),("Horizonte mínimo","3 años")],
                "teoria":"Markowitz: portafolio sobre la Frontera Eficiente que maximiza el Sharpe Ratio.",
            },
            "Conservador": {
                "label":"CONSERVADOR","icon":"🛡️","color":"#1a7a4a",
                "bg":"linear-gradient(135deg,#edfaf3 0%,#d5f5e3 100%)","border":"#1a7a4a",
                "objetivo":"Minimizar la varianza — preservar capital ante todo",
                "horizonte":"Corto–mediano plazo · 1–4 años","horizonte_years":2,
                "tolerancia":"Baja","tolerancia_pct":20,
                "volatilidad_range":(8,14),"retorno_obj":(10,16),"sharpe_obj":0.8,
                "perfil":"Inversionista adverso al riesgo. Prefiere activos defensivos (oro, tabaco, industria estable). Acepta retornos más bajos a cambio de protección en caídas.",
                "stats":[("Volatilidad esperada","~8–14%"),("Retorno objetivo","~10–16%"),("Sharpe objetivo","> 0.8"),("Horizonte mínimo","1 año")],
                "teoria":"Markowitz: portafolio de Mínima Varianza — punto más a la izquierda de la Frontera Eficiente.",
            },
            "Agresivo": {
                "label":"AGRESIVO","icon":"🚀","color":"#c0392b",
                "bg":"linear-gradient(135deg,#fdf0ef 0%,#fadbd8 100%)","border":"#c0392b",
                "objetivo":"Maximizar el retorno absoluto — crecimiento acelerado",
                "horizonte":"Largo plazo · 5–10 años","horizonte_years":8,
                "tolerancia":"Alta","tolerancia_pct":85,
                "volatilidad_range":(35,55),"retorno_obj":(50,80),"sharpe_obj":1.0,
                "perfil":"Inversionista con alta convicción en tendencias disruptivas (IA, EVs, tecnología). Concentra posiciones en activos de alto beta. Requiere horizonte largo para absorber ciclos.",
                "stats":[("Volatilidad esperada","~35–55%"),("Retorno objetivo","~50–80%"),("Sharpe objetivo","> 1.0"),("Horizonte mínimo","5 años")],
                "teoria":"Portafolio en la parte superior derecha de la Frontera Eficiente — máximo retorno esperado.",
            },
            "Familiar": {
                "label":"PATRIMONIAL","icon":"🏛️","color":"#2d4a6b",
                "bg":"linear-gradient(135deg,#eef3f8 0%,#d6e4f0 100%)","border":"#2d4a6b",
                "objetivo":"Preservar y transmitir patrimonio intergeneracional",
                "horizonte":"Muy largo plazo · 10+ años","horizonte_years":12,
                "tolerancia":"Muy baja","tolerancia_pct":12,
                "volatilidad_range":(10,16),"retorno_obj":(12,18),"sharpe_obj":0.7,
                "perfil":"Gestión de riqueza familiar. Combina activos refugio (oro, tabaco) con activos de valor estable (banca, industria). Minimiza riesgo sistémico.",
                "stats":[("Volatilidad esperada","~10–16%"),("Retorno objetivo","~12–18%"),("Sharpe objetivo","> 0.7"),("Horizonte mínimo","10 años")],
                "teoria":"Estrategia mixta: activos de baja correlación para maximizar diversificación.",
            },
        }
        pf     = INVESTOR_PROFILES_FULL.get(selected_port, INVESTOR_PROFILES_FULL["Tangencia"])
        pc     = pf["color"]
        tol    = pf["tolerancia_pct"]        # 0–100
        hyr    = pf["horizonte_years"]       # numeric years
        vlo,vhi= pf["volatilidad_range"]
        rlo,rhi= pf["retorno_obj"]
        sobj   = pf["sharpe_obj"]

        # ── Tolerance bar segments (5 bands: Muy baja → Muy alta)
        tol_bands = [
            (0,20,"Muy baja","#1a7a4a"),
            (20,40,"Baja","#3b8f5c"),
            (40,60,"Media","#c9a227"),
            (60,80,"Alta","#d4820a"),
            (80,100,"Muy alta","#c0392b"),
        ]
        def _tol_bars(pct, color):
            bars = ""
            for lo, hi, lbl, c in tol_bands:
                filled = "opacity:1" if lo < pct <= hi or (pct == 0 and lo == 0) else "opacity:0.15"
                bars += (
                    f"<div style='display:flex;flex-direction:column;align-items:center;flex:1;gap:3px;'>"
                    f"<div style='background:{c};border-radius:4px;height:22px;width:100%;{filled};'></div>"
                    f"<div style='font-size:9px;color:#6b7c93;text-align:center;line-height:1.2;'>{lbl}</div>"
                    f"</div>"
                )
            return bars

        # ── Horizon timeline dots
        def _horizon_dots(years):
            milestones = [1, 3, 5, 7, 10, 15]
            dots = ""
            for m in milestones:
                active = years >= m
                c = pc if active else "#dde3ea"
                tc = "#1a2332" if active else "#b0bac8"
                dots += (
                    f"<div style='display:flex;flex-direction:column;align-items:center;flex:1;'>"
                    f"<div style='width:18px;height:18px;border-radius:50%;background:{c};"
                    f"margin-bottom:4px;box-shadow:{'0 0 0 3px ' + pc + '33' if active else 'none'};'></div>"
                    f"<div style='font-size:9px;color:{tc};font-weight:{'700' if active else '400'};'>{m}a</div>"
                    f"</div>"
                )
            return dots

        # ── Volatility range bar
        def _vol_range_bar(lo, hi, color):
            full = 60   # scale: 0–60% is full bar
            left_pct = lo / full * 100
            width_pct = (hi - lo) / full * 100
            return (
                f"<div style='position:relative;height:14px;background:#e8f0f8;"
                f"border-radius:7px;overflow:hidden;margin:6px 0;'>"
                f"<div style='position:absolute;left:{left_pct:.0f}%;width:{width_pct:.0f}%;"
                f"height:100%;background:{color};border-radius:7px;opacity:0.85;'></div>"
                f"</div>"
                f"<div style='display:flex;justify-content:space-between;font-size:9px;color:#6b7c93;'>"
                f"<span>0%</span><span>20%</span><span>40%</span><span>60%</span>"
                f"</div>"
            )

        # ─── Render card ───────────────────────────────────────────────────────
        _h = (
            f"<div style='background:{pf['bg']};border-radius:18px;padding:24px 28px;"
            f"border-left:5px solid {pf['border']};box-shadow:0 4px 20px rgba(0,0,0,0.09);"
            f"margin-bottom:6px;'>"
            # Header row
            f"<div style='display:flex;align-items:center;gap:14px;margin-bottom:18px;'>"
            f"<div style='font-size:44px;line-height:1;'>{pf['icon']}</div>"
            f"<div>"
            f"<div style='font-size:10px;letter-spacing:0.12em;font-weight:700;color:{pc};"
            f"text-transform:uppercase;'>Tu perfil de inversión</div>"
            f"<div style='font-size:26px;font-weight:700;color:#1a2332;line-height:1.1;'>{pf['label']}</div>"
            f"<div style='font-size:12px;color:#6b7c93;margin-top:2px;'>{pf['perfil'][:90]}…</div>"
            f"</div>"
            f"</div>"
            # 3 columns: objective text | risk gauge | horizon
            f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;'>"
            # Col 1 — Objective + theory
            f"<div style='background:rgba(255,255,255,0.7);border-radius:12px;padding:14px;'>"
            f"<div style='font-size:10px;font-weight:700;color:{pc};text-transform:uppercase;"
            f"margin-bottom:8px;'>🎯 Objetivo</div>"
            f"<div style='font-size:12px;color:#1a2332;line-height:1.6;margin-bottom:8px;'>{pf['objetivo']}</div>"
            f"<div style='font-size:10px;color:#6b7c93;line-height:1.5;border-top:1px solid rgba(0,0,0,0.07);"
            f"padding-top:8px;'>📐 {pf['teoria']}</div>"
            f"</div>"
            # Col 2 — Risk tolerance gauge + volatility bar
            f"<div style='background:rgba(255,255,255,0.7);border-radius:12px;padding:14px;'>"
            f"<div style='font-size:10px;font-weight:700;color:{pc};text-transform:uppercase;"
            f"margin-bottom:8px;'>💓 Tolerancia al riesgo</div>"
            f"<div style='font-size:18px;font-weight:700;color:{pc};margin-bottom:8px;'>{pf['tolerancia']}</div>"
            f"<div style='display:flex;gap:4px;margin-bottom:12px;'>{_tol_bars(tol, pc)}</div>"
            f"<div style='font-size:10px;font-weight:700;color:{pc};text-transform:uppercase;"
            f"margin-bottom:4px;'>📊 Volatilidad esperada</div>"
            f"<div style='font-size:12px;color:#1a2332;font-weight:600;'>{vlo}% – {vhi}%</div>"
            f"{_vol_range_bar(vlo, vhi, pc)}"
            f"</div>"
            # Col 3 — Horizon timeline + return obj
            f"<div style='background:rgba(255,255,255,0.7);border-radius:12px;padding:14px;'>"
            f"<div style='font-size:10px;font-weight:700;color:{pc};text-transform:uppercase;"
            f"margin-bottom:8px;'>⏱ Horizonte de inversión</div>"
            f"<div style='font-size:12px;color:#1a2332;margin-bottom:10px;'>{pf['horizonte']}</div>"
            f"<div style='display:flex;align-items:flex-end;gap:2px;margin-bottom:12px;'>"
            f"{_horizon_dots(hyr)}"
            f"</div>"
            f"<div style='font-size:10px;font-weight:700;color:{pc};text-transform:uppercase;"
            f"margin-bottom:4px;'>📈 Retorno objetivo</div>"
            f"<div style='font-size:18px;font-weight:700;color:{pc};'>{rlo}% – {rhi}%</div>"
            f"<div style='font-size:10px;color:#6b7c93;'>Sharpe objetivo: &gt; {sobj}</div>"
            f"</div>"
            f"</div>"  # end grid
            f"</div>"  # end card
        )
        st.markdown(_h, unsafe_allow_html=True)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── Valor de la Acción Actual

        # ── KPI cards — plain language + interpretation ─────────────────────
        k1, k2, k3 = st.columns(3)
        sign      = "+" if total_ganancia >= 0 else ""
        gan_color = "positive" if total_ganancia >= 0 else "negative"

        # Derive contextual interpretation per profile
        _prof_vol_ok = {
            "Tangencia":   0.28, "Conservador": 0.14,
            "Agresivo":    0.55, "Familiar":    0.16,
        }
        _vol_thresh = _prof_vol_ok.get(selected_port, 0.25)
        _vol_now    = risk_metrics.get("vol_annual", 0)
        _sharpe_now = risk_metrics.get("sharpe", 0)

        if gan_pct >= 15:
            gan_interp = "Tu inversión está creciendo muy bien 🚀"
        elif gan_pct >= 5:
            gan_interp = "Buen progreso — en línea con tu estrategia"
        elif gan_pct >= 0:
            gan_interp = "Ligeramente positivo — dentro de lo normal"
        elif gan_pct >= -10:
            gan_interp = "Pérdida moderada — evalúa con tu asesor"
        else:
            gan_interp = "Pérdida importante — revisa con tu asesor"

        if _vol_now <= _vol_thresh * 0.7:
            vol_interp = f"Muy estable para tu perfil {prof['icon']}"
        elif _vol_now <= _vol_thresh:
            vol_interp = f"Volatilidad dentro de lo esperado para tu perfil"
        else:
            vol_interp = f"Más volátil de lo habitual — es temporal"

        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Valor de mi inversión</div>
                <div class="kpi-value">${total_mxn:,.0f}</div>
                <div class="kpi-sub">MXN · tipo de cambio {tc_actual:.2f}</div>
                <div style="font-size:11px;color:#6b7c93;margin-top:6px;border-top:1px solid #f0f4f9;padding-top:6px;">
                    Equivale a <b>${total_usd:,.0f} USD</b>
                </div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Ganancia / Pérdida total</div>
                <div class="kpi-value {gan_color}">{sign}${total_ganancia:,.0f} USD</div>
                <div class="kpi-sub">{sign}{gan_pct:.1f}% sobre lo invertido</div>
                <div style="font-size:11px;color:#6b7c93;margin-top:6px;border-top:1px solid #f0f4f9;padding-top:6px;">
                    {gan_interp}
                </div>
            </div>""", unsafe_allow_html=True)
        with k3:
            n_ok   = len(alerts_df[alerts_df["Alerta"] == "OK"]) if not alerts_df.empty else 0
            n_tot  = len(alerts_df) if not alerts_df.empty else 0
            health_color = "#1a7a4a" if not has_actions else "#c9a227"
            health_label = "Todo en orden" if not has_actions else f"{n_ok}/{n_tot} activos en rango"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Estado del portafolio</div>
                <div class="kpi-value" style="font-size:20px;color:{health_color};">
                    {"✅ En equilibrio" if not has_actions else "⚠️ Necesita ajuste"}
                </div>
                <div class="kpi-sub">{health_label}</div>
                <div style="font-size:11px;color:#6b7c93;margin-top:6px;border-top:1px solid #f0f4f9;padding-top:6px;">
                    {vol_interp}
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Semáforo visual ───────────────────────────────────────────────────
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        _sr   = risk_metrics.get("sharpe", 0)
        _vol  = risk_metrics.get("vol_annual", 0)
        _dd   = risk_metrics.get("max_dd", 0)
        _var  = risk_metrics.get("var_95", 0)

        def _semaforo(val, good, warn):
            if val >= good:   return "#1a7a4a", "🟢"
            if val >= warn:   return "#d4820a", "🟡"
            return "#c0392b", "🔴"

        sr_col,  sr_ico  = _semaforo(_sr,   1.0,  0.5)
        vol_col, vol_ico = _semaforo(-_vol, -_vol_thresh, -_vol_thresh*1.4)  # lower is better
        dd_col,  dd_ico  = _semaforo(_dd,   -0.10, -0.20)   # less negative is better
        var_col, var_ico = _semaforo(-abs(_var), -0.04, -0.08)

        # Plain-language gauge
        sr_txt  = ("Excelente — tu dinero trabaja bien" if _sr >= 1.0
                   else "Aceptable — dentro de lo normal" if _sr >= 0.5
                   else "Bajo — revisa con tu asesor")
        vol_txt = ("Estable" if _vol <= _vol_thresh
                   else "Moderada" if _vol <= _vol_thresh*1.4 else "Alta")
        dd_txt  = ("Mínima caída histórica" if _dd >= -0.10
                   else "Caída moderada" if _dd >= -0.20 else "Caída importante")
        var_txt = ("Riesgo mensual bajo" if abs(_var) <= 0.04
                   else "Riesgo moderado" if abs(_var) <= 0.08 else "Riesgo elevado")

        sem_cols = st.columns(4)
        for col, ico, lbl, val_str, txt, color in [
            (sem_cols[0], sr_ico,  "Eficiencia",        f"{_sr:.2f}",            sr_txt,  sr_col),
            (sem_cols[1], vol_ico, "Volatilidad anual", f"{_vol*100:.1f}%",      vol_txt, vol_col),
            (sem_cols[2], dd_ico,  "Caída máx. histórica", f"{_dd*100:.1f}%",   dd_txt,  dd_col),
            (sem_cols[3], var_ico, "Riesgo mensual",    f"{abs(_var)*100:.1f}%", var_txt, var_col),
        ]:
            with col:
                st.markdown(f"""
                <div style='background:white;border-radius:12px;padding:16px;text-align:center;
                            box-shadow:0 2px 12px rgba(13,27,42,0.07);border-top:4px solid {color};'>
                    <div style='font-size:22px;margin-bottom:4px;'>{ico}</div>
                    <div style='font-size:10px;text-transform:uppercase;letter-spacing:0.08em;
                                color:#6b7c93;margin-bottom:4px;'>{lbl}</div>
                    <div style='font-family:"DM Serif Display",serif;font-size:24px;color:{color};'>{val_str}</div>
                    <div style='font-size:11px;color:{color};font-weight:600;margin-top:4px;'>{txt}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Action instructions — most important section ───────────────────────
        if has_actions:
            st.markdown("<div class='section-title'>📋 Acciones que debes tomar</div>",
                        unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:12px;color:#6b7c93;margin-bottom:14px;'>"
                "El modelo indica que tu portafolio se desvió de los porcentajes óptimos. "
                "Ejecuta estas operaciones para mantener la estrategia funcionando.</div>",
                unsafe_allow_html=True)

            order_cols = st.columns(len(pending))
            for i, (_, row) in enumerate(pending.iterrows()):
                with order_cols[i]:
                    desv_pct = row["Desviación"] * 100
                    if row["Alerta"] == "COMPRAR":
                        mxn_c = row["Compra Sugerida MXN"]
                        usd_c = mxn_c / tc_actual if tc_actual > 0 else 0
                        st.markdown(
                            "<div class='order-card order-buy'>"
                            "<div style='font-size:26px'>🟢</div>"
                            "<div style='flex:1'>"
                            f"<div style='font-weight:700;font-size:16px;color:#1a3a5c'>{row['Ticker']}</div>"
                            "<div style='font-size:12px;font-weight:700;color:#1a7a4a;margin-bottom:6px'>COMPRAR — tienes poco de este</div>"
                            f"<div style='font-size:13px'><b>{row['Acciones a Comprar']:.4f} acciones</b>"
                            f" a ${row['Precio Actual']:,.2f}</div>"
                            f"<div style='font-size:13px;color:#1a7a4a'><b>${usd_c:,.2f} USD</b>"
                            f" · ${mxn_c:,.0f} MXN</div>"
                            f"<div style='font-size:11px;color:#888;margin-top:4px'>"
                            f"Tienes {row['Peso Actual']*100:.1f}% · deberías tener {row['Peso Objetivo']*100:.1f}%</div>"
                            "</div></div>", unsafe_allow_html=True)
                    else:
                        mxn_v = row["Monto a Vender MXN"]
                        usd_v = mxn_v / tc_actual if tc_actual > 0 else 0
                        st.markdown(
                            "<div class='order-card order-sell'>"
                            "<div style='font-size:26px'>🔴</div>"
                            "<div style='flex:1'>"
                            f"<div style='font-weight:700;font-size:16px;color:#1a3a5c'>{row['Ticker']}</div>"
                            "<div style='font-size:12px;font-weight:700;color:#c0392b;margin-bottom:6px'>VENDER — tienes demasiado de este</div>"
                            f"<div style='font-size:13px'><b>{row['Acciones a Vender']:.4f} acciones</b>"
                            f" a ${row['Precio Actual']:,.2f}</div>"
                            f"<div style='font-size:13px;color:#c0392b'><b>${usd_v:,.2f} USD</b>"
                            f" · ${mxn_v:,.0f} MXN</div>"
                            f"<div style='font-size:11px;color:#888;margin-top:4px'>"
                            f"Tienes {row['Peso Actual']*100:.1f}% · deberías tener {row['Peso Objetivo']*100:.1f}%</div>"
                            "</div></div>", unsafe_allow_html=True)
        else:
            st.success("✅ Todo en orden — no hay acciones pendientes este periodo.")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── My positions — simple table ────────────────────────────────────────
        st.markdown("<div class='section-title'>📊 Mis acciones hoy</div>",
                    unsafe_allow_html=True)

        if not alerts_df.empty:
            def simple_badge(a):
                if a == "COMPRAR": return "🟢 Comprar más"
                if a == "VENDER":  return "🔴 Vender parte"
                return "✅ OK"

            rows_html = ""
            for _, row in alerts_df.iterrows():
                gan_c = "#1a7a4a" if row["Ganancia USD"] >= 0 else "#c0392b"
                gsign = "+" if row["Ganancia USD"] >= 0 else ""
                rows_html += (
                    f"<tr>"
                    f"<td><b>{row['Ticker']}</b></td>"
                    f"<td>{row['Cantidad']:.4f} acc</td>"
                    f"<td>${row['Precio Actual']:,.2f}</td>"
                    f"<td><b>${row['Valor MXN']:,.0f} MXN</b></td>"
                    f"<td>{row['Peso Actual']*100:.1f}%</td>"
                    f"<td style='color:#6b7c93'>{row['Peso Objetivo']*100:.1f}%</td>"
                    f"<td style='color:{gan_c}'>{gsign}${row['Ganancia USD']:,.0f}</td>"
                    f"<td>{simple_badge(row['Alerta'])}</td>"
                    f"</tr>"
                )
            st.markdown(f"""
            <table class="styled-table">
                <thead><tr>
                    <th>Acción</th><th>Cantidad</th><th>Precio actual</th>
                    <th>Valor</th><th>Tengo %</th><th>Debo tener %</th>
                    <th>Ganancia</th><th>¿Qué hacer?</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>""", unsafe_allow_html=True)
        else:
            st.info("Aún no tienes operaciones registradas. Ve a 'Registrar Operación' para comenzar.")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # [profile card moved to top — rendered above KPIs]

        # ── Valor de la Acción Actual ──────────────────────────────────────────
        st.markdown("<div class='section-title'>💹 Valor de la Acción Actual</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:12px;color:#6b7c93;margin-bottom:12px;'>"
            "Precio actual de cada acción en tu portafolio y su posición dentro del rango de las últimas 52 semanas. "
            "Recuerda: el modelo se basa en datos históricos a 5 años — no reacciones a movimientos del día.</div>",
            unsafe_allow_html=True)

        if not YF_AVAILABLE:
            st.info("📡 Precios en tiempo real no disponibles — agrega yfinance al requirements.txt")
        else:
            with st.spinner("Cargando cotizaciones..."):
                live_q_inv = get_live_quotes(list(target_wts.keys()))
            if live_q_inv:
                st.markdown(
                    f'<div style="font-size:11px;color:#6b7c93;margin-bottom:12px;">'                    f'📡 Yahoo Finance · tiempo real · actualiza cada 5 min · {datetime.now().strftime("%H:%M:%S")}</div>',
                    unsafe_allow_html=True)
                val_cols = st.columns(min(len(target_wts), 4))
                for i, tkr in enumerate(target_wts.keys()):
                    q          = live_q_inv.get(tkr, {})
                    price      = q.get("price")
                    chg_pct    = q.get("change_pct")
                    chg_7d     = q.get("chg_7d_pct")
                    hist_7d    = q.get("hist_7d", [])
                    vs_ma52w   = q.get("vs_ma52w")
                    w52_high   = q.get("week52_high")
                    w52_low    = q.get("week52_low")
                    with val_cols[i % 4]:
                        if not price:
                            st.markdown(
                                f"<div style='background:white;border-radius:14px;padding:16px;"
                                f"border:1px solid #e8eef5;text-align:center;'>"
                                f"<b>{tkr}</b><br><span style='color:#aab;'>Sin datos</span></div>",
                                unsafe_allow_html=True)
                            continue
                        # Color día
                        chg_color = "#1a7a4a" if (chg_pct or 0) >= 0 else "#c0392b"
                        chg_bg    = "#d5f5e3" if (chg_pct or 0) >= 0 else "#fadbd8"
                        chg_icon  = "▲" if (chg_pct or 0) >= 0 else "▼"
                        chg_str   = f"{chg_icon} {abs(chg_pct):.2f}%" if chg_pct is not None else "—"
                        # Color 7d
                        c7_color = "#1a7a4a" if (chg_7d or 0) >= 0 else "#c0392b"
                        c7_icon  = "▲" if (chg_7d or 0) >= 0 else "▼"
                        c7_str   = f"{c7_icon} {abs(chg_7d):.2f}% (7d)" if chg_7d is not None else ""
                        # Señal 52w
                        if vs_ma52w is None:
                            sig_label,sig_color,sig_bg,sig_icon = "Sin datos","#6b7c93","#f4f7fb","⚪"
                        elif vs_ma52w > 20:
                            sig_label,sig_color,sig_bg,sig_icon = "MUY CARA","#c0392b","#fadbd8","🔴"
                        elif vs_ma52w > 5:
                            sig_label,sig_color,sig_bg,sig_icon = "SOBREVALORADA","#d4820a","#fef9e7","🟡"
                        elif vs_ma52w >= -5:
                            sig_label,sig_color,sig_bg,sig_icon = "PRECIO JUSTO","#2d4a6b","#e8f0f8","🔵"
                        elif vs_ma52w >= -20:
                            sig_label,sig_color,sig_bg,sig_icon = "INFRAVALORADA","#1a7a4a","#d5f5e3","🟢"
                        else:
                            sig_label,sig_color,sig_bg,sig_icon = "MUY BARATA","#1a7a4a","#d5f5e3","🟢"
                        # Sparkline 7d con plotly
                        spark_html = ""
                        if len(hist_7d) >= 2:
                            import plotly.graph_objects as _go
                            sp_color = "#1a7a4a" if hist_7d[-1] >= hist_7d[0] else "#c0392b"
                            fig_sp = _go.Figure(_go.Scatter(
                                y=hist_7d, mode="lines",
                                line=dict(color=sp_color, width=2),
                                fill="tozeroy", fillcolor=f"{sp_color}22"
                            ))
                            fig_sp.update_layout(
                                height=50, margin=dict(l=0,r=0,t=0,b=0),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                xaxis=dict(visible=False), yaxis=dict(visible=False),
                                showlegend=False
                            )
                            st.plotly_chart(fig_sp, use_container_width=True, config={"displayModeBar":False})
                        st.markdown(f"""
                        <div style='background:white;border-radius:14px;padding:14px 16px;
                             border:1px solid #e8eef5;margin-bottom:8px;'>
                          <div style='font-weight:700;font-size:15px;color:#1a3a5c;'>{tkr}</div>
                          <div style='font-size:22px;font-weight:800;color:#1a3a5c;margin:4px 0;'>
                            ${price:,.2f}
                          </div>
                          <div style='display:inline-block;background:{chg_bg};color:{chg_color};
                               border-radius:6px;padding:2px 8px;font-size:12px;font-weight:700;'>
                            {chg_str} hoy
                          </div>
                          <div style='font-size:12px;color:{c7_color};margin-top:4px;font-weight:600;'>
                            {c7_str}
                          </div>
                          <div style='margin-top:8px;font-size:11px;color:#6b7c93;'>
                            52s: ${w52_low:,.2f} — ${w52_high:,.2f}
                          </div>
                          <div style='margin-top:4px;display:inline-block;background:{sig_bg};
                               color:{sig_color};border-radius:6px;padding:2px 8px;font-size:11px;'>
                            {sig_icon} {sig_label}
                          </div>
                        </div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    
        # ── Positions table + pie chart ───────────────────────────────────────────
        col_left, col_right = st.columns([3, 2])
    
        with col_left:
            st.markdown('<div class="section-title anim-fadein">Posiciones Actuales</div>', unsafe_allow_html=True)
        
            if not alerts_df.empty:
                def alert_badge(a):
                    if a == "REDUCIR":   return '<span class="badge-danger">🔴 REDUCIR</span>'
                    if a == "REBALANCEAR": return '<span class="badge-warn">🟡 REBALANCEAR</span>'
                    return '<span class="badge-ok">✅ OK</span>'
            
                rows_html = ""
                for _, row in alerts_df.iterrows():
                    gan_color = "#1a7a4a" if row["Ganancia USD"] >= 0 else "#c0392b"
                    gan_sign  = "+" if row["Ganancia USD"] >= 0 else ""
                    rows_html += f"""
                    <tr>
                        <td><b>{row['Ticker']}</b></td>
                        <td>{row['Cantidad']:.4f}</td>
                        <td>${row['Precio Actual']:,.2f}</td>
                        <td><b>${row['Valor MXN']:,.0f}</b></td>
                        <td>{row['Peso Actual']*100:.1f}%</td>
                        <td>{row['Peso Objetivo']*100:.1f}%</td>
                        <td style="color:{('#c0392b' if row['Desviación']>0.05 else '#1a7a4a')}">{row['Desviación']*100:+.1f}%</td>
                        <td style="color:{gan_color}">{gan_sign}${row['Ganancia USD']:,.0f}</td>
                        <td>{alert_badge(row['Alerta'])}</td>
                    </tr>"""
            
                st.markdown(f"""
                <table class="styled-table">
                    <thead><tr>
                        <th>Ticker</th><th>Cantidad</th><th>Precio</th>
                        <th>Valor MXN</th><th>Peso %</th><th>Obj %</th>
                        <th>Desv.</th><th>Ganancia</th><th>Alerta</th>
                    </tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>""", unsafe_allow_html=True)
            else:
                st.info("Sin posiciones registradas. Registra operaciones primero.")
    
        with col_right:
            st.markdown('<div class="section-title">Distribución</div>', unsafe_allow_html=True)
        
            if not alerts_df.empty:
                fig_pie = make_subplots(rows=1, cols=2, specs=[[{"type":"pie"},{"type":"pie"}]],
                                        subplot_titles=["Actual", "Objetivo"])
            
                colors_act = ["#1a3a5c","#c9a227","#1a7a4a","#c0392b","#2d4a6b","#d4820a"]
            
                fig_pie.add_trace(go.Pie(
                    labels=alerts_df["Ticker"],
                    values=alerts_df["Peso Actual"],
                    customdata=list(zip(
                        alerts_df["Valor MXN"],
                        alerts_df["Peso Objetivo"]*100,
                        alerts_df["Desviación"]*100
                    )),
                    hole=0.5, marker_colors=colors_act,
                    textinfo="label+percent", textfont_size=11,
                    showlegend=False,
                    hovertemplate=(
                        "<b>%{label}</b><br>"
                        "Peso actual: %{percent}<br>"
                        "Valor: $%{customdata[0]:,.0f} MXN<br>"
                        "Objetivo: %{customdata[1]:.1f}%<br>"
                        "Desviación: %{customdata[2]:+.1f}%<extra></extra>"
                    )
                ), row=1, col=1)
            
                tickers_obj = list(target_wts.keys())
                weights_obj = list(target_wts.values())
                fig_pie.add_trace(go.Pie(
                    labels=tickers_obj,
                    values=weights_obj,
                    hole=0.5, marker_colors=colors_act,
                    textinfo="label+percent", textfont_size=11,
                    showlegend=False,
                    hovertemplate="<b>%{label}</b><br>Peso objetivo: %{percent}<extra></extra>"
                ), row=1, col=2)
            
                fig_pie.update_layout(
                    height=300, margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="white", plot_bgcolor="#fafbfe",
                    font=dict(family="DM Sans", color="#1a2332"),
                    annotations=[
                        dict(text="Actual", x=0.18, y=0.5, font_size=12, showarrow=False),
                        dict(text="Objetivo", x=0.82, y=0.5, font_size=12, showarrow=False)
                    ]
                )
    
            st.markdown(
                "<div style='background:#e8f0f8; border-left:3px solid #1a3a5c; "
                "border-radius:0 8px 8px 0; padding:8px 12px; margin-bottom:6px; "
                "font-size:11px; color:#2d4a6b; line-height:1.5;'>"
                "<b style='font-size:12px;'>ℹ</b>  Las <b>donas</b> comparan la distribución actual del portafolio vs los pesos objetivo de Markowitz. Una diferencia amplia indica necesidad de rebalanceo.</div>",
                unsafe_allow_html=True
            )
            st.markdown('<div class="chart3d">', unsafe_allow_html=True)
            st.plotly_chart(fig_pie, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
    
        # ── Valor de la Acción Actual ─────────────────────────────────────────────
        st.markdown('<div class="section-title anim-fadein anim-delay-1">Valor de la Acción Actual</div>', unsafe_allow_html=True)

        if not YF_AVAILABLE:
            st.warning("Instala yfinance: `pip3 install yfinance`")
        else:
            with st.spinner("Cargando cotizaciones..."):
                live_q = get_live_quotes(list(target_wts.keys()))

            if live_q:
                st.markdown(
                    f'<div style="font-size:11px; color:#6b7c93; margin-bottom:12px;">' +
                    f'📡 Yahoo Finance · tiempo real · actualiza cada 5 min · {datetime.now().strftime("%H:%M:%S")}</div>',
                    unsafe_allow_html=True
                )
                val_cols = st.columns(len(target_wts))
                for i, tkr in enumerate(target_wts.keys()):
                    q = live_q.get(tkr, {})
                    price     = q.get("price")
                    chg_pct   = q.get("change_pct")
                    vs_ma52w  = q.get("vs_ma52w")
                    w52_high  = q.get("week52_high")
                    w52_low   = q.get("week52_low")
                    ma52w     = q.get("ma52w")

                    with val_cols[i]:
                        if not price:
                            st.markdown(
                                f"<div style='background:white; border-radius:14px; padding:16px;"
                                f"border:1px solid #e8eef5; text-align:center;'>"
                                f"<b>{tkr}</b><br><span style='color:#aab;'>Sin datos</span></div>",
                                unsafe_allow_html=True
                            )
                            continue

                        # Variación del día
                        chg_color = "#1a7a4a" if (chg_pct or 0) >= 0 else "#c0392b"
                        chg_bg    = "#d5f5e3" if (chg_pct or 0) >= 0 else "#fadbd8"
                        chg_icon  = "▲" if (chg_pct or 0) >= 0 else "▼"
                        chg_str   = f"{chg_icon} {abs(chg_pct):.2f}%" if chg_pct is not None else "—"

                        # Señal vs promedio 52 semanas
                        if vs_ma52w is None:
                            sig_label, sig_color, sig_bg, sig_icon = "Sin datos", "#6b7c93", "#f4f7fb", "⚪"
                        elif vs_ma52w > 20:
                            sig_label, sig_color, sig_bg, sig_icon = "MUY CARA", "#c0392b", "#fadbd8", "🔴"
                        elif vs_ma52w > 5:
                            sig_label, sig_color, sig_bg, sig_icon = "SOBREVALORADA", "#d4820a", "#fef9e7", "🟡"
                        elif vs_ma52w >= -5:
                            sig_label, sig_color, sig_bg, sig_icon = "PRECIO JUSTO", "#2d4a6b", "#e8f0f8", "🔵"
                        elif vs_ma52w >= -20:
                            sig_label, sig_color, sig_bg, sig_icon = "INFRAVALORADA", "#1a7a4a", "#d5f5e3", "🟢"
                        else:
                            sig_label, sig_color, sig_bg, sig_icon = "MUY BARATA", "#1a7a4a", "#d5f5e3", "🟢"

                        vs_str   = f"{vs_ma52w:+.1f}% vs prom 52s" if vs_ma52w is not None else ""
                        ma_str   = f"${ma52w:,.2f}" if ma52w else "—"
                        h52_str  = f"${w52_high:,.2f}" if w52_high else "—"
                        l52_str  = f"${w52_low:,.2f}"  if w52_low  else "—"

                        # 52-week range bar
                        if w52_high and w52_low and w52_high > w52_low and price:
                            bar_pct = max(0, min(100, (price - w52_low) / (w52_high - w52_low) * 100))
                        else:
                            bar_pct = 50

                        card = (
                            f"<div style='background:white; border-radius:14px; padding:16px 14px;"
                            f"border:1px solid #e8eef5; box-shadow:0 2px 10px rgba(13,27,42,0.07);'>"
                            # Header: ticker + day change badge
                            f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>"
                            f"<span style='font-weight:700; font-size:15px; color:#1a2332;'>{tkr}</span>"
                            f"<span style='background:{chg_bg}; color:{chg_color}; padding:2px 8px;"
                            f"border-radius:10px; font-size:11px; font-weight:700;'>{chg_str}</span></div>"
                            # Price
                            f"<div style='font-size:26px; font-weight:700; color:#1a3a5c; margin-bottom:10px;'>${price:,.2f}</div>"
                            # Valuation signal
                            f"<div style='background:{sig_bg}; border-radius:8px; padding:7px 10px; margin-bottom:10px;"
                            f"border-left:3px solid {sig_color};'>"
                            f"<div style='font-size:12px; font-weight:700; color:{sig_color};'>{sig_icon} {sig_label}</div>"
                            f"<div style='font-size:10px; color:#6b7c93; margin-top:2px;'>{vs_str}</div></div>"
                            # 52-week range
                            f"<div style='font-size:10px; color:#6b7c93; margin-bottom:4px;'>"
                            f"Rango 52 semanas · Prom: <b style='color:#1a3a5c;'>{ma_str}</b></div>"
                            f"<div style='background:#eef2f7; border-radius:4px; height:6px; position:relative; margin-bottom:3px;'>"
                            f"<div style='background:linear-gradient(to right,#c0392b,#c9a227,#1a7a4a);"
                            f"border-radius:4px; height:6px; opacity:0.3;'></div>"
                            f"<div style='position:absolute; left:{bar_pct:.0f}%; top:-4px;"
                            f"width:14px; height:14px; border-radius:50%; background:#1a3a5c;"
                            f"transform:translateX(-50%); border:2px solid white;"
                            f"box-shadow:0 1px 4px rgba(0,0,0,0.2);'></div></div>"
                            f"<div style='display:flex; justify-content:space-between; font-size:9px; color:#aab;'>"
                            f"<span>Min {l52_str}</span><span>Max {h52_str}</span></div>"
                            f"</div>"
                        )
                        st.markdown(card, unsafe_allow_html=True)
            else:
                st.info("No se pudieron cargar cotizaciones en tiempo real.")

        # ── Historical chart ──────────────────────────────────────────────────────
        st.markdown('<div class="section-title anim-fadein anim-delay-2">Evolución del Portafolio</div>', unsafe_allow_html=True)
    
        if not history.empty:
            fig_hist = go.Figure()
        
            # Area fill
            fig_hist.add_trace(go.Scatter(
                x=history["Mes"], y=history["Valor"],
                fill="tozeroy", fillcolor="rgba(26,58,92,0.08)",
                line=dict(color="#3b82f6", width=2.5),
                mode="lines+markers",
                marker=dict(size=8, color="#c9a227", line=dict(width=2, color="white")),
                name="Valor MXN",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Valor: <b>$%{y:,.0f} MXN</b><extra></extra>"
                )
            ))
        
            # Returns as bar
            vals = history["Valor"].values
            rets = np.concatenate([[0], np.diff(vals)/vals[:-1]*100])
            colors_ret = ["#1a7a4a" if r >= 0 else "#c0392b" for r in rets]
        
            fig_hist.add_trace(go.Bar(
                x=history["Mes"], y=rets,
                name="Retorno mensual %",
                marker_color=colors_ret,
                opacity=0.7,
                yaxis="y2",
                hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>"
            ))
        
            fig_hist.update_layout(
                height=320, paper_bgcolor="white", plot_bgcolor="#fafbfe",
                font=dict(family="DM Sans", color="#1a2332"),
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, font_size=11),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=True, gridcolor="#eef2f7", zeroline=False,
                           tickprefix="$", tickformat=",.0f"),
                yaxis2=dict(overlaying="y", side="right", showgrid=False,
                            ticksuffix="%", zeroline=True, zerolinecolor="#ddd"),
                hovermode="x unified",
                bargap=0.3,
                transition=dict(duration=700, easing="cubic-in-out"),

            )

            st.markdown(
                "<div style='background:#e8f0f8; border-left:3px solid #1a3a5c; "
                "border-radius:0 8px 8px 0; padding:8px 12px; margin-bottom:6px; "
                "font-size:11px; color:#2d4a6b; line-height:1.5;'>"
                "<b style='font-size:12px;'>ℹ</b>  La <b>línea</b> muestra la evolución del valor total en MXN. Las <b>barras</b> son el retorno mensual — verde = ganancia, rojo = pérdida. Un portafolio sano muestra barras verdes más frecuentes y línea ascendente.</div>",
                unsafe_allow_html=True
            )
            st.markdown('<div class="chart3d">', unsafe_allow_html=True)
            st.plotly_chart(fig_hist, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Live Market Data ──────────────────────────────────────────────────────
        st.markdown('<div class="section-title anim-fadein anim-delay-3">Mercado en Tiempo Real</div>', unsafe_allow_html=True)

        port_tickers = list(target_wts.keys())

        if not YF_AVAILABLE:
            st.warning("Instala yfinance para ver datos en tiempo real: `pip install yfinance`")
        else:
            with st.spinner("Cargando datos de mercado..."):
                quotes = get_live_quotes(port_tickers)

            if not quotes:
                st.info("No se pudieron cargar cotizaciones. Verifica tu conexión a internet.")
            else:
                # ── Build carousel cards ───────────────────────────────────────
                cards_html = ""
                for tkr in port_tickers:
                    q         = quotes.get(tkr, {})
                    price     = q.get("price")
                    open_p    = q.get("open")
                    prev_c    = q.get("prev_close")
                    day_high  = q.get("day_high")
                    day_low   = q.get("day_low")
                    chg_pct   = q.get("change_pct")

                    if price is None:
                        cards_html += (
                            f"<div class='mkt-card'><div style='font-weight:700;font-size:14px;color:#1a2332;'>{tkr}</div>"
                            "<div style='color:#aab;font-size:12px;margin-top:8px;'>Sin datos</div></div>"
                        )
                        continue

                    chg_color = "#1a7a4a" if (chg_pct or 0) >= 0 else "#c0392b"
                    chg_bg    = "#d5f5e3" if (chg_pct or 0) >= 0 else "#fadbd8"
                    chg_icon  = "▲" if (chg_pct or 0) >= 0 else "▼"
                    chg_str   = f"{chg_icon} {abs(chg_pct):.2f}%" if chg_pct is not None else "—"

                    if day_high and day_low and day_high > day_low:
                        range_pct = max(0, min(100, (price - day_low) / (day_high - day_low) * 100))
                    else:
                        range_pct = 50

                    open_str  = f"${open_p:,.2f}"   if open_p   else "—"
                    prev_str  = f"${prev_c:,.2f}"   if prev_c   else "—"
                    high_str  = f"${day_high:,.2f}" if day_high else "—"
                    low_str   = f"${day_low:,.2f}"  if day_low  else "—"

                    cards_html += (
                        "<div class='mkt-card'>"
                        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>"
                        f"<span style='font-weight:700;font-size:15px;color:#1a2332;'>{tkr}</span>"
                        f"<span style='background:{chg_bg};color:{chg_color};padding:2px 8px;"
                        f"border-radius:10px;font-size:11px;font-weight:700;'>{chg_str}</span></div>"
                        f"<div style='font-size:26px;font-weight:700;color:#1a3a5c;margin-bottom:8px;'>${price:,.2f}</div>"
                        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:3px;"
                        "font-size:11px;color:#6b7c93;margin-bottom:10px;'>"
                        f"<div>Apertura <b style='color:#1a2332;'>{open_str}</b></div>"
                        f"<div>Cierre ant. <b style='color:#1a2332;'>{prev_str}</b></div>"
                        f"<div>Max <b style='color:#1a7a4a;'>{high_str}</b></div>"
                        f"<div>Min <b style='color:#c0392b;'>{low_str}</b></div></div>"
                        "<div style='font-size:9px;color:#aab;margin-bottom:4px;'>Rango del día</div>"
                        "<div style='background:#eef2f7;border-radius:4px;height:8px;position:relative;'>"
                        "<div style='background:linear-gradient(to right,#c0392b,#c9a227,#1a7a4a);"
                        "border-radius:4px;height:8px;opacity:0.25;'></div>"
                        f"<div style='position:absolute;left:{range_pct:.0f}%;top:-3px;"
                        "width:14px;height:14px;border-radius:50%;background:#1a3a5c;"
                        "transform:translateX(-50%);border:2px solid white;"
                        "box-shadow:0 1px 4px rgba(0,0,0,0.2);'></div></div>"
                        f"<div style='display:flex;justify-content:space-between;font-size:9px;color:#aab;margin-top:3px;'>"
                        f"<span>{low_str}</span><span>{high_str}</span></div>"
                        "</div>"
                    )

                # Duplicate cards for seamless infinite loop
                n_cards = len(port_tickers)
                carousel_html = f"""
    <div style='font-size:11px;color:#6b7c93;margin-bottom:10px;'>
      📡 Datos Yahoo Finance · Refresco cada 5 min · {datetime.now().strftime("%H:%M:%S")}
    </div>
    <style>
      .mkt-carousel-wrap {{
        overflow: hidden;
        width: 100%;
        position: relative;
      }}
      .mkt-carousel-wrap::before, .mkt-carousel-wrap::after {{
        content: "";
        position: absolute;
        top: 0; bottom: 0;
        width: 60px;
        z-index: 2;
        pointer-events: none;
      }}
      .mkt-carousel-wrap::before {{ left:0;  background: linear-gradient(to right, #f7f9fc, transparent); }}
      .mkt-carousel-wrap::after  {{ right:0; background: linear-gradient(to left,  #f7f9fc, transparent); }}
      .mkt-track {{
        display: flex;
        gap: 16px;
        width: max-content;
        animation: mkt-scroll {n_cards * 5}s linear infinite;
      }}
      .mkt-track:hover {{ animation-play-state: paused; }}
      @keyframes mkt-scroll {{
        0%   {{ transform: translateX(0); }}
        100% {{ transform: translateX(-50%); }}
      }}
      .mkt-card {{
        background: white;
        border-radius: 14px;
        padding: 16px 14px;
        border: 1px solid #e8eef5;
        box-shadow: 0 2px 10px rgba(13,27,42,0.07);
        min-width: 230px;
        max-width: 230px;
        flex-shrink: 0;
        transition: box-shadow 0.2s, transform 0.2s;
      }}
      .mkt-card:hover {{
        box-shadow: 0 6px 20px rgba(13,27,42,0.14);
        transform: translateY(-3px);
      }}
    </style>
    <div class="mkt-carousel-wrap">
      <div class="mkt-track">
        {cards_html}{cards_html}
      </div>
    </div>"""
                st.markdown(carousel_html, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════════
    # PAGE: OPERACIONES
    # ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Operaciones" and user_data.get("role") == "admin":
    
    # ── Órdenes de Rebalanceo ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">Órdenes de Rebalanceo</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:12px; color:#6b7c93; margin-bottom:12px;'>"
        "Acciones necesarias para llevar el portafolio a sus pesos objetivo.</div>",
        unsafe_allow_html=True
    )
    if not alerts_df.empty:
        has_orders = False
        order_cols = st.columns(len(alerts_df))
        for i, (_, row) in enumerate(alerts_df.iterrows()):
            with order_cols[i]:
                desv_pct = row["Desviación"] * 100
                if row["Alerta"] == "COMPRAR":
                    has_orders = True
                    mxn_c = row["Compra Sugerida MXN"]
                    usd_c = mxn_c / tc_actual if tc_actual > 0 else 0
                    st.markdown(
                        "<div class='order-card order-buy' style='border-left:4px solid #1a7a4a; background:#d5f5e3;'>"
                        "<div style='font-size:22px;'>🟢</div><div style='flex:1;'>"
                        f"<div style='font-weight:700; font-size:15px; color:#1a3a5c;'>{row['Ticker']}</div>"
                        "<div style='font-size:12px; font-weight:700; color:#1a7a4a;'>COMPRAR · subponderado</div>"
                        f"<div style='margin-top:5px; font-size:12px;'><b>{row['Acciones a Comprar']:.4f} acc</b>"
                        f" @ ${row['Precio Actual']:,.2f}<br><b>${usd_c:,.2f} USD</b> · <b>${mxn_c:,.0f} MXN</b></div>"
                        f"<div style='font-size:11px; color:#1a7a4a; margin-top:3px;'>Desviación: {desv_pct:+.1f}%</div>"
                        "</div></div>", unsafe_allow_html=True)
                elif row["Alerta"] == "VENDER":
                    has_orders = True
                    mxn_v = row["Monto a Vender MXN"]
                    usd_v = mxn_v / tc_actual if tc_actual > 0 else 0
                    isr   = usd_v * 0.048
                    st.markdown(
                        "<div class='order-card order-sell' style='border-left:4px solid #c0392b; background:#fadbd8;'>"
                        "<div style='font-size:22px;'>🔴</div><div style='flex:1;'>"
                        f"<div style='font-weight:700; font-size:15px; color:#1a3a5c;'>{row['Ticker']}</div>"
                        "<div style='font-size:12px; font-weight:700; color:#c0392b;'>VENDER · sobreponderado</div>"
                        f"<div style='margin-top:5px; font-size:12px;'><b>{row['Acciones a Vender']:.4f} acc</b>"
                        f" @ ${row['Precio Actual']:,.2f}<br><b>${usd_v:,.2f} USD</b> · <b>${mxn_v:,.0f} MXN</b></div>"
                        f"<div style='font-size:11px; color:#c0392b; margin-top:3px;'>Desv: {desv_pct:+.1f}% · ISR est. ${isr:,.2f} USD</div>"
                        "</div></div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<div class='order-card order-hold'>"
                        "<div style='font-size:22px;'>✅</div><div>"
                        f"<div style='font-weight:700; font-size:15px;'>{row['Ticker']}</div>"
                        "<div style='font-size:12px; color:#2d4a6b; font-weight:600;'>MANTENER</div>"
                        f"<div style='font-size:11px; color:#888; margin-top:3px;'>"
                        f"Peso {row['Peso Actual']*100:.1f}% · Obj {row['Peso Objetivo']*100:.1f}%<br>"
                        f"Desv: {desv_pct:+.1f}%</div></div></div>", unsafe_allow_html=True)
        if not has_orders:
            st.success("✅ Portafolio en equilibrio — no se requieren órdenes.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Registrar Nueva Operación ─────────────────────────────────────────────
    st.markdown('<div class="section-title">Registrar Nueva Operación</div>', unsafe_allow_html=True)
    
    with st.form("new_operation", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            fecha     = st.date_input("Fecha", value=date.today())
            tipo      = st.selectbox("Tipo", ["Compra", "Venta"])
        with c2:
            ticker    = st.text_input("Ticker", placeholder="NVDA").upper()
            cantidad  = st.number_input("Cantidad (acciones)", min_value=0.0001, value=1.0, step=0.0001, format="%.4f")
        with c3:
            precio    = st.number_input("Precio USD", min_value=0.01, value=100.0, step=0.01, format="%.2f")
            comision  = st.number_input("Comisión USD", min_value=0.0, value=0.50, step=0.01, format="%.2f")
        
        tc_op = st.number_input("TC MXN/USD al momento de la operación", value=tc_actual, step=0.01, format="%.2f")
        
        total_usd_op = cantidad * precio + comision
        total_mxn_op = total_usd_op * tc_op
        
        st.markdown(f"""
        <div style='background:#e8f0f8; border-radius:8px; padding:12px 16px; margin:8px 0;'>
            <span style='color:#6b7c93; font-size:12px;'>Total USD: </span>
            <b>${total_usd_op:,.2f}</b>
            &nbsp;&nbsp;&nbsp;
            <span style='color:#6b7c93; font-size:12px;'>Total MXN: </span>
            <b>${total_mxn_op:,.2f}</b>
        </div>""", unsafe_allow_html=True)
        
        submitted = st.form_submit_button("💾 Registrar Operación", width="stretch")
        
        if submitted:
            if ticker:
                new_row = {
                    "Fecha": str(fecha),
                    "Ticker": ticker,
                    "Tipo": tipo,
                    "Cantidad": cantidad,
                    "Precio_USD": precio,
                    "Comision_USD": comision,
                    "TC_MXN": tc_op
                }
                if use_sheets:
                    row_list = [str(fecha), ticker, tipo, cantidad, precio, comision, tc_op]
                    ok = save_row(sheet_id, f"Ops_{selected_port}", row_list)
                    if ok:
                        st.success(f"✅ Operación registrada en Google Sheets: {tipo} {cantidad:.4f} {ticker}")
                        st.cache_data.clear()
                    else:
                        st.error("Error guardando en Google Sheets")
                else:
                    st.success(f"✅ Operación registrada (demo): {tipo} {cantidad:.4f} {ticker} @ ${precio:.2f}")
                    st.info("Para guardar permanentemente, conecta Google Sheets en la configuración.")
            else:
                st.error("Ingresa un ticker válido")
    
    # ── Operations history ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Historial de Operaciones</div>', unsafe_allow_html=True)
    
    if not ops_df.empty:
        # Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            filter_ticker = st.multiselect("Ticker", ops_df["Ticker"].unique().tolist(), 
                                           default=ops_df["Ticker"].unique().tolist())
        with fc2:
            filter_tipo = st.multiselect("Tipo", ["Compra", "Venta"], default=["Compra", "Venta"])
        with fc3:
            st.markdown("<br>", unsafe_allow_html=True)
        
        filtered_ops = ops_df[
            ops_df["Ticker"].isin(filter_ticker) & 
            ops_df["Tipo"].isin(filter_tipo)
        ].copy()
        
        filtered_ops["Total USD"] = filtered_ops["Cantidad"] * filtered_ops["Precio_USD"] + filtered_ops["Comision_USD"]
        filtered_ops["Total MXN"] = filtered_ops["Total USD"] * filtered_ops["TC_MXN"]
        
        rows_html = ""
        for _, row in filtered_ops.iterrows():
            tipo_color = "#1a7a4a" if row["Tipo"] == "Compra" else "#c0392b"
            tipo_badge = f'<span style="background:{"#d5f5e3" if row["Tipo"]=="Compra" else "#fadbd8"}; color:{tipo_color}; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:600;">{row["Tipo"]}</span>'
            rows_html += f"""
            <tr>
                <td>{row['Fecha']}</td>
                <td><b>{row['Ticker']}</b></td>
                <td>{tipo_badge}</td>
                <td>{row['Cantidad']:.4f}</td>
                <td>${row['Precio_USD']:,.2f}</td>
                <td>${row['Comision_USD']:,.2f}</td>
                <td>{row['TC_MXN']:.2f}</td>
                <td><b>${row['Total USD']:,.2f}</b></td>
                <td><b>${row['Total MXN']:,.0f}</b></td>
            </tr>"""
        
        st.markdown(f"""
        <table class="styled-table">
            <thead><tr>
                <th>Fecha</th><th>Ticker</th><th>Tipo</th><th>Cantidad</th>
                <th>Precio USD</th><th>Comisión</th><th>TC</th><th>Total USD</th><th>Total MXN</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)
        
        # Summary
        total_invertido = filtered_ops["Total MXN"].sum()
        st.markdown(f"""
        <div style='background:#1a3a5c; color:white; border-radius:8px; padding:12px 20px; margin-top:12px; display:flex; justify-content:space-between;'>
            <span>{len(filtered_ops)} operaciones</span>
            <span>Total invertido: <b>${total_invertido:,.0f} MXN</b></span>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Sin operaciones registradas para este portafolio.")
    
    # ── Historial mensual ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Actualizar Historial Mensual</div>', unsafe_allow_html=True)
    
    with st.form("update_history"):
        hc1, hc2 = st.columns(2)
        with hc1:
            mes_hist = st.text_input("Mes (ej: Ene 2026)", placeholder="Ene 2026")
        with hc2:
            val_hist = st.number_input("Valor total portafolio (MXN)", min_value=0.0, value=total_mxn, format="%.2f")
        
        submitted_hist = st.form_submit_button("📅 Guardar en Historial", width="stretch")
        if submitted_hist and mes_hist:
            st.success(f"✅ Registrado: {mes_hist} → ${val_hist:,.2f} MXN")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANÁLISIS DE RIESGO
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Cómo va mi inversión":
    # ═══════════════════════════════════════════════════════════════════════════
    # INVESTOR INSIGHT PAGE — educative, visual, calm language
    # ═══════════════════════════════════════════════════════════════════════════

    st.markdown("""
    <div style='background:linear-gradient(135deg,#eef3f8 0%,#e8f0f8 100%);
                border-radius:16px;padding:20px 26px;margin-bottom:24px;
                border-left:5px solid #1a3a5c;'>
        <div style='font-family:"DM Serif Display",serif;font-size:24px;
                    color:#1a2332;margin-bottom:6px;'>📈 Cómo va mi inversión</div>
        <div style='font-size:13px;color:#6b7c93;line-height:1.6;'>
            Aquí encontrarás los indicadores clave de tu portafolio explicados en términos simples.
            No necesitas ser experto — cada número tiene su contexto y significado.
            <b>La clave es entender la tendencia, no reaccionar al día a día.</b>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Section 1: Los 4 números más importantes ──────────────────────────────
    st.markdown("<div class='section-title'>🔢 Los 4 números que debes conocer</div>",
                unsafe_allow_html=True)

    _sharpe  = risk_metrics.get("sharpe", 0)
    _vol     = risk_metrics.get("vol_annual", 0)
    _ret     = risk_metrics.get("ret_annual", 0)
    _maxdd   = risk_metrics.get("max_dd", 0)
    _var     = risk_metrics.get("var_95", 0)
    _nmonths = risk_metrics.get("n_months", 0)

    _sl, _sc = sharpe_label(_sharpe)

    # Plain language per metric
    def _sr_plain(v):
        if v >= 1.0: return "Tu portafolio genera buen retorno por el riesgo que toma — es eficiente"
        if v >= 0.5: return "Retorno aceptable para el nivel de riesgo — en rango normal"
        return "El retorno no está compensando bien el riesgo — platica con tu asesor"

    def _vol_plain(v, port):
        ranges = {"Tangencia":(0.20,0.28),"Conservador":(0.08,0.14),"Agresivo":(0.35,0.55),"Familiar":(0.10,0.16)}
        lo, hi = ranges.get(port, (0.15, 0.30))
        if v <= lo: return f"Volatilidad baja para tu perfil — tu inversión se mueve de forma estable"
        if v <= hi: return f"Volatilidad dentro de lo esperado para el perfil {port}"
        return f"Volatilidad algo elevada — normal en momentos de mercado agitado"

    def _ret_plain(v):
        if v >= 0.20: return "Retorno anual muy sólido — por encima del promedio histórico"
        if v >= 0.10: return "Retorno positivo y consistente con tu estrategia"
        if v >= 0:    return "Retorno moderado positivo — en línea con mercado estable"
        return "Retorno negativo este periodo — evalúa con tu asesor si persiste"

    def _dd_plain(v):
        if v >= -0.05: return "Caída mínima desde el máximo — portafolio muy estable"
        if v >= -0.15: return "Caída moderada — dentro de rangos normales para inversión"
        if v >= -0.25: return "Caída notable — el modelo está diseñado para recuperarse"
        return "Caída importante — recuerda que los datos de 5 años absorben estos ciclos"

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    for col, icon, title, number, color, meaning, learn in [
        (c1, "⚡", "Eficiencia de tu inversión",
         f"{_sharpe:.2f}",  _sc,
         _sr_plain(_sharpe),
         "Este número se llama <b>Sharpe Ratio</b>. Compara cuánto ganaste contra cuánto riesgo tomaste. "
         "Un valor mayor a 1 significa que el riesgo valió la pena. El modelo Markowitz busca maximizar este número."),
        (c2, "📊", "Cuánto varía tu inversión al año",
         f"{_vol*100:.1f}%", "#2d4a6b",
         _vol_plain(_vol, selected_port),
         "Esto se llama <b>Volatilidad</b>. Si tu portafolio vale $100,000 y la volatilidad es 20%, "
         "en un año normal podría estar entre $80,000 y $120,000. No es pérdida — es el rango de movimiento esperado."),
        (c3, "📈", "Rendimiento anual de tu portafolio",
         f"{_ret*100:.1f}%", "#1a7a4a" if _ret >= 0 else "#c0392b",
         _ret_plain(_ret),
         "Es el <b>retorno anualizado</b> — cuánto ha crecido tu inversión en promedio por año. "
         "Se calcula sobre el historial registrado. El objetivo de tu perfil es consistencia, no picos aislados."),
        (c4, "📉", "Mayor caída histórica registrada",
         f"{_maxdd*100:.1f}%", "#c0392b" if _maxdd < -0.15 else "#d4820a",
         _dd_plain(_maxdd),
         "Se llama <b>Drawdown máximo</b>. Es la caída más grande desde un pico histórico. "
         "El modelo usa datos de Bloomberg a 5 años precisamente para que estos eventos extremos "
         "no distorsionen tu estrategia. La disciplina del rebalanceo protege frente a esto."),
    ]:
        with col:
            st.markdown(f"""
            <div style='background:white;border-radius:16px;padding:22px;
                        box-shadow:0 2px 16px rgba(13,27,42,0.08);margin-bottom:8px;
                        border-left:4px solid {color};'>
                <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>
                    <span style='font-size:28px;'>{icon}</span>
                    <span style='font-size:12px;font-weight:700;color:#6b7c93;
                                 text-transform:uppercase;letter-spacing:0.08em;'>{title}</span>
                </div>
                <div style='font-family:"DM Serif Display",serif;font-size:36px;
                            color:{color};margin-bottom:8px;'>{number}</div>
                <div style='font-size:13px;color:#1a2332;margin-bottom:12px;
                            line-height:1.5;'>{meaning}</div>
                <details style='cursor:pointer;'>
                    <summary style='font-size:11px;color:#c9a227;font-weight:700;
                                    list-style:none;'>📖 ¿Qué significa este número?</summary>
                    <div style='font-size:11px;color:#6b7c93;margin-top:8px;
                                line-height:1.6;border-top:1px solid #f0f4f9;padding-top:8px;'>
                        {learn}
                    </div>
                </details>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Section 2: Cómo se ha movido tu dinero ────────────────────────────────
    st.markdown("<div class='section-title'>📆 Cómo se ha movido tu dinero mes a mes</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:13px;color:#6b7c93;margin-bottom:14px;line-height:1.6;'>"
        "La línea azul muestra el valor total de tu portafolio. Las barras muestran si cada mes "
        "ganaste o perdiste valor. <b>Verde = mes positivo. Rojo = mes negativo.</b> "
        "Los meses rojos son normales — lo que importa es que la línea tienda hacia arriba.</div>",
        unsafe_allow_html=True)

    if not history.empty:
        vals_h = history["Valor"].values.astype(float)
        rets_h = np.concatenate([[0], np.diff(vals_h)/vals_h[:-1]*100])
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=history["Mes"], y=history["Valor"],
            fill="tozeroy", fillcolor="rgba(26,58,92,0.07)",
            line=dict(color="#3b82f6", width=2.5),
            mode="lines+markers",
            marker=dict(size=8, color="#c9a227", line=dict(width=2, color="white")),
            name="Valor total MXN",
            hovertemplate="<b>%{x}</b><br>Valor: $%{y:,.0f} MXN<extra></extra>"
        ))
        fig_hist.add_trace(go.Bar(
            x=history["Mes"], y=rets_h,
            marker_color=["#1a7a4a" if r >= 0 else "#c0392b" for r in rets_h],
            opacity=0.65, yaxis="y2", name="Cambio mensual %",
            hovertemplate="<b>%{x}</b><br>%{y:+.1f}%<extra></extra>"
        ))
        fig_hist.update_layout(
            height=320, paper_bgcolor="white", plot_bgcolor="#fafbfe",
            font=dict(family="DM Sans", color="#1a2332"),
            margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font_size=11),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#eef2f7", tickprefix="$", tickformat=",.0f",
                       title="Valor MXN"),
            yaxis2=dict(overlaying="y", side="right", showgrid=False, ticksuffix="%",
                        title="Cambio mensual"),
            hovermode="x unified",
        )
        st.markdown('<div class="chart3d">', unsafe_allow_html=True)
        st.plotly_chart(fig_hist, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 3: Meses buenos vs malos ─────────────────────────────────────
    st.markdown("<div class='section-title'>📊 Distribución de tus retornos mensuales</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:13px;color:#6b7c93;margin-bottom:14px;line-height:1.6;'>"
        "Esta gráfica muestra con qué frecuencia tu portafolio tuvo meses positivos o negativos. "
        "<b>Mientras más barras estén a la derecha del cero, mejor.</b> "
        "La línea roja punteada muestra el peor mes esperado en condiciones normales "
        "(esto se llama VaR — Valor en Riesgo).</div>",
        unsafe_allow_html=True)

    col_dist, col_stats = st.columns([2, 1])

    if not history.empty and len(history) > 3:
        vals_d = history["Valor"].values.astype(float)
        rets_d = np.diff(vals_d) / vals_d[:-1] * 100
        var_95_d  = float(np.percentile(rets_d, 5))
        mean_d    = float(np.mean(rets_d))
        pos_months = int(np.sum(rets_d > 0))
        neg_months = int(np.sum(rets_d <= 0))
        total_m    = len(rets_d)

        with col_dist:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=rets_d, nbinsx=10,
                marker_color="#3b82f6", opacity=0.8,
                hovertemplate="%{x:.1f}%<br>Frecuencia: %{y} meses<extra></extra>"
            ))
            fig_dist.add_vline(x=var_95_d, line_dash="dash", line_color="#c0392b", line_width=2,
                annotation_text=f"Peor mes esperado: {var_95_d:.1f}%",
                annotation_font=dict(color="#c0392b", size=11))
            fig_dist.add_vline(x=mean_d, line_dash="dot", line_color="#c9a227", line_width=2,
                annotation_text=f"Promedio: {mean_d:.1f}%",
                annotation_font=dict(color="#c9a227", size=11))
            fig_dist.update_layout(
                height=280, paper_bgcolor="white", plot_bgcolor="#fafbfe",
                font=dict(family="DM Sans"), margin=dict(l=0,r=0,t=20,b=0),
                xaxis=dict(title="Retorno mensual %", showgrid=False, zeroline=True,
                           zerolinecolor="#1a2332", zerolinewidth=2),
                yaxis=dict(title="Número de meses", showgrid=True, gridcolor="#eef2f7"),
                showlegend=False
            )
            st.markdown('<div class="chart3d">', unsafe_allow_html=True)
            st.plotly_chart(fig_dist, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_stats:
            pct_pos   = pos_months / total_m * 100
            bar_color = "#1a7a4a" if pct_pos >= 60 else "#d4820a" if pct_pos >= 50 else "#c0392b"
            best_m  = float(np.max(rets_d))
            worst_m = float(np.min(rets_d))
            _h  = "<div style='background:white;border-radius:14px;padding:20px;box-shadow:0 2px 12px rgba(13,27,42,0.07);'>"
            _h += "<div style='font-size:11px;font-weight:700;color:#6b7c93;text-transform:uppercase;margin-bottom:16px;'>Resumen</div>"
            _h += "<div style='text-align:center;margin-bottom:16px;'>"
            _h += f"<div style='font-size:42px;font-weight:700;color:{bar_color};'>{pct_pos:.0f}%</div>"
            _h += "<div style='font-size:12px;color:#6b7c93;'>de los meses fueron positivos</div>"
            _h += "</div>"
            _h += "<div style='display:flex;justify-content:space-around;margin-bottom:16px;text-align:center;'>"
            _h += f"<div><div style='font-size:22px;font-weight:700;color:#1a7a4a;'>{pos_months}</div><div style='font-size:11px;color:#6b7c93;'>meses &#128994;</div></div>"
            _h += f"<div><div style='font-size:22px;font-weight:700;color:#c0392b;'>{neg_months}</div><div style='font-size:11px;color:#6b7c93;'>meses &#128308;</div></div>"
            _h += "</div>"
            _h += "<div style='background:#f4f7fb;border-radius:10px;padding:12px;font-size:12px;color:#1a2332;line-height:1.8;'>"
            _h += f"<b>Promedio mensual:</b> {mean_d:+.1f}%<br>"
            _h += f"<b>Mejor mes:</b> {best_m:+.1f}%<br>"
            _h += f"<b>Peor mes:</b> {worst_m:+.1f}%<br>"
            _h += f"<b>VaR 95%:</b> {var_95_d:.1f}%"
            _h += "<div style='font-size:10px;color:#6b7c93;margin-top:6px;'>VaR: en 19 de 20 meses la p&#233;rdida no deber&#237;a superar este valor</div>"
            _h += "</div></div>"
            st.markdown(_h, unsafe_allow_html=True)

    # ── Section 4: Estabilidad — la caída desde el máximo ─────────────────────
    st.markdown("<div class='section-title'>🏔️ Qué tan lejos está tu inversión de su máximo</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:13px;color:#6b7c93;margin-bottom:14px;line-height:1.6;'>"
        "Esta gráfica muestra la distancia entre el valor actual de tu portafolio y su punto más alto. "
        "<b>Cuando la línea está en 0%, tu inversión está en máximos históricos.</b> "
        "Cuando baja, muestra cuánto ha caído desde ese pico. "
        "La disciplina del modelo es que el rebalanceo periódico evita que estas caídas se profundicen.</div>",
        unsafe_allow_html=True)

    if not history.empty:
        vals_dd = history["Valor"].values.astype(float)
        peak_dd = np.maximum.accumulate(vals_dd)
        drawdowns_dd = (vals_dd - peak_dd) / peak_dd * 100
        max_dd_val = float(drawdowns_dd.min())
        curr_dd    = float(drawdowns_dd[-1])

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=history["Mes"], y=drawdowns_dd,
            fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
            line=dict(color="#c0392b", width=2),
            mode="lines+markers",
            marker=dict(size=5, color="#c0392b"),
            hovertemplate="<b>%{x}</b><br>Distancia del máximo: %{y:.1f}%<extra></extra>"
        ))
        fig_dd.add_hline(y=max_dd_val, line_dash="dash", line_color="#8e0000",
            annotation_text=f"Peor caída: {max_dd_val:.1f}%",
            annotation_font=dict(color="#8e0000", size=11))
        fig_dd.add_hline(y=0, line_color="#1a7a4a", line_width=1.5,
            annotation_text="Máximo histórico", annotation_position="bottom right",
            annotation_font=dict(color="#1a7a4a", size=10))
        fig_dd.update_layout(
            height=260, paper_bgcolor="white", plot_bgcolor="#fafbfe",
            font=dict(family="DM Sans"), margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#eef2f7", ticksuffix="%",
                       title="Distancia del máximo"),
            showlegend=False
        )
        dd_cols = st.columns([3,1])
        with dd_cols[0]:
            st.markdown('<div class="chart3d">', unsafe_allow_html=True)
            st.plotly_chart(fig_dd, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
        with dd_cols[1]:
            dd_color = "#1a7a4a" if curr_dd >= -0.05 else "#d4820a" if curr_dd >= -0.15 else "#c0392b"
            dd_msg   = ("En máximos ✅" if curr_dd >= -0.01
                        else "Cerca del máximo" if curr_dd >= -0.05
                        else "Recuperando" if curr_dd >= -0.15 else "En proceso de recuperación")
            st.markdown(f"""
            <div style='background:white;border-radius:14px;padding:20px;
                        box-shadow:0 2px 12px rgba(13,27,42,0.07);text-align:center;'>
                <div style='font-size:11px;font-weight:700;color:#6b7c93;
                            text-transform:uppercase;margin-bottom:8px;'>Hoy estás</div>
                <div style='font-family:"DM Serif Display",serif;font-size:34px;
                            color:{dd_color};'>{curr_dd:.1f}%</div>
                <div style='font-size:12px;color:{dd_color};font-weight:600;
                            margin-top:4px;'>{dd_msg}</div>
                <div style='margin-top:14px;font-size:11px;color:#6b7c93;'>
                    Peor caída histórica:<br>
                    <b style='color:#c0392b;font-size:16px;'>{max_dd_val:.1f}%</b>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Section 5: Beta y Alpha — ¿cómo te comparas con el mercado? ───────────
    st.markdown("<div class='section-title'>🌍 Tu portafolio vs el mercado</div>",
                unsafe_allow_html=True)

    _AD_inv = {
        "NVDA":(1.83,+0.38),"CAT":(1.07,+0.08),"NEM":(0.88,+0.05),
        "AAPL":(1.08,+0.12),"BATS":(0.62,-0.02),"TSLA":(1.65,+0.15),
        "ASTS":(1.43,-0.18),"NOW":(1.27,+0.22),"JPM":(1.13,+0.06),
        "9988HK":(1.42,-0.09)
    }
    if not alerts_df.empty:
        _tks = alerts_df["Ticker"].tolist()
        _wts = np.array([target_wts.get(t,0) for t in _tks], dtype=float)
        _wts = _wts / _wts.sum()
        _betas_i  = np.array([_AD_inv.get(t,(1.0,0.0))[0] for t in _tks])
        _alphas_i = np.array([_AD_inv.get(t,(1.0,0.0))[1] for t in _tks])
        _pb = float(np.dot(_wts, _betas_i))
        _pa = float(np.dot(_wts, _alphas_i))

        beta_msg = (
            "Tu portafolio es <b>menos volátil que el mercado</b> — cae menos cuando el mercado cae, "
            "pero también sube menos cuando sube." if _pb < 0.9
            else "Tu portafolio <b>se mueve similar al mercado</b> — va aproximadamente al ritmo del S&P 500." if _pb <= 1.1
            else "Tu portafolio es <b>más dinámico que el mercado</b> — amplifica los movimientos del índice. "
                 "Sube más en mercados alcistas, pero también puede caer más en correcciones."
        )
        alpha_msg = (
            f"Tu portafolio genera un <b>rendimiento adicional de {_pa*100:.1f}% sobre lo que el mercado "
            f"esperaría</b> dado su nivel de riesgo. Eso es un alpha positivo — señal de que la "
            f"selección de activos está funcionando." if _pa > 0
            else f"Tu portafolio está ligeramente por debajo del retorno esperado por su nivel de riesgo "
                 f"({_pa*100:.1f}%). El modelo ajusta continuamente para corregir esto."
        )

        beta_color = "#1a7a4a" if _pb < 0.9 else "#2d4a6b" if _pb <= 1.1 else "#d4820a"
        alpha_color = "#1a7a4a" if _pa >= 0 else "#d4820a"

        bm1, bm2 = st.columns(2)
        with bm1:
            st.markdown(f"""
            <div style='background:white;border-radius:16px;padding:22px;
                        box-shadow:0 2px 16px rgba(13,27,42,0.08);border-left:4px solid {beta_color};'>
                <div style='font-size:11px;font-weight:700;color:#6b7c93;
                            text-transform:uppercase;margin-bottom:8px;'>
                    Beta (β) — Sensibilidad al mercado</div>
                <div style='font-family:"DM Serif Display",serif;font-size:40px;
                            color:{beta_color};margin-bottom:8px;'>{_pb:.2f}</div>
                <div style='font-size:13px;color:#1a2332;margin-bottom:10px;
                            line-height:1.6;'>{beta_msg}</div>
                <div style='background:#f4f7fb;border-radius:8px;padding:10px;
                            font-size:11px;color:#6b7c93;'>
                    <b>β &lt; 1</b> → Más estable que el S&P 500<br>
                    <b>β = 1</b> → Se mueve igual que el mercado<br>
                    <b>β &gt; 1</b> → Más dinámico que el mercado
                </div>
            </div>""", unsafe_allow_html=True)
        with bm2:
            st.markdown(f"""
            <div style='background:white;border-radius:16px;padding:22px;
                        box-shadow:0 2px 16px rgba(13,27,42,0.08);border-left:4px solid {alpha_color};'>
                <div style='font-size:11px;font-weight:700;color:#6b7c93;
                            text-transform:uppercase;margin-bottom:8px;'>
                    Alpha (α) — Valor añadido sobre el mercado</div>
                <div style='font-family:"DM Serif Display",serif;font-size:40px;
                            color:{alpha_color};margin-bottom:8px;'>{_pa:+.2f}</div>
                <div style='font-size:13px;color:#1a2332;margin-bottom:10px;
                            line-height:1.6;'>{alpha_msg}</div>
                <div style='background:#f4f7fb;border-radius:8px;padding:10px;
                            font-size:11px;color:#6b7c93;'>
                    <b>α &gt; 0</b> → La selección de activos supera al mercado<br>
                    <b>α = 0</b> → Retorno exactamente en línea con el riesgo<br>
                    <b>α &lt; 0</b> → Por debajo del retorno esperado por el riesgo
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Section 6: Nota educativa final ───────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0d1b2a 0%,#1a3a5c 100%);
                border-radius:16px;padding:24px 28px;color:white;'>
        <div style='font-family:"DM Serif Display",serif;font-size:20px;
                    color:#c9a227;margin-bottom:12px;'>
            💡 Lo más importante que debes recordar
        </div>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;font-size:13px;
                    line-height:1.7;color:rgba(232,240,248,0.9);'>
            <div>
                <b style='color:#c9a227;'>Los datos de Bloomberg a 5 años</b> eliminan el ruido
                de eventos temporales. Un buen trimestre o una caída puntual no cambian la estrategia.
                Lo que importa es la tendencia de largo plazo.
            </div>
            <div>
                <b style='color:#c9a227;'>El rebalanceo periódico</b> es la herramienta clave.
                Cuando el sistema te dice "compra X" o "vende Y", esa instrucción es matemática —
                no emocional. Seguirla es lo que mantiene el portafolio en su punto óptimo.
            </div>
            <div>
                <b style='color:#c9a227;'>No reacciones al mercado del día.</b>
                Las noticias, los precios que suben o bajan fuerte, los eventos sorpresa —
                el modelo los absorbe. Tu trabajo es registrar tus operaciones y rebalancear
                cuando el sistema lo indique.
            </div>
            <div>
                <b style='color:#c9a227;'>Cada número en esta página</b> tiene un rango esperado
                para tu perfil. Si algo te genera dudas, es el momento ideal para consultar
                con tu asesor — no para actuar por cuenta propia.
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


elif page == "📋 Registrar Operación":
    # Investor-only simple operations page
    st.markdown("<div class='section-title'>📋 Registrar Operación</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:13px;color:#6b7c93;margin-bottom:18px;'>"
        "Ingresa aquí cada compra o venta que realices. Esto mantiene tu portafolio "
        "actualizado para que el sistema pueda darte las instrucciones correctas.</div>",
        unsafe_allow_html=True)

    with st.form("inv_op_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            i_fecha  = st.date_input("Fecha de la operación", value=date.today())
            i_tipo   = st.selectbox("¿Compraste o vendiste?", ["Compra", "Venta"])
        with c2:
            # Only show tickers from this portfolio
            port_tickers_list = list(target_wts.keys())
            i_ticker = st.selectbox("Acción", port_tickers_list)
            i_cant   = st.number_input("Cantidad de acciones", min_value=0.0001,
                                        value=1.0, step=0.0001, format="%.4f")
        with c3:
            i_precio = st.number_input("Precio por acción (USD)",
                                        min_value=0.01, value=100.0, step=0.01)
            i_com    = st.number_input("Comisión (USD)", min_value=0.0,
                                        value=0.50, step=0.01)
        i_tc = st.number_input("Tipo de cambio MXN/USD del día",
                                value=tc_actual, step=0.01, format="%.2f")

        total_usd_i = i_cant * i_precio + i_com
        total_mxn_i = total_usd_i * i_tc
        st.markdown(
            f"<div style='background:#e8f0f8;border-radius:10px;padding:14px 18px;margin:8px 0;'>"
            f"<span style='color:#6b7c93;font-size:12px;'>Total USD: </span>"
            f"<b style='font-size:15px'>${total_usd_i:,.2f}</b>"
            f"&nbsp;&nbsp;&nbsp;"
            f"<span style='color:#6b7c93;font-size:12px;'>Total MXN: </span>"
            f"<b style='font-size:15px'>${total_mxn_i:,.0f}</b>"
            f"</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("💾 Guardar operación", use_container_width=True)
        if submitted:
            if use_sheets:
                ok = save_row(sheet_id, f"Ops_{selected_port}",
                              [str(i_fecha), i_ticker, i_tipo, i_cant, i_precio, i_com, i_tc])
                if ok:
                    st.success(f"✅ Operación registrada: {i_tipo} de {i_cant:.4f} {i_ticker} a ${i_precio:.2f} USD")
                    st.cache_data.clear()
                else:
                    st.error("❌ Error al guardar. Verifica tu conexión a Google Sheets.")
            else:
                st.success(f"✅ Operación registrada (demo): {i_tipo} {i_cant:.4f} {i_ticker} @ ${i_precio:.2f}")

    st.divider()

    # Show recent ops
    st.markdown("### Mis operaciones recientes")
    if not ops_df.empty:
        st.dataframe(
            ops_df.sort_values("Fecha", ascending=False).head(20)
            .style.format({
                "Precio_USD": "${:,.2f}",
                "Cantidad": "{:.4f}",
                "TC_MXN": "{:.2f}",
                "Comision_USD": "${:.2f}",
            }),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("Aún no hay operaciones registradas.")

elif page == "⚡ Análisis de Riesgo" and user_data.get("role") == "admin":
    
    # ── Risk KPIs ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Métricas de Riesgo</div>', unsafe_allow_html=True)
    
    r1, r2, r3, r4, r5 = st.columns(5)
    
    sharpe_val = risk_metrics.get("sharpe", 0)
    sr_label, sr_color = sharpe_label(sharpe_val)
    
    metrics_display = [
        (r1, "Sharpe Ratio", f"{sharpe_val:.4f}", sr_label, sr_color),
        (r2, "Volatilidad Anual", f"{risk_metrics.get('vol_annual',0)*100:.2f}%", "Desv. estándar anualizada", "#2d4a6b"),
        (r3, "Retorno Anual", f"{risk_metrics.get('ret_annual',0)*100:.2f}%", "Promedio anualizado", "#1a7a4a" if risk_metrics.get('ret_annual',0)>0 else "#c0392b"),
        (r4, "VaR 95%", f"{abs(risk_metrics.get('var_95',0))*100:.2f}%", "Pérdida máx. mensual 95%", "#c0392b"),
        (r5, "CVaR 95%", f"{abs(risk_metrics.get('cvar_95',0))*100:.2f}%", "Pérdida esperada peor 5%", "#8e0000"),
    ]
    
    for col, label, value, sub, color in metrics_display:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color}; font-size:22px;">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)
    
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    
    # ── Charts: Returns distribution + Drawdown ───────────────────────────────
    ch1, ch2 = st.columns(2)
    
    with ch1:
        st.markdown('<div class="section-title">Distribución de Retornos</div>', unsafe_allow_html=True)
        
        if not history.empty and len(history) > 3:
            vals    = history["Valor"].values.astype(float)
            returns = np.diff(vals) / vals[:-1] * 100
            
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=returns, nbinsx=12,
                marker_color="#3b82f6", opacity=0.8,
                name="Retornos mensuales",
                hovertemplate="%{x:.2f}%<br>Frecuencia: %{y}<extra></extra>"
            ))
            
            var_95 = np.percentile(returns, 5)
            fig_dist.add_vline(x=var_95, line_dash="dash", line_color="#c0392b",
                               annotation_text=f"VaR 95%: {var_95:.2f}%",
                               annotation_font_color="#c0392b")
            
            mean_ret = np.mean(returns)
            fig_dist.add_vline(x=mean_ret, line_dash="dot", line_color="#c9a227",
                               annotation_text=f"Media: {mean_ret:.2f}%",
                               annotation_font_color="#c9a227")
            
            fig_dist.update_layout(
                height=280, paper_bgcolor="white", plot_bgcolor="#fafbfe",
                font=dict(family="DM Sans"), margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(title="Retorno mensual %", showgrid=False),
                yaxis=dict(title="Frecuencia", showgrid=True, gridcolor="#eef2f7"),
                showlegend=False
            )

        st.markdown(
            "<div style='background:#e8f0f8; border-left:3px solid #1a3a5c; "
            "border-radius:0 8px 8px 0; padding:8px 12px; margin-bottom:6px; "
            "font-size:11px; color:#2d4a6b; line-height:1.5;'>"
            "<b style='font-size:12px;'>ℹ</b>  El <b>histograma</b> muestra la distribución de retornos mensuales. Una curva simétrica y centrada a la derecha del 0 es ideal. Colas largas a la izquierda indican riesgo de pérdidas extremas.</div>",
            unsafe_allow_html=True
        )
        st.markdown('<div class="chart3d">', unsafe_allow_html=True)
        st.plotly_chart(fig_dist, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with ch2:
        st.markdown('<div class="section-title">Drawdown</div>', unsafe_allow_html=True)
        
        if not history.empty:
            vals = history["Valor"].values.astype(float)
            peak = np.maximum.accumulate(vals)
            drawdowns = (vals - peak) / peak * 100
            
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=history["Mes"], y=drawdowns,
                fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
                line=dict(color="#c0392b", width=2),
                mode="lines+markers",
                marker=dict(size=5, color="#c0392b"),
                name="Drawdown %",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Drawdown: <b>%{y:.2f}%</b><extra></extra>"
                )
            ))
            
            max_dd = drawdowns.min()
            fig_dd.add_hline(y=max_dd, line_dash="dash", line_color="#8e0000",
                             annotation_text=f"Máx: {max_dd:.2f}%",
                             annotation_font_color="#8e0000")
            
            fig_dd.update_layout(
                height=280, paper_bgcolor="white", plot_bgcolor="#fafbfe",
                font=dict(family="DM Sans"), margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#eef2f7", ticksuffix="%"),
                showlegend=False,
                transition=dict(duration=600, easing="cubic-in-out"),
            )

        st.markdown(
            "<div style='background:#e8f0f8; border-left:3px solid #1a3a5c; "
            "border-radius:0 8px 8px 0; padding:8px 12px; margin-bottom:6px; "
            "font-size:11px; color:#2d4a6b; line-height:1.5;'>"
            "<b style='font-size:12px;'>ℹ</b>  El <b>drawdown</b> mide la caída desde el máximo histórico. La línea punteada roja es el peor drawdown registrado. Valores cercanos a 0% son mejores; -20% o peor indica alta volatilidad bajista.</div>",
            unsafe_allow_html=True
        )
        st.markdown('<div class="chart3d">', unsafe_allow_html=True)
        st.plotly_chart(fig_dd, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ── Rolling Sharpe & Volatility ───────────────────────────────────────────
    st.markdown('<div class="section-title">Métricas Dinámicas (Rolling)</div>', unsafe_allow_html=True)
    
    if not history.empty and len(history) >= 4:
        vals    = history["Valor"].values.astype(float)
        returns_s = pd.Series(np.diff(vals) / vals[:-1])
        
        window = min(6, len(returns_s))
        roll_vol    = returns_s.rolling(window).std() * np.sqrt(12) * 100
        roll_ret    = returns_s.rolling(window).mean() * 12
        rf_daily    = 0.0264
        roll_sharpe = (roll_ret - rf_daily) / (roll_vol/100)
        
        months_ret = history["Mes"].values[1:]
        
        fig_roll = make_subplots(rows=1, cols=2,
                                  subplot_titles=["Volatilidad Rolling (6M)", "Sharpe Ratio Rolling (6M)"])
        
        fig_roll.add_trace(go.Scatter(
            x=months_ret, y=roll_vol,
            line=dict(color="#2d4a6b", width=2.5),
            fill="tozeroy", fillcolor="rgba(45,74,107,0.1)",
            mode="lines+markers",
            marker=dict(size=5, color="#2d4a6b"),
            name="Volatilidad %",
            hovertemplate="<b>%{x}</b><br>Volatilidad: <b>%{y:.2f}%</b><extra></extra>"
        ), row=1, col=1)
        
        colors_sharpe = ["#1a7a4a" if v > 1 else "#d4820a" if v > 0.5 else "#c0392b" 
                         for v in roll_sharpe.fillna(0)]
        fig_roll.add_trace(go.Bar(
            x=months_ret, y=roll_sharpe,
            marker_color=colors_sharpe, name="Sharpe",
            hovertemplate="<b>%{x}</b><br>Sharpe: <b>%{y:.3f}</b><extra></extra>"
        ), row=1, col=2)
        
        fig_roll.add_hline(y=1.0, line_dash="dash", line_color="#c9a227", row=1, col=2)
        
        fig_roll.update_layout(
            height=280, paper_bgcolor="white", plot_bgcolor="#fafbfe",
            font=dict(family="DM Sans"), margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False,
            xaxis=dict(showgrid=False), xaxis2=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#eef2f7", ticksuffix="%"),
            yaxis2=dict(showgrid=True, gridcolor="#eef2f7"),
            transition=dict(duration=600, easing="cubic-in-out"),
        )

        st.markdown(
            "<div style='background:#e8f0f8; border-left:3px solid #1a3a5c; "
            "border-radius:0 8px 8px 0; padding:8px 12px; margin-bottom:6px; "
            "font-size:11px; color:#2d4a6b; line-height:1.5;'>"
            "<b style='font-size:12px;'>ℹ</b>  Izquierda: <b>volatilidad rolling 3 meses</b> — periodos altos implican más riesgo de mercado. Derecha: <b>Sharpe rolling</b> — valores &gt;1 indican que el retorno compensa el riesgo; la línea dorada marca Sharpe = 1 como umbral de eficiencia.</div>",
            unsafe_allow_html=True
        )
        st.markdown('<div class="chart3d">', unsafe_allow_html=True)
        st.plotly_chart(fig_roll, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ── Risk indicator ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Semáforo del Portafolio</div>', unsafe_allow_html=True)
    
    sr_label_full, sr_col = sharpe_label(sharpe_val)
    vol_val   = risk_metrics.get("vol_annual", 0)
    max_dd_val = risk_metrics.get("max_dd", 0)
    var_val   = risk_metrics.get("var_95", 0)
    
    semaph_cols = st.columns(4)
    indicators = [
        (semaph_cols[0], "Sharpe Ratio", f"{sharpe_val:.3f}", sr_label_full, sr_col),
        (semaph_cols[1], "Volatilidad", f"{vol_val*100:.1f}%", 
         "Controlada" if vol_val < 0.20 else "Elevada", 
         "#1a7a4a" if vol_val < 0.20 else "#d4820a"),
        (semaph_cols[2], "Máx. Drawdown", f"{max_dd_val*100:.1f}%",
         "Aceptable" if max_dd_val > -0.15 else "Severo",
         "#1a7a4a" if max_dd_val > -0.15 else "#c0392b"),
        (semaph_cols[3], "VaR 95%", f"{abs(var_val)*100:.1f}%",
         "Bajo riesgo" if abs(var_val) < 0.05 else "Riesgo moderado",
         "#1a7a4a" if abs(var_val) < 0.05 else "#d4820a"),
    ]
    
    for col, label, value, status, color in indicators:
        with col:
            st.markdown(f"""
            <div style='background:white; border-radius:12px; padding:20px; text-align:center;
                        box-shadow:0 2px 12px rgba(13,27,42,0.08); border-top:4px solid {color};'>
                <div style='font-size:11px; text-transform:uppercase; letter-spacing:0.08em; color:#6b7c93; margin-bottom:8px;'>{label}</div>
                <div style='font-family:"DM Serif Display",serif; font-size:26px; color:{color};'>{value}</div>
                <div style='font-size:12px; color:{color}; font-weight:600; margin-top:4px;'>{status}</div>
            </div>""", unsafe_allow_html=True)
    
    # ── Full Markowitz Output ────────────────────────────────────────────────
    st.markdown('<div class="section-title">📐 Análisis Markowitz Completo</div>', unsafe_allow_html=True)

    _mkz    = st.session_state.get("mkz")
    _ret_df = st.session_state.get("ret_df")
    _ret_src= st.session_state.get("ret_source", "Demo")
    _mkz_ok = st.session_state.get("mkz_ok", False)

    if _mkz_ok and _mkz:
        tickers_mk = _mkz["tickers"]
        wts_tang   = _mkz["tangencia"]
        wts_mv     = _mkz["min_var"]
        exp_rets   = _mkz["exp_rets"]
        corr_mat   = _mkz["corr_matrix"]
        cov_mat    = _mkz["cov_matrix"]
        frontier   = _mkz["frontier"]
        sr_tang    = _mkz["sharpe_tangencia"]
        vol_tang   = _mkz["vol_tang"]
        ret_tang   = _mkz["ret_tang"]
        n_mk       = len(tickers_mk)

        RF_MK  = 0.0264
        MRP_MK = 0.0550

        _AD = {
            "NVDA":(1.83,+0.38),"CAT":(1.07,+0.08),"NEM":(0.88,+0.05),
            "AAPL":(1.08,+0.12),"BATS":(0.62,-0.02),"TSLA":(1.65,+0.15),
            "ASTS":(1.43,-0.18),"NOW":(1.27,+0.22),"JPM":(1.13,+0.06),
            "9988HK":(1.42,-0.09),"SPX":(1.00,0.00),
        }
        betas_mk  = np.array([_AD.get(t,(1.0,0.0))[0] for t in tickers_mk])
        alphas_mk = np.array([_AD.get(t,(1.0,0.0))[1] for t in tickers_mk])
        wts_arr   = np.array([wts_tang.get(t,0) for t in tickers_mk])
        wts_arr  /= wts_arr.sum()

        port_beta_mk  = float(np.dot(wts_arr, betas_mk))
        port_alpha_mk = float(np.dot(wts_arr, alphas_mk))
        port_vol_mk   = vol_tang
        port_ret_mk   = ret_tang
        port_sharpe   = sr_tang
        spx_ret       = RF_MK + MRP_MK
        port_treynor  = (port_ret_mk - RF_MK) / port_beta_mk if port_beta_mk > 0 else 0
        tracking_err  = port_vol_mk * abs(1 - port_beta_mk)
        info_ratio    = port_alpha_mk / tracking_err if tracking_err > 0 else 0

        # Source badge
        src_color = "#1a7a4a" if "Sheet" in _ret_src else "#d4820a"
        st.markdown(
            f"<div style='display:inline-block;background:{src_color}22;border:1px solid {src_color};"
            f"border-radius:20px;padding:3px 12px;font-size:11px;color:{src_color};"
            f"font-weight:700;margin-bottom:14px;'>⚙ Fuente de retornos: {_ret_src}</div>",
            unsafe_allow_html=True)

        # ── 6 KPIs ────────────────────────────────────────────────────────────
        mk1,mk2,mk3,mk4,mk5,mk6 = st.columns(6)
        _sl, _sc = sharpe_label(port_sharpe)
        for col, lbl, val, sub, col_c in [
            (mk1, "Retorno Esperado",  f"{port_ret_mk*100:.2f}%",   f"SPX: {spx_ret*100:.1f}%",  "#1a7a4a" if port_ret_mk>spx_ret else "#d4820a"),
            (mk2, "Beta Ponderado",    f"{port_beta_mk:.3f}",        "< 1 Def · > 1 Agr",         "#1a7a4a" if port_beta_mk<1 else "#c0392b"),
            (mk3, "Alpha Ponderado",   f"{port_alpha_mk:+.3f}",      "Retorno sobre CAPM",         "#1a7a4a" if port_alpha_mk>=0 else "#c0392b"),
            (mk4, "Sharpe Ratio",      f"{port_sharpe:.4f}",         _sl,                          _sc),
            (mk5, "Ratio de Treynor",  f"{port_treynor*100:.2f}%",   "Ret. exceso / Beta",         "#2d4a6b"),
            (mk6, "Information Ratio", f"{info_ratio:.3f}",          "Alpha / Tracking Error",     "#1a7a4a" if info_ratio>0.5 else "#d4820a"),
        ]:
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label" style="font-size:10px;">{lbl}</div>
                    <div class="kpi-value" style="color:{col_c};font-size:20px;">{val}</div>
                    <div class="kpi-sub" style="font-size:10px;">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # ── Pesos óptimos: Tangencia vs Mínima Varianza ───────────────────────
        st.markdown("**Pesos Óptimos** — Tangencia (máx. Sharpe) vs Mínima Varianza")
        wt_cols = st.columns(min(n_mk, 5))
        for i, tkr in enumerate(tickers_mk):
            w_t = wts_tang.get(tkr, 0)
            w_m = wts_mv.get(tkr, 0)
            er  = exp_rets.get(tkr, 0)
            col_c = "#c9a227" if w_t > 0.30 else "#2d4a6b"
            with wt_cols[i % len(wt_cols)]:
                bar_w = int(w_t * 100)
                _wh = (
                    f"<div style='background:white;border-radius:10px;padding:12px 14px;"
                    f"box-shadow:0 2px 8px rgba(13,27,42,0.07);margin-bottom:6px;'>"
                    f"<div style='font-weight:700;font-size:13px;color:#1a2332;'>{tkr}</div>"
                    f"<div style='font-size:10px;color:#6b7c93;margin-bottom:6px;'>E(r): {er*100:.1f}% anual</div>"
                    f"<div style='font-size:11px;color:#6b7c93;margin-bottom:2px;'>Tangencia</div>"
                    f"<div style='height:10px;background:#e8f0f8;border-radius:5px;margin-bottom:4px;'>"
                    f"<div style='height:100%;width:{bar_w}%;background:{col_c};"
                    f"border-radius:5px;'></div></div>"
                    f"<div style='font-size:16px;font-weight:700;color:{col_c};'>{w_t*100:.1f}%</div>"
                    f"<div style='font-size:10px;color:#6b7c93;margin-top:4px;'>Min. Varianza: {w_m*100:.1f}%</div>"
                    f"</div>"
                )
                st.markdown(_wh, unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # ── CAPM breakdown table ───────────────────────────────────────────────
        st.markdown("**CAPM por activo** — E(rᵢ) = Rƒ + βᵢ × MRP + αᵢ")
        capm_rows = ""
        for t in tickers_mk:
            w  = wts_tang.get(t, 0)
            b  = _AD.get(t,(1.0,0.0))[0]
            a  = _AD.get(t,(1.0,0.0))[1]
            er = (RF_MK + b*MRP_MK + a)*100
            req= (RF_MK + b*MRP_MK)*100
            a_col = "#1a7a4a" if a>=0 else "#c0392b"
            a_ico = "▲" if a>=0 else "▼"
            capm_rows += (
                f"<tr><td><b>{t}</b></td>"
                f"<td style='text-align:center'>{w*100:.1f}%</td>"
                f"<td style='text-align:center'>{b:.3f}</td>"
                f"<td style='text-align:center'>{req:.2f}%</td>"
                f"<td style='text-align:center;color:{a_col}'>{a_ico} {abs(a*100):.2f}%</td>"
                f"<td style='text-align:center;font-weight:700;color:{a_col}'>{er:.2f}%</td>"
                f"<td style='text-align:center;color:#2d4a6b'>{w*er:.2f}%</td></tr>"
            )
        a_col_p = "#1a7a4a" if port_alpha_mk>=0 else "#c0392b"
        a_ico_p = "▲" if port_alpha_mk>=0 else "▼"
        ret_col_p = "#1a7a4a" if port_ret_mk>spx_ret else "#d4820a"
        capm_rows += (
            f"<tr style='background:#f0f4f9;font-weight:700;'>"
            f"<td>⭐ PORTAFOLIO</td><td style='text-align:center'>100%</td>"
            f"<td style='text-align:center'>{port_beta_mk:.3f}</td>"
            f"<td style='text-align:center'>{(RF_MK+port_beta_mk*MRP_MK)*100:.2f}%</td>"
            f"<td style='text-align:center;color:{a_col_p}'>{a_ico_p} {abs(port_alpha_mk*100):.2f}%</td>"
            f"<td style='text-align:center;color:{ret_col_p}'>{port_ret_mk*100:.2f}%</td>"
            f"<td style='text-align:center'>—</td></tr>"
        )
        st.markdown(f"""
        <table class="styled-table">
            <thead><tr>
                <th>Ticker</th><th>Peso Tang.</th><th>Beta (β)</th>
                <th>Ret. req. CAPM</th><th>Alpha (α)</th>
                <th>E(r) total</th><th>Contribución</th>
            </tr></thead><tbody>{capm_rows}</tbody>
        </table>""", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # ── Correlation heatmap (from real returns) ────────────────────────────
        st.markdown("**Matriz de Correlación** — calculada con retornos históricos reales")
        fig_corr = go.Figure(go.Heatmap(
            z=corr_mat, x=tickers_mk, y=tickers_mk,
            colorscale=[[0.0,"#1a7a4a"],[0.35,"#d5f5e3"],[0.5,"#f4f7fb"],
                        [0.7,"#fef9e7"],[1.0,"#c0392b"]],
            zmid=0.5,
            text=[[f"{corr_mat[i][j]:.2f}" for j in range(n_mk)] for i in range(n_mk)],
            texttemplate="%{text}", textfont=dict(size=12, color="#1a2332"),
            hovertemplate="<b>%{x} · %{y}</b><br>ρ = %{z:.3f}<extra></extra>",
            showscale=True, colorbar=dict(title="ρ", thickness=14, len=0.85)
        ))
        fig_corr.update_layout(
            height=max(300, n_mk*62), paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="DM Sans", color="#1a2332"),
            margin=dict(l=70,r=20,t=20,b=70), xaxis=dict(side="bottom")
        )
        st.markdown('<div class="chart3d">', unsafe_allow_html=True)
        st.plotly_chart(fig_corr, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:11px;color:#6b7c93;margin-top:4px;'>"
            "🟢 Correlación baja = mejor diversificación &nbsp;·&nbsp; "
            "🔴 Correlación alta = activos se mueven juntos (menor beneficio de diversificar)"
            "</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # ── Variance decomposition ─────────────────────────────────────────────
        wts_np   = np.array([wts_tang.get(t,0) for t in tickers_mk])
        wts_np  /= wts_np.sum()
        port_var = float(wts_np @ cov_mat @ wts_np)
        sigma_m  = 0.18
        syst_var = (port_beta_mk**2) * (sigma_m**2)
        idio_var = max(port_var - syst_var, 0)
        syst_pct = syst_var / port_var * 100 if port_var > 0 else 50
        idio_pct = 100 - syst_pct

        vd1,vd2,vd3,vd4 = st.columns(4)
        for col, lbl, val, sub, col_c in [
            (vd1,"Varianza portafolio",    f"{port_var*100:.4f}%²", "σ²ₚ total ponderada",    "#2d4a6b"),
            (vd2,"Volatilidad Markowitz",  f"{np.sqrt(port_var)*100:.2f}%","√σ²ₚ calculada", "#2d4a6b"),
            (vd3,"Riesgo sistemático",     f"{syst_pct:.1f}%",      "β²σ²ₘ — no diversif.",  "#c0392b"),
            (vd4,"Riesgo idiosincrático",  f"{idio_pct:.1f}%",      "Diversificado por pesos","#1a7a4a"),
        ]:
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label" style="font-size:10px;">{lbl}</div>
                    <div class="kpi-value" style="color:{col_c};font-size:18px;">{val}</div>
                    <div class="kpi-sub" style="font-size:10px;">{sub}</div>
                </div>""", unsafe_allow_html=True)

        dcol1, dcol2 = st.columns([1,2])
        with dcol1:
            fig_var = go.Figure(go.Pie(
                labels=["Sistemático","Idiosincrático"],
                values=[syst_pct, idio_pct], hole=0.55,
                marker_colors=["#c0392b","#1a7a4a"],
                textinfo="label+percent", textfont_size=11,
            ))
            fig_var.update_layout(height=200, paper_bgcolor="white",
                font=dict(family="DM Sans"), margin=dict(l=10,r=10,t=10,b=10),
                showlegend=False,
                annotations=[dict(text="Varianza", x=0.5, y=0.5, font_size=11, showarrow=False)])
            st.plotly_chart(fig_var, width="stretch")
        with dcol2:
            st.markdown(
                f"<div style='background:#f4f7fb;border-radius:12px;padding:16px 18px;"
                f"font-size:12px;color:#1a2332;'>"
                f"<b>Tracking Error vs S&P 500:</b> {tracking_err*100:.2f}% — "
                f"{'Gestión activa bien diferenciada' if tracking_err>0.05 else 'Muy correlado con benchmark'}<br><br>"
                f"<b>Information Ratio:</b> {info_ratio:.3f} — "
                f"{'Gestión activa justificada ✅' if info_ratio>0.5 else 'Alpha insuficiente vs benchmark ⚠️'}<br><br>"
                f"<b>Retorno Tang. vs SPX:</b> {port_ret_mk*100:.2f}% vs {spx_ret*100:.1f}% — "
                f"{'Portafolio supera benchmark ✅' if port_ret_mk>spx_ret else 'Benchmark supera portafolio ⚠️'}"
                f"</div>", unsafe_allow_html=True)

    elif not _mkz_ok:
        st.info("Instalando scipy... ejecuta: pip install scipy --break-system-packages")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Universo completo de activos ──────────────────────────────────────────
    st.markdown('<div class="section-title">🌍 Universo de Activos Analizados</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:13px;color:#6b7c93;margin-bottom:14px;line-height:1.6;'>"
        "Todos los activos evaluados en el proceso de optimización. Los <b>seleccionados</b> "
        "maximizan el Sharpe del portafolio para el perfil indicado. Los <b>descartados</b> "
        "presentaban alta correlación, alpha negativo o redundancia con activos ya incluidos.</div>",
        unsafe_allow_html=True)

    # Tabs: Seleccionados | Descartados | Comparativa
    tab_sel, tab_desc, tab_comp = st.tabs(["✅ Seleccionados", "❌ Descartados / No incluidos", "📊 Comparativa completa"])

    with tab_sel:
        sel_items = [(t, d) for t, d in UNIVERSE_FULL.items() if d[5]]
        for tkr, (name, sector, beta, alpha, reason, selected, ports) in sel_items:
            a_col  = "#1a7a4a" if alpha >= 0 else "#c0392b"
            a_ico  = "▲" if alpha >= 0 else "▼"
            b_col  = "#1a7a4a" if beta < 0.9 else "#2d4a6b" if beta <= 1.2 else "#c0392b"
            ports_str = " · ".join([f"<span style='background:#f0f4f9;padding:2px 7px;border-radius:10px;font-size:10px;'>{p}</span>" for p in ports])
            _h = (
                f"<div style='background:white;border-radius:12px;padding:16px 18px;"
                f"box-shadow:0 2px 10px rgba(13,27,42,0.07);margin-bottom:10px;"
                f"border-left:4px solid #1a7a4a;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:flex-start;'>"
                f"<div>"
                f"<span style='font-weight:700;font-size:15px;color:#1a2332;'>{tkr}</span>"
                f"<span style='font-size:12px;color:#6b7c93;margin-left:8px;'>{name}</span>"
                f"<span style='font-size:11px;color:#2d4a6b;margin-left:8px;background:#eef3f8;"
                f"padding:2px 8px;border-radius:8px;'>{sector}</span>"
                f"</div>"
                f"<div style='display:flex;gap:8px;align-items:center;'>"
                f"<span style='font-size:11px;color:{b_col};font-weight:700;'>β={beta}</span>"
                f"<span style='font-size:11px;color:{a_col};font-weight:700;'>{a_ico}α={abs(alpha):.2f}</span>"
                f"</div>"
                f"</div>"
                f"<div style='font-size:12px;color:#1a2332;margin-top:8px;line-height:1.6;'>{reason}</div>"
                f"<div style='margin-top:8px;'>{ports_str}</div>"
                f"</div>"
            )
            st.markdown(_h, unsafe_allow_html=True)

    with tab_desc:
        desc_items = [(t, d) for t, d in UNIVERSE_FULL.items() if not d[5]]
        for tkr, (name, sector, beta, alpha, reason, selected, ports) in desc_items:
            a_col = "#1a7a4a" if alpha >= 0 else "#c0392b"
            a_ico = "▲" if alpha >= 0 else "▼"
            _h = (
                f"<div style='background:#fafbfc;border-radius:12px;padding:16px 18px;"
                f"box-shadow:0 1px 6px rgba(13,27,42,0.05);margin-bottom:10px;"
                f"border-left:4px solid #c0392b;opacity:0.85;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:flex-start;'>"
                f"<div>"
                f"<span style='font-weight:700;font-size:15px;color:#1a2332;'>{tkr}</span>"
                f"<span style='font-size:12px;color:#6b7c93;margin-left:8px;'>{name}</span>"
                f"<span style='font-size:11px;color:#2d4a6b;margin-left:8px;background:#eef3f8;"
                f"padding:2px 8px;border-radius:8px;'>{sector}</span>"
                f"</div>"
                f"<div style='display:flex;gap:8px;'>"
                f"<span style='font-size:11px;color:#6b7c93;font-weight:700;'>β={beta}</span>"
                f"<span style='font-size:11px;color:{a_col};font-weight:700;'>{a_ico}α={abs(alpha):.2f}</span>"
                f"</div>"
                f"</div>"
                f"<div style='font-size:12px;color:#555;margin-top:8px;line-height:1.6;'>"
                f"⚠ <b>Razón de exclusión:</b> {reason}</div>"
                f"</div>"
            )
            st.markdown(_h, unsafe_allow_html=True)

    with tab_comp:
        all_items = list(UNIVERSE_FULL.items())
        comp_rows = ""
        for tkr, (name, sector, beta, alpha, reason, selected, ports) in sorted(all_items, key=lambda x: -x[1][3]):
            sel_icon  = "✅" if selected else "❌"
            a_col     = "#1a7a4a" if alpha>=0 else "#c0392b"
            b_col     = "#1a7a4a" if beta<0.9 else "#2d4a6b" if beta<=1.2 else "#c0392b"
            er_capm   = (RF_MK + beta*MRP_MK + alpha)*100
            ports_str = ", ".join(ports) if ports else "—"
            comp_rows += (
                f"<tr>"
                f"<td>{sel_icon}</td>"
                f"<td><b>{tkr}</b></td>"
                f"<td style='font-size:11px;color:#6b7c93;'>{name}</td>"
                f"<td style='font-size:11px;color:#2d4a6b;'>{sector}</td>"
                f"<td style='text-align:center;color:{b_col};font-weight:700;'>{beta}</td>"
                f"<td style='text-align:center;color:{a_col};font-weight:700;'>{'▲' if alpha>=0 else '▼'}{abs(alpha):.2f}</td>"
                f"<td style='text-align:center;font-weight:700;'>{er_capm:.1f}%</td>"
                f"<td style='font-size:11px;color:#6b7c93;'>{ports_str}</td>"
                f"</tr>"
            )
        st.markdown(f"""
        <table class="styled-table">
            <thead><tr>
                <th></th><th>Ticker</th><th>Nombre</th><th>Sector</th>
                <th>Beta</th><th>Alpha</th><th>E(r) CAPM</th><th>Portafolios</th>
            </tr></thead><tbody>{comp_rows}</tbody>
        </table>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── P/E Valuation module (continues below) ────────────────────────────────
    _mkz_for_pe = _mkz_ok  # used as flag below

    if True:  # PE/SML always render if we have target_wts
        tickers_mk = list(target_wts.keys())
        wts_mk     = np.array([target_wts.get(t, 0) for t in tickers_mk], dtype=float)
        wts_mk     = wts_mk / wts_mk.sum() if wts_mk.sum() > 0 else wts_mk

        RF_MK  = 0.0264
        MRP_MK = 0.0550

        # Re-use ASSET_DATA already loaded below; define betas/alphas here from defaults
        _AD = {
            "NVDA":(1.83,+0.38),"CAT":(1.07,+0.08),"NEM":(0.88,+0.05),
            "AAPL":(1.08,+0.12),"BATS":(0.62,-0.02),"TSLA":(1.65,+0.15),
            "ASTS":(1.43,-0.18),"NOW":(1.27,+0.22),"JPM":(1.13,+0.06),
            "9988HK":(1.42,-0.09),"SPX":(1.00,0.00),
        }
        betas_mk  = np.array([_AD.get(t,(1.0,0.0))[0] for t in tickers_mk])
        alphas_mk = np.array([_AD.get(t,(1.0,0.0))[1] for t in tickers_mk])
        exp_rets_mk = RF_MK + betas_mk * MRP_MK + alphas_mk

        port_ret_mk   = float(np.dot(wts_mk, exp_rets_mk))
        port_beta_mk  = float(np.dot(wts_mk, betas_mk))
        port_alpha_mk = float(np.dot(wts_mk, alphas_mk))
        port_vol_mk   = risk_metrics.get("vol_annual", 0.20)
        port_sharpe_mk = (port_ret_mk - RF_MK) / port_vol_mk if port_vol_mk > 0 else 0
        port_treynor   = (port_ret_mk - RF_MK) / port_beta_mk if port_beta_mk > 0 else 0
        spx_ret        = RF_MK + MRP_MK
        tracking_err   = port_vol_mk * abs(1 - port_beta_mk)
        info_ratio     = port_alpha_mk / tracking_err if tracking_err > 0 else 0

        # ── 6 Markowitz KPIs ──────────────────────────────────────────────────
        mk1,mk2,mk3,mk4,mk5,mk6 = st.columns(6)
        _sl, _sc = sharpe_label(port_sharpe_mk)
        for col, lbl, val, sub, col_c in [
            (mk1, "Retorno Esperado",   f"{port_ret_mk*100:.2f}%",    f"SPX: {spx_ret*100:.1f}%",   "#1a7a4a" if port_ret_mk>spx_ret else "#d4820a"),
            (mk2, "Beta Ponderado",     f"{port_beta_mk:.3f}",        "< 1 Def · > 1 Agr",          "#1a7a4a" if port_beta_mk<1 else "#c0392b"),
            (mk3, "Alpha Ponderado",    f"{port_alpha_mk:+.3f}",      "Retorno sobre CAPM",          "#1a7a4a" if port_alpha_mk>=0 else "#c0392b"),
            (mk4, "Sharpe Ratio",       f"{port_sharpe_mk:.4f}",      _sl,                           _sc),
            (mk5, "Ratio de Treynor",   f"{port_treynor*100:.2f}%",   "Ret. exceso / Beta",          "#2d4a6b"),
            (mk6, "Information Ratio",  f"{info_ratio:.3f}",          "Alpha / Tracking Error",      "#1a7a4a" if info_ratio>0.5 else "#d4820a"),
        ]:
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label" style="font-size:10px;">{lbl}</div>
                    <div class="kpi-value" style="color:{col_c};font-size:20px;">{val}</div>
                    <div class="kpi-sub" style="font-size:10px;">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── CAPM breakdown table ───────────────────────────────────────────────
        st.markdown("**CAPM por activo** — E(rᵢ) = Rƒ + βᵢ × MRP + αᵢ")
        capm_rows = ""
        for t, w, b, a in zip(tickers_mk, wts_mk, betas_mk, alphas_mk):
            er     = (RF_MK + b*MRP_MK + a)*100
            req    = (RF_MK + b*MRP_MK)*100
            a_col  = "#1a7a4a" if a>=0 else "#c0392b"
            a_ico  = "▲" if a>=0 else "▼"
            capm_rows += (
                f"<tr><td><b>{t}</b></td>"
                f"<td style='text-align:center'>{w*100:.1f}%</td>"
                f"<td style='text-align:center'>{b:.3f}</td>"
                f"<td style='text-align:center'>{req:.2f}%</td>"
                f"<td style='text-align:center;color:{a_col}'>{a_ico} {abs(a*100):.2f}%</td>"
                f"<td style='text-align:center;font-weight:700;color:{a_col}'>{er:.2f}%</td>"
                f"<td style='text-align:center;color:#2d4a6b'>{w*er:.2f}%</td></tr>"
            )
        a_col_p = "#1a7a4a" if port_alpha_mk>=0 else "#c0392b"
        a_ico_p = "▲" if port_alpha_mk>=0 else "▼"
        ret_col_p = "#1a7a4a" if port_ret_mk>spx_ret else "#d4820a"
        capm_rows += (
            f"<tr style='background:#f0f4f9;font-weight:700;'>"
            f"<td>⭐ PORTAFOLIO</td><td style='text-align:center'>100%</td>"
            f"<td style='text-align:center'>{port_beta_mk:.3f}</td>"
            f"<td style='text-align:center'>{(RF_MK+port_beta_mk*MRP_MK)*100:.2f}%</td>"
            f"<td style='text-align:center;color:{a_col_p}'>{a_ico_p} {abs(port_alpha_mk*100):.2f}%</td>"
            f"<td style='text-align:center;color:{ret_col_p}'>{port_ret_mk*100:.2f}%</td>"
            f"<td style='text-align:center'>—</td></tr>"
        )
        st.markdown(f"""
        <table class="styled-table">
            <thead><tr>
                <th>Ticker</th><th>Peso</th><th>Beta (β)</th>
                <th>Ret. req. CAPM</th><th>Alpha (α)</th>
                <th>E(r) total</th><th>Contribución</th>
            </tr></thead><tbody>{capm_rows}</tbody>
        </table>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Correlation heatmap ───────────────────────────────────────────────
        n = len(tickers_mk)
        sigma_m = 0.18
        _idio   = {"NVDA":0.15,"CAT":0.08,"NEM":0.12,"AAPL":0.09,"BATS":0.06,
                   "TSLA":0.18,"ASTS":0.22,"NOW":0.14,"JPM":0.07,"9988HK":0.16}
        sigmas  = betas_mk * sigma_m + np.array([_idio.get(t,0.10) for t in tickers_mk])
        cov_mat = np.outer(betas_mk, betas_mk) * sigma_m**2
        for i in range(n): cov_mat[i,i] = sigmas[i]**2
        std_d   = np.sqrt(np.diag(cov_mat))
        corr_mat = cov_mat / np.outer(std_d, std_d)
        np.fill_diagonal(corr_mat, 1.0)

        st.markdown("**Matriz de correlación estimada** — menor correlación = mejor diversificación")
        fig_corr = go.Figure(go.Heatmap(
            z=corr_mat, x=tickers_mk, y=tickers_mk,
            colorscale=[[0.0,"#1a7a4a"],[0.4,"#d5f5e3"],[0.5,"#f4f7fb"],
                        [0.7,"#fef9e7"],[1.0,"#c0392b"]],
            zmid=0.5,
            text=[[f"{corr_mat[i][j]:.2f}" for j in range(n)] for i in range(n)],
            texttemplate="%{text}", textfont=dict(size=11),
            hovertemplate="<b>%{x} · %{y}</b><br>ρ = %{z:.3f}<extra></extra>",
            showscale=True, colorbar=dict(title="ρ", thickness=12, len=0.8)
        ))
        fig_corr.update_layout(
            height=max(280, n*55), paper_bgcolor="white", plot_bgcolor="white",
            font=dict(family="DM Sans", color="#1a2332"),
            margin=dict(l=60,r=20,t=20,b=60), xaxis=dict(side="bottom")
        )
        st.markdown('<div class="chart3d">', unsafe_allow_html=True)
        st.plotly_chart(fig_corr, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Variance decomposition ────────────────────────────────────────────
        port_var  = float(wts_mk @ cov_mat @ wts_mk)
        syst_var  = (port_beta_mk**2) * (sigma_m**2)
        idio_var  = port_var - syst_var
        syst_pct  = syst_var / port_var * 100 if port_var > 0 else 50
        idio_pct  = 100 - syst_pct

        vd1,vd2,vd3,vd4 = st.columns(4)
        for col, lbl, val, sub in [
            (vd1,"Varianza portafolio",    f"{port_var*100:.4f}%²",      "σ²ₚ total ponderada"),
            (vd2,"Volatilidad Markowitz",  f"{np.sqrt(port_var)*100:.2f}%","√σ²ₚ calculada"),
            (vd3,"Riesgo sistemático",     f"{syst_pct:.1f}%",           "β²σ²ₘ — no diversificable"),
            (vd4,"Riesgo idiosincrático",  f"{idio_pct:.1f}%",           "Diversificado por pesos"),
        ]:
            col_c = "#c0392b" if lbl=="Riesgo sistemático" else "#1a7a4a" if lbl=="Riesgo idiosincrático" else "#2d4a6b"
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label" style="font-size:10px;">{lbl}</div>
                    <div class="kpi-value" style="color:{col_c};font-size:18px;">{val}</div>
                    <div class="kpi-sub" style="font-size:10px;">{sub}</div>
                </div>""", unsafe_allow_html=True)

        dcol1, dcol2 = st.columns([1,2])
        with dcol1:
            fig_var = go.Figure(go.Pie(
                labels=["Sistemático","Idiosincrático"],
                values=[syst_pct, idio_pct], hole=0.55,
                marker_colors=["#c0392b","#1a7a4a"],
                textinfo="label+percent", textfont_size=11,
                hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>"
            ))
            fig_var.update_layout(
                height=200, paper_bgcolor="white",
                font=dict(family="DM Sans"), margin=dict(l=10,r=10,t=10,b=10),
                showlegend=False,
                annotations=[dict(text="Varianza", x=0.5, y=0.5, font_size=11, showarrow=False)]
            )
            st.plotly_chart(fig_var, width="stretch")
        with dcol2:
            st.markdown(
                f"<div style='background:#f4f7fb;border-radius:12px;padding:16px 18px;"
                f"font-size:12px;color:#1a2332;height:100%;'>"
                f"<b>Tracking Error vs S&P 500:</b> {tracking_err*100:.2f}% &nbsp;—&nbsp; "
                f"Mide cuánto se aleja el portafolio del índice. Menor = más parecido al benchmark.<br><br>"
                f"<b>Information Ratio:</b> {info_ratio:.3f} &nbsp;—&nbsp; "
                f"Alpha generado por unidad de desviación del benchmark. "
                f"{'> 0.5 = gestión activa justificada ✅' if info_ratio>0.5 else '< 0.5 = alpha insuficiente vs benchmark ⚠️'}<br><br>"
                f"<b>Retorno esperado portafolio:</b> {port_ret_mk*100:.2f}% &nbsp;vs&nbsp; "
                f"<b>S&P 500:</b> {spx_ret*100:.1f}% &nbsp;—&nbsp; "
                f"{'Portafolio supera al benchmark ✅' if port_ret_mk>spx_ret else 'Benchmark supera al portafolio ⚠️'}"
                f"</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


    # ── P/E Valuation module ─────────────────────────────────────────────────
    # ── P/E + Alpha + Beta + SML ─────────────────────────────────────────────
    st.markdown('<div class="section-title">Valuación por Activo: P/E · Alpha · Beta</div>', unsafe_allow_html=True)

    # (pe_actual, pe_hist_5a, beta, alpha_anual, descripcion)
    # alpha > 0 → supera al mercado ajustado por riesgo
    # ── Load ASSET_DATA from Sheet tab 'Activos', fallback to hardcoded ──────
    _ASSET_DATA_DEFAULT = {
        "NVDA":   (35.2,  55.8,  1.83,  +0.38, "Semiconductores · IA"),
        "CAT":    (17.4,  19.2,  1.07,  +0.08, "Maquinaria Industrial"),
        "NEM":    (22.1,  28.5,  0.88,  +0.05, "Minería de Oro"),
        "AAPL":   (31.8,  28.4,  1.08,  +0.12, "Tecnología Consumer"),
        "BATS":   (8.3,   10.1,  0.62,  -0.02, "Tabaco · Defensivo"),
        "TSLA":   (82.4,  95.3,  1.65,  +0.15, "Vehículos Eléctricos"),
        "ASTS":   (None,  None,  1.43,  -0.18, "Sin utilidades aún"),
        "NOW":    (55.2,  62.1,  1.27,  +0.22, "SaaS Enterprise"),
        "JPM":    (13.1,  12.8,  1.13,  +0.06, "Banca · Financiero"),
        "9988HK": (18.4,  22.0,  1.42,  -0.09, "E-Commerce · China"),
        "SPX":    (22.1,  20.5,  1.00,   0.00, "Índice S&P 500"),
    }
    _asset_from_sheet = load_asset_data_from_sheets(sheet_id) if (use_sheets and sheet_id) else None
    ASSET_DATA = _asset_from_sheet if _asset_from_sheet else _ASSET_DATA_DEFAULT
    _asset_source = "Sheet (Activos)" if _asset_from_sheet else "Demo hardcoded"
    RF   = 0.0264
    MRP  = 0.0550

    if not alerts_df.empty:
        tickers_in_port = alerts_df["Ticker"].tolist()
        pe_cols = st.columns(len(tickers_in_port))

        for i, tkr in enumerate(tickers_in_port):
            dat = ASSET_DATA.get(tkr, (None, None, 1.0, 0.0, ""))
            pe_cur, pe_avg, beta, alpha, desc = dat
            with pe_cols[i]:
                # P/E signal
                if pe_cur is None:
                    val_icon, val_label, val_color, val_bg = "⚪", "S/D", "#6b7c93", "#f4f7fb"
                    pct_pe = None
                else:
                    pct_pe = (pe_cur / pe_avg - 1) * 100
                    if pct_pe < -15:
                        val_icon, val_label, val_color, val_bg = "🟢", f"INFRAVALORADA ({pct_pe:+.0f}%)", "#1a7a4a", "#d5f5e3"
                    elif pct_pe < 10:
                        val_icon, val_label, val_color, val_bg = "🔵", f"PRECIO JUSTO ({pct_pe:+.0f}%)", "#2d4a6b", "#e8f0f8"
                    elif pct_pe < 30:
                        val_icon, val_label, val_color, val_bg = "🟡", f"SOBREVALORADA ({pct_pe:+.0f}%)", "#d4820a", "#fef9e7"
                    else:
                        val_icon, val_label, val_color, val_bg = "🔴", f"MUY SOBREVALUADA ({pct_pe:+.0f}%)", "#c0392b", "#fadbd8"

                # Alpha signal
                alpha_color = "#1a7a4a" if alpha >= 0 else "#c0392b"
                alpha_icon  = "▲" if alpha >= 0 else "▼"
                alpha_label = "Supera mercado" if alpha >= 0 else "Bajo mercado"

                # Beta label
                if beta < 0.8:
                    beta_label, beta_color = "Defensivo", "#1a7a4a"
                elif beta <= 1.2:
                    beta_label, beta_color = "Mercado", "#2d4a6b"
                else:
                    beta_label, beta_color = "Agresivo", "#c0392b"

                pe_cur_str  = f"{pe_cur:.1f}x" if pe_cur else "N/D"
                pe_avg_str  = f"{pe_avg:.1f}x" if pe_avg else "N/D"
                ret_req_str = f"{(RF + beta*MRP)*100:.1f}%"
                html_card = (
                    f"<div style='background:{val_bg}; border-radius:12px; padding:12px 14px;"
                    f"border-left:4px solid {val_color}; font-size:11px;'>"
                    f"<div style='font-weight:700; font-size:13px; color:#1a2332;'>{tkr}</div>"
                    f"<div style='color:#6b7c93; font-size:10px; margin-bottom:8px;'>{desc}</div>"
                    f"<div style='display:flex; justify-content:space-between; margin-bottom:8px;'>"
                    f"<div style='text-align:center;'><div style='color:#6b7c93; font-size:9px;'>P/E HOY</div>"
                    f"<div style='font-weight:700; font-size:15px; color:{val_color};'>{pe_cur_str}</div></div>"
                    f"<div style='text-align:center;'><div style='color:#6b7c93; font-size:9px;'>P/E 5A</div>"
                    f"<div style='font-weight:700; font-size:15px; color:#1a3a5c;'>{pe_avg_str}</div></div>"
                    f"<div style='text-align:center;'><div style='color:#6b7c93; font-size:9px;'>BETA</div>"
                    f"<div style='font-weight:700; font-size:15px; color:{beta_color};'>{beta:.2f}</div></div>"
                    f"<div style='text-align:center;'><div style='color:#6b7c93; font-size:9px;'>ALPHA</div>"
                    f"<div style='font-weight:700; font-size:15px; color:{alpha_color};'>{alpha_icon}{abs(alpha):.2f}</div></div>"
                    f"</div>"
                    f"<div style='border-top:1px solid rgba(0,0,0,0.08); padding-top:6px;'>"
                    f"<span style='background:{val_color}; color:white; padding:2px 8px;"
                    f"border-radius:10px; font-size:10px; font-weight:700;'>{val_icon} {val_label}</span>"
                    f"<span style='margin-left:6px; color:{alpha_color}; font-size:10px; font-weight:600;'>"
                    f"a {alpha_icon} {alpha_label}</span></div>"
                    f"<div style='margin-top:5px; font-size:10px; color:{beta_color};'>"
                    f"b={beta:.2f} {beta_label} · req. {ret_req_str}</div>"
                    f"</div>"
                )
                st.markdown(html_card, unsafe_allow_html=True)

    st.markdown('<div style="font-size:10px; color:#aab; margin-top:6px;">⚠ P/E, Alpha, Beta y Descripción leídos del tab Activos del Sheet. Si no hay conexión se usan valores demo.</div>', unsafe_allow_html=True)

    # ── Theory reference card ─────────────────────────────────────────────────
    with st.expander("📖 Guía de interpretación — Postulados teóricos", expanded=False):
        theory_html = """
        <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; font-size:12px;'>

            <div style='background:#f4f7fb; border-radius:10px; padding:16px; border-top:3px solid #1a3a5c;'>
                <div style='font-weight:700; font-size:13px; color:#1a3a5c; margin-bottom:8px;'>📐 Beta (β) — Riesgo Sistemático</div>
                <div style='line-height:1.8;'>
                    <b style='color:#1a7a4a;'>β &lt; 0.8</b> · Defensivo — menos volátil que el mercado<br>
                    <b style='color:#2d4a6b;'>β = 1.0</b> · Neutro — se mueve igual que el mercado<br>
                    <b style='color:#d4820a;'>β 1.0–1.5</b> · Agresivo — amplifica movimientos del mercado<br>
                    <b style='color:#c0392b;'>β &gt; 1.5</b> · Muy agresivo — alta sensibilidad al ciclo<br>
                    <hr style='border:none; border-top:1px solid #ddd; margin:8px 0;'>
                    <span style='color:#6b7c93; font-size:11px;'>CAPM: E(r) = Rf + β × (Rm − Rf)<br>
                    Rf = 2.64% · MRP = 5.50%</span>
                </div>
            </div>

            <div style='background:#f4f7fb; border-radius:10px; padding:16px; border-top:3px solid #c9a227;'>
                <div style='font-weight:700; font-size:13px; color:#1a3a5c; margin-bottom:8px;'>⭐ Alpha (α) — Retorno Anormal</div>
                <div style='line-height:1.8;'>
                    <b style='color:#1a7a4a;'>α &gt; +0.10</b> · Supera claramente al mercado<br>
                    <b style='color:#2d4a6b;'>α 0 a +0.10</b> · Ligera ventaja — buena señal<br>
                    <b style='color:#d4820a;'>α −0.10 a 0</b> · Por debajo — evaluar posición<br>
                    <b style='color:#c0392b;'>α &lt; −0.10</b> · Destruye valor ajustado por riesgo<br>
                    <hr style='border:none; border-top:1px solid #ddd; margin:8px 0;'>
                    <span style='color:#6b7c93; font-size:11px;'>Fama-French: α positivo sostenido<br>
                    indica ventaja competitiva real</span>
                </div>
            </div>

            <div style='background:#f4f7fb; border-radius:10px; padding:16px; border-top:3px solid #1a7a4a;'>
                <div style='font-weight:700; font-size:13px; color:#1a3a5c; margin-bottom:8px;'>📊 P/E vs Histórico — Valuación</div>
                <div style='line-height:1.8;'>
                    <b style='color:#1a7a4a;'>−15% o más</b> · Infravalorada · señal de compra<br>
                    <b style='color:#2d4a6b;'>−15% a +10%</b> · Precio justo · mantener<br>
                    <b style='color:#d4820a;'>+10% a +30%</b> · Sobrevalorada · precaución<br>
                    <b style='color:#c0392b;'>&gt; +30%</b> · Muy sobrevaluada · riesgo alto<br>
                    <hr style='border:none; border-top:1px solid #ddd; margin:8px 0;'>
                    <span style='color:#6b7c93; font-size:11px;'>Graham: comprar bajo P/E histórico<br>
                    Buffett: P/E justo ≈ crecimiento esperado</span>
                </div>
            </div>

            <div style='background:#f4f7fb; border-radius:10px; padding:16px; border-top:3px solid #c0392b;'>
                <div style='font-weight:700; font-size:13px; color:#1a3a5c; margin-bottom:8px;'>📉 Sharpe Ratio — Eficiencia</div>
                <div style='line-height:1.8;'>
                    <b style='color:#c0392b;'>SR &lt; 0.5</b> · Débil — retorno no justifica el riesgo<br>
                    <b style='color:#d4820a;'>SR 0.5–1.0</b> · Moderado — aceptable<br>
                    <b style='color:#2d4a6b;'>SR 1.0–1.5</b> · Sólido — buen balance riesgo/retorno<br>
                    <b style='color:#1a7a4a;'>SR &gt; 1.5</b> · Muy fuerte — eficiencia alta<br>
                    <hr style='border:none; border-top:1px solid #ddd; margin:8px 0;'>
                    <span style='color:#6b7c93; font-size:11px;'>Markowitz: portafolio eficiente maximiza<br>
                    Sharpe en la frontera eficiente</span>
                </div>
            </div>

            <div style='background:#f4f7fb; border-radius:10px; padding:16px; border-top:3px solid #2d4a6b;'>
                <div style='font-weight:700; font-size:13px; color:#1a3a5c; margin-bottom:8px;'>⚖️ Rebalanceo — Regla de Pesos</div>
                <div style='line-height:1.8;'>
                    <b style='color:#1a7a4a;'>Desviación &lt; 5%</b> · ✅ Mantener — dentro del rango<br>
                    <b style='color:#d4820a;'>Desviación 5–15%</b> · 🟡 Rebalancear con aportación<br>
                    <b style='color:#c0392b;'>Desviación &gt; 15%</b> · 🔴 Rebalancear urgente<br>
                    <hr style='border:none; border-top:1px solid #ddd; margin:8px 0;'>
                    <span style='color:#6b7c93; font-size:11px;'>Regla práctica: rebalancear cada 6 meses<br>
                    o cuando desviación &gt; 5% en cualquier activo</span>
                </div>
            </div>

            <div style='background:#f4f7fb; border-radius:10px; padding:16px; border-top:3px solid #8e0000;'>
                <div style='font-weight:700; font-size:13px; color:#1a3a5c; margin-bottom:8px;'>🎯 SML — Posición Relativa</div>
                <div style='line-height:1.8;'>
                    <b style='color:#1a7a4a;'>Sobre la SML</b> · Punto verde — α positivo · COMPRAR<br>
                    <b style='color:#1a3a5c;'>En la SML</b> · Precio justo para su nivel de riesgo<br>
                    <b style='color:#c0392b;'>Bajo la SML</b> · Punto rojo — α negativo · EVALUAR<br>
                    <hr style='border:none; border-top:1px solid #ddd; margin:8px 0;'>
                    <span style='color:#6b7c93; font-size:11px;'>CAPM: activos sobre SML están<br>
                    subvaluados según su riesgo sistemático</span>
                </div>
            </div>

        </div>
        """
        st.markdown(theory_html, unsafe_allow_html=True)

    # ── SML Chart ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Security Market Line (SML)</div>', unsafe_allow_html=True)

    if not alerts_df.empty:
        tickers_in_port = alerts_df["Ticker"].tolist()

        # SML line: E(r) = RF + beta * MRP
        beta_range = np.linspace(0, 2.2, 100)
        sml_returns = RF + beta_range * MRP

        fig_sml = go.Figure()

        # SML line
        fig_sml.add_trace(go.Scatter(
            x=beta_range, y=sml_returns * 100,
            mode="lines",
            line=dict(color="#1a3a5c", width=2.5, dash="solid"),
            name="SML: E(r) = Rf + β·MRP",
            hovertemplate="β=%{x:.2f}<br>Retorno requerido: %{y:.2f}%<extra>SML</extra>"
        ))

        # Market point
        fig_sml.add_trace(go.Scatter(
            x=[1.0], y=[(RF + MRP) * 100],
            mode="markers+text",
            marker=dict(size=12, color="#1a3a5c", symbol="diamond"),
            text=["S&P500"], textposition="top center",
            textfont=dict(size=10),
            name="Mercado",
            hovertemplate="<b>Mercado (S&P500)</b><br>β=1.00<br>E(r)=%{y:.2f}%<extra></extra>"
        ))

        # Risk-free point
        fig_sml.add_trace(go.Scatter(
            x=[0], y=[RF * 100],
            mode="markers+text",
            marker=dict(size=10, color="#6b7c93", symbol="square"),
            text=["Rf"], textposition="top right",
            textfont=dict(size=10),
            name=f"Tasa libre Rf={RF*100:.2f}%",
            hovertemplate=f"<b>Tasa libre de riesgo</b><br>β=0<br>Rf={RF*100:.2f}%<extra></extra>"
        ))

        # Individual assets
        for tkr in tickers_in_port:
            dat = ASSET_DATA.get(tkr)
            if not dat:
                continue
            pe_c, pe_a, beta, alpha, desc = dat
            ret_requerido = (RF + beta * MRP) * 100
            ret_real      = ret_requerido + alpha * 100   # alpha shifts actual return above/below SML

            above = alpha >= 0
            color  = "#1a7a4a" if above else "#c0392b"
            symbol = "circle" if above else "circle-open"
            tip    = "Sobre SML · COMPRAR" if above else "Bajo SML · EVALUAR"

            # Vertical line from SML to actual point (alpha arrow)
            fig_sml.add_trace(go.Scatter(
                x=[beta, beta],
                y=[ret_requerido, ret_real],
                mode="lines",
                line=dict(color=color, width=1.5, dash="dot"),
                showlegend=False,
                hoverinfo="skip"
            ))

            fig_sml.add_trace(go.Scatter(
                x=[beta], y=[ret_real],
                mode="markers+text",
                marker=dict(size=14, color=color, symbol=symbol,
                            line=dict(width=2, color=color)),
                text=[tkr], textposition="top center",
                textfont=dict(size=10, color=color),
                name=tkr,
                hovertemplate=(
                    f"<b>{tkr}</b> · {desc}<br>"
                    f"β = {beta:.2f}<br>"
                    f"Ret. requerido SML: {ret_requerido:.1f}%<br>"
                    f"Ret. real estimado: {ret_real:.1f}%<br>"
                    f"Alpha: {alpha:+.2f} ({tip})<extra></extra>"
                )
            ))

        # Tangency portfolio point
        port_betas  = [ASSET_DATA.get(t, (None,None,1.0,0.0,""))[2] for t in tickers_in_port]
        port_alphas = [ASSET_DATA.get(t, (None,None,1.0,0.0,""))[3] for t in tickers_in_port]
        port_wts    = [target_wts.get(t, 0) for t in tickers_in_port]
        total_wt    = sum(port_wts) if sum(port_wts) > 0 else 1
        norm_wts    = [w/total_wt for w in port_wts]

        port_beta  = sum(b*w for b,w in zip(port_betas, norm_wts))
        port_alpha = sum(a*w for a,w in zip(port_alphas, norm_wts))
        port_ret_req  = (RF + port_beta * MRP) * 100
        port_ret_real = port_ret_req + port_alpha * 100

        fig_sml.add_trace(go.Scatter(
            x=[port_beta], y=[port_ret_real],
            mode="markers+text",
            marker=dict(size=18, color="#c9a227", symbol="star",
                        line=dict(width=2, color="#1a3a5c")),
            text=["Portafolio"], textposition="top center",
            textfont=dict(size=11, color="#c9a227"),
            name="Portafolio Tangencia",
            hovertemplate=(
                f"<b>⭐ Portafolio</b><br>"
                f"β ponderado = {port_beta:.2f}<br>"
                f"Ret. req. SML: {port_ret_req:.1f}%<br>"
                f"Ret. estimado: {port_ret_real:.1f}%<br>"
                f"Alpha port: {port_alpha:+.2f}<extra></extra>"
            )
        ))

        fig_sml.update_layout(
            height=420,
            paper_bgcolor="white", plot_bgcolor="#fafbfe",
            font=dict(family="DM Sans", color="#1a2332"),
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(
                title="Beta (β) — Riesgo sistemático",
                showgrid=True, gridcolor="#eef2f7", zeroline=True, zerolinecolor="#ddd",
                range=[-0.1, 2.3]
            ),
            yaxis=dict(
                title="Retorno esperado (%)",
                showgrid=True, gridcolor="#eef2f7", ticksuffix="%"
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font_size=10),
            hovermode="closest",
            transition=dict(duration=800, easing="elastic-in"),
        )

        # Shade above/below SML
        fig_sml.add_hrect(
            y0=(RF + 0*MRP)*100, y1=(RF + 2.2*MRP)*100,
            fillcolor="rgba(26,122,74,0.03)", line_width=0,
            annotation_text="Zona ALPHA+", annotation_position="top left",
            annotation_font=dict(color="#1a7a4a", size=10)
        )


        st.markdown(
            "<div style='background:#e8f0f8; border-left:3px solid #1a3a5c; "
            "border-radius:0 8px 8px 0; padding:8px 12px; margin-bottom:6px; "
            "font-size:11px; color:#2d4a6b; line-height:1.5;'>"
            "<b style='font-size:12px;'>ℹ</b>  La <b>SML</b> (Security Market Line) muestra el retorno esperado según el riesgo sistemático (Beta). Puntos <b style='color:#1a7a4a'>verdes</b> sobre la línea = alpha positivo (superan el mercado). Puntos <b style='color:#c0392b'>rojos</b> bajo la línea = alpha negativo (destruyen valor ajustado por riesgo).</div>",
            unsafe_allow_html=True
        )
        st.markdown('<div class="chart3d">', unsafe_allow_html=True)
        st.plotly_chart(fig_sml, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:10px; color:#aab;">⚠ Puntos sobre la SML (verde) = alpha positivo, supera retorno requerido. Puntos bajo la SML (rojo) = alpha negativo. ⭐ = portafolio ponderado.</div>', unsafe_allow_html=True)


    # ── Multi-portfolio comparison ────────────────────────────────────────────
    st.markdown('<div class="section-title">Comparativa de Portafolios</div>', unsafe_allow_html=True)
    
    compare_data = []
    for pname, pdata in get_demo_portfolios().items():
        h = get_demo_history()
        # Add slight variation per portfolio for demo
        noise = {"Tangencia": 1.0, "Conservador": 0.7, "Agresivo": 1.4, "Familiar": 0.6}
        h["Valor"] = h["Valor"] * noise.get(pname, 1.0) + np.random.normal(0, 500, len(h))
        rm = calc_risk_metrics(h)
        compare_data.append({
            "Portafolio": pname,
            "Sharpe": rm.get("sharpe", 0),
            "Volatilidad %": rm.get("vol_annual", 0) * 100,
            "Retorno %": rm.get("ret_annual", 0) * 100,
            "Max DD %": rm.get("max_dd", 0) * 100,
            "VaR 95% %": abs(rm.get("var_95", 0)) * 100,
            "Color": pdata["color"]
        })
    
    comp_df = pd.DataFrame(compare_data)
    
    fig_comp = go.Figure()

    # ── Efficient Frontier / Sharpe concave curve ──────────────────────────────
    vols    = comp_df["Volatilidad %"].values
    rets    = comp_df["Retorno %"].values
    sharpes = comp_df["Sharpe"].values

    # Build smooth concave frontier using the Markowitz risk-return tradeoff
    # interpolate a curve through the portfolios: σ = f(μ) concave in risk-return space
    v_min, v_max = vols.min() * 0.85, vols.max() * 1.10
    r_min, r_max = rets.min() * 0.90, rets.max() * 1.08
    frontier_vols = np.linspace(v_min, v_max, 120)
    # Concave frontier: r = r_min + (r_max - r_min) * sqrt((σ - σ_min) / (σ_max - σ_min))
    frontier_rets = r_min + (r_max - r_min) * np.sqrt(
        np.clip((frontier_vols - v_min) / (v_max - v_min), 0, 1)
    )

    fig_comp.add_trace(go.Scatter(
        x=frontier_vols, y=frontier_rets,
        mode="lines",
        line=dict(color="#e11d48", width=2.5, dash="dot"),
        name="Frontera Eficiente",
        hovertemplate="Frontera: σ=%.1f%% · r=%.1f%%<extra></extra>",
        showlegend=True,
    ))

    # Shaded area under frontier
    fig_comp.add_trace(go.Scatter(
        x=np.concatenate([frontier_vols, frontier_vols[::-1]]),
        y=np.concatenate([frontier_rets, np.full(len(frontier_rets), r_min * 0.95)]),
        fill="toself",
        fillcolor="rgba(37,99,235,0.06)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Tangency line from Rf through best Sharpe portfolio
    rf_pct = RF * 100
    best_idx = np.argmax(sharpes)
    best_v, best_r = vols[best_idx], rets[best_idx]
    slope = (best_r - rf_pct) / best_v if best_v > 0 else 1
    cap_v = np.linspace(0, v_max * 1.15, 80)
    cap_r = rf_pct + slope * cap_v
    fig_comp.add_trace(go.Scatter(
        x=cap_v, y=cap_r,
        mode="lines",
        line=dict(color="#f59e0b", width=2, dash="dash"),
        name="CML (Línea de Mercado de Capital)",
        hoverinfo="skip",
        showlegend=True,
    ))

    # Portfolio bubbles
    for _, row_c in comp_df.iterrows():
        fig_comp.add_trace(go.Scatter(
            x=[row_c["Volatilidad %"]], y=[row_c["Retorno %"]],
            mode="markers+text",
            marker=dict(size=row_c["Sharpe"]*30, color=row_c["Color"],
                       opacity=0.88, line=dict(width=2, color="white")),
            text=[row_c["Portafolio"]],
            textposition="top center",
            textfont=dict(size=11, family="DM Sans"),
            name=row_c["Portafolio"],
            hovertemplate=(
                f"<b>{row_c['Portafolio']}</b><br>"
                f"Riesgo: {row_c['Volatilidad %']:.1f}%<br>"
                f"Retorno: {row_c['Retorno %']:.1f}%<br>"
                f"Sharpe: {row_c['Sharpe']:.3f}<extra></extra>"
            )
        ))

    # Rf point
    fig_comp.add_trace(go.Scatter(
        x=[0], y=[rf_pct],
        mode="markers+text",
        marker=dict(size=10, color="#6b7c93", symbol="diamond"),
        text=["Rf"],
        textposition="top right",
        textfont=dict(size=10, color="#6b7c93"),
        name="Tasa libre de riesgo",
        showlegend=False,
        hovertemplate=f"Tasa libre de riesgo: {rf_pct:.2f}%<extra></extra>",
    ))

    fig_comp.update_layout(
        height=400, paper_bgcolor="white", plot_bgcolor="#fafbfe",
        font=dict(family="DM Sans", color="#1a2332"),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title="Riesgo — Volatilidad Anual (%)", showgrid=True,
                   gridcolor="#eef2f7", zeroline=False),
        yaxis=dict(title="Retorno Anual (%)", showgrid=True,
                   gridcolor="#eef2f7", zeroline=False),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
    )


    st.markdown(
        "<div style='background:#e8f0f8; border-left:3px solid #1a3a5c; "
        "border-radius:0 8px 8px 0; padding:8px 12px; margin-bottom:6px; "
        "font-size:11px; color:#2d4a6b; line-height:1.5;'>"
        "<b style='font-size:12px;'>ℹ</b>  La <b>curva cóncava roja</b> es la Frontera Eficiente de Markowitz — ningún portafolio puede tener más retorno con el mismo riesgo. La <b>línea dorada</b> (CML) conecta la tasa libre de riesgo con el portafolio de máximo Sharpe. Las burbujas más grandes = mayor Sharpe ratio.</div>",
        unsafe_allow_html=True
    )
    st.markdown('<div class="chart3d">', unsafe_allow_html=True)
    st.plotly_chart(fig_comp, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    rows_html = ""
    for _, row_c in comp_df.sort_values("Sharpe", ascending=False).iterrows():
        is_best = row_c["Portafolio"] == comp_df.loc[comp_df["Sharpe"].idxmax(), "Portafolio"]
        sr_lbl, sr_c = sharpe_label(row_c["Sharpe"])
        rows_html += f"""
        <tr {'style="background:#fef9e7;"' if is_best else ''}>
            <td><b style='color:{row_c["Color"]};'>{'★ ' if is_best else ''}{row_c['Portafolio']}</b></td>
            <td style='color:{sr_c};'><b>{row_c['Sharpe']:.3f}</b></td>
            <td>{row_c['Volatilidad %']:.1f}%</td>
            <td style='color:{"#1a7a4a" if row_c["Retorno %"]>0 else "#c0392b"}'>{row_c['Retorno %']:.1f}%</td>
            <td style='color:#c0392b;'>{row_c['Max DD %']:.1f}%</td>
            <td>{row_c['VaR 95% %']:.1f}%</td>
            <td><span style='color:{sr_c}; font-weight:600;'>{sr_lbl}</span></td>
        </tr>"""
    
    st.markdown(f"""
    <table class="styled-table">
        <thead><tr>
            <th>Portafolio</th><th>Sharpe</th><th>Volatilidad</th>
            <th>Retorno</th><th>Max Drawdown</th><th>VaR 95%</th><th>Estado</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ── USUARIOS (admin only) ────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
elif page == "👥 Usuarios":
    if user_data.get("role") != "admin":
        st.error("⛔ Acceso restringido a administradores.")
        st.stop()

    st.markdown("<div class='section-title'>👥 Administración de Usuarios</div>", unsafe_allow_html=True)

    st.info("Los cambios se guardan directamente en **app.py**. Reinicia la app para que el login refleje los cambios.", icon="💡")

    if "admin_users" not in st.session_state:
        st.session_state.admin_users = {k: dict(v) for k, v in USERS.items()}

    users_edit = st.session_state.admin_users
    all_ports  = list(get_demo_portfolios().keys())

    # ── EXISTING USERS ────────────────────────────────────────────────────────
    st.markdown("### Usuarios registrados")
    to_delete = None

    for ukey, ud in list(users_edit.items()):
        ports_now = ud.get("portfolios") or all_ports
        is_admin  = ud.get("role") == "admin"
        icon      = "🔑" if is_admin else "👤"
        with st.expander(f"{icon} **{ud['name']}** · `{ukey}` · {ud['role']}", expanded=False):
            c1, c2, c3 = st.columns([1.2, 1.2, 1])
            with c1:
                ud["name"] = st.text_input("Nombre completo", value=ud["name"], key=f"u_name_{ukey}")
                ud["pin"]  = st.text_input("PIN (4 dígitos)", value=ud["pin"], max_chars=4, type="password", key=f"u_pin_{ukey}")
                ud["role"] = st.selectbox("Rol", ["admin", "investor"], index=0 if ud.get("role") == "admin" else 1, key=f"u_role_{ukey}")
            with c2:
                ud["avatar"] = st.text_input("Avatar (2 letras)", value=ud["avatar"], max_chars=2, key=f"u_av_{ukey}")
                ud["color"]  = st.color_picker("Color", value=ud["color"], key=f"u_col_{ukey}")
            with c3:
                sel = st.multiselect("Portafolios", options=all_ports, default=ports_now, key=f"u_ports_{ukey}")
                ud["portfolios"] = None if set(sel) == set(all_ports) else sel
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                if not is_admin:
                    if st.button(f"🗑 Eliminar", key=f"del_{ukey}", use_container_width=True):
                        to_delete = ukey

    if to_delete:
        del st.session_state.admin_users[to_delete]
        st.rerun()

    st.divider()

    # ── ADD NEW USER ──────────────────────────────────────────────────────────
    st.markdown("### ➕ Agregar usuario")
    with st.expander("Nuevo usuario", expanded=False):
        na1, na2, na3 = st.columns([1.2, 1.2, 1])
        with na1:
            n_key  = st.text_input("Clave única (sin espacios)", key="nu_key", placeholder="ej: pedro").lower().strip()
            n_name = st.text_input("Nombre completo", key="nu_name", placeholder="Pedro Ramírez")
            n_pin  = st.text_input("PIN (4 dígitos)", key="nu_pin", max_chars=4, type="password")
        with na2:
            n_avatar = st.text_input("Avatar (2 letras)", key="nu_av", max_chars=2, placeholder="PR")
            n_color  = st.color_picker("Color", value="#8b5cf6", key="nu_col")
            n_role   = st.selectbox("Rol", ["investor", "admin"], key="nu_role")
        with na3:
            n_ports = st.multiselect("Portafolios", options=all_ports, default=all_ports, key="nu_ports")

        if st.button("Agregar usuario", type="primary", use_container_width=True, key="nu_add"):
            errs = []
            if not n_key:           errs.append("Clave requerida")
            if n_key in users_edit: errs.append(f"Clave '{n_key}' ya existe")
            if len(n_pin) != 4:     errs.append("PIN debe tener 4 dígitos")
            if not n_name:          errs.append("Nombre requerido")
            if not n_avatar:        errs.append("Avatar requerido")
            if errs:
                for e in errs:
                    st.error(e)
            else:
                ports_val = None if set(n_ports) == set(all_ports) else n_ports
                st.session_state.admin_users[n_key] = {
                    "name": n_name, "pin": n_pin, "avatar": n_avatar.upper(),
                    "color": n_color, "sheet_id": "", "portfolios": ports_val, "role": n_role,
                }
                st.success(f"✅ Usuario '{n_name}' agregado. Guarda los cambios para persistir.")
                st.rerun()

    st.divider()

    # ── SAVE ──────────────────────────────────────────────────────────────────
    sc1, sc2 = st.columns([1, 2])
    with sc1:
        if st.button("💾 Guardar todos los cambios", type="primary", use_container_width=True, key="save_users"):
            USERS.clear()
            USERS.update(st.session_state.admin_users)
            ok, msg = write_users_to_file(USERS)
            if ok:
                st.success("✅ " + msg + " — reinicia la app para que el login refleje los cambios")
            else:
                st.error("❌ " + msg)
    with sc2:
        st.markdown(
            "<div style='padding:10px 0;font-size:12px;color:#6b7c93;'>"
            "Guarda para reescribir el bloque <b>USERS</b> en <code>app.py</code>. "
            "Los cambios de PIN y acceso toman efecto al reiniciar la app."
            "</div>", unsafe_allow_html=True
        )

# PAGE: QUICKVIEW (admin only)
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔭 QuickView" and user_data.get("role") == "admin":
    st.markdown("<div class='section-title'>🔭 QuickView — Todos los Portafolios</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:12px;color:#6b7c93;margin-bottom:16px;'>"
        "Vista rápida del estado de todos los usuarios y sus portafolios.</div>",
        unsafe_allow_html=True)

    for uname, udata in USERS.items():
        uports = udata.get("portfolios") or list(get_demo_portfolios().keys())
        if not uports:
            continue
        uid_sheet = udata.get("sheet_id", "")

        st.markdown(f"""
        <div style='background:#1a3a5c;color:white;border-radius:12px 12px 0 0;
             padding:10px 18px;margin-top:18px;display:flex;align-items:center;gap:12px;'>
          <div style='background:#c9a227;border-radius:50%;width:36px;height:36px;
               display:flex;align-items:center;justify-content:center;
               font-weight:800;font-size:14px;'>{udata.get("avatar","?")}</div>
          <div>
            <div style='font-weight:700;font-size:15px;'>{udata.get("name", uname)}</div>
            <div style='font-size:11px;color:#c9d6e3;'>
              {udata.get("role","investor").upper()} · {len(uports)} portafolio(s)
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        for pname in uports:
            port_info = get_demo_portfolios().get(pname, {})
            # Load ops
            u_ops = pd.DataFrame()
            if uid_sheet and use_sheets:
                try:
                    _raw = load_sheet(uid_sheet, f"Ops_{pname}")
                    if _raw is not None and not _raw.empty:
                        u_ops = _raw
                except:
                    pass
            if u_ops.empty:
                u_ops = get_demo_operations(pname)

            # Calc positions
            u_pos = calc_positions(u_ops, {}, 1.0)
            total_usd_u = u_pos["Valor USD"].sum() if not u_pos.empty and "Valor USD" in u_pos.columns else 0
            total_mxn_u = u_pos["Valor MXN"].sum() if not u_pos.empty and "Valor MXN" in u_pos.columns else 0
            n_ops = len(u_ops)

            # Load target weights for this portfolio
            u_tgt = port_info.get("target", {})

            # Risk metrics from sheets if available
            u_rm = {}
            if uid_sheet and use_sheets:
                try:
                    u_ret_df = load_price_history_from_sheets(uid_sheet, pname)
                    if u_ret_df is not None and u_tgt:
                        u_rm = calc_risk_metrics_bloomberg(u_ret_df, u_tgt)
                except:
                    pass

            sharpe_u = u_rm.get("sharpe", 0)
            vol_u    = u_rm.get("vol_annual", 0)
            ret_u    = u_rm.get("ret_annual", 0)
            sharpe_color = "#1a7a4a" if sharpe_u >= 1.0 else "#d4820a" if sharpe_u >= 0.5 else "#c0392b"

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div style='background:white;border:1px solid #e8eef5;border-radius:0 0 0 0;
                     padding:12px 16px;'>
                  <div style='font-size:11px;color:#6b7c93;font-weight:700;'>📁 {pname}</div>
                  <div style='font-size:18px;font-weight:800;color:#1a3a5c;margin-top:4px;'>
                    ${total_mxn_u:,.0f} MXN
                  </div>
                  <div style='font-size:11px;color:#6b7c93;'>${total_usd_u:,.0f} USD</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div style='background:white;border:1px solid #e8eef5;padding:12px 16px;'>
                  <div style='font-size:11px;color:#6b7c93;font-weight:700;'>📋 Operaciones</div>
                  <div style='font-size:18px;font-weight:800;color:#1a3a5c;margin-top:4px;'>
                    {n_ops}
                  </div>
                  <div style='font-size:11px;color:#6b7c93;'>registradas</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div style='background:white;border:1px solid #e8eef5;padding:12px 16px;'>
                  <div style='font-size:11px;color:#6b7c93;font-weight:700;'>⚡ Sharpe</div>
                  <div style='font-size:18px;font-weight:800;color:{sharpe_color};margin-top:4px;'>
                    {sharpe_u:.2f}
                  </div>
                  <div style='font-size:11px;color:#6b7c93;'>Ret {ret_u*100:.1f}% | Vol {vol_u*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                # Weights mini-table
                wts_str = " · ".join([f"{t} {w*100:.0f}%" for t,w in list(u_tgt.items())[:4]])
                st.markdown(f"""
                <div style='background:white;border:1px solid #e8eef5;
                     border-radius:0 0 12px 0;padding:12px 16px;'>
                  <div style='font-size:11px;color:#6b7c93;font-weight:700;'>🎯 Pesos objetivo</div>
                  <div style='font-size:11px;color:#1a3a5c;margin-top:6px;line-height:1.8;'>
                    {wts_str}
                  </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
