"""
EGADA Portfolio Manager v4 — clean rebuild
Core = PORT_MAT-DEEP_PYTHON.py exacto, sin modificaciones al algoritmo.
UI   = mínima pero completa: login, portafolios desde Sheet, dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GS_OK = True
except ImportError:
    GS_OK = False

try:
    import yfinance as yf
    YF_OK = True
except ImportError:
    YF_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# CORE MODEL — traducción línea a línea de PORT_MAT-DEEP_PYTHON.py
# SIN cambios al algoritmo. Solo devuelve un dict con todos los resultados.
# ─────────────────────────────────────────────────────────────────────────────
def get_rf() -> float:
    """^TNX live. Fallback 4.28% igual que el archivo Python."""
    try:
        if YF_OK:
            d = yf.Ticker("^TNX").history(period="1d")["Close"]
            if not d.empty:
                v = float(d.iloc[-1]) / 100.0
                if 0.001 < v < 0.20:
                    return v
    except Exception:
        pass
    return 0.0428

def run_model(Precios: np.ndarray, AssetList: list, rf: float) -> dict:
    """
    Réplica exacta de PORT_MAT-DEEP_PYTHON.py.
    Recibe matriz de precios numpy (filas=fechas, cols=activos) ya limpia.
    """
    max_peso = 0.30
    nAssets = len(AssetList)

    # Verificar SPX
    idx_spx = next((i for i,n in enumerate(AssetList) if "SPX" in n.upper()), None)

    # Eliminar filas con NaN
    if np.any(np.isnan(Precios)):
        mask = ~np.any(np.isnan(Precios), axis=1)
        Precios = Precios[mask, :]

    if len(Precios) < 14:
        return {"error": f"Solo {len(Precios)} filas válidas (mínimo 14)"}

    # Rendimientos logarítmicos
    Rendimientos = np.diff(np.log(Precios), axis=0)
    num_obs = Rendimientos.shape[0]

    # Estadísticos anualizados
    AssetMean  = np.mean(Rendimientos, axis=0) * 12
    AssetCovar = np.cov(Rendimientos, rowvar=False) * 12
    mean_mensual = AssetMean / 12
    cov_mensual  = AssetCovar / 12

    # Funciones
    def port_stats(w):
        ret  = float(np.dot(w, AssetMean))
        risk = float(np.sqrt(w @ AssetCovar @ w))
        return ret, risk

    def neg_sharpe(w):
        ret, risk = port_stats(w)
        return -(ret - rf) / risk if risk > 1e-10 else 0.0

    def portfolio_volatility(w, *_):
        return float(np.sqrt(w @ AssetCovar @ w))

    def portfolio_return(w, *_):
        return float(np.dot(w, AssetMean))

    init_guess  = np.ones(nAssets) / nAssets
    bounds      = tuple((0.0, max_peso) for _ in range(nAssets))
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    # Tangencia
    opt = minimize(neg_sharpe, init_guess, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 5000})
    w_tang = opt.x
    ret_t, risk_t = port_stats(w_tang)
    sharpe_t = (ret_t - rf) / risk_t if risk_t > 1e-10 else 0.0

    # Frontera eficiente (20 puntos)
    numPorts       = 20
    target_returns = np.linspace(AssetMean.min(), AssetMean.max(), numPorts)
    frontier_risk_l, frontier_return_l, frontier_weights_l = [], [], []
    for ret_target in target_returns:
        cons_ret = [
            {"type":"eq","fun": lambda x: np.sum(x) - 1},
            {"type":"eq","fun": lambda x, r=ret_target: portfolio_return(x) - r},
        ]
        o = minimize(portfolio_volatility, init_guess, args=(AssetMean, AssetCovar),
                     method="SLSQP", bounds=bounds, constraints=cons_ret,
                     options={"ftol":1e-10,"maxiter":500})
        if o.success:
            r2, v2 = port_stats(o.x)
            frontier_return_l.append(r2)
            frontier_risk_l.append(v2)
            frontier_weights_l.append(o.x)

    frontier_return  = np.array(frontier_return_l)
    frontier_risk    = np.array(frontier_risk_l)
    frontier_weights = np.array(frontier_weights_l) if frontier_weights_l else np.zeros((0,nAssets))
    n_frontier = len(frontier_risk)

    # SML / Betas
    Betas = np.zeros(nAssets)
    if idx_spx is not None:
        Rm    = Rendimientos[:, idx_spx]
        var_m = np.var(Rm, ddof=1)
        for i in range(nAssets):
            cov_im   = np.cov(Rendimientos[:,i], Rm)[0,1]
            Betas[i] = cov_im / var_m if var_m > 0 else 1.0
    Rm_promedio = np.mean(Rendimientos[:,idx_spx])*12 if idx_spx is not None else rf
    MRP  = Rm_promedio - rf
    CAPM = rf + Betas * MRP
    Alfa = AssetMean - CAPM

    # Correlación
    corr_matrix = np.corrcoef(Rendimientos, rowvar=False)

    # MC 10 000 sims — 1 año
    np.random.seed(0)
    nSim_mc = 10000
    sim_returns = np.zeros(nSim_mc)
    for i in range(nSim_mc):
        sr = np.random.multivariate_normal(mean_mensual, cov_mensual, size=12)
        sim_returns[i] = float(np.prod(1 + sr @ w_tang) - 1)

    ret_p5  = float(np.percentile(sim_returns,  5))
    ret_p25 = float(np.percentile(sim_returns, 25))
    ret_p50 = float(np.percentile(sim_returns, 50))
    ret_p75 = float(np.percentile(sim_returns, 75))
    ret_p95 = float(np.percentile(sim_returns, 95))
    prob_loss = float(np.mean(sim_returns < 0))

    # MC simulaciones G9 (nSim=500, T=24 meses)
    nSim, T_sim = 500, 24
    np.random.seed(0)
    RetSeries = np.random.multivariate_normal(mean_mensual, cov_mensual, size=(nSim, T_sim))
    AccumulatedReturns = np.prod(1 + RetSeries, axis=1) - 1  # (nSim, nAssets)
    if AccumulatedReturns.shape[0] != nAssets:
        AccumulatedReturns = AccumulatedReturns.T

    # VaR/CVaR por activo
    alpha = 0.05
    VaR  = np.zeros(nAssets)
    CVaR = np.zeros(nAssets)
    for i in range(nAssets):
        s    = np.sort(AccumulatedReturns[i])
        idx  = max(1, int(alpha * nSim))
        VaR[i]  = -s[idx]
        CVaR[i] = -np.mean(s[:idx])

    # Intervalos por horizonte [1,3,5,10,20] años
    horizontes = [1, 3, 5, 10, 20]
    rend_h = {}
    np.random.seed(0)
    for años in horizontes:
        meses = años * 12
        sim_h = np.zeros(nSim_mc)
        for i in range(nSim_mc):
            sr       = np.random.multivariate_normal(mean_mensual, cov_mensual, size=meses)
            sim_h[i] = (np.prod(1 + sr @ w_tang) - 1) * 100
        rend_h[años] = {
            "p5":   float(np.percentile(sim_h,  5)),
            "mean": float(np.mean(sim_h)),
            "p95":  float(np.percentile(sim_h, 95)),
        }

    # Riesgo histórico
    port_r = Rendimientos @ w_tang
    cum    = np.exp(np.cumsum(port_r))
    mx_c   = np.maximum.accumulate(cum)
    max_dd = float(np.min((cum - mx_c) / mx_c))
    var95h = float(np.percentile(port_r, 5))
    mask   = port_r < var95h
    cvar95h= float(port_r[mask].mean()) if mask.any() else var95h

    return {
        "AssetList":   AssetList,
        "nAssets":     nAssets,
        "num_obs":     num_obs,
        "rf":          rf,
        "idx_spx":     idx_spx,
        # Tangencia
        "w_tang":      w_tang,
        "ret_t":       ret_t,
        "risk_t":      risk_t,
        "sharpe_t":    sharpe_t,
        # Frontera
        "frontier_risk":    frontier_risk,
        "frontier_return":  frontier_return,
        "frontier_weights": frontier_weights,
        "n_frontier":       n_frontier,
        # SML
        "AssetMean":   AssetMean,
        "Betas":       Betas,
        "Alfa":        Alfa,
        "CAPM":        CAPM,
        "MRP":         MRP,
        # Correlación
        "corr_matrix": corr_matrix,
        # MC 10k
        "sim_returns": sim_returns,
        "ret_p5":      ret_p5,  "ret_p25": ret_p25,
        "ret_p50":     ret_p50, "ret_p75": ret_p75,
        "ret_p95":     ret_p95, "prob_loss": prob_loss,
        # VaR por activo
        "VaR":   VaR,  "CVaR": CVaR,
        # Horizontes
        "rend_h": rend_h,
        # Riesgo histórico
        "max_dd":  max_dd,
        "var95h":  var95h,
        "cvar95h": cvar95h,
        "error":   None,
    }

# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE SHEETS — leer precios
# ─────────────────────────────────────────────────────────────────────────────
SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive"]

@st.cache_resource
def get_gc():
    if not GS_OK: return None
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=SCOPES)
        return gspread.authorize(creds)
    except Exception:
        return None

def parse_num(s: str) -> float:
    """Parsea cualquier formato numérico de Google Sheets."""
    s = str(s).strip().replace(" ","")
    if not s: return float("nan")
    if "," in s and "." in s:
        s = s.replace(".","").replace(",",".") if s.rfind(",")>s.rfind(".") else s.replace(",","")
    elif "," in s:
        parts = s.split(",")
        s = s.replace(",",".") if len(parts)==2 and len(parts[1])<=2 else s.replace(",","")
    try:    return float(s)
    except: return float("nan")

@st.cache_data(ttl=300)
def sheet_load(sheet_id: str, tab: str) -> pd.DataFrame:
    """Lee cualquier hoja del Sheet."""
    gc = get_gc()
    if gc is None: return pd.DataFrame()
    try:
        data = gc.open_by_key(sheet_id).worksheet(tab).get_all_values()
        if len(data) < 2: return pd.DataFrame()
        return pd.DataFrame(data[1:], columns=data[0])
    except Exception:
        return pd.DataFrame()

def load_prices_for_model(sheet_id: str, port_name: str):
    """
    Lee Precios_<port_name> y devuelve:
      - Precios (np.ndarray)  — solo filas donde TODOS tienen precio
      - AssetList (list)
      - fechas_raw (list)     — para diagnóstico
      - raw_df (DataFrame)    — sin filtrar, para diagnóstico
    """
    gc = get_gc()
    if gc is None:
        return None, [], [], pd.DataFrame()
    try:
        ws   = gc.open_by_key(sheet_id).worksheet(f"Precios_{port_name}")
        data = ws.get_all_values()
        if len(data) < 3:
            return None, [], [], pd.DataFrame()

        headers = data[0]
        fecha_idx = None
        asset_hdrs = []   # (col_idx, ticker)
        for j, col in enumerate(headers):
            c = col.strip()
            if c.lower() in ("fecha","date"):
                fecha_idx = j
            elif c:
                asset_hdrs.append((j, c.upper()))

        if len(asset_hdrs) < 2:
            return None, [], [], pd.DataFrame()

        AssetList = [t for _, t in asset_hdrs]
        nA = len(AssetList)

        # Leer todas las filas (con NaN para valores faltantes)
        fechas_raw, price_rows = [], []
        for row in data[1:]:
            if all(str(v).strip() == "" for v in row): continue
            fecha = row[fecha_idx] if fecha_idx is not None and fecha_idx < len(row) else ""
            fechas_raw.append(fecha)
            vals = []
            for j, _ in asset_hdrs:
                raw = row[j] if j < len(row) else ""
                vals.append(parse_num(raw))
            price_rows.append(vals)

        raw_df = pd.DataFrame(price_rows, columns=AssetList)
        raw_df.insert(0, "__fecha__", fechas_raw)

        # Filtrar filas donde TODOS tienen precio > 0 (igual que MATLAB)
        price_data = np.array(price_rows, dtype=float)
        valid_mask = np.all(price_data > 0, axis=1) & ~np.any(np.isnan(price_data), axis=1)
        Precios = price_data[valid_mask]

        if len(Precios) < 14:
            return Precios, AssetList, fechas_raw, raw_df

        return Precios, AssetList, [fechas_raw[i] for i in range(len(fechas_raw)) if valid_mask[i]], raw_df

    except Exception as e:
        return None, [], [], pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# USERS
# ─────────────────────────────────────────────────────────────────────────────
USERS = {  # USERS_START
    "david": {
        "name":     "David Jassan",
        "pin":      "1234",
        "avatar":   "DJ",
        "color":    "#1a3a5c",
        "sheet_id": "1eApNRcJSnqYYkUxK2uDWqUOwXh6lNkoOZ-zvzVSoKFw",
        "role":     "admin",
    },
}  # USERS_END

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="EGADA v4", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*{font-family:'Inter',sans-serif;}
.stApp,[data-testid="stAppViewContainer"]{background:#f5f7fa!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b2a,#162233)!important;}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important;}
.main .block-container{padding:1.5rem 2rem 3rem;}
.card{background:white;border-radius:12px;padding:18px 20px;
      border:1px solid #e5e7eb;box-shadow:0 1px 4px rgba(0,0,0,.05);margin-bottom:4px;}
.kpi-lbl{font-size:10px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
         color:#6b7280;margin-bottom:6px;}
.kpi-val{font-size:28px;font-weight:700;line-height:1;margin-bottom:4px;}
.kpi-sub{font-size:11px;color:#6b7280;}
.pos{color:#059669;} .neg{color:#ef4444;}
.sem{background:white;border-radius:12px;padding:14px;text-align:center;
     border:1px solid #e5e7eb;}
.tbl{overflow-x:auto;border-radius:12px;border:1px solid #e5e7eb;background:white;}
table{width:100%;border-collapse:collapse;font-size:12px;}
th{background:#1a3a5c;color:white;padding:9px 12px;font-size:10px;
   text-transform:uppercase;letter-spacing:.07em;}
td{padding:8px 12px;text-align:center;border-bottom:1px solid #f3f4f6;}
tr:last-child td{border-bottom:none;}
tr:hover td{background:#f9fafb;}
.sec{font-size:18px;font-weight:600;margin:20px 0 12px;
     border-bottom:2px solid rgba(201,162,39,.3);padding-bottom:6px;}
.diag{border-radius:10px;padding:10px 16px;font-size:12px;
      margin-bottom:12px;border-left:4px solid;}
.ok{background:#ecfdf5;border-color:#059669;}
.warn{background:#fffbeb;border-color:#d97706;}
.err{background:#fef2f2;border-color:#ef4444;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("auth", False), ("user", None), ("sel_login", None),
              ("portfolios", {}), ("pin_err", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.auth:
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("<h2 style='text-align:center;margin-bottom:24px;'>EGADA v4</h2>",
                    unsafe_allow_html=True)
        if not st.session_state.sel_login:
            for uk, ud in USERS.items():
                if st.button(f"{ud['avatar']} — {ud['name']} ({ud['role']})",
                             key=f"lb_{uk}", use_container_width=True):
                    st.session_state.sel_login = uk
                    st.rerun()
        else:
            uk = st.session_state.sel_login
            ud = USERS[uk]
            st.write(f"**{ud['name']}** — PIN:")
            pin = st.text_input("", type="password", max_chars=4,
                                key="pin_f", label_visibility="collapsed")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("← Volver", use_container_width=True):
                    st.session_state.sel_login = None; st.rerun()
            with c2:
                if st.button("Entrar →", type="primary", use_container_width=True):
                    if pin == ud["pin"]:
                        st.session_state.auth = True
                        st.session_state.user = uk
                        st.rerun()
                    else:
                        st.error("PIN incorrecto")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# AUTHENTICATED
# ─────────────────────────────────────────────────────────────────────────────
cur = st.session_state.user
ud  = USERS[cur]
is_a = ud.get("role") == "admin"
sheet_id = ud.get("sheet_id", "").strip()

# Detect portfolio switch → clear cache
if st.session_state.get("_last_port") != st.session_state.get("sel_port"):
    st.session_state["_last_port"] = st.session_state.get("sel_port")
    st.cache_data.clear()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""<div style='padding:16px 0 12px;color:white;'>
        <div style='font-size:22px;font-weight:700;'>EGADA <span style='color:#c9a227;'>v4</span></div>
        <div style='font-size:11px;color:#6b7280;margin-top:2px;'>Portfolio Manager</div>
    </div>""", unsafe_allow_html=True)

    # TC
    tc = 20.50
    if YF_OK:
        try:
            v = getattr(yf.Ticker("MXN=X").fast_info, "last_price", None)
            if v and 15 < float(v) < 35: tc = round(float(v), 2)
        except Exception: pass

    st.markdown(f"""<div style='background:rgba(255,255,255,.08);border-radius:10px;
                    padding:10px 14px;margin-bottom:12px;'>
        <div style='font-size:9px;color:#c9a227;font-weight:700;text-transform:uppercase;'>
            USD/MXN</div>
        <div style='font-size:22px;font-weight:800;color:white;'>{tc:.2f}</div>
    </div>""", unsafe_allow_html=True)

    if st.button("⬡ Cerrar sesión", use_container_width=True):
        st.session_state.auth = False
        st.session_state.user = None
        st.rerun()

    st.divider()

    # Portafolios
    all_ports = st.session_state.portfolios
    st.markdown("<div style='font-size:10px;font-weight:700;color:#c9a227;"
               "text-transform:uppercase;margin-bottom:8px;'>Portafolio Activo</div>",
               unsafe_allow_html=True)

    if not all_ports:
        st.markdown("<div style='color:rgba(255,255,255,.4);font-size:12px;'>"
                   "Sin portafolios. Ve a Admin.</div>", unsafe_allow_html=True)
        selected = None
    else:
        if "sel_port" not in st.session_state or \
           st.session_state.sel_port not in all_ports:
            st.session_state.sel_port = list(all_ports.keys())[0]
        for pn, pc in all_ports.items():
            sel = st.session_state.sel_port == pn
            bg  = pc["color"] if sel else "rgba(255,255,255,.07)"
            st.markdown(f"<div style='background:{bg};border-radius:9px;"
                       f"padding:9px 12px;margin-bottom:2px;pointer-events:none;'>"
                       f"<div style='font-size:13px;font-weight:{'700' if sel else '400'};"
                       f"color:white;'>{pn}</div>"
                       f"<div style='font-size:10px;color:rgba(255,255,255,.5);'>"
                       f"{pc.get('desc','')}</div></div>", unsafe_allow_html=True)
            if st.button("​", key=f"pb_{pn}", use_container_width=True):
                st.session_state.sel_port = pn
                st.cache_data.clear()
                st.rerun()
        selected = st.session_state.sel_port

    st.divider()

    if is_a:
        tc = st.number_input("TC MXN/USD", value=tc, step=0.01, format="%.2f")
        thresh_reb = st.slider("Umbral rebalanceo", 0.03, 0.15, 0.05, 0.01,
                               format="%.0f%%")
    else:
        thresh_reb = 0.05

    st.divider()
    pages = ["📊 Modelo", "📡 Posiciones", "📋 Operaciones", "⚡ Riesgo",
             "👥 Admin"] if is_a else ["📊 Modelo", "📡 Posiciones", "📋 Operaciones"]
    page = st.radio("nav", pages, label_visibility="collapsed")

# ─────────────────────────────────────────────────────────────────────────────
# SIN PORTAFOLIO
# ─────────────────────────────────────────────────────────────────────────────
if not selected and page != "👥 Admin":
    st.markdown("<div style='text-align:center;padding:60px;'>"
               "<div style='font-size:48px;'>📂</div>"
               "<div style='font-size:22px;font-weight:700;margin:12px 0;'>"
               "Sin portafolios</div>"
               "<div style='color:#6b7280;'>Ve a Admin para crear uno.</div>"
               "</div>", unsafe_allow_html=True)
    if page != "👥 Admin": st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# CARGAR DATOS
# ─────────────────────────────────────────────────────────────────────────────
if selected:
    pcfg = all_ports.get(selected, {})

    # Precios del Excel → modelo
    @st.cache_data(ttl=300)
    def cached_model(sid, pname):
        Precios, AssetList, fechas, raw_df = load_prices_for_model(sid, pname)
        if Precios is None or len(Precios) < 14:
            return None, AssetList, fechas, raw_df
        rf = get_rf()
        result = run_model(Precios, AssetList, rf)
        return result, AssetList, fechas, raw_df

    result, AssetList, fechas_validas, raw_df = cached_model(sheet_id, selected)

    # Pesos objetivo (w > 0.01 igual que MATLAB)
    target_wts = {}
    if result and not result.get("error"):
        for i, t in enumerate(result["AssetList"]):
            if result["w_tang"][i] > 0.01:
                target_wts[t] = float(result["w_tang"][i])

    # Diagnóstico de datos
    n_raw    = len(raw_df) if not raw_df.empty else 0
    n_valid  = len(fechas_validas)
    n_drop   = n_raw - n_valid
    missing  = []
    if not raw_df.empty and n_drop > 0:
        for c in [x for x in raw_df.columns if not x.startswith("__")]:
            n_nan = int(raw_df[c].apply(lambda v: pd.isna(v) or parse_num(str(v)) <= 0).sum())
            if n_nan > 0:
                missing.append(f"<b>{c}</b>: {n_nan} filas sin precio")

    if not result or result.get("error"):
        diag_cls = "err"
        diag_msg = (f"❌ {result.get('error','Sin datos')} — "
                    f"Hoja Precios_{selected} · {n_raw} filas totales · "
                    f"{n_valid} filas válidas · {n_drop} descartadas")
    elif n_drop > 0:
        diag_cls = "warn"
        diag_msg = (f"⚠ Precios_{selected} · {n_raw} filas en Sheet · "
                    f"<b>{n_valid} usadas</b> · "
                    f"<b style='color:#ef4444;'>{n_drop} descartadas</b> "
                    f"(datos faltantes)<br>"
                    f"{'  ·  '.join(missing)}<br>"
                    f"<b>Solución:</b> Llena todas las celdas vacías para ese período.")
    else:
        n_obs = result.get("num_obs", n_valid-1)
        rf_v  = result.get("rf", 0)
        f0    = fechas_validas[0]  if fechas_validas else "—"
        f1    = fechas_validas[-1] if fechas_validas else "—"
        diag_cls = "ok"
        diag_msg = (f"✅ Precios_{selected} · <b>{n_valid} filas</b> · "
                    f"<b>{n_obs} retornos</b> · {f0} → {f1} · "
                    f"RF: <b>{rf_v*100:.2f}%</b> · "
                    f"Activos: <b>{', '.join(AssetList)}</b>")

    st.markdown(f"<div class='diag {diag_cls}'>{diag_msg}</div>",
               unsafe_allow_html=True)

    # Operaciones
    ops_df = pd.DataFrame()
    if sheet_id:
        raw_ops = sheet_load(sheet_id, f"Ops_{selected}")
        if not raw_ops.empty:
            for col in ["Cantidad","Precio_USD","Comision_USD","TC_MXN"]:
                if col in raw_ops.columns:
                    raw_ops[col] = pd.to_numeric(raw_ops[col], errors="coerce").fillna(0)
            ops_df = raw_ops

    # Precios live (Yahoo)
    yf_prices = {}
    if YF_OK and target_wts:
        try:
            for t in list(target_wts.keys()):
                sym = "^GSPC" if t == "SPX" else t
                px  = getattr(yf.Ticker(sym).fast_info, "last_price", None)
                if px: yf_prices[t] = float(px)
        except Exception: pass

    # Precios para valorar: YF si está disponible, último Excel si no
    last_excel = {}
    if not raw_df.empty:
        lr = raw_df.iloc[-1]
        for c in [x for x in raw_df.columns if not x.startswith("__")]:
            v = parse_num(str(lr[c]))
            if v > 0: last_excel[c] = v
    prices = {**last_excel, **yf_prices}

    # Posiciones
    def calc_pos():
        tks = list(target_wts.keys())
        if not ops_df.empty:
            for t in ops_df["Ticker"].unique():
                if t not in tks: tks.append(t)
        rows = []
        for t in tks:
            qty = cost = 0.0
            if not ops_df.empty and t in ops_df["Ticker"].values:
                sub = ops_df[ops_df["Ticker"] == t]
                qc  = sub[sub["Tipo"]=="Compra"]["Cantidad"].sum()
                qv  = sub[sub["Tipo"]=="Venta"]["Cantidad"].sum()
                qty = qc - qv
                cost= ((sub[sub["Tipo"]=="Compra"]["Cantidad"] *
                        sub[sub["Tipo"]=="Compra"]["Precio_USD"]).sum() / qc
                       if qc > 0 else 0.0)
            px   = prices.get(t, 0.0)
            rows.append({"Ticker":t,"Cantidad":qty,"Precio":px,
                         "Valor USD":qty*px,"Valor MXN":qty*px*tc,
                         "Costo Prom":cost,
                         "Ganancia USD":(px-cost)*qty,
                         "Peso Obj": target_wts.get(t,0)})
        return pd.DataFrame(rows)

    pos_df = calc_pos()
    total_mxn  = pos_df["Valor MXN"].sum() if not pos_df.empty else 0.0
    total_usd  = pos_df["Valor USD"].sum()  if not pos_df.empty else 0.0
    total_gain = pos_df["Ganancia USD"].sum() if not pos_df.empty else 0.0
    cost_base  = total_usd - total_gain
    gan_pct    = (total_gain/cost_base*100) if cost_base > 0 else 0.0

    # Peso actual y alertas
    if not pos_df.empty and total_mxn > 0:
        pos_df["Peso Actual"] = pos_df["Valor MXN"] / total_mxn
    else:
        pos_df["Peso Actual"] = 0.0
    pos_df["Desviacion"] = pos_df["Peso Actual"] - pos_df["Peso Obj"]
    def _alerta(r):
        if r["Peso Obj"]==0 and r["Peso Actual"]>0: return "VENDER"
        if r["Desviacion"] >  thresh_reb: return "VENDER"
        if r["Desviacion"] < -thresh_reb: return "COMPRAR"
        return "OK"
    pos_df["Alerta"] = pos_df.apply(_alerta, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: MODELO ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Modelo" and selected:
    if not result or result.get("error"):
        st.warning("Modelo no disponible. Verifica que la hoja del Sheet tenga datos completos.")
        st.stop()

    st.markdown(f"<div class='sec'>📊 Modelo Markowitz — {selected}</div>",
               unsafe_allow_html=True)

    # ── 4 KPIs del modelo ─────────────────────────────────────────────────
    k1,k2,k3,k4 = st.columns(4)
    sr = result["sharpe_t"]
    src = "#059669" if sr>=1 else ("#d97706" if sr>=0.5 else "#ef4444")
    with k1:
        st.markdown(f"""<div class='card'>
            <div class='kpi-lbl'>Sharpe Ratio</div>
            <div class='kpi-val' style='color:{src};'>{sr:.3f}</div>
            <div class='kpi-sub'>{"Excelente" if sr>=1 else "Aceptable" if sr>=0.5 else "Débil"}</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class='card'>
            <div class='kpi-lbl'>Retorno Esperado Anual</div>
            <div class='kpi-val pos'>{result["ret_t"]*100:.2f}%</div>
            <div class='kpi-sub'>Portafolio tangencia</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class='card'>
            <div class='kpi-lbl'>Volatilidad Anual</div>
            <div class='kpi-val' style='color:#d97706;'>{result["risk_t"]*100:.2f}%</div>
            <div class='kpi-sub'>Riesgo del portafolio</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class='card'>
            <div class='kpi-lbl'>Intervalo 90% (1 año)</div>
            <div class='kpi-val' style='font-size:18px;color:#1a3a5c;'>
                {result["ret_p5"]*100:.1f}% — {result["ret_p95"]*100:.1f}%</div>
            <div class='kpi-sub'>RF: {result["rf"]*100:.2f}% · {result["num_obs"]} retornos</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Composición óptima ────────────────────────────────────────────────
    st.markdown(f"<div class='sec'>🎯 Composición Óptima (w > 1%)</div>",
               unsafe_allow_html=True)
    pal = ["#1a3a5c","#c9a227","#059669","#ef4444","#5b4fcf",
           "#0e7490","#b45309","#6d28d9","#be185d","#0891b2"]
    tw_sorted = sorted(target_wts.items(), key=lambda x: -x[1])
    wcols = st.columns(min(len(tw_sorted), 5))
    for qi, (t, w) in enumerate(tw_sorted):
        c2 = pal[qi % len(pal)]
        with wcols[qi % len(wcols)]:
            st.markdown(
                f"<div class='card' style='text-align:center;border-top:4px solid {c2};'>"
                f"<div style='font-size:14px;font-weight:700;'>{t}</div>"
                f"<div style='font-size:24px;font-weight:800;color:{c2};'>{w*100:.1f}%</div>"
                f"</div>", unsafe_allow_html=True)

    # Activos con peso < 1% (descartados)
    disc = [f"{t} ({result['w_tang'][i]*100:.2f}%)"
            for i, t in enumerate(result["AssetList"])
            if result["w_tang"][i] <= 0.01]
    if disc:
        st.caption(f"Descartados por optimizador (peso ≤ 1%): {', '.join(disc)}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Gráficos en tabs ──────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs(["🗺 Frontera + CML",
                               "📈 Intervalos Horizonte",
                               "🔵 Correlaciones",
                               "📊 SML"])
    with t1:
        fig = go.Figure()
        fr  = result["frontier_risk"]
        fret= result["frontier_return"]
        fig.add_trace(go.Scatter(x=fr*100, y=fret*100,
            mode="lines+markers", name="Frontera Eficiente",
            line=dict(color="#1a3a5c",width=2.5),
            marker=dict(size=7, color="#1a3a5c"),
            hovertemplate="σ=%{x:.2f}%  R=%{y:.2f}%<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[result["risk_t"]*100], y=[result["ret_t"]*100],
            mode="markers", name=f"Tangencia  SR={sr:.3f}",
            marker=dict(size=16,color="#c9a227",symbol="star",
                        line=dict(width=2,color="#0d1b2a"))))
        rf_p = result["rf"]*100
        vt   = result["risk_t"]*100
        rt   = result["ret_t"]*100
        xc   = [0, max(fr)*110 if len(fr) else 30]
        yc   = [rf_p, rf_p + (rt-rf_p)/vt*xc[1]] if vt > 0 else [rf_p, rf_p]
        fig.add_trace(go.Scatter(x=xc, y=yc, mode="lines",
            name=f"CML (RF={rf_p:.2f}%)",
            line=dict(color="#ef4444",dash="dash",width=1.5)))
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(243,244,246,.8)",
            xaxis=dict(title="Riesgo σ (%)", gridcolor="#e5e7eb"),
            yaxis=dict(title="Retorno (%)", gridcolor="#e5e7eb"),
            margin=dict(l=10,r=10,t=10,b=10),
            legend=dict(orientation="h",y=1.05,font_size=10))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        rh  = result["rend_h"]
        hrs = list(rh.keys())
        means = [rh[y]["mean"]  for y in hrs]
        p5s   = [rh[y]["p5"]    for y in hrs]
        p95s  = [rh[y]["p95"]   for y in hrs]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=hrs+hrs[::-1], y=p95s+p5s[::-1],
            fill="toself", fillcolor="rgba(26,58,92,.12)",
            line=dict(color="rgba(0,0,0,0)"), name="Intervalo 90%"))
        fig2.add_trace(go.Scatter(
            x=hrs, y=means, mode="lines+markers",
            line=dict(color="#1a3a5c",width=2.5),
            marker=dict(size=9,color="#c9a227"),
            name="Rendimiento esperado",
            hovertemplate="<b>%{x} años</b><br>%{y:.1f}%<extra></extra>"))
        fig2.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(243,244,246,.8)",
            xaxis=dict(title="Horizonte (años)",gridcolor="#e5e7eb"),
            yaxis=dict(title="Rendimiento total (%)",gridcolor="#e5e7eb"),
            margin=dict(l=10,r=10,t=10,b=10),
            legend=dict(orientation="h",y=1.05,font_size=10))
        st.plotly_chart(fig2, use_container_width=True)

        # Tabla resumen
        rows_h = ""
        for y in hrs:
            rows_h += (f"<tr><td><b>{y} año{'s' if y>1 else ''}</b></td>"
                       f"<td>{rh[y]['p5']:.1f}%</td>"
                       f"<td>{rh[y]['mean']:.1f}%</td>"
                       f"<td>{rh[y]['p95']:.1f}%</td></tr>")
        st.markdown(
            f"<div class='tbl'><table><thead><tr>"
            f"<th>Horizonte</th><th>P5 (pesimista)</th>"
            f"<th>Esperado</th><th>P95 (optimista)</th>"
            f"</tr></thead><tbody>{rows_h}</tbody></table></div>",
            unsafe_allow_html=True)

    with t3:
        cm  = result["corr_matrix"]
        als = result["AssetList"]
        fig3 = go.Figure(go.Heatmap(
            z=cm, x=als, y=als, colorscale="RdBu_r", zmin=-1, zmax=1,
            text=[[f"{cm[i][j]:.2f}" for j in range(len(als))]
                  for i in range(len(als))],
            texttemplate="%{text}", textfont=dict(size=9),
            hovertemplate="<b>%{x} vs %{y}</b>: %{z:.3f}<extra></extra>"))
        fig3.update_layout(title="Matriz de Correlación",
            height=420, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with t4:
        betas = result["Betas"]
        als   = result["AssetList"]
        amu   = result["AssetMean"]
        alfa  = result["Alfa"]
        idx_s = result["idx_spx"]
        fig4 = go.Figure()
        x_l = [0, float(np.max(betas))*1.1]
        y_l = [result["rf"]*100, (result["rf"] + x_l[1]*result["MRP"])*100]
        fig4.add_trace(go.Scatter(x=x_l, y=y_l, mode="lines",
            line=dict(color="#ef4444",width=2), name="SML"))
        for i, t in enumerate(als):
            color = "#ef4444" if i == idx_s else "#1a3a5c"
            size  = 14 if i == idx_s else 9
            fig4.add_trace(go.Scatter(
                x=[betas[i]], y=[amu[i]*100],
                mode="markers+text",
                marker=dict(size=size, color=color,
                            line=dict(width=1, color="white")),
                text=[f"{t} (α={alfa[i]*100:+.1f}%)"],
                textposition="top center", textfont=dict(size=8),
                name=t, showlegend=False))
        fig4.update_layout(
            title="Security Market Line (SML)",
            height=420, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(243,244,246,.8)",
            xaxis=dict(title="Beta", gridcolor="#e5e7eb"),
            yaxis=dict(title="Retorno Esperado (%)", gridcolor="#e5e7eb"),
            margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: POSICIONES (Yahoo Finance) ─────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📡 Posiciones" and selected:
    st.markdown(f"<div class='sec'>📡 Posiciones · {selected}</div>",
               unsafe_allow_html=True)

    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class='card'>
            <div class='kpi-lbl'>Valor Total</div>
            <div class='kpi-val'>${total_mxn:,.0f} MXN</div>
            <div class='kpi-sub'>${total_usd:,.0f} USD · TC {tc:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        sg = "pos" if total_gain>=0 else "neg"
        si = "+" if total_gain>=0 else ""
        st.markdown(f"""<div class='card'>
            <div class='kpi-lbl'>Ganancia / Pérdida</div>
            <div class='kpi-val {sg}'>{si}${total_gain:,.0f} USD</div>
            <div class='kpi-sub'>{si}{gan_pct:.1f}% sobre costo</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        yf_src = "📡 Yahoo Finance" if yf_prices else "📊 Excel (YF no disponible)"
        st.markdown(f"""<div class='card'>
            <div class='kpi-lbl'>Fuente de precios</div>
            <div class='kpi-val' style='font-size:14px;color:#1a3a5c;'>{yf_src}</div>
            <div class='kpi-sub'>{len(yf_prices)} tickers con precio live</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        n_ops = len(ops_df) if not ops_df.empty else 0
        st.markdown(f"""<div class='card'>
            <div class='kpi-lbl'>Operaciones</div>
            <div class='kpi-val' style='color:#1a3a5c;'>{n_ops}</div>
            <div class='kpi-sub'>En Ops_{selected}</div>
        </div>""", unsafe_allow_html=True)

    # Tabla posiciones + donut
    pc1, pc2 = st.columns([3,2])
    with pc1:
        def badge(a):
            if a=="COMPRAR": return "<span style='background:#dcfce7;color:#166534;padding:2px 8px;border-radius:8px;font-size:10px;font-weight:700;'>🟢 COMPRAR</span>"
            if a=="VENDER":  return "<span style='background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:8px;font-size:10px;font-weight:700;'>🔴 VENDER</span>"
            return "<span style='background:#f0fdf4;color:#166534;padding:2px 8px;border-radius:8px;font-size:10px;font-weight:700;'>✅ OK</span>"
        rows_t = ""
        for _, r in pos_df.iterrows():
            gc2 = "#059669" if r["Ganancia USD"]>=0 else "#ef4444"
            gs  = "+" if r["Ganancia USD"]>=0 else ""
            dc  = "#ef4444" if abs(r["Desviacion"])>thresh_reb else "#059669"
            rows_t += (f"<tr><td><b>{r['Ticker']}</b></td>"
                       f"<td>{r['Cantidad']:.4f}</td>"
                       f"<td>${r['Precio']:,.2f}</td>"
                       f"<td><b>${r['Valor MXN']:,.0f}</b></td>"
                       f"<td>{r['Peso Actual']*100:.1f}%</td>"
                       f"<td>{r['Peso Obj']*100:.1f}%</td>"
                       f"<td style='color:{dc}'>{r['Desviacion']*100:+.1f}%</td>"
                       f"<td style='color:{gc2}'>{gs}${r['Ganancia USD']:,.0f}</td>"
                       f"<td>{badge(r['Alerta'])}</td></tr>")
        st.markdown(
            f"<div class='tbl'><table><thead><tr>"
            f"<th>Ticker</th><th>Cantidad</th><th>Precio</th>"
            f"<th>Valor MXN</th><th>Peso%</th><th>Obj%</th>"
            f"<th>Desv.</th><th>Ganancia</th><th>Alerta</th>"
            f"</tr></thead><tbody>{rows_t}</tbody></table></div>",
            unsafe_allow_html=True)

    with pc2:
        pal2 = ["#1a3a5c","#c9a227","#059669","#ef4444",
                "#5b4fcf","#0e7490","#b45309","#6d28d9"]
        all_t = list(dict.fromkeys(list(pos_df["Ticker"]) + list(target_wts.keys())))
        tc3   = {t: pal2[i%len(pal2)] for i,t in enumerate(all_t)}
        has_actual = pos_df["Peso Actual"].sum() > 0
        if has_actual:
            fig_d = make_subplots(1,2, specs=[[{"type":"pie"},{"type":"pie"}]],
                                  subplot_titles=["Actual","Objetivo"])
            fig_d.add_trace(go.Pie(
                labels=pos_df["Ticker"], values=pos_df["Peso Actual"],
                hole=0.52, textinfo="label+percent", textfont_size=10,
                marker=dict(colors=[tc3.get(t,"#888") for t in pos_df["Ticker"]],
                            line=dict(color="white",width=2))), 1,1)
            fig_d.add_trace(go.Pie(
                labels=list(target_wts.keys()), values=list(target_wts.values()),
                hole=0.52, textinfo="label+percent", textfont_size=10,
                marker=dict(colors=[tc3.get(t,"#888") for t in target_wts],
                            line=dict(color="white",width=2)),
                showlegend=False), 1,2)
            fig_d.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10,r=10,t=40,b=60),
                legend=dict(orientation="h",y=-0.25,x=0.5,xanchor="center",font_size=10))
        else:
            fig_d = go.Figure(go.Pie(
                labels=list(target_wts.keys()), values=list(target_wts.values()),
                hole=0.52, textinfo="label+percent", textfont_size=11,
                marker=dict(colors=[tc3.get(t,"#888") for t in target_wts],
                            line=dict(color="white",width=2)),
                title=dict(text="Objetivo<br>Modelo", font_size=13)))
            fig_d.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10,r=10,t=30,b=60),
                legend=dict(orientation="h",y=-0.25,x=0.5,xanchor="center",font_size=10))
            st.caption("Sin operaciones — composición objetivo del modelo")
        st.plotly_chart(fig_d, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: OPERACIONES ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Operaciones" and selected:
    st.markdown(f"<div class='sec'>📋 Operaciones · {selected}</div>",
               unsafe_allow_html=True)
    from datetime import date as date_type
    with st.form(f"form_{selected}", clear_on_submit=True):
        a,b,c2 = st.columns(3)
        with a:
            ff = st.date_input("Fecha", value=date_type.today())
            ft = st.selectbox("Tipo", ["Compra","Venta"])
        with b:
            fk = st.text_input("Ticker").strip().upper()
            fq = st.number_input("Cantidad", min_value=0.0001, value=1.0, step=0.001, format="%.4f")
        with c2:
            fp  = st.number_input("Precio USD", min_value=0.01, value=100.0, step=0.01)
            fc  = st.number_input("Comisión USD", min_value=0.0, value=0.5, step=0.01)
            ftc = st.number_input("TC MXN/USD", value=tc, step=0.01, format="%.2f")
        if st.form_submit_button("💾 Guardar", type="primary", use_container_width=True):
            if fk and sheet_id:
                gc2 = get_gc()
                if gc2:
                    try:
                        gc2.open_by_key(sheet_id).worksheet(f"Ops_{selected}") \
                           .append_row([str(ff), fk, ft, fq, fp, fc, ftc, cur])
                        st.success("✅ Guardado")
                        st.cache_data.clear()
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.error("Ticker requerido / Sheet no configurado")

    if not ops_df.empty:
        st.markdown("**Historial**", unsafe_allow_html=False)
        st.dataframe(ops_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: RIESGO ──────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚡ Riesgo" and selected:
    st.markdown(f"<div class='sec'>⚡ Análisis de Riesgo · {selected}</div>",
               unsafe_allow_html=True)
    if not result or result.get("error"):
        st.warning("Modelo no disponible.")
    else:
        t1, t2, t3 = st.tabs(["🎲 Monte Carlo","📊 VaR por Activo","📈 Histórico"])
        with t1:
            sims = result["sim_returns"] * 100
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=sims, nbinsx=60,
                marker_color="#1a3a5c", opacity=0.75,
                hovertemplate="Ret: %{x:.1f}%<extra></extra>"))
            for pv,pn,pc2 in [(result["ret_p5"]*100,"P5","#ef4444"),
                               (result["ret_p50"]*100,"Mediana","#c9a227"),
                               (result["ret_p95"]*100,"P95","#059669")]:
                fig.add_vline(x=pv, line=dict(color=pc2,dash="dash",width=2),
                             annotation_text=f"{pn}: {pv:.1f}%")
            fig.update_layout(title="Distribución rendimientos anuales (10 000 sims)",
                xaxis_title="Rendimiento (%)", height=340,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(243,244,246,.8)",
                margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)
            cc = st.columns(5)
            for i,(n2,v2,c2) in enumerate([
                ("P5",    result["ret_p5"] *100,"#ef4444"),
                ("P25",   result["ret_p25"]*100,"#d97706"),
                ("Med.",  result["ret_p50"]*100,"#c9a227"),
                ("P75",   result["ret_p75"]*100,"#2d7a4a"),
                ("P95",   result["ret_p95"]*100,"#059669")]):
                with cc[i]:
                    st.markdown(
                        f"<div class='card' style='text-align:center;border-top:3px solid {c2};'>"
                        f"<div style='font-size:9px;color:#6b7280;'>{n2}</div>"
                        f"<div style='font-size:18px;font-weight:700;color:{c2};'>{v2:.1f}%</div>"
                        f"</div>", unsafe_allow_html=True)
        with t2:
            als2 = result["AssetList"]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=als2, y=result["VaR"]*100,
                name="VaR 95%", marker_color="#ef4444", opacity=0.8))
            fig2.add_trace(go.Bar(x=als2, y=result["CVaR"]*100,
                name="CVaR 95%", marker_color="#7f1d1d", opacity=0.8))
            fig2.update_layout(title="VaR y CVaR por activo (sim 500 · 24 meses)",
                barmode="group", height=320,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(243,244,246,.8)",
                margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig2, use_container_width=True)
        with t3:
            st.metric("Caída Máxima Histórica", f"{result['max_dd']*100:.2f}%")
            st.metric("VaR 95% Histórico (mensual)", f"{result['var95h']*100:.2f}%")
            st.metric("CVaR 95% Histórico (mensual)", f"{result['cvar95h']*100:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: ADMIN ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "👥 Admin" and is_a:
    st.markdown("<div class='sec'>👥 Administración</div>", unsafe_allow_html=True)

    with st.expander("➕ Crear Portafolio", expanded=not bool(st.session_state.portfolios)):
        st.info("Solo ingresa nombre y color. El modelo leerá **Precios_{nombre}** y "
                "**Ops_{nombre}** automáticamente del Sheet.")
        a1, a2 = st.columns([3,1])
        with a1:
            np_n = st.text_input("Nombre (sin espacios)", key="npn",
                                  placeholder="AltoRendimiento")
            np_d = st.text_input("Descripción", key="npd",
                                  placeholder="Perfil agresivo")
        with a2:
            np_c = st.color_picker("Color", "#ef4444", key="npc")
        if st.button("➕ Crear", type="primary"):
            n = np_n.strip().replace(" ","")
            if not n:
                st.error("Nombre requerido")
            elif n in st.session_state.portfolios:
                st.error(f"Ya existe '{n}'")
            else:
                st.session_state.portfolios[n] = {"color":np_c,"desc":np_d.strip()}
                st.success(f"✅ '{n}' creado. Crea la hoja Precios_{n} en tu Sheet.")
                st.cache_data.clear()
                st.rerun()

    # Lista de portafolios
    if st.session_state.portfolios:
        st.markdown("<div class='sec'>Portafolios</div>", unsafe_allow_html=True)
        for pn, pcfg2 in list(st.session_state.portfolios.items()):
            c1, c2 = st.columns([5,1])
            with c1:
                # Correr modelo para mostrar resultados
                res2, _, _, _ = cached_model(sheet_id, pn) if sheet_id else (None,[],[],pd.DataFrame())
                if res2 and not res2.get("error"):
                    info = (f"Sharpe: <b>{res2['sharpe_t']:.3f}</b> · "
                            f"Ret: <b>{res2['ret_t']*100:.2f}%</b> · "
                            f"Vol: <b>{res2['risk_t']*100:.2f}%</b> · "
                            f"RF: <b>{res2['rf']*100:.2f}%</b> · "
                            f"{res2['num_obs']} retornos")
                else:
                    info = res2.get("error","Sin datos") if res2 else "Sin datos"
                st.markdown(
                    f"<div class='card' style='border-left:4px solid {pcfg2['color']};'>"
                    f"<div style='font-size:14px;font-weight:700;'>{pn}</div>"
                    f"<div style='font-size:12px;color:#6b7280;'>{info}</div>"
                    f"<div style='font-size:10px;color:#9ca3af;margin-top:4px;'>"
                    f"Hoja: Precios_{pn} + Ops_{pn}</div></div>",
                    unsafe_allow_html=True)
            with c2:
                if st.button("🗑", key=f"del_{pn}"):
                    del st.session_state.portfolios[pn]
                    st.rerun()
