"""
EGADA — Estrategia Global de Activos Dinámicos Avanzados
Portfolio Manager v3.0 — Arquitectura limpia, un portafolio de prueba,
diagnóstico completo de Excel/Sheet.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, date
import io, warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════
# LAYER 1 — CORE MODEL  (réplica exacta del MATLAB madre)
# Entrada : DataFrame de precios mensuales, columnas = tickers (+ SPX opcional)
# Salida  : dict con todos los outputs: tangencia, frontera, SML, MC, VaR
# ══════════════════════════════════════════════════════════════════════════
def _get_rf() -> float:
    """
    Obtiene la tasa libre de riesgo del bono del Tesoro USA a 10 años (^TNX).
    Exactamente como el archivo Python madre.
    Fallback: 0.0428 si no hay conexión.
    """
    try:
        if YF_AVAILABLE:
            import yfinance as yf
            tnx = yf.Ticker("^TNX")
            rf_data = tnx.history(period="1d")["Close"]
            if not rf_data.empty:
                return float(rf_data.iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.0428   # valor de respaldo igual que el archivo Python madre


def core_model_run(prices_df: pd.DataFrame,
                   rf: float = None,
                   max_w: float = 0.30) -> dict:
    """
    Traducción EXACTA de PORT_MAT-DEEP_PYTHON.py / Project_R_30_max__1_.mlx
    =========================================================================
    Pasos idénticos al archivo Python proporcionado:
      1.  RF automático desde ^TNX (igual que el archivo Python madre)
      2.  Eliminar filas con NaN: filas_validas = all(~isnan(Precios), 2)
      3.  Rendimientos = diff(log(Precios))
      4.  AssetMean = mean(Rendimientos) * 12
      5.  AssetCovar = cov(Rendimientos) * 12
      6.  mean_mensual = AssetMean / 12
      7.  cov_mensual  = AssetCovar / 12
      8.  Tangencia: minimize(neg_sharpe) con bounds [0, max_w]
      9.  Frontera: 20 puntos, linspace(min_ret, max_ret, 20)
      10. SML/Betas usando SPX como benchmark
      11. MC: nSim=500, T=24 meses → AccumulatedReturns, VaR, CVaR
      12. MC intervalos: nSim_mc=10000, T=12 meses → percentiles
      13. Horizontes: [1, 3, 5, 10, 20] años, nSim_mc=10000
      14. Filtro composición: w > 0.01 (igual que w_show = w_tang(w_tang>0.01))
    """
    from scipy.optimize import minimize

    if prices_df is None or prices_df.empty or len(prices_df) < 13:
        return {"error": "insuficientes_datos",
                "n_rows": len(prices_df) if prices_df is not None else 0}
    try:
        # ── RF automático desde mercado ────────────────────────────────────
        if rf is None:
            rf = _get_rf()

        df = prices_df.copy()
        if "__fecha__" in df.columns:
            df = df.drop(columns=["__fecha__"])

        # ── Extraer AssetList y Precios ────────────────────────────────────
        AssetList  = list(df.columns)
        nAssets    = len(AssetList)
        if nAssets < 2:
            return {"error": "menos_de_2_activos"}

        Precios = df.values.astype(float)

        # ── Identificar SPX ────────────────────────────────────────────────
        idx_spx = None
        for i, nombre in enumerate(AssetList):
            if "SPX" in nombre.upper():
                idx_spx = i
                break

        # ── Eliminar filas con NaN (filas_validas = all(~isnan(P), 2)) ─────
        if np.any(np.isnan(Precios)):
            filas_validas = ~np.any(np.isnan(Precios), axis=1)
            Precios = Precios[filas_validas, :]

        # ── Rendimientos logarítmicos ──────────────────────────────────────
        Rendimientos = np.diff(np.log(Precios), axis=0)
        num_obs      = Rendimientos.shape[0]

        # ── Estadísticos anualizados ───────────────────────────────────────
        AssetMean  = np.mean(Rendimientos, axis=0) * 12
        AssetCovar = np.cov(Rendimientos, rowvar=False) * 12

        # Variables mensuales (usadas en MC, igual que el archivo Python)
        mean_mensual = AssetMean / 12
        cov_mensual  = AssetCovar / 12

        # ── Funciones de portafolio ────────────────────────────────────────
        def port_stats(w, mu, cov):
            ret  = float(np.dot(w, mu))
            risk = float(np.sqrt(w @ cov @ w))
            return ret, risk

        def neg_sharpe(w, mu, cov, rf_):
            ret, risk = port_stats(w, mu, cov)
            return -(ret - rf_) / risk if risk > 1e-10 else 0.0

        def portfolio_return(w, mu, cov):
            return float(np.dot(w, mu))

        def portfolio_volatility(w, mu, cov):
            return float(np.sqrt(w @ cov @ w))

        # ── Restricciones y límites ────────────────────────────────────────
        init_guess  = np.ones(nAssets) / nAssets
        bounds      = tuple((0.0, max_w) for _ in range(nAssets))
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

        # ── Portafolio Tangencia (maximize Sharpe) ─────────────────────────
        opt_result = minimize(neg_sharpe, init_guess,
                              args=(AssetMean, AssetCovar, rf),
                              method="SLSQP", bounds=bounds,
                              constraints=constraints,
                              options={"ftol": 1e-12, "maxiter": 5000})
        w_tang        = opt_result.x
        ret_t, risk_t = port_stats(w_tang, AssetMean, AssetCovar)
        sharpe_t      = (ret_t - rf) / risk_t if risk_t > 1e-10 else 0.0

        # ── Frontera Eficiente (20 puntos) ─────────────────────────────────
        numPorts       = 20
        target_returns = np.linspace(AssetMean.min(), AssetMean.max(), numPorts)
        frontier_risk_arr    = []
        frontier_return_arr  = []
        frontier_weights_arr = []

        for ret_target in target_returns:
            cons_ret = [
                {"type": "eq", "fun": lambda x: np.sum(x) - 1},
                {"type": "eq", "fun": lambda x, r=ret_target:
                    portfolio_return(x, AssetMean, AssetCovar) - r},
            ]
            opt = minimize(portfolio_volatility, init_guess,
                           args=(AssetMean, AssetCovar),
                           method="SLSQP", bounds=bounds,
                           constraints=cons_ret,
                           options={"ftol": 1e-10, "maxiter": 500})
            if opt.success:
                r_p, v_p = port_stats(opt.x, AssetMean, AssetCovar)
                frontier_risk_arr.append(v_p)
                frontier_return_arr.append(r_p)
                frontier_weights_arr.append(opt.x)

        frontier_risk_arr    = np.array(frontier_risk_arr)
        frontier_return_arr  = np.array(frontier_return_arr)
        frontier_weights_arr = (np.array(frontier_weights_arr)
                                if len(frontier_weights_arr) else np.zeros((0, nAssets)))
        n_frontier = len(frontier_risk_arr)

        # Estructura de frontera para el dashboard
        frontier = [
            {"vol": float(frontier_risk_arr[i]),
             "ret": float(frontier_return_arr[i]),
             "weights": {AssetList[j]: float(frontier_weights_arr[i, j])
                         for j in range(nAssets)}}
            for i in range(n_frontier)
        ]

        # Mín varianza y máx retorno de la frontera
        if n_frontier > 0:
            idx_mv  = int(np.argmin(frontier_risk_arr))
            idx_mr  = int(np.argmax(frontier_return_arr))
            ret_v   = float(frontier_return_arr[idx_mv])
            vol_v   = float(frontier_risk_arr[idx_mv])
            w_mv    = frontier_weights_arr[idx_mv]
        else:
            ret_v = vol_v = 0.0
            w_mv  = init_guess

        # ── SML / Betas usando SPX ─────────────────────────────────────────
        betas, alphas = {}, {}
        if idx_spx is not None:
            Rm       = Rendimientos[:, idx_spx]
            Rm_ann   = float(np.mean(Rm) * 12)
            MRP      = Rm_ann - rf
            var_m    = float(np.var(Rm, ddof=1))
            for i, t in enumerate(AssetList):
                cov_im   = float(np.cov(Rendimientos[:, i], Rm)[0, 1])
                beta_i   = cov_im / var_m if var_m > 0 else 1.0
                betas[t] = beta_i
                alphas[t]= AssetMean[i] - (rf + beta_i * MRP)
        else:
            for t in AssetList:
                betas[t] = 1.0; alphas[t] = 0.0

        # ── Correlación ────────────────────────────────────────────────────
        corr_matrix = np.corrcoef(Rendimientos, rowvar=False)

        # ── MC: Simulaciones (nSim=500, T=24 meses) ────────────────────────
        # Exactamente como G9-G13 del archivo Python
        T_sim  = 24
        nSim   = 500
        np.random.seed(0)
        RetSeries = np.random.multivariate_normal(
            mean_mensual, cov_mensual, size=(nSim, T_sim))  # (nSim, T, nAssets)

        # Rendimientos acumulados por activo (nAssets, nSim)
        AccumulatedReturns = np.prod(1 + RetSeries, axis=1) - 1  # (nSim, nAssets)
        if AccumulatedReturns.shape[0] != nAssets:
            AccumulatedReturns = AccumulatedReturns.T

        # VaR / CVaR por activo (G13)
        alpha_var = 0.05
        var_asset  = {}
        cvar_asset = {}
        for i, t in enumerate(AssetList):
            sorted_r = np.sort(AccumulatedReturns[i])
            idx_v    = max(1, int(alpha_var * nSim))
            var_asset[t]  = float(-sorted_r[idx_v])
            cvar_asset[t] = float(-np.mean(sorted_r[:idx_v]))

        # VaR / CVaR por portafolio en frontera (G14)
        nSim3  = 500
        T_p    = 12
        var_frontier  = []
        cvar_frontier = []
        for j in range(n_frontier):
            pw   = frontier_weights_arr[j]
            sims = np.zeros(nSim3)
            for k in range(nSim3):
                sr      = np.random.multivariate_normal(mean_mensual, cov_mensual, size=T_p)
                sims[k] = float(np.prod(1 + sr @ pw) - 1)
            sorted_l  = np.sort(-sims)
            idx_vf    = max(1, int(alpha_var * nSim3))
            var_frontier.append(float(sorted_l[idx_vf]))
            cvar_frontier.append(float(np.mean(sorted_l[:idx_vf])))

        # ── MC Intervalos de confianza (nSim_mc=10000) ─────────────────────
        # Igual que el archivo Python: sim_returns para 1 año con w_tang
        nSim_mc = 10000
        np.random.seed(0)
        sim_returns = np.zeros(nSim_mc)
        for i in range(nSim_mc):
            sim_ret      = np.random.multivariate_normal(mean_mensual, cov_mensual, size=12)
            sim_returns[i] = float(np.prod(1 + sim_ret @ w_tang) - 1)

        ret_p5  = float(np.percentile(sim_returns, 5))
        ret_p25 = float(np.percentile(sim_returns, 25))
        ret_p50 = float(np.percentile(sim_returns, 50))
        ret_p75 = float(np.percentile(sim_returns, 75))
        ret_p95 = float(np.percentile(sim_returns, 95))
        prob_loss = float(np.mean(sim_returns < 0))

        # ── Horizontes [1,3,5,10,20] años (igual que el archivo Python) ────
        horizontes = [1, 3, 5, 10, 20]
        hz = {}
        for años in horizontes:
            meses  = años * 12
            sim_h  = np.zeros(nSim_mc)
            for i in range(nSim_mc):
                sim_ret  = np.random.multivariate_normal(
                    mean_mensual, cov_mensual, size=meses)
                sim_h[i] = (np.prod(1 + sim_ret @ w_tang) - 1) * 100
            hz[años] = {
                "p5":   float(np.percentile(sim_h, 5)),
                "mean": float(np.mean(sim_h)),
                "p95":  float(np.percentile(sim_h, 95)),
            }

        # ── Riesgo histórico del portafolio tangencia ──────────────────────
        port_r = Rendimientos @ w_tang
        cum    = np.exp(np.cumsum(port_r))
        mx_cum = np.maximum.accumulate(cum)
        max_dd = float(np.min((cum - mx_cum) / mx_cum))
        var95  = float(np.percentile(port_r, 5))
        mask   = port_r < var95
        cvar95 = float(port_r[mask].mean()) if mask.any() else var95

        # ── Resultado completo ─────────────────────────────────────────────
        return {
            # Identificación
            "tickers":      AssetList,
            "spx_col":      AssetList[idx_spx] if idx_spx is not None else None,
            "n_obs":        num_obs,
            "n_assets":     nAssets,
            "rf":           rf,
            "max_w":        max_w,
            # Tangencia (w > 0.01 igual que w_show = w_tang(w_tang > 0.01) en MATLAB)
            "w_tang":       {AssetList[i]: float(w_tang[i]) for i in range(nAssets)},
            "ret_tang":     ret_t,
            "vol_tang":     risk_t,
            "sharpe_tang":  sharpe_t,
            # Mín Varianza
            "w_mv":         {AssetList[i]: float(w_mv[i]) for i in range(nAssets)},
            "ret_mv":       ret_v,
            "vol_mv":       vol_v,
            # Frontera eficiente
            "frontier":     frontier,
            # Matrices estadísticas
            "mu_ann":       {AssetList[i]: float(AssetMean[i])  for i in range(nAssets)},
            "sd_ann":       {AssetList[i]: float(np.sqrt(AssetCovar[i,i])) for i in range(nAssets)},
            "cov_ann":      AssetCovar,
            "corr":         corr_matrix,
            # SML
            "betas":        betas,
            "alphas":       alphas,
            # Riesgo histórico
            "max_dd":       max_dd,
            "var_95":       var95,
            "cvar_95":      cvar95,
            # MC portafolio (10 000 sims × 1 año)
            "mc_sims":      sim_returns.tolist(),
            "mc_p5":        ret_p5,
            "mc_p25":       ret_p25,
            "mc_p50":       ret_p50,
            "mc_p75":       ret_p75,
            "mc_p95":       ret_p95,
            "mc_ploss":     prob_loss,
            # VaR por activo
            "var_asset":    var_asset,
            "cvar_asset":   cvar_asset,
            # VaR por portafolio en frontera
            "var_frontier": var_frontier,
            "cvar_frontier":cvar_frontier,
            # Horizontes
            "horizons":     hz,
            "error":        None,
        }
    except Exception as ex:
        return {"error": str(ex)}



# ══════════════════════════════════════════════════════════════════════════
# LAYER 2 — GOOGLE SHEETS
# ══════════════════════════════════════════════════════════════════════════
SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive"]

# Mapa Bloomberg → ticker corto (acepta ambos formatos en el Excel)
B2T = {
    "AAPL US Equity":"AAPL","ASTS US Equity":"ASTS","JPM US Equity":"JPM",
    "9988 HK Equity":"9988HK","CAT US Equity":"CAT","NOW US Equity":"NOW",
    "NVDA US Equity":"NVDA","NEM US Equity":"NEM","TSLA US Equity":"TSLA",
    "BATS LN Equity":"BATS","CIEN US Equity":"CIEN","EXPE US Equity":"EXPE",
    "EXPE US":"EXPE","SPX Index":"SPX","SPX":"SPX",
    "GLD US Equity":"GLD","IAU US Equity":"IAU","GE US Equity":"GE",
    "SHY US Equity":"SHY","IEI US Equity":"IEI","VCSH US Equity":"VCSH",
    "QQQ US Equity":"QQQ","WMT US Equity":"WMT","TIP US Equity":"TIP",
    "LQD US Equity":"LQD","HYG US Equity":"HYG","AGG US Equity":"AGG",
    # Tickers cortos directos
    "SHY":"SHY","IEI":"IEI","VCSH":"VCSH","QQQ":"QQQ","WMT":"WMT",
    "GLD":"GLD","IAU":"IAU","NEM":"NEM","AAPL":"AAPL","CAT":"CAT",
    "NVDA":"NVDA","TIP":"TIP","LQD":"LQD","HYG":"HYG","AGG":"AGG",
}

YF_MAP = {
    "SHY":"SHY","IEI":"IEI","VCSH":"VCSH","TIP":"TIP","LQD":"LQD",
    "HYG":"HYG","AGG":"AGG","QQQ":"QQQ","GLD":"GLD","IAU":"IAU",
    "WMT":"WMT","AAPL":"AAPL","CAT":"CAT","NVDA":"NVDA","NEM":"NEM",
    "SPX":"^GSPC","TSLA":"TSLA","JPM":"JPM","BATS":"BATS.L",
    "ASTS":"ASTS","NOW":"NOW","9988HK":"9988.HK",
}

@st.cache_resource
def get_gc():
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=SCOPES)
        return gspread.authorize(creds)
    except Exception:
        return None

@st.cache_data(ttl=60)
def load_tab(sheet_id: str, tab: str) -> pd.DataFrame:
    gc = get_gc()
    if gc is None: return pd.DataFrame()
    try:
        data = gc.open_by_key(sheet_id).worksheet(tab).get_all_records()
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def save_tab_row(sheet_id: str, tab: str, row: list) -> bool:
    gc = get_gc()
    if gc is None: return False
    try:
        gc.open_by_key(sheet_id).worksheet(tab).append_row(row)
        return True
    except Exception:
        return False

def _parse_price(raw: str) -> float:
    """
    Convierte cualquier formato numérico de Google Sheets a float.
    Maneja: 1234.56 | 1,234.56 | 1.234,56 | 1 234,56 | 1234,56
    """
    s = str(raw).strip().replace(" ", "")
    if not s:
        return 0.0
    # Detectar formato europeo: punto como miles, coma como decimal
    # Ej: "1.234,56" → tiene coma después del último punto → europeo
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            # Europeo: 1.234,56 → quitar puntos, cambiar coma por punto
            s = s.replace(".", "").replace(",", ".")
        else:
            # Americano: 1,234.56 → quitar comas
            s = s.replace(",", "")
    elif "," in s:
        # Solo coma: puede ser decimal europeo (99,14) o miles (1,234)
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) <= 2:
            # Decimal: 99,14 → 99.14
            s = s.replace(",", ".")
        else:
            # Miles sin decimal: 1,234 → 1234
            s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


@st.cache_data(ttl=3600)
def load_prices(sheet_id: str, port_name: str) -> pd.DataFrame:
    """
    Lee Precios_<port_name> del Sheet.
    - Parsing robusto de números (todos los formatos de Google Sheets)
    - Solo incluye filas donde TODOS los activos tienen precio > 0
      (replica MATLAB: filas_validas = all(~isnan(Precios), 2))
    - Guarda __fecha__ para diagnóstico (no entra al modelo)
    """
    gc = get_gc()
    if gc is None: return pd.DataFrame()
    try:
        ws   = gc.open_by_key(sheet_id).worksheet(f"Precios_{port_name}")
        data = ws.get_all_values()
        if len(data) < 3: return pd.DataFrame()
        headers = data[0]

        # ── Identificar columnas ──────────────────────────────────────────
        fecha_idx  = None
        asset_hdrs = []   # lista de (col_idx, ticker_name)
        for j, col in enumerate(headers):
            col_clean = col.strip()
            if col_clean.lower() in ("fecha", "date"):
                fecha_idx = j
            elif col_clean == "":
                pass   # columna vacía, ignorar
            else:
                tkr = B2T.get(col_clean) or col_clean.upper()
                asset_hdrs.append((j, tkr))

        if len(asset_hdrs) < 2:
            return pd.DataFrame()

        n_assets = len(asset_hdrs)
        rows = []

        # ── Leer filas ────────────────────────────────────────────────────
        for row in data[1:]:
            # Saltar filas completamente vacías
            if all(str(v).strip() == "" for v in row):
                continue

            vals = {}

            # Fecha (solo para diagnóstico)
            if fecha_idx is not None and fecha_idx < len(row):
                f = str(row[fecha_idx]).strip()
                if f:
                    vals["__fecha__"] = f

            # Precios — todos deben ser > 0 (igual que MATLAB filas_validas)
            all_valid = True
            for j, tkr in asset_hdrs:
                raw = row[j] if j < len(row) else ""
                px  = _parse_price(str(raw))
                if px > 0:
                    vals[tkr] = px
                else:
                    all_valid = False
                    break   # fila inválida, descartar

            if all_valid:
                rows.append(vals)

        if len(rows) < 13:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_quotes(tickers: list) -> dict:
    if not YF_AVAILABLE: return {}
    out = {}
    for t in tickers:
        sym = YF_MAP.get(t, t)
        try:
            info = yf.Ticker(sym).fast_info
            px   = getattr(info, "last_price", None)
            prev = getattr(info, "previous_close", None)
            h52  = getattr(info, "year_high", None)
            l52  = getattr(info, "year_low", None)
            chg  = (px/prev-1)*100 if px and prev else None
            ma52 = (h52+l52)/2     if h52 and l52 else None
            vsma = (px/ma52-1)*100 if px and ma52 else None
            out[t] = {"price":px,"prev":prev,"chg":chg,
                      "h52":h52,"l52":l52,"ma52":ma52,"vsma":vsma}
        except Exception:
            out[t] = {}
    return out

# ══════════════════════════════════════════════════════════════════════════
# LAYER 3 — PORTFOLIO REGISTRY
# Un portafolio = una hoja Precios_<nombre> + Ops_<nombre>
# Los portafolios de producción se crean desde Admin
# ══════════════════════════════════════════════════════════════════════════
def get_all_ports() -> dict:
    """Solo portafolios creados por admin (custom_portfolios en session_state)."""
    return st.session_state.get("custom_portfolios", {})

def get_user_ports(username: str) -> list:
    user    = USERS.get(username, {})
    allowed = user.get("portfolios")
    all_p   = list(get_all_ports().keys())
    return all_p if allowed is None else [p for p in allowed if p in all_p]

# ══════════════════════════════════════════════════════════════════════════
# LAYER 4 — POSITIONS & ALERTS
# ══════════════════════════════════════════════════════════════════════════
def calc_positions(ops: pd.DataFrame, prices: dict, tc: float,
                   target_tickers: list = None) -> pd.DataFrame:
    tks = list(ops["Ticker"].unique()) if not ops.empty else []
    for t in (target_tickers or []):
        if t not in tks: tks.append(t)
    if not tks: return pd.DataFrame()
    rows = []
    for t in tks:
        qty = cost = 0.0
        if not ops.empty and t in ops["Ticker"].values:
            sub = ops[ops["Ticker"] == t]
            qc  = sub[sub["Tipo"]=="Compra"]["Cantidad"].sum()
            qv  = sub[sub["Tipo"]=="Venta"]["Cantidad"].sum()
            qty = qc - qv
            cost = (sub[sub["Tipo"]=="Compra"]["Cantidad"] *
                    sub[sub["Tipo"]=="Compra"]["Precio_USD"]).sum() / qc if qc > 0 else 0.0
        px    = prices.get(t, 0.0)
        vusd  = qty * px
        vmxn  = vusd * tc
        gain  = (px - cost) * qty
        gpct  = (px/cost - 1) if cost > 0 else 0.0
        rows.append({"Ticker":t,"Cantidad":qty,"Precio Actual":px,
                     "Costo Prom":cost,"Valor USD":vusd,"Valor MXN":vmxn,
                     "Ganancia USD":gain,"Ganancia %":gpct})
    return pd.DataFrame(rows)

def calc_alerts(pos: pd.DataFrame, target_wts: dict, tc: float,
                contrib: float, thr: float = 0.05) -> pd.DataFrame:
    if pos.empty: return pd.DataFrame()
    total = pos["Valor MXN"].sum()
    df    = pos.copy()
    df["Peso Objetivo"] = df["Ticker"].map(target_wts).fillna(0.0)
    if total == 0:
        df["Peso Actual"] = 0.0; df["Desviación"] = -df["Peso Objetivo"]
        df["Alerta"] = df["Peso Objetivo"].apply(lambda w: "COMPRAR" if w>0 else "OK")
        df["Compra Sugerida MXN"] = df["Peso Objetivo"] * contrib
        df["Acciones a Comprar"]  = df.apply(
            lambda r: r["Compra Sugerida MXN"]/(r["Precio Actual"]*tc)
            if r["Precio Actual"]>0 else 0, axis=1)
        df["Monto a Vender MXN"] = 0.0; df["Acciones a Vender"] = 0.0
        return df
    df["Peso Actual"]  = df["Valor MXN"] / total
    df["Desviación"]   = df["Peso Actual"] - df["Peso Objetivo"]
    def alert(r):
        if r["Peso Objetivo"]==0 and r["Peso Actual"]>0: return "VENDER"
        if r["Desviación"] >  thr: return "VENDER"
        if r["Desviación"] < -thr: return "COMPRAR"
        return "OK"
    df["Alerta"] = df.apply(alert, axis=1)
    df["MXN Dif"]  = df["Peso Objetivo"]*total - df["Valor MXN"]
    df["Compra Sugerida MXN"] = df["MXN Dif"].clip(lower=0)
    df["Acciones a Comprar"]  = df.apply(
        lambda r: r["Compra Sugerida MXN"]/(r["Precio Actual"]*tc)
        if r["Precio Actual"]>0 else 0, axis=1)
    df["Monto a Vender MXN"] = df["MXN Dif"].clip(upper=0).abs()
    df["Acciones a Vender"]  = df.apply(
        lambda r: r["Monto a Vender MXN"]/(r["Precio Actual"]*tc)
        if r["Precio Actual"]>0 else 0, axis=1)
    return df

def calc_twr(ops: pd.DataFrame, live: dict, tc: float) -> dict:
    if ops.empty: return {}
    try:
        o = ops.copy()
        o["Fecha"] = pd.to_datetime(o["Fecha"], errors="coerce")
        o = o.dropna(subset=["Fecha"]).sort_values("Fecha")
        inv = (o[o["Tipo"]=="Compra"]["Cantidad"]*o[o["Tipo"]=="Compra"]["Precio_USD"]).sum()
        inv -= (o[o["Tipo"]=="Venta"]["Cantidad"]*o[o["Tipo"]=="Venta"]["Precio_USD"]).sum()
        val = sum((
            (o[o["Ticker"]==t]["Cantidad"]*(1 if o[o["Ticker"]==t]["Tipo"]=="Compra" else -1)).sum()
            * live.get(t,0)
            for t in o["Ticker"].unique()))
        twr = (val-inv)/inv*100 if inv>0 else 0.0
        dias = (datetime.now() - o["Fecha"].min()).days
        ann  = ((1+twr/100)**(365/dias)-1)*100 if dias>30 else None
        return {"first":o["Fecha"].min(),"inv":inv,"val":val,"twr":twr,"ann":ann,"dias":dias}
    except Exception:
        return {}

# ══════════════════════════════════════════════════════════════════════════
# LAYER 5 — USERS
# ══════════════════════════════════════════════════════════════════════════
def write_users(users_dict):
    import os, re
    p = os.path.abspath(__file__)
    with open(p,"r",encoding="utf-8") as f: src=f.read()
    nl="\n"; parts=["USERS = {  # USERS_START"+nl]
    for k,u in users_dict.items():
        pv=u.get("portfolios"); pr="None" if pv is None else repr(pv)
        parts+=["    "+repr(k)+": {"+nl,
                '        "name":       '+repr(u.get("name",""))+","+nl,
                '        "pin":        '+repr(u.get("pin","0000"))+","+nl,
                '        "avatar":     '+repr(u.get("avatar","??"))+","+nl,
                '        "color":      '+repr(u.get("color","#1a3a5c"))+","+nl,
                '        "sheet_id":   "",'+nl,
                '        "portfolios": '+pr+","+nl,
                '        "role":       '+repr(u.get("role","investor"))+","+nl,
                '        "email":      '+repr(u.get("email",""))+","+nl,
                "    },"+nl]
    parts.append("}  # USERS_END"+nl)
    nb="".join(parts)
    pat=r"USERS = \{  # USERS_START\n.*?\}  # USERS_END\n"
    if re.search(pat,src,re.DOTALL):
        with open(p,"w",encoding="utf-8") as f:
            f.write(re.sub(pat,nb,src,flags=re.DOTALL))
        return True,"Guardado"
    return False,"Marcadores no encontrados"

USERS = {  # USERS_START
    "david": {
        "name":       "David Jassan",
        "pin":        "1234",
        "avatar":     "DJ",
        "color":      "#1a3a5c",
        "sheet_id":   "1eApNRcJSnqYYkUxK2uDWqUOwXh6lNkoOZ-zvzVSoKFw",
        "portfolios": None,
        "role":       "admin",
        "email":      "",
    },
}  # USERS_END

# ══════════════════════════════════════════════════════════════════════════
# LAYER 6 — STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="EGADA v3", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--navy:#0d1b2a;--gold:#c9a227;--green:#059669;--red:#ef4444;
      --amber:#d97706;--bg:#f5f7fa;--card:#ffffff;--border:#e5e7eb;
      --txt:#111827;--sub:#6b7280;}
*{font-family:'Inter',sans-serif;color:var(--txt);}
.stApp,[data-testid="stAppViewContainer"]{background:var(--bg)!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b2a,#162233)!important;
    border-right:1px solid rgba(201,162,39,0.2)!important;}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stDeployButton"],[class*="viewerBadge"]{display:none!important;}
.main .block-container{padding:1.5rem 2rem 3rem;}

/* ── Cards ── */
.card{background:var(--card);border-radius:14px;padding:20px 22px;
      border:1px solid var(--border);box-shadow:0 1px 4px rgba(0,0,0,.06);}
.card:hover{box-shadow:0 4px 16px rgba(0,0,0,.10);transform:translateY(-2px);
            transition:all .2s ease;}
.kpi-lbl{font-size:10px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
         color:var(--sub);margin-bottom:6px;}
.kpi-val{font-family:'Space Grotesk',sans-serif;font-size:30px;font-weight:700;
         line-height:1;margin-bottom:4px;}
.kpi-sub{font-size:11px;color:var(--sub);}
.pos{color:var(--green);} .neg{color:var(--red);}

/* ── Semáforo ── */
.sem{background:var(--card);border-radius:14px;padding:16px;text-align:center;
     border:1px solid var(--border);}
.sem-ico{font-size:22px;margin-bottom:6px;}
.sem-lbl{font-size:9px;text-transform:uppercase;letter-spacing:.08em;color:var(--sub);margin-bottom:4px;}
.sem-val{font-family:'Space Grotesk',sans-serif;font-size:24px;font-weight:700;margin-bottom:4px;}
.sem-txt{font-size:10px;font-weight:600;line-height:1.4;}

/* ── Tables ── */
.tbl{width:100%;overflow-x:auto;border-radius:12px;
     border:1px solid var(--border);background:var(--card);}
table{width:100%;border-collapse:collapse;font-size:12px;}
th{background:#1a3a5c;color:white;padding:10px 12px;text-align:center;
   font-size:10px;text-transform:uppercase;letter-spacing:.07em;font-weight:600;}
td{padding:9px 12px;text-align:center;border-bottom:1px solid var(--border);}
tr:last-child td{border-bottom:none;}
tr:nth-child(even) td{background:#f9fafb;}
tr:hover td{background:#f0f4ff;}

/* ── Diagnóstico Excel ── */
.diag{border-radius:12px;padding:14px 18px;font-size:12px;margin-bottom:16px;
      border-left:4px solid;}
.diag-ok{background:#ecfdf5;border-color:var(--green);}
.diag-warn{background:#fffbeb;border-color:var(--amber);}
.diag-err{background:#fef2f2;border-color:var(--red);}

/* ── Debug bar (admin) ── */
.dbg{background:#1a3a5c;color:white;border-radius:10px;padding:8px 16px;
     font-size:11px;margin-bottom:14px;font-family:monospace;}

/* ── Section title ── */
.sec{font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:600;
     margin:24px 0 14px;padding-bottom:8px;
     border-bottom:2px solid rgba(201,162,39,0.35);}

/* ── Charts ── */
div[data-testid="stPlotlyChart"]{border-radius:14px!important;
    box-shadow:0 2px 10px rgba(0,0,0,.07)!important;}

/* ── Login ── */
.lglass{background:white;border-radius:20px;padding:32px 36px;
        box-shadow:0 4px 24px rgba(0,0,0,.10);border-top:4px solid var(--gold);}
[data-testid="stAppViewContainer"] [data-testid="stButton"]>div>button{
    background:#f9fafb!important;border:1.5px solid var(--border)!important;
    border-radius:12px!important;font-weight:600!important;min-height:64px!important;
    padding:12px 18px!important;text-align:left!important;white-space:pre-line!important;
    transition:all .18s!important;}
[data-testid="stAppViewContainer"] [data-testid="stButton"]>div>button:hover{
    border-color:var(--gold)!important;background:white!important;
    transform:translateX(4px)!important;}
</style>
""", unsafe_allow_html=True)

# ── Session init ──────────────────────────────────────────────────────────────
for k,v in [("auth",False),("user",None),("sel_login",None),
             ("pin_err",False),("custom_portfolios",{})]:
    if k not in st.session_state: st.session_state[k]=v

# ══════════════════════════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════════════════════════
if not st.session_state.auth:
    st.markdown("""<style>
    .stApp{background:radial-gradient(ellipse at 30% 40%,#112240,#050810)!important;}
    </style>""", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        import base64
        _svg_str = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 80">'
            '<rect width="6" height="80" fill="#c8102e" rx="2"/>'
            '<text x="16" y="52" font-family="Georgia" font-size="36" font-weight="700"'
            ' letter-spacing="3" fill="white">EGADA</text>'
            '<text x="17" y="68" font-family="Arial" font-size="8.5" letter-spacing="2"'
            ' fill="rgba(255,255,255,0.5)">ESTRATEGIA GLOBAL DE ACTIVOS</text>'
            '<text x="17" y="19" font-family="Arial" font-size="8" letter-spacing="1.5"'
            ' fill="#c8102e">DINAMICOS AVANZADOS</text>'
            '</svg>')
        svg = base64.b64encode(_svg_str.encode()).decode()
        st.markdown(
            f'<div style="text-align:center;padding:48px 0 24px;">'
            f'<div style="display:inline-block;background:rgba(255,255,255,.97);'
            f'border-radius:18px;padding:20px 36px;box-shadow:0 8px 32px rgba(0,0,0,.2);">'
            f'<img src="data:image/svg+xml;base64,{svg}" style="width:240px;display:block;"></div>'
            f'<div style="margin-top:12px;font-size:11px;color:rgba(201,162,39,.8);'
            f'letter-spacing:.18em;font-weight:700;text-transform:uppercase;">'
            f'Portfolio Manager v3.0</div></div>', unsafe_allow_html=True)

        if not st.session_state.sel_login:
            st.markdown('<div class="lglass"><div style="font-size:22px;font-weight:700;'
                       'margin-bottom:4px;">Bienvenido</div>'
                       '<div style="font-size:12px;color:#6b7280;margin-bottom:24px;">'
                       'Selecciona tu perfil</div></div>', unsafe_allow_html=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            for uk, ud in USERS.items():
                pts = get_user_ports(uk) or ["Sin portafolios"]
                lbl = f"{ud['avatar']}  {ud['name']}\n{' · '.join(pts[:4])}"
                if st.button(lbl, key=f"lb_{uk}", use_container_width=True):
                    st.session_state.sel_login = uk
                    st.session_state.pin_err = False
                    st.rerun()
        else:
            uk   = st.session_state.sel_login
            ud   = USERS[uk]
            err  = ('<div style="background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.3);'
                    'border-radius:8px;padding:8px 14px;color:#c0392b;font-size:12px;text-align:center;'
                    'margin-top:8px;">⚠ PIN incorrecto</div>') if st.session_state.pin_err else ""
            st.markdown(f'<div class="lglass">'
                       f'<div style="display:flex;align-items:center;gap:12px;'
                       f'margin-bottom:16px;padding-bottom:14px;border-bottom:1px solid #e5e7eb;">'
                       f'<div style="width:44px;height:44px;border-radius:50%;background:{ud["color"]};'
                       f'display:flex;align-items:center;justify-content:center;'
                       f'font-weight:700;color:white;font-size:15px;">{ud["avatar"]}</div>'
                       f'<div><div style="font-weight:700;font-size:16px;">{ud["name"]}</div>'
                       f'<div style="font-size:11px;color:#6b7280;">{ud["role"].upper()}</div>'
                       f'</div></div>'
                       f'<div style="font-size:10px;font-weight:700;color:#c9a227;'
                       f'letter-spacing:.14em;text-transform:uppercase;margin-bottom:8px;">'
                       f'PIN de acceso</div>{err}</div>', unsafe_allow_html=True)
            pin = st.text_input("PIN","",type="password",max_chars=4,
                                placeholder="● ● ● ●",label_visibility="collapsed",key="pin_f")
            c1,c2=st.columns([1,1.4])
            with c1:
                if st.button("← Volver",use_container_width=True,key="bk"):
                    st.session_state.sel_login=None; st.session_state.pin_err=False; st.rerun()
            with c2:
                ok=st.button("Entrar →",use_container_width=True,type="primary",key="ok")
            if ok or (pin and len(pin)==4):
                if pin==ud["pin"]:
                    st.session_state.auth=True; st.session_state.user=uk
                    st.session_state.pin_err=False; st.rerun()
                elif ok:
                    st.session_state.pin_err=True
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# AUTHENTICATED
# ══════════════════════════════════════════════════════════════════════════
cur  = st.session_state.user
ud   = USERS[cur]
is_a = ud.get("role")=="admin"

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    import base64 as _b64
    _svg_s2 = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 280 72">'
        '<rect width="6" height="72" fill="#c8102e" rx="2"/>'
        '<text x="14" y="47" font-family="Georgia" font-size="32" font-weight="700"'
        ' letter-spacing="3" fill="white">EGADA</text>'
        '<text x="15" y="62" font-family="Arial" font-size="8" letter-spacing="1.8"'
        ' fill="rgba(255,255,255,0.45)">ESTRATEGIA GLOBAL DE ACTIVOS</text>'
        '</svg>')
    _svg = _b64.b64encode(_svg_s2.encode()).decode()
    st.markdown(f"<div style='padding:16px 14px 12px;border-bottom:1px solid rgba(201,162,39,0.2);'>"
               f"<img src='data:image/svg+xml;base64,{_svg}' style='width:175px;'>"
               f"</div>", unsafe_allow_html=True)

    # TC live
    tc_live = 20.50
    if YF_AVAILABLE:
        try:
            v = getattr(yf.Ticker("MXN=X").fast_info,"last_price",None)
            if v and 15<float(v)<35: tc_live=round(float(v),2)
        except Exception: pass

    st.markdown(f"""<div style='padding:12px 0 10px;'>
        <div style='font-size:17px;font-weight:600;color:#e8f0f8;margin-bottom:8px;'>
            Portfolio <span style='color:#c9a227;'>Manager</span>
            <span style='font-size:10px;color:#6b7280;'> v3</span></div>
        <div style='background:rgba(255,255,255,.07);border-radius:10px;padding:10px 12px;
                    display:flex;align-items:center;gap:10px;'>
            <div style='width:30px;height:30px;border-radius:50%;background:{ud["color"]};
                        display:flex;align-items:center;justify-content:center;
                        font-weight:700;font-size:12px;color:white;'>{ud["avatar"]}</div>
            <div><div style='font-size:12px;font-weight:600;color:#e8f0f8;'>{ud["name"]}</div>
                 <div style='font-size:9px;color:#6b7280;'>{ud["role"].capitalize()}</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style='background:rgba(13,27,42,.9);border:1px solid rgba(201,162,39,.3);
                    border-radius:10px;padding:10px 14px;margin-bottom:12px;'>
        <div style='font-size:9px;color:#c9a227;font-weight:700;letter-spacing:.15em;
                    text-transform:uppercase;margin-bottom:3px;'>USD / MXN</div>
        <div style='font-size:24px;font-weight:800;color:white;line-height:1;'>{tc_live:.2f}</div>
        <div style='font-size:9px;color:rgba(201,162,39,.7);'>● Yahoo Finance</div>
    </div>""", unsafe_allow_html=True)

    if st.button("⬡ Cerrar sesión", use_container_width=True):
        st.session_state.auth=False; st.session_state.user=None
        st.session_state.sel_login=None; st.rerun()

    st.divider()

    # Sheet ID
    _own   = ud.get("sheet_id","").strip()
    _admin = next((u["sheet_id"] for u in USERS.values()
                   if u.get("role")=="admin" and u.get("sheet_id","").strip()),"")
    sheet_id  = _own if _own else _admin
    use_sheets = bool(sheet_id) and get_gc() is not None

    # Portafolios disponibles
    user_ports = get_user_ports(cur)
    st.markdown("<div style='font-size:10px;font-weight:700;letter-spacing:.1em;color:#c9a227;"
               "text-transform:uppercase;margin-bottom:8px;'>Portafolio Activo</div>",
               unsafe_allow_html=True)

    if not user_ports:
        st.markdown("<div style='color:rgba(255,255,255,0.4);font-size:12px;padding:8px;'>"
                   "Sin portafolios. El admin debe crear uno.</div>", unsafe_allow_html=True)
        selected = None
    else:
        if "sel_port" not in st.session_state or st.session_state.sel_port not in user_ports:
            st.session_state.sel_port = user_ports[0]

        # Detectar cambio de portafolio → limpiar caché para forzar recarga limpia
        if st.session_state.get('_last_port') != st.session_state.sel_port:
            st.session_state['_last_port'] = st.session_state.sel_port
            st.cache_data.clear()

        st.markdown("""<style>
        [data-testid="stSidebar"] [data-testid="stButton"] button{
            position:relative!important;margin-top:-42px!important;height:42px!important;
            background:transparent!important;border:none!important;box-shadow:none!important;
            color:transparent!important;font-size:0!important;cursor:pointer!important;
            z-index:999!important;width:100%!important;}
        </style>""", unsafe_allow_html=True)

        for pn in user_ports:
            pd_cfg = get_all_ports().get(pn,{})
            pc  = pd_cfg.get("color","#3b82f6")
            pds = pd_cfg.get("description","")
            sel = st.session_state.sel_port == pn
            bg  = pc if sel else "rgba(255,255,255,.07)"
            gl  = f"0 0 0 2px {pc},0 4px 14px {pc}44" if sel else f"0 0 0 1px {pc}66"
            st.markdown(f"<div style='background:{bg};border-radius:10px;padding:9px 12px;"
                       f"margin-bottom:2px;box-shadow:{gl};pointer-events:none;'>"
                       f"<div style='font-size:13px;font-weight:{'700' if sel else '400'};"
                       f"color:white;'>{pn}</div>"
                       f"<div style='font-size:10px;color:rgba(255,255,255,.5);'>{pds}</div>"
                       f"</div>", unsafe_allow_html=True)
            if st.button("​", key=f"pb_{pn}", use_container_width=True):
                st.session_state.sel_port=pn; st.rerun()

        selected = st.session_state.sel_port

    st.divider()

    # Admin params
    if is_a:
        st.markdown("<div style='font-size:10px;font-weight:700;letter-spacing:.1em;color:#c9a227;"
                   "text-transform:uppercase;margin-bottom:8px;'>Parámetros</div>",
                   unsafe_allow_html=True)
        tc_actual    = st.number_input("TC MXN/USD", value=tc_live, step=0.01, format="%.2f")
        monthly_add  = st.number_input("Aportación mensual MXN", value=2000.0, step=100.0)
        thresh_rebal = st.slider("Umbral rebalanceo",0.03,0.15,0.05,0.01,format="%.0f%%")
    else:
        tc_actual=tc_live; monthly_add=2000.0; thresh_rebal=0.05

    st.divider()
    st.markdown("<div style='font-size:10px;font-weight:700;letter-spacing:.1em;color:#c9a227;"
               "text-transform:uppercase;margin-bottom:8px;'>Navegación</div>",
               unsafe_allow_html=True)
    pages = (["📊 Dashboard","📋 Operaciones","⚡ Riesgo","🔀 Modelo vs Real","👥 Admin"]
             if is_a else
             ["📊 Mi Portafolio","📋 Registrar Operación","📈 Mi Inversión","🔀 Modelo vs Real"])
    page = st.radio("nav", pages, label_visibility="collapsed")

# ══════════════════════════════════════════════════════════════════════════
# SIN PORTAFOLIO SELECCIONADO
# ══════════════════════════════════════════════════════════════════════════
if not selected:
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;'>
        <div style='font-size:48px;margin-bottom:16px;'>📂</div>
        <div style='font-size:24px;font-weight:700;margin-bottom:8px;'>Sin portafolios configurados</div>
        <div style='font-size:14px;color:#6b7280;'>Ve a <b>👥 Admin</b> para crear tu primer portafolio.</div>
    </div>""", unsafe_allow_html=True)
    if page != "👥 Admin": st.stop()

# ── run_model: función a nivel de módulo — caché estable por portafolio ──────
@st.cache_data(ttl=300)
def run_model(sid: str, pname: str) -> dict:
    """
    Corre el modelo exacto de PORT_MAT-DEEP_PYTHON.py para el portafolio.
    RF se obtiene automáticamente desde ^TNX (o fallback 4.28%).
    Cacheado 5 min por (sheet_id, nombre_portafolio).
    """
    df = load_prices(sid, pname)
    if df.empty: return {}
    return core_model_run(df, max_w=0.30)


# ══════════════════════════════════════════════════════════════════════════
# CARGAR DATOS DEL PORTAFOLIO ACTIVO
# ══════════════════════════════════════════════════════════════════════════
if selected:
    port_cfg = get_all_ports().get(selected, {})
    port_color = port_cfg.get("color","#3b82f6")

    # ── Operaciones: Ops_<selected>
    ops_df = pd.DataFrame()
    if use_sheets and sheet_id:
        _raw = load_tab(sheet_id, f"Ops_{selected}")
        if not _raw.empty:
            if "Usuario" in _raw.columns:
                un = ud.get("name",cur).lower()
                uc = _raw["Usuario"].astype(str).str.strip().str.lower()
                mk = (uc==cur.lower())|(uc==un)|(uc==un.split()[0])
                ops_df = (_raw[mk].copy() if mk.any() else _raw.copy())
            else:
                ops_df = _raw.copy()
    if not ops_df.empty:
        for col in ["Cantidad","Precio_USD","Comision_USD","TC_MXN"]:
            if col in ops_df.columns:
                ops_df[col] = pd.to_numeric(ops_df[col],errors="coerce").fillna(0)

    # ── Precios históricos: Precios_<selected>
    prices_hist = pd.DataFrame()
    if use_sheets and sheet_id:
        prices_hist = load_prices(sheet_id, selected)

    # ── Correr modelo (run_model definido a nivel de módulo)
    model = {}
    if use_sheets and sheet_id and not prices_hist.empty:
        model = run_model(sheet_id, selected)

    # ── Pesos objetivo: del modelo o semilla del config
    seed_wts = port_cfg.get("target", {})
    if model and not model.get("error") and model.get("w_tang"):
        # Filtrar w > 0.01 (1%) igual que MATLAB: w_show = w_tang(w_tang > 0.01)
        # Esto replica exactamente el pie chart de MATLAB que muestra solo 4 activos
        target_wts = {t:w for t,w in model["w_tang"].items() if w > 0.01}
        wt_source  = f"Modelo Markowitz · {model['n_obs']} meses"
    else:
        target_wts = seed_wts
        wt_source  = "Pesos semilla (conecta Sheet para el modelo)"

    # ═══════════════════════════════════════════════════════════════════
    # PIPELINE B — PRECIOS: separado del modelo
    # El modelo (Pipeline A) ya corrió SOLO con el Excel histórico.
    # Yahoo Finance se usa ÚNICAMENTE para valorar posiciones reales.
    # ═══════════════════════════════════════════════════════════════════

    # Tickers con operaciones reales (los únicos que necesitan precio live)
    _op_tickers = ops_df["Ticker"].unique().tolist() if not ops_df.empty else []

    # Último precio del Excel histórico (para el modelo y cotizaciones del portafolio)
    _last_excel_price = {}
    if not prices_hist.empty:
        _last_row = prices_hist.iloc[-1].to_dict()
        _last_excel_price = {}
        for t, v in _last_row.items():
            if t.startswith("__"): continue
            try:
                fv = float(v)
                if fv > 0:
                    _last_excel_price[t] = fv
            except (TypeError, ValueError):
                pass

    # Precios live de Yahoo solo para tickers con operaciones reales
    yf_prices = {}
    if _op_tickers:
        _raw_q = get_quotes(_op_tickers)
        yf_prices = {t: d.get("price") or 0
                     for t,d in _raw_q.items() if d.get("price")}

    # Precio registrado en operaciones (costo histórico del usuario)
    ops_prices = {}
    if not ops_df.empty and "Precio_USD" in ops_df.columns:
        ops_prices = (ops_df.sort_values("Fecha")
                      .groupby("Ticker")["Precio_USD"].last().to_dict())

    # Precio final para valorar posiciones:
    #   · Tickers con ops reales → Yahoo Finance live
    #   · Tickers sin ops → último precio del Excel histórico
    prices = {}
    for t in set(list(target_wts.keys()) + _op_tickers):
        if t in yf_prices and t in _op_tickers:
            prices[t] = yf_prices[t]          # ops reales → precio live
        elif t in _last_excel_price:
            prices[t] = _last_excel_price[t]  # sin ops → Excel histórico
        else:
            prices[t] = 0.0

    # Etiquetas de fuente (para debug y banner)
    _n_filas_excel  = len(prices_hist) if not prices_hist.empty else 0
    _n_obs_modelo   = model.get("n_obs", 0) if model and not model.get("error") else 0
    px_source_model = (f"Excel {_n_filas_excel} filas · modelo {_n_obs_modelo} retornos"
                       if _n_obs_modelo else f"Excel {_n_filas_excel} filas · sin modelo")
    px_source_live  = ("Yahoo Finance (" + str(len(yf_prices)) + " tickers)"
                       if yf_prices else "Sin operaciones")
    px_source = px_source_model  # compatibilidad con referencias antiguas

    # ── Posiciones y alertas
    act_tickers = list(target_wts.keys())
    for t in _op_tickers:
        if t not in act_tickers: act_tickers.append(t)

    pos_df    = calc_positions(ops_df, prices, tc_actual, target_tickers=act_tickers)
    alerts_df = calc_alerts(pos_df, target_wts, tc_actual, monthly_add, thresh_rebal)
    has_act   = not alerts_df.empty and alerts_df["Alerta"].isin(["COMPRAR","VENDER"]).any()
    pending   = (alerts_df[alerts_df["Alerta"].isin(["COMPRAR","VENDER"])]
                 if not alerts_df.empty else pd.DataFrame())

    total_usd  = pos_df["Valor USD"].sum() if not pos_df.empty else 0.0
    total_mxn  = total_usd * tc_actual
    total_gain = pos_df["Ganancia USD"].sum() if not pos_df.empty else 0.0
    cost_base  = total_usd - total_gain
    gan_pct    = (total_gain / cost_base * 100) if cost_base > 0 else 0.0

    # ── Métricas del modelo
    # "modelo"    = calculado del Excel de ESTE portafolio
    # "sin_datos" = Sheet no conectado o datos insuficientes
    if model and not model.get("error"):
        rm = {
            "sharpe":  model["sharpe_tang"],
            "vol":     model["vol_tang"],
            "ret":     model["ret_tang"],
            "dd":      model["max_dd"],
            "var95":   model["var_95"],
            "n":       model["n_obs"],
            "source":  "modelo",
            "activos": model["tickers"],
            "vol_mv":  model.get("vol_mv", model["vol_tang"]),
        }
    else:
        rm = {
            "sharpe": 0.0, "vol": 0.0, "ret": 0.0,
            "dd": 0.0, "var95": 0.0, "n": 0,
            "source": "sin_datos",
            "activos": list(target_wts.keys()),
            "vol_mv": 0.0,
        }

    # TWR
    twr = calc_twr(ops_df, yf_prices or ops_prices, tc_actual)

    # ── DIAGNÓSTICO EXCEL — siempre visible ──────────────────────────────────
    # Muestra: hoja leída, número de filas reales, rango de fechas, activos, meses del modelo
    if sheet_id:
        _price_cols = [c for c in (prices_hist.columns if not prices_hist.empty else [])
                       if not c.startswith("__")]
        _rows_total = len(prices_hist)
        # Rango de fechas del Excel
        _fecha_ini = _fecha_fin = ""
        if not prices_hist.empty and "__fecha__" in prices_hist.columns:
            _fechas_validas = prices_hist["__fecha__"].dropna().tolist()
            if _fechas_validas:
                _fecha_ini = str(_fechas_validas[0])
                _fecha_fin = str(_fechas_validas[-1])
        _fecha_rango = (f" · Período: <b>{_fecha_ini}</b> a <b>{_fecha_fin}</b>"
                       if _fecha_ini else "")
        _n_meses_modelo = model.get("n_obs", 0) if model and not model.get("error") else 0

        if prices_hist.empty:
            _diag_cls = "diag-err"
            _diag_msg = (f"❌ No se encontró la hoja <b>Precios_{selected}</b> — "
                        f"crea la hoja en el Sheet con esa nombre exacto.")
        elif _rows_total < 13:
            _diag_cls = "diag-warn"
            _diag_msg = (f"⚠ Hoja <b>Precios_{selected}</b>: solo <b>{_rows_total} filas</b> "
                        f"(mínimo 13 para correr el modelo).{_fecha_rango}")
        else:
            _model_err = model.get("error") if model else "sin_sheet"
            if _model_err:
                _diag_cls = "diag-warn"
                _diag_msg = (f"⚠ Datos cargados ({_rows_total} filas) pero modelo falló: "
                            f"<b>{_model_err}</b>.{_fecha_rango}")
            else:
                _diag_cls = "diag-ok"
                _diag_msg = (
                    f"✅ Sheet conectado · Hoja: <b>Precios_{selected}</b> · "
                    f"<b>{_rows_total} filas</b> en Excel · "
                    f"Modelo corrió con <b>{_n_meses_modelo} retornos mensuales</b>"
                    f"{_fecha_rango} · "
                    f"Activos: <b>{', '.join(_price_cols)}</b>")
        st.markdown(f"<div class='diag {_diag_cls}'>{_diag_msg}</div>",
                   unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='diag diag-warn'>"
                   f"⚠ Sheet no configurado. Conecta tu Google Sheet para activar el modelo.</div>",
                   unsafe_allow_html=True)


    # ── Banner ────────────────────────────────────────────────────────────────
    hora = datetime.now().hour
    sal  = "Buenos días" if 6<=hora<13 else "Buenas tardes" if hora<20 else "Buenas noches"
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0d1b2a,#1a3a5c);border-radius:16px;
                padding:22px 28px;margin-bottom:20px;display:flex;
                justify-content:space-between;align-items:flex-start;
                box-shadow:0 4px 20px rgba(13,27,42,0.2);'>
        <div>
            <div style='font-family:"Space Grotesk",sans-serif;font-size:28px;
                        font-weight:700;color:white;margin-bottom:2px;'>{selected}</div>
            <div style='font-size:12px;color:#c9a227;'>{port_cfg.get("description","")}</div>
            <div style='font-size:10px;color:rgba(255,255,255,0.4);margin-top:4px;'>
                Modelo: {px_source_model} · Pesos: {wt_source}</div>
            <div style='font-size:10px;color:rgba(201,162,39,0.6);margin-top:1px;'>
                {'Posiciones: ' + px_source_live if _op_tickers else 'Sin operaciones aun - ingresa la primera compra'}</div>
        </div>
        <div style='text-align:right;'>
            <div style='font-size:10px;color:rgba(255,255,255,0.45);'>{sal},</div>
            <div style='font-size:16px;font-weight:600;color:white;'>
                <b style='color:#c9a227;'>{ud["name"]}</b></div>
            <div style='font-size:10px;color:rgba(255,255,255,0.4);margin-top:4px;'>
                {datetime.now().strftime("%d %b %Y %H:%M")}</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── DEBUG BAR (admin) ─────────────────────────────────────────────────────
    if is_a:
        st.markdown(
            f"<div class='dbg'>"
            f"MODELO [{selected}] "
            f"Activos: <b>{rm['activos']}</b> · "
            f"Sharpe: <b>{rm['sharpe']:.4f}</b> · "
            f"Vol: <b>{rm['vol']*100:.2f}%</b> · "
            f"Ret: <b>{rm['ret']*100:.2f}%</b> · "
            f"n_meses: <b>{rm['n']}</b> · "
            f"Fuente modelo: <b>{px_source_model}</b> · "
            f"{'Posiciones: ' + str(len(_op_tickers)) + ' tickers con ops | ' + px_source_live if _op_tickers else 'Sin operaciones aun'}"
            f"</div>",
            unsafe_allow_html=True)

    # El modelo corre con exactamente los activos que están en el Excel.
    # No hay validación contra una lista fija — el universo lo define el Excel.

# ══════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════

# ── DASHBOARD ─────────────────────────────────────────────────────────────
if page in ("📊 Dashboard","📊 Mi Portafolio") and selected:

    # ── Sin datos del modelo: mostrar aviso claro ─────────────────────────
    if rm["source"] == "sin_datos":
        st.info(
            f"📊 **{selected}** — El modelo aún no tiene datos. "
            f"Asegúrate de que la hoja **Precios_{selected}** exista en tu Sheet "
            f"con al menos 13 filas de precios históricos mensuales. "
            f"Mientras tanto, las tarjetas y semáforos mostrarán ceros."
        )

    # ── Valores del modelo (todos vienen de rm — calculados del Excel) ────
    _sharpe  = rm["sharpe"]   # Sharpe del portafolio tangencia
    _vol     = rm["vol"]      # Volatilidad anual
    _ret     = rm["ret"]      # Retorno anual esperado
    _dd      = rm["dd"]       # Máxima caída histórica
    _var95   = rm["var95"]    # VaR 95% mensual
    _n       = rm["n"]        # Número de observaciones

    # ── Umbrales: se derivan de los resultados del modelo, no de config ───
    # Sharpe: bueno si > 1.0, aceptable si > 0.5, débil si < 0.5
    _sr_bueno    = _sharpe >= 1.0
    _sr_aceptable= _sharpe >= 0.5
    _sr_color    = "#059669" if _sr_bueno else ("#d97706" if _sr_aceptable else "#ef4444")
    _sr_txt      = ("Excelente" if _sr_bueno else
                    "Aceptable" if _sr_aceptable else "Débil — revisar estrategia")

    # Volatilidad: comparar contra la volatilidad del portafolio de mín. varianza
    # Si no hay modelo, usar la volatilidad observada como referencia neutra
    _vol_ref     = model.get("vol_mv", _vol * 1.5) if model and not model.get("error") else _vol * 1.5
    _vol_alta    = _vol > _vol_ref * 1.3
    _vol_color   = "#059669" if not _vol_alta else "#d97706"
    _vol_txt     = ("Volatilidad controlada" if not _vol_alta else
                    f"Alta vs mín. varianza ({_vol_ref*100:.1f}%)")

    # Caída máxima: rojo si > 25%, amarillo si 10–25%, verde si < 10%
    _dd_pct      = abs(_dd * 100)
    _dd_color    = "#059669" if _dd_pct < 10 else ("#d97706" if _dd_pct < 25 else "#ef4444")
    _dd_txt      = ("Caída mínima" if _dd_pct < 10 else
                    "Caída moderada" if _dd_pct < 25 else "Caída importante")

    # VaR 95% mensual: rojo si > 10%, amarillo si 5–10%, verde si < 5%
    _var_pct     = abs(_var95 * 100)
    _var_color   = "#059669" if _var_pct < 5 else ("#d97706" if _var_pct < 10 else "#ef4444")
    _var_txt     = ("Riesgo bajo" if _var_pct < 5 else
                    "Riesgo moderado" if _var_pct < 10 else "Riesgo elevado")

    # Fuente de datos (modelo real o semilla)
    _fuente = "modelo Excel" if rm["source"] == "modelo" else "referencia"

    # ── KPI Row — 4 tarjetas ──────────────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)

    with c1:
        st.markdown(f"""<div class='card'>
            <div class='kpi-lbl'>Valor · {selected}</div>
            <div class='kpi-val'>${total_mxn:,.0f}</div>
            <div class='kpi-sub'>${total_usd:,.0f} USD · TC {tc_actual:.2f}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        sg = "pos" if total_gain>=0 else "neg"
        si = "+" if total_gain>=0 else ""
        st.markdown(f"""<div class='card'>
            <div class='kpi-lbl'>Ganancia / Pérdida · {selected}</div>
            <div class='kpi-val {sg}'>{si}${total_gain:,.0f} USD</div>
            <div class='kpi-sub'>{si}{gan_pct:.1f}% sobre costo</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        _src_badge = (
            "📊 Excel histórico" if rm["source"]=="modelo"
            else "⚠ Sin datos"
        )
        st.markdown(f"""<div class='card'>
            <div class='kpi-lbl'>Sharpe Ratio · {selected}</div>
            <div class='kpi-val' style='color:{_sr_color};'>{_sharpe:.3f}</div>
            <div class='kpi-sub'>{_sr_txt}</div>
            <div style='font-size:10px;color:#9ca3af;margin-top:4px;'>{_src_badge} · {_n} meses</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""<div class='card'>
            <div class='kpi-lbl'>Retorno Esperado · {selected}</div>
            <div class='kpi-val' style='color:#059669;'>{_ret*100:.2f}%</div>
            <div class='kpi-sub'>Volatilidad: {_vol*100:.1f}%</div>
            <div style='font-size:10px;color:#9ca3af;margin-top:4px;'>📊 Excel histórico · {_n} meses</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Semáforos — todos basados en resultados del modelo ─────────────────
    s1,s2,s3,s4 = st.columns(4)

    def sem(col, ico, lbl, val, txt, color):
        with col:
            st.markdown(f"""<div class='sem' style='border-top:4px solid {color};'>
                <div class='sem-ico'>{ico}</div>
                <div class='sem-lbl'>{lbl}</div>
                <div class='sem-val' style='color:{color};'>{val}</div>
                <div class='sem-txt' style='color:{color};'>{txt}</div>
            </div>""", unsafe_allow_html=True)

    _src_note = "📊 modelo Excel" if rm["source"]=="modelo" else "⚠ sin datos"

    sem(s1,
        "🟢" if _sr_bueno else ("🟡" if _sr_aceptable else "🔴"),
        f"Sharpe · {_src_note}",
        f"{_sharpe:.3f}",
        _sr_txt,
        _sr_color)

    sem(s2,
        "🟢" if not _vol_alta else "🟡",
        f"Volatilidad · {_src_note}",
        f"{_vol*100:.1f}%",
        _vol_txt,
        _vol_color)

    sem(s3,
        "🟢" if _dd_pct < 10 else ("🟡" if _dd_pct < 25 else "🔴"),
        f"Caída Máxima · {_src_note}",
        f"{_dd*100:.1f}%",
        _dd_txt,
        _dd_color)

    sem(s4,
        "🟢" if _var_pct < 5 else ("🟡" if _var_pct < 10 else "🔴"),
        f"VaR 95% · {_src_note}",
        f"{_var_pct:.2f}%",
        _var_txt,
        _var_color)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Alertas ───────────────────────────────────────────────────────────
    if has_act:
        st.markdown(f"<div class='sec'>📋 Acciones · {selected}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px;color:#6b7280;margin-bottom:12px;'>"
                   f"Desviaciones detectadas en <b>{selected}</b> — rebalancear:</div>",
                   unsafe_allow_html=True)
        ocols = st.columns(min(len(pending),4))
        for i,(_,r) in enumerate(pending.iterrows()):
            with ocols[i%len(ocols)]:
                if r["Alerta"]=="COMPRAR":
                    m=r["Compra Sugerida MXN"]; u=m/tc_actual
                    st.markdown(f"""<div style='background:#ecfdf5;border-radius:12px;
                        padding:14px;border-left:4px solid #059669;margin-bottom:8px;'>
                        <div style='font-weight:700;font-size:14px;color:#059669;'>🟢 COMPRAR {r["Ticker"]}</div>
                        <div style='font-size:12px;margin-top:6px;'>{r["Acciones a Comprar"]:.4f} acc
                        @ ${r["Precio Actual"]:,.2f}</div>
                        <div style='font-size:12px;color:#059669;font-weight:600;'>
                        ${u:,.2f} USD · ${m:,.0f} MXN</div>
                        <div style='font-size:10px;color:#6b7280;margin-top:4px;'>
                        {r["Peso Actual"]*100:.1f}% → {r["Peso Objetivo"]*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    m=r["Monto a Vender MXN"]; u=m/tc_actual
                    st.markdown(f"""<div style='background:#fef2f2;border-radius:12px;
                        padding:14px;border-left:4px solid #ef4444;margin-bottom:8px;'>
                        <div style='font-weight:700;font-size:14px;color:#ef4444;'>🔴 VENDER {r["Ticker"]}</div>
                        <div style='font-size:12px;margin-top:6px;'>{r["Acciones a Vender"]:.4f} acc
                        @ ${r["Precio Actual"]:,.2f}</div>
                        <div style='font-size:12px;color:#ef4444;font-weight:600;'>
                        ${u:,.2f} USD · ${m:,.0f} MXN</div>
                        <div style='font-size:10px;color:#6b7280;margin-top:4px;'>
                        {r["Peso Actual"]*100:.1f}% → {r["Peso Objetivo"]*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
    else:
        st.success(f"✅ {selected} en equilibrio.")

    # ── Cotizaciones ──────────────────────────────────────────────────────
    st.markdown(f"<div class='sec'>📡 Cotizaciones · {selected}</div>", unsafe_allow_html=True)
    if yf_prices:
        qcols = st.columns(min(len(target_wts),5))
        for qi,tkr in enumerate(target_wts.keys()):
            q = _yf.get(tkr,{})
            px = q.get("price")
            with qcols[qi%len(qcols)]:
                if not px:
                    st.markdown(f"<div class='card' style='text-align:center;'>"
                               f"<b>{tkr}</b><br><span style='color:#9ca3af;font-size:12px;'>"
                               f"Sin datos</span></div>", unsafe_allow_html=True)
                    continue
                chg=q.get("chg",0) or 0
                cc="#059669" if chg>=0 else "#ef4444"
                cbg="#ecfdf5" if chg>=0 else "#fef2f2"
                vsma=q.get("vsma")
                if vsma is None: sig,sc="#6b7280","Sin datos"
                elif vsma>20: sig,sc="#ef4444","MUY CARA"
                elif vsma>5:  sig,sc="#d97706","ALTA"
                elif vsma>=-5:sig,sc="#1e40af","JUSTA"
                elif vsma>=-20:sig,sc="#059669","BAJA"
                else:          sig,sc="#059669","MUY BAJA"
                h52=q.get("h52"); l52=q.get("l52")
                bp=max(0,min(100,(px-(l52 or 0))/((h52 or px+1)-(l52 or 0))*100)) if h52 and l52 else 50
                st.markdown(f"""<div class='card'>
                    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;'>
                        <span style='font-weight:700;font-size:14px;'>{tkr}</span>
                        <span style='background:{cbg};color:{cc};padding:2px 7px;
                                     border-radius:8px;font-size:10px;font-weight:700;'>
                            {'▲' if chg>=0 else '▼'} {abs(chg):.2f}%</span></div>
                    <div style='font-size:24px;font-weight:700;color:#111827;margin-bottom:8px;'>
                        ${px:,.2f}</div>
                    <div style='background:{cbg};border-radius:7px;padding:5px 9px;
                                margin-bottom:8px;border-left:3px solid {sc};'>
                        <div style='font-size:10px;font-weight:700;color:{sc};'>{sig}</div>
                        <div style='font-size:9px;color:#6b7280;'>
                        {f"{vsma:+.1f}% vs prom 52s" if vsma is not None else ""}</div></div>
                    <div style='font-size:9px;color:#9ca3af;margin-bottom:3px;'>
                        52s: ${l52:,.2f}–${h52:,.2f}</div>
                    <div style='background:#f3f4f6;border-radius:4px;height:4px;position:relative;'>
                        <div style='position:absolute;left:{bp:.0f}%;top:-4px;width:12px;height:12px;
                                    border-radius:50%;background:#1a3a5c;transform:translateX(-50%);
                                    border:2px solid white;'></div></div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("Yahoo Finance no disponible.")

    # ── Posiciones + Donuts ───────────────────────────────────────────────
    st.markdown(f"<div class='sec'>📋 Posiciones · {selected}</div>", unsafe_allow_html=True)
    pc1,pc2 = st.columns([3,2])
    with pc1:
        if not alerts_df.empty:
            def badge(a):
                if a=="COMPRAR": return "<span style='background:#dcfce7;color:#166534;padding:2px 8px;border-radius:8px;font-size:10px;font-weight:700;'>🟢 COMPRAR</span>"
                if a=="VENDER":  return "<span style='background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:8px;font-size:10px;font-weight:700;'>🔴 VENDER</span>"
                return "<span style='background:#f0fdf4;color:#166534;padding:2px 8px;border-radius:8px;font-size:10px;font-weight:700;'>✅ OK</span>"
            rows=""
            for _,r in alerts_df.iterrows():
                gc2="#059669" if r["Ganancia USD"]>=0 else "#ef4444"
                gs="+" if r["Ganancia USD"]>=0 else ""
                dc="#ef4444" if abs(r["Desviación"])>thresh_rebal else "#059669"
                rows+=(f"<tr><td><b>{r['Ticker']}</b></td>"
                       f"<td>{r['Cantidad']:.4f}</td>"
                       f"<td>${r['Precio Actual']:,.2f}</td>"
                       f"<td><b>${r['Valor MXN']:,.0f}</b></td>"
                       f"<td>{r['Peso Actual']*100:.1f}%</td>"
                       f"<td>{r['Peso Objetivo']*100:.1f}%</td>"
                       f"<td style='color:{dc}'>{r['Desviación']*100:+.1f}%</td>"
                       f"<td style='color:{gc2}'>{gs}${r['Ganancia USD']:,.0f}</td>"
                       f"<td>{badge(r['Alerta'])}</td></tr>")
            st.markdown(f"<div class='tbl'><table><thead><tr>"
                       f"<th>Ticker</th><th>Cantidad</th><th>Precio</th>"
                       f"<th>Valor MXN</th><th>Peso%</th><th>Obj%</th>"
                       f"<th>Desv.</th><th>Ganancia</th><th>Alerta</th>"
                       f"</tr></thead><tbody>{rows}</tbody></table></div>",
                       unsafe_allow_html=True)
        else:
            st.info("Sin posiciones. Registra operaciones.")

    with pc2:
        st.markdown(f"<div style='font-size:13px;font-weight:600;margin-bottom:8px;'>"
                   f"Distribución Objetivo · {selected}</div>", unsafe_allow_html=True)
        # El donut SIEMPRE muestra los pesos objetivo del modelo.
        # Si hay operaciones, también muestra el peso actual real.
        _pal  = ["#1a3a5c","#c9a227","#059669","#ef4444","#5b4fcf","#0e7490","#b45309","#6d28d9"]
        _all_t = list(dict.fromkeys(
            (list(alerts_df["Ticker"]) if not alerts_df.empty else []) +
            list(target_wts.keys())))
        _tc3  = {t:_pal[i%len(_pal)] for i,t in enumerate(_all_t)}
        _has_actual = (not alerts_df.empty and
                       "Peso Actual" in alerts_df.columns and
                       alerts_df["Peso Actual"].sum() > 0)

        if _has_actual:
            # Dos donuts: Actual + Objetivo
            _fig = make_subplots(1,2,
                specs=[[{"type":"pie"},{"type":"pie"}]],
                subplot_titles=["Actual","Objetivo Modelo"])
            _fig.add_trace(go.Pie(
                labels=alerts_df["Ticker"],
                values=alerts_df["Peso Actual"],
                hole=0.52, textinfo="label+percent", textfont_size=10,
                marker=dict(colors=[_tc3.get(t,"#888") for t in alerts_df["Ticker"]],
                            line=dict(color="white",width=2)),
                showlegend=True,
                hovertemplate="<b>%{label}</b><br>Actual: %{percent}<extra></extra>"
            ),1,1)
            _fig.add_trace(go.Pie(
                labels=list(target_wts.keys()),
                values=list(target_wts.values()),
                hole=0.52, textinfo="label+percent", textfont_size=10,
                marker=dict(colors=[_tc3.get(t,"#888") for t in target_wts],
                            line=dict(color="white",width=2)),
                showlegend=False,
                hovertemplate="<b>%{label}</b><br>Objetivo: %{percent}<extra></extra>"
            ),1,2)
            _fig.update_layout(
                height=300, paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10,r=10,t=40,b=60),
                legend=dict(orientation="h",y=-0.25,x=0.5,xanchor="center",font_size=10),
                annotations=[
                    dict(text="Actual", x=0.18,y=0.5,font_size=12,showarrow=False),
                    dict(text="Objetivo",x=0.82,y=0.5,font_size=12,showarrow=False)])
        else:
            # Solo donut de Objetivo (siempre visible aunque no haya operaciones)
            _fig = go.Figure()
            _fig.add_trace(go.Pie(
                labels=list(target_wts.keys()),
                values=list(target_wts.values()),
                hole=0.52, textinfo="label+percent", textfont_size=11,
                marker=dict(colors=[_tc3.get(t,"#888") for t in target_wts],
                            line=dict(color="white",width=2)),
                showlegend=True,
                hovertemplate="<b>%{label}</b><br>Objetivo: %{percent}<extra></extra>",
                title=dict(text="Objetivo<br>Modelo",font_size=13)
            ))
            _fig.update_layout(
                height=300, paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10,r=10,t=30,b=60),
                legend=dict(orientation="h",y=-0.25,x=0.5,xanchor="center",font_size=10))
            st.caption("Sin operaciones — mostrando composición objetivo del modelo")

        st.plotly_chart(_fig, use_container_width=True)

    # ── Gráfico frontera (si hay modelo) ─────────────────────────────────
    if model and not model.get("error") and model.get("frontier") and is_a:
        st.markdown(f"<div class='sec'>🗺️ Frontera Eficiente · {selected}</div>",
                   unsafe_allow_html=True)
        fr=model["frontier"]
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=[p["vol"]*100 for p in fr],y=[p["ret"]*100 for p in fr],
            mode="lines+markers",line=dict(color=port_color,width=2.5),
            marker=dict(size=7,color=port_color),name="Frontera",
            hovertemplate="σ=%{x:.1f}% · R=%{y:.1f}%<extra></extra>"))
        fig2.add_trace(go.Scatter(x=[model["vol_tang"]*100],y=[model["ret_tang"]*100],
            mode="markers",marker=dict(size=14,color="#c9a227",symbol="star",
                                        line=dict(width=2,color="#0d1b2a")),
            name=f"Tangencia SR={model['sharpe_tang']:.3f}"))
        fig2.add_trace(go.Scatter(x=[model["vol_mv"]*100],y=[model["ret_mv"]*100],
            mode="markers",marker=dict(size=12,color="#059669",symbol="diamond"),
            name="Mín. Varianza"))
        rf_p=model["rf"]*100
        vt=model["vol_tang"]*100; rt=model["ret_tang"]*100
        xc=[0,max([p["vol"]*100 for p in fr])*1.1]
        yc=[rf_p,rf_p+(rt-rf_p)/vt*xc[1]] if vt>0 else [rf_p,rf_p]
        fig2.add_trace(go.Scatter(x=xc,y=yc,mode="lines",
            line=dict(color="#ef4444",width=1.5,dash="dash"),
            name=f"CML (Rf={rf_p:.2f}%)"))
        fig2.update_layout(height=360,paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(243,244,246,0.8)",
            xaxis=dict(title="Riesgo σ (%)",gridcolor="#e5e7eb"),
            yaxis=dict(title="Retorno esperado (%)",gridcolor="#e5e7eb"),
            margin=dict(l=10,r=10,t=10,b=10),
            legend=dict(orientation="h",y=1.05,font_size=10))
        st.plotly_chart(fig2, use_container_width=True)


# ── OPERACIONES ────────────────────────────────────────────────────────────
elif page in ("📋 Operaciones","📋 Registrar Operación") and selected:
    st.markdown(f"<div class='sec'>📋 Operaciones · {selected}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:12px;color:#6b7280;margin-bottom:14px;'>"
               f"Se guardan en <b>Ops_{selected}</b>. Solo afectan a este portafolio.</div>",
               unsafe_allow_html=True)

    with st.form(f"form_{selected}", clear_on_submit=True):
        a,b,c = st.columns(3)
        with a:
            ff=st.date_input("Fecha",value=date.today())
            ft=st.selectbox("Tipo",["Compra","Venta"])
        with b:
            fk=st.text_input("Ticker",placeholder="SHY").strip().upper()
            fq=st.number_input("Cantidad",min_value=0.0001,value=1.0,step=0.001,format="%.4f")
        with c:
            fp=st.number_input("Precio USD",min_value=0.01,value=100.0,step=0.01)
            fc=st.number_input("Comisión USD",min_value=0.0,value=0.50,step=0.01)
        ftc=st.number_input("TC MXN/USD",value=tc_actual,step=0.01,format="%.2f")
        if st.form_submit_button("💾 Registrar",use_container_width=True,type="primary"):
            if fk:
                if use_sheets:
                    ok=save_tab_row(sheet_id,f"Ops_{selected}",
                                   [str(ff),fk,ft,fq,fp,fc,ftc,cur])
                    st.success(f"✅ Guardado en Ops_{selected}") if ok else st.error("Error Sheets")
                    if ok: st.cache_data.clear()
                else:
                    st.info("Demo: conecta Sheet para persistencia.")
            else: st.error("Ingresa un ticker")

    if not ops_df.empty:
        st.markdown("<div class='sec'>Historial</div>", unsafe_allow_html=True)
        f1,f2=st.columns(2)
        with f1: ftk=st.multiselect("Ticker",ops_df["Ticker"].unique().tolist(),
                                      default=ops_df["Ticker"].unique().tolist(),key="otk")
        with f2: ftp=st.multiselect("Tipo",["Compra","Venta"],default=["Compra","Venta"],key="otp")
        fo=ops_df[ops_df["Ticker"].isin(ftk)&ops_df["Tipo"].isin(ftp)].copy()
        fo["Total USD"]=fo["Cantidad"]*fo["Precio_USD"]+fo["Comision_USD"]
        fo["Total MXN"]=fo["Total USD"]*fo["TC_MXN"]
        rh=""
        for _,r in fo.iterrows():
            tc2="#059669" if r["Tipo"]=="Compra" else "#ef4444"
            tb2="#dcfce7" if r["Tipo"]=="Compra" else "#fee2e2"
            bd=f"<span style='background:{tb2};color:{tc2};padding:2px 8px;border-radius:8px;font-size:10px;font-weight:700;'>{r['Tipo']}</span>"
            rh+=(f"<tr><td>{r['Fecha']}</td><td><b>{r['Ticker']}</b></td><td>{bd}</td>"
                 f"<td>{r['Cantidad']:.4f}</td><td>${r['Precio_USD']:,.2f}</td>"
                 f"<td>${r['Comision_USD']:,.2f}</td><td>{r['TC_MXN']:.2f}</td>"
                 f"<td><b>${r['Total USD']:,.2f}</b></td><td><b>${r['Total MXN']:,.0f}</b></td></tr>")
        st.markdown(f"<div class='tbl'><table><thead><tr>"
                   f"<th>Fecha</th><th>Ticker</th><th>Tipo</th><th>Cant.</th>"
                   f"<th>Precio</th><th>Comisión</th><th>TC</th>"
                   f"<th>Total USD</th><th>Total MXN</th></tr></thead>"
                   f"<tbody>{rh}</tbody></table></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background:#1a3a5c;color:white;border-radius:8px;"
                   f"padding:8px 16px;margin-top:8px;display:flex;justify-content:space-between;'>"
                   f"<span>{len(fo)} operaciones</span>"
                   f"<span>Total: <b>${fo['Total MXN'].sum():,.0f} MXN</b></span></div>",
                   unsafe_allow_html=True)


# ── RIESGO ─────────────────────────────────────────────────────────────────
elif page in ("⚡ Riesgo","📈 Mi Inversión") and selected:
    st.markdown(f"<div class='sec'>⚡ Análisis de Riesgo · {selected}</div>",
               unsafe_allow_html=True)
    if not model or model.get("error"):
        st.warning(f"Modelo no disponible: {model.get('error','sin Sheet') if model else 'sin Sheet'}. "
                  f"Conecta el Sheet con la hoja Precios_{selected}.")
    else:
        t1,t2,t3 = st.tabs(["📊 Correlaciones","🎲 Monte Carlo","📈 VaR por Activo"])
        with t1:
            corr=model["corr"]; tks=model["tickers"]
            fig=go.Figure(go.Heatmap(z=corr,x=tks,y=tks,colorscale="RdBu_r",
                zmin=-1,zmax=1,
                text=[[f"{corr[i][j]:.2f}" for j in range(len(tks))] for i in range(len(tks))],
                texttemplate="%{text}",textfont=dict(size=10),
                hovertemplate="<b>%{x} vs %{y}</b>: %{z:.3f}<extra></extra>"))
            fig.update_layout(title=f"Correlaciones · {selected}",height=420,
                paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)
        with t2:
            sims=np.array(model["mc_sims"])*100
            fig=go.Figure()
            fig.add_trace(go.Histogram(x=sims,nbinsx=60,marker_color=port_color,
                opacity=0.75,hovertemplate="Ret: %{x:.1f}%<br>Frec: %{y}<extra></extra>"))
            for pv,pn,pc2 in [(model["mc_p5"]*100,"P5","#ef4444"),
                               (model["mc_p50"]*100,"Mediana","#c9a227"),
                               (model["mc_p95"]*100,"P95","#059669")]:
                fig.add_vline(x=pv,line=dict(color=pc2,dash="dash",width=2),
                             annotation_text=f"{pn}: {pv:.1f}%",annotation_position="top")
            fig.update_layout(title=f"Distribución rendimientos anuales · {selected}",
                xaxis_title="Rendimiento (%)",height=340,
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(243,244,246,0.8)",
                margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)
            _mc_cards = "".join([
                f"<div class='card' style='text-align:center;border-top:3px solid {_c2};'>"
                f"<div style='font-size:9px;color:#6b7280;'>{_n2}</div>"
                f"<div style='font-size:18px;font-weight:700;color:{_c2};'>{_v2:.1f}%</div></div>"
                for _n2,_v2,_c2 in [
                    ("P5",    model["mc_p5"] *100,"#ef4444"),
                    ("P25",   model["mc_p25"]*100,"#d97706"),
                    ("Med.",  model["mc_p50"]*100,"#c9a227"),
                    ("P75",   model["mc_p75"]*100,"#2d7a4a"),
                    ("P95",   model["mc_p95"]*100,"#059669"),
                ]])
            st.markdown(
                f"<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:8px;'>"
                f"{_mc_cards}</div>",
                unsafe_allow_html=True)
        with t3:
            va=model["var_asset"]; ca=model["cvar_asset"]; tks2=list(va.keys())
            fig=go.Figure()
            fig.add_trace(go.Bar(x=tks2,y=[va[t]*100 for t in tks2],
                name="VaR 95%",marker_color="#ef4444",opacity=0.8))
            fig.add_trace(go.Bar(x=tks2,y=[ca[t]*100 for t in tks2],
                name="CVaR 95%",marker_color="#7f1d1d",opacity=0.8))
            fig.update_layout(title=f"VaR y CVaR por activo · {selected}",
                barmode="group",height=320,paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(243,244,246,0.8)",margin=dict(l=10,r=10,t=50,b=10))
            st.plotly_chart(fig, use_container_width=True)


# ── MODELO VS REAL ──────────────────────────────────────────────────────────
elif page == "🔀 Modelo vs Real" and selected:
    st.markdown(f"<div class='sec'>🔀 Modelo vs Realidad · {selected}</div>",
               unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown(f"""<div class='card' style='border-left:5px solid #1a3a5c;'>
            <div class='kpi-lbl'>📊 Rendimiento Modelo</div>
            <div class='kpi-val' style='color:#1a3a5c;'>
                {f"{rm['ret']*100:+.2f}%" if rm['ret'] else 'Sin datos'}</div>
            <div class='kpi-sub'>Retorno anual esperado · {rm['source']}</div>
            <div style='margin-top:10px;padding-top:10px;border-top:1px solid #e5e7eb;font-size:11px;color:#6b7280;'>
                Vol: {rm['vol']*100:.1f}% · Sharpe: {rm['sharpe']:.3f}</div>
        </div>""", unsafe_allow_html=True)
    twr_pct = twr.get("twr",None); twr_inv=twr.get("inv",0); twr_val=twr.get("val",0)
    twr_ann=twr.get("ann",None); fd=twr.get("first",None)
    with c2:
        tc2c="#059669" if (twr_pct or 0)>=0 else "#ef4444"
        st.markdown(f"""<div class='card' style='border-left:5px solid {tc2c};'>
            <div class='kpi-lbl'>📈 Tu Rendimiento (TWR)</div>
            <div class='kpi-val' style='color:{tc2c};'>
                {f"{twr_pct:+.2f}%" if twr_pct is not None else "Sin operaciones"}</div>
            <div class='kpi-sub'>Desde {fd.strftime("%d %b %Y") if fd and pd.notna(fd) else "—"}</div>
            <div style='margin-top:10px;padding-top:10px;border-top:1px solid #e5e7eb;font-size:11px;color:#6b7280;'>
                Invertido: ${twr_inv:,.2f} USD · Actual: ${twr_val:,.2f} USD<br>
                Anualizado: {f"{twr_ann:.1f}%" if twr_ann else "—"}</div>
        </div>""", unsafe_allow_html=True)
    diff = (twr_pct - rm['ret']*100) if twr_pct is not None and rm['ret'] else None
    with c3:
        dc="#059669" if (diff or 0)>=0 else "#ef4444"
        if diff is None: interp="Conecta Sheet y registra operaciones"
        elif diff>=5: interp="🚀 Superas al modelo"
        elif diff>=0: interp="✅ En línea con el modelo"
        elif diff>=-10: interp="🟡 Ligeramente bajo el modelo"
        else: interp="⚠ Diferencia importante"
        st.markdown(f"""<div class='card' style='border-left:5px solid {dc};'>
            <div class='kpi-lbl'>⚖️ Diferencia</div>
            <div class='kpi-val' style='color:{dc};'>
                {f"{diff:+.2f}%" if diff is not None else "—"}</div>
            <div class='kpi-sub'>Real menos Modelo</div>
            <div style='margin-top:10px;padding-top:10px;border-top:1px solid #e5e7eb;
                        font-size:11px;color:#1a2332;line-height:1.5;'>{interp}</div>
        </div>""", unsafe_allow_html=True)

    # Horizonte
    if model and not model.get("error") and model.get("horizons"):
        hz=model["horizons"]
        fig=go.Figure()
        hy=list(hz.keys()); means=[hz[y]["mean"] for y in hy]
        p5s=[hz[y]["p5"] for y in hy]; p95s=[hz[y]["p95"] for y in hy]
        fig.add_trace(go.Scatter(x=hy+hy[::-1],y=p95s+p5s[::-1],
            fill="toself",fillcolor=f"rgba(59,130,246,0.1)",line=dict(color="rgba(0,0,0,0)"),
            name="Intervalo 90%",hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=hy,y=means,mode="lines+markers",
            line=dict(color=port_color,width=2.5),
            marker=dict(size=9,color="#c9a227"),name="Rendimiento esperado",
            hovertemplate="<b>%{x} años</b><br>%{y:.1f}%<extra></extra>"))
        if twr_pct is not None:
            fig.add_hline(y=twr_pct,line=dict(color="#059669",dash="dash",width=2),
                         annotation_text=f"Tu rendimiento actual: {twr_pct:.1f}%")
        fig.update_layout(title=f"Proyección por horizonte · {selected}",
            xaxis_title="Horizonte (años)",yaxis_title="Rendimiento Total (%)",
            height=340,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(243,244,246,0.8)",
            margin=dict(l=10,r=10,t=50,b=10),legend=dict(orientation="h",y=1.05,font_size=10))
        st.plotly_chart(fig, use_container_width=True)


# ── ADMIN ──────────────────────────────────────────────────────────────────
elif page == "👥 Admin" and is_a:
    st.markdown("<div class='sec'>👥 Administración</div>", unsafe_allow_html=True)

    # ── Crear portafolio — solo 3 campos, el modelo calcula todo lo demás ───
    with st.expander("➕ Crear Nuevo Portafolio", expanded=not bool(get_all_ports())):
        st.markdown("""<div class='diag diag-ok'>
            <b>Cómo funciona:</b><br>
            1. Ingresa solo nombre, descripción y color<br>
            2. Crea la hoja <b>Precios_NOMBRE</b> en tu Sheet con los precios históricos<br>
            3. El modelo calcula automáticamente: Sharpe, volatilidad, retorno, pesos óptimos<br>
            4. Nada está hardcodeado — todo viene de tus datos
        </div>""", unsafe_allow_html=True)

        a1, a2 = st.columns([3, 1])
        with a1:
            np_name  = st.text_input("Nombre del portafolio (sin espacios)",
                                      placeholder="Ej: Conservador, AltoRendimiento",
                                      key="npn")
            np_desc  = st.text_input("Descripción breve",
                                      placeholder="Ej: Perfil de alto rendimiento",
                                      key="npd")
        with a2:
            np_color = st.color_picker("Color identificador", value="#3b82f6", key="npc")

        st.markdown(
            "<div style='font-size:11px;color:#6b7280;margin-top:4px;'>"
            "Una vez creado, la app buscará la hoja <b>Precios_{nombre}</b> en tu Sheet. "
            "El modelo corre automáticamente: calcula retornos log, "
            "optimiza Sharpe con límite 30%, "
            "y produce Sharpe, volatilidad, retorno y pesos óptimos. "
            "La tasa libre de riesgo es 2.64% (igual que MATLAB)."
            "</div>",
            unsafe_allow_html=True)

        if st.button("➕ Crear Portafolio", key="btn_create", type="primary"):
            n = np_name.strip().replace(" ", "")
            if not n:
                st.error("El nombre es requerido.")
            elif n in get_all_ports():
                st.error(f"Ya existe un portafolio llamado '{n}'.")
            else:
                st.session_state.custom_portfolios[n] = {
                    "description": np_desc.strip() or n,
                    "color":       np_color,
                    "border":      np_color,
                    "rf":          0.0264,   # fijo igual que MATLAB
                    "target":      {},
                    "profile":     "personalizado",
                    "icon":        "📁",
                    "bg":          "linear-gradient(135deg,#f9fafb,#f3f4f6)",
                }
                st.success(
                    f"✅ Portafolio **{n}** creado. "
                    f"Asegúrate de que la hoja **Precios_{n}** exista en tu Sheet.")
                st.cache_data.clear()
                st.rerun()

    # ── Lista portafolios activos ──────────────────────────────────────────
    all_p=get_all_ports()
    if all_p:
        st.markdown("<div class='sec'>Portafolios Activos</div>", unsafe_allow_html=True)
        for pn,pd2 in list(all_p.items()):
            c1,c2=st.columns([5,1])
            with c1:
                _mod=run_model(sheet_id,pn) if use_sheets and sheet_id else {}
                _ph =load_prices(sheet_id,pn) if use_sheets and sheet_id else pd.DataFrame()
                _sr =_mod.get("sharpe_tang","—") if _mod and not _mod.get("error") else "—"
                _rt =_mod.get("ret_tang","—")    if _mod and not _mod.get("error") else "—"
                _vol=_mod.get("vol_tang","—")    if _mod and not _mod.get("error") else "—"
                _nobs=_mod.get("n_obs",0)        if _mod and not _mod.get("error") else 0
                _sr_s  = f"{_sr:.3f}"      if isinstance(_sr,float)  else _sr
                _rt_s  = f"{_rt*100:.2f}%" if isinstance(_rt,float)  else _rt
                _vol_s = f"{_vol*100:.2f}%" if isinstance(_vol,float) else _vol
                _nfilas= len(_ph) if not _ph.empty else 0
                _cols_p= [c for c in (_ph.columns if not _ph.empty else [])
                          if not c.startswith("__")]
                # Fechas del Excel
                _f0=_fn=""
                if not _ph.empty and "__fecha__" in _ph.columns:
                    _flist=_ph["__fecha__"].dropna().tolist()
                    if _flist: _f0,_fn=str(_flist[0]),str(_flist[-1])

                st.markdown(
                    f"<div class='card' style='border-left:4px solid {pd2['color']};'>"
                    f"<div style='font-size:14px;font-weight:700;margin-bottom:6px;'>{pn}"
                    f" <span style='font-size:11px;color:#6b7280;font-weight:400;'>"
                    f"— {pd2.get('description','')}</span></div>"
                    # Resultados del modelo
                    f"<div style='display:grid;grid-template-columns:repeat(4,1fr);"
                    f"gap:8px;margin-bottom:8px;'>"
                    f"<div style='background:#f9fafb;border-radius:8px;padding:8px 10px;"
                    f"border-top:3px solid {pd2['color']};'>"
                    f"<div style='font-size:9px;color:#6b7280;'>Sharpe</div>"
                    f"<div style='font-size:16px;font-weight:700;color:{pd2['color']};'>{_sr_s}</div></div>"
                    f"<div style='background:#f9fafb;border-radius:8px;padding:8px 10px;"
                    f"border-top:3px solid #059669;'>"
                    f"<div style='font-size:9px;color:#6b7280;'>Retorno</div>"
                    f"<div style='font-size:16px;font-weight:700;color:#059669;'>{_rt_s}</div></div>"
                    f"<div style='background:#f9fafb;border-radius:8px;padding:8px 10px;"
                    f"border-top:3px solid #d97706;'>"
                    f"<div style='font-size:9px;color:#6b7280;'>Volatilidad</div>"
                    f"<div style='font-size:16px;font-weight:700;color:#d97706;'>{_vol_s}</div></div>"
                    f"<div style='background:#f9fafb;border-radius:8px;padding:8px 10px;"
                    f"border-top:3px solid #6b7280;'>"
                    f"<div style='font-size:9px;color:#6b7280;'>n retornos</div>"
                    f"<div style='font-size:16px;font-weight:700;color:#6b7280;'>{_nobs}</div></div>"
                    f"</div>"
                    # Datos del Excel
                    f"<div style='font-size:10px;color:#6b7280;'>"
                    f"📊 Excel: <b>{_nfilas} filas</b>"
                    f"{f' · {_f0} → {_fn}' if _f0 else ''} · "
                    f"Activos: <b>{', '.join(_cols_p)}</b><br>"
                    f"Hojas: <code>Precios_{pn}</code> + <code>Ops_{pn}</code>"
                    f"</div></div>",
                    unsafe_allow_html=True)

                # ── Verificación de datos: tabla de primeros y últimos precios ──
                with st.expander(f"🔍 Verificar datos de Precios_{pn} vs MATLAB"):
                    if _ph.empty:
                        st.error(f"No se encontró la hoja Precios_{pn}")
                    else:
                        _show = _ph[[c for c in _ph.columns if not c.startswith("__")]].copy()
                        st.markdown("**Primeras 3 filas del Excel** (deben coincidir con las primeras filas de tu precios.xlsx de MATLAB):")
                        st.dataframe(_show.head(3).round(4), use_container_width=True)
                        st.markdown("**Últimas 3 filas del Excel:**")
                        st.dataframe(_show.tail(3).round(4), use_container_width=True)
                        st.markdown(
                            f"<div class='diag diag-ok'>"
                            f"ℹ Si estos precios coinciden con tu archivo MATLAB <code>precios.xlsx</code>, "
                            f"el modelo producirá resultados idénticos. "
                            f"Si hay diferencias en cualquier valor → los resultados diferirán.<br>"
                            f"<b>Total: {len(_show)} filas × {len(_show.columns)} activos</b>"
                            f"</div>", unsafe_allow_html=True)
                        # Estadísticos de retornos (para comparar con MATLAB output)
                        if len(_show) >= 2:
                            import numpy as _np
                            _P = _show.values.astype(float)
                            _R = _np.diff(_np.log(_P), axis=0)
                            _mu = _R.mean(axis=0) * 12
                            _sd = _R.std(axis=0, ddof=1) * _np.sqrt(12)
                            _stats = pd.DataFrame({
                                "Activo": _show.columns,
                                "Ret. anual esperado": [f"{x*100:.2f}%" for x in _mu],
                                "Vol. anual": [f"{x*100:.2f}%" for x in _sd],
                                "Sharpe individual": [f"{(_mu[i]-0.0264)/_sd[i]:.3f}" if _sd[i]>0 else "—"
                                                     for i in range(len(_mu))]
                            })
                            st.markdown("**Estadísticos por activo** (retornos log anualizados, igual que MATLAB):")
                            st.dataframe(_stats, use_container_width=True, hide_index=True)
                            st.caption("Si el Sharpe de AltoRendimiento en MATLAB es 1.23, los retornos individuales de NVDA, AAPL, CAT deben ser muy altos aquí.")
            with c2:
                if st.button("🗑", key=f"del_{pn}"):
                    del st.session_state.custom_portfolios[pn]; st.rerun()

    # ── Usuarios ──────────────────────────────────────────────────────────
    st.markdown("<div class='sec'>Gestión de Usuarios</div>", unsafe_allow_html=True)
    if "adm_users" not in st.session_state:
        st.session_state.adm_users={k:dict(v) for k,v in USERS.items()}
    ap=list(get_all_ports().keys())
    for uk2,ud2 in list(st.session_state.adm_users.items()):
        with st.expander(f"{ud2.get('avatar','?')} {ud2.get('name',uk2)} — {ud2.get('role','').upper()}"):
            e1,e2,e3=st.columns(3)
            with e1:
                nn=st.text_input("Nombre",value=ud2.get("name",""),key=f"nn_{uk2}")
                np2=st.text_input("PIN (4 dig)",value=ud2.get("pin",""),key=f"np_{uk2}",max_chars=4)
                ne=st.text_input("Email",value=ud2.get("email",""),key=f"ne_{uk2}")
            with e2:
                na=st.text_input("Avatar",value=ud2.get("avatar",""),key=f"na_{uk2}",max_chars=2)
                nc=st.color_picker("Color",value=ud2.get("color","#1a3a5c"),key=f"nc_{uk2}")
                nr=st.selectbox("Rol",["investor","admin"],
                    index=0 if ud2.get("role")!="admin" else 1,key=f"nr_{uk2}")
            with e3:
                curr_p=ud2.get("portfolios")
                sel_p=st.multiselect("Portafolios",ap,
                    default=ap if curr_p is None else [x for x in curr_p if x in ap],
                    key=f"npt_{uk2}")
                all_acc=st.checkbox("Todos",value=(curr_p is None),key=f"nall_{uk2}")
            ux1,ux2=st.columns([1,5])
            with ux1:
                if st.button("💾",key=f"upd_{uk2}"):
                    st.session_state.adm_users[uk2].update({
                        "name":nn,"pin":np2,"avatar":na.upper(),"color":nc,
                        "role":nr,"email":ne,
                        "portfolios":None if all_acc else sel_p})
                    st.success("✅")
            with ux2:
                if uk2!=cur and st.button("🗑 Eliminar",key=f"del_u_{uk2}"):
                    del st.session_state.adm_users[uk2]; st.rerun()

    # Agregar usuario
    with st.expander("➕ Nuevo Usuario"):
        b1,b2,b3=st.columns(3)
        with b1:
            nuk=st.text_input("Username",key="nuk"); nun=st.text_input("Nombre",key="nun")
            nup=st.text_input("PIN",max_chars=4,key="nup"); nue=st.text_input("Email",key="nue")
        with b2:
            nua=st.text_input("Avatar",max_chars=2,key="nua")
            nuc=st.color_picker("Color","#1a7a4a",key="nuc")
            nur=st.selectbox("Rol",["investor","admin"],key="nur")
        with b3:
            nup2=st.multiselect("Portafolios",ap,key="nup2")
            nue_=st.text_input("Email",key="nue2")
        if st.button("➕ Crear",key="btn_nu"):
            errs=[]
            if not nuk.strip(): errs.append("Username requerido")
            if nuk in st.session_state.adm_users: errs.append("Ya existe")
            if len(nup)!=4: errs.append("PIN 4 dígitos")
            for e in errs: st.error(e)
            if not errs:
                st.session_state.adm_users[nuk.strip()]={
                    "name":nun,"pin":nup,"avatar":nua.upper(),"color":nuc,
                    "sheet_id":"","role":nur,"email":nue_,
                    "portfolios":None if not nup2 else nup2}
                st.success(f"Usuario '{nun}' creado."); st.rerun()

    st.divider()
    sc1,sc2=st.columns([1,3])
    with sc1:
        if st.button("💾 Guardar cambios",type="primary",use_container_width=True):
            USERS.clear(); USERS.update(st.session_state.adm_users)
            ok,msg=write_users(USERS)
            st.success(f"✅ {msg}") if ok else st.error(f"❌ {msg}")
    with sc2:
        st.markdown("<div style='padding:10px 0;font-size:12px;color:#6b7280;'>"
                   "Guarda para persistir en app.py. Reinicia la app para que los PINs surtan efecto.</div>",
                   unsafe_allow_html=True)
