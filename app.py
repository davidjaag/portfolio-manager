"""
EGADA — Estrategia Global de Activos Dinámicos Avanzados
Portfolio Manager v2.0 — Core MATLAB Model Integration
Arquitectura limpia: modelo separado de UI, portafolios independientes.
"""

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
import io

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════
import plotly.io as pio
pio.templates["egada"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(232,240,248,0.25)",
        font=dict(family="DM Sans, sans-serif", color="#1a2332", size=12),
        colorway=["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#06b6d4","#84cc16","#f43f5e"],
        xaxis=dict(gridcolor="rgba(13,27,42,0.06)", gridwidth=1,
                   linecolor="rgba(13,27,42,0.12)", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="rgba(13,27,42,0.06)", gridwidth=1,
                   linecolor="rgba(13,27,42,0.12)", showgrid=True, zeroline=False),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(13,27,42,0.1)",
                    borderwidth=1, font=dict(size=11)),
        margin=dict(l=16, r=16, t=48, b=16),
    )
)
pio.templates.default = "egada"


# ══════════════════════════════════════════════════════════════════════════════
# ─── LAYER 1: CORE MODEL (traducción exacta del MATLAB madre) ─────────────
# Sustituye cualquier modelo anterior. Cada portafolio usa solo sus datos.
# ══════════════════════════════════════════════════════════════════════════════

def core_model_run(prices_df: pd.DataFrame, rf: float = 0.0264, max_weight: float = 0.30) -> dict:
    """
    Réplica exacta del modelo MATLAB madre (Project_R_30_max).
    Recibe DataFrame de precios históricos mensuales (filas=fechas, cols=tickers).
    Incluye SPX si está presente como benchmark.
    Devuelve dict con todos los outputs del modelo: tangencia, frontera, SML, VaR, etc.
    max_weight = 0.30 (restricción de 30% máximo por activo, igual que MATLAB).
    """
    from scipy.optimize import minimize

    if prices_df is None or prices_df.empty or len(prices_df) < 13:
        return {}

    try:
        # ─── Inicialización (igual que MATLAB clc/clear) ───────────────────
        prices_df = prices_df.copy().dropna()

        # Detectar y separar SPX si existe
        spx_col = None
        for c in prices_df.columns:
            if "SPX" in str(c).upper():
                spx_col = c
                break

        # Activos del portafolio (sin SPX)
        asset_cols = [c for c in prices_df.columns if c != spx_col]
        n_assets = len(asset_cols)

        if n_assets < 2:
            return {}

        # ─── Rendimientos logarítmicos (diff(log(P))) ─────────────────────
        prices_assets = prices_df[asset_cols].values.astype(float)
        log_returns = np.diff(np.log(prices_assets), axis=0)
        n_obs = log_returns.shape[0]

        # Estadísticos anualizados
        mu_ann = log_returns.mean(axis=0) * 12          # AssetMean * 12
        cov_ann = np.cov(log_returns.T, ddof=1) * 12    # cov * 12
        rf_ann = rf

        # ─── Portafolio Tangencia: max Sharpe (con límite 30%) ────────────
        def neg_sharpe(w):
            ret = float(np.dot(w, mu_ann))
            vol = float(np.sqrt(w @ cov_ann @ w))
            return -(ret - rf_ann) / vol if vol > 1e-9 else 0.0

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.0, max_weight)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        res_tang = minimize(neg_sharpe, w0, method="SLSQP",
                           bounds=bounds, constraints=constraints,
                           options={"ftol": 1e-10, "maxiter": 2000})
        w_tang = np.clip(res_tang.x, 0, max_weight)
        w_tang /= w_tang.sum()

        ret_tang = float(np.dot(w_tang, mu_ann))
        vol_tang = float(np.sqrt(w_tang @ cov_ann @ w_tang))
        sharpe_tang = (ret_tang - rf_ann) / vol_tang if vol_tang > 1e-9 else 0.0

        # ─── Mínima Varianza ──────────────────────────────────────────────
        def port_var(w):
            return float(w @ cov_ann @ w)

        res_mv = minimize(port_var, w0, method="SLSQP",
                         bounds=bounds, constraints=constraints,
                         options={"ftol": 1e-10, "maxiter": 2000})
        w_mv = np.clip(res_mv.x, 0, max_weight)
        w_mv /= w_mv.sum()
        ret_mv = float(np.dot(w_mv, mu_ann))
        vol_mv = float(np.sqrt(w_mv @ cov_ann @ w_mv))

        # ─── Frontera Eficiente (20 puntos, igual que MATLAB numPorts=20) ─
        n_frontier = 20
        ret_min = ret_mv
        ret_max = float(np.max(mu_ann)) * 0.98
        frontier_pts = []
        for target_r in np.linspace(ret_min, ret_max, n_frontier):
            cons_f = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, r=target_r: np.dot(w, mu_ann) - r},
            ]
            res_f = minimize(port_var, w0, method="SLSQP",
                            bounds=bounds, constraints=cons_f,
                            options={"ftol": 1e-9, "maxiter": 500})
            if res_f.success:
                wf = np.clip(res_f.x, 0, max_weight)
                wf /= wf.sum()
                vf = float(np.sqrt(wf @ cov_ann @ wf))
                frontier_pts.append({
                    "vol": vf, "ret": float(np.dot(wf, mu_ann)),
                    "weights": {asset_cols[i]: float(wf[i]) for i in range(n_assets)}
                })

        # ─── SML / Betas (usando SPX si existe) ──────────────────────────
        betas = {}
        alphas_capm = {}
        if spx_col and spx_col in prices_df.columns:
            spx_prices = prices_df[spx_col].values.astype(float)
            spx_rets = np.diff(np.log(spx_prices))
            # Alinear longitud
            min_len = min(len(log_returns), len(spx_rets))
            Rm = spx_rets[-min_len:]
            Ra = log_returns[-min_len:]
            Rm_ann = float(np.mean(Rm) * 12)
            MRP = Rm_ann - rf_ann
            for i, ticker in enumerate(asset_cols):
                cov_im = float(np.cov(Ra[:, i], Rm)[0, 1])
                var_m = float(np.var(Rm, ddof=1))
                beta_i = cov_im / var_m if var_m > 0 else 1.0
                capm_i = rf_ann + beta_i * MRP
                betas[ticker] = beta_i
                alphas_capm[ticker] = mu_ann[i] - capm_i
        else:
            for i, ticker in enumerate(asset_cols):
                betas[ticker] = 1.0
                alphas_capm[ticker] = 0.0

        # ─── Correlación (G1) ─────────────────────────────────────────────
        std_v = np.sqrt(np.diag(cov_ann))
        corr_matrix = cov_ann / np.outer(std_v, std_v)
        np.fill_diagonal(corr_matrix, 1.0)

        # ─── VaR / CVaR 95% — Simulación MC 10,000 (igual que MATLAB) ────
        n_sim = 10000
        T_sim = 12  # 1 año (12 meses)
        sim_port_annual = np.zeros(n_sim)
        np.random.seed(42)
        for k in range(n_sim):
            sim_r = np.random.multivariate_normal(mu_ann / 12, cov_ann / 12, T_sim)
            sim_port_annual[k] = float(np.prod(1 + sim_r @ w_tang) - 1)

        p5  = float(np.percentile(sim_port_annual, 5))
        p25 = float(np.percentile(sim_port_annual, 25))
        p50 = float(np.percentile(sim_port_annual, 50))
        p75 = float(np.percentile(sim_port_annual, 75))
        p95 = float(np.percentile(sim_port_annual, 95))
        prob_loss = float(np.mean(sim_port_annual < 0))

        # VaR / CVaR individuales por activo (G13)
        var_by_asset = {}
        cvar_by_asset = {}
        acc_rets_by_asset = np.zeros((n_assets, n_sim))
        for k in range(n_sim):
            sim_r = np.random.multivariate_normal(mu_ann / 12, cov_ann / 12, T_sim)
            for i in range(n_assets):
                acc_rets_by_asset[i, k] = float(np.prod(1 + sim_r[:, i]) - 1)
        alpha_var = 0.05
        for i, ticker in enumerate(asset_cols):
            sorted_r = np.sort(acc_rets_by_asset[i])
            idx_var = max(0, int(alpha_var * n_sim) - 1)
            var_by_asset[ticker] = -float(sorted_r[idx_var])
            cvar_by_asset[ticker] = -float(np.mean(sorted_r[:idx_var + 1]))

        # ─── VaR por portafolios en frontera (G14) ────────────────────────
        var_frontier = []
        cvar_frontier = []
        for fp in frontier_pts:
            wf_arr = np.array([fp["weights"].get(t, 0) for t in asset_cols])
            sims_fp = np.zeros(500)
            for k in range(500):
                sim_r = np.random.multivariate_normal(mu_ann / 12, cov_ann / 12, T_sim)
                sims_fp[k] = float(np.prod(1 + sim_r @ wf_arr) - 1)
            sorted_fp = np.sort(-sims_fp)
            idx_v = max(0, int(alpha_var * 500) - 1)
            var_frontier.append(float(sorted_fp[idx_v]))
            cvar_frontier.append(float(np.mean(sorted_fp[:idx_v + 1])))

        # ─── Intervalos por horizonte (Intervalos por Horizonte) ──────────
        horizontes = [1, 3, 5, 10, 20]
        horizon_intervals = {}
        for h_yr in horizontes:
            meses_h = h_yr * 12
            sim_h = np.zeros(n_sim)
            for k in range(n_sim):
                sim_r_h = np.random.multivariate_normal(mu_ann / 12, cov_ann / 12, meses_h)
                sim_h[k] = float((np.prod(1 + sim_r_h @ w_tang) - 1) * 100)
            horizon_intervals[h_yr] = {
                "p5":   float(np.percentile(sim_h, 5)),
                "mean": float(np.mean(sim_h)),
                "p95":  float(np.percentile(sim_h, 95)),
            }

        # ─── Retorno histórico real del portafolio tangencia ──────────────
        port_log_rets = log_returns @ w_tang
        cum_values = np.exp(np.cumsum(port_log_rets))
        mx_cum = np.maximum.accumulate(cum_values)
        max_dd = float(np.min((cum_values - mx_cum) / mx_cum))
        var_95_hist = float(np.percentile(port_log_rets, 5))
        cvar_mask = port_log_rets < var_95_hist
        cvar_95_hist = float(port_log_rets[cvar_mask].mean()) if cvar_mask.any() else var_95_hist

        # ─── Resultado completo ────────────────────────────────────────────
        return {
            # Identificación
            "tickers":        asset_cols,
            "n_obs":          n_obs,
            "rf":             rf_ann,
            "max_weight":     max_weight,
            # Tangencia
            "w_tang":         {asset_cols[i]: float(w_tang[i]) for i in range(n_assets)},
            "ret_tang":       ret_tang,
            "vol_tang":       vol_tang,
            "sharpe_tang":    sharpe_tang,
            # Mínima varianza
            "w_mv":           {asset_cols[i]: float(w_mv[i]) for i in range(n_assets)},
            "ret_mv":         ret_mv,
            "vol_mv":         vol_mv,
            # Frontera
            "frontier":       frontier_pts,
            # Matrices
            "mu_ann":         {asset_cols[i]: float(mu_ann[i]) for i in range(n_assets)},
            "cov_ann":        cov_ann,
            "corr_matrix":    corr_matrix,
            "std_ann":        {asset_cols[i]: float(std_v[i]) for i in range(n_assets)},
            # SML
            "betas":          betas,
            "alphas_capm":    alphas_capm,
            # Riesgo histórico
            "max_dd":         max_dd,
            "var_95":         var_95_hist,
            "cvar_95":        cvar_95_hist,
            "port_log_rets":  port_log_rets.tolist(),
            # MC Intervals
            "mc_p5":          p5,
            "mc_p25":         p25,
            "mc_p50":         p50,
            "mc_p75":         p75,
            "mc_p95":         p95,
            "mc_prob_loss":   prob_loss,
            "sim_annual":     sim_port_annual.tolist(),
            # VaR por activo
            "var_by_asset":   var_by_asset,
            "cvar_by_asset":  cvar_by_asset,
            # VaR frontera
            "var_frontier":   var_frontier,
            "cvar_frontier":  cvar_frontier,
            # Horizontes
            "horizon_intervals": horizon_intervals,
        }
    except Exception as e:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# ─── LAYER 2: DATOS — Google Sheets (Excel) por portafolio ───────────────
# Cada portafolio lee SOLO sus hojas: Precios_<nombre> y Ops_<nombre>
# ══════════════════════════════════════════════════════════════════════════════

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Mapeo Bloomberg → ticker corto (acepta ambos formatos)
BLOOMBERG_TO_TICKER = {
    # Nombres Bloomberg formato largo → ticker corto
    "AAPL US Equity": "AAPL", "ASTS US Equity": "ASTS",
    "JPM US Equity":  "JPM",  "9988 HK Equity": "9988HK",
    "CAT US Equity":  "CAT",  "NOW US Equity":  "NOW",
    "NVDA US Equity": "NVDA", "NEM US Equity":  "NEM",
    "TSLA US Equity": "TSLA", "BATS LN Equity": "BATS",
    "CIEN US Equity": "CIEN", "EXPE US Equity": "EXPE",
    "EXPE US": "EXPE",        "SPX Index": "SPX", "SPX": "SPX",
    "GLD US Equity": "GLD",   "IAU US Equity":  "IAU",
    "GE US Equity":  "GE",
    # Portafolios MATLAB nuevos
    "SHY US Equity":  "SHY",  "IEI US Equity":  "IEI",
    "VCSH US Equity": "VCSH", "QQQ US Equity":  "QQQ",
    "WMT US Equity":  "WMT",
    # Tickers cortos directos (sin sufijo) — acepta ambos formatos
    "SHY": "SHY", "IEI": "IEI", "VCSH": "VCSH",
    "QQQ": "QQQ", "WMT": "WMT", "GLD": "GLD",
    "IAU": "IAU", "NEM": "NEM", "AAPL": "AAPL",
    "CAT": "CAT", "NVDA": "NVDA",
}

YF_TICKER_MAP = {
    # Portafolios MATLAB actuales
    "SHY":    "SHY",
    "IEI":    "IEI",
    "VCSH":   "VCSH",
    "GLD":    "GLD",
    "QQQ":    "QQQ",
    "IAU":    "IAU",
    "WMT":    "WMT",
    "AAPL":   "AAPL",
    "CAT":    "CAT",
    "NVDA":   "NVDA",
    "NEM":    "NEM",
    "SPX":    "^GSPC",
    # Otros activos posibles
    "BATS":   "BATS.L",
    "TSLA":   "TSLA",
    "ASTS":   "ASTS",
    "NOW":    "NOW",
    "JPM":    "JPM",
    "9988HK": "9988.HK",
    "GE":     "GE",
    "CIEN":   "CIEN",
    "EXPE":   "EXPE",
}


@st.cache_resource
def get_gspread_client():
    try:
        creds_dict = st.secrets["gcp_service_account"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        return gspread.authorize(creds)
    except Exception:
        return None


@st.cache_data(ttl=30)
def load_sheet(sheet_id: str, tab_name: str) -> pd.DataFrame:
    client = get_gspread_client()
    if client is None:
        return pd.DataFrame()
    try:
        sh = client.open_by_key(sheet_id)
        ws = sh.worksheet(tab_name)
        data = ws.get_all_records()
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def save_row(sheet_id: str, tab_name: str, row_data: list) -> bool:
    client = get_gspread_client()
    if client is None:
        return False
    try:
        sh = client.open_by_key(sheet_id)
        ws = sh.worksheet(tab_name)
        ws.append_row(row_data)
        return True
    except Exception:
        return False


@st.cache_data(ttl=3600)
def load_price_history(sheet_id: str, portfolio_name: str) -> pd.DataFrame:
    """
    Lee Precios_<portfolio_name> del Sheet.
    Devuelve DataFrame de precios mensuales con columnas = tickers.
    Acepta nombres Bloomberg o tickers cortos.
    NUNCA mezcla datos entre portafolios.
    """
    client = get_gspread_client()
    if client is None:
        return pd.DataFrame()
    try:
        sh = client.open_by_key(sheet_id)
        tab = f"Precios_{portfolio_name}"
        try:
            ws = sh.worksheet(tab)
        except Exception:
            return pd.DataFrame()

        data = ws.get_all_values()
        if len(data) < 3:
            return pd.DataFrame()

        headers = data[0]
        rows = []
        for row in data[1:]:
            if all(str(v).strip() == "" for v in row):
                continue
            vals = {}
            for j, col in enumerate(headers):
                if col.lower() in ("fecha", "date", ""):
                    continue
                tkr = BLOOMBERG_TO_TICKER.get(col) or col.strip().upper()
                try:
                    v = float(str(row[j]).replace(",", ".").replace(" ", ""))
                    if v > 0:
                        vals[tkr] = v
                except Exception:
                    pass
            if len(vals) >= 2:
                rows.append(vals)

        if len(rows) < 13:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_live_quotes(tickers: list) -> dict:
    if not YF_AVAILABLE:
        return {}
    results = {}
    for tkr in tickers:
        yf_sym = YF_TICKER_MAP.get(tkr, tkr)
        try:
            obj = yf.Ticker(yf_sym)
            info = obj.fast_info
            price = getattr(info, "last_price", None)
            prev_close = getattr(info, "previous_close", None)
            w52_high = getattr(info, "year_high", None)
            w52_low = getattr(info, "year_low", None)
            chg_pct = (price / prev_close - 1) * 100 if price and prev_close else None
            ma52w = ((w52_high or 0) + (w52_low or 0)) / 2 if w52_high and w52_low else None
            vs_ma52w = (price / ma52w - 1) * 100 if price and ma52w else None
            results[tkr] = {
                "price": price, "prev_close": prev_close,
                "change_pct": chg_pct, "week52_high": w52_high,
                "week52_low": w52_low, "ma52w": ma52w, "vs_ma52w": vs_ma52w,
                "open": getattr(info, "open", None),
                "day_high": getattr(info, "day_high", None),
                "day_low": getattr(info, "day_low", None),
            }
        except Exception:
            results[tkr] = {}
    return results


@st.cache_data(ttl=300)
def get_live_prices(tickers: list) -> dict:
    q = get_live_quotes(tickers)
    return {t: (q[t].get("price") or 0) for t in tickers if t in q}


# ══════════════════════════════════════════════════════════════════════════════
# ─── LAYER 3: CÁLCULOS DE POSICIONES Y ALERTAS ───────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def calc_positions(ops_df: pd.DataFrame, prices: dict, tc: float,
                   target_tickers: list = None) -> pd.DataFrame:
    op_tickers = list(ops_df["Ticker"].unique()) if not ops_df.empty else []
    all_tickers = list(op_tickers)
    if target_tickers:
        for t in target_tickers:
            if t not in all_tickers:
                all_tickers.append(t)
    if not all_tickers:
        return pd.DataFrame()

    rows = []
    for tkr in all_tickers:
        qty_total = 0.0
        costo_prom = 0.0
        if not ops_df.empty and tkr in ops_df["Ticker"].values:
            tkr_ops = ops_df[ops_df["Ticker"] == tkr]
            compras = tkr_ops[tkr_ops["Tipo"] == "Compra"]
            ventas = tkr_ops[tkr_ops["Tipo"] == "Venta"]
            qty_c = compras["Cantidad"].sum()
            qty_v = ventas["Cantidad"].sum() if not ventas.empty else 0.0
            qty_total = qty_c - qty_v
            if qty_c > 0:
                costo_prom = (compras["Cantidad"] * compras["Precio_USD"]).sum() / qty_c

        precio_actual = prices.get(tkr, 0.0)
        valor_usd = qty_total * precio_actual
        valor_mxn = valor_usd * tc
        ganancia_usd = (precio_actual - costo_prom) * qty_total
        ganancia_pct = (precio_actual / costo_prom - 1) if costo_prom > 0 else 0.0

        rows.append({
            "Ticker": tkr, "Cantidad": qty_total,
            "Precio Actual": precio_actual, "Costo Prom": costo_prom,
            "Valor USD": valor_usd, "Valor MXN": valor_mxn,
            "Ganancia USD": ganancia_usd, "Ganancia %": ganancia_pct,
        })
    return pd.DataFrame(rows)


def calc_alerts(positions_df: pd.DataFrame, target_weights: dict,
                tc: float, monthly_contrib: float,
                threshold_rebal: float = 0.05) -> pd.DataFrame:
    if positions_df.empty:
        return pd.DataFrame()
    total_mxn = positions_df["Valor MXN"].sum()
    df = positions_df.copy()
    df["Peso Objetivo"] = df["Ticker"].map(target_weights).fillna(0.0)

    if total_mxn == 0:
        df["Peso Actual"] = 0.0
        df["Desviación"] = -df["Peso Objetivo"]
        df["Alerta"] = df["Peso Objetivo"].apply(lambda w: "COMPRAR" if w > 0 else "OK")
        df["MXN Diferencia"] = df["Peso Objetivo"] * monthly_contrib
        df["Compra Sugerida MXN"] = df["MXN Diferencia"].clip(lower=0)
        df["Acciones a Comprar"] = df.apply(
            lambda r: r["Compra Sugerida MXN"] / (r["Precio Actual"] * tc)
            if r["Precio Actual"] > 0 else 0, axis=1)
        df["Monto a Vender MXN"] = 0.0
        df["Acciones a Vender"] = 0.0
        return df

    df["Peso Actual"] = df["Valor MXN"] / total_mxn
    df["Desviación"] = df["Peso Actual"] - df["Peso Objetivo"]

    def get_alert(row):
        if row["Peso Objetivo"] == 0 and row["Peso Actual"] > 0:
            return "VENDER"
        if row["Desviación"] > threshold_rebal:
            return "VENDER"
        if row["Desviación"] < -threshold_rebal:
            return "COMPRAR"
        return "OK"

    df["Alerta"] = df.apply(get_alert, axis=1)
    df["MXN Diferencia"] = df.apply(
        lambda r: r["Peso Objetivo"] * total_mxn - r["Valor MXN"], axis=1)
    df["Compra Sugerida MXN"] = df["MXN Diferencia"].clip(lower=0)
    df["Acciones a Comprar"] = df.apply(
        lambda r: r["Compra Sugerida MXN"] / (r["Precio Actual"] * tc)
        if r["Precio Actual"] > 0 else 0, axis=1)
    df["Monto a Vender MXN"] = df["MXN Diferencia"].clip(upper=0).abs()
    df["Acciones a Vender"] = df.apply(
        lambda r: r["Monto a Vender MXN"] / (r["Precio Actual"] * tc)
        if r["Precio Actual"] > 0 else 0, axis=1)
    return df


def calc_twr(ops_df: pd.DataFrame, live_prices: dict, tc: float) -> dict:
    """
    Time-Weighted Return desde la primera compra real del usuario.
    Compara con rendimiento modelo usando precios actuales de Yahoo Finance.
    """
    if ops_df.empty:
        return {}
    try:
        ops = ops_df.copy()
        ops["Fecha"] = pd.to_datetime(ops["Fecha"], errors="coerce")
        ops = ops.dropna(subset=["Fecha"]).sort_values("Fecha")
        first_date = ops["Fecha"].min()

        # Valor invertido total (compras - ventas a costo)
        compras = ops[ops["Tipo"] == "Compra"]
        ventas = ops[ops["Tipo"] == "Venta"]
        costo_total = (compras["Cantidad"] * compras["Precio_USD"]).sum()
        ventas_total = (ventas["Cantidad"] * ventas["Precio_USD"]).sum()
        net_invested = costo_total - ventas_total

        # Valor actual en USD
        valor_actual_usd = 0.0
        for tkr in ops["Ticker"].unique():
            tkr_ops = ops[ops["Ticker"] == tkr]
            qty_c = tkr_ops[tkr_ops["Tipo"] == "Compra"]["Cantidad"].sum()
            qty_v = tkr_ops[tkr_ops["Tipo"] == "Venta"]["Cantidad"].sum() if not ventas.empty else 0
            qty = qty_c - qty_v
            price_live = live_prices.get(tkr, 0.0)
            valor_actual_usd += qty * price_live

        twr_pct = (valor_actual_usd - net_invested) / net_invested * 100 if net_invested > 0 else 0.0
        dias = (datetime.now() - first_date).days if pd.notna(first_date) else 0
        retorno_anualizado = ((1 + twr_pct / 100) ** (365 / dias) - 1) * 100 if dias > 30 else None

        return {
            "first_date": first_date,
            "net_invested_usd": net_invested,
            "valor_actual_usd": valor_actual_usd,
            "twr_pct": twr_pct,
            "retorno_anualizado": retorno_anualizado,
            "dias": dias,
        }
    except Exception:
        return {}


def calc_model_return_history(prices_df: pd.DataFrame, model: dict) -> pd.DataFrame:
    """
    Construye la evolución mensual del portafolio tangencia con precios históricos.
    Retorna DataFrame {"Mes", "Valor", "Retorno"}.
    """
    if prices_df.empty or not model:
        return pd.DataFrame()
    try:
        w_tang = model.get("w_tang", {})
        tickers = [t for t in w_tang if t in prices_df.columns]
        if not tickers:
            return pd.DataFrame()
        w_arr = np.array([w_tang[t] for t in tickers])
        p_sub = prices_df[tickers].values.astype(float)
        # Rendimiento mensual del portafolio tangencia
        p_rets = np.diff(p_sub, axis=0) / p_sub[:-1]
        port_rets = p_rets @ w_arr
        cum = np.cumprod(1 + port_rets)
        rows = []
        for i, val in enumerate(cum):
            rows.append({
                "Mes": f"M{i+2:02d}",
                "Valor": float(val * 100),  # base 100
                "Retorno": float(port_rets[i] * 100),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# ─── LAYER 4: CONFIGURACIÓN DE PORTAFOLIOS BASE ───────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def get_base_portfolios() -> dict:
    """
    4 portafolios base.

    UNIVERSO = DINÁMICO: se lee de la hoja Precios_<nombre> en el Excel.
    El modelo core_model_run() recibe los activos que estén en esa hoja
    en el momento de correr — no hay universo hardcodeado.

    'target' = pesos semilla de MATLAB (respaldo cuando Sheets no está conectado).
    En runtime estos SE SOBREESCRIBEN con los pesos calculados por el optimizador
    usando los datos reales del Excel.

    Métricas de referencia (de MATLAB, imagen confirmada):
      Conservador   → Ret  3.48% | Vol  6.29% | Sharpe 0.13 | Intervalo 90%: -6.8% a 14.7%
      Moderado      → Ret 12.25% | Vol 10.26% | Sharpe 0.94 | Intervalo 90%: -4.8% a 32.7%
      Balanceado    → Ret 17.93% | Vol 12.38% | Sharpe 1.24 | Intervalo 90%: -2.7% a 44.9%
      AltoRend.     → Ret 32.44% | Vol 24.18% | Sharpe 1.23 | Intervalo 90%: -9.7% a 96.7%

    Los activos de cada portafolio los define tu Excel — no esta función.
    Si agregas o quitas tickers de Precios_<nombre>, el modelo los toma
    automáticamente en el siguiente recálculo sin tocar el código.
    """
    return {
        # ── PORTAFOLIO CONSERVADOR ─────────────────────────────────────────
        # Universo real: lo define Precios_Conservador en el Excel.
        # Pesos semilla MATLAB (respaldo): SHY 30%, IEI 30%, VCSH 30%, SPX 10%
        "Conservador": {
            "description": "Perfil Conservador",
            "color": "#10b981",
            "rf": 0.0264,
            "profile": "conservador",
            # Universo semilla — sobreescrito por columnas del Excel en runtime
            "universe": ["SHY", "IEI", "VCSH", "SPX"],
            # Pesos semilla MATLAB (respaldo sin Sheet)
            "target": {"SHY": 0.30, "IEI": 0.30, "VCSH": 0.30, "SPX": 0.10},
            # Métricas de referencia MATLAB
            "matlab_ret":   0.0348,
            "matlab_vol":   0.0629,
            "matlab_sharpe": 0.13,
            "matlab_p5":   -0.068,
            "matlab_p95":   0.147,
            "objetivo": "Preservar capital con mínima volatilidad",
            "horizonte": "1–4 años",
            "tolerancia": "Muy Baja",
            "tolerancia_pct": 15,
            "horizonte_years": 3,
            "volatilidad_range": (4, 8),
            "retorno_obj": (3, 8),
            "sharpe_obj": 0.13,
            "icon": "🛡️",
            "bg": "linear-gradient(135deg,#edfaf3 0%,#d5f5e3 100%)",
            "border": "#1a7a4a",
        },
        # ── PORTAFOLIO MODERADO ────────────────────────────────────────────
        # Universo real: lo define Precios_Moderado en el Excel.
        # Pesos semilla MATLAB: GLD 30%, SHY 27%, QQQ 30%, SPX 13%
        "Moderado": {
            "description": "Perfil Moderado",
            "color": "#f59e0b",
            "rf": 0.0264,
            "profile": "moderado",
            "universe": ["GLD", "SHY", "QQQ", "SPX"],
            "target": {"GLD": 0.30, "SHY": 0.27, "QQQ": 0.30, "SPX": 0.13},
            "matlab_ret":   0.1225,
            "matlab_vol":   0.1026,
            "matlab_sharpe": 0.94,
            "matlab_p5":   -0.048,
            "matlab_p95":   0.327,
            "objetivo": "Equilibrio entre crecimiento y estabilidad",
            "horizonte": "3–7 años",
            "tolerancia": "Media",
            "tolerancia_pct": 45,
            "horizonte_years": 5,
            "volatilidad_range": (8, 12),
            "retorno_obj": (10, 15),
            "sharpe_obj": 0.94,
            "icon": "⚖️",
            "bg": "linear-gradient(135deg,#fffbee 0%,#fef3cc 100%)",
            "border": "#c9a227",
        },
        # ── PORTAFOLIO BALANCEADO ──────────────────────────────────────────
        # Universo real: lo define Precios_Balanceado en el Excel.
        # Pesos semilla MATLAB: GLD 30%, IAU 30%, WMT 30%, SPX 10%
        "Balanceado": {
            "description": "Perfil Balanceado",
            "color": "#3b82f6",
            "rf": 0.0264,
            "profile": "balanceado",
            "universe": ["GLD", "IAU", "WMT", "SPX"],
            "target": {"GLD": 0.30, "IAU": 0.30, "WMT": 0.30, "SPX": 0.10},
            "matlab_ret":   0.1793,
            "matlab_vol":   0.1238,
            "matlab_sharpe": 1.24,
            "matlab_p5":   -0.027,
            "matlab_p95":   0.449,
            "objetivo": "Maximizar Sharpe Ratio — eficiencia óptima",
            "horizonte": "5–10 años",
            "tolerancia": "Media-Alta",
            "tolerancia_pct": 55,
            "horizonte_years": 7,
            "volatilidad_range": (10, 15),
            "retorno_obj": (15, 22),
            "sharpe_obj": 1.24,
            "icon": "📊",
            "bg": "linear-gradient(135deg,#eff6ff 0%,#dbeafe 100%)",
            "border": "#3b82f6",
        },
        # ── PORTAFOLIO ALTO RENDIMIENTO ────────────────────────────────────
        # Universo real: lo define Precios_AltoRendimiento en el Excel.
        # Pesos semilla MATLAB: AAPL 21%, CAT 30%, NVDA 30%, NEM 19%
        "AltoRendimiento": {
            "description": "Perfil Agresivo",
            "color": "#ef4444",
            "rf": 0.0264,
            "profile": "agresivo",
            "universe": ["AAPL", "CAT", "NVDA", "NEM"],
            "target": {"AAPL": 0.21, "CAT": 0.30, "NVDA": 0.30, "NEM": 0.19},
            "matlab_ret":   0.3244,
            "matlab_vol":   0.2418,
            "matlab_sharpe": 1.23,
            "matlab_p5":   -0.097,
            "matlab_p95":   0.967,
            "objetivo": "Maximizar retorno absoluto — crecimiento agresivo",
            "horizonte": "5–10 años",
            "tolerancia": "Alta",
            "tolerancia_pct": 80,
            "horizonte_years": 8,
            "volatilidad_range": (20, 28),
            "retorno_obj": (28, 40),
            "sharpe_obj": 1.23,
            "icon": "🚀",
            "bg": "linear-gradient(135deg,#fdf0ef 0%,#fadbd8 100%)",
            "border": "#c0392b",
        },
    }


def get_all_portfolios() -> dict:
    base = get_base_portfolios()
    custom = st.session_state.get("custom_portfolios", {})
    return {**base, **custom}


def get_user_portfolios(username: str) -> list:
    user = USERS.get(username, {})
    allowed = user.get("portfolios")
    all_ports = list(get_all_portfolios().keys())
    return all_ports if allowed is None else [p for p in allowed if p in all_ports]


# ══════════════════════════════════════════════════════════════════════════════
# ─── LAYER 5: DEMO DATA ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def get_demo_ops(portfolio_name: str) -> pd.DataFrame:
    """
    Operaciones de ejemplo alineadas con los universos reales de MATLAB.
    Solo se usan cuando no hay Sheet conectado (modo demo/desarrollo).
    """
    ops_map = {
        # Conservador: SHY, IEI, VCSH (+ SPX benchmark)
        "Conservador": [
            {"Fecha": "2025-01-15", "Ticker": "SHY",  "Tipo": "Compra", "Cantidad": 120.0, "Precio_USD": 82.50,  "Comision_USD": 0.50, "TC_MXN": 20.05},
            {"Fecha": "2025-01-15", "Ticker": "IEI",  "Tipo": "Compra", "Cantidad": 86.0,  "Precio_USD": 114.80, "Comision_USD": 0.50, "TC_MXN": 20.05},
            {"Fecha": "2025-01-15", "Ticker": "VCSH", "Tipo": "Compra", "Cantidad": 130.0, "Precio_USD": 75.40,  "Comision_USD": 0.50, "TC_MXN": 20.05},
            {"Fecha": "2025-06-01", "Ticker": "SHY",  "Tipo": "Compra", "Cantidad": 30.0,  "Precio_USD": 83.10,  "Comision_USD": 0.50, "TC_MXN": 20.00},
        ],
        # Moderado: GLD, SHY, QQQ (+ SPX benchmark)
        "Moderado": [
            {"Fecha": "2025-01-20", "Ticker": "GLD",  "Tipo": "Compra", "Cantidad": 43.0,  "Precio_USD": 185.00, "Comision_USD": 0.50, "TC_MXN": 20.10},
            {"Fecha": "2025-01-20", "Ticker": "SHY",  "Tipo": "Compra", "Cantidad": 130.0, "Precio_USD": 82.50,  "Comision_USD": 0.50, "TC_MXN": 20.10},
            {"Fecha": "2025-01-20", "Ticker": "QQQ",  "Tipo": "Compra", "Cantidad": 25.0,  "Precio_USD": 460.00, "Comision_USD": 0.50, "TC_MXN": 20.10},
            {"Fecha": "2025-06-01", "Ticker": "GLD",  "Tipo": "Compra", "Cantidad": 10.0,  "Precio_USD": 220.00, "Comision_USD": 0.50, "TC_MXN": 19.80},
        ],
        # Balanceado: GLD, IAU, WMT (+ SPX benchmark)
        "Balanceado": [
            {"Fecha": "2025-01-15", "Ticker": "GLD",  "Tipo": "Compra", "Cantidad": 43.0,  "Precio_USD": 185.00, "Comision_USD": 0.50, "TC_MXN": 20.60},
            {"Fecha": "2025-01-15", "Ticker": "IAU",  "Tipo": "Compra", "Cantidad": 216.0, "Precio_USD": 37.00,  "Comision_USD": 0.50, "TC_MXN": 20.60},
            {"Fecha": "2025-01-15", "Ticker": "WMT",  "Tipo": "Compra", "Cantidad": 105.0, "Precio_USD": 88.00,  "Comision_USD": 0.50, "TC_MXN": 20.60},
            {"Fecha": "2025-06-01", "Ticker": "GLD",  "Tipo": "Compra", "Cantidad": 10.0,  "Precio_USD": 220.00, "Comision_USD": 0.50, "TC_MXN": 19.80},
        ],
        # Alto Rendimiento: AAPL, CAT, NVDA, NEM
        "AltoRendimiento": [
            {"Fecha": "2025-01-20", "Ticker": "AAPL", "Tipo": "Compra", "Cantidad": 9.0,   "Precio_USD": 229.00, "Comision_USD": 0.50, "TC_MXN": 20.15},
            {"Fecha": "2025-01-20", "Ticker": "CAT",  "Tipo": "Compra", "Cantidad": 3.0,   "Precio_USD": 350.00, "Comision_USD": 0.50, "TC_MXN": 20.15},
            {"Fecha": "2025-01-20", "Ticker": "NVDA", "Tipo": "Compra", "Cantidad": 12.0,  "Precio_USD": 136.00, "Comision_USD": 0.50, "TC_MXN": 20.15},
            {"Fecha": "2025-01-20", "Ticker": "NEM",  "Tipo": "Compra", "Cantidad": 50.0,  "Precio_USD": 41.00,  "Comision_USD": 0.50, "TC_MXN": 20.15},
            {"Fecha": "2025-06-01", "Ticker": "NVDA", "Tipo": "Compra", "Cantidad": 4.0,   "Precio_USD": 128.00, "Comision_USD": 0.50, "TC_MXN": 19.70},
        ],
    }
    return pd.DataFrame(ops_map.get(portfolio_name, []))


def get_demo_prices() -> dict:
    """Precios de referencia (fallback sin Yahoo Finance conectado)."""
    return {
        # Portafolio Conservador (SHY, IEI, VCSH + SPX)
        "SHY":    83.50,    # iShares 1-3yr Treasury Bond ETF
        "IEI":    116.20,   # iShares 3-7yr Treasury Bond ETF
        "VCSH":   76.40,    # Vanguard Short-Term Corporate Bond ETF
        # Portafolio Moderado (GLD, SHY, QQQ + SPX)
        "QQQ":    480.00,   # Invesco QQQ Trust (Nasdaq 100)
        # Portafolio Balanceado (GLD, IAU, WMT + SPX)
        "GLD":    230.00,   # SPDR Gold Shares
        "IAU":    46.00,    # iShares Gold Trust
        "WMT":    95.00,    # Walmart Inc.
        # Portafolio Alto Rendimiento (AAPL, CAT, NVDA, NEM)
        "AAPL":   228.30,
        "CAT":    345.20,
        "NVDA":   138.50,
        "NEM":    44.10,
        # Benchmark
        "SPX":    5720.00,
        # Otros activos posibles
        "BATS":   36.80, "TSLA": 395.00, "ASTS": 18.50,
        "NOW":    985.00, "JPM": 238.00, "9988HK": 95.00,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ─── LAYER 6: EMAIL + WRITE USERS ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def send_alert_email(user_name, user_email, portfolio, alerts_detail):
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        cfg = st.secrets.get("email", {}) if hasattr(st, "secrets") else {}
        if not cfg or not user_email:
            return False
        n_buy = sum(1 for r in alerts_detail if r.get("Alerta") == "COMPRAR")
        n_sell = sum(1 for r in alerts_detail if r.get("Alerta") == "VENDER")
        subject = f"⚠ Alerta {portfolio} — {n_buy} compras · {n_sell} ventas"
        rows_html = "".join(
            f"<tr><td style='padding:8px 12px;border-bottom:1px solid #f0f4f9;font-weight:600;'>"
            f"{r.get('Ticker','')}</td>"
            f"<td style='padding:8px 12px;border-bottom:1px solid #f0f4f9;color:"
            f"{'#1a7a4a' if r.get('Alerta')=='COMPRAR' else '#c0392b'};font-weight:700;'>"
            f"{r.get('Alerta','')}</td>"
            f"<td style='padding:8px 12px;border-bottom:1px solid #f0f4f9;'>"
            f"{r.get('Peso Actual',0)*100:.1f}%</td>"
            f"<td style='padding:8px 12px;border-bottom:1px solid #f0f4f9;'>"
            f"{r.get('Peso Objetivo',0)*100:.1f}%</td></tr>"
            for r in alerts_detail
        )
        html = f"""<html><body style="font-family:Arial,sans-serif;background:#edf2f8;">
        <div style="max-width:560px;margin:32px auto;background:white;border-radius:16px;overflow:hidden;">
          <div style="background:linear-gradient(135deg,#0d2240,#1a3a5c);padding:28px 32px;">
            <div style="font-size:22px;color:white;font-weight:700;">EGADA Portfolio Manager</div>
          </div>
          <div style="padding:28px 32px;">
            <div style="background:#f9d100;border-radius:10px;padding:14px 20px;margin-bottom:18px;">
              <div style="font-size:17px;font-weight:700;color:#1a1400;">⚠ Tu portafolio necesita ajuste</div>
              <div style="font-size:12px;color:#5c4d00;">{portfolio}</div>
            </div>
            <p>Hola <b>{user_name}</b>,</p>
            <table style="width:100%;border-collapse:collapse;font-size:13px;">
              <thead><tr style="background:#1a3a5c;color:white;">
                <th style="padding:10px;">Activo</th><th>Acción</th><th>Actual</th><th>Objetivo</th>
              </tr></thead><tbody>{rows_html}</tbody>
            </table>
            <p style="color:#6b7c93;font-size:11px;margin-top:18px;">
              {datetime.now().strftime("%d %b %Y %H:%M")} · EGADA Portfolio Manager</p>
          </div></div></body></html>"""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"EGADA Portfolio <{cfg['smtp_user']}>"
        msg["To"] = user_email
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP(cfg["smtp_host"], int(cfg.get("smtp_port", 587))) as srv:
            srv.starttls()
            srv.login(cfg["smtp_user"], cfg["smtp_pass"])
            srv.send_message(msg)
        return True
    except Exception:
        return False


def write_users_to_file(users_dict):
    import os, re
    app_path = os.path.abspath(__file__)
    with open(app_path, "r", encoding="utf-8") as f:
        src = f.read()
    nl = "\n"
    parts = ["USERS = {  # USERS_START" + nl]
    for key, ud in users_dict.items():
        ports = ud.get("portfolios")
        ports_repr = "None" if ports is None else repr(ports)
        parts.append("    " + repr(key) + ": {" + nl)
        parts.append('        "name":       ' + repr(ud.get("name", "")) + "," + nl)
        parts.append('        "pin":        ' + repr(ud.get("pin", "0000")) + "," + nl)
        parts.append('        "avatar":     ' + repr(ud.get("avatar", "??")) + "," + nl)
        parts.append('        "color":      ' + repr(ud.get("color", "#1a3a5c")) + "," + nl)
        parts.append('        "sheet_id":   "",' + nl)
        parts.append('        "portfolios": ' + ports_repr + "," + nl)
        parts.append('        "role":       ' + repr(ud.get("role", "investor")) + "," + nl)
        parts.append('        "email":      ' + repr(ud.get("email", "")) + "," + nl)
        parts.append("    }," + nl)
    parts.append("}  # USERS_END" + nl)
    new_block = "".join(parts)
    pattern = r"USERS = \{  # USERS_START\n.*?\}  # USERS_END\n"
    if re.search(pattern, src, re.DOTALL):
        new_src = re.sub(pattern, new_block, src, flags=re.DOTALL)
        with open(app_path, "w", encoding="utf-8") as f:
            f.write(new_src)
        return True, f"{len(users_dict)} usuarios guardados en app.py"
    return False, "Marcadores no encontrados"


# ══════════════════════════════════════════════════════════════════════════════
# ─── LAYER 7: USERS CONFIG ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

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
    "ana": {
        "name":       "Ana García",
        "pin":        "2222",
        "avatar":     "AG",
        "color":      "#1a7a4a",
        "sheet_id":   "",
        "portfolios": ["Conservador", "Moderado"],
        "role":       "investor",
        "email":      "",
    },
    "carlos": {
        "name":       "Carlos López",
        "pin":        "3333",
        "avatar":     "CL",
        "color":      "#c0392b",
        "sheet_id":   "",
        "portfolios": ["AltoRendimiento"],
        "role":       "investor",
        "email":      "",
    },
}  # USERS_END


# ══════════════════════════════════════════════════════════════════════════════
# ─── LAYER 8: PDF REPORT (Admin download) ─────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def build_pdf_report(port_name: str, model: dict, positions_df: pd.DataFrame,
                     alerts_df: pd.DataFrame, twr_data: dict, history_df: pd.DataFrame) -> bytes:
    """
    Genera PDF con todos los gráficos del modelo explicados detalladamente.
    Usa reportlab si disponible, else HTML convertible.
    """
    try:
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter,
                                rightMargin=0.75*inch, leftMargin=0.75*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle("Title2", parent=styles["Title"],
                                     fontSize=18, spaceAfter=6)
        sub_style = ParagraphStyle("Sub", parent=styles["Normal"],
                                   fontSize=10, textColor=colors.HexColor("#6b7c93"),
                                   spaceAfter=12)
        h2_style = ParagraphStyle("H2", parent=styles["Heading2"],
                                  fontSize=13, textColor=colors.HexColor("#1a3a5c"),
                                  spaceBefore=14, spaceAfter=6)
        body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                    fontSize=10, leading=14, spaceAfter=8)

        story.append(Paragraph(f"EGADA — Portfolio Manager", title_style))
        story.append(Paragraph(f"Reporte Completo: {port_name}", title_style))
        story.append(Paragraph(
            f"Generado: {datetime.now().strftime('%d %b %Y %H:%M')} · "
            f"Estrategia Global de Activos Dinámicos Avanzados",
            sub_style))
        story.append(Spacer(1, 0.2*inch))

        # G1: Descripción del modelo
        story.append(Paragraph("G1 — Modelo Core: Markowitz con Restricción 30%", h2_style))
        story.append(Paragraph(
            f"El modelo sigue exactamente la metodología del código madre MATLAB. "
            f"Se calculan rendimientos logarítmicos mensuales, se construye la frontera eficiente "
            f"con 20 puntos óptimos y se maximiza el Sharpe Ratio con un límite máximo de "
            f"{model.get('max_weight', 0.3)*100:.0f}% por activo. "
            f"Tasa libre de riesgo: {model.get('rf', 0.0264)*100:.2f}%.", body_style))

        # G2: Portafolio de Tangencia
        story.append(Paragraph("G2 — Portafolio de Tangencia (Máximo Sharpe)", h2_style))
        w_tang = model.get("w_tang", {})
        if w_tang:
            ret_t = model.get("ret_tang", 0) * 100
            vol_t = model.get("vol_tang", 0) * 100
            sr_t = model.get("sharpe_tang", 0)
            story.append(Paragraph(
                f"Rendimiento anual esperado: {ret_t:.2f}% | "
                f"Riesgo (volatilidad): {vol_t:.2f}% | Sharpe Ratio: {sr_t:.2f}",
                body_style))
            comp_data = [["Activo", "Peso Óptimo", "Descripción"]] + [
                [t, f"{w*100:.1f}%", ""] for t, w in w_tang.items() if w > 0.01
            ]
            t_comp = Table(comp_data, colWidths=[1.5*inch, 1.2*inch, 4*inch])
            t_comp.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e8f0")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(t_comp)

        # G3: Mínima Varianza
        story.append(Paragraph("G3 — Portafolio de Mínima Varianza", h2_style))
        w_mv = model.get("w_mv", {})
        if w_mv:
            story.append(Paragraph(
                f"Retorno: {model.get('ret_mv',0)*100:.2f}% | "
                f"Vol: {model.get('vol_mv',0)*100:.2f}% — "
                f"Punto más a la izquierda de la frontera eficiente.", body_style))

        # G4: Intervalos de Confianza MC
        story.append(Paragraph("G4 — Intervalos de Confianza (Monte Carlo 10,000 sim.)", h2_style))
        story.append(Paragraph(
            f"Rendimiento esperado 1 año: {model.get('mc_p50',0)*100:.2f}% (mediana) | "
            f"Intervalo 90%: [{model.get('mc_p5',0)*100:.1f}%, {model.get('mc_p95',0)*100:.1f}%] | "
            f"Probabilidad de pérdida: {model.get('mc_prob_loss',0)*100:.1f}%", body_style))

        # G5: VaR
        story.append(Paragraph("G5 — Value at Risk (VaR) y CVaR al 95%", h2_style))
        story.append(Paragraph(
            f"VaR mensual 95%: {abs(model.get('var_95',0))*100:.2f}% — "
            f"En el peor 5% de los meses, se espera perder al menos este porcentaje. "
            f"CVaR 95%: {abs(model.get('cvar_95',0))*100:.2f}% — "
            f"Pérdida promedio en escenarios extremos.", body_style))

        # G6: Horizontes
        story.append(Paragraph("G6 — Rendimiento Proyectado por Horizonte", h2_style))
        hz = model.get("horizon_intervals", {})
        if hz:
            hz_data = [["Horizonte", "Mínimo (P5)", "Esperado", "Máximo (P95)"]] + [
                [f"{yr} año{'s' if yr > 1 else ''}",
                 f"{d['p5']:.1f}%", f"{d['mean']:.1f}%", f"{d['p95']:.1f}%"]
                for yr, d in hz.items()
            ]
            t_hz = Table(hz_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            t_hz.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e8f0")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(t_hz)

        # G7: Posiciones actuales
        story.append(Paragraph("G7 — Posiciones Actuales del Usuario", h2_style))
        if not positions_df.empty:
            pos_data = [["Ticker", "Cantidad", "Precio", "Valor USD", "Ganancia"]] + [
                [str(r["Ticker"]), f"{r['Cantidad']:.4f}", f"${r['Precio Actual']:,.2f}",
                 f"${r['Valor USD']:,.2f}", f"{'+' if r['Ganancia USD']>=0 else ''}{r['Ganancia USD']:,.2f}"]
                for _, r in positions_df.iterrows()
            ]
            t_pos = Table(pos_data, colWidths=[1.2*inch, 1.2*inch, 1.2*inch, 1.4*inch, 1.4*inch])
            t_pos.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e8f0")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(t_pos)

        # G8: TWR
        story.append(Paragraph("G8 — Rendimiento Real del Usuario (TWR)", h2_style))
        if twr_data:
            story.append(Paragraph(
                f"Fecha primera compra: {twr_data.get('first_date','N/A')} | "
                f"Capital invertido: ${twr_data.get('net_invested_usd',0):,.2f} USD | "
                f"Valor actual: ${twr_data.get('valor_actual_usd',0):,.2f} USD | "
                f"TWR: {twr_data.get('twr_pct',0):+.2f}%", body_style))

        # G9: SML
        story.append(Paragraph("G9 — Security Market Line (SML) y Betas", h2_style))
        betas = model.get("betas", {})
        alphas = model.get("alphas_capm", {})
        mu_ann = model.get("mu_ann", {})
        if betas:
            sml_data = [["Activo", "Beta", "Alpha CAPM", "Retorno Esperado"]] + [
                [t, f"{betas.get(t,0):.2f}", f"{alphas.get(t,0)*100:+.2f}%",
                 f"{mu_ann.get(t,0)*100:.2f}%"]
                for t in betas
            ]
            t_sml = Table(sml_data, colWidths=[1.5*inch, 1.2*inch, 1.5*inch, 2*inch])
            t_sml.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e8f0")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(t_sml)
            story.append(Paragraph(
                "Beta > 1: más volátil que el mercado. Beta < 1: más defensivo. "
                "Alpha positivo = activo genera retorno por encima de lo esperado por CAPM.",
                body_style))

        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(
            f"Reporte generado por EGADA Portfolio Manager · "
            f"{datetime.now().strftime('%d %b %Y %H:%M')} · "
            f"Modelo Core: Markowitz max Sharpe con restricción 30%",
            sub_style))

        doc.build(story)
        buf.seek(0)
        return buf.read()
    except Exception:
        # Fallback: CSV
        buf = io.BytesIO()
        if not positions_df.empty:
            buf.write(positions_df.to_csv(index=False).encode())
        else:
            buf.write(b"No data available")
        buf.seek(0)
        return buf.read()


# ══════════════════════════════════════════════════════════════════════════════
# ─── LAYER 9: STREAMLIT APP ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="EGADA Portfolio Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Global ────────────────────────────────────────────────────────────────
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

#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stDeployButton"],[class*="viewerBadge"],[class*="StatusWidget"],
button[title="View fullscreen"],a[href*="streamlit.io"],a[href*="github.com"],
.stActionButton,[data-testid="stBaseButton-headerNoPadding"],
[data-testid="stAppDeployButton"]{display:none !important;}

.stApp,[data-testid="stAppViewContainer"]{
    background:linear-gradient(135deg,#f0f4f9 0%,#e8f0f8 50%,#eef3f8 100%) !important;
}
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0d1b2a 0%,#1a2d45 100%) !important;
    border-right:1px solid rgba(201,162,39,0.2) !important;
}
.main .block-container{padding:1.5rem 2rem 2rem;}
@media(max-width:768px){.main .block-container{padding:1rem;}}

/* KPI Cards */
.kpi-card{
    background:linear-gradient(135deg,rgba(255,255,255,0.88),rgba(232,240,248,0.7));
    border-radius:16px;padding:18px 20px;
    border:1px solid rgba(26,58,92,0.1);
    box-shadow:0 4px 16px rgba(13,27,42,0.08),0 1px 0 rgba(255,255,255,0.9) inset;
    position:relative;overflow:hidden;
    transition:transform 0.25s ease,box-shadow 0.25s ease;
}
.kpi-card:hover{transform:translateY(-3px);box-shadow:0 10px 32px rgba(13,27,42,0.13);}
.kpi-label{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:6px;}
.kpi-value{font-family:'DM Serif Display',serif;font-size:28px;color:var(--navy2);line-height:1.1;margin-bottom:4px;}
.kpi-sub{font-size:11px;color:var(--muted);}
.positive{color:var(--green);}
.negative{color:var(--red);}

/* Section titles */
.section-title{
    font-family:'DM Serif Display',serif;font-size:22px;
    color:var(--navy);margin:24px 0 14px;
    padding-bottom:8px;border-bottom:2px solid rgba(201,162,39,0.3);
}

/* Banner */
.app-banner{
    background:linear-gradient(135deg,#0d1b2a 0%,#1a3a5c 60%,#0d2240 100%);
    border-radius:20px;padding:24px 32px;margin-bottom:24px;
    display:flex;justify-content:space-between;align-items:flex-start;
    box-shadow:0 8px 32px rgba(13,27,42,0.25);
}
.banner-title{font-family:'DM Serif Display',serif;font-size:30px;color:#fff;margin:0;}
.banner-sub{color:var(--gold2);font-size:13px;margin-top:4px;}
.banner-date{color:rgba(232,240,248,0.45);font-size:11px;margin-top:4px;}
.banner-greeting{font-size:16px;color:rgba(255,255,255,0.82);font-weight:500;}
.banner-count{font-family:'DM Serif Display',serif;font-size:36px;color:#fff;line-height:1;}
.banner-count-label{font-size:11px;color:rgba(232,240,248,0.6);}

/* Chart wrapper */
.chart3d{
    background:linear-gradient(145deg,rgba(255,255,255,0.78),rgba(232,240,248,0.6));
    backdrop-filter:blur(16px);border-radius:18px;
    border:1px solid rgba(26,58,92,0.12);
    padding:6px;margin-bottom:14px;
    box-shadow:0 8px 32px rgba(13,27,42,0.09),0 1px 0 rgba(255,255,255,0.85) inset;
    transition:transform 0.3s ease,box-shadow 0.3s ease;
}
.chart3d:hover{transform:translateY(-4px);box-shadow:0 16px 48px rgba(13,27,42,0.14);}

/* Tables */
.table-wrap{
    width:100%;overflow-x:auto;border-radius:14px;
    background:rgba(255,255,255,0.75);
    border:1px solid rgba(26,58,92,0.1);
    box-shadow:0 4px 16px rgba(13,27,42,0.07);
    padding:2px;
}
.styled-table{width:100%;min-width:580px;border-collapse:separate;border-spacing:0;font-size:13px;}
.styled-table th{
    background:linear-gradient(135deg,#1a3a5c,#0d2240);
    color:#e8f0f8;padding:11px 14px;text-align:center;
    font-weight:700;font-size:10px;letter-spacing:.08em;text-transform:uppercase;
    white-space:nowrap;border-bottom:2px solid rgba(201,162,39,0.4);
}
.styled-table th:first-child{border-radius:12px 0 0 0;}
.styled-table th:last-child{border-radius:0 12px 0 0;}
.styled-table td{
    padding:10px 14px;text-align:center;
    border-bottom:1px solid rgba(26,58,92,0.06);
    background:rgba(255,255,255,0.5);white-space:nowrap;
}
.styled-table tr:last-child td{border-bottom:none;}
.styled-table tr:hover td{background:rgba(26,58,92,0.05);}
.styled-table tr:nth-child(even) td{background:#f8fafd;}

/* Badges */
.badge-ok{background:#d1fae5;color:#065f46;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;}
.badge-warn{background:#fef3c7;color:#92400e;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;}
.badge-danger{background:#fee2e2;color:#991b1b;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;}

/* Order cards */
.order-card{border-radius:12px;padding:16px 20px;margin:8px 0;display:flex;align-items:center;gap:16px;}
.order-buy{background:linear-gradient(135deg,#d5f5e3,#e8f8f0);border-left:4px solid #059669;}
.order-sell{background:linear-gradient(135deg,#fadbd8,#fde8e6);border-left:4px solid #ef4444;}

/* Comparison card */
.compare-card{
    background:linear-gradient(135deg,rgba(255,255,255,0.9),rgba(240,244,249,0.8));
    border-radius:14px;padding:18px 20px;
    border:1px solid rgba(26,58,92,0.1);margin-bottom:10px;
}

/* Login */
.gpanel{
    background:#ffffff;border-radius:22px;
    box-shadow:0 4px 20px rgba(0,0,0,0.10),0 1px 4px rgba(0,0,0,0.06);
    padding:32px 36px;position:relative;overflow:visible;
    border-top:4px solid #c9a227;
}
.user-pill{
    display:flex;align-items:center;gap:12px;padding:14px 16px;
    border-radius:12px;border:2px solid var(--ice);margin-bottom:10px;
    background:var(--fog);transition:border .2s,background .2s,transform .15s;
}
.user-pill:hover{border-color:var(--gold);background:white;transform:translateX(3px);}

/* Animations */
@keyframes fadeInUp{from{opacity:0;transform:translateY(16px);}to{opacity:1;transform:translateY(0);}}
@keyframes countUp{from{opacity:0;transform:translateY(8px) scale(0.85);}to{opacity:1;transform:translateY(0) scale(1);}}
.kpi-value,.banner-count{animation:countUp 0.6s cubic-bezier(.22,.68,0,1.25) both;}
.anim-fadeinup{animation:fadeInUp .5s cubic-bezier(.22,.68,0,1.2) both;}
.page-wrap{animation:fadeInUp .4s ease both;}

/* Plotly charts */
div[data-testid="stPlotlyChart"]{
    filter:drop-shadow(0 4px 12px rgba(13,27,42,0.1)) !important;
    border-radius:14px !important;transition:filter 0.3s,transform 0.3s !important;
}
div[data-testid="stPlotlyChart"]:hover{
    filter:drop-shadow(0 14px 36px rgba(13,27,42,0.17)) !important;
    transform:translateY(-4px) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────
for _k, _v in [("authenticated", False), ("current_user", None),
               ("selected_user_login", None), ("login_error", False),
               ("custom_portfolios", {})]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════════
# ── LOGIN SCREEN ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    st.markdown("""
    <style>
    .stApp,[data-testid="stAppViewContainer"]{
        background:radial-gradient(ellipse at 30% 40%,#112240 0%,#0a1622 45%,#050810 100%) !important;
    }
    @keyframes floatOrb{0%,100%{transform:translateY(0) scale(1);opacity:.22;}50%{transform:translateY(-28px) scale(1.06);opacity:.30;}}
    .orb{position:fixed;border-radius:50%;pointer-events:none;z-index:0;animation:floatOrb 10s ease-in-out infinite;}
    .orb1{width:560px;height:560px;left:-140px;top:-140px;background:radial-gradient(circle,rgba(37,99,235,.28) 0%,transparent 68%);animation-delay:0s;}
    .orb2{width:400px;height:400px;right:-100px;bottom:0;background:radial-gradient(circle,rgba(201,162,39,.2) 0%,transparent 65%);animation-delay:4s;}
    .orb3{width:250px;height:250px;left:38%;top:52%;background:radial-gradient(circle,rgba(139,92,246,.16) 0%,transparent 65%);animation-delay:7s;}
    [data-testid="stAppViewContainer"] [data-testid="stButton"] > div > button{
        background:#f4f7fb !important;border:1.5px solid #e2e8f0 !important;
        border-radius:14px !important;color:#1a2332 !important;font-size:14px !important;
        font-weight:600 !important;text-align:left !important;min-height:68px !important;
        padding:14px 20px !important;margin-bottom:10px !important;white-space:pre-line !important;
        line-height:1.5 !important;transition:all .2s !important;
        box-shadow:0 2px 8px rgba(13,27,42,0.06) !important;
    }
    [data-testid="stAppViewContainer"] [data-testid="stButton"] > div > button:hover{
        background:#fff !important;border-color:#c9a227 !important;
        transform:translateX(5px) !important;
        box-shadow:0 4px 20px rgba(201,162,39,.25),-3px 0 0 #c9a227 !important;
    }
    [data-testid="stAppViewContainer"] .stTextInput input{
        background:#f4f7fb !important;border:1.5px solid #d1dae6 !important;
        border-radius:12px !important;color:#1a2332 !important;
        font-size:24px !important;letter-spacing:14px !important;
        text-align:center !important;padding:16px !important;
    }
    [data-testid="stAppViewContainer"] .stTextInput input:focus{
        border-color:#c9a227 !important;box-shadow:0 0 0 3px rgba(201,162,39,.15) !important;
    }
    [data-testid="stAppViewContainer"] .stTextInput label{display:none !important;}
    </style>
    <div class="orb orb1"></div><div class="orb orb2"></div><div class="orb orb3"></div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.3, 1])
    with col:
        # EGADA Logo SVG (nombre completo)
        egada_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 90" width="320" height="90">
  <rect x="0" y="0" width="7" height="90" fill="#c8102e" rx="2"/>
  <text x="20" y="58" font-family="Georgia,serif" font-size="40" font-weight="700"
        letter-spacing="3" fill="#1a2332">EGADA</text>
  <text x="21" y="74" font-family="Arial,sans-serif" font-size="9.5" font-weight="400"
        letter-spacing="2.5" fill="#6b7c93">ESTRATEGIA GLOBAL DE ACTIVOS</text>
  <text x="21" y="22" font-family="Arial,sans-serif" font-size="9" font-weight="400"
        letter-spacing="1.5" fill="#c8102e">DINÁMICOS AVANZADOS</text>
</svg>"""
        import base64
        egada_b64 = base64.b64encode(egada_svg.encode()).decode()

        st.markdown(
            f'<div style="text-align:center;padding:48px 0 24px;position:relative;z-index:1;">'
            f'<div style="display:inline-block;background:rgba(255,255,255,0.97);'
            f'border-radius:18px;padding:20px 36px;box-shadow:0 8px 32px rgba(0,0,0,.18);">'
            f'<img src="data:image/svg+xml;base64,{egada_b64}" style="width:240px;display:block;">'
            f'</div>'
            f'<div style="margin-top:14px;font-size:11px;color:rgba(201,162,39,.8);'
            f'letter-spacing:.18em;font-weight:700;text-transform:uppercase;">'
            f'Portfolio Manager · v2.0</div></div>',
            unsafe_allow_html=True
        )

        if st.session_state.selected_user_login is None:
            st.markdown("""
            <div class="gpanel">
                <div style="font-family:'DM Serif Display',serif;font-size:24px;color:#0d1b2a;margin-bottom:4px;">Bienvenido</div>
                <div style="font-size:12px;color:#6b7c93;margin-bottom:24px;">Selecciona tu perfil para continuar</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            for ukey, udata in USERS.items():
                port_list = get_user_portfolios(ukey)
                port_str = "  ·  ".join(port_list[:4])
                label = f"{udata['avatar']}  {udata['name']}\n{port_str}"
                if st.button(label, key=f"user_btn_{ukey}", use_container_width=True):
                    st.session_state.selected_user_login = ukey
                    st.session_state.login_error = False
                    st.rerun()
        else:
            ukey = st.session_state.selected_user_login
            udata = USERS[ukey]
            port_list = get_user_portfolios(ukey)
            uc, ua, un = udata["color"], udata["avatar"], udata["name"]
            err_html = (
                "<div style='margin-top:8px;background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.3);"
                "border-radius:10px;padding:9px 14px;color:#c0392b;font-size:12px;text-align:center;'>"
                "⚠ PIN incorrecto — intenta de nuevo</div>"
            ) if st.session_state.get("login_error") else ""

            st.markdown(
                f"<div class='gpanel' style='padding-bottom:24px;'>"
                f"<div style='display:flex;align-items:center;gap:14px;"
                f"padding-bottom:16px;margin-bottom:16px;border-bottom:1px solid #e8eef5;'>"
                f"<div style='width:52px;height:52px;border-radius:50%;background:{uc};"
                f"display:flex;align-items:center;justify-content:center;"
                f"font-size:22px;flex-shrink:0;color:white;font-weight:700;'>{ua}</div>"
                f"<div><div style='font-weight:700;font-size:18px;color:#0d1b2a;'>{un}</div>"
                f"<div style='font-size:11px;color:#6b7c93;'>{' · '.join(port_list[:3])}</div>"
                f"</div></div>"
                f"<div style='font-size:10px;font-weight:700;color:#c9a227;letter-spacing:.14em;text-transform:uppercase;margin-bottom:10px;'>PIN de acceso</div>"
                f"{err_html}</div>",
                unsafe_allow_html=True
            )

            pin_input = st.text_input("PIN", type="password", max_chars=4,
                                      placeholder="● ● ● ●", label_visibility="collapsed",
                                      key="pin_field")
            c1, c2 = st.columns([1, 1.4])
            with c1:
                if st.button("← Volver", use_container_width=True, key="back_btn"):
                    st.session_state.selected_user_login = None
                    st.session_state.login_error = False
                    st.rerun()
            with c2:
                enter = st.button("Entrar →", use_container_width=True,
                                  type="primary", key="enter_btn")
            if enter or (pin_input and len(pin_input) == 4):
                if pin_input == udata["pin"]:
                    st.session_state.authenticated = True
                    st.session_state.current_user = ukey
                    st.session_state.login_error = False
                    st.rerun()
                elif enter:
                    st.session_state.login_error = True
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# ── AUTHENTICATED SESSION ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
current_user = st.session_state.current_user
user_data = USERS[current_user]
is_admin = user_data.get("role") == "admin"
user_ports = get_user_portfolios(current_user)
all_portfolios = get_all_portfolios()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    import base64 as _b64
    _egada_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 80" width="300" height="80">
  <rect x="0" y="0" width="6" height="80" fill="#c8102e" rx="2"/>
  <text x="16" y="52" font-family="Georgia,serif" font-size="36" font-weight="700"
        letter-spacing="3" fill="white">EGADA</text>
  <text x="17" y="68" font-family="Arial,sans-serif" font-size="9" font-weight="400"
        letter-spacing="2" fill="rgba(255,255,255,0.55)">ESTRATEGIA GLOBAL DE ACTIVOS</text>
  <text x="17" y="19" font-family="Arial,sans-serif" font-size="8.5" font-weight="400"
        letter-spacing="1.5" fill="#c8102e">DINÁMICOS AVANZADOS</text>
</svg>"""
    _egada_b64 = _b64.b64encode(_egada_svg.encode()).decode()
    st.markdown(
        f"<div style='padding:16px 14px 12px;border-bottom:1px solid rgba(201,162,39,0.2);margin-bottom:14px;'>"
        f"<img src='data:image/svg+xml;base64,{_egada_b64}' style='width:190px;display:block;'>"
        f"</div>",
        unsafe_allow_html=True
    )

    # TC en tiempo real
    _tc_live = 20.50
    if YF_AVAILABLE:
        try:
            _tc_live_raw = getattr(yf.Ticker("MXN=X").fast_info, "last_price", None)
            if _tc_live_raw and 15 < float(_tc_live_raw) < 35:
                _tc_live = round(float(_tc_live_raw), 2)
        except Exception:
            pass

    # User card
    st.markdown(f"""
    <div style='padding:10px 0 8px;'>
        <div style='font-family:"DM Serif Display",serif;font-size:18px;color:#e8f0f8;margin-bottom:2px;'>
            Portfolio <span style='color:#c9a227;'>Manager</span>
        </div>
        <div style='font-size:9px;color:#6b7c93;margin-bottom:10px;'>Sistema de Gestión v2.0</div>
        <div style='background:rgba(255,255,255,0.07);border-radius:10px;padding:10px 12px;
                    display:flex;align-items:center;gap:10px;'>
            <div style='width:32px;height:32px;border-radius:50%;background:{user_data["color"]};
                        display:flex;align-items:center;justify-content:center;
                        font-weight:700;font-size:13px;color:white;flex-shrink:0;'>{user_data["avatar"]}</div>
            <div>
                <div style='font-size:13px;font-weight:600;color:#e8f0f8;'>{user_data["name"]}</div>
                <div style='font-size:10px;color:#6b7c93;'>{user_data["role"].capitalize()}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # TC Card
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,rgba(13,27,42,0.95),rgba(26,58,92,0.9));
                border:1px solid rgba(201,162,39,0.3);border-radius:12px;
                padding:12px 14px;margin:8px 0 12px;'>
        <div style='font-size:9px;color:#c9a227;font-weight:700;letter-spacing:.15em;
                    text-transform:uppercase;margin-bottom:4px;'>USD / MXN</div>
        <div style='display:flex;align-items:baseline;gap:6px;'>
            <span style='font-size:26px;font-weight:800;color:#fff;
                font-family:"DM Serif Display",serif;line-height:1;'>{_tc_live:.2f}</span>
            <span style='font-size:9px;color:rgba(201,162,39,0.7);'>● Yahoo Finance</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⬡ Cerrar sesión", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.selected_user_login = None
        st.rerun()

    st.divider()

    # Sheet config
    _own_sheet = user_data.get("sheet_id", "").strip()
    _admin_sheet = next(
        (u["sheet_id"] for u in USERS.values()
         if u.get("role") == "admin" and u.get("sheet_id", "").strip()),
        ""
    )
    sheet_id = _own_sheet if _own_sheet else _admin_sheet
    use_sheets = bool(sheet_id) and get_gspread_client() is not None
    if not use_sheets and sheet_id:
        st.warning("⚠ Credenciales GCP no configuradas")

    # Portfolio selector
    port_names = [p for p in list(all_portfolios.keys()) if p in user_ports]
    st.markdown("<div style='font-size:10px;font-weight:700;letter-spacing:.1em;color:#c9a227;"
                "text-transform:uppercase;margin-bottom:8px;'>Portafolio Activo</div>",
                unsafe_allow_html=True)

    if "selected_port" not in st.session_state or st.session_state.selected_port not in port_names:
        st.session_state.selected_port = port_names[0] if port_names else ""

    # Invisible button overlay trick
    st.markdown("""
    <style>
    [data-testid="stSidebar"] [data-testid="stButton"] button{
        position:relative !important;margin-top:-44px !important;height:44px !important;
        background:transparent !important;border:none !important;box-shadow:none !important;
        color:transparent !important;font-size:0px !important;cursor:pointer !important;
        z-index:999 !important;width:100% !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover,
    [data-testid="stSidebar"] [data-testid="stButton"] button:focus{
        background:transparent !important;border:none !important;box-shadow:none !important;
    }
    </style>""", unsafe_allow_html=True)

    for pn in port_names:
        pdata = all_portfolios.get(pn, {})
        pcolor = pdata.get("color", "#1a3a5c")
        pdesc = pdata.get("description", "")
        is_sel = st.session_state.selected_port == pn
        bg = pcolor if is_sel else "rgba(255,255,255,0.07)"
        glow = f"0 0 0 2px {pcolor},0 4px 16px {pcolor}55" if is_sel else f"0 0 0 1px {pcolor}77"
        fw = "700" if is_sel else "400"
        st.markdown(
            f"<div style='background:{bg};border-radius:10px;padding:10px 12px;"
            f"margin-bottom:2px;box-shadow:{glow};pointer-events:none;'>"
            f"<div style='font-size:13px;font-weight:{fw};color:white;'>{pn}</div>"
            f"<div style='font-size:10px;color:rgba(255,255,255,0.55);margin-top:1px;'>{pdesc}</div>"
            f"</div>", unsafe_allow_html=True
        )
        if st.button("​", key=f"portbtn_{pn}", use_container_width=True):
            st.session_state.selected_port = pn
            st.rerun()

    selected_port = st.session_state.selected_port
    if not selected_port or selected_port not in all_portfolios:
        st.stop()

    port_data = all_portfolios[selected_port]
    port_color = port_data.get("color", "#1a3a5c")

    st.divider()

    # Admin params
    if is_admin:
        st.markdown("<div style='font-size:10px;font-weight:700;letter-spacing:.1em;color:#c9a227;"
                    "text-transform:uppercase;margin-bottom:8px;'>Parámetros</div>",
                    unsafe_allow_html=True)
        tc_actual = st.number_input("TC MXN/USD", value=_tc_live, step=0.01, format="%.2f")
        monthly_add = st.number_input("Aportación mensual (MXN)", value=2000.0, step=100.0)
        thresh_rebal = st.slider("Umbral rebalanceo", 0.03, 0.15, 0.05, 0.01, format="%.0f%%")
    else:
        tc_actual = _tc_live
        monthly_add = 2000.0
        thresh_rebal = 0.05

    st.divider()
    # Navigation
    st.markdown("<div style='font-size:10px;font-weight:700;letter-spacing:.1em;color:#c9a227;"
                "text-transform:uppercase;margin-bottom:8px;'>Navegación</div>",
                unsafe_allow_html=True)
    if is_admin:
        _pages = ["📊 Dashboard", "📋 Operaciones", "⚡ Análisis de Riesgo",
                  "🔀 Modelo vs Realidad", "👥 Usuarios", "🔭 QuickView"]
    else:
        _pages = ["📊 Mi Portafolio", "📋 Registrar Operación",
                  "📈 Cómo va mi inversión", "🔀 Modelo vs Realidad"]
    page = st.radio("Nav", _pages, label_visibility="collapsed")


# ══════════════════════════════════════════════════════════════════════════════
# ── LOAD DATA (per-portfolio, NEVER cross-contaminated) ───────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# 1. Operaciones: solo Ops_<selected_port>
if use_sheets and sheet_id:
    _raw_ops = load_sheet(sheet_id, f"Ops_{selected_port}")
    if not _raw_ops.empty:
        if "Usuario" in _raw_ops.columns:
            _uname_full = user_data.get("name", current_user).lower()
            _ucol = _raw_ops["Usuario"].astype(str).str.strip().str.lower()
            _mask = ((_ucol == current_user.lower()) | (_ucol == _uname_full) |
                     (_ucol == _uname_full.split()[0]))
            _f = _raw_ops[_mask].copy()
            ops_df = _f if not _f.empty else _raw_ops.copy()
        else:
            ops_df = _raw_ops.copy()
        ops_source = "Google Sheets"
    else:
        ops_df = get_demo_ops(selected_port)
        ops_source = "Demo"
else:
    ops_df = get_demo_ops(selected_port)
    ops_source = "Demo"

if not ops_df.empty:
    for _col in ["Cantidad", "Precio_USD", "Comision_USD", "TC_MXN"]:
        if _col in ops_df.columns:
            ops_df[_col] = pd.to_numeric(ops_df[_col], errors="coerce").fillna(0)

# 2. Precios históricos: solo Precios_<selected_port>
# El universo (columnas) lo define el Excel — no el código.
prices_hist_df = pd.DataFrame()
if use_sheets and sheet_id:
    prices_hist_df = load_price_history(sheet_id, selected_port)

# Si el Excel tiene datos, actualizamos el universo del portafolio dinámicamente
# para que coincida exactamente con las columnas de Precios_<portafolio>
if not prices_hist_df.empty:
    # Las columnas del DataFrame SON el universo real (sin SPX si existe)
    _excel_tickers = [c for c in prices_hist_df.columns if "SPX" not in c.upper()]
    if _excel_tickers:
        port_data["universe"] = _excel_tickers  # actualiza en runtime, no persiste

# 3. Correr modelo CORE solo con datos de este portafolio
@st.cache_data(ttl=3600)
def run_portfolio_model(sheet_id_key, port_name, rf_val):
    """
    Cached model run — keyed by (sheet, portfolio, rf).
    El universo lo determinan las columnas del Excel, no el código.
    Cada portafolio corre completamente independiente.
    """
    if not sheet_id_key:
        return {}
    _df = load_price_history(sheet_id_key, port_name)
    if _df.empty:
        return {}
    return core_model_run(_df, rf=rf_val, max_weight=0.30)

model_result = {}
if use_sheets and sheet_id and not prices_hist_df.empty:
    model_result = run_portfolio_model(sheet_id, selected_port, port_data.get("rf", 0.0264))

# ── PESOS OBJETIVO ─────────────────────────────────────────────────────────
# Prioridad:
#   1. Modelo Markowitz corrido con datos reales del Excel  ← SIEMPRE preferido
#   2. Pesos semilla hardcodeados de MATLAB                 ← solo sin Sheet
#
# NUNCA se mezclan universos entre portafolios.
# Si el Excel cambia activos → el modelo los toma en el siguiente recálculo.
if model_result and model_result.get("w_tang"):
    # Pesos del optimizador con datos reales — filtrar pesos < 0.1% (ruido numérico)
    target_wts = {t: w for t, w in model_result["w_tang"].items() if w >= 0.001}
    if not target_wts:
        target_wts = port_data.get("target", {})
    _wt_source = f"Modelo Markowitz · {model_result.get('n_obs', 0)} meses de datos reales"
elif not prices_hist_df.empty:
    # Excel conectado pero modelo no corrió (< 13 filas de datos)
    target_wts = port_data.get("target", {})
    _wt_source = "Pesos MATLAB (datos históricos insuficientes para optimizar)"
else:
    # Sin Sheet: usar pesos semilla MATLAB como referencia
    target_wts = port_data.get("target", {})
    _wt_source = "Pesos semilla MATLAB (sin datos históricos conectados)"

# 4. Precios actuales: Yahoo Finance > ops_df > demo
_yf_prices = {}
if YF_AVAILABLE:
    try:
        _all_tkrs = list(target_wts.keys())
        if not ops_df.empty and "Ticker" in ops_df.columns:
            _all_tkrs += [t for t in ops_df["Ticker"].unique() if t not in _all_tkrs]
        _yf_prices = get_live_prices(_all_tkrs)
    except Exception:
        pass

_ops_prices = {}
if not ops_df.empty and "Precio_USD" in ops_df.columns:
    _ops_prices = ops_df.sort_values("Fecha").groupby("Ticker")["Precio_USD"].last().to_dict()

prices = {**get_demo_prices(), **_ops_prices, **_yf_prices}
prices_source = "Yahoo Finance (tiempo real)" if _yf_prices else ("Operaciones" if _ops_prices else "Demo")

# 5. Posiciones y alertas
_active_tickers = list(target_wts.keys()) + [t for t in (ops_df["Ticker"].unique().tolist() if not ops_df.empty else []) if t not in target_wts]
positions_df = calc_positions(ops_df, prices, tc_actual, target_tickers=_active_tickers)
alerts_df = calc_alerts(positions_df, target_wts, tc_actual, monthly_add, thresh_rebal)

has_actions = not alerts_df.empty and alerts_df["Alerta"].isin(["COMPRAR", "VENDER"]).any()
pending = alerts_df[alerts_df["Alerta"].isin(["COMPRAR", "VENDER"])] if not alerts_df.empty else pd.DataFrame()

# 6. Totales
total_usd = positions_df["Valor USD"].sum() if not positions_df.empty else 0.0
total_mxn = total_usd * tc_actual
total_ganancia = positions_df["Ganancia USD"].sum() if not positions_df.empty else 0.0
costo_base = total_usd - total_ganancia
gan_pct = (total_ganancia / costo_base * 100) if costo_base > 0 else 0.0

# 7. Métricas de riesgo del modelo
# Prioridad:
#   1. Modelo corrido con datos reales del Excel   ← más preciso
#   2. Valores de referencia MATLAB del config     ← respaldo exacto por portafolio
#   Nunca queda vacío ni muestra datos de otro portafolio.
_matlab_ref = {
    "sharpe":     port_data.get("matlab_sharpe", 0),
    "vol_annual": port_data.get("matlab_vol",    0),
    "ret_annual": port_data.get("matlab_ret",    0),
    "max_dd":     port_data.get("matlab_p5",     0),   # proxy conservador
    "var_95":     port_data.get("matlab_p5",     0) / 12 if port_data.get("matlab_p5") else 0,
    "cvar_95":    port_data.get("matlab_p5",     0) / 12 if port_data.get("matlab_p5") else 0,
    "n_months":   0,
    "source":     "referencia_matlab",
}

if model_result and model_result.get("sharpe_tang", 0) != 0:
    risk_metrics = {
        "sharpe":      model_result.get("sharpe_tang", 0),
        "vol_annual":  model_result.get("vol_tang",    0),
        "ret_annual":  model_result.get("ret_tang",    0),
        "max_dd":      model_result.get("max_dd",      0),
        "var_95":      model_result.get("var_95",      0),
        "cvar_95":     model_result.get("cvar_95",     0),
        "n_months":    model_result.get("n_obs",       0),
        "source":      "modelo_real",
    }
else:
    # Sin Sheet o modelo sin correr: usar valores exactos de MATLAB por portafolio
    risk_metrics = _matlab_ref

# 8. TWR (rendimiento real usuario)
twr_data = calc_twr(ops_df, _yf_prices or _ops_prices, tc_actual) if not ops_df.empty else {}

# 9. Historia modelo
model_history_df = pd.DataFrame()
if not prices_hist_df.empty and model_result:
    model_history_df = calc_model_return_history(prices_hist_df, model_result)

# ECG sidebar state
st.session_state[f"has_alerts_{selected_port}"] = has_actions

# ── Banner ────────────────────────────────────────────────────────────────────
_hora = datetime.now().hour
_saludo = "Buenos días" if 6 <= _hora < 13 else "Buenas tardes" if _hora < 20 else "Buenas noches"
_nombre = user_data.get("name", current_user)

st.markdown(f"""
<div class="app-banner">
    <div>
        <div class="banner-title">{selected_port}</div>
        <div class="banner-sub">{port_data.get('description','')}</div>
        <div style="font-size:10px;color:rgba(201,162,39,0.75);margin-top:3px;">
            Precios: {prices_source} · Pesos: {_wt_source}
        </div>
        <div class="banner-date">Actualizado: {datetime.now().strftime('%d %b %Y, %H:%M')}</div>
    </div>
    <div style="text-align:right;">
        <div class="banner-count">{len(target_wts)}</div>
        <div class="banner-count-label">activos en portafolio</div>
        <div class="banner-greeting" style="margin-top:8px;">{_saludo}, <b style="color:#c9a227;">{_nombre}</b></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ─── HELPER: PROFILE CARD ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def render_profile_card(port_data_dict: dict, selected_port_name: str, rm: dict = None):
    rm = rm or {}
    icon = port_data_dict.get("icon", "📊")
    pc = port_data_dict.get("color", "#1a3a5c")
    bg = port_data_dict.get("bg", "linear-gradient(135deg,#f0f4f9,#e8f0f8)")
    border = port_data_dict.get("border", pc)
    tol = port_data_dict.get("tolerancia_pct", 50)
    hyr = port_data_dict.get("horizonte_years", 5)
    vlo, vhi = port_data_dict.get("volatilidad_range", (15, 25))
    rlo, rhi = port_data_dict.get("retorno_obj", (15, 30))
    # Sharpe objetivo: usar matlab_sharpe (valor exacto de MATLAB por portafolio)
    sobj     = port_data_dict.get("matlab_sharpe", port_data_dict.get("sharpe_obj", 1.0))
    # Valores calculados: del modelo real o referencia MATLAB — nunca de otro portafolio
    sr_calc  = rm.get("sharpe",     sobj)
    vol_calc = rm.get("vol_annual", (vlo + vhi) / 200)
    ret_calc = rm.get("ret_annual", (rlo + rhi) / 200)
    src_label = "datos reales" if rm.get("source") == "modelo_real" else "ref. MATLAB"

    tol_bands = [(0,20,"Muy baja","#1a7a4a"),(20,40,"Baja","#3b8f5c"),
                 (40,60,"Media","#c9a227"),(60,80,"Alta","#d4820a"),(80,100,"Muy alta","#c0392b")]
    bars_html = ""
    for lo, hi, lbl, c in tol_bands:
        filled = "opacity:1" if lo < tol <= hi or (tol == 0 and lo == 0) else "opacity:0.14"
        bars_html += (f"<div style='display:flex;flex-direction:column;align-items:center;flex:1;gap:2px;'>"
                     f"<div style='background:{c};border-radius:4px;height:20px;width:100%;{filled};'></div>"
                     f"<div style='font-size:8px;color:#6b7c93;text-align:center;'>{lbl}</div></div>")

    milestones = [1, 3, 5, 7, 10, 15]
    dots_html = ""
    for m in milestones:
        active = hyr >= m
        c = pc if active else "#dde3ea"
        tc2 = "#1a2332" if active else "#b0bac8"
        dots_html += (f"<div style='display:flex;flex-direction:column;align-items:center;flex:1;'>"
                     f"<div style='width:16px;height:16px;border-radius:50%;background:{c};"
                     f"margin-bottom:3px;{'box-shadow:0 0 0 3px ' + pc + '33;' if active else ''}'></div>"
                     f"<div style='font-size:8px;color:{tc2};font-weight:{'700' if active else '400'};'>{m}a</div></div>")

    full = 60
    vol_bar = (f"<div style='position:relative;height:12px;background:#e8f0f8;border-radius:6px;overflow:hidden;margin:5px 0;'>"
               f"<div style='position:absolute;left:{vlo/full*100:.0f}%;width:{(vhi-vlo)/full*100:.0f}%;height:100%;background:{pc};border-radius:6px;opacity:0.8;'></div></div>"
               f"<div style='display:flex;justify-content:space-between;font-size:8px;color:#6b7c93;'><span>0%</span><span>20%</span><span>40%</span><span>60%</span></div>")

    st.markdown(f"""
    <div style='background:{bg};border-radius:16px;padding:20px 24px;
                border-left:5px solid {border};box-shadow:0 3px 16px rgba(0,0,0,0.08);margin-bottom:8px;'>
        <div style='display:flex;align-items:center;gap:12px;margin-bottom:16px;'>
            <div style='font-size:38px;line-height:1;'>{icon}</div>
            <div>
                <div style='font-size:9px;letter-spacing:.12em;font-weight:700;color:{pc};text-transform:uppercase;'>Tu perfil de inversión</div>
                <div style='font-size:22px;font-weight:700;color:#1a2332;line-height:1.1;'>{port_data_dict.get("tolerancia","").upper()}</div>
                <div style='font-size:11px;color:#6b7c93;margin-top:1px;'>{port_data_dict.get("objetivo","")}</div>
            </div>
        </div>
        <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;'>
            <div style='background:rgba(255,255,255,0.75);border-radius:11px;padding:12px;'>
                <div style='font-size:9px;font-weight:700;color:{pc};text-transform:uppercase;margin-bottom:7px;'>💓 Tolerancia al Riesgo</div>
                <div style='font-size:16px;font-weight:700;color:{pc};margin-bottom:7px;'>{port_data_dict.get("tolerancia","")}</div>
                <div style='display:flex;gap:3px;margin-bottom:10px;'>{bars_html}</div>
                <div style='font-size:9px;font-weight:700;color:{pc};text-transform:uppercase;margin-bottom:3px;'>📊 Volatilidad esperada</div>
                <div style='font-size:13px;color:#1a2332;font-weight:600;'>{vlo}% – {vhi}%</div>
                {vol_bar}
            </div>
            <div style='background:rgba(255,255,255,0.75);border-radius:11px;padding:12px;'>
                <div style='font-size:9px;font-weight:700;color:{pc};text-transform:uppercase;margin-bottom:7px;'>⏱ Horizonte de Inversión</div>
                <div style='font-size:11px;color:#1a2332;margin-bottom:8px;'>{port_data_dict.get("horizonte","")}</div>
                <div style='display:flex;align-items:flex-end;gap:1px;margin-bottom:10px;'>{dots_html}</div>
                <div style='font-size:9px;font-weight:700;color:{pc};text-transform:uppercase;margin-bottom:3px;'>📈 Retorno objetivo</div>
                <div style='font-size:22px;font-weight:800;color:{pc};letter-spacing:-0.01em;'>{rlo}% – {rhi}%</div>
                <div style='font-size:10px;color:#1a3a5c;font-weight:600;margin-top:4px;'>
                    Retorno calculado: <b>{ret_calc*100:.2f}%</b> · <i style='color:#6b7c93;'>{src_label}</i>
                </div>
                <div style='font-size:10px;color:#6b7c93;margin-top:3px;'>
                    Sharpe objetivo: &gt; {sobj:.2f} · Calculado: <b style='color:{pc};'>{sr_calc:.3f}</b>
                </div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ─── HELPER: SHARPE LABEL ────────────────────────────────────────────────────
def sharpe_label(sr):
    if sr > 1.5: return "⭐⭐⭐ Muy Fuerte", "#1a7a4a"
    if sr > 1.0: return "⭐⭐ Sólido", "#2d7a4a"
    if sr > 0.5: return "⭐ Moderado", "#d4820a"
    return "⚠ Débil", "#c0392b"


# ══════════════════════════════════════════════════════════════════════════════
# ─── PAGE: DASHBOARD / MI PORTAFOLIO ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# ─── PAGE: DASHBOARD / MI PORTAFOLIO — rebuilt clean ─────────────────────────
# Todas las tarjetas, gráficos y semáforos leen SOLO de:
#   risk_metrics  → calculado para el portafolio activo (o referencia MATLAB)
#   port_data     → config del portafolio activo
#   positions_df  → posiciones del portafolio activo
#   alerts_df     → alertas del portafolio activo
# CERO referencias cruzadas entre portafolios.
# ══════════════════════════════════════════════════════════════════════════════
if page in ("📊 Dashboard", "📊 Mi Portafolio"):

    # ── 0. Fuente de datos: debug visible para admin ──────────────────────────
    _rm_source = risk_metrics.get("source", "sin_datos")
    _rm_sharpe = risk_metrics.get("sharpe",     0.0)
    _rm_vol    = risk_metrics.get("vol_annual", 0.0)
    _rm_ret    = risk_metrics.get("ret_annual", 0.0)
    _rm_dd     = risk_metrics.get("max_dd",     0.0)
    _rm_var    = risk_metrics.get("var_95",     0.0)
    _vlo, _vhi = port_data.get("volatilidad_range", (4, 8))
    _rlo, _rhi = port_data.get("retorno_obj",   (3, 15))
    _sr_obj    = port_data.get("matlab_sharpe", port_data.get("sharpe_obj", 1.0))

    if is_admin:
        st.markdown(
            f"<div style='background:#1a3a5c;color:white;border-radius:10px;"
            f"padding:8px 16px;font-size:11px;margin-bottom:12px;'>"
            f"🔍 <b>Datos activos — {selected_port}</b> · "
            f"Fuente: <b>{_rm_source}</b> · "
            f"Sharpe: <b>{_rm_sharpe:.3f}</b> · "
            f"Vol: <b>{_rm_vol*100:.1f}%</b> · "
            f"Ret: <b>{_rm_ret*100:.2f}%</b> · "
            f"n_meses: <b>{risk_metrics.get('n_months',0)}</b></div>",
            unsafe_allow_html=True)

    # ── 1. Tarjeta de perfil ──────────────────────────────────────────────────
    render_profile_card(port_data, selected_port, rm=risk_metrics)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── 2. KPI Row — 4 tarjetas principales ──────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    # k1: Valor total portafolio
    with k1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Valor del Portafolio · {selected_port}</div>
            <div class="kpi-value">${total_mxn:,.0f}</div>
            <div class="kpi-sub">MXN · ${total_usd:,.0f} USD · TC {tc_actual:.2f}</div>
        </div>""", unsafe_allow_html=True)

    # k2: Ganancia/Pérdida sobre costo
    with k2:
        _sign = "+" if total_ganancia >= 0 else ""
        _gc   = "positive" if total_ganancia >= 0 else "negative"
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Ganancia / Pérdida · {selected_port}</div>
            <div class="kpi-value {_gc}">{_sign}${total_ganancia:,.0f} USD</div>
            <div class="kpi-sub">{_sign}{gan_pct:.1f}% sobre lo invertido</div>
        </div>""", unsafe_allow_html=True)

    # k3: Sharpe del modelo — comparado contra el objetivo MATLAB de ESTE portafolio
    with k3:
        _sr_ok    = _rm_sharpe >= _sr_obj * 0.85
        _sr_color = "#1a7a4a" if _rm_sharpe >= _sr_obj else ("#d4820a" if _rm_sharpe >= _sr_obj * 0.5 else "#c0392b")
        _sr_tag   = "datos reales" if _rm_source == "modelo_real" else "ref. MATLAB"
        if _rm_sharpe >= _sr_obj:
            _sr_txt = f"✅ Cumple objetivo ({_sr_obj:.2f})"
        elif _rm_sharpe >= _sr_obj * 0.5:
            _sr_txt = f"🟡 Cerca del objetivo ({_sr_obj:.2f})"
        else:
            _sr_txt = f"⚠ Bajo objetivo ({_sr_obj:.2f})"
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Sharpe Ratio · {selected_port}</div>
            <div class="kpi-value" style="color:{_sr_color};">{_rm_sharpe:.3f}</div>
            <div class="kpi-sub">{_sr_txt} · <i>{_sr_tag}</i></div>
        </div>""", unsafe_allow_html=True)

    # k4: Volatilidad — comparada contra rango del portafolio
    with k4:
        _vol_pct = _rm_vol * 100
        _vol_ok  = _vol_pct <= _vhi
        _vc      = "#1a7a4a" if _vol_ok else "#c0392b"
        _vol_lbl = "✅ En rango" if _vol_ok else "⚠ Fuera de rango"
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Volatilidad Anual · {selected_port}</div>
            <div class="kpi-value" style="color:{_vc};">{_vol_pct:.1f}%</div>
            <div class="kpi-sub">Objetivo: {_vlo}%–{_vhi}% · {_vol_lbl}<br>
            Retorno esperado: <b style="color:#1a3a5c;">{_rm_ret*100:.2f}%</b>
            (obj. {_rlo}%–{_rhi}%)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── 3. Semáforos — 4 indicadores clave ───────────────────────────────────
    # Cada umbral viene del portafolio activo, nunca hardcodeado globalmente.
    _profile  = port_data.get("profile", "moderado")
    _dd_thr   = -0.08 if _profile == "conservador" else (
                -0.15 if _profile == "moderado"    else (
                -0.25 if _profile == "balanceado"  else -0.40))
    _var_thr  = abs(port_data.get("matlab_p5", -0.07)) / 12  # mensual aprox.

    sem1, sem2, sem3, sem4 = st.columns(4)

    def _sem_card(col, icon_ok, icon_bad, label, value_str, txt_ok, txt_bad, is_good):
        ico   = icon_ok if is_good else icon_bad
        color = "#1a7a4a" if is_good else "#c0392b"
        txt   = txt_ok   if is_good else txt_bad
        with col:
            st.markdown(f"""
            <div style='background:white;border-radius:12px;padding:14px;text-align:center;
                        box-shadow:0 2px 10px rgba(13,27,42,0.07);border-top:4px solid {color};'>
                <div style='font-size:20px;margin-bottom:4px;'>{ico}</div>
                <div style='font-size:9px;text-transform:uppercase;letter-spacing:.08em;
                            color:#6b7c93;margin-bottom:4px;'>{label}</div>
                <div style='font-family:"DM Serif Display",serif;font-size:22px;
                            color:{color};margin-bottom:4px;'>{value_str}</div>
                <div style='font-size:10px;color:{color};font-weight:600;
                            line-height:1.4;'>{txt}</div>
            </div>""", unsafe_allow_html=True)

    _sem_card(sem1, "🟢", "🔴",
        f"Eficiencia (Sharpe) · {selected_port}",
        f"{_rm_sharpe:.3f}",
        f"Cumple objetivo ({_sr_obj:.2f})",
        f"Bajo objetivo ({_sr_obj:.2f}) — revisa estrategia",
        _rm_sharpe >= _sr_obj * 0.85)

    _sem_card(sem2, "🟢", "🟡",
        f"Volatilidad · {selected_port}",
        f"{_rm_vol*100:.1f}%",
        f"Dentro del rango {_vlo}%–{_vhi}%",
        f"Alta para este perfil (obj ≤{_vhi}%)",
        _rm_vol * 100 <= _vhi)

    _sem_card(sem3, "🟢", "🔴",
        f"Caída Máxima · {selected_port}",
        f"{_rm_dd*100:.1f}%",
        f"Controlada para perfil {_profile}",
        f"Mayor a lo esperado (ref {_dd_thr*100:.0f}%)",
        _rm_dd >= _dd_thr)

    _sem_card(sem4, "🟢", "🟡",
        f"VaR 95% mensual · {selected_port}",
        f"{abs(_rm_var)*100:.2f}%",
        f"Riesgo aceptable para {selected_port}",
        f"Elevado para este perfil",
        abs(_rm_var) <= _var_thr * 1.2)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── 4. Alertas del modelo ─────────────────────────────────────────────────
    if has_actions:
        st.markdown("<div class='section-title'>📋 Acciones Recomendadas por el Modelo</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:12px;color:#6b7c93;margin-bottom:12px;'>"
            f"El modelo Markowitz detectó desviaciones en <b>{selected_port}</b>. "
            f"Estas operaciones restauran los pesos óptimos.</div>",
            unsafe_allow_html=True)
        _order_cols = st.columns(min(len(pending), 4))
        for _i, (_, _row) in enumerate(pending.iterrows()):
            with _order_cols[_i % len(_order_cols)]:
                if _row["Alerta"] == "COMPRAR":
                    _mxn = _row["Compra Sugerida MXN"]
                    _usd = _mxn / tc_actual
                    st.markdown(
                        "<div class='order-card order-buy'>"
                        "<div style='font-size:24px'>🟢</div><div style='flex:1'>"
                        f"<div style='font-weight:700;font-size:15px;color:#1a3a5c;'>{_row['Ticker']}</div>"
                        "<div style='font-size:11px;font-weight:700;color:#1a7a4a;margin-bottom:4px;'>COMPRAR</div>"
                        f"<div style='font-size:12px;'><b>{_row['Acciones a Comprar']:.4f} acc</b>"
                        f" @ ${_row['Precio Actual']:,.2f}</div>"
                        f"<div style='font-size:12px;color:#1a7a4a;'><b>${_usd:,.2f} USD</b>"
                        f" · ${_mxn:,.0f} MXN</div>"
                        f"<div style='font-size:10px;color:#888;margin-top:3px;'>"
                        f"Actual {_row['Peso Actual']*100:.1f}% → Obj {_row['Peso Objetivo']*100:.1f}%</div>"
                        "</div></div>", unsafe_allow_html=True)
                else:
                    _mxn = _row["Monto a Vender MXN"]
                    _usd = _mxn / tc_actual
                    st.markdown(
                        "<div class='order-card order-sell'>"
                        "<div style='font-size:24px'>🔴</div><div style='flex:1'>"
                        f"<div style='font-weight:700;font-size:15px;color:#1a3a5c;'>{_row['Ticker']}</div>"
                        "<div style='font-size:11px;font-weight:700;color:#c0392b;margin-bottom:4px;'>VENDER</div>"
                        f"<div style='font-size:12px;'><b>{_row['Acciones a Vender']:.4f} acc</b>"
                        f" @ ${_row['Precio Actual']:,.2f}</div>"
                        f"<div style='font-size:12px;color:#c0392b;'><b>${_usd:,.2f} USD</b>"
                        f" · ${_mxn:,.0f} MXN</div>"
                        f"<div style='font-size:10px;color:#888;margin-top:3px;'>"
                        f"Actual {_row['Peso Actual']*100:.1f}% → Obj {_row['Peso Objetivo']*100:.1f}%</div>"
                        "</div></div>", unsafe_allow_html=True)
    else:
        st.success(f"✅ {selected_port} en equilibrio — no hay acciones pendientes.")

    # ── 5. Cotizaciones en tiempo real ────────────────────────────────────────
    st.markdown(f"<div class='section-title'>📡 Cotizaciones en Tiempo Real · {selected_port}</div>",
                unsafe_allow_html=True)
    _port_tickers = list(target_wts.keys())
    if _yf_prices and _port_tickers:
        _qt = get_live_quotes(_port_tickers)
        _qcols = st.columns(min(len(_port_tickers), 5))
        for _qi, _tkr in enumerate(_port_tickers):
            _q    = _qt.get(_tkr, {})
            _px   = _q.get("price")
            _chg  = _q.get("change_pct")
            _w52h = _q.get("week52_high")
            _w52l = _q.get("week52_low")
            _vsma = _q.get("vs_ma52w")
            with _qcols[_qi % len(_qcols)]:
                if not _px:
                    st.markdown(
                        f"<div style='background:white;border-radius:12px;padding:14px;"
                        f"text-align:center;border:1px solid #e8eef5;'>"
                        f"<b>{_tkr}</b><br><span style='color:#aab;font-size:12px;'>Sin datos</span></div>",
                        unsafe_allow_html=True)
                    continue
                _cc  = "#1a7a4a" if (_chg or 0) >= 0 else "#c0392b"
                _cbg = "#d5f5e3" if (_chg or 0) >= 0 else "#fadbd8"
                _cic = "▲" if (_chg or 0) >= 0 else "▼"
                _cst = f"{_cic} {abs(_chg):.2f}%" if _chg is not None else "—"
                if _vsma is None:   _sig, _sc, _sbg = "Sin datos", "#6b7c93", "#f4f7fb"
                elif _vsma > 20:    _sig, _sc, _sbg = "MUY CARA 🔴", "#c0392b", "#fadbd8"
                elif _vsma > 5:     _sig, _sc, _sbg = "ALTA 🟡", "#d4820a", "#fef9e7"
                elif _vsma >= -5:   _sig, _sc, _sbg = "JUSTO 🔵", "#2d4a6b", "#e8f0f8"
                elif _vsma >= -20:  _sig, _sc, _sbg = "BAJA 🟢", "#1a7a4a", "#d5f5e3"
                else:               _sig, _sc, _sbg = "MUY BAJA 🟢", "#1a7a4a", "#d5f5e3"
                _bp = max(0, min(100,
                    (_px - (_w52l or 0)) / ((_w52h or _px+1) - (_w52l or 0)) * 100
                )) if _w52h and _w52l else 50
                st.markdown(
                    f"<div style='background:white;border-radius:12px;padding:14px;"
                    f"border:1px solid #e8eef5;box-shadow:0 2px 8px rgba(13,27,42,0.06);'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;'>"
                    f"<span style='font-weight:700;font-size:14px;'>{_tkr}</span>"
                    f"<span style='background:{_cbg};color:{_cc};padding:2px 7px;"
                    f"border-radius:9px;font-size:10px;font-weight:700;'>{_cst}</span></div>"
                    f"<div style='font-size:24px;font-weight:700;color:#1a3a5c;margin-bottom:8px;'>${_px:,.2f}</div>"
                    f"<div style='background:{_sbg};border-radius:7px;padding:5px 9px;margin-bottom:8px;"
                    f"border-left:3px solid {_sc};'>"
                    f"<div style='font-size:10px;font-weight:700;color:{_sc};'>{_sig}</div>"
                    f"<div style='font-size:9px;color:#6b7c93;'>"
                    f"{f'{_vsma:+.1f}% vs prom 52s' if _vsma is not None else ''}</div></div>"
                    f"<div style='font-size:9px;color:#6b7c93;margin-bottom:3px;'>"
                    f"52s: ${_w52l:,.2f}–${_w52h:,.2f}</div>"
                    f"<div style='background:#eef2f7;border-radius:4px;height:5px;position:relative;'>"
                    f"<div style='position:absolute;left:{_bp:.0f}%;top:-4px;"
                    f"width:13px;height:13px;border-radius:50%;background:#1a3a5c;"
                    f"transform:translateX(-50%);border:2px solid white;"
                    f"box-shadow:0 1px 4px rgba(0,0,0,0.2);'></div></div>"
                    f"</div>", unsafe_allow_html=True)
    else:
        st.info(f"📡 Sin cotizaciones en tiempo real para {selected_port}. Instala yfinance.")

    # ── 6. Posiciones + Donuts ────────────────────────────────────────────────
    st.markdown(f"<div class='section-title'>📋 Posiciones · {selected_port}</div>",
                unsafe_allow_html=True)
    _pc1, _pc2 = st.columns([3, 2])

    with _pc1:
        if not alerts_df.empty:
            def _badge(a):
                if a == "COMPRAR": return '<span class="badge-ok">🟢 COMPRAR</span>'
                if a == "VENDER":  return '<span class="badge-danger">🔴 VENDER</span>'
                return '<span class="badge-ok">✅ OK</span>'
            _rows_h = ""
            for _, _r in alerts_df.iterrows():
                _gc2 = "#1a7a4a" if _r["Ganancia USD"] >= 0 else "#c0392b"
                _gs  = "+" if _r["Ganancia USD"] >= 0 else ""
                _dev_color = "#c0392b" if abs(_r["Desviación"]) > thresh_rebal else "#1a7a4a"
                _rows_h += (
                    f"<tr><td><b>{_r['Ticker']}</b></td>"
                    f"<td>{_r['Cantidad']:.4f}</td>"
                    f"<td>${_r['Precio Actual']:,.2f}</td>"
                    f"<td><b>${_r['Valor MXN']:,.0f}</b></td>"
                    f"<td>{_r['Peso Actual']*100:.1f}%</td>"
                    f"<td>{_r['Peso Objetivo']*100:.1f}%</td>"
                    f"<td style='color:{_dev_color}'>{_r['Desviación']*100:+.1f}%</td>"
                    f"<td style='color:{_gc2}'>{_gs}${_r['Ganancia USD']:,.0f}</td>"
                    f"<td>{_badge(_r['Alerta'])}</td></tr>")
            st.markdown(
                f"<div class='table-wrap'><table class='styled-table'>"
                f"<thead><tr><th>Ticker</th><th>Cantidad</th><th>Precio</th>"
                f"<th>Valor MXN</th><th>Peso%</th><th>Obj%</th>"
                f"<th>Desv.</th><th>Ganancia</th><th>Alerta</th></tr></thead>"
                f"<tbody>{_rows_h}</tbody></table></div>",
                unsafe_allow_html=True)
        else:
            st.info(f"Sin posiciones para {selected_port}. Registra operaciones.")

    with _pc2:
        st.markdown(f"<div style='font-size:13px;font-weight:700;color:#1a3a5c;"
                    f"margin-bottom:8px;'>Actual vs Objetivo · {selected_port}</div>",
                    unsafe_allow_html=True)
        if not alerts_df.empty and not alerts_df["Peso Actual"].isna().all():
            _pal = ["#1a3a5c","#c9a227","#1a7a4a","#c0392b","#5b4fcf","#0e7490","#b45309","#6d28d9"]
            _all_t = list(dict.fromkeys(list(alerts_df["Ticker"]) + list(target_wts.keys())))
            _tc3   = {t: _pal[i % len(_pal)] for i, t in enumerate(_all_t)}
            _fig_pie = make_subplots(1, 2, specs=[[{"type":"pie"},{"type":"pie"}]],
                                     subplot_titles=["Actual","Objetivo"])
            _fig_pie.add_trace(go.Pie(
                labels=alerts_df["Ticker"],
                values=alerts_df["Peso Actual"],
                hole=0.52,
                marker=dict(colors=[_tc3.get(t,"#888") for t in alerts_df["Ticker"]],
                            line=dict(color="rgba(255,255,255,0.7)", width=2)),
                textinfo="label+percent", textfont_size=10, showlegend=True,
                hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>"
            ), 1, 1)
            _fig_pie.add_trace(go.Pie(
                labels=list(target_wts.keys()),
                values=list(target_wts.values()),
                hole=0.52,
                marker=dict(colors=[_tc3.get(t,"#888") for t in target_wts],
                            line=dict(color="rgba(255,255,255,0.7)", width=2)),
                textinfo="label+percent", textfont_size=10, showlegend=False,
                hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>"
            ), 1, 2)
            _fig_pie.update_layout(
                height=300, paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10,r=10,t=40,b=50),
                legend=dict(orientation="h",y=-0.25,x=0.5,xanchor="center",font_size=10),
                annotations=[
                    dict(text="Actual",  x=0.18, y=0.5, font_size=12, showarrow=False),
                    dict(text="Objetivo",x=0.82, y=0.5, font_size=12, showarrow=False),
                ])
            st.markdown('<div class="chart3d">', unsafe_allow_html=True)
            st.plotly_chart(_fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Sin posiciones para graficar.")

    # ── 7. Evolución histórica del modelo ─────────────────────────────────────
    if not model_history_df.empty:
        st.markdown(f"<div class='section-title'>📈 Evolución del Modelo · {selected_port}</div>",
                    unsafe_allow_html=True)
        _fig_ev = go.Figure()
        _fig_ev.add_trace(go.Scatter(
            x=model_history_df["Mes"], y=model_history_df["Valor"],
            fill="tozeroy", fillcolor="rgba(26,58,92,0.08)",
            line=dict(color=port_color, width=2.5),
            mode="lines+markers",
            marker=dict(size=6, color="#c9a227", line=dict(width=1.5, color="white")),
            name=f"Índice {selected_port} (base 100)",
            hovertemplate="<b>%{x}</b><br>Valor: %{y:.2f}<extra></extra>"))
        _fig_ev.add_trace(go.Bar(
            x=model_history_df["Mes"], y=model_history_df["Retorno"],
            marker_color=["#1a7a4a" if r >= 0 else "#c0392b"
                          for r in model_history_df["Retorno"]],
            opacity=0.65, yaxis="y2", name="Retorno mensual %",
            hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>"))
        _fig_ev.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(232,240,248,0.2)",
            margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation="h", y=1.05, font_size=10),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#eef2f7", title="Valor (base 100)"),
            yaxis2=dict(overlaying="y", side="right", showgrid=False, ticksuffix="%"),
            hovermode="x unified")
        st.markdown('<div class="chart3d">', unsafe_allow_html=True)
        st.plotly_chart(_fig_ev, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── 8. Frontera eficiente + descarga PDF (solo admin) ─────────────────────
    if is_admin and model_result and model_result.get("frontier"):
        st.markdown(f"<div class='section-title'>🗺️ Frontera Eficiente · {selected_port}</div>",
                    unsafe_allow_html=True)
        _frontier = model_result["frontier"]
        _fv = [p["vol"] * 100 for p in _frontier]
        _fr = [p["ret"] * 100 for p in _frontier]
        _fig_front = go.Figure()
        _fig_front.add_trace(go.Scatter(
            x=_fv, y=_fr, mode="lines+markers",
            line=dict(color=port_color, width=2.5),
            marker=dict(size=7, color=port_color),
            name=f"Frontera · {selected_port}",
            hovertemplate="σ=%{x:.1f}% · R=%{y:.1f}%<extra></extra>"))
        _vt = model_result.get("vol_tang", 0) * 100
        _rt = model_result.get("ret_tang", 0) * 100
        _fig_front.add_trace(go.Scatter(
            x=[_vt], y=[_rt], mode="markers",
            marker=dict(size=14, color="#c9a227", symbol="star",
                        line=dict(width=2, color="#1a3a5c")),
            name=f"Tangencia SR={model_result.get('sharpe_tang',0):.2f}",
            hovertemplate=f"Tangencia<br>σ={_vt:.1f}% · R={_rt:.1f}%<extra></extra>"))
        _vm = model_result.get("vol_mv", 0) * 100
        _rm2 = model_result.get("ret_mv", 0) * 100
        _fig_front.add_trace(go.Scatter(
            x=[_vm], y=[_rm2], mode="markers",
            marker=dict(size=12, color="#10b981", symbol="diamond"),
            name="Mínima Varianza"))
        _rf_pct = model_result.get("rf", 0.0264) * 100
        _cml_x  = [0, max(_fv) * 1.1] if _fv else [0, 30]
        _cml_y  = [_rf_pct, _rf_pct + (_rt - _rf_pct) / _vt * _cml_x[1]] if _vt > 0 else [_rf_pct, _rf_pct]
        _fig_front.add_trace(go.Scatter(
            x=_cml_x, y=_cml_y, mode="lines",
            line=dict(color="#ef4444", width=2, dash="dash"),
            name=f"CML (Rf={_rf_pct:.2f}%)"))
        _fig_front.update_layout(
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(232,240,248,0.2)",
            xaxis=dict(title="Riesgo — Volatilidad Anual (%)", gridcolor="#eef2f7"),
            yaxis=dict(title="Rendimiento Esperado Anual (%)", gridcolor="#eef2f7"),
            margin=dict(l=10,r=10,t=10,b=10),
            legend=dict(orientation="h", y=1.05, font_size=10))
        st.markdown('<div class="chart3d">', unsafe_allow_html=True)
        st.plotly_chart(_fig_front, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # PDF download
        _pdf_bytes = build_pdf_report(
            selected_port, model_result, positions_df, alerts_df, twr_data, model_history_df)
        st.download_button(
            f"📥 Descargar Reporte PDF — {selected_port}",
            data=_pdf_bytes,
            file_name=f"EGADA_{selected_port}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ─── PAGE: MODELO VS REALIDAD ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔀 Modelo vs Realidad":
    st.markdown("<div class='section-title'>🔀 Rendimiento Modelo vs Tu Rendimiento Real</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:12px;color:#6b7c93;margin-bottom:18px;'>"
        "Comparativo entre el rendimiento teórico del modelo Markowitz "
        "y tu rendimiento real medido con Time-Weighted Return (TWR) desde tu primera compra.</div>",
        unsafe_allow_html=True)

    col_m, col_r, col_d = st.columns(3)

    # Modelo
    model_ret_pct = model_result.get("ret_tang", 0) * 100 if model_result else None
    model_vol_pct = model_result.get("vol_tang", 0) * 100 if model_result else None
    model_sr = model_result.get("sharpe_tang", 0) if model_result else None

    # Real
    twr_pct = twr_data.get("twr_pct", None)
    twr_ann = twr_data.get("retorno_anualizado", None)
    first_date = twr_data.get("first_date", None)
    net_inv = twr_data.get("net_invested_usd", 0)
    val_act = twr_data.get("valor_actual_usd", 0)

    with col_m:
        mc_color = "#1a3a5c"
        st.markdown(f"""
        <div class="compare-card" style="border-left:5px solid {mc_color};">
            <div style="font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
                        color:{mc_color};margin-bottom:10px;">📊 Rendimiento Modelo</div>
            <div style="font-size:26px;font-weight:700;color:{mc_color};">
                {f'{model_ret_pct:+.2f}%' if model_ret_pct is not None else 'Sin datos históricos'}</div>
            <div style="font-size:11px;color:#6b7c93;margin-top:4px;">Retorno anual esperado (Markowitz)</div>
            <div style="margin-top:10px;padding-top:10px;border-top:1px solid #e8f0f8;">
                <div style="font-size:10px;color:#6b7c93;">Volatilidad: <b>{f'{model_vol_pct:.1f}%' if model_vol_pct is not None else '—'}</b></div>
                <div style="font-size:10px;color:#6b7c93;">Sharpe: <b>{f'{model_sr:.2f}' if model_sr is not None else '—'}</b></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_r:
        twr_c = "#1a7a4a" if (twr_pct or 0) >= 0 else "#c0392b"
        st.markdown(f"""
        <div class="compare-card" style="border-left:5px solid {twr_c};">
            <div style="font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
                        color:{twr_c};margin-bottom:10px;">📈 Tu Rendimiento Hoy (TWR)</div>
            <div style="font-size:26px;font-weight:700;color:{twr_c};">
                {f'{twr_pct:+.2f}%' if twr_pct is not None else 'Sin operaciones registradas'}</div>
            <div style="font-size:11px;color:#6b7c93;margin-top:4px;">
                Desde {first_date.strftime('%d %b %Y') if first_date and pd.notna(first_date) else '—'} · Precios Yahoo Finance</div>
            <div style="margin-top:10px;padding-top:10px;border-top:1px solid #e8f0f8;">
                <div style="font-size:10px;color:#6b7c93;">Invertido: <b>${net_inv:,.2f} USD</b></div>
                <div style="font-size:10px;color:#6b7c93;">Valor actual: <b>${val_act:,.2f} USD</b></div>
                <div style="font-size:10px;color:#6b7c93;">Retorno anualizado: <b>{f'{twr_ann:.1f}%' if twr_ann else '—'}</b></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_d:
        if model_ret_pct is not None and twr_pct is not None:
            diff = twr_pct - model_ret_pct
            diff_c = "#1a7a4a" if diff >= 0 else "#c0392b"
            if diff >= 5:
                interp = "🚀 Tu rendimiento supera al modelo — excelente ejecución"
            elif diff >= 0:
                interp = "✅ En línea con el modelo — estrategia funcionando"
            elif diff >= -10:
                interp = "🟡 Ligeramente por debajo del modelo — mercado actual vs teórico"
            else:
                interp = "⚠ Diferencia importante — el mercado actual vs condiciones históricas"
        else:
            diff = None
            diff_c = "#6b7c93"
            interp = "Necesitas datos históricos en Sheets y operaciones registradas"

        st.markdown(f"""
        <div class="compare-card" style="border-left:5px solid {diff_c};">
            <div style="font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
                        color:{diff_c};margin-bottom:10px;">⚖️ Diferencia</div>
            <div style="font-size:26px;font-weight:700;color:{diff_c};">
                {f'{diff:+.2f}%' if diff is not None else '—'}</div>
            <div style="font-size:11px;color:#6b7c93;margin-top:4px;">Real menos Modelo</div>
            <div style="margin-top:10px;padding-top:10px;border-top:1px solid #e8f0f8;
                        font-size:11px;color:#1a2332;line-height:1.5;">{interp}</div>
        </div>""", unsafe_allow_html=True)

    # ── Gráfico comparativo ───────────────────────────────────────────────────
    if model_result and model_result.get("mc_p5") is not None:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='background:#e8f0f8;border-left:3px solid #1a3a5c;border-radius:0 10px 10px 0;"
            "padding:10px 14px;font-size:12px;color:#2d4a6b;margin-bottom:12px;'>"
            f"<b>Intervalo de confianza 90% (MC 10,000 sim.):</b> "
            f"{model_result['mc_p5']*100:.1f}% a {model_result['mc_p95']*100:.1f}% · "
            f"Mediana: {model_result['mc_p50']*100:.1f}% · "
            f"Probabilidad de pérdida anual: {model_result['mc_prob_loss']*100:.1f}%</div>",
            unsafe_allow_html=True)

        # Gráfico de horizontes
        hz = model_result.get("horizon_intervals", {})
        if hz:
            fig_hz = go.Figure()
            h_years = list(hz.keys())
            means = [hz[y]["mean"] for y in h_years]
            p5s = [hz[y]["p5"] for y in h_years]
            p95s = [hz[y]["p95"] for y in h_years]

            fig_hz.add_trace(go.Scatter(
                x=h_years + h_years[::-1],
                y=p95s + p5s[::-1],
                fill="toself", fillcolor="rgba(59,130,246,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Intervalo 90%",
                hoverinfo="skip"
            ))
            fig_hz.add_trace(go.Scatter(
                x=h_years, y=means, mode="lines+markers",
                line=dict(color="#3b82f6", width=2.5),
                marker=dict(size=9, color="#c9a227"),
                name="Rendimiento esperado",
                hovertemplate="<b>%{x} años</b><br>Esperado: %{y:.1f}%<extra></extra>"
            ))
            if twr_pct is not None:
                fig_hz.add_hline(y=twr_pct, line=dict(color="#1a7a4a", dash="dash", width=2),
                                annotation_text=f"Tu rendimiento actual: {twr_pct:.1f}%",
                                annotation_position="right")

            fig_hz.update_layout(
                title="Proyección de Rendimiento por Horizonte (con tu rendimiento actual)",
                height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(232,240,248,0.2)",
                xaxis=dict(title="Horizonte (años)", gridcolor="#eef2f7"),
                yaxis=dict(title="Rendimiento Total Acumulado (%)", gridcolor="#eef2f7"),
                legend=dict(orientation="h", y=1.05, font_size=10),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.markdown('<div class="chart3d">', unsafe_allow_html=True)
            st.plotly_chart(fig_hz, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Alerta automática si diferencia es grande ─────────────────────────────
    if diff is not None and abs(diff) >= 10:
        st.warning(
            f"⚠ **Alerta de seguimiento**: Tu rendimiento actual ({twr_pct:.1f}%) difiere "
            f"{diff:+.1f}% respecto al modelo ({model_ret_pct:.1f}%). "
            "Considera revisar tu cartera con tu asesor. El sistema puede enviar esta alerta por email.",
            icon="⚠️"
        )
    elif diff is not None and abs(diff) < 5:
        st.success(f"✅ Tu portafolio está funcionando en línea con el modelo (diferencia: {diff:+.1f}%).")

    # ── Actualización mensual automática ──────────────────────────────────────
    st.markdown("<div class='section-title'>📅 Historial de Precios — Actualización Mensual</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='background:#e8f0f8;border-left:3px solid #1a3a5c;border-radius:0 10px 10px 0;"
        "padding:12px 16px;font-size:12px;color:#2d4a6b;margin-bottom:14px;'>"
        "<b>Historial base:</b> Los datos históricos llegan hasta 30/01/2026. "
        "El sistema actualiza mensualmente el primer día disponible después del cierre del mes. "
        "Los nuevos precios se agregan a la hoja <code>Precios_{selected_port}</code> y el modelo se recalcula automáticamente.</div>",
        unsafe_allow_html=True)

    if is_admin:
        with st.expander("➕ Agregar precios del mes (actualización mensual manual)"):
            with st.form(f"update_prices_{selected_port}"):
                cols_p = [c for c in (prices_hist_df.columns.tolist() if not prices_hist_df.empty else list(target_wts.keys()))]
                col1p, col2p = st.columns(2)
                with col1p:
                    fecha_cierre = st.date_input("Fecha cierre del mes")
                precio_entries = {}
                for tkr_p in cols_p[:8]:
                    precio_entries[tkr_p] = st.number_input(f"{tkr_p} (USD)", min_value=0.0, value=prices.get(tkr_p, 0.0), step=0.01)
                if st.form_submit_button("💾 Guardar precios del mes"):
                    if use_sheets:
                        row_new = [str(fecha_cierre)] + [precio_entries.get(t, "") for t in cols_p[:8]]
                        ok = save_row(sheet_id, f"Precios_{selected_port}", row_new)
                        if ok:
                            st.success(f"✅ Precios de {fecha_cierre} guardados en Precios_{selected_port}")
                            st.cache_data.clear()
                        else:
                            st.error("Error guardando en Sheets")
                    else:
                        st.info("Conecta Google Sheets para guardar permanentemente.")


# ══════════════════════════════════════════════════════════════════════════════
# ─── PAGE: OPERACIONES ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
elif page in ("📋 Operaciones", "📋 Registrar Operación"):
    st.markdown("<div class='section-title'>📋 Registro de Operaciones</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:12px;color:#6b7c93;margin-bottom:14px;'>"
        f"Las operaciones se guardan en <b>Ops_{selected_port}</b>. Solo afectan a este portafolio.</div>",
        unsafe_allow_html=True)

    with st.form(f"form_op_{selected_port}", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            f_fecha = st.date_input("Fecha", value=date.today())
            f_tipo = st.selectbox("Tipo", ["Compra", "Venta"])
        with c2:
            f_ticker = st.text_input("Ticker", placeholder="NVDA").strip().upper()
            f_cant = st.number_input("Cantidad", min_value=0.0001, value=1.0, step=0.001, format="%.4f")
        with c3:
            f_precio = st.number_input("Precio USD", min_value=0.01, value=100.0, step=0.01)
            f_com = st.number_input("Comisión USD", min_value=0.0, value=0.50, step=0.01)
        f_tc = st.number_input("TC MXN/USD", value=tc_actual, step=0.01, format="%.2f")

        if st.form_submit_button("💾 Registrar Operación", use_container_width=True, type="primary"):
            if f_ticker:
                total_usd_op = f_cant * f_precio + f_com
                st.markdown(
                    f"<div style='background:#d5f5e3;border-radius:10px;padding:12px 16px;"
                    f"font-size:13px;color:#1a2332;margin:8px 0;'>"
                    f"✅ <b>{f_tipo}</b> · {f_cant:.4f} <b>{f_ticker}</b> @ ${f_precio:.2f} USD "
                    f"· Total: ${total_usd_op:.2f} USD · ${total_usd_op*f_tc:,.0f} MXN</div>",
                    unsafe_allow_html=True)
                if use_sheets:
                    ok = save_row(sheet_id, f"Ops_{selected_port}",
                                  [str(f_fecha), f_ticker, f_tipo, f_cant, f_precio, f_com, f_tc, current_user])
                    if ok:
                        st.success(f"✅ Guardado en Ops_{selected_port}")
                        st.cache_data.clear()
                    else:
                        st.error("Error guardando en Google Sheets")
                else:
                    st.info("Demo: conecta Google Sheets para persistencia.")
            else:
                st.error("Ingresa un ticker válido")

    # Historial
    st.markdown("<div class='section-title'>Historial de Operaciones</div>", unsafe_allow_html=True)
    if not ops_df.empty:
        fc1, fc2 = st.columns(2)
        with fc1:
            ftk = st.multiselect("Ticker", ops_df["Ticker"].unique().tolist(),
                                  default=ops_df["Ticker"].unique().tolist(), key="flt_tk")
        with fc2:
            ftp = st.multiselect("Tipo", ["Compra","Venta"], default=["Compra","Venta"], key="flt_tp")
        fops = ops_df[ops_df["Ticker"].isin(ftk) & ops_df["Tipo"].isin(ftp)].copy()
        fops["Total USD"] = fops["Cantidad"] * fops["Precio_USD"] + fops["Comision_USD"]
        fops["Total MXN"] = fops["Total USD"] * fops["TC_MXN"]
        rows_h = ""
        for _, row in fops.iterrows():
            tc_color = "#1a7a4a" if row["Tipo"] == "Compra" else "#c0392b"
            tc_bg = "#d5f5e3" if row["Tipo"] == "Compra" else "#fadbd8"
            badge = f'<span style="background:{tc_bg};color:{tc_color};padding:3px 10px;border-radius:12px;font-size:10px;font-weight:700;">{row["Tipo"]}</span>'
            rows_h += (f"<tr><td>{row['Fecha']}</td><td><b>{row['Ticker']}</b></td>"
                      f"<td>{badge}</td><td>{row['Cantidad']:.4f}</td>"
                      f"<td>${row['Precio_USD']:,.2f}</td><td>${row['Comision_USD']:,.2f}</td>"
                      f"<td>{row['TC_MXN']:.2f}</td>"
                      f"<td><b>${row['Total USD']:,.2f}</b></td><td><b>${row['Total MXN']:,.0f}</b></td></tr>")
        st.markdown(
            f"<div class='table-wrap'><table class='styled-table'>"
            f"<thead><tr><th>Fecha</th><th>Ticker</th><th>Tipo</th><th>Cantidad</th>"
            f"<th>Precio USD</th><th>Comisión</th><th>TC</th><th>Total USD</th><th>Total MXN</th></tr></thead>"
            f"<tbody>{rows_h}</tbody></table></div>", unsafe_allow_html=True)
        tot_inv = fops["Total MXN"].sum()
        st.markdown(
            f"<div style='background:#1a3a5c;color:white;border-radius:8px;padding:10px 18px;margin-top:10px;"
            f"display:flex;justify-content:space-between;'>"
            f"<span>{len(fops)} operaciones</span><span>Total: <b>${tot_inv:,.0f} MXN</b></span></div>",
            unsafe_allow_html=True)
    else:
        st.info("Sin operaciones registradas para este portafolio.")


# ══════════════════════════════════════════════════════════════════════════════
# ─── PAGE: ANÁLISIS DE RIESGO ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
elif page in ("⚡ Análisis de Riesgo", "📈 Cómo va mi inversión"):
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#eef3f8,#e8f0f8);border-radius:14px;
                padding:18px 24px;margin-bottom:20px;border-left:5px solid #1a3a5c;'>
        <div style='font-family:"DM Serif Display",serif;font-size:22px;color:#1a2332;margin-bottom:4px;'>
            {'⚡ Análisis de Riesgo' if is_admin else '📈 Cómo va mi inversión'}
        </div>
        <div style='font-size:12px;color:#6b7c93;line-height:1.6;'>
            Métricas del modelo Markowitz para <b>{selected_port}</b>. 
            {'Todos los cálculos provienen del código core y usan solo datos de este portafolio.' if is_admin else
             'Aquí encontrarás los números clave de tu portafolio explicados en términos simples.'}
        </div>
    </div>""", unsafe_allow_html=True)

    # 4 KPIs de riesgo
    rk1, rk2, rk3, rk4 = st.columns(4)
    _sr = risk_metrics.get("sharpe", 0)
    _vol = risk_metrics.get("vol_annual", 0)
    _dd = risk_metrics.get("max_dd", 0)
    _var = risk_metrics.get("var_95", 0)
    _sl, _sc = sharpe_label(_sr)

    for (col, icon, lbl, val, sub, color) in [
        (rk1, "🎯", "Sharpe Ratio", f"{_sr:.3f}", _sl, _sc),
        (rk2, "📊", "Volatilidad Anual", f"{_vol*100:.1f}%",
         f"Obj: {port_data.get('volatilidad_range',(15,25))[0]}%–{port_data.get('volatilidad_range',(15,25))[1]}%",
         "#1a7a4a" if _vol*100 <= port_data.get("volatilidad_range",(15,25))[1] else "#c0392b"),
        (rk3, "📉", "Caída Máxima", f"{_dd*100:.1f}%", "Peor caída histórica del modelo",
         "#1a7a4a" if _dd >= -0.15 else "#c0392b"),
        (rk4, "⚠️", "VaR 95%", f"{abs(_var)*100:.2f}%", "Pérdida max mensual en peor 5%",
         "#1a7a4a" if abs(_var) <= 0.05 else "#c0392b"),
    ]:
        with col:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">{icon} {lbl}</div>
                <div class="kpi-value" style="color:{color};">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    if not model_result:
        st.info("Conecta Google Sheets con datos históricos de Precios_{selected_port} para activar el modelo completo.")
        st.stop()

    # ── Gráficos del modelo ───────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Frontera & SML", "📊 Correlaciones", "🎲 Monte Carlo", "📈 VaR por Activo"])

    with tab1:
        c1f, c2f = st.columns(2)
        with c1f:
            st.markdown("**Frontera Eficiente + CML**")
            frontier = model_result.get("frontier", [])
            if frontier:
                f_vols = [pt["vol"] * 100 for pt in frontier]
                f_rets = [pt["ret"] * 100 for pt in frontier]
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(x=f_vols, y=f_rets, mode="lines+markers",
                    line=dict(color="#3b82f6", width=2.5), marker=dict(size=7, color="#3b82f6"),
                    name="Frontera Eficiente",
                    hovertemplate="σ=%{x:.1f}% · R=%{y:.1f}%<extra></extra>"))
                vt = model_result.get("vol_tang", 0) * 100
                rt = model_result.get("ret_tang", 0) * 100
                fig_f.add_trace(go.Scatter(x=[vt], y=[rt], mode="markers",
                    marker=dict(size=14, color="#c9a227", symbol="star"),
                    name=f"Tangencia SR={model_result.get('sharpe_tang',0):.2f}"))
                vm = model_result.get("vol_mv", 0) * 100
                rm = model_result.get("ret_mv", 0) * 100
                fig_f.add_trace(go.Scatter(x=[vm], y=[rm], mode="markers",
                    marker=dict(size=11, color="#10b981", symbol="diamond"), name="Mín. Varianza"))
                rf_pct = model_result.get("rf", 0.0264) * 100
                cml_x = [0, max(f_vols) * 1.1 if f_vols else 50]
                cml_y = [rf_pct, rf_pct + (rt - rf_pct) / vt * cml_x[1]] if vt > 0 else [rf_pct, rf_pct]
                fig_f.add_trace(go.Scatter(x=cml_x, y=cml_y, mode="lines",
                    line=dict(color="#ef4444", width=2, dash="dash"), name=f"CML (Rf={rf_pct:.2f}%)"))
                fig_f.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(232,240,248,0.2)",
                    xaxis_title="Riesgo (%)", yaxis_title="Retorno (%)",
                    margin=dict(l=10,r=10,t=10,b=10),
                    legend=dict(orientation="h", y=1.05, font_size=9))
                st.markdown('<div class="chart3d">', unsafe_allow_html=True)
                st.plotly_chart(fig_f, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with c2f:
            st.markdown("**Security Market Line (SML)**")
            betas = model_result.get("betas", {})
            mu_ann_m = model_result.get("mu_ann", {})
            alphas_c = model_result.get("alphas_capm", {})
            if betas:
                tickers_sml = list(betas.keys())
                betas_v = [betas[t] for t in tickers_sml]
                rets_v = [mu_ann_m.get(t, 0) * 100 for t in tickers_sml]
                alphas_v = [alphas_c.get(t, 0) * 100 for t in tickers_sml]
                colors_sml = ["#1a7a4a" if a >= 0 else "#c0392b" for a in alphas_v]
                fig_sml = go.Figure()
                # SML line
                rf_p = model_result.get("rf", 0.0264) * 100
                beta_range = [0, max(betas_v) * 1.15]
                # Using midpoint estimate for Rm
                Rm_est = rf_p + 5.0  # approx 5% MRP
                sml_y = [rf_p + b * (Rm_est - rf_p) for b in beta_range]
                fig_sml.add_trace(go.Scatter(x=beta_range, y=sml_y, mode="lines",
                    line=dict(color="#ef4444", width=2, dash="dash"), name="SML"))
                fig_sml.add_trace(go.Scatter(x=betas_v, y=rets_v, mode="markers+text",
                    marker=dict(size=12, color=colors_sml),
                    text=tickers_sml, textposition="top center", textfont=dict(size=9),
                    name="Activos",
                    customdata=alphas_v,
                    hovertemplate="<b>%{text}</b><br>Beta=%{x:.2f}<br>Ret=%{y:.2f}%<br>Alpha=%{customdata:+.2f}%<extra></extra>"))
                fig_sml.add_hline(y=rf_p, line=dict(color="#6b7c93", dash="dot"),
                                  annotation_text=f"Rf={rf_p:.2f}%")
                fig_sml.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(232,240,248,0.2)",
                    xaxis_title="Beta", yaxis_title="Retorno Esperado Anual (%)",
                    margin=dict(l=10,r=10,t=10,b=10),
                    legend=dict(orientation="h", y=1.05, font_size=9))
                st.markdown('<div class="chart3d">', unsafe_allow_html=True)
                st.plotly_chart(fig_sml, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Tabla Betas + Alphas
                sml_df = pd.DataFrame({
                    "Activo": tickers_sml,
                    "Beta": [f"{betas[t]:.2f}" for t in tickers_sml],
                    "Alpha CAPM": [f"{alphas_c.get(t,0)*100:+.2f}%" for t in tickers_sml],
                    "Ret. Esperado": [f"{mu_ann_m.get(t,0)*100:.2f}%" for t in tickers_sml],
                })
                rows_sml = ""
                for _, r in sml_df.iterrows():
                    ac = "#1a7a4a" if "+" in r["Alpha CAPM"] else "#c0392b"
                    rows_sml += (f"<tr><td><b>{r['Activo']}</b></td><td>{r['Beta']}</td>"
                                f"<td style='color:{ac};font-weight:700;'>{r['Alpha CAPM']}</td>"
                                f"<td>{r['Ret. Esperado']}</td></tr>")
                st.markdown(
                    f"<div class='table-wrap'><table class='styled-table'>"
                    f"<thead><tr><th>Activo</th><th>Beta</th><th>Alpha</th><th>Retorno Esp.</th></tr></thead>"
                    f"<tbody>{rows_sml}</tbody></table></div>", unsafe_allow_html=True)

    with tab2:
        corr = model_result.get("corr_matrix")
        tickers_m = model_result.get("tickers", [])
        if corr is not None and len(tickers_m) > 0:
            fig_corr = go.Figure(go.Heatmap(
                z=corr, x=tickers_m, y=tickers_m,
                colorscale="RdBu_r", zmin=-1, zmax=1,
                text=[[f"{corr[i][j]:.2f}" for j in range(len(tickers_m))] for i in range(len(tickers_m))],
                texttemplate="%{text}", textfont=dict(size=10),
                hovertemplate="<b>%{x} vs %{y}</b><br>Corr: %{z:.3f}<extra></extra>"
            ))
            fig_corr.update_layout(
                title="G1 — Matriz de Correlación (Rendimientos Logarítmicos Mensuales)",
                height=450, paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=50, b=10)
            )
            st.markdown('<div class="chart3d">', unsafe_allow_html=True)
            st.plotly_chart(fig_corr, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(
                "<div style='font-size:11px;color:#6b7c93;line-height:1.6;'>"
                "<b>Cómo leer esto:</b> Valores cercanos a +1 (rojo) = movimiento conjunto — "
                "menor diversificación. Valores cercanos a -1 (azul) = movimiento opuesto — "
                "mayor diversificación. El modelo Markowitz selecciona activos con baja correlación "
                "entre sí para reducir el riesgo del portafolio.</div>", unsafe_allow_html=True)

    with tab3:
        sim_data = model_result.get("sim_annual", [])
        if sim_data:
            sim_arr = np.array(sim_data) * 100
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(
                x=sim_arr, nbinsx=60,
                marker_color="#3b82f6", opacity=0.75, name="Distribución",
                hovertemplate="Retorno: %{x:.1f}%<br>Frecuencia: %{y}<extra></extra>"
            ))
            p5, p25, p50, p75, p95 = (model_result.get(k, 0) * 100 for k in
                                       ["mc_p5", "mc_p25", "mc_p50", "mc_p75", "mc_p95"])
            for pval, pname, pc_col in [(p5,"P5","#c0392b"),(p50,"Mediana","#c9a227"),(p95,"P95","#1a7a4a")]:
                fig_mc.add_vline(x=pval, line=dict(color=pc_col, dash="dash", width=2),
                                annotation_text=f"{pname}: {pval:.1f}%",
                                annotation_position="top")
            fig_mc.update_layout(
                title="Distribución de Rendimientos Anuales (MC 10,000 simulaciones)",
                xaxis_title="Rendimiento Anual (%)", yaxis_title="Frecuencia",
                height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(232,240,248,0.2)",
                margin=dict(l=10,r=10,t=50,b=10)
            )
            st.markdown('<div class="chart3d">', unsafe_allow_html=True)
            st.plotly_chart(fig_mc, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            prob_loss = model_result.get("mc_prob_loss", 0)
            st.markdown(f"""
            <div style='display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-top:10px;'>
                <div style='background:white;border-radius:10px;padding:12px;text-align:center;border-top:3px solid #c0392b;'>
                    <div style='font-size:9px;color:#6b7c93;'>P5 (peor 5%)</div>
                    <div style='font-size:18px;font-weight:700;color:#c0392b;'>{p5:.1f}%</div>
                </div>
                <div style='background:white;border-radius:10px;padding:12px;text-align:center;border-top:3px solid #d4820a;'>
                    <div style='font-size:9px;color:#6b7c93;'>P25</div>
                    <div style='font-size:18px;font-weight:700;color:#d4820a;'>{p25:.1f}%</div>
                </div>
                <div style='background:white;border-radius:10px;padding:12px;text-align:center;border-top:3px solid #c9a227;'>
                    <div style='font-size:9px;color:#6b7c93;'>Mediana</div>
                    <div style='font-size:18px;font-weight:700;color:#c9a227;'>{p50:.1f}%</div>
                </div>
                <div style='background:white;border-radius:10px;padding:12px;text-align:center;border-top:3px solid #2d7a4a;'>
                    <div style='font-size:9px;color:#6b7c93;'>P75</div>
                    <div style='font-size:18px;font-weight:700;color:#2d7a4a;'>{p75:.1f}%</div>
                </div>
                <div style='background:white;border-radius:10px;padding:12px;text-align:center;border-top:3px solid #1a7a4a;'>
                    <div style='font-size:9px;color:#6b7c93;'>P95 (mejor 5%)</div>
                    <div style='font-size:18px;font-weight:700;color:#1a7a4a;'>{p95:.1f}%</div>
                </div>
            </div>
            <div style='margin-top:10px;font-size:12px;color:#6b7c93;'>
                Probabilidad de pérdida anual: <b style='color:{"#c0392b" if prob_loss>0.3 else "#1a7a4a"};'>{prob_loss*100:.1f}%</b>
            </div>""", unsafe_allow_html=True)

    with tab4:
        var_data = model_result.get("var_by_asset", {})
        cvar_data = model_result.get("cvar_by_asset", {})
        if var_data:
            tkrs_v = list(var_data.keys())
            fig_var = go.Figure()
            fig_var.add_trace(go.Bar(x=tkrs_v, y=[var_data[t]*100 for t in tkrs_v],
                name="VaR 95%", marker_color="#ef4444", opacity=0.8))
            fig_var.add_trace(go.Bar(x=tkrs_v, y=[cvar_data.get(t,0)*100 for t in tkrs_v],
                name="CVaR 95%", marker_color="#7f1d1d", opacity=0.8))
            fig_var.update_layout(
                title="G13 — VaR y CVaR al 95% por Activo (simulado)",
                barmode="group", xaxis_title="Activos", yaxis_title="Pérdida (%)",
                height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(232,240,248,0.2)",
                margin=dict(l=10,r=10,t=50,b=10)
            )
            st.markdown('<div class="chart3d">', unsafe_allow_html=True)
            st.plotly_chart(fig_var, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # VaR frontera
            vf = model_result.get("var_frontier", [])
            cf = model_result.get("cvar_frontier", [])
            fp_vols_f = [pt["vol"]*100 for pt in model_result.get("frontier", [])[:len(vf)]]
            if vf and fp_vols_f:
                fig_vf = go.Figure()
                fig_vf.add_trace(go.Scatter(x=fp_vols_f, y=[v*100 for v in vf],
                    mode="lines+markers", line=dict(color="#ef4444", width=2),
                    marker=dict(size=6), name="VaR 95%"))
                fig_vf.add_trace(go.Scatter(x=fp_vols_f, y=[v*100 for v in cf],
                    mode="lines+markers", line=dict(color="#7f1d1d", width=2),
                    marker=dict(size=6), name="CVaR 95%"))
                fig_vf.update_layout(
                    title="G14 — VaR y CVaR a lo largo de la Frontera Eficiente",
                    xaxis_title="Riesgo del Portafolio (%)", yaxis_title="Pérdida (%)",
                    height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(232,240,248,0.2)",
                    margin=dict(l=10,r=10,t=50,b=10)
                )
                st.markdown('<div class="chart3d">', unsafe_allow_html=True)
                st.plotly_chart(fig_vf, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ─── PAGE: USUARIOS (admin) ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Usuarios" and is_admin:
    st.markdown("<div class='section-title'>👥 Gestión de Usuarios</div>", unsafe_allow_html=True)

    if "admin_users" not in st.session_state:
        st.session_state.admin_users = {k: dict(v) for k, v in USERS.items()}

    all_ports_names = list(get_all_portfolios().keys())

    for ukey, udata_u in list(st.session_state.admin_users.items()):
        with st.expander(f"{udata_u.get('avatar','?')} {udata_u.get('name', ukey)} — {udata_u.get('role','investor').upper()}"):
            eu1, eu2, eu3 = st.columns(3)
            with eu1:
                new_name = st.text_input("Nombre", value=udata_u.get("name",""), key=f"un_{ukey}")
                new_pin = st.text_input("PIN (4 dígitos)", value=udata_u.get("pin",""), key=f"up_{ukey}", max_chars=4)
                new_email = st.text_input("Email (alertas)", value=udata_u.get("email",""), key=f"ue_{ukey}")
            with eu2:
                new_avatar = st.text_input("Avatar (2 letras)", value=udata_u.get("avatar",""), key=f"ua_{ukey}", max_chars=2)
                new_color = st.color_picker("Color", value=udata_u.get("color","#1a3a5c"), key=f"uc_{ukey}")
                new_role = st.selectbox("Rol", ["investor","admin"],
                                        index=0 if udata_u.get("role")!="admin" else 1, key=f"ur_{ukey}")
            with eu3:
                curr_ports = udata_u.get("portfolios")
                sel_ports = st.multiselect("Portafolios", all_ports_names,
                                           default=all_ports_names if curr_ports is None else curr_ports,
                                           key=f"uprt_{ukey}")
                all_access = st.checkbox("Acceso a todos", value=(curr_ports is None), key=f"uall_{ukey}")

            uc1, uc2 = st.columns([1, 5])
            with uc1:
                if st.button("💾 Actualizar", key=f"upd_{ukey}"):
                    st.session_state.admin_users[ukey].update({
                        "name": new_name, "pin": new_pin, "avatar": new_avatar.upper(),
                        "color": new_color, "role": new_role, "email": new_email,
                        "portfolios": None if all_access else sel_ports,
                    })
                    st.success("Actualizado ✓")
            with uc2:
                if ukey != current_user and st.button("🗑 Eliminar", key=f"del_{ukey}"):
                    del st.session_state.admin_users[ukey]
                    st.rerun()

    st.divider()
    # Nuevo usuario
    with st.expander("➕ Agregar Nuevo Usuario"):
        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            n_key = st.text_input("Username (clave)", key="n_key")
            n_name = st.text_input("Nombre completo", key="n_name")
            n_pin = st.text_input("PIN (4 dígitos)", max_chars=4, key="n_pin")
        with nc2:
            n_avatar = st.text_input("Avatar (2 letras)", max_chars=2, key="n_avatar")
            n_color = st.color_picker("Color", "#1a7a4a", key="n_color")
            n_role = st.selectbox("Rol", ["investor","admin"], key="n_role")
        with nc3:
            n_ports = st.multiselect("Portafolios", all_ports_names, key="n_ports")
            n_email = st.text_input("Email", key="n_email")

        if st.button("➕ Crear Usuario", key="btn_add_user"):
            errs = []
            if not n_key.strip(): errs.append("Username requerido")
            if n_key in st.session_state.admin_users: errs.append("Username ya existe")
            if len(n_pin) != 4: errs.append("PIN debe ser 4 dígitos")
            if errs:
                for e in errs: st.error(e)
            else:
                st.session_state.admin_users[n_key.strip()] = {
                    "name": n_name, "pin": n_pin, "avatar": n_avatar.upper(),
                    "color": n_color, "sheet_id": "", "role": n_role,
                    "portfolios": None if not n_ports else n_ports, "email": n_email,
                }
                st.success(f"Usuario '{n_name}' creado.")
                st.rerun()

    st.divider()
    sc1, sc2 = st.columns([1, 3])
    with sc1:
        if st.button("💾 Guardar todos los cambios", type="primary", use_container_width=True):
            USERS.clear()
            USERS.update(st.session_state.admin_users)
            ok, msg = write_users_to_file(USERS)
            if ok:
                st.success(f"✅ {msg} — reinicia la app para que surta efecto")
            else:
                st.error(f"❌ {msg}")
    with sc2:
        st.markdown("<div style='padding:10px 0;font-size:12px;color:#6b7c93;'>"
                    "Guarda para persistir cambios en app.py. Los cambios de PIN toman efecto al reiniciar.</div>",
                    unsafe_allow_html=True)

    # ── Crear nuevo portafolio ────────────────────────────────────────────────
    st.divider()
    with st.expander("➕ Crear Nuevo Portafolio"):
        st.caption(
            "El nuevo portafolio leerá Precios_NOMBRE y Ops_NOMBRE del Sheet configurado. "
            "Opera como universo completamente independiente.")
        _pc1, _pc2, _pc3 = st.columns([2, 2, 1])
        with _pc1: _new_pn = st.text_input("Nombre (sin espacios)", placeholder="Especulativo", key="new_pname")
        with _pc2: _new_pd = st.text_input("Descripción", placeholder="Perfil Especulativo", key="new_pdesc")
        with _pc3: _new_pc = st.color_picker("Color", "#6366f1", key="new_pcolor")
        _new_pu = st.text_input("Tickers del universo (coma-separados)", placeholder="NVDA, TSLA, META", key="new_puniv")
        _new_pw = st.text_input("Pesos % semilla (deben sumar 100)", placeholder="40, 35, 25", key="new_pwts")

        if st.button("➕ Crear Portafolio", key="btn_create_port"):
            _n = _new_pn.strip().replace(" ", "")
            _u = [t.strip().upper() for t in _new_pu.split(",") if t.strip()]
            _wr = [w.strip() for w in _new_pw.split(",") if w.strip()]
            if not _n: st.error("Nombre requerido.")
            elif _n in get_all_portfolios(): st.error(f"Ya existe '{_n}'.")
            elif not _u: st.error("Agrega al menos 2 tickers.")
            else:
                try:
                    _wf = [float(w) for w in _wr] if _wr else [100/len(_u)] * len(_u)
                    _ws = sum(_wf)
                    _wn = {t: w/_ws for t, w in zip(_u, _wf)}
                    st.session_state.custom_portfolios[_n] = {
                        "description": _new_pd.strip() or _n,
                        "color": _new_pc, "rf": 0.0264, "profile": "personalizado",
                        "universe": _u, "target": _wn,
                        "objetivo": "Portafolio personalizado",
                        "horizonte": "A definir", "tolerancia": "A definir",
                        "tolerancia_pct": 50, "horizonte_years": 5,
                        "volatilidad_range": (15, 30), "retorno_obj": (15, 35),
                        "sharpe_obj": 1.0, "icon": "📁",
                        "bg": "linear-gradient(135deg,#f4f7fb,#e8f0f8)",
                        "border": _new_pc,
                    }
                    st.success(f"✅ Portafolio '{_n}' creado. Agrega hojas Precios_{_n} y Ops_{_n} en tu Sheet.")
                    st.rerun()
                except Exception as _ex:
                    st.error(f"Error: {_ex}")

        # Lista portafolios custom
        for _pn, _pd in list(st.session_state.get("custom_portfolios", {}).items()):
            _pp1, _pp2 = st.columns([5, 1])
            with _pp1:
                st.markdown(
                    f"<div style='background:#f0f4f8;border-left:4px solid {_pd['color']};"
                    f"border-radius:6px;padding:7px 12px;font-size:13px;'>"
                    f"<b>{_pn}</b> — {_pd.get('description','')} · {len(_pd.get('universe',[]))} tickers</div>",
                    unsafe_allow_html=True)
            with _pp2:
                if st.button("🗑", key=f"del_p_{_pn}"):
                    del st.session_state.custom_portfolios[_pn]
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ─── PAGE: QUICKVIEW (admin) ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔭 QuickView" and is_admin:
    st.markdown("<div class='section-title'>🔭 QuickView — Todos los Usuarios</div>",
                unsafe_allow_html=True)

    all_ports_qv = get_all_portfolios()
    _qv_opts = ["Todos"] + [v.get("name", k) for k, v in USERS.items()]
    _qv_sel = st.selectbox("Filtrar usuario", _qv_opts, key="qv_sel")
    _qv_fk = None if _qv_sel == "Todos" else next(
        (k for k, v in USERS.items() if v.get("name", k) == _qv_sel), None)
    _qv_list = [(k, v) for k, v in USERS.items() if _qv_fk is None or k == _qv_fk]

    for uname, udata_qv in _qv_list:
        u_ports = udata_qv.get("portfolios") or list(all_ports_qv.keys())
        if not u_ports:
            continue
        st.markdown(
            f"<div style='background:#1a3a5c;color:white;border-radius:10px;padding:10px 16px;margin-top:18px;'>"
            f"<b>{udata_qv.get('name',uname)}</b> · {udata_qv.get('role','investor').upper()} · {len(u_ports)} portafolio(s)</div>",
            unsafe_allow_html=True)

        for pname in u_ports:
            _pdata = all_ports_qv.get(pname, {})
            _uid = udata_qv.get("sheet_id","").strip() or sheet_id
            _uops = pd.DataFrame()
            if _uid and use_sheets:
                try:
                    _raw = load_sheet(_uid, f"Ops_{pname}")
                    if not _raw.empty:
                        if "Usuario" in _raw.columns:
                            _un = udata_qv.get("name", uname).lower()
                            _uc = _raw["Usuario"].astype(str).str.strip().str.lower()
                            _mk = ((_uc == uname.lower()) | (_uc == _un) | (_uc == _un.split()[0]))
                            _f = _raw[_mk].copy()
                            _uops = _f if not _f.empty else _raw
                        else:
                            _uops = _raw
                except Exception:
                    pass
            if _uops.empty:
                _uops = get_demo_ops(pname)

            _u_tgt = _pdata.get("target", {})
            _u_pos = calc_positions(_uops, get_demo_prices(), tc_actual)
            _tot_usd = _u_pos["Valor USD"].sum() if not _u_pos.empty else 0
            _tot_mxn = _tot_usd * tc_actual
            _gain_usd = _u_pos["Ganancia USD"].sum() if not _u_pos.empty else 0
            _cost = _tot_usd - _gain_usd
            _rend_pct = (_gain_usd / _cost * 100) if _cost > 0 else 0
            _rc = "#1a7a4a" if _rend_pct >= 0 else "#c0392b"
            _rs = "+" if _rend_pct >= 0 else ""
            _wts_str = " · ".join([f"{t} {w*100:.0f}%" for t, w in list(_u_tgt.items())[:4]])

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"<div class='kpi-card' style='padding:12px;'>"
                           f"<div class='kpi-label'>📁 {pname}</div>"
                           f"<div class='kpi-value' style='font-size:20px;'>${_tot_mxn:,.0f}</div>"
                           f"<div class='kpi-sub'>MXN</div></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='kpi-card' style='padding:12px;'>"
                           f"<div class='kpi-label'>📈 Rendimiento</div>"
                           f"<div class='kpi-value' style='font-size:20px;color:{_rc};'>{_rs}{_rend_pct:.1f}%</div>"
                           f"<div class='kpi-sub'>Desde costo base</div></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='kpi-card' style='padding:12px;'>"
                           f"<div class='kpi-label'>📋 Operaciones</div>"
                           f"<div class='kpi-value' style='font-size:20px;'>{len(_uops)}</div>"
                           f"<div class='kpi-sub'>registradas</div></div>", unsafe_allow_html=True)
            with c4:
                st.markdown(f"<div class='kpi-card' style='padding:12px;'>"
                           f"<div class='kpi-label'>🎯 Pesos objetivo</div>"
                           f"<div style='font-size:10px;color:#1a3a5c;line-height:1.8;margin-top:4px;'>{_wts_str}</div>"
                           f"</div>", unsafe_allow_html=True)
