# =============================================================================
# ANÁLISIS COMPLETO DE PORTAFOLIO — MÓDULO PYTHON
# Autor original: traducido desde MATLAB
# Incluye: Frontera eficiente, Tangencia, SML, VaR, Correlación, CAPM, Monte Carlo
# Restricción: Máximo 30% por activo (configurable)
#
# USO COMO MÓDULO (Streamlit / app.py):
#   from PORT_MAT_DEEP_PYTHON import run_complete_analysis
#   results = run_complete_analysis(prices_df, rf=0.0264)
#
# USO STANDALONE (línea de comandos):
#   python PORT_MAT_DEEP_PYTHON.py
#   → Lee precios.xlsx, genera gráficas y muestra resultados en consola
# =============================================================================

import pandas as pd
import numpy as np
import warnings
from scipy.optimize import minimize
from scipy.stats import norm

warnings.filterwarnings('ignore')


# =============================================================================
# FUNCIÓN PRINCIPAL — IMPORTABLE DESDE STREAMLIT
# =============================================================================
def run_complete_analysis(prices_df, rf=None, max_peso=0.30):
    """
    Ejecuta el análisis completo de portafolio Markowitz.

    Parámetros
    ----------
    prices_df : pd.DataFrame
        DataFrame con fechas como índice y columnas = tickers (valores = precios).
        Debe incluir una columna cuyo nombre contenga 'SPX' como benchmark de mercado.
    rf : float, opcional
        Tasa libre de riesgo anual (ej. 0.0428 = 4.28 %).
        Si es None se intenta obtener de Yahoo Finance (^TNX); si falla usa 0.0428.
    max_peso : float
        Límite máximo de peso por activo en la optimización (default 0.30 = 30 %).

    Retorna
    -------
    dict con las claves:
        AssetList, AssetMean, AssetCovar,
        w_tang, ret_t, risk_t, sharpe_t,
        w_minvar, ret_minvar, risk_minvar,
        frontier_returns, frontier_risks, frontier_weights,
        corr_matrix, Betas, Alfa, CAPM, Rm_promedio, MRP, rf, idx_spx,
        ret_p5, ret_p25, ret_p50, ret_p75, ret_p95, sim_returns,
        tangencia (dict ticker→peso), min_var (dict ticker→peso),
        exp_rets (dict ticker→retorno esperado anualizado),
        sharpe_tangencia, vol_tang, tickers, cov_matrix,
        prob_perdida, sim_returns_pct
    """
    # ── Imports locales (seguros en entorno Streamlit) ──────────────────────
    try:
        import yfinance as yf
        _yf_ok = True
    except ImportError:
        _yf_ok = False

    # ── Validación de entrada ───────────────────────────────────────────────
    prices_df = prices_df.copy()
    # Eliminar filas con cualquier NaN
    prices_df = prices_df.dropna()
    if len(prices_df) < 13:
        raise ValueError(
            f"Se requieren al menos 13 filas de precios para calcular 12 retornos mensuales. "
            f"Se recibieron {len(prices_df)}."
        )

    # ── Localizar SPX (benchmark) ───────────────────────────────────────────
    AssetList = prices_df.columns.tolist()
    nAssets   = len(AssetList)
    idx_spx   = None
    for i, nombre in enumerate(AssetList):
        if 'SPX' in str(nombre).upper():
            idx_spx = i
            break
    if idx_spx is None:
        raise ValueError(
            "El DataFrame debe incluir una columna llamada 'SPX' (o que contenga 'SPX') "
            "como benchmark de mercado para calcular betas y la SML."
        )

    Precios = prices_df.values.astype(float)

    # ── Rendimientos logarítmicos mensuales ─────────────────────────────────
    Rendimientos = np.diff(np.log(Precios), axis=0)   # shape: (n-1, nAssets)
    num_obs      = Rendimientos.shape[0]

    # Estadísticos anualizados (asume datos mensuales)
    AssetMean  = np.mean(Rendimientos, axis=0) * 12
    AssetCovar = np.cov(Rendimientos, rowvar=False) * 12

    # ── Tasa libre de riesgo ────────────────────────────────────────────────
    if rf is None:
        if _yf_ok:
            try:
                tnx     = yf.Ticker("^TNX")
                rf_data = tnx.history(period="1d")['Close']
                rf      = float(rf_data.iloc[-1]) / 100.0 if not rf_data.empty else 0.0428
            except Exception:
                rf = 0.0428
        else:
            rf = 0.0428

    # ── Funciones auxiliares internas ───────────────────────────────────────
    def port_stats(weights):
        ret  = float(np.dot(weights, AssetMean))
        risk = float(np.sqrt(weights @ AssetCovar @ weights))
        return ret, risk

    def neg_sharpe(weights):
        ret, risk = port_stats(weights)
        return -(ret - rf) / risk if risk > 1e-9 else 0.0

    def port_volatility(weights):
        return float(np.sqrt(weights @ AssetCovar @ weights))

    constraints_sum = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds          = tuple((0, max_peso) for _ in range(nAssets))
    init_guess      = np.ones(nAssets) / nAssets

    # ── Portafolio de Tangencia (máximo Sharpe) ─────────────────────────────
    opt_tang = minimize(
        neg_sharpe, init_guess, method='SLSQP',
        bounds=bounds, constraints=constraints_sum,
        options={'ftol': 1e-10, 'maxiter': 1000}
    )
    w_tang        = np.clip(opt_tang.x, 0, 1)
    w_tang       /= w_tang.sum()
    ret_t, risk_t = port_stats(w_tang)
    sharpe_t      = (ret_t - rf) / risk_t if risk_t > 1e-9 else 0.0

    # ── Portafolio de Mínima Varianza ───────────────────────────────────────
    opt_mv   = minimize(
        port_volatility, init_guess, method='SLSQP',
        bounds=bounds, constraints=constraints_sum,
        options={'ftol': 1e-10, 'maxiter': 1000}
    )
    w_mv            = np.clip(opt_mv.x, 0, 1)
    w_mv           /= w_mv.sum()
    ret_mv, risk_mv = port_stats(w_mv)

    # ── Frontera eficiente (20 puntos) ──────────────────────────────────────
    numPorts       = 20
    target_returns = np.linspace(float(AssetMean.min()), float(AssetMean.max()), numPorts)
    frontier_risk, frontier_return, frontier_weights = [], [], []

    for ret_target in target_returns:
        cons = [
            constraints_sum,
            {'type': 'eq',
             'fun':  lambda x, r=ret_target: float(np.dot(x, AssetMean)) - r}
        ]
        opt = minimize(
            port_volatility, init_guess, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'ftol': 1e-9, 'maxiter': 500}
        )
        if opt.success:
            w_f = np.clip(opt.x, 0, 1); w_f /= w_f.sum()
            r_f, v_f = port_stats(w_f)
            frontier_return.append(r_f)
            frontier_risk.append(v_f)
            frontier_weights.append(w_f)

    frontier_return  = np.array(frontier_return)
    frontier_risk    = np.array(frontier_risk)
    frontier_weights = np.array(frontier_weights) if frontier_weights else np.empty((0, nAssets))

    # ── Matriz de correlación ───────────────────────────────────────────────
    corr_matrix = np.corrcoef(Rendimientos, rowvar=False)
    # Asegurar diagonal unitaria exacta
    np.fill_diagonal(corr_matrix, 1.0)

    # ── SML: Betas y Alphas (regresión contra SPX) ──────────────────────────
    Rm      = Rendimientos[:, idx_spx]   # retornos mensuales del mercado
    Betas   = np.zeros(nAssets)
    for i in range(nAssets):
        cov_i_m  = np.cov(Rendimientos[:, i], Rm)[0, 1]
        var_m    = float(np.var(Rm))
        Betas[i] = cov_i_m / var_m if var_m > 1e-12 else 1.0

    Rm_promedio = float(np.mean(Rm)) * 12   # retorno anualizado del mercado
    MRP         = Rm_promedio - rf           # prima de riesgo de mercado
    CAPM        = rf + Betas * MRP           # retorno requerido por CAPM
    Alfa        = AssetMean - CAPM           # alpha de Jensen (exceso sobre CAPM)

    # ── Monte Carlo: distribución del retorno anual del portafolio Tang. ────
    np.random.seed(42)
    nSim_mc      = 10_000
    mean_mensual = AssetMean / 12
    cov_mensual  = AssetCovar / 12
    sim_returns  = np.zeros(nSim_mc)
    for i in range(nSim_mc):
        sim_ret      = np.random.multivariate_normal(mean_mensual, cov_mensual, size=12)
        sim_returns[i] = float(np.prod(1.0 + sim_ret @ w_tang) - 1.0)

    ret_p5  = float(np.percentile(sim_returns, 5))  * 100
    ret_p25 = float(np.percentile(sim_returns, 25)) * 100
    ret_p50 = float(np.percentile(sim_returns, 50)) * 100
    ret_p75 = float(np.percentile(sim_returns, 75)) * 100
    ret_p95 = float(np.percentile(sim_returns, 95)) * 100
    prob_perdida = float(np.sum(sim_returns < 0) / nSim_mc * 100)

    # ── Diccionarios de pesos (formato ticker→peso) ─────────────────────────
    tangencia_dict = {t: float(w) for t, w in zip(AssetList, w_tang)}
    minvar_dict    = {t: float(w) for t, w in zip(AssetList, w_mv)}
    exp_rets_dict  = {t: float(r) for t, r in zip(AssetList, AssetMean)}

    # ── Frontera en formato de lista de tuplas (compatibilidad con markowitz_optimize) ─
    frontier_list = []
    for i in range(len(frontier_return)):
        wf  = frontier_weights[i]
        frontier_list.append((
            float(frontier_risk[i]) * 100,
            float(frontier_return[i]) * 100,
            {t: float(wf[j]) for j, t in enumerate(AssetList)}
        ))

    # ── Resultado completo ───────────────────────────────────────────────────
    return {
        # Datos crudos
        'AssetList':         AssetList,
        'AssetMean':         AssetMean,          # retornos anualizados por activo
        'AssetCovar':        AssetCovar,          # covarianza anualizada
        'num_obs':           num_obs,

        # Portafolio tangencia
        'w_tang':            w_tang,              # array de pesos
        'ret_t':             ret_t,               # retorno anual
        'risk_t':            risk_t,              # volatilidad anual
        'sharpe_t':          sharpe_t,

        # Portafolio mínima varianza
        'w_minvar':          w_mv,
        'ret_minvar':        ret_mv,
        'risk_minvar':       risk_mv,

        # Frontera eficiente
        'frontier_returns':  frontier_return,     # array
        'frontier_risks':    frontier_risk,       # array
        'frontier_weights':  frontier_weights,    # array 2D

        # Correlación y riesgo
        'corr_matrix':       corr_matrix,
        'idx_spx':           idx_spx,

        # CAPM / SML (betas y alphas calculados desde datos reales)
        'Betas':             Betas,
        'Alfa':              Alfa,
        'CAPM':              CAPM,
        'Rm_promedio':       Rm_promedio,
        'MRP':               MRP,
        'rf':                rf,

        # Monte Carlo
        'ret_p5':            ret_p5,
        'ret_p25':           ret_p25,
        'ret_p50':           ret_p50,
        'ret_p75':           ret_p75,
        'ret_p95':           ret_p95,
        'sim_returns':       sim_returns,
        'sim_returns_pct':   sim_returns * 100,
        'prob_perdida':      prob_perdida,

        # Formatos compatibles con markowitz_optimize legacy
        'tickers':           AssetList,
        'tangencia':         tangencia_dict,
        'min_var':           minvar_dict,
        'exp_rets':          exp_rets_dict,
        'frontier':          frontier_list,       # list of (vol%, ret%, {ticker:w})
        'cov_matrix':        AssetCovar,
        'sharpe_tangencia':  sharpe_t,
        'vol_tang':          risk_t,
        'ret_tang':          ret_t,
        'vol_minvar':        risk_mv,
        'ret_minvar_scalar': ret_mv,
    }


# =============================================================================
# EJECUCIÓN STANDALONE — solo cuando se corre directamente con Python
# =============================================================================
# =============================================================================
# ── UTILIDADES STANDALONE: conexión a Google Sheets sin Streamlit ─────────────
# =============================================================================

# Mapeo de nombres Bloomberg → ticker corto (mismo que en app.py)
_BLOOMBERG_TO_TICKER = {
    "AAPL US Equity": "AAPL",  "ASTS US Equity": "ASTS",
    "JPM US Equity":  "JPM",   "9988 HK Equity": "9988HK",
    "CAT US Equity":  "CAT",   "NOW US Equity":  "NOW",
    "NVDA US Equity": "NVDA",  "NEM US Equity":  "NEM",
    "TSLA US Equity": "TSLA",  "BATS LN Equity": "BATS",
    "CIEN US Equity": "CIEN",  "EXPE US Equity": "EXPE",
    "EXPE US":        "EXPE",  "SPX Index":      "SPX",
    "SPX":            "SPX",
}


def _load_credentials():
    """
    Carga credenciales de Google en este orden de prioridad:
      1. .streamlit/secrets.toml  (mismo archivo que usa app.py)
      2. gcp_credentials.json     (alternativa para entornos CI/CD)
    Devuelve un objeto google.oauth2.service_account.Credentials o None.
    """
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    # ── Intento 1: .streamlit/secrets.toml ──────────────────────────────────
    import os, pathlib
    secrets_path = pathlib.Path(".streamlit") / "secrets.toml"
    if secrets_path.exists():
        try:
            # Python 3.11+ trae tomllib; versiones anteriores necesitan tomli
            try:
                import tomllib
                with open(secrets_path, "rb") as f:
                    secrets = tomllib.load(f)
            except ImportError:
                import tomli as tomllib          # pip install tomli
                with open(secrets_path, "rb") as f:
                    secrets = tomllib.load(f)

            creds_dict = secrets.get("gcp_service_account", {})
            if creds_dict:
                from google.oauth2.service_account import Credentials
                creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
                print("✅ Credenciales cargadas desde .streamlit/secrets.toml")
                return creds
        except Exception as e:
            print(f"⚠️  secrets.toml encontrado pero error al leer: {e}")

    # ── Intento 2: gcp_credentials.json ────────────────────────────────────
    json_path = pathlib.Path("gcp_credentials.json")
    if json_path.exists():
        try:
            from google.oauth2.service_account import Credentials
            creds = Credentials.from_service_account_file(str(json_path), scopes=SCOPES)
            print("✅ Credenciales cargadas desde gcp_credentials.json")
            return creds
        except Exception as e:
            print(f"⚠️  gcp_credentials.json encontrado pero error al leer: {e}")

    return None


def _get_sheet_id():
    """
    Obtiene el sheet_id en este orden:
      1. Variable de entorno  PORTFOLIO_SHEET_ID
      2. .streamlit/secrets.toml  → [sheet] id
      3. Constante hardcodeada al final de esta función (edita si es necesario)
    """
    import os, pathlib

    # ── Variable de entorno ─────────────────────────────────────────────────
    env_id = os.environ.get("PORTFOLIO_SHEET_ID", "").strip()
    if env_id:
        print(f"✅ Sheet ID desde variable de entorno: {env_id[:20]}...")
        return env_id

    # ── secrets.toml → sección [sheet] id ──────────────────────────────────
    secrets_path = pathlib.Path(".streamlit") / "secrets.toml"
    if secrets_path.exists():
        try:
            try:
                import tomllib
                with open(secrets_path, "rb") as f:
                    secrets = tomllib.load(f)
            except ImportError:
                import tomli as tomllib
                with open(secrets_path, "rb") as f:
                    secrets = tomllib.load(f)
            sid = secrets.get("sheet", {}).get("id", "").strip()
            if sid:
                print(f"✅ Sheet ID desde secrets.toml [sheet] id: {sid[:20]}...")
                return sid
            # Intentar también en gcp_service_account si algún usuario lo puso ahí
            sid = secrets.get("sheet_id", "").strip()
            if sid:
                return sid
        except Exception:
            pass

    # ── Hardcodeado como último recurso ─────────────────────────────────────
    # ✏️  Cambia este valor por el ID de tu hoja si los métodos anteriores fallan.
    # Lo encuentras en la URL: docs.google.com/spreadsheets/d/<SHEET_ID>/edit
    HARDCODED_SHEET_ID = ""   # ← pega aquí si es necesario

    if HARDCODED_SHEET_ID:
        print(f"✅ Sheet ID hardcodeado: {HARDCODED_SHEET_ID[:20]}...")
        return HARDCODED_SHEET_ID

    return None


def _fetch_prices_from_sheet(client, sheet_id, tab_name):
    """
    Lee un tab Precios_<portfolio> del Google Sheet.
    Devuelve pd.DataFrame con índice de fechas y columnas = tickers,
    o None si el tab no existe o tiene datos insuficientes.
    """
    try:
        sh   = client.open_by_key(sheet_id)
        ws   = sh.worksheet(tab_name)
        data = ws.get_all_values()
        if len(data) < 3:
            return None
        headers = data[0]
        rows    = data[1:]
        fechas, price_rows = [], []
        for row in rows:
            if not row or all(v == "" for v in row):
                continue
            fecha_str = str(row[0]).strip()
            if not fecha_str:
                continue
            fechas.append(fecha_str)
            vals = {}
            for j, col in enumerate(headers[1:], start=1):
                if j >= len(row):
                    continue
                tkr = _BLOOMBERG_TO_TICKER.get(col) or col.strip().upper()
                if not tkr:
                    continue
                try:
                    v = float(str(row[j]).replace(",", ".").replace(" ", ""))
                    if v > 0:
                        vals[tkr] = v
                except (ValueError, TypeError):
                    pass
            price_rows.append(vals)

        if len(price_rows) < 13:
            print(f"   ⚠️  {tab_name}: solo {len(price_rows)} filas (mínimo 13)")
            return None

        df = pd.DataFrame(price_rows, index=fechas)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=1, how="any")          # elimina columnas con NaN
        df.index = pd.to_datetime(df.index, dayfirst=True, errors="coerce")
        df = df.dropna(axis=0, how="any")           # elimina filas con fecha inválida
        df = df.sort_index()
        return df if not df.empty else None
    except Exception as e:
        print(f"   ⚠️  Error leyendo {tab_name}: {e}")
        return None


def _plot_portfolio(R, portfolio_name):
    """Genera las 5 gráficas estándar para un portafolio dado."""
    import matplotlib.pyplot as plt

    AssetList = R['AssetList']
    nAssets   = len(AssetList)
    colors    = plt.cm.tab10(np.linspace(0, 1, nAssets))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f'Análisis Markowitz — {portfolio_name}', fontsize=15, fontweight='bold')

    # ── G1: Frontera Eficiente + CML ────────────────────────────────────────
    ax1 = axes[0, 0]
    if len(R['frontier_risks']) > 0:
        ax1.plot(R['frontier_risks'] * 100, R['frontier_returns'] * 100,
                 'b-', linewidth=2.5, label='Frontera Eficiente')
    for i, t in enumerate(AssetList):
        vol_i = float(np.sqrt(R['AssetCovar'][i, i]))
        ax1.scatter(vol_i * 100, R['AssetMean'][i] * 100,
                    color=colors[i], s=100, zorder=5)
        ax1.annotate(t, (vol_i * 100, R['AssetMean'][i] * 100),
                     textcoords='offset points', xytext=(5, 3), fontsize=8)
    ax1.scatter(R['risk_t'] * 100, R['ret_t'] * 100,
                color='red', s=180, marker='*', zorder=10, label='Tangencia ⭐')
    ax1.scatter(R['risk_minvar'] * 100, R['ret_minvar'] * 100,
                color='navy', s=100, marker='D', zorder=10, label='Mín. Var. ◆')
    if R['risk_t'] > 0:
        vols_c = np.linspace(0, R['risk_t'] * 100 * 1.6, 80)
        slope  = (R['ret_t'] - R['rf']) / R['risk_t']
        ax1.plot(vols_c, R['rf'] * 100 + slope * vols_c,
                 'g--', linewidth=1.5, label='CML')
    ax1.set_xlabel('Volatilidad (%)'); ax1.set_ylabel('Retorno (%)')
    ax1.set_title('Frontera Eficiente & CML'); ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # ── G2: SML ─────────────────────────────────────────────────────────────
    ax2 = axes[0, 1]
    b_max = max(R['Betas']) if len(R['Betas']) > 0 else 2.0
    beta_range = np.linspace(0, b_max * 1.2, 200)
    ax2.plot(beta_range, R['rf'] * 100 + beta_range * R['MRP'] * 100,
             'b-', linewidth=2, label='SML')
    for i, t in enumerate(AssetList):
        clr = 'green' if R['Alfa'][i] >= 0 else 'red'
        ax2.scatter(R['Betas'][i], R['AssetMean'][i] * 100,
                    color=clr, s=100, zorder=5)
        ax2.annotate(t, (R['Betas'][i], R['AssetMean'][i] * 100),
                     textcoords='offset points', xytext=(5, 3), fontsize=8)
    ax2.set_xlabel('Beta (β)'); ax2.set_ylabel('Retorno esperado (%)')
    ax2.set_title('Security Market Line (SML)')
    ax2.grid(True, alpha=0.3); ax2.legend()

    # ── G3: Distribución Monte Carlo ────────────────────────────────────────
    ax3 = axes[0, 2]
    ax3.hist(R['sim_returns_pct'], bins=60, color='steelblue',
             alpha=0.7, edgecolor='white', density=True)
    ax3.axvline(R['ret_p5'],  color='red',  linestyle='--',
                label=f'VaR 95%: {R["ret_p5"]:.1f}%')
    ax3.axvline(R['ret_p50'], color='gold', linestyle='--',
                label=f'Mediana: {R["ret_p50"]:.1f}%')
    x_n   = np.linspace(R['sim_returns_pct'].min(), R['sim_returns_pct'].max(), 200)
    mu_n  = float(np.mean(R['sim_returns_pct']))
    sd_n  = float(np.std(R['sim_returns_pct']))
    ax3.plot(x_n, norm.pdf(x_n, mu_n, sd_n), 'r-', linewidth=2, label='Normal')
    ax3.set_xlabel('Retorno anual (%)'); ax3.set_ylabel('Densidad')
    ax3.set_title('Monte Carlo (10,000 sims)')
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # ── G4: Matriz de correlación ────────────────────────────────────────────
    ax4 = axes[1, 0]
    im = ax4.imshow(R['corr_matrix'], cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax4, label='ρ')
    ax4.set_xticks(range(nAssets)); ax4.set_yticks(range(nAssets))
    ax4.set_xticklabels(AssetList, rotation=45, ha='right', fontsize=8)
    ax4.set_yticklabels(AssetList, fontsize=8)
    for i in range(nAssets):
        for j in range(nAssets):
            ax4.text(j, i, f'{R["corr_matrix"][i, j]:.2f}',
                     ha='center', va='center', fontsize=7,
                     color='black' if abs(R['corr_matrix'][i, j]) < 0.7 else 'white')
    ax4.set_title('Matriz de Correlación')

    # ── G5: Composición tangencia (pastel) ───────────────────────────────────
    ax5 = axes[1, 1]
    wts_nz = {t: w for t, w in R['tangencia'].items() if w > 0.005}
    ax5.pie(list(wts_nz.values()), labels=list(wts_nz.keys()),
            autopct='%1.1f%%', startangle=90,
            colors=plt.cm.Set3(np.linspace(0, 1, len(wts_nz))))
    ax5.set_title('Pesos Tangencia')

    # ── G6: Tabla resumen ────────────────────────────────────────────────────
    ax6 = axes[1, 2]
    ax6.axis('off')
    resumen = [
        ['Retorno Tang.',   f'{R["ret_t"]*100:.2f}%'],
        ['Volatilidad',     f'{R["risk_t"]*100:.2f}%'],
        ['Sharpe Ratio',    f'{R["sharpe_t"]:.4f}'],
        ['Rf utilizada',    f'{R["rf"]*100:.2f}%'],
        ['MRP',             f'{R["MRP"]*100:.2f}%'],
        ['Ret. Mín.Var.',   f'{R["ret_minvar"]*100:.2f}%'],
        ['Vol. Mín.Var.',   f'{R["risk_minvar"]*100:.2f}%'],
        ['P50 MC',          f'{R["ret_p50"]:.1f}%'],
        ['P5 (VaR)',        f'{R["ret_p5"]:.1f}%'],
        ['Prob. pérdida',   f'{R["prob_perdida"]:.1f}%'],
    ]
    tbl = ax6.table(cellText=resumen, colLabels=['Métrica', 'Valor'],
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.1, 1.4)
    ax6.set_title('Resumen', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# =============================================================================
# ── EJECUCIÓN STANDALONE ──────────────────────────────────────────────────────
# Corre cuando haces:  python PORT_MAT_DEEP_PYTHON.py
# Lee precios directamente desde Google Sheets (Portfolio_Manager_Data).
# Descubre automáticamente todas las pestañas Precios_* y analiza cada una.
# No requiere ningún archivo Excel local.
# =============================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')   # usa 'Agg' si estás en un servidor sin pantalla
    import matplotlib.pyplot as plt
    import gspread

    print('=' * 60)
    print('  ANÁLISIS COMPLETO DE PORTAFOLIO — Standalone')
    print('  Fuente de datos: Google Sheets (Portfolio_Manager_Data)')
    print('=' * 60 + '\n')

    # ── 1. Obtener tasa libre de riesgo ──────────────────────────────────────
    try:
        import yfinance as yf
        tnx     = yf.Ticker("^TNX")
        rf_data = tnx.history(period="1d")['Close']
        rf_global = float(rf_data.iloc[-1]) / 100.0 if not rf_data.empty else 0.0428
        print(f'✅ Tasa libre de riesgo: {rf_global*100:.2f}% (^TNX)\n')
    except Exception as e:
        rf_global = 0.0428
        print(f'⚠️  No se pudo obtener Rf de mercado ({e}). Usando 4.28%\n')

    # ── 2. Conectar a Google Sheets ──────────────────────────────────────────
    creds = _load_credentials()
    if creds is None:
        print(
            '\n❌ No se encontraron credenciales de Google.\n'
            '   Crea .streamlit/secrets.toml con la sección [gcp_service_account]\n'
            '   o coloca gcp_credentials.json en este directorio.\n'
            '   Ver README para instrucciones detalladas.'
        )
        raise SystemExit(1)

    client = gspread.authorize(creds)

    # ── 3. Obtener Sheet ID ──────────────────────────────────────────────────
    sheet_id = _get_sheet_id()
    if not sheet_id:
        print(
            '\n❌ No se encontró el Sheet ID.\n'
            '   Opciones:\n'
            '     a) Agrega en .streamlit/secrets.toml:\n'
            '        [sheet]\n'
            '        id = "1eApNRcJSnqYYkUxK2uDWqUOwXh6lNkoOZ-zvzVSoKFw"\n'
            '     b) Exporta la variable de entorno:\n'
            '        export PORTFOLIO_SHEET_ID="1eApNRcJSnqYYkUxK2uDWqUOwXh6lNkoOZ-zvzVSoKFw"\n'
            '     c) Edita HARDCODED_SHEET_ID en _get_sheet_id() dentro de este archivo.'
        )
        raise SystemExit(1)

    # ── 4. Descubrir automáticamente todos los tabs Precios_* ────────────────
    print(f'📡 Conectando a Google Sheets...')
    try:
        sh         = client.open_by_key(sheet_id)
        all_tabs   = [ws.title for ws in sh.worksheets()]
        price_tabs = [t for t in all_tabs if t.startswith('Precios_')]
    except Exception as e:
        print(f'❌ No se pudo abrir la hoja: {e}')
        raise SystemExit(1)

    if not price_tabs:
        print(
            '❌ No se encontraron pestañas con el prefijo "Precios_" en la hoja.\n'
            f'   Pestañas disponibles: {all_tabs}'
        )
        raise SystemExit(1)

    print(f'✅ Hoja abierta: "{sh.title}"')
    print(f'   Pestañas de precios encontradas: {price_tabs}\n')

    # ── 5. Analizar cada portafolio ──────────────────────────────────────────
    resultados = {}   # portfolio_name → dict de resultados
    figs       = []   # figuras para mostrar al final

    for tab in price_tabs:
        portfolio_name = tab.replace('Precios_', '')
        print(f'{"─"*55}')
        print(f'📂 Portafolio: {portfolio_name}  (tab: {tab})')

        prices_df = _fetch_prices_from_sheet(client, sheet_id, tab)

        if prices_df is None:
            print(f'   ⚠️  Datos insuficientes o error. Saltando.\n')
            continue

        print(f'   Período : {prices_df.index[0].strftime("%b %Y")} → '
              f'{prices_df.index[-1].strftime("%b %Y")}  '
              f'({len(prices_df)} observaciones)')
        print(f'   Activos : {list(prices_df.columns)}')

        # Verificar SPX
        has_spx = any('SPX' in c.upper() for c in prices_df.columns)
        if not has_spx:
            print(f'   ❌ No se encontró columna SPX. '
                  f'Agrega el índice S&P500 a la pestaña {tab}. Saltando.')
            continue

        # Correr análisis completo
        try:
            R = run_complete_analysis(
                prices_df.dropna(axis=1),
                rf=rf_global,
                max_peso=0.30
            )
            resultados[portfolio_name] = R
        except Exception as e:
            print(f'   ❌ Error en análisis: {e}. Saltando.')
            continue

        # Imprimir resumen en consola
        print(f'\n   ┌── PORTAFOLIO DE TANGENCIA ──────────────────────┐')
        print(f'   │  Retorno esperado : {R["ret_t"]*100:7.2f}%                  │')
        print(f'   │  Volatilidad      : {R["risk_t"]*100:7.2f}%                  │')
        print(f'   │  Sharpe Ratio     : {R["sharpe_t"]:7.4f}                  │')
        print(f'   │  Composición:                                    │')
        for t, w in sorted(R['tangencia'].items(), key=lambda x: -x[1]):
            if w > 0.005:
                bar = '█' * int(w * 30)
                print(f'   │    {t:<10s} {w*100:5.1f}%  {bar:<30s}│')
        print(f'   └──────────────────────────────────────────────────┘')

        print(f'\n   ┌── MÍNIMA VARIANZA ──────────────────────────────┐')
        print(f'   │  Retorno  : {R["ret_minvar"]*100:7.2f}%  '
              f'Volatilidad: {R["risk_minvar"]*100:6.2f}%           │')
        print(f'   └──────────────────────────────────────────────────┘')

        print(f'\n   ┌── BETAS & ALFAS (regresión vs SPX) ────────────┐')
        for i, t in enumerate(R['AssetList']):
            b_str = f'β={R["Betas"][i]:.3f}'
            a_str = f'α={R["Alfa"][i]*100:+.2f}%'
            c_str = f'CAPM={R["CAPM"][i]*100:.2f}%'
            print(f'   │  {t:<10s}  {b_str:<10s}  {a_str:<12s}  {c_str:<16s}│')
        print(f'   └──────────────────────────────────────────────────┘')

        print(f'\n   ┌── MONTE CARLO (10,000 sims) ────────────────────┐')
        print(f'   │  P50 (mediana) : {R["ret_p50"]:+7.1f}%                       │')
        print(f'   │  IC 90%        : [{R["ret_p5"]:+.1f}%, {R["ret_p95"]:+.1f}%]'
              f'                   │')
        print(f'   │  Prob. pérdida : {R["prob_perdida"]:6.1f}%                       │')
        print(f'   └──────────────────────────────────────────────────┘\n')

        # Generar gráficas
        fig = _plot_portfolio(R, portfolio_name)
        figs.append(fig)

    # ── 6. Resumen global y mostrar gráficas ─────────────────────────────────
    print(f'\n{"="*60}')
    print(f'✅ ANÁLISIS COMPLETADO — {len(resultados)} portafolio(s) procesado(s)')
    if resultados:
        print(f'\n  {"Portafolio":<18} {"Sharpe":>8} {"Retorno":>9} {"Vol":>8}')
        print(f'  {"-"*46}')
        for pn, R in sorted(resultados.items(), key=lambda x: -x[1]['sharpe_t']):
            print(f'  {pn:<18} {R["sharpe_t"]:8.4f} '
                  f'{R["ret_t"]*100:8.2f}%  {R["risk_t"]*100:7.2f}%')
    print(f'\n📌 {len(figs)} figura(s) generada(s). Cerrando abre la siguiente.')
    print('=' * 60)

    plt.show()
