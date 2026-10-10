# -*- coding: utf-8 -*-
"""
Genera las gráficas de la tesina en resultados/graficas/ (PDF vectorial para LaTeX + PNG).

    python graficas.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats

from var import delta_normal, historico, montecarlo
from var.comun import cargar_fix, log_rendimientos

OUT = Path(__file__).resolve().parent / "resultados" / "graficas"
OUT.mkdir(parents=True, exist_ok=True)

# Paleta: 3 tonos categóricos y un gris neutro para datos de contexto
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRIS, GRIS_CLARO = "#8a8984", "#d6d5cf"
TINTA, TINTA_2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": GRIS, "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe6",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 100,
    "savefig.bbox": "tight", "axes.titlelocation": "left", "axes.titlecolor": TINTA,
})
miles = mtick.FuncFormatter(lambda x, _: f"{x:,.0f}")
millones = mtick.FuncFormatter(lambda x, _: f"{x/1e6:,.0f}")


def guardar(fig, nombre):
    fig.savefig(OUT / f"{nombre}.pdf")
    fig.savefig(OUT / f"{nombre}.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- 1. FIX 2019–2024
def g1_fix():
    fix = cargar_fix()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(fix.index, fix.values, color=AZUL, lw=1.2)
    ax.set_ylabel("Pesos por dólar")
    ax.set_title("Tipo de cambio FIX USD/MXN, 30/04/2019 – 30/04/2024")
    ax.text(0.98, 0.95, f"30/04/2024: {fix.iloc[-1]:.4f}", transform=ax.transAxes, ha="right", va="top",
            color=TINTA_2, fontsize=9)
    ax.plot(fix.index[-1], fix.iloc[-1], "o", ms=4, color=AZUL)
    guardar(fig, "g1_fix_2019_2024")


# --------------------------------------------------------------------------- 2. Caminatas GBM
def g2_caminatas(n_trayectorias=100, dias=100, semilla=11):
    p = montecarlo.parametros_gbm()
    S0 = 17.1268
    dt = 1 / 252
    Z = np.random.default_rng(semilla).standard_normal((n_trayectorias, dias))
    inc = (p["mu"] - 0.5 * p["sigma"]**2) * dt + p["sigma"] * np.sqrt(dt) * Z
    S = S0 * np.exp(np.hstack([np.zeros((n_trayectorias, 1)), inc.cumsum(axis=1)]))
    t = np.arange(dias + 1)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(t, S.T, color=AZUL, lw=0.5, alpha=0.25)
    lo, hi = S0 * np.exp((p["mu"] - 0.5 * p["sigma"]**2) * t * dt + np.array([[-1.645], [1.645]]) * p["sigma"] * np.sqrt(t * dt))
    ax.plot(t, lo, color=NARANJA, lw=1.4, ls="--")
    ax.plot(t, hi, color=NARANJA, lw=1.4, ls="--")
    ax.text(t[-1] + 1, hi[-1], "percentil 95 %", color=TINTA_2, fontsize=8.5, va="center")
    ax.text(t[-1] + 1, lo[-1], "percentil 5 %", color=TINTA_2, fontsize=8.5, va="center")
    ax.set_xlabel("Días hábiles simulados")
    ax.set_ylabel("Pesos por dólar")
    ax.set_title(f"Trayectorias simuladas del USD/MXN (GBM, {n_trayectorias} trayectorias)")
    ax.set_xlim(0, dias + 14)
    guardar(fig, "g2_trayectorias_gbm")


# --------------------------------------------------------------------------- 3. Tipo de cambio simulado a 1 día
def g3_fx_simulado(sim):
    S0 = sim.attrs["S0"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(sim.S_simulado, bins=80, color=AZUL, edgecolor="white", linewidth=0.4)
    ax.axvline(S0, color=TINTA, lw=1, ls="--")
    ax.text(0.98, 0.95, f"línea punteada: $S_0$ = {S0:.4f}", transform=ax.transAxes, ha="right", va="top", color=TINTA_2, fontsize=9)
    ax.set_xlabel("Tipo de cambio simulado a un día (pesos por dólar)")
    ax.set_ylabel("Número de escenarios")
    ax.set_title(f"Histograma del tipo de cambio simulado ({len(sim):,} escenarios)")
    guardar(fig, "g3_fx_simulado")


# --------------------------------------------------------------------------- 4. P&L histórico vs VaR
def g4_pnl_historico_vs_var(esc, var_dn, var_hs, var_mc):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(esc.index, esc.portafolio, color=GRIS, lw=0.6)
    for v, color, ls, txt in [(var_dn, AZUL, "-", "Delta-Normal"), (var_mc, NARANJA, "--", "Monte Carlo"),
                              (var_hs, AQUA, "-.", "Simulación histórica")]:
        ax.axhline(-v, color=color, lw=1.4, ls=ls, label=f"VaR 95 % {txt}: {v/1e6:,.2f} M")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.axhline(0, color=GRIS, lw=0.6)
    ax.yaxis.set_major_formatter(millones)
    ax.set_ylabel("P&L (millones de MXN)")
    ax.set_title("P&L del portafolio al 30/04/2024 con el rendimiento del FIX de cada fecha")
    ax.yaxis.set_major_locator(mtick.MultipleLocator(2e6))
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(esc.index[0], esc.index[-1])
    ax.set_ylim(-11e6, 13e6)
    guardar(fig, "g4_pnl_historico_vs_var")


# --------------------------------------------------------------------------- 5. Distribuciones de P&L (HS vs MC)
def g5_distribucion_pnl(esc, sim, var_dn, var_hs, var_mc):
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    lim = (-6e6, 6e6)
    bins = np.linspace(-10e6, 10e6, 161)
    for ax, datos, nombre, v, color in [(axes[0], esc.portafolio, "Simulación histórica (1,260 escenarios)", var_hs, AQUA),
                                        (axes[1], sim.portafolio, "Monte Carlo (10,000 escenarios)", var_mc, NARANJA)]:
        ax.hist(datos, bins=bins, density=True, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(-v, color=TINTA, lw=1.4, label=f"VaR 95 % del método: {v/1e6:,.2f} M")
        ax.axvline(-var_dn, color=AZUL, lw=1.3, ls="--", label=f"VaR 95 % Delta-Normal: {var_dn/1e6:,.2f} M")
        fuera = int((datos < lim[0]).sum())
        if fuera:
            ax.text(0.01, 0.35, f"{fuera} escenarios con\npérdida mayor a 6 M\nquedan fuera del eje", transform=ax.transAxes,
                    fontsize=8, color=TINTA_2, va="top")
        ax.set_title(nombre)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.legend(loc="upper right", fontsize=8.5)
    axes[1].set_xlim(*lim)
    axes[1].xaxis.set_major_formatter(millones)
    axes[1].set_xlabel("P&L del portafolio a un día (millones de MXN)")
    guardar(fig, "g5_distribucion_pnl")


# --------------------------------------------------------------------------- 6. Normalidad de los rendimientos
def g6_normalidad():
    r = log_rendimientos(cargar_fix()) * 100
    mu, sd = r.mean(), r.std()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 3.3), gridspec_kw={"width_ratios": [1.4, 1]})
    a1.hist(r, bins=80, density=True, color=AZUL, alpha=0.85, edgecolor="white", linewidth=0.3)
    x = np.linspace(r.min(), r.max(), 400)
    a1.plot(x, stats.norm.pdf(x, mu, sd), color=NARANJA, lw=1.5)
    a1.text(0.97, 0.95, f"Normal($\\bar r$, $s$)\ncurtosis en exceso: {stats.kurtosis(r):.1f}", transform=a1.transAxes,
            ha="right", va="top", fontsize=8.5, color=TINTA_2)
    a1.set_xlabel("Log-rendimiento diario (%)")
    a1.set_yticks([])
    a1.grid(axis="y", visible=False)
    a1.set_title("Distribución de los rendimientos del FIX")
    (osm, osr), (m_, b_, _) = stats.probplot(r, dist="norm")
    a2.plot(osm, osr, "o", ms=2.5, color=AZUL, alpha=0.6)
    a2.plot(osm, m_ * osm + b_, color=NARANJA, lw=1.3)
    a2.set_xlabel("Cuantiles de la normal")
    a2.set_ylabel("Cuantiles observados (%)")
    a2.set_title("Gráfica Q-Q")
    guardar(fig, "g6_normalidad_rendimientos")


# --------------------------------------------------------------------------- 7. Curvas e interpolación
def g7_curvas_interpolacion():
    from valuacion import opciones
    ins = opciones.cargar_insumos()
    res = opciones.valuar_portafolio_opciones(ins)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, curva, nombre in [(axes[0], ins["curva_mxn"], "Curva MXN"), (axes[1], ins["curva_usd"], "Curva USD")]:
        c = curva[curva.iloc[:, 0] <= 120]
        ax.plot(c.iloc[:, 0], c.iloc[:, 1], color=GRIS, lw=1, label="Interpolación lineal entre nodos")
        ax.plot(c.iloc[:, 0], c.iloc[:, 1], "o", ms=5, color=AZUL, label="Nodos de la curva")
        dias = sorted({int((r.fecha_spot - r.fecha_valuacion).days) for r in res.itertuples()} |
                      {int((r.liquidacion - r.fecha_valuacion).days) for r in res.itertuples()})
        vals = [opciones.interpolar_lineal(curva, d) for d in dias]
        ax.plot(dias, vals, "D", ms=5, color=NARANJA, label="Tasas interpoladas")
        for d, v in zip(dias, vals):
            ax.annotate(f"{d} d", (d, v), xytext=(4, -10), textcoords="offset points", fontsize=8, color=TINTA_2)
        ax.set_title(f"{nombre} al 30/04/2024 (hasta 120 días)")
        ax.set_xlabel("Plazo (días)")
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.3f} %"))
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout(w_pad=2)
    guardar(fig, "g7_curvas_interpolacion")


if __name__ == "__main__":
    dn = delta_normal.var_delta_normal()["var_portafolio"]
    esc = historico.escenarios_historicos()
    sim = montecarlo.simular()
    hs, mc = historico.var_historico(esc).portafolio, montecarlo.var_mc(sim).portafolio
    g1_fix(); g2_caminatas(); g3_fx_simulado(sim); g4_pnl_historico_vs_var(esc, dn, hs, mc)
    g5_distribucion_pnl(esc, sim, dn, hs, mc); g6_normalidad(); g7_curvas_interpolacion()
    print("Gráficas generadas en", OUT)
    for f in sorted(OUT.glob("*.png")):
        print(" -", f.name)
