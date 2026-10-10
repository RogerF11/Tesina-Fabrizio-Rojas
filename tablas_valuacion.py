# -*- coding: utf-8 -*-
"""Genera las tablas de valuación de la tesina (CSV + LaTeX) en resultados/."""
from pathlib import Path
from valuacion import futuros

OUT = Path(__file__).resolve().parent / "resultados"
OUT.mkdir(exist_ok=True)


def tabla_futuros():
    ins = futuros.cargar_insumos()
    df = futuros.valuar_portafolio_futuros(ins)
    sis = ins["posicion"].set_index("CONTRACT_ID")["VAL_DE_LOS_DERIVADOS_BM"]
    df["valor_sistema"] = df.contrato.map(sis)
    df["dif_pct"] = (df.valor_presente_mxn / df.valor_sistema - 1) * 100
    df.to_csv(OUT / "tabla1_futuros.csv", index=False)

    f = lambda x, d=2: f"{x:,.{d}f}"
    filas = []
    for i, r in enumerate(df.itertuples(), 1):
        filas.append(" & ".join([
            f"Futuro {i}", r.posicion.capitalize(), f(r.contratos, 0), f(r.contratos * futuros.TAMANO_CONTRATO_USD, 0),
            f(r.precio_pactado, 4), r.vencimiento.strftime("%d/%m/%Y"), r.liquidacion.strftime("%d/%m/%Y"),
            str(r.T_dias), f(r.rd_fwd_pct, 4) + r"\%", f(r.rf_fwd_pct, 4) + r"\%", f(r.forward, 4),
            f(r.valor_presente_mxn), f(r.valor_sistema), f"{r.dif_pct:+.3f}" + r"\%"]) + r" \\")
    r0 = df.iloc[0]
    tex = r"""\begin{table}[H]
\centering
\scriptsize
\caption{Valuación de contratos de futuros USD/MXN}
\label{tab:futuros}
\begin{tabular}{llrrrccrrrrrrr}
\toprule
 & Posición & Contratos & Nocional USD & $P_0$ & Vencimiento & Liquidación & $T$ & $r_d^{fwd}$ & $r_f^{fwd}$ & $F$ & Modelo (MXN) & Sistema (MXN) & Dif. \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}

\vspace{2pt}
\parbox{\linewidth}{\footnotesize Nota: fecha de valuación """ + r0.fecha_valuacion.strftime("%d/%m/%Y") + \
        ", fecha spot " + r0.fecha_spot.strftime("%d/%m/%Y") + f", tipo de cambio {r0.spot:.4f}" + \
        r""". $T$ en días naturales de la fecha spot a la liquidación; tasas forward simples, base 360. Tamaño de contrato: 10,000 USD. Valor del modelo: $(F-P_0)\,Q\,N$.}
\end{table}
"""
    (OUT / "tabla1_futuros.tex").write_text(tex, encoding="utf-8")
    return df


def tabla_swaps():
    from valuacion import swaps
    ins = swaps.cargar_insumos()
    df = swaps.valuar_portafolio_swaps(ins)
    sis = ins["posicion"].set_index("NB")["DISCOUNTED_MARKET_VALUE"]
    df["valor_sistema"] = df.nb.map(sis)
    df["dif_pct"] = (df.valor_presente_mxn / df.valor_sistema - 1) * 100
    df.to_csv(OUT / "tabla3_swaps.csv", index=False)
    f = lambda x, d=2: f"{x:,.{d}f}"
    filas = [" & ".join([f"Swap {i}", r.pata_usd.capitalize(), str(r.plazo_remanente_dias), f"{r.flujos_usd}/{r.flujos_mxn}",
                         f(r.nocional_usd), f(r.nocional_mxn), f(r.tasa_usd_pct) + r"\%", f(r.tasa_mxn_pct) + r"\%",
                         f(r.vp_usd), f(r.vp_mxn), f(r.valor_presente_mxn), f(r.valor_sistema), f"{r.dif_pct:+.3f}" + r"\%"]) + r" \\"
             for i, r in enumerate(df.itertuples(), 1)]
    tex = r"""\begin{table}[H]
\centering
\scriptsize
\caption{Valuación de swaps cross currency USD/MXN 1M--1M}
\label{tab:swaps}
\begin{tabular}{llrrrrrrrrrrr}
\toprule
 & Pata USD & Plazo (días) & Flujos USD/MXN & Nocional USD & Nocional MXN & Tasa USD & Tasa MXN & $VP_{USD}$ & $VP_{MXN}$ & Modelo (MXN) & Sistema (MXN) & Dif. \\
\midrule
""" + "\n".join(filas) + r"""
\bottomrule
\end{tabular}

\vspace{2pt}
\parbox{\linewidth}{\footnotesize Nota: fecha de valuación 30/04/2024; tipo de cambio """ + f"{df.spot.iloc[0]:.4f}" + r""". $VP_{USD}$ y $VP_{MXN}$ sin signo, a la fecha spot. Valor del swap: $\varepsilon\,(S\,VP_{USD}-VP_{MXN})$, con $\varepsilon=+1$ si se recibe la pata en dólares y $-1$ si se paga.}
\end{table}
"""
    (OUT / "tabla3_swaps.tex").write_text(tex, encoding="utf-8")
    return df


if __name__ == "__main__":
    print(tabla_futuros().T)
    print(tabla_swaps().T)
