# -*- coding: utf-8 -*-
"""
Genera todas las tablas de la tesina en resultados/ (un .tex listo para pegar y un .csv por tabla).

    python tablas.py
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

import tablas_valuacion as tv
from valuacion import opciones
from var import delta_normal, historico, montecarlo

OUT = Path(__file__).resolve().parent / "resultados"
OUT.mkdir(exist_ok=True)

m = lambda x, d=2: f"{x:,.{d}f}"                    # número con separador de miles
pct = lambda x, d=4: f"{x:.{d}f}\\%"
fecha = lambda t: pd.Timestamp(t).strftime("%d/%m/%Y")


def _tabla(nombre, caption, label, columnas, alineacion, filas, nota=None, extra_final=None):
    cuerpo = "\n".join(" & ".join(f) + r" \\" for f in filas)
    if extra_final:
        cuerpo += "\n\\midrule\n" + " & ".join(extra_final) + r" \\"
    tex = (f"\\begin{{table}}[H]\n\\centering\n\\small\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
           f"\\begin{{tabular}}{{{alineacion}}}\n\\toprule\n" + " & ".join(columnas) + " \\\\\n\\midrule\n"
           + cuerpo + "\n\\bottomrule\n\\end{tabular}\n")
    if nota:
        tex += "\n\\vspace{2pt}\n\\parbox{0.95\\linewidth}{\\footnotesize " + nota + "}\n"
    tex += "\\end{table}\n"
    (OUT / f"{nombre}.tex").write_text(tex, encoding="utf-8")


# --------------------------------------------------------------------------- Tabla 2
def tabla2_opciones():
    ins = opciones.cargar_insumos()
    df = opciones.valuar_portafolio_opciones(ins)
    sis = ins["posicion"].set_index("CONTRACT_ID")["UNITARY_PRICE"]
    df["precio_sistema"] = df.contrato.map(sis)
    df["dif_pct"] = (df.precio_unitario / df.precio_sistema - 1) * 100
    df.to_csv(OUT / "tabla2_opciones.csv", index=False)
    filas = [[f"Opción {i}", "Call" if r.tipo == "C" else "Put", r.posicion.capitalize(), m(r.nocional_usd, 0),
              m(r.strike, 2), fecha(r.vencimiento), str(r.T_vol_dias), str(r.T_desc_dias), pct(r.vol_pct, 3),
              pct(r.rd_fwd_pct), pct(r.rf_fwd_pct), m(r.forward, 4), m(r.precio_unitario, 6), m(r.valor_presente_mxn),
              f"{r.dif_pct:+.4f}\\%"] for i, r in enumerate(df.itertuples(), 1)]
    r0 = df.iloc[0]
    _tabla("tabla2_opciones", "Valuación de opciones vanilla USD/MXN", "tab:opciones",
           ["", "Tipo", "Posición", "Nocional USD", "$K$", "Vencimiento", "$T$", "$T'$", "$\\sigma$",
            "$r_d^{fwd}$", "$r_f^{fwd}$", "$F$", "Precio unitario", "VP (MXN)", "Dif. sistema"],
           "lllrrcrrrrrrrrr", filas,
           nota=(f"Nota: fecha de valuación {fecha(r0.fecha_valuacion)}, fecha spot {fecha(r0.fecha_spot)}, tipo de cambio "
                 f"{r0.spot:.4f}. $T$: días del 30/04/2024 al vencimiento (plazo de la volatilidad, base 365); $T'$: días de la "
                 "fecha spot a la liquidación (plazo de descuento, base 360). Tasas forward simples. Precio unitario en MXN por "
                 "USD de nocional. Dif. sistema: diferencia del precio unitario contra el del sistema de valuación de la institución."))
    return df


# --------------------------------------------------------------------------- Tabla 4
def tabla4_portafolio():
    from valuacion import futuros, swaps
    o, f, s = opciones.valuar_portafolio_opciones(), futuros.valuar_portafolio_futuros(), swaps.valuar_portafolio_swaps()
    filas = []
    for r in o.itertuples():
        filas.append(["Opción " + ("Call" if r.tipo == "C" else "Put"), r.posicion.capitalize(), m(r.nocional_usd),
                      str((r.vencimiento - r.fecha_valuacion).days), m(r.valor_presente_mxn)])
    for r in s.itertuples():
        filas.append(["Swap USD/MXN", ("Recibe" if r.pata_usd == "recibe" else "Paga") + " USD", m(r.nocional_usd),
                      str(r.plazo_remanente_dias), m(r.valor_presente_mxn)])
    for r in f.itertuples():
        filas.append(["Futuro", r.posicion.capitalize(), m(r.contratos * futuros.TAMANO_CONTRATO_USD),
                      str((r.vencimiento - r.fecha_valuacion).days), m(r.valor_presente_mxn)])
    total = o.valor_presente_mxn.sum() + s.valor_presente_mxn.sum() + f.valor_presente_mxn.sum()
    pd.DataFrame(filas, columns=["instrumento", "posicion", "nocional_usd", "plazo_dias", "vp_mxn"]).to_csv(OUT / "tabla4_portafolio.csv", index=False)
    _tabla("tabla4_portafolio", "Portafolio de derivados al 30/04/2024", "tab:portafolio",
           ["Instrumento", "Posición", "Nocional (USD)", "Plazo (días)", "Valor presente (MXN)"], "llrrr", filas,
           extra_final=["\\textbf{Total}", "", "", "", f"\\textbf{{{m(total)}}}"],
           nota="Nota: montos en USD para nocionales y en MXN para valores presentes. En futuros, nocional = contratos × 10,000 USD "
                "(negativo: posición corta). En opciones, nocional positivo: opción comprada. Plazo: días naturales del 30/04/2024 "
                "al vencimiento del contrato o del último flujo.")


# --------------------------------------------------------------------------- Tabla 5
def tabla5_delta_normal():
    res = delta_normal.var_delta_normal()
    d = res["detalle"]
    d.to_csv(OUT / "tabla5_delta_normal.csv", index=False)
    filas = [[r.instrumento, r.posicion[:1].upper() + r.posicion[1:], m(r.delta_usd), m(r.exposicion_mxn), m(r.var_individual_mxn)]
             for r in d.itertuples()]
    _tabla("tabla5_delta_normal", "VaR Delta-Normal a un día (95\\% de confianza)", "tab:var_dn",
           ["Instrumento", "Posición", "$\\Delta_i$ (USD)", "$S_0\\,\\Delta_i$ (MXN)", "VaR individual (MXN)"], "llrrr", filas,
           extra_final=["\\textbf{Portafolio}", "", f"\\textbf{{{m(res['delta_portafolio_usd'])}}}",
                        f"\\textbf{{{m(res['S0'] * res['delta_portafolio_usd'])}}}", f"\\textbf{{{m(res['var_portafolio'])}}}"],
           nota=(f"Nota: $\\Delta_i = \\partial V_i/\\partial S$ de toda la posición. VaR$_i = z_{{0.95}}\\,\\sigma\\,S_0\\,|\\Delta_i|$ "
                 f"(informativo) y VaR del portafolio $= z_{{0.95}}\\,\\sigma\\,S_0\\,|\\sum_i\\Delta_i|$, con $z_{{0.95}}={res['z']:.4f}$, "
                 f"$\\sigma={100*res['sigma_diaria']:.4f}\\%$ diaria (desviación estándar muestral de {res['n_rendimientos']:,} "
                 f"log-rendimientos diarios del FIX, {fecha(res['inicio'])}--{fecha(res['fin'])}) y $S_0={res['S0']:.4f}$. "
                 "Los VaR individuales no son aditivos: las posiciones de signo contrario se compensan en el portafolio."))
    return res


# --------------------------------------------------------------------------- Tablas 6 y 7
def tabla6_historico(esc=None):
    esc = esc if esc is not None else historico.escenarios_historicos()
    v = historico.var_historico(esc)
    peor = esc.portafolio.idxmin()
    filas = [["Opciones", m(v.opciones)], ["Swaps", m(v.swaps)], ["Futuros", m(v.futuros)],
             ["\\textbf{Portafolio}", f"\\textbf{{{m(v.portafolio)}}}"]]
    v.to_csv(OUT / "tabla6_historico.csv")
    _tabla("tabla6_historico", "VaR por simulación histórica a un día (95\\% de confianza)", "tab:var_hs",
           ["Clase de instrumento", "VaR (MXN)"], "lr", filas,
           nota=(f"Nota: {len(esc):,} escenarios, uno por cada log-rendimiento diario del FIX entre {fecha(esc.index.min())} y "
                 f"{fecha(esc.index.max())}. En cada escenario, $S^{{(t)}}=S_0\\,e^{{r_t}}$ con $S_0={esc.attrs['S0']:.4f}$, y se "
                 "revalúa el portafolio al 30/04/2024 (curvas y volatilidades fijas). VaR = $-$percentil 5\\% del P\\&L simulado. "
                 f"Peor escenario: {fecha(peor)} ($r_t={100*esc.loc[peor,'r']:.2f}\\%$), pérdida de {m(-esc.portafolio.min(), 0)} MXN. "
                 "El VaR por clase no es aditivo."))
    return esc


def tabla7_montecarlo(sim=None):
    sim = sim if sim is not None else montecarlo.simular()
    v = montecarlo.var_mc(sim)
    est = montecarlo.estabilidad()
    a = sim.attrs
    filas = [["Opciones", m(v.opciones)], ["Swaps", m(v.swaps)], ["Futuros", m(v.futuros)],
             ["\\textbf{Portafolio}", f"\\textbf{{{m(v.portafolio)}}}"]]
    v.to_csv(OUT / "tabla7_montecarlo.csv")
    _tabla("tabla7_montecarlo", "VaR por simulación Monte Carlo a un día (95\\% de confianza)", "tab:var_mc",
           ["Clase de instrumento", "VaR (MXN)"], "lr", filas,
           nota=(f"Nota: movimiento browniano geométrico con $\\hat\\mu={100*a['mu']:.2f}\\%$ y $\\hat\\sigma={100*a['sigma']:.2f}\\%$ anuales "
                 f"($\\Delta t=1/252$), estimados con {a['n']:,} log-rendimientos diarios del FIX. {a['n_sim']:,} escenarios con semilla "
                 f"{a['semilla']}, los mismos para todos los instrumentos, y revaluación completa del portafolio. Con 200 semillas "
                 f"distintas, el VaR del portafolio tiene media {m(est['mean'], 0)} MXN y el 95\\% de los resultados cae entre "
                 f"{m(est['2.5%'], 0)} y {m(est['97.5%'], 0)} MXN. El VaR por clase no es aditivo."))
    return sim


# --------------------------------------------------------------------------- Tabla 8
def tabla8_comparativo(dn, esc, sim):
    vals = {"Delta-Normal": dn["var_portafolio"], "Simulación histórica": historico.var_historico(esc).portafolio,
            "Monte Carlo": montecarlo.var_mc(sim).portafolio}
    pd.Series(vals, name="var_95").to_csv(OUT / "tabla8_comparativo.csv")
    filas = [[k, m(v)] for k, v in vals.items()]
    _tabla("tabla8_comparativo", "VaR del portafolio a un día por metodología (95\\% de confianza)", "tab:var_comparativo",
           ["Método", "VaR (MXN)"], "lr", filas)


def ajustar_ancho(max_columnas: int = 6):
    """Las tablas anchas (más de `max_columnas` columnas) se escalan al ancho del texto (requiere \\usepackage{graphicx})."""
    import re
    for f in OUT.glob("tabla*.tex"):
        t = f.read_text(encoding="utf-8")
        mo = re.search(r"\\begin\{tabular\}\{([^}]*)\}", t)
        if mo and len(mo.group(1)) > max_columnas and "resizebox" not in t:
            t = t.replace(mo.group(0), "\\resizebox{\\textwidth}{!}{%\n" + mo.group(0), 1)
            t = t.replace("\\end{tabular}", "\\end{tabular}}", 1)
            f.write_text(t, encoding="utf-8")


if __name__ == "__main__":
    tv.tabla_futuros()
    tv.tabla_swaps()
    tabla2_opciones()
    tabla4_portafolio()
    dn = tabla5_delta_normal()
    esc = tabla6_historico()
    sim = tabla7_montecarlo()
    tabla8_comparativo(dn, esc, sim)
    ajustar_ancho()
    print("Tablas generadas en", OUT)
    for f in sorted(OUT.glob("tabla*.tex")):
        print(" -", f.name)
