# -*- coding: utf-8 -*-
"""
VaR por Simulación Histórica a 1 día (un factor: USD/MXN), con revaluación completa.

Para cada día t de la ventana 30/04/2019–30/04/2024:
    r_t      = ln(S_t / S_{t-1})                (rendimiento histórico del FIX)
    S^(t)    = S0 * exp(r_t)                    (escenario: el movimiento de t aplicado HOY)
    dV_t     = V(S^(t)) - V(S0)                 (P&L del portafolio ACTUAL en ese escenario)
    VaR_95   = - percentil_5 %( {dV_t} )

Curvas de tasas y volatilidades se mantienen en su nivel del 30/04/2024 (un solo factor de riesgo).
S0 es el spot con el que se valúa cada instrumento (17.1268).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from var.comun import cargar_fix, log_rendimientos


def escenarios_historicos() -> pd.DataFrame:
    """Revaluación completa (exacta) del portafolio en cada escenario histórico; ver var.comun.revaluador."""
    from var.comun import revaluador
    S0, V0, valor = revaluador()
    r = log_rendimientos(cargar_fix())
    S = S0 * np.exp(r.to_numpy())
    dV = valor(S) - V0.to_numpy()
    dV.index = r.index
    dV.index.name = "fecha"
    df = pd.concat([r.rename("r"), pd.Series(S, index=r.index, name="S_escenario"), dV], axis=1)
    df["portafolio"] = df[["opciones", "swaps", "futuros"]].sum(axis=1)
    df.attrs["S0"] = S0
    df.attrs["V0"] = V0.to_dict()
    return df


def var_historico(esc: pd.DataFrame, alpha: float = 0.95) -> pd.Series:
    """VaR (positivo) por clase y del portafolio; percentil empírico con interpolación lineal."""
    return -esc[["opciones", "swaps", "futuros", "portafolio"]].quantile(1 - alpha)


if __name__ == "__main__":
    import time
    t = time.time()
    esc = escenarios_historicos()
    print(f"{len(esc)} escenarios en {time.time()-t:.1f} s; S0 = {esc.attrs['S0']}")
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print("\nVaR histórico 95 %:\n", var_historico(esc).to_string())
    print("\n5 peores escenarios del portafolio:\n", esc.nsmallest(5, "portafolio")[["r", "S_escenario", "portafolio"]].to_string())
    esc.to_csv("resultados/var_historico_escenarios.csv")
