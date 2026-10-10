# -*- coding: utf-8 -*-
"""
VaR Delta-Normal a 1 día con un solo factor de riesgo (USD/MXN).

    r_t = ln(S_t / S_{t-1})                  log-rendimientos diarios del FIX
    sigma = sqrt( 1/(n-1) * sum (r_t - r_bar)^2 )
    Delta_i = dV_i/dS  (USD)                 sensibilidad de TODA la posición i
    dV ~ S0 * Delta_port * r,  Delta_port = sum_i Delta_i
    VaR_port = z_alpha * sigma * S0 * |Delta_port|
    VaR_i    = z_alpha * sigma * S0 * |Delta_i|          (informativo)
    VaR_bruto = z_alpha * sigma * S0 * sum_i |Delta_i|   (sin compensación)
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import norm

from var.comun import cargar_fix, log_rendimientos, portafolio, SPOT_DN


def var_delta_normal(alpha: float = 0.95, S0: float = SPOT_DN) -> dict:
    r = log_rendimientos(cargar_fix())
    sigma = float(r.std(ddof=1))
    z = float(norm.ppf(alpha))
    port = portafolio()
    port["exposicion_mxn"] = S0 * port.delta_usd
    port["var_individual_mxn"] = z * sigma * port.exposicion_mxn.abs()
    delta_port = float(port.delta_usd.sum())
    return {
        "detalle": port,
        "sigma_diaria": sigma, "n_rendimientos": len(r), "media_diaria": float(r.mean()),
        "inicio": r.index.min(), "fin": r.index.max(),
        "z": z, "S0": S0,
        "delta_portafolio_usd": delta_port,
        "var_portafolio": z * sigma * S0 * abs(delta_port),
        "var_bruto": z * sigma * S0 * float(port.delta_usd.abs().sum()),
    }


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    res = var_delta_normal()
    print(res["detalle"].to_string(index=False))
    for k in ["sigma_diaria", "n_rendimientos", "media_diaria", "inicio", "fin", "z", "S0", "delta_portafolio_usd", "var_portafolio", "var_bruto"]:
        print(f"{k:>22}: {res[k]}")
