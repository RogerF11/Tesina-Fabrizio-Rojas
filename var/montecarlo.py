# -*- coding: utf-8 -*-
"""
VaR por simulación Monte Carlo a 1 día (un factor: USD/MXN, movimiento browniano geométrico).

    dS/S = mu dt + sigma dW     =>   S_{t+dt} = S_t exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z),  Z ~ N(0,1)

Estimación con los log-rendimientos diarios del FIX (30/04/2019–30/04/2024), dt = 1/252 (días hábiles):
    r_t = ln(S_t/S_{t-1}) ~ N((mu - sigma^2/2) dt, sigma^2 dt)
    sigma_hat = std(r) * sqrt(252)
    mu_hat    = 252 * mean(r) + sigma_hat^2 / 2
(Con estos estimadores, el rendimiento simulado a un día tiene exactamente la media y la varianza
muestrales de los rendimientos diarios observados.)

Un solo vector de escenarios S^(s) para TODO el portafolio (misma Z para todos los instrumentos),
revaluación completa de cada escenario y  VaR_alpha = -percentil_{1-alpha}(dV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from var.comun import cargar_fix, log_rendimientos, revaluador

N_SIM = 10_000
SEMILLA = 11
DIAS_ANIO = 252


def parametros_gbm() -> dict:
    r = log_rendimientos(cargar_fix())
    sigma = float(r.std(ddof=1) * np.sqrt(DIAS_ANIO))
    mu = float(DIAS_ANIO * r.mean() + 0.5 * sigma**2)
    return {"mu": mu, "sigma": sigma, "n": len(r), "media_diaria": float(r.mean()), "sigma_diaria": float(r.std(ddof=1))}


_CACHE: dict = {}


def _insumos():
    if not _CACHE:
        _CACHE["p"] = parametros_gbm()
        _CACHE["rev"] = revaluador()
    return _CACHE["p"], _CACHE["rev"]


def simular(n_sim: int = N_SIM, semilla: int = SEMILLA, horizonte_dias: int = 1) -> pd.DataFrame:
    p, (S0, V0, valor) = _insumos()
    dt = horizonte_dias / DIAS_ANIO
    Z = np.random.default_rng(semilla).standard_normal(n_sim)
    S = S0 * np.exp((p["mu"] - 0.5 * p["sigma"]**2) * dt + p["sigma"] * np.sqrt(dt) * Z)
    dV = valor(S) - V0.to_numpy()
    dV["portafolio"] = dV.sum(axis=1)
    dV.insert(0, "S_simulado", S)
    dV.attrs.update(p | {"S0": S0, "semilla": semilla, "n_sim": n_sim})
    return dV


def var_mc(sim: pd.DataFrame, alpha: float = 0.95) -> pd.Series:
    return -sim[["opciones", "swaps", "futuros", "portafolio"]].quantile(1 - alpha)


def estabilidad(n_sim: int = N_SIM, n_semillas: int = 200, alpha: float = 0.95) -> pd.Series:
    """Distribución del VaR del portafolio al cambiar la semilla (error de muestreo con N simulaciones)."""
    v = [var_mc(simular(n_sim, s), alpha)["portafolio"] for s in range(n_semillas)]
    return pd.Series(v).describe(percentiles=[0.025, 0.975])


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    sim = simular()
    a = sim.attrs
    print(f"mu = {a['mu']:.6f}  sigma = {a['sigma']:.6f}  (diarios: media {a['media_diaria']:.6f}, sigma {a['sigma_diaria']:.6f}; n = {a['n']})")
    print(f"S0 = {a['S0']}, N = {a['n_sim']}, semilla = {a['semilla']}")
    print("\nVaR MC 95 %:\n" + var_mc(sim).to_string())
    print("\nEstabilidad del VaR 95 % del portafolio (200 semillas):\n" + estabilidad().to_string())
    sim.to_csv("resultados/var_montecarlo_escenarios.csv", index=False)
