# -*- coding: utf-8 -*-
"""
Valuación de opciones vanilla europeas USD/MXN (FXD).

Convenciones:

* Tasas de las curvas: porcentaje anual SIMPLE, base Actual/360
  (una tasa de 11.25 % se guarda como 11.25).
* Fechas:  t0 = fecha de valuación, ts = fecha spot (t0 + 2 días hábiles),
  te = vencimiento, tl = liquidación (te + 2 días hábiles).
* Plazo de descuento   T' = tl - ts   (días naturales, base 360).
  La prima se paga en la fecha spot, por eso se descuenta de tl a ts.
* Plazo de volatilidad tau = (te - t0) / 365.
* Tasas forward simples de ts a tl para cada moneda:
      r_fwd = [ (1 + r_l * d_l/36000) / (1 + r_s * d_s/36000) - 1 ] * 36000 / (d_l - d_s)
* Factores de descuento simples:  DF_d = 1/(1 + r_d,fwd T'/36000),
                                  DF_f = 1/(1 + r_f,fwd T'/36000)
* Forward:  F = S * DF_f / DF_d
* Black sobre el forward:
      d1 = [ln(F/K) + sigma^2 tau / 2] / (sigma sqrt(tau)),  d2 = d1 - sigma sqrt(tau)
      Call = DF_f S N(d1) - DF_d K N(d2)
      Put  = DF_d K N(-d2) - DF_f S N(-d1)
* Delta spot (derivada exacta del precio respecto a S):
      Call:  DF_f N(d1)        Put: -DF_f N(-d1)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

BASE_TASAS = 36000.0   # 360 días x 100 (tasas en porcentaje)
BASE_VOL = 365.0

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "opciones"


# ----------------------------------------------------------------------------
# Utilidades de fechas y curvas
# ----------------------------------------------------------------------------
def fecha_mas_dias_habiles(fecha: pd.Timestamp, n: int, inhabiles) -> pd.Timestamp:
    """Suma n días hábiles (lunes-viernes, excluyendo `inhabiles`)."""
    fechas = [x for x in inhabiles if isinstance(x, (pd.Timestamp, np.datetime64)) or hasattr(x, "year")]
    inh = set(pd.to_datetime(pd.Series(fechas)).dt.normalize())
    f = pd.Timestamp(fecha).normalize()
    contados = 0
    while contados < n:
        f += pd.Timedelta(days=1)
        if f.weekday() < 5 and f not in inh:
            contados += 1
    return f


def interpolar_lineal(curva: pd.DataFrame, dias: float) -> float:
    """Interpolación lineal en (plazo, valor); extrapolación plana en extremos."""
    x = curva.iloc[:, 0].to_numpy(dtype=float)
    y = curva.iloc[:, 1].to_numpy(dtype=float)
    return float(np.interp(dias, x, y))


def tasa_forward_simple(r_spot: float, r_liq: float, d_liq: float, d_spot: float) -> float:
    """Tasa forward simple (en %, base 360) entre d_spot y d_liq."""
    fl = 1 + r_liq * d_liq / BASE_TASAS
    fs = 1 + r_spot * d_spot / BASE_TASAS
    return (fl / fs - 1) * BASE_TASAS / (d_liq - d_spot)


def factor_descuento_simple(r: float, dias: float) -> float:
    return 1.0 / (1 + r * dias / BASE_TASAS)


# ----------------------------------------------------------------------------
# Precio y delta
# ----------------------------------------------------------------------------
def precio_y_delta(tipo: str, S: float, K: float, vol_pct: float,
                   rd_fwd: float, rf_fwd: float, T_desc: float, T_vol: float) -> tuple[float, float]:
    """Precio unitario (MXN por USD de nocional) y delta spot de una vanilla."""
    if tipo not in ("C", "P"):
        raise ValueError(f"Tipo de opción no reconocido: {tipo!r} (se espera 'C' o 'P')")
    sigma = vol_pct / 100.0
    tau = T_vol / BASE_VOL
    df_d = factor_descuento_simple(rd_fwd, T_desc)
    df_f = factor_descuento_simple(rf_fwd, T_desc)
    F = S * df_f / df_d
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if tipo == "C":
        precio = df_f * S * norm.cdf(d1) - df_d * K * norm.cdf(d2)
        delta = df_f * norm.cdf(d1)
    else:
        precio = df_d * K * norm.cdf(-d2) - df_f * S * norm.cdf(-d1)
        delta = -df_f * norm.cdf(-d1)
    return float(precio), float(delta)


def precio_vectorizado(tipo: str, S, K: float, vol_pct: float, rd_fwd: float, rf_fwd: float,
                       T_desc: float, T_vol: float):
    """Misma fórmula que `precio_y_delta`, pero acepta un arreglo de spots (para simulaciones)."""
    S = np.asarray(S, dtype=float)
    sigma, tau = vol_pct / 100.0, T_vol / BASE_VOL
    df_d, df_f = factor_descuento_simple(rd_fwd, T_desc), factor_descuento_simple(rf_fwd, T_desc)
    F = S * df_f / df_d
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if tipo == "C":
        return df_f * S * norm.cdf(d1) - df_d * K * norm.cdf(d2)
    if tipo == "P":
        return df_d * K * norm.cdf(-d2) - df_f * S * norm.cdf(-d1)
    raise ValueError(f"Tipo de opción no reconocido: {tipo!r}")


@dataclass
class ResultadoOpcion:
    contrato: int
    tipo: str
    posicion: str          # "larga" (comprada) o "corta" (vendida)
    nocional_usd: float
    strike: float
    spot: float
    fecha_valuacion: pd.Timestamp
    fecha_spot: pd.Timestamp
    vencimiento: pd.Timestamp
    liquidacion: pd.Timestamp
    T_vol_dias: int
    T_desc_dias: int
    vol_pct: float
    rd_fwd_pct: float
    rf_fwd_pct: float
    forward: float
    precio_unitario: float
    valor_presente_mxn: float
    delta_unitaria: float
    delta_posicion_usd: float   # dV/dS de toda la posición (USD equivalentes)


def valuar_opcion(op: pd.Series, curva_mxn: pd.DataFrame, curva_usd: pd.DataFrame,
                  curva_vol: pd.DataFrame, inhabiles, spot: float | None = None) -> ResultadoOpcion:
    """Valúa una opción del archivo de posición. Si `spot` es None usa MARKET_SPOT."""
    S = float(op.MARKET_SPOT if spot is None else spot)
    t0 = pd.Timestamp(op.FCH_DATAMART)
    te = pd.Timestamp(op.DATE_PERIOD_EXPIRY_DATE)
    ts = fecha_mas_dias_habiles(t0, 2, inhabiles)
    tl = fecha_mas_dias_habiles(te, 2, inhabiles)

    d_s = (ts - t0).days
    d_l = (tl - t0).days
    T_desc = (tl - ts).days
    T_vol = (te - t0).days

    rd_fwd = tasa_forward_simple(interpolar_lineal(curva_mxn, d_s), interpolar_lineal(curva_mxn, d_l), d_l, d_s)
    rf_fwd = tasa_forward_simple(interpolar_lineal(curva_usd, d_s), interpolar_lineal(curva_usd, d_l), d_l, d_s)
    vol = interpolar_lineal(curva_vol, T_vol)

    precio, delta = precio_y_delta(op.CALL_PUT, S, float(op.STRIKE), vol, rd_fwd, rf_fwd, T_desc, T_vol)
    q = float(op.LIVE_QUANTITY_SIGNED)
    F = S * factor_descuento_simple(rf_fwd, T_desc) / factor_descuento_simple(rd_fwd, T_desc)
    return ResultadoOpcion(
        contrato=int(op.CONTRACT_ID), tipo=op.CALL_PUT, posicion="larga" if q > 0 else "corta",
        nocional_usd=q, strike=float(op.STRIKE), spot=S,
        fecha_valuacion=t0, fecha_spot=ts, vencimiento=te, liquidacion=tl,
        T_vol_dias=T_vol, T_desc_dias=T_desc, vol_pct=vol, rd_fwd_pct=rd_fwd, rf_fwd_pct=rf_fwd,
        forward=F, precio_unitario=precio, valor_presente_mxn=precio * q,
        delta_unitaria=delta, delta_posicion_usd=delta * q,
    )


# ----------------------------------------------------------------------------
# Carga de insumos
# ----------------------------------------------------------------------------
try:
    from config_privado import CONTRAPARTES_INTERNAS_OPCIONES as CONTRAPARTES_INTERNAS
except ImportError:
    CONTRAPARTES_INTERNAS = []


def cargar_insumos(data_dir: Path = DATA_DIR) -> dict:
    pos = pd.read_excel(data_dir / "Posicion_vigente.xlsx")
    pos = pos[(pos.TYPOLOGY == "Vanilla Option FXD")
              & (~pos.COUNTERPART.isin(CONTRAPARTES_INTERNAS))
              & (pos.TRN_GROUP == "OPT")
              & (pos.INSTRUMENT == "USD/MXN")]
    return {
        "posicion": pos,
        "curva_mxn": pd.read_excel(data_dir / "Curvas_tasas.xlsx", sheet_name="MXN_USD"),
        "curva_usd": pd.read_excel(data_dir / "Curvas_tasas.xlsx", sheet_name="USD"),
        "curva_vol": pd.read_excel(data_dir / "volatididad.xlsx", sheet_name="USD_MXN"),
        "inhabiles": list(pd.to_datetime(pd.read_csv(data_dir.parent / "comun" / "dias_inhabiles_mxn_usd.csv")["fecha"])),
    }


try:
    from config_privado import CONTRATOS_OPCIONES as CONTRATOS_PORTAFOLIO
except ImportError:
    CONTRATOS_PORTAFOLIO = None


def valuar_portafolio_opciones(insumos: dict | None = None, contratos=CONTRATOS_PORTAFOLIO,
                               spot: float | None = None) -> pd.DataFrame:
    ins = insumos or cargar_insumos()
    pos = ins["posicion"]
    sel = pos[pos.CONTRACT_ID.isin(contratos)]
    res = [valuar_opcion(op, ins["curva_mxn"], ins["curva_usd"], ins["curva_vol"], ins["inhabiles"], spot)
           for _, op in sel.iterrows()]
    return pd.DataFrame([asdict(r) for r in res])


def validar_contra_sistema(insumos: dict | None = None) -> pd.DataFrame:
    """Compara precio y delta contra los del sistema de valuación para toda la cartera USD/MXN."""
    ins = insumos or cargar_insumos()
    filas = []
    for _, op in ins["posicion"].iterrows():
        r = valuar_opcion(op, ins["curva_mxn"], ins["curva_usd"], ins["curva_vol"], ins["inhabiles"])
        delta_sis = op.DELTA_CURR_SWAPS_ES_FX_DELTA / op.LIVE_QUANTITY_SIGNED if op.LIVE_QUANTITY_SIGNED else np.nan
        filas.append({"contrato": r.contrato, "precio_calc": r.precio_unitario, "precio_sistema": op.UNITARY_PRICE,
                      "delta_calc": r.delta_unitaria, "delta_sistema": delta_sis})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    ins = cargar_insumos()
    print(valuar_portafolio_opciones(ins).T)
    v = validar_contra_sistema(ins)
    ok = v[v.precio_sistema.abs() > 1e-6]
    err_p = (ok.precio_calc / ok.precio_sistema - 1).abs() * 100
    err_d = (v.delta_calc / v.delta_sistema - 1).abs().replace(np.inf, np.nan)
    print(f"\nValidación contra el sistema ({len(v)} opciones USD/MXN):")
    print(f"  error de precio  |%|: mediana {err_p.median():.5f}  p95 {err_p.quantile(.95):.5f}  máx {err_p.max():.5f}")
    print(f"  error de delta   |rel|: mediana {err_d.median():.2e}  p95 {err_d.quantile(.95):.2e}")
