# -*- coding: utf-8 -*-
"""
Valuación de futuros USD/MXN listados en MexDer.

Convenciones:
* Tasas de las curvas en porcentaje anual SIMPLE, base Actual/360.
* t0 = fecha de valuación; ts = t0 + 2 días hábiles (fecha spot);
  te = vencimiento; tl = te + 2 días hábiles (liquidación).
  Días hábiles con calendario conjunto MXN + USD.
* Tasas forward simples de ts a tl, igual que en opciones.
* Precio forward teórico (no arbitraje):
      F = S * (1 + r_d,fwd T'/36000) / (1 + r_f,fwd T'/36000),   T' = tl - ts
* Valor de la posición (MXN):  V = (F - P0) * Q * N,  N = 10,000 USD por contrato,
  Q con signo (+ larga, - corta), P0 = precio pactado.
* Delta de la posición (USD equivalentes):  dV/dS = Q * N * F / S.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from valuacion.opciones import fecha_mas_dias_habiles, interpolar_lineal, tasa_forward_simple, BASE_TASAS

DATA = Path(__file__).resolve().parents[1] / "data"
TAMANO_CONTRATO_USD = 10_000
try:
    from config_privado import CONTRATOS_FUTUROS as CONTRATOS_PORTAFOLIO
except ImportError:
    CONTRATOS_PORTAFOLIO = None


@dataclass
class ResultadoFuturo:
    contrato: int
    posicion: str
    contratos: float
    precio_pactado: float
    spot: float
    fecha_valuacion: pd.Timestamp
    fecha_spot: pd.Timestamp
    vencimiento: pd.Timestamp
    liquidacion: pd.Timestamp
    T_dias: int
    rd_fwd_pct: float
    rf_fwd_pct: float
    forward: float
    valor_presente_mxn: float
    delta_posicion_usd: float


def leer_inhabiles_mxn_usd(ruta: Path = DATA / "comun" / "dias_inhabiles_mxn_usd.csv") -> list:
    return list(pd.to_datetime(pd.read_csv(ruta)["fecha"]))


def valuar_futuro(fut: pd.Series, curva_mxn: pd.DataFrame, curva_usd: pd.DataFrame, inhabiles,
                  spot: float | None = None, N: float = TAMANO_CONTRATO_USD) -> ResultadoFuturo:
    """Valúa un futuro. Si `spot` es None usa MARKET_SPOT del archivo de posición."""
    S = float(fut.MARKET_SPOT if spot is None else spot)
    t0 = pd.Timestamp(fut.FCH_DATAMART)
    te = pd.Timestamp(fut.DATE_PERIOD_EXPIRY_DATE)
    ts = fecha_mas_dias_habiles(t0, 2, inhabiles)
    tl = fecha_mas_dias_habiles(te, 2, inhabiles)
    d_s, d_l = (ts - t0).days, (tl - t0).days
    T = d_l - d_s
    rd = tasa_forward_simple(interpolar_lineal(curva_mxn, d_s), interpolar_lineal(curva_mxn, d_l), d_l, d_s)
    rf = tasa_forward_simple(interpolar_lineal(curva_usd, d_s), interpolar_lineal(curva_usd, d_l), d_l, d_s)
    F = S * (1 + rd * T / BASE_TASAS) / (1 + rf * T / BASE_TASAS)
    Q = float(fut.LIVE_QUANTITY_SIGNED)
    return ResultadoFuturo(
        contrato=int(fut.CONTRACT_ID), posicion="larga" if Q > 0 else "corta", contratos=Q,
        precio_pactado=float(fut.PRECIO_INICIAL), spot=S, fecha_valuacion=t0, fecha_spot=ts,
        vencimiento=te, liquidacion=tl, T_dias=T, rd_fwd_pct=rd, rf_fwd_pct=rf, forward=F,
        valor_presente_mxn=(F - float(fut.PRECIO_INICIAL)) * Q * N,
        delta_posicion_usd=Q * N * F / S,
    )


def cargar_insumos(data_dir: Path = DATA) -> dict:
    x = pd.ExcelFile(data_dir / "futuros" / "Insumos.xlsx")
    pos = pd.read_excel(x, "POS")
    return {
        "posicion": pos[pos.INSTRUMENT == "MXD USDMXN"],
        "curva_mxn": pd.read_excel(x, "Tasa Local", header=None),
        "curva_usd": pd.read_excel(x, "Tasa Extranjera", header=None),
        "inhabiles": leer_inhabiles_mxn_usd(data_dir / "comun" / "dias_inhabiles_mxn_usd.csv"),
    }


def valuar_portafolio_futuros(insumos: dict | None = None, contratos=CONTRATOS_PORTAFOLIO,
                              spot: float | None = None) -> pd.DataFrame:
    ins = insumos or cargar_insumos()
    sel = ins["posicion"]
    if contratos is not None:
        sel = sel[sel.CONTRACT_ID.isin(contratos)]
    return pd.DataFrame([asdict(valuar_futuro(f, ins["curva_mxn"], ins["curva_usd"], ins["inhabiles"], spot))
                         for _, f in sel.iterrows()])


def validar_contra_sistema(insumos: dict | None = None) -> pd.DataFrame:
    ins = insumos or cargar_insumos()
    df = valuar_portafolio_futuros(ins, contratos=None)
    df["valor_sistema"] = ins["posicion"]["VAL_DE_LOS_DERIVADOS_BM"].to_numpy()
    df["dif_pct"] = (df.valor_presente_mxn / df.valor_sistema - 1) * 100
    return df[["contrato", "vencimiento", "forward", "valor_presente_mxn", "valor_sistema", "dif_pct"]]


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    ins = cargar_insumos()
    print(valuar_portafolio_futuros(ins).T)
    print("\nValidación contra el sistema:")
    print(validar_contra_sistema(ins).to_string(index=False))
