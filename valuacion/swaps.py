# -*- coding: utf-8 -*-
"""
Valuación de swaps cross currency USD/MXN 1M-1M de tasa fija contra tasa fija.

Convenciones:
* Pata 1 en USD, pata 2 en MXN. Tasas en porcentaje anual SIMPLE, base Actual/360.
* Cada flujo = amortización de capital + interés (capital vigente x tasa x días/360).
* Cada flujo se descuenta con su curva (USD o MXN) de la fecha de pago a t0 y se lleva
  a la fecha spot ts = t0 + 2 días hábiles dividiendo entre el factor de descuento a ts.
  VP_USD y VP_MXN son SIN signo (valor de los flujos que se reciben o pagan).
* Valor del swap en MXN, con el tipo de cambio spot S de la fecha de valuación:
      V = e * (S * VP_USD - VP_MXN),   e = +1 si se recibe la pata USD, -1 si se paga.
* Delta (derivada exacta de V respecto a S, en USD):   Delta = e * VP_USD.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "swaps"
PRODUCTO = "USD MXN 1M 1M N"
BASE = 36000.0

try:
    from config_privado import CONTRATOS_SWAPS as CONTRATOS_PORTAFOLIO
except ImportError:
    CONTRATOS_PORTAFOLIO = None


def interpolar_curva(curva: pd.DataFrame, dias: float) -> float:
    """Interpolación lineal sobre (Dias, tasa) usando solo nodos con plazo > 0."""
    c = curva[curva.iloc[:, 0] > 0]
    x = c.iloc[:, 0].to_numpy(float)
    y = c.iloc[:, 1].to_numpy(float)
    i = np.where(x <= dias)[0][-1]
    return float(y[i] + (y[i + 1] - y[i]) / (x[i + 1] - x[i]) * (dias - x[i]))


def fecha_spot(t0: pd.Timestamp, feriados) -> pd.Timestamp:
    habiles = sorted(set(pd.date_range(t0, t0 + pd.DateOffset(days=30), freq="B")) - set(pd.to_datetime(list(feriados))))
    return habiles[2]   # tercer día hábil del rango, contando t0


def vp_pata(flujos_pata: pd.DataFrame, curva: pd.DataFrame, t0: pd.Timestamp, factor_spot: float) -> tuple[float, int]:
    """VP (a fecha spot) de los flujos de una pata y número de flujos pendientes."""
    P = flujos_pata.reset_index(drop=True)
    monto = P.REMAINING_CAPITAL.to_numpy(float)
    tasa = float(P.FIXED_RATE.iloc[0] - P.MARGIN.iloc[0])
    vp = 0.0
    for i in range(len(P)):
        d_desde = (pd.Timestamp(P.START_DATE[i]) - t0).days
        d_hasta = max(0, (pd.Timestamp(P.END_DATE[i]) - t0).days)
        capital = monto[i] - monto[i + 1] if i < len(P) - 1 else monto[i]
        pago = capital + monto[i] * tasa * (d_hasta - d_desde) / BASE
        r = interpolar_curva(curva, 4 if d_hasta <= 1 else d_hasta)
        vp += pago / (1 + r * d_hasta / BASE) / factor_spot
    return vp, len(P)


@dataclass
class ResultadoSwap:
    nb: int
    contrato: int
    pata_usd: str              # "recibe" o "paga"
    plazo_remanente_dias: int
    flujos_usd: int
    flujos_mxn: int
    nocional_usd: float
    nocional_mxn: float
    tasa_usd_pct: float
    tasa_mxn_pct: float
    spot: float
    vp_usd: float
    vp_mxn: float
    valor_presente_mxn: float
    delta_usd: float


def cargar_insumos(data_dir: Path = DATA) -> dict:
    cur = pd.ExcelFile(data_dir / "Curvas.xlsx")
    fx = pd.read_excel(data_dir / "USDMXN Cierre.xlsx")
    pos = pd.read_csv(data_dir / "posicion_ccs_usdmxn_1m1m.csv", parse_dates=["FCH_DATAMART", "DATE_PERIOD_EXPIRY_DATE"])
    flu = pd.read_csv(data_dir / "flujos_ccs_usdmxn_1m1m.csv", parse_dates=["START_DATE", "END_DATE"])
    t0 = pd.Timestamp(pos.FCH_DATAMART.iloc[0])
    return {
        "posicion": pos, "flujos": flu, "t0": t0,
        "curva_usd": pd.read_excel(cur, "CURVA DCTO USD")[["Dias", PRODUCTO]],
        "curva_mxn": pd.read_excel(cur, "CURVA DCTO MXN")[["Dias", PRODUCTO]],
        "spot": float(fx[fx["Fecha diaria"] == t0.strftime("%d/%m/%Y")].iloc[0, 1]),   # cierre de jornada
        # calendario conjunto México + EE. UU. para la fecha spot
        "feriados_mxn": pd.read_csv(data_dir.parent / "comun" / "dias_inhabiles_mxn_usd.csv")["fecha"],
    }


def valuar_swap(fila: pd.Series, ins: dict, spot: float | None = None) -> ResultadoSwap:
    S = ins["spot"] if spot is None else spot
    t0 = ins["t0"]
    ts = fecha_spot(t0, ins["feriados_mxn"])
    d_s = (ts - t0).days
    fs_usd = 1 / (1 + interpolar_curva(ins["curva_usd"], d_s) * d_s / BASE)
    fs_mxn = 1 / (1 + interpolar_curva(ins["curva_mxn"], d_s) * d_s / BASE)
    fl = ins["flujos"][ins["flujos"].DEAL == fila.NB]
    p1, p2 = fl[fl.LEG == "LEG 1"], fl[fl.LEG == "LEG 2"]
    vp_usd, n1 = vp_pata(p1, ins["curva_usd"], t0, fs_usd)
    vp_mxn, n2 = vp_pata(p2, ins["curva_mxn"], t0, fs_mxn)
    e = 1.0 if fila.RT_PAY_REC_1ST_LEG_FIXRATE == "R" else -1.0
    return ResultadoSwap(
        nb=int(fila.NB), contrato=int(fila.CONTRACT_ID), pata_usd="recibe" if e > 0 else "paga",
        plazo_remanente_dias=int((pd.Timestamp(fl.END_DATE.max()) - t0).days), flujos_usd=n1, flujos_mxn=n2,
        nocional_usd=float(p1.REMAINING_CAPITAL.iloc[0]), nocional_mxn=float(p2.REMAINING_CAPITAL.iloc[0]),
        tasa_usd_pct=float(p1.FIXED_RATE.iloc[0] - p1.MARGIN.iloc[0]), tasa_mxn_pct=float(p2.FIXED_RATE.iloc[0] - p2.MARGIN.iloc[0]),
        spot=S, vp_usd=vp_usd, vp_mxn=vp_mxn,
        valor_presente_mxn=e * (S * vp_usd - vp_mxn), delta_usd=e * vp_usd,
    )


def valuar_portafolio_swaps(ins: dict | None = None, contratos=CONTRATOS_PORTAFOLIO, spot: float | None = None) -> pd.DataFrame:
    ins = ins or cargar_insumos()
    pos = ins["posicion"]
    if contratos is not None:
        pos = pos.set_index("NB").loc[contratos].reset_index()
    return pd.DataFrame([asdict(valuar_swap(f, ins, spot)) for _, f in pos.iterrows()])


def validar_contra_sistema(ins: dict | None = None) -> pd.DataFrame:
    ins = ins or cargar_insumos()
    df = valuar_portafolio_swaps(ins, contratos=None)
    sis = ins["posicion"].set_index("NB")
    df["valor_sistema"] = df.nb.map(sis.DISCOUNTED_MARKET_VALUE)
    df["delta_sistema"] = df.nb.map(sis.DELTA_CURR_SWAPS_ES_FX_DELTA)
    df["dif_valor_pct"] = (df.valor_presente_mxn / df.valor_sistema - 1) * 100
    df["dif_delta_pct"] = (df.delta_usd / df.delta_sistema - 1) * 100
    return df[["nb", "pata_usd", "valor_presente_mxn", "valor_sistema", "dif_valor_pct", "delta_usd", "delta_sistema", "dif_delta_pct"]]


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    ins = cargar_insumos()
    print(valuar_portafolio_swaps(ins).T)
    print("\nValidación contra el sistema:")
    print(validar_contra_sistema(ins).to_string(index=False))
