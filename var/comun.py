# -*- coding: utf-8 -*-
"""Insumos comunes a los tres métodos de VaR: serie FIX y deltas del portafolio."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "comun"
FECHA_VALUACION = pd.Timestamp("2024-04-30")
INICIO_VENTANA = pd.Timestamp("2019-04-30")
SPOT_DN = 17.1268        # S0 del Delta-Normal: tipo de cambio de cierre con el que se valúa el portafolio


def cargar_fix(ruta: Path = DATA / "FIX 0419-0424.xlsx", inicio=INICIO_VENTANA, fin=FECHA_VALUACION) -> pd.Series:
    """FIX de Banxico por FECHA DE DETERMINACIÓN (serie SF43718), solo días hábiles, en la ventana [inicio, fin]."""
    d = pd.read_excel(ruta, skiprows=17)[["Fecha", "SF43718"]]
    d["Fecha"] = pd.to_datetime(d["Fecha"], errors="coerce")
    d["SF43718"] = pd.to_numeric(d["SF43718"], errors="coerce")
    d = d.dropna().sort_values("Fecha")
    d = d[(d.Fecha >= inicio) & (d.Fecha <= fin)]
    return d.set_index("Fecha")["SF43718"].rename("fix")


def log_rendimientos(fix: pd.Series) -> pd.Series:
    return np.log(fix / fix.shift(1)).dropna().rename("r")


def portafolio(spot_valuacion: float | None = None) -> pd.DataFrame:
    """Los 9 instrumentos de la tesina con valor presente (MXN) y delta de la posición (USD)."""
    from valuacion import opciones, futuros, swaps
    o = opciones.valuar_portafolio_opciones(spot=spot_valuacion)
    f = futuros.valuar_portafolio_futuros(spot=spot_valuacion)
    s = swaps.valuar_portafolio_swaps(spot=spot_valuacion)
    filas = []
    for r in o.itertuples():
        filas.append(("Opción " + ("Put" if r.tipo == "P" else "Call"), r.contrato, r.posicion, r.valor_presente_mxn, r.delta_posicion_usd))
    for r in s.itertuples():
        filas.append(("Swap USD/MXN", r.nb, ("recibe" if r.pata_usd == "recibe" else "paga") + " USD", r.valor_presente_mxn, r.delta_usd))
    for r in f.itertuples():
        filas.append(("Futuro", r.contrato, r.posicion, r.valor_presente_mxn, r.delta_posicion_usd))
    return pd.DataFrame(filas, columns=["instrumento", "contrato", "posicion", "valor_presente_mxn", "delta_usd"])


def revaluador():
    """
    Devuelve (S0, V0, f) donde f(S) da el valor por clase para un arreglo de spots S.
    Revaluación completa y exacta:
      * opciones: Black sobre el forward, repreciada para cada S (curvas y volatilidad fijas);
      * futuros:  V = (F0 * S/S0 - P0) * Q * N          (F es proporcional a S)
      * swaps:    V = e * (S * VP_USD - VP_MXN)          (VP_USD y VP_MXN no dependen de S)
    Las dos últimas son exactas, no aproximaciones: el valor de futuros y swaps es lineal en S.
    """
    from valuacion import opciones, futuros, swaps
    o = opciones.valuar_portafolio_opciones()
    f = futuros.valuar_portafolio_futuros()
    s = swaps.valuar_portafolio_swaps()
    S0 = float(s.spot.iloc[0])
    assert np.allclose(o.spot, S0) and np.allclose(f.spot, S0)

    def valor(S):
        S = np.atleast_1d(np.asarray(S, dtype=float))
        v_o = sum(r.nocional_usd * opciones.precio_vectorizado(r.tipo, S, r.strike, r.vol_pct, r.rd_fwd_pct,
                                                               r.rf_fwd_pct, r.T_desc_dias, r.T_vol_dias)
                  for r in o.itertuples())
        v_f = sum((r.forward * S / S0 - r.precio_pactado) * r.contratos * futuros.TAMANO_CONTRATO_USD
                  for r in f.itertuples())
        e = np.where(s.pata_usd == "recibe", 1.0, -1.0)
        v_s = sum(ei * (S * r.vp_usd - r.vp_mxn) for ei, r in zip(e, s.itertuples()))
        return pd.DataFrame({"opciones": v_o, "swaps": v_s, "futuros": v_f})

    V0 = valor(S0).iloc[0]
    assert np.isclose(V0.opciones, o.valor_presente_mxn.sum()) and np.isclose(V0.futuros, f.valor_presente_mxn.sum()) \
        and np.isclose(V0.swaps, s.valor_presente_mxn.sum())
    return S0, V0, valor
