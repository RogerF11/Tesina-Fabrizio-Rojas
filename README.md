# Código de la tesina

Código de la tesina **"Análisis de pérdidas de un portafolio de tipo de cambio mediante metodologías de Valor en Riesgo"**
(Licenciatura en Actuaría, ITAM).

El código valúa un portafolio de derivados sobre el tipo de cambio USD/MXN (futuros, opciones vanilla y swaps cross
currency) y estima su Valor en Riesgo a un día con 95 % de confianza mediante tres métodos: Delta-Normal, simulación
histórica y simulación Monte Carlo.

* `valuacion/`: calculadoras de valuación de futuros, opciones y swaps.
* `var/`: estimación del VaR con cada método.
* `tablas.py`, `tablas_valuacion.py`, `graficas.py`: generan las tablas y gráficas de la tesina.
