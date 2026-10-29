"""Predicción de conflicto y eventos de violencia organizada en celdas discretas del mapa (PRIO-GRID, 0,5°).

Pregunta que responde: ¿cuál es la probabilidad de que en esta celda de ~55 x 55 km haya al menos un evento
de violencia organizada (estatal ``sb``, no estatal ``ns``, unilateral contra civiles ``os``, o cualquiera)
en cada uno de los próximos 12 meses, cuántos eventos y cuántas muertes estatales, y por qué?

Diseño (detalle y referencias en ``docs/F11-informe.md``):

* **Unidad y datos** como ViEWS ``pgm`` (Hegre et al. 2019, 2021): celda PRIO-GRID 2.0 x mes, eventos de
  UCDP GED 26.1 + UCDP Candidate (``world4.data.conflict_grid``). A diferencia de ViEWS: mundo entero, los
  tres tipos de violencia, y pronóstico de eventos además de muertes.
* **Dos modelos de familias distintas, promediados** (``forecast``): un umbral logístico directo por horizonte
  con historia del conflicto, difusión espacial y geografía (``Logistic``; la parte umbral de los hurdle de
  ViEWS y de Fritz et al. 2024) y un proceso de Hawkes espacio-temporal en tiempo discreto con fondo
  estructural, excitación propia, de los anillos 1-2 y del país, cruzada entre tipos (``Hawkes``; Browning
  et al. 2025, Held et al. 2005), estimado por EM y simulado hacia delante.
* **Muertes** = P(evento estatal) x tamaño log-normal (``Size``).
* **Validación** (``validate``) con origen rodante 2019-2025 contra los modelos de referencia del reto ViEWS
  2023/24 (``benchmarks``), con Brier, AUPR y CRPS (``metrics``), aparte en celdas sin historia (el "problema
  difícil" de Mueller y Rauh 2022), y 2026 contra los eventos candidatos.
* **Salida** para la interfaz (``service.payload``): JSON por columnas con probabilidades por celda y mes,
  eventos y muertes esperados, la descomposición fondo / contagio y los agregados por país.
"""
