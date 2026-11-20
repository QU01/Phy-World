"""L2: la capa de eventos. Mensual, por celda de 0,5° (PRIO-GRID), con actores armados, díadas y bloques de
población; produce eventos con el mismo esquema que los datos (UCDP GED) para poder puntuarse contra ellos.

El nombre «L2» lo dejó libre la hoja de ruta: FaIR y los cultivos, que iban a ser un L2 aparte, viven dentro
del L1 desde F4/F5. Esta fase es F12 (F11 sigue reservada al estimador estructural sobre el L1 completo).

    regions       qué países forman cada una de las cinco regiones de UCDP; ``world`` son las cinco
    panel         el panel celda-mes y país-mes, las díadas y los actores, cortado «a fecha de»
    baseline      la vara: referencias sin modelo y un modelo estadístico de valla (país-mes y celda-mes)
    agents        el simulador de actores, díadas, huella espacial y nacimiento de díadas
    displacement  los bloques de población: huida, destino y retorno
    forecast      ensembles desde un origen; puente con el L1 en las dos direcciones

Todo lo que sale de aquí es ``exploratory`` salvo los horizontes y niveles que W13 midió fuera de muestra.
"""
