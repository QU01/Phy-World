"""L0 vigente: World3-03 con las correcciones que piden los datos, una por fase.

V1: la población deja de ser cuatro cajas y pasa a cohortes por edad y sexo (``cohorts``, ``lifetable``),
acopladas a ``world4.core.l0`` por su gancho (``model``).
V2: el mundo se parte en grupos de países (``groups``) con esperanza de vida y fecundidad estimadas con datos
(``equations``); la historia hasta un año origen es la observada y desde ahí corre el modelo.
El escenario 1 original no cambia.
"""
