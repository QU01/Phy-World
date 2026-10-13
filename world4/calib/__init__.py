"""F7: calibración bayesiana e incertidumbre.

* ``gp``: emulador GP (Matérn 5/2 con ARD, media lineal, krigeado universal) en NumPy.
* ``history_matching``: oleadas de *history matching* (Vernon, Goldstein y Bower 2010; Andrianakis et
  al. 2015) con la implausibilidad del emulador.
* ``abc``: ABC-SMC (Beaumont et al. 2009) con el error del modelo de Wilkinson (2013) sobre la región
  no descartada.
* ``discrepancy``: discrepancia AR1 por variable (``phycore.uq.DiscrepancyGP``) con su marca.
* ``l0r``: el L0 regional calibrado (posterior de producción y comprobación fuera de muestra).
* ``l1``: el L1 calibrado con pseudo-hindcasts antes del origen (W4).
* ``posterior``: ``models/posterior.npz``, el posterior que viaja con el producto.
"""
