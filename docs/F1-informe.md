# F1 de Phy-World (World4): población y capital humano endógenos

- **Fecha:** 2026-10-12
- **Estado:** construida y medida.
  - D1 y D2 cerrados en `api.query`: después del origen, ni la población ni el `hc` se leen del dato.
  - **W3: ÁMBAR.** No hay degradación (al contrario), pero el mutante declarado no lo pone en rojo.
  - **W4 (solo población): ROJO.** La causa está nombrada y medida.
  - **W5: VERDE.**
  - W2 sigue bit a bit.
- **Pruebas:** 73/73 (`pytest -q`, ~30 s); 13 son nuevas (12 de F1 y 1 de la API).
- **Validación:** `world4_cli.py --validate-f1` (49 orígenes × 4 modos, R = 32, unos 6 min con 6 procesos, más el *hindcast*).

## 1. Qué se construyó

| Pieza | Fichero | Qué hace |
|---|---|---|
| Datos WPP por edad | `world4/data/wpp.py`, `wpp_manifest.json` | Población por edad simple y sexo de 1950-2023; mortalidad por quinquenio y sexo; patrón de fecundidad; razón de sexos al nacer. Descarga de `PPgp/wpp2024` (CC BY 3.0 IGO) con SHA-256 y conversión a `data_cache/wpp2024/wpp_age.npz`. Los 180 países del L1 se enlazan por código M49. |
| Ajuste de ecuaciones | `world4/sectors/_panel.py` | MCO agrupado con covarianza robusta por país, restricciones de signo declaradas, efecto propio de cada país encogido, escala robusta del residuo, choque común por año AR(1) y choque propio por país AR(1) |
| Población | `world4/sectors/population.py` | Cohortes `(R, N, 2, 101)` con paso anual |
| Capital humano | `world4/sectors/human_capital.py` | Escolaridad por cohorte; `hc` como stock |
| Acoplamiento | `world4/core/coupling.py` | `CoupledSimulator`, subclase del `Simulator` del Atlas. `MemberSeries` sustituye a `pop` y `hc`: devuelve el dato hasta el origen y el estado `(R, N)` de cada miembro después. El bulto juvenil del MMP también sale de las cohortes. `coupling.forecast(pop_mode=observed / trend / endogenous)` |
| Validación | `world4/validate/rolling.py`, `hindcast.py` | W3/W5 (origen móvil) y W4-población (*hindcast* libre 1990→2023) |
| CLI | `world4_cli.py` | `--install-wpp` y `--validate-f1`; `--query population` y `--query human_capital` |
| Campaña | `world4/campaign.py` | W3, W4 y W5 medidos desde `runs/` |

### Ecuaciones

Todas se estiman con datos ≤ origen. Las variables explicativas son del año anterior.

- **Fecundidad:** Δln TFR depende de ln TFR, su cambio anterior, el ingreso, la escolaridad y la urbanización. Se reparte por edad con el patrón de WPP del último quinquenio cerrado.
- **Mortalidad:** Δe0 depende de e0, su cambio anterior, el ingreso, el crecimiento, el conflicto (≤ 0) y el calor (≤ 0). Las tasas por edad son las de WPP en el origen, multiplicadas por un factor común exp(κ). κ se resuelve por Newton para reproducir la e0 de cada miembro (error 4e-10).
- **Migración neta:** depende de su valor anterior, el ingreso relativo al mundo y el conflicto (≤ 0). Tiene un techo (percentil 99 del dato) y la suma mundial se fuerza a cero. El perfil de edad es fijo [S].
- **Urbanización:** Δlogit con ajuste parcial y el ingreso.
- **Escolaridad:**
  - `s` sale de invertir la fórmula de la PWT, ln hc = φ(s).
  - La ganancia por cohorte es b = f(s adulta, ingreso, canal r del L1). El signo del canal r es ≥ 0 y la estimación lo cumple.
  - La cohorte que cumple 25 años entra con la escolaridad nueva.
  - hc_t = hc_origen · exp(φ(s_t) − φ(s_origen)).
  - La política educativa llega al stock con el retardo de una generación.

Coeficientes con corte 1990, por ejemplo:

- conflicto → e0: −0,010 años/año (EE 0,062);
- canal r → ganancia de escolaridad: +0,33 años/año por unidad de I+D + educación / PIB (EE 0,20).

Los tamaños son plausibles, pero la evidencia es débil.

## 2. W3: origen móvil (49 orígenes, mismo protocolo que el Atlas, en esta máquina)

| | Atlas (población observada) | F0.5 (tendencia de 5 años) | **F1** | Cociente F1 / Atlas, IC95 por orígenes |
|---|---|---|---|---|
| CRPS log PIB pc, 1 año | 0,0637 | 0,0636 | 0,0636 | 0,998 [0,996, 1,001] |
| 5 años | 0,1191 | 0,1184 | 0,1183 | 0,993 [0,989, 0,998] |
| 10 años | 0,1805 | 0,1785 | 0,1778 | 0,985 [0,979, 0,993] |
| CRPS log población, 10 años | — (es el dato) | 0,0580 | **0,0388** | −33 % frente a la tendencia |
| Cobertura p10-p90 de la población por país, 1/5/10 años | — | 0,05 / 0,05 / 0,04 | 0,74 / 0,75 / 0,71 | nominal 0,80 |

- **Sin degradación.** El PIB pc mejora un poco a 5 y 10 años, más allá del ruido de semilla (Atlas contra Atlas con otra semilla: 0,995-0,999).
- **4 de 6 variables baten a su mejor referencia a 10 años:** PIB, rentas, deuda y GMST. Conflicto y democracia no lo hacen; tampoco lo hacía el Atlas.
- **Test de fuga: el mutante no funciona.** Volver a la población observada (el Atlas) no mejora el CRPS del PIB pc; lo empeora en 0,2-1,5 %. Esa métrica no ve esta fuga: el PIB pc es casi insensible a errores de población de pocos puntos.
  - La fuga se vigila con una prueba de invariancia: multiplicar por 3 la población y por 0,5 el `hc` posteriores al origen deja el pronóstico de F1 idéntico bit a bit, y el mutante sí cambia.
  - Por eso W3 queda en **ÁMBAR**: el listón se cumple, pero el criterio no demuestra que vea la fuga.

## 3. W4 (solo población): *hindcast* libre 1990 → 2023

Los parámetros, la conducta y la demografía se estiman con datos ≤ 1990. Después no se reancla nada. R = 64.

| Año | Dato (miles de millones) | p10 | p50 | p90 | Tendencia de F0.5 |
|---|---|---|---|---|---|
| 1999 | 5,971 | 6,011 | 6,039 | 6,087 | 6,173 |
| 2008 | 6,677 | 6,719 | 6,800 | 6,917 | 7,303 |
| 2017 | 7,418 | 7,379 | 7,525 | 7,739 | 8,697 |
| 2023 | 7,836 | 7,746 | 7,940 | 8,255 | 9,805 |

- **La mediana acaba en +1,3 %** (la tendencia de F0.5, en +25 %). El peor año se aparta 1,9 %.
- **Dentro de la banda solo el 30 % de los años** (listón: 80 %) → **ROJO**.
- **Causa medida** con una descomposición, poniendo un impulsor observado cada vez y sin ruido:

| Error mundial | 1995 | 2000 | 2005 | 2010 | 2019 |
|---|---|---|---|---|---|
| Modelo | +0,6 % | +1,4 % | +2,0 % | +2,1 % | +1,5 % |
| TFR observada | −0,3 % | −0,6 % | −0,9 % | −1,3 % | −2,7 % |
| e0 observada | +0,5 % | +1,3 % | +2,0 % | +2,3 % | +2,4 % |
| Migración observada | +0,6 % | +1,4 % | +2,0 % | +2,2 % | +1,8 % |
| Todo observado | −0,4 % | −0,6 % | −0,8 % | −0,9 % | −1,6 % |

Hay dos causas:

1. **La ecuación de fecundidad,** estimada hasta 1990, no anticipa la caída rápida de los noventa.
2. **Las cohortes alimentadas con los impulsores observados quedan por debajo de la población PWT,** que es la verdad del L1. Ese sesgo contable llega a −1,6 % en 29 años. Posibles fuentes: la PWT y WPP no coinciden en crecimiento, las tasas por edad del quinquenio 1985-1990 quedan fijas en la forma, y el perfil de migración es supuesto.

La banda es estrecha al principio porque la población parte del dato y la incertidumbre de las tasas solo se acumula año a año. Un sesgo sistemático de +0,5 % en 1995 ya basta para quedar fuera.

## 4. W5: cero recalibración

No se recalibró nada del L1. Ninguna de las 15 celdas (conflicto, rentas, deuda, democracia y GMST, a 1, 5 y 10 años) se mueve fuera de su banda, que es el máximo entre 2 % y el ruido de semilla. El mayor movimiento es el de rentas a 5 años, con cociente 0,995. **VERDE.**

## 5. Desviaciones del plan y decisiones

- **Edad simple, no quinquenios.** El estado lleva 101 edades. Las tasas sí van por quinquenio, como pide el plan. Es un superconjunto de lo pedido.
- **El estimador estructural de F0.5 no cambia.** Sigue con la tendencia de 5 años y el almacén no se reconstruye, así que W8 y W9a no se mueven. Pasarlo al L1 con F1 es trabajo de F11. Con el 33 % menos de error de población, es un candidato claro.
- **La mano de obra del L1 sigue siendo la población total, no la de 15-64 años.** Cambiarla movería la función de producción y obligaría a recalibrar, y eso choca con W5. El dividendo demográfico queda fuera.
- **Términos ausentes, declarados en `MISSING_TERMS` y en los avisos de la API:**
  - alimento → mortalidad (F5);
  - desastres → mortalidad (F4: el L1 no expone sus desastres por miembro).
- **Supuestos [S]:**
  - el perfil lineal de escolaridad por cohorte en el origen;
  - el perfil de edad de los migrantes;
  - riesgos proporcionales en la mortalidad.
- **Colas no simuladas.** El ruido usa una escala robusta (MAD); genocidios, hambrunas y éxodos quedan fuera. Por eso la cobertura de población por país es del 71-75 % y no del 80 %.
- **Fuga de revisión de WPP 2024.** La estructura por edad y las tasas de 1990 son historia revisada en 2024. No hay ediciones antiguas cargadas para medirla.
- **Una aproximación del Atlas cambia de significado (declarado en `coupling.py`).** La frontera de `hc` (cuantil 90 entre países) se calcula con todos los miembros juntos.
- **Dependencia nueva solo para instalar:** `rdata` (extra `world4[wpp]`).

## 6. Qué sigue

1. **Fecundidad.** Es la causa del rojo de W4. Opciones: una forma de transición (logística en el tiempo, como la de Bayesian TFR de la ONU) en lugar del ajuste parcial lineal, o fecundidad por educación de la madre al estilo Wittgenstein. Se mide con el mismo *hindcast*, sin tocarlo con datos posteriores a 1990.
2. **Sesgo contable entre cohortes y PWT.** Escalar con la diferencia PWT/WPP del último quinquenio, o resolver contra WPP y reescalar solo la salida.
3. **Pasar el estimador de F0.5 a la población de F1 (F11)** y volver a medir W8 y W9a.
