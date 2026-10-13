# F2 de Phy-World (World4): escasez global endógena

- **Fecha:** 2026-10-12
- **Estado:** construida y medida. D3 cerrado: `psi_t` (rentas de recursos del mundo / PIB mundial) ya no entra como dato después del origen; es una **salida** de cuatro mercados mundiales. `api.query` usa F1+F2 (`MODEL_REVISION` 3).
  - **W3: ÁMBAR** (la misma salvedad que en F1). Con F2 mejoran el PIB, las rentas por país y la `psi`.
  - **W4: ROJO** por la población de F1. La `psi` cumple su listón de F2 desde 1990 (72 % ≥ 70 %), pero no desde 1980 (54 %).
  - **W5: VERDE** (30/30 celdas).
  - **W6: ÁMBAR.** La relación de F2 se cumple; las demás esperan a F6.
  - W2 sigue bit a bit.
- **Pruebas:** 79/79; 6 son nuevas (5 de F2 y 1 de la API).
- **Validación:** `world4_cli.py --validate` (49 orígenes × 6 modos más 3 *hindcasts*, unos 12 min).

## 1. Qué se construyó

| Pieza | Fichero | Qué hace |
|---|---|---|
| Mercados por clase | `world4/sectors/resources.py` | Petróleo, gas, carbón y metales: demanda, capacidad de agotamiento, precio y renta por miembro del ensemble |
| Ajuste | `resources.fit` | Ecuaciones estimadas con datos ≤ origen (covarianza MCO para series mundiales en `_panel.fit`) |
| Acoplamiento | `world4/core/coupling.py` | `F2Config`. La subclase sustituye `_psi(t)` del Atlas por la `psi` de los mercados y **normaliza** la Cobb-Douglas (§3). El PIB mundial de cada miembro alimenta la demanda |
| Validación | `rolling.py` (modos `f2` y `f2_persist`, métrica `psi`), `hindcast.run_psi` | W3, W5 y W4-`psi` |
| API | `api.query` | F1+F2 por defecto; variables nuevas `resource_rent_share_world` y `oil_price` (WLD) |
| Campaña | `campaign.py` | W3 con las rentas; W4 con la `psi`; W5 por capas; W6 medido en parte |

### El mercado de cada clase, cada año después del origen

- **Demanda:** Δln Q = gY + Δln(Q/PIB).
  - La intensidad responde al precio del año anterior y a su propio cambio anterior.
  - Se estima con los **20 años previos al origen**.
  - La elasticidad renta es 1 [S]. Con ella libre, el signo salía al revés y la constante absorbía el crecimiento de 1950-1990: la demanda de petróleo crecía al 4 % anual.
  - gY es el crecimiento del PIB mundial del propio miembro del L1, con un año de retraso.
- **Capacidad:** no se extrae más del 10 % del recurso que queda por año [S]. El recurso último es el acumulado en el origen más las reservas conocidas en el origen por un factor de crecimiento de reservas g ~ log-uniforme [1, 3] [S].
  - Si no hay reservas en el origen (antes de 1980, o el carbón siempre): 30 años de producción para el petróleo, 40 para el gas y 150 para el carbón [S].
  - Los metales no tienen agotamiento (no hay recurso último).
- **Precio:**
  - AR(1) **estacionario** alrededor de la media histórica hasta el origen, sin deriva y con φ ≤ 0,95 [S].
  - Ruido con la desviación típica completa: los choques del petróleo son la historia, no colas.
  - Prima de escasez κ·ln(Qd/Cap) cuando la demanda supera la capacidad [S].
- **Renta:** Δln ψ_c = θ·Δln p + Δln(Q/PIB), sin constante, más una desviación de nivel AR(1) estacionaria (ρ = 0,5, escala robusta acotada a 0,5) [S].
- **Total:** `psi` = Σ ψ_c + la renta forestal, congelada en su media de 10 años (renovable).

Coeficientes con corte 2019, por ejemplo:

- θ = 0,92 (petróleo), 1,13 (gas), 2,05 (carbón) y 1,70 (metales);
- φ del precio = 0,93 (petróleo) y 0,81 (carbón);
- la elasticidad-precio de la demanda de petróleo, de carbón y de metales sale con el signo contrario y queda en 0 (restricción declarada). Por eso el pico de largo plazo lo produce la capacidad, no el precio.

## 2. Tres correcciones que el dato obligó a hacer, y por qué

| Síntoma | Causa | Corrección |
|---|---|---|
| Con datos ≤ 1990, la linealización de Hubbert da un recurso último del petróleo de 1,53e6 TWh. Hasta 2019 se extrajeron 2,35e6 | Crecimiento de reservas: el clásico fallo de Hubbert antes del pico | El recurso último no se ajusta: son las reservas conocidas en el origen por un factor incierto |
| Con datos ≤ 1980, el precio sale como raíz unitaria con deriva de +7 % anual y la `psi` explota (0,5 en 2010) | Los años setenta dominan la muestra | Precio estacionario sin deriva; demanda con los 20 años previos |
| El índice de metales cambiaba ligeramente al alterar datos de 2010 (la prueba de fuga lo cazó) | Ponderaba con precios y cantidades de **2010** | Año base del índice = el origen |

Además, en 1970 el carbón, el gas y los metales no tienen 10 años de precios. Esas clases quedan **congeladas** en su renta media y la corrida no se rompe (antes producía NaN en todo el L1).

## 3. La normalización: un hallazgo sobre el Atlas

En el Atlas, `psi` es el **exponente** de una Cobb-Douglas con insumos en niveles absolutos. Medido: subir `psi` en 0,01 sube el PIB per cápita de **todos** los países un 5,3 % de golpe (mediana; hasta 8,6 %). Es un efecto de unidades, no económico. El Atlas lo arrastra en silencio de dos formas:

- **En la historia,** porque `psi` cambia cada año (de 0,0075 en 1950 a 0,076 en 1980).
- **En cada pronóstico,** porque en el origen salta de la `psi` observada a su media de 10 años.

F2 añade en la subclase un desplazamiento de la TFP que anula ese efecto a insumos dados: es la normalización local de De La Grandville.

- **Con normalización:** +0,01 de `psi` baja el PIB per cápita de los importadores un 0,5 % (mediana) y sube el de los exportadores hasta un 3,7 %, que es la economía que se busca.
- **Identidad (prueba):** si la `psi` del mercado es igual a la del Atlas, el resultado es el Atlas a 3,6e-15.

## 4. W3: origen móvil (49 orígenes, R = 32)

| | Atlas | F1 | Ablación: `psi` del origen + normalización | **F1 + F2** | F2/Atlas, IC95 |
|---|---|---|---|---|---|
| CRPS log PIB pc, 1 año | 0,0637 | 0,0636 | 0,0565 | **0,0563** | 0,883 [0,868, 0,899] |
| 5 años | 0,1191 | 0,1183 | 0,1136 | **0,1133** | 0,951 [0,937, 0,964] |
| 10 años | 0,1805 | 0,1778 | 0,1764 | **0,1748** | 0,969 [0,952, 0,987] |
| RMSE rentas por país, 1 año | 0,0493 | 0,0492 | 0,0389 | **0,0382** | 0,776 [0,695, 0,847] |
| 5 años | 0,0663 | 0,0660 | 0,0637 | **0,0580** | 0,875 [0,831, 0,919] |
| 10 años | 0,0680 | 0,0679 | 0,0735 | **0,0641** | 0,942 [0,901, 0,982] |
| CRPS `psi`, 1 / 5 / 10 años | 0,014 / 0,019 / 0,018 | igual | 0,008 / 0,016 / 0,020 | **0,006 / 0,011 / 0,011** | mejor referencia: 0,007 / 0,015 / 0,018 |
| Cobertura p10-p90 de `psi` | — | — | — | 0,57 / 0,64 / 0,65 | nominal 0,80 |

**De dónde viene cada mejora** (la ablación lo separa):

- **PIB:** casi toda la ganancia viene de **quitar el salto de `psi` del Atlas en el origen** (0,886 / 0,954 / 0,978 sin mercados). Los mercados añaden poco (0,969 a 10 años).
- **Rentas y `psi` a 5-10 años:** aquí sí trabajan los mercados. Sin ellos, las rentas a 10 años empeoran (1,08) y la `psi` a 10 años es peor que la del Atlas.
- **Conflicto, deuda, democracia y GMST:** no se mueven (W5).

## 5. W4 (parte de F2): la `psi` en *hindcast* libre

| Origen → 2019 | Dentro de p10-p90 | MAE de la mediana | MAE de la media de 10 años (Atlas) |
|---|---|---|---|
| 1980 | 54 % | 0,0160 | 0,0143 |
| **1990** | **72 %** (listón 70 %) | 0,0131 | 0,0104 |

- El listón de F2 decía «1970-2019». **No puede medirse fuera de muestra:** las rentas por clase del WDI empiezan en 1970. El criterio se mide desde 1990 (20 años de rentas para ajustar) y 1980 se informa. Es una desviación declarada.
- **Ni en 1980 ni en 1990 la mediana bate a la media de 10 años en error absoluto.** El modelo no ve venir el auge de materias primas de 2003-2014 (China) ni el mínimo de 1998. Su ventaja está en la distribución (CRPS de W3), no en la mediana a 20-40 años.

## 6. W6: la relación metamórfica de F2

Mercados solos desde 2019 con el PIB mundial al 2 % anual:

| Recurso último | ×0,5 | ×1 | ×1,5 | Sin límite |
|---|---|---|---|---|
| Pico del petróleo | 2054 | 2097 | 2141 | sin pico hasta 2169 |
| Pico del gas | 2043 | 2065 | 2082 | sin pico hasta 2169 |

Más recurso da un pico más tarde y sin límites no hay pico. Hay una prueba con cada caso. **Limitación:** después del pico, la cantidad cae y el precio sube poco (κ = 1 [S]), así que la `psi` **baja** en lugar de subir. Sin sustitutos (F3) no hay precio de equilibrio razonable para un mundo que solo se agota; ese tramo es **exploratorio**.

## 7. Desviaciones y lo que no está

- **No hay suma de Hubbert por país.** La linealización con datos ≤ origen infraestima el recurso (§2). La capacidad mundial es agregada, con recurso incierto; el reparto entre países sigue siendo el Hubbert por país del Atlas.
- **Metales:** sin agotamiento (no hay recurso último) y sin metales críticos (litio, tierras raras: no hay series).
- **Supuestos [S]:**
  - tasa máxima de extracción del 10 %;
  - κ = 1;
  - g en [1, 3];
  - precio estacionario;
  - elasticidad renta 1;
  - ρ = 0,5 en el nivel de la renta;
  - R/P de respaldo.
- **El estimador de Prime Radiant (F0.5) no cambia** (W8 y W9a, igual). Llevar F1+F2 al estimador es F11, y la mejora del PIB a 1 año (−12 %) lo hace más interesante.

**Nota posterior (F3, 2026-10-12).** F3 encontró un fallo que afectaba a F2: el crecimiento del PIB mundial del primer año tras el origen comparaba el PIB reanclado al dato con el de la historia simulada sin reanclar. Corregido (crecimiento observado para ese año), la validación repetida da las mismas cifras de F2 a la cuarta decimal (PIB 0,0563 / 0,1133 / 0,1748; rentas 0,0383 / 0,0580 / 0,0641).

## 8. Qué sigue

1. **F3 (energía con EROI):** sustitutos y costes de la transición. Arregla el tramo post-pico, donde la `psi` cae.
2. **Bandas de `psi`:** cobertura del 57-65 %. Recalibración (D) o choques comunes entre clases (los precios del petróleo, el gas y el carbón se mueven juntos y hoy se simulan independientes).
3. **La fecundidad de F1** sigue siendo el rojo de W4.
