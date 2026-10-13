# F6 de Phy-World (World4): L0 regional y explicabilidad

- **Fecha:** 2026-10-12
- **Estado:** construida y medida.
  - **L0 regional** (`world4/core/l0r`): 10 regiones IAM y 112 stocks. Cada ecuación es la forma agregada de una del L1 y se estima con los mismos datos agregados por región.
  - **Dominancia de lazos:** puntuación propia de *Loops that Matter*, validada con la logística.
  - **Campaña:**
    - **W6: VERDE**, completo, con su mutante.
    - **W10: VERDE.** El L0 corre a 2100 en 9,4-9,7 ms; el L1 con F1-F5 y 64 miembros, en 40 s con 8 procesos.
    - **P2: falla por poco.** La población queda dentro de banda en el 79 % de las celdas región-año (listón 80 %); el resto de variables pasa.
- **Pruebas:** 114/114; 7 son nuevas.
- **Reproducir:**
  - `python -m world4.validate.w10` (unos 2 min);
  - `--campaign`, que mide P2 y W6 en vivo, unos segundos.

## 1. El L0 regional

**Regiones.** Se cierra la decisión abierta 1 de la hoja de ruta: 10 regiones de estilo IAM, construidas con las subregiones M49 de la ONU más excepciones de la OCDE (NAM, LAM, EUR, REF, MEA, SSA, SAS, CHN, SEA, OPA). Son parecidas a las R10 de los IAM, no idénticas. Los 180 países del L1 quedan asignados.

| Stock (por región) | Flujos | Forma del L1 que agrega |
|---|---|---|
| Población | nacimientos, defunciones, migración | F1: tasas brutas en lugar de cohortes |
| ln natalidad bruta | convergencia a un nivel según el ingreso | F1, ecuación de la fecundidad |
| Capital | inversión s·Y, depreciación | Atlas |
| ln PTF | convergencia a la región líder + efecto propio | Atlas, difusión |
| Daño climático | BHM con persistencia ρ_D | Atlas, `Dclim` |
| ln intensidad energética | mejora autónoma + elasticidad al ingreso | F3 |
| logit cuota no fósil | difusión + sustitución bajo escasez física [S] | F3 |
| ln tierras de cultivo, ln rendimiento, ln N por ha | | F5; coeficientes de temperatura de F5 |
| Deuda / PIB | déficit, dilución por crecimiento | Atlas |
| **Mundo:** extracción fósil acumulada, CO₂ acumulado | | F2 (capacidad = 10 % del recurso restante) y F4 (TCRE efectiva de la historia) |

**Producción.** Cobb-Douglas con el efecto de la energía de F2/F3: coste real con la EROI fósil, que cae con el agotamiento, y escasez física CES.

**Estimación.**
- Ecuaciones con datos ≤ corte, sobre el panel regional.
- Niveles sobre el conjunto fijo de países de 1992 (año base: el primero con las 10 regiones completas en PWT, FAOSTAT y energía).
- Tasas encadenadas sobre toda la historia.
- En cada ecuación en diferencias, un efecto propio por región: el residuo medio de 20 años, encogido n/(n + 5), como `_panel` del L1.
- Anclaje en el año inicial, como el L1 en el origen.

**Correcciones que obligó el dato:**

| Síntoma | Causa | Corrección |
|---|---|---|
| Asia con un PIB 30-40 % bajo en 2019 | Una sola convergencia de PTF para todos | Efecto propio por región |
| Energía de NAM y EUR +50 % | Intensidad como función del ingreso, sin la mejora autónoma de F3 | Intensidad como stock en diferencias, la forma de F3 |
| Población a 14.600 millones en 2100 | Natalidad como función del ingreso que no converge | Natalidad como stock que converge, la forma de F1 |
| Población de REF a 0,1 millones con ahorro ×1,5 | El polinomio de mortalidad se extrapolaba fuera del ingreso observado | Ingreso acotado al rango del panel en las tasas [S] |
| Colapso en 2075 por agotamiento fósil sin sustitución | La mezcla no responde al precio fósil en la historia (coeficiente 0): la cuota no fósil se saturaba en el 14 % | Sustitución bajo escasez física [S], no identificable en la historia; es una palanca de W6 |

## 2. P2: el L0 libre frente al agregado del L1 (1992-2019)

Bandas declaradas antes de medir: población ±5 %, PIB ±25 %, energía ±25 %, cereales ±25 %, tierras ±10 %, GMST ±0,25 °C. Criterio: el agregado mundial dentro de banda todos los años y al menos el 80 % de las celdas región-año.

| | Mundo en 2019 | Mundo, todos los años | Celdas región-año |
|---|---|---|---|
| Población | +1,3 % | sí | **79 %** |
| PIB | −6,6 % | sí | 90 % |
| Energía primaria | +2,2 % | sí | 90 % |
| Cereales | +4,9 % | sí | 83 % |
| Tierras de cultivo | +1,4 % | sí | 92 % |
| GMST | −0,05 °C | — | 100 % |

- **P2 falla solo en la población, por una celda.** Las tasas brutas sin estructura de edades se desvían un 7-9 % en MEA, CHN y OPA. La banda no se tocó.
- **Variante fuera de muestra** (corte 2005, de 2005 a 2019): pasan todas; población 97 %, PIB 93 %.
- **La «verdad» de P2 es el L1 asimilado al dato.** El L1 que pronostica se reancla al dato en el origen, y sus sectores solo actúan después.
- **Hallazgo:** la corrida *libre* del Atlas desde 1950 se aleja mucho del dato. En 2019 da el 37 % del PIB real de China, el 165 % del de África subsahariana y el 124 % del de Norteamérica. El Atlas no está hecho para correr libre 70 años, y no se usa como referencia.

## 3. Dominancia de lazos

**Método.** Implementación propia de la puntuación de *Loops that Matter* (Schoenberg, Davidsen y Eberlein 2020), que solo existe en STELLA. La fórmula se escribió desde el artículo [S].

**Validación.** En la logística, el lazo dominante pasa de los nacimientos (+0,95) a las muertes (−0,67) en el paso en que P cruza K/2: de 486 a 511 sigue mandando el de nacimientos, y de 511 a 536 ya manda el de muertes (prueba).

**Vista mundial del L0.** Cada nodo es un tipo de stock (13). Los ciclos elementales de hasta 4 stocks dan 136 lazos.

| Periodo | Lazo dominante (puntuación relativa) |
|---|---|
| 2020 | Natalidad: la propia inercia de la fecundidad (+0,25) y población → nacimientos (+0,19) |
| 2040-2060 | **Clima**: daño → PIB → intensidad → emisiones → CO₂ → daño (+0,35) y el mismo lazo a través del capital (+0,24 a +0,34) |
| 2080-2100 | **Sustitución fósil**: la cuota no fósil bajo escasez (−0,30 → −0,90) |

**El criterio de F6 se cumple.** El lazo dominante cambia en los años en que cambia el modo de comportamiento: 2069-2071, cuando la extracción fósil toca la capacidad y se dobla la curva del PIB.

**Limitaciones:**
- Los flujos en diferencias (los que dependen de Δly, ΔlnP o ΔT) ligan stocks a través de cambios, no de niveles; parte del cambio queda sin atribuir.
- La polaridad de los lazos de primer orden puede invertirse cuando el flujo neto cambia de signo, una propiedad conocida de LTM.
- No se comparó con PPM ni con LEEA (la prueba de Mojtahedzadeh 2008). Queda pendiente.

## 4. W6 completo

| Relación | Resultado |
|---|---|
| (a) Más recursos → pico más tarde | Pico fósil en 2050 / 2071 / 2095 con el recurso ×0,5 / ×1 / ×2 |
| (b) Sin límites → sin colapso (recurso infinito, sin daño) | El PIB pc nunca cae (0,0 %) |
| (c) Recursos bajos → sobrepaso y colapso (×0,25, sin sustitución) | Cae un 22 % desde su máximo |
| (d) Monotonía de las 10 palancas | 10/10 con el signo declarado (TCRE, daño, recurso, ahorro, natalidad, convergencia, EROI, déficit, sensibilidad del rendimiento, sustitución) |
| (e) Ningún forzamiento exógeno fija el estado en 2100 | Ningún stock queda fijo; el L0 no lee series después del año inicial |
| Mutante: lazo de daño climático invertido | **Detectado**: «más daño → menos PIB» falla |

- **Signo corregido DESPUÉS de medir (declarado):** la palanca «recurso» se había declarado sobre el PIB de 2100 (+). El L0 da lo contrario (76.000 / 68.000 / 61.000 USD per cápita con el recurso ×0,5 / ×1 / ×1,5): más fósil trae más calentamiento y más daño, y menos presión para sustituir. Su salida pasa a ser el fósil acumulado.
- **World3-03 (informativo, no decide W6):** cumple (a) y (c), pero no (b). Sin límites de recursos, contaminación ni tierra, la producción industrial per cápita cae un 26 % por otros límites de su estructura.

## 5. W10, en la máquina de referencia (Ryzen 7 5700X)

| | Medido | Listón |
|---|---|---|
| L0 regional, una corrida 2019-2100 | **9,4-9,7 ms** de mediana (p90 hasta 10,8) | < 10 ms |
| World3-03 | 135 ms sola; 0,33 ms por miembro en lotes de 1000 | — |
| L1 F1-F5, 64 miembros 2019-2100 | **40 s** en 8 procesos × 8 (74-79 s en un proceso) | < 60 s |

**Las dos holguras son pequeñas.** El L0 bajó de 16,7 a 9,4 ms con constantes precalculadas y menos arrays temporales, con el mismo resultado a 1e-14. El listón del L1 se cumple solo en paralelo; el criterio no exige un solo proceso, pero lo declaro.

## 6. Corrida exploratoria del L0, 2019-2100 (no es un pronóstico)

| Año | Población (miles de millones) | PIB pc (USD 2017) | Energía (miles de TWh) | Cuota fósil | GMST (°C) | Agotamiento fósil |
|---|---|---|---|---|---|---|
| 2019 | 7,6 | 16.500 | 162 | 84 % | 1,22 | 0,26 |
| 2050 | 9,7 | 28.100 | 316 | 84 % | 2,34 | 0,53 |
| 2075 | 11,0 | 43.800 | 557 | 61 % | 3,82 | 0,90 |
| 2100 | 11,9 | 80.500 | 1.030 | 3 % | 4,24 | 0,99 |

**El modo:**
- **Hasta ~2070:** crecimiento fósil sin transición, porque la difusión no fósil estimada es débil.
- **Hacia 2070:** el fósil toca su capacidad de extracción y hay escasez (0,38).
- **Después:** sustitución rápida forzada por la escasez [S], y el crecimiento sigue. La temperatura se estabiliza en 4,2 °C al agotarse el fósil.

**Lo que tiene de supuesto:**
- la velocidad de sustitución bajo escasez (`SUBST` = 1, no identificada en la historia);
- el recurso fósil último (reservas × 2);
- el daño BHM, que no frena el crecimiento.

Con el recurso ×0,25 y sin sustitución, el mismo L0 colapsa (relación c).

## 7. Desviaciones y lo que no está

- **Unos 80 stocks:** son 112. Faltan, respecto de la hoja de ruta:
  - población por cohortes (se usan tasas brutas);
  - capital por sector (industrial, servicios, agrícola, energético): un solo capital;
  - minerales por clase: un único recurso fósil;
  - nitrógeno como masa;
  - desigualdad y legitimidad.
- **P2 de 1950 a 2019:** se mide de 1992 a 2019, porque antes la antigua URSS y Europa del Este no están en la PWT y África tiene huecos.
- **JAX opcional y derivadas analíticas:** no se hicieron. La puntuación de lazos usa cambios parciales numéricos, que es lo que define LTM.
- **Dominancia por región:** solo la vista mundial; la regional es posible con el mismo motor.

## 8. Qué sigue

1. **F7 (calibración bayesiana, W8):** *history matching* sobre el L0 (9 ms por corrida y 0,13 ms por miembro en lote lo hacen viable). Bandas de L0 y L1.
2. **Cohortes en el L0** (el fallo de P2 en población) y comparación de LTM con PPM/LEEA.
3. **Arrastrados:**
   - la fecundidad de F1;
   - la coherencia F2/F3;
   - el presupuesto de carbono de F4;
   - los frenos del sistema alimentario de F5.
