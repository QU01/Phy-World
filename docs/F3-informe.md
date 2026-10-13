# F3 de Phy-World (World4): energía con EROI

- **Fecha:** 2026-10-12
- **Estado:** construida y medida. Cada país tiene una mezcla de energía primaria endógena (fósil, nuclear, hidro, eólica + solar y otras). Con ella llegan la EROI, la energía neta y los materiales de la transición. `api.query` usa F1+F2+F3 (`MODEL_REVISION` 4).
  - **W3: ÁMBAR** (la salvedad de F1).
  - **W4: ROJO** por la población de F1. Energía primaria mundial dentro de banda el 94 % de los años desde 2005 y el 88 % desde 1990.
  - **W5: VERDE** (45/45).
  - **W7: ÁMBAR.** La energía cierra a 0; faltan las masas, el balance SFC y el empleo.
  - W2 sigue bit a bit.
- **Pruebas:** 88/88; 9 son nuevas.
- **Validación:** `world4_cli.py --validate` (49 orígenes × 7 modos más 5 *hindcasts*, unos 25 min).

## 1. Qué se construyó

| Pieza | Fichero | Qué hace |
|---|---|---|
| Datos | `world4/data/energy.py`, `energy_manifest.json` | OWID energy-data fijado al commit `f8de88a` (Energy Institute + EIA + Ember), con SHA-256. Las seis regiones de la ONU salen de WPP |
| Sector | `world4/sectors/energy.py` | Mezcla por país y miembro, EROI, cuentas bruta/autoconsumo/neta, coste real del sistema, escasez física y materiales |
| Acoplamiento | `coupling.py` (`F3Config`) | Energía → mercados fósiles de F2 → L1, cada año (§2) |
| Validación | `rolling.py` (modo `f3`, métrica `pe`) y `hindcast.run_energy` | W3, W5 y W4-energía |
| Campaña | `campaign.py` | W4 con energía, W5 con la capa F3, W7 medido |
| CLI / API | `--install-energy`; `--query primary_energy` | |

**Desglose imputado (declarado por país y año):**

- El Energy Institute da el desglose por fuente de 79 de los 180 países, que suman el 96,5 % de la energía mundial.
- Al resto se le imputa desde la generación eléctrica, con la eficiencia de sustitución **medida** en los países que tienen ambas series. En 2019, el 2,6 % de la energía mundial sale de esa imputación.
- Antes de 2000 muchos países pequeños no tienen generación eléctrica y su energía cuenta como toda fósil. Eso abre un **salto en la «verdad»** en 2000; por eso el criterio de mezcla se mide desde 2005.

## 2. El modelo

Cada año, por país y miembro, se estiman con datos ≤ origen, mínimos cuadrados **ponderados por la energía del país** y choques AR(1) comunes y propios:

| Ecuación | Forma | Corte 2019 |
|---|---|---|
| Intensidad | Δln(PE/PIB) = a + b Δln(PE/PIB)₋₁ + c Δln pF₋₁ (c ≤ 0) | c = −0,026 (EE 0,007) |
| Cuotas nuclear, hidro y otras | Δlogit = a + b·logit₋₁ + c·Δlogit₋₁, sin efecto propio por país | dinámicas estimadas |
| Cuota eólica + solar | ídem + d·G₋₁; G = ln pF + β ln(acumulado mundial) − K·cuota² (d ≥ 0) | **d = 0,054 (EE 0,029); con datos ≤ 2005 no se identifica (d = 0)** |
| Fósil | el resto de la energía primaria | |

**Acoplamiento:**

- **Energía → F2:** la demanda de petróleo, gas y carbón de F2 deja de seguir al PIB y sigue al consumo fósil de F3. Es la sustitución que le faltaba a F2. El cobre y el níquel de la eólica + solar nueva entran como demanda en el mercado de metales.
- **F2 → energía:** el precio fósil entra en la intensidad y en la brecha de coste de la eólica + solar; el agotamiento hace caer la EROI fósil.
- **Energía → L1** (TFP, normalizada a 0 en el origen como F2):
  - −θ_E·ln(coste real del sistema / el del origen), con θ_E = 0,05 [S]. El coste incluye el aprendizaje de la eólica + solar, su coste de integración y el autoconsumo por la EROI. El precio fósil **no** entra aquí, porque ya lo lleva F2 en las rentas.
  - Si la capacidad de F2 no cubre la demanda fósil, la producción cae según un CES con σ = 0,3 [S].

**Parámetros supuestos [S, sin verificar con la fuente primaria]:**

- EROI en el origen: fósil 20 (cae con el agotamiento de F2), nuclear 15, hidro 40, eólica + solar 10 (sube con el aprendizaje, techo 30), otras 4. Llevan ±30 % log por miembro.
- Tasa de aprendizaje entre 10 y 20 % por duplicación; integración K = 2.
- Intensidades de material de la IEA (2021): cobre 2,85 t/MW, níquel 0,2 t/MW, tierras raras 0,007 t/MW.
- Factor de capacidad 0,25.

## 3. Correcciones que obligaron los datos

| Síntoma | Causa | Corrección |
|---|---|---|
| Energía mundial +13-15 % a 10 años | (a) el primer año tras el origen medía el «crecimiento» entre el PIB reanclado y el de la historia simulada sin reanclar; (b) MCO no ponderado: dominan los países pequeños, con intensidad creciente | (a) crecimiento observado para ese año, **corrige también F2**; (b) MCO ponderado por energía. Queda +2,9 % (1990) y +8,8 % (2005) a 10 años |
| Eólica + solar del 33 % en Oceanía y nuclear del 10 % en Latinoamérica (2023) | El efecto propio de un país que arranca extrapola su crecimiento inicial, porque la reversión es débil | Sin efecto propio en las cuotas |
| Eólica + solar disparada desde 1990 | 11 observaciones en 1990; todos los países sembrados en el mínimo | Solo crece donde existe en el origen; con < 50 observaciones queda congelada |
| NaN en 1970 | Ninguna clase fósil tiene 10 años de demanda sobre energía fósil (el panel empieza en 1965) | Agotamiento y precio constantes si todas están congeladas |

## 4. W3 y W5: origen móvil (49 orígenes)

| | Atlas | F1+F2 | **F1+F2+F3** | F3/Atlas, IC95 |
|---|---|---|---|---|
| CRPS PIB pc, 1/5/10 años | 0,0637 / 0,1191 / 0,1805 | 0,0563 / 0,1133 / 0,1748 | 0,0569 / 0,1132 / 0,1743 | 0,892 / 0,950 / 0,966 [0,949, 0,983] |
| RMSE rentas, 1/5/10 años | 0,0493 / 0,0663 / 0,0680 | 0,0383 / 0,0580 / 0,0641 | 0,0395 / 0,0602 / 0,0658 | 0,80 / 0,91 / 0,97 [0,93, 1,01] |
| CRPS `psi`, 1/5/10 años | 0,0144 / 0,0192 / 0,0183 | 0,0060 / 0,0105 / 0,0106 | 0,0069 / 0,0121 / 0,0120 | mejor referencia 0,0074 / 0,0152 / 0,0175 |
| **CRPS log energía primaria por país** | — | — | **0,064 / 0,155 / 0,253** | mejor referencia 0,073 / 0,227 / 0,415 |
| Cobertura p10-p90 de la energía | — | — | 0,63 / 0,73 / 0,69 | nominal 0,80 |

- **La energía por país bate a la persistencia y a la tendencia** en los tres horizontes, y a 10 años lo hace por un 39 %.
- **F3 empeora un poco la `psi` y las rentas respecto de F2,** aunque siguen mejor que el Atlas. Al mover la demanda fósil con la mezcla en lugar del PIB se pierde algo de precisión a corto plazo. A 10 años el IC de las rentas ya incluye el 1.
- **W5:** conflicto, deuda, democracia, temperatura y población no se mueven fuera de su banda al añadir F3 (45/45 celdas en las tres capas).

## 5. W4: *hindcast* libre de la energía (hasta 2023)

| Origen | Energía primaria mundial dentro de p10-p90 | Mediana en 2023 | Mezcla (6 regiones × 5 fuentes) dentro | Error medio de la mezcla | Persistencia |
|---|---|---|---|---|---|
| **2005** | **94 %** | +8,7 % | 77 % | **0,0113** | 0,0135 |
| 1990 | 88 % | +40,3 % | 62 % | 0,0168 | 0,0140 |

- **Desde 2005, F3 mejora a la persistencia en un 16 %.** En 2023, por ejemplo: Europa con 6,7 % de eólica + solar (observado 7,4 %), Norteamérica con 3,8 % (5,8 %), Asia con 1,2 % (5,2 %). El aprendizaje no está identificado con datos ≤ 2005, así que la eólica + solar sale baja.
- **Desde 1990, F3 es peor que congelar la mezcla.** La eólica + solar casi no existía (queda congelada en 0); la nuclear se sobreestima porque la dinámica común la lleva al 10-12 % (Latinoamérica 7,8 % frente a 0,8 %); y la «verdad» tiene el salto de imputación de 2000. Es honesto: en 1990 no se podía prever.
- **Materiales:** en el *hindcast* desde 2005, la eólica + solar nueva de 2023 pide unas 320.000 t de cobre, 22.000 t de níquel y 790 t de tierras raras (mediana). La oferta de tierras raras y el litio no están modelados.

## 6. W7: cuentas

- **Energía:** |bruta − autoconsumo − neta| / bruta = 0 y |Σ fuentes − energía primaria| / energía primaria = 2e-16. La generación nunca es negativa.
- **Cohortes de F1:** cierran a 1e-9 (pruebas).
- **ÁMBAR:** faltan las masas de C y N (F4/F5), el balance SFC de deuda y activos y el empleo ≤ población en edad de trabajar, porque el L1 no tiene mercado de trabajo ni cuentas financieras.

## 7. Corrida exploratoria 2019 → 2100 (no es un pronóstico)

F1+F2+F3, R = 4, sin clima (F4) ni alimentos (F5):

| Año | Energía (miles de TWh) | Fósil | Hidro | Eólica + solar | `psi` | Población (miles de millones) |
|---|---|---|---|---|---|---|
| 2019 | 162 | 84 % | 6 % | 3 % | 0,029 | 7,6 |
| 2050 | 180 | 75 % | 10 % | 7 % | 0,014 | 9,1 |
| 2100 | 431 | 72 % | 12 % | 11 % | 0,009 | 7,7 |

La corrida no explota, pero destapa límites declarados:

- **Transición lenta:** la difusión estimada es débil y el aprendizaje solo se identifica con datos hasta 2019.
- **Hidro sin techo de potencial.**
- **El petróleo no escasea antes de 2100** con los recursos supuestos.
- **La población cae a 7.700 millones en 2100,** frente a 10.200 de la variante media de WPP. Es la fecundidad de F1 extrapolada.

## 8. Desviaciones y lo que no está

- **Sin capital energético por fuente** (stock, inversión, vida útil). La mezcla se modela como **cuotas con dinámica estimada**, porque no hay series de capital por fuente para estimar inversión y depreciación sin supuestos. La EROI, el aprendizaje y la integración actúan a través de la brecha de coste y del coste del sistema.
- **«Mezcla por región 1965-2023»:** se mide desde 1990 y 2005, hasta 2023. Antes de 1990 casi no hay desglose fuera de la OCDE.
- **No están:**
  - electricidad como vector, almacenamiento, vehículos y litio;
  - biomasa tradicional (el EI no la cuenta);
  - costes absolutos (se suponen iguales en el origen [S]);
  - techo de hidro;
  - nuclear en países que no la tienen.
- **Licencia:** OWID es CC BY; la licencia del Energy Institute no es verificable ([?]). No viaja en el repo.

**Nota posterior (F4, 2026-10-12).** F4 encontró una incoherencia entre F2 y F3. F3 mueve la demanda fósil de F2 con el *crecimiento* de su consumo fósil, pero cada clase de F2 conserva su propia tendencia estimada (Δln(Q/F)), así que la suma de petróleo, gas y carbón de F2 no es el fósil de F3. En el *hindcast* desde 1990 (R = 16), en 2023 el fósil de F3 vale ×2,07 su valor de 1990 (observado ×1,69), mientras que F2 da petróleo ×0,84, gas ×0,97 y carbón ×1,82. F4 toma la cantidad de F3 y de F2 solo el reparto entre combustibles (§2 del informe de F4). Cerrar la incoherencia, haciendo que F2 reparta el fósil de F3, cambia las cifras de F2 y F3 y queda pendiente con su revalidación.

## 9. Qué sigue

1. **F4 (clima):** las emisiones salen ya de la mezcla fósil de F3; FaIR en lugar del TCRE.
2. **Identificar el aprendizaje antes de 2010:** series de costes de IRENA (2010+) y precio de módulos (1976+), con su licencia verificada. Hoy la tasa de aprendizaje es un supuesto.
3. **Bandas:** energía 63-73 % y `psi` 50-60 % frente al 80 % nominal; es candidato a la recalibración (D).
4. **La fecundidad de F1** sigue siendo el rojo de W4.
