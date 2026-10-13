# F5 de Phy-World (World4): alimentos, suelo, nitrógeno y agua

- **Fecha:** 2026-10-12
- **Estado:** construida y medida. Cada país lleva cereales, tierras de cultivo, pastos, bosque, N mineral, riego y suministro de calorías (FAOSTAT), por miembro del ensemble. **D4 cerrado:** la huella ecológica y la biocapacidad que el Atlas leía congeladas en el origen se calculan con los flujos del propio modelo. Además:
  - las emisiones agrícolas y la pérdida de bosque alimentan FaIR (F4);
  - el déficit calórico entra en la esperanza de vida de F1.
  - `api.query` usa F1-F5 (`MODEL_REVISION` 6).
- **Campaña:**
  - **W3: ÁMBAR** (la salvedad de F1). Cereales, tierras de cultivo y calorías baten a la persistencia en los tres horizontes.
  - **W4: ROJO.** Pesan la población de F1 y la GMST de F4, y se añade la superficie cultivada desde 1990, dentro de banda solo el 3 % de los años. Desde 2005, cereales y tierras de cultivo quedan dentro de banda el 100 % de los años.
  - **W5: VERDE**, 72/72 celdas.
  - **W6: ÁMBAR.** Se mide la relación de F5.
  - **W7: ÁMBAR.** Suelo y N cierran a 3e-16 y 0.
  - **W12: ÁMBAR.** La licencia de FAOSTAT queda verificada (CC BY 4.0 más los términos de FAO).
- **Pruebas:** 107/107; 8 son nuevas.
- **Validación:** `world4_cli.py --validate` (49 orígenes × 9 modos más los *hindcasts*; unos 50 min).

## 1. Qué se construyó

| Pieza | Fichero | Qué hace |
|---|---|---|
| Datos | `world4/data/food.py`, `food_manifest.json` | FAOSTAT por descarga masiva y agua del WDI. Conjuntos de FAOSTAT: QCL (cereales), RL (uso del suelo), RFN (N mineral), ESB (balance de N de las tierras de cultivo), GT (CH₄ y N₂O agrícolas, CO₂ de conversión de bosque), FBSH + FBS (kcal). SHA-256 fijado con la fecha de actualización de cada conjunto: FAOSTAT reemplaza sus ficheros sin versión, así que otro SHA es otra *vintage* y se rechaza |
| Sector | `world4/sectors/food.py` | Diez ecuaciones de panel (≤ origen, ponderadas por tamaño), cuentas de suelo y N, huella y biocapacidad, indicadores de límites planetarios |
| F1 | `population.fit_e0_food`, `Population.enable_food` | La e0 con el déficit calórico del año anterior (solo con F5) |
| F4 | `climate.Climate.set_agri` | Con F5:<br>• el N₂O y el NH₃ siguen las entradas de N de las tierras de cultivo;<br>• el 35 % del CH₄ sigue las calorías;<br>• el CO₂ de uso del suelo suma la pérdida de bosque modelada |
| Acoplamiento | `coupling.py` (`F5Config`) | Tras cada año del L1: alimentos (con la temperatura local, el PIB y la población del año) → mortalidad de F1 al año siguiente → emisiones de FaIR. El Atlas lee la huella y la biocapacidad calculadas (`ex`) |
| Validación, CLI, API | `rolling.py` (modo `f5`; métricas `cereal`, `cropland`, `kcal`), `hindcast.run_food`, `--install-food`, `--query cereal_production / cropland / food_supply` | |

**Cobertura.** Los países del L1 con datos en el origen son 131 en 1990 (falta la ex URSS: el 13 % de los cereales del mundo) y 154 en 2005. La verdad de los *hindcasts* se suma sobre esos mismos países.

## 2. El modelo (coeficientes con corte 2019)

| Ecuación | Forma | Lo esencial |
|---|---|---|
| N mineral por ha | Δln n = b₁ (frontera − ln n) + b₂ Δly + b₃ Δln n₋₁, **sin constante** [S] | Converge al percentil 90 del origen (134 kg N/ha) y no lo rebasa |
| Rendimiento de cereales | Δln Y = c + b₁ brecha + b₂ Δln n + b₃ ΔT + b₄ ΔT·(T̄ − 15)/10 + b₅ Δln Y₋₁ | ΔT es la anomalía de temperatura local del año, la del L1 (con F4, la de FaIR). **+1 °C: −2,7 % en un país templado; −5 % a 25 °C** |
| Superficie de cereales, tierras de cultivo, pastos, riego | Dinámica en diferencias con población y rendimiento | |
| Bosque | ΔF/L sobre la expansión agrícola | Solo con 8 años o más de FRA (orígenes ≥ 1999). La expansión agrícola no explica la pérdida de bosque (b ≈ 0): el bosque sigue su propia tendencia |
| Calorías | Δln k = c + b₁ ln k₋₁ + b₂ Δly + b₃ Δln(cereal propio per cápita) + b₄ Δln k₋₁ | b₃ = 0,04 (EE 0,01). La cosecha mundial no entra (b ≈ 0); la propia del año sí |
| Huella sin carbono / biocapacidad | Elasticidades a calorías y tierras / a rendimiento × tierras | La huella de carbono es proporcional al fósil per cápita de F3, porque la de GFN es lineal en el CO₂ |
| e0 (F1) con déficit | El déficit es max(0, ln 2300 − ln kcal) [S] | **No identificado:** 0 por el signo con datos ≤ 2005; −0,12 ± 0,29 con datos ≤ 2019 |

**Cuentas (W7):**
- **Suelo:** cultivo + pastos + bosque + resto = superficie de tierra. Si el resto saliera negativo, se recortan los pastos.
- **N de las tierras de cultivo:** entradas = mineral + estiércol + fijación + deposición; salidas = cosecha + residuos + semilla. El estiércol, la fijación y la deposición por hectárea quedan en su valor del origen [S]; las salidas siguen la cosecha.

**Huella sin MRIO (decisión de la hoja de ruta).** No se usa EXIOBASE (no comercial) ni GLORIA (licencia sin verificar). La huella sale de los flujos físicos del modelo, sin matriz entre regiones, así que el comercio solo está implícito en las elasticidades estimadas.

## 3. Tres correcciones que obligaron los datos (o un fallo)

| Síntoma | Causa | Corrección |
|---|---|---|
| Hindcast desde 1990: 300 Tg de N intencional en 2023 (real ~165) y cereales +56 % | Con datos de 1961-1990 (Revolución Verde) la ecuación daba +4 % anual de N también en la frontera | N sin constante, con convergencia a la frontera del origen [S]. **Se cambió DESPUÉS de ver ese hindcast**, y se declara. Desde 1990 los cereales pasan a +41 % y el N a 200 Tg |
| Las calorías no respondían a la cosecha: el clima no llegaba a la alimentación | Se usaba la cosecha mundial per cápita, con b ≈ 0 | La cosecha propia del año, con b > 0 y significativa |
| A 2100, el N se disparaba hasta 1e6 Tg, el NH₃ también y la GMST bajaba a −30 °C | Un fallo: `Food.step` no actualizaba Δln n, y Mongolia declara 0 fertilizante en 2019 (Δ = −3,45 en el origen) | Corregido; las diferencias iniciales se acotan a ±0,3 y un 0 aislado en el origen toma el último valor > 0. La validación se repitió entera después |

## 4. W3 y W5: origen móvil (49 orígenes, R = 32)

| CRPS del log por país, 1/5/10 años | F1-F5 | Persistencia (la mejor referencia) | Cobertura p10-p90 |
|---|---|---|---|
| Producción de cereales | **0,136 / 0,228 / 0,326** | 0,169 / 0,289 / 0,411 | 0,65 / 0,66 / 0,66 |
| Tierras de cultivo | **0,018 / 0,061 / 0,108** | 0,021 / 0,077 / 0,138 | 0,63 / 0,48 / 0,38 |
| Calorías por persona | **0,015 / 0,035 / 0,053** | 0,019 / 0,048 / 0,074 | 0,81 / 0,77 / 0,74 |

- **Baten a la persistencia en los tres horizontes,** entre un 13 y un 29 %.
- **Las bandas de las tierras de cultivo son estrechas** a 10 años: cubren el 38 % de los casos.
- **PIB pc frente al Atlas:** 0,892 / 0,953 / 0,969. F5 casi no mueve la economía (con F4 era 0,970).
- **W5:** con la capa F5, deuda, democracia, energía y `psi` no se mueven (72/72 celdas en las cinco capas).

## 5. W4: *hindcast* libre hasta 2023

| Origen | Cereales dentro de p10-p90 | Cereales en 2023 (mediana) | Tierras de cultivo dentro | Tierras de cultivo en 2023 | Calorías dentro | Balance de N dentro |
|---|---|---|---|---|---|---|
| 1990 (131 países) | 85 % | **+41 %** | **3 %** | +14 % | 100 % | 97 % |
| **2005** (154 países) | **100 %** | +0,6 % | **100 %** | +1,7 % | 72 % | 83 % |

- **Desde 2005 el modelo acierta:** cereales y tierras de cultivo quedan a ±2 % en 2023.
- **Desde 1990 extrapola la Revolución Verde.**
  - Los rendimientos y las tierras siguen el ritmo de 1961-1990.
  - Lo que vino después no se veía en los datos ≤ 1990: el estancamiento del fertilizante en la frontera y el ahorro de tierra en los países ricos.
  - Las tierras de cultivo crecen un 14 % donde la realidad apenas se movió, y su banda es demasiado estrecha para contenerlo (3 %). Es el rojo nuevo de W4.
  - Que los cereales queden dentro de banda el 85 % de los años se debe a lo anchas que son sus bandas, no a que la mediana acierte (+41 %).

## 6. W6 y W7

- **W6 (F5):** +2 °C de anomalía local bajan el rendimiento de cereales un 10,4 % en los países cálidos (T media > 24 °C) y un 1,7 % en los frescos (< 12 °C), con los coeficientes estimados hasta 2019.
- **W7:** el suelo cierra a 3e-16 y el N de las tierras de cultivo a 0. Con F3 y F4, todas las cuentas físicas del modelo (energía, carbono, suelo y N) cierran. Siguen faltando la contabilidad stock-flujo (SFC) y el empleo.

## 7. Límites planetarios (indicadores [S])

| | 2019 | 2050 (exploratorio) | Límite |
|---|---|---|---|
| N intencional (N mineral + fijación en cultivos), Tg N/año | 144 | 241 | 62 (Steffen et al. 2015) |
| Extracción agrícola de agua, km³/año | 2.640 | 3.590 | 4.000 de uso **consuntivo** (Rockström et al. 2009); la extracción es una cota superior |

## 8. Corrida exploratoria 2019 → 2100 (no es un pronóstico)

F1-F5, R = 8:

| Año | GMST p50 / p90 (°C) | Cereales (Gt) | Tierras de cultivo (Mha) | kcal/persona/día | N intencional (Tg) | Agua agrícola (km³) | Bosque (Mha) |
|---|---|---|---|---|---|---|---|
| 2019 | 1,24 | 2,92 | 1.500 | 2.971 | 144 | 2.640 | 3.957 |
| 2050 | 2,20 / 2,43 | 5,52 | 1.732 | 3.207 | 241 | 3.590 | 3.869 |
| 2100 | 4,26 / 5,85 | 11,1 | 2.385 | 3.442 | 412 | 12.100 | 3.479 |

**No es creíble a largo plazo,** y lo digo:
- **Cereales:** se multiplican por 1,9 hasta 2050 (las proyecciones habituales rondan ×1,2-1,3 [S]) y por 3,8 hasta 2100. Las tasas de rendimiento y de superficie estimadas no se frenan.
- **Calor:** los +4,3 °C solo restan algunos puntos al rendimiento.
- **Agua:** la extracción se multiplica por 4,6 por el riego de unos pocos países.
- **Lo que falta para frenar el sistema:** saturación del rendimiento, límite de tierra cultivable de calidad y precio de los alimentos. Los lazos de escasez, que son el corazón de World3, no se cierran aquí para los alimentos.

## 9. Desviaciones y lo que no está

- **Emulador de cultivos:** no hay GGCMI. Los datos de la fase II pesan 7,4 GB y OSCAR-crop tiene la licencia del código sin confirmar. La respuesta al clima es estadística (anomalías de temperatura de 1961-origen), sin CO₂ ni precipitación.
- **Producción:** solo cereales; no hay ganadería, pastos productivos ni comercio explícito.
- **Tierra degradada:** no hay serie.
- **LUH2:** no se usó (licencia de los datos sin verificar).
- **Hindcast desde 1961:** el criterio decía «1961-2023»; el L1 arranca en 1970 o después, así que se mide desde 1990 y 2005.
- **Alimento → PSI** (presión estructural-demográfica del Atlas): no conectado, porque exige recalibrar el modelo de conflicto del Atlas (F7).
- **Alimento → mortalidad:** conectado, pero los datos no lo identifican; el efecto es ≈ 0 (§2).
- **Supuestos [S]:**
  - umbral de 2.300 kcal;
  - estiércol, fijación y deposición por hectárea fijos;
  - frontera de N fija;
  - límites planetarios citados de fuentes secundarias;
  - parte agrícola del CH₄ = 35 %.

## 10. Qué sigue

1. **Frenos del sistema alimentario:** saturación de rendimientos, techo de tierra cultivable y un precio de los alimentos (mercado mundial como en F2). Sin ellos la corrida larga no tiene sentido.
2. **Emulador de cultivos de GGCMI** (fase II, CC BY 4.0) para CO₂, agua y N.
3. **F6 (L0 de 10 regiones y lazos)** y **F7 (calibración bayesiana y bandas, W8):** las bandas de tierras de cultivo, GMST y energía son estrechas.
4. **Arrastrados:**
   - la fecundidad de F1 (W4);
   - la coherencia F2/F3 del fósil;
   - el presupuesto de carbono de F4.
