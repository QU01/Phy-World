# F10 de Phy-World (World4): addon de Quasar Studio

- **Fecha:** 2026-10-13
- **Estado:** construida y medida. La card de World4 sale `available` en Studio, lanza escenarios y búsquedas de políticas en un proceso aparte con cancelación, y trae una vista propia: el **explorador del mundo** (globo 3D con los países como agentes, grafo de la corrida y preguntas a QUO y Prime Radiant), en inglés y en español.
- **Pruebas:**
  - World4: 7 nuevas en `tests/test_f10_studio.py`.
  - Studio: 10 nuevas en `backend/tests/test_model_world4.py` (incluida una corrida L1 en proceso hijo con cancelación) y 226 tests previos de permisos, tools, Protocol, aislamiento, registro, router y MCP en verde.
  - Frontend: `tsc` y `eslint` limpios; revisado en el navegador con datos de corridas reales.
- **Parche de Studio:** `docs/F10-qstudio.patch`, solo con mis cambios (Studio tenía cambios del usuario sin commit en `platform_tools.py`, `permissions.py`, `quo_system.md`, `en.ts` y `es.ts`; en esos archivos el parche solo añade líneas).

## 1. Qué se construyó

| Pieza | Dónde | Nota |
|---|---|---|
| Shim del contrato | `world4/studio_compat.py` | `Fidelity`, `FAMILIES`, `VAR_NAMES`, cajas, `denormalize`, `evaluate`, `run_studio` |
| Datos del explorador | `world4/explorer.py` | `world.json`, `geo.json`, `graph.json` (ver §3) |
| Manifiesto | Studio `Phy-Models/manifests/world4.json` | `society.world_system`; familias `scenario` / `design`; slot `world_explorer` |
| Adaptador | Studio `backend/app/services/phy_models/models/world4.py` | 138 líneas; toda la lógica vive en el paquete |
| Requisitos | Studio `backend/requirements-world4.txt` | `fair`, `xarray`, `pooch` fijados |
| Coste medido | Studio `costs/world4_fidelity_costs.json` | L0 y L1 (§4) |
| Tool de QUO | Studio `platform_tools.world4_read` | clase lectura; deriva a `quo` y `design_copilot`, no al verificador |
| Vista | Studio `frontend/src/components/world4/` | globo, grafo y Ask; i18n en `lib/i18n/world4.ts` |

## 2. Cómo encaja en el contrato de Studio

- **θ** = las 7 palancas de F8. **Familias** = los dos modos de la CLI que Studio ofrece: `scenario` (`--theta`: una política sobre el ensemble con su PRIM) y `design` (`--design`: la búsqueda robusta con `phycore.Designer`).
- **Fidelidades:**
  - L0 = el L0 regional a 2100;
  - L1 = lo mismo más los 180 países-agente 2019-2029 (F1-F5);
  - **L2 se declara y no se ofrece** (`excluded_fidelities`): sus emuladores (FaIR, cultivos) ya están dentro del L1 desde F4/F5. Si llega por otra vía, `run_studio` la degrada a L1 con aviso, como la CLI.
- **Capacidades:** `evaluate_theta`, `cancel_run`, `report`. Sin `trace`: el árbol del optimizador de la familia no tiene nada que mostrar para un ensemble de políticas; la vista de una corrida es el explorador. Sin `geometry`.
- **Aislamiento:** todo vive bajo `world4.`, sin `physics_core.py` en la raíz, así que no choca en `sys.modules` con el resto de la familia.
- **Proceso aparte:** las corridas van por `run_process.spawn_child`, el camino de producción de Studio; el test lento lo ejercita de verdad (`pid` del hijo y `cancel.flag`).

**Hallazgo que cambia el plan: `fair` NO es opcional en Studio.** La hoja de ruta lo daba como opcional, con degradación de L2 a L1. Pero hasta el L0 lee el CO₂ fósil observado a través de `world4.data.climate`, que importa FaIR. Sin él no corre nada (medido), así que es un requisito y la card sale `dependency_missing` con el remedio. Instalarlo en el venv de Studio solo añade `fair`, `pooch` y `xarray` (comprobado con `pip install --dry-run`); se instaló.

**Versiones:** Studio corre con numpy 2.5.1, scipy 1.18.0 y pandas 3.0.5, no con las que World4 fija para la paridad bit a bit de W2. La política de referencia da los mismos objetivos que en el venv de World4 con diferencias de 1e-15. W2 se mide en el paquete independiente, no en Studio, y la `honesty_note` del manifiesto lo dice.

**Submódulo:** `QU01/Quasar-World4` no está publicado. En esta máquina, `Phy-Models/World4` es una unión de directorios al checkout, excluida en `.git/info/exclude` y nunca comiteada. El `git submodule add` queda pendiente de publicar el repo.

## 3. Los datos del explorador

**`world.json` (esquema `world4-explorer-1`):**

- **Países como agentes** (solo L1). El L1 F1-F5 desde el origen vivo, 2019, a 10 años. Para cada país y año:
  - p10/p50/p90 del ensemble crudo de 16 series: PIB per cápita, población, centralidad, asabiya, deuda, democracia, riesgo de conflicto, drenaje, rentas netas, rentas de recursos, CO₂, energía, calorías…;
  - las frecuencias de conflicto, crisis y golpe;
  - **las decisiones del agente:** las medianas de sus seis canales de la regla de conducta del Atlas (capital, tecnología propia, importar del centro, extracción propia, recursos fuera, redistribución).
- **Flujos:**
  - **Comercio:** cuotas de exportación COW (la red de 2014, mantenida) × apertura × PIB.
  - **Drenaje por intercambio desigual:** el L1 registra solo el total por país; el reparto entre socios repite su propia regla (`θ_d·X·S_exp·gap`) sobre las medianas. El test comprueba que lo repartido no supera lo registrado.
  - Se guardan los 160 y 120 mayores enlaces del año y, por país, los 3 mayores socios en cada sentido. **Sin esto último**, la ficha de un país pequeño salía sin socios, porque nunca entra en el top mundial. Se vio en el navegador y se corrigió.
- **L0 regional a 2100** con la política y la narrativa de la corrida: bandas por región y del mundo, más la línea sin política. También la tabla país → región, para que una corrida L0 pueda colorear el globo.

**`graph.json`:** el flujo de información de la corrida, con 16-19 nodos y sus aristas, y en cada nodo los números de ESTA corrida, su estado y sus salvedades:

> datos (con SHA-256 y licencia) → sectores F1-F5 → agentes y flujos → L0 → palancas ⇄ búsqueda → objetivos y límites → Prime Radiant y QUO

Lleva además el grafo de lazos del L0 (13 stocks, 54 enlaces) con la dominancia por década.

Las tres cargas pasan `honesty.audit` (W11) sin violaciones.

## 4. Coste medido

Medido el 2026-10-13 con `launch_run` del servicio genérico en modo hilo, con el venv de Studio, seed 71 y la narrativa ssp2.

| Corrida | L0 | L1 |
|---|---|---|
| `design`, preset quick (40 + 2 × 40 evaluaciones del ensemble) | 56,0 s | 64,6 s |
| `scenario` (200 estados del mundo + PRIM + explorador + informe) | 3,0 s | 13,0 s |

Con el preset quick, la búsqueda dejó 22 políticas en el frente verificado y 0 factibles: el techo de tierras es un WALL en el L0 y se informa sin restringir, como en F8.

## 5. La vista

- **Globo:**
  - three.js puro, sin globe.gl. Los países se pintan en una textura equirectangular, así que recolorear por capa o año es redibujar 177 polígonos.
  - Arcos de círculo máximo con partículas que van del origen al destino.
  - Agentes como puntos (tamaño = PIB; color = el canal hacia el que se movieron ese año) y anillos que laten con un evento (conflicto en la mitad de los miembros, crisis en una cuarta parte o golpe en una décima).
  - Línea de tiempo con reproducción; selección por raycast con giro de la cámara hacia el país.
  - A la derecha, el **pulso del mundo** (GMST, rentas, petróleo, los agentes que más se movieron, los eventos y la GMST a 2100 con y sin política) o la **ficha del país-agente** (estado con bandas, trayectoria, columnas de decisión, socios y los botones a QUO y Prime Radiant).
- **Grafo de la corrida** (@xyflow, ya en Studio):
  - flujo de información por columnas con ficha por nodo;
  - lazos de realimentación con selector de década; el lazo elegido se resalta, y uno de un solo stock se pinta como anillo en el nodo;
  - banda inferior con la política frente a no hacer nada.
- **Preguntar:**
  - **QUO:** va por el MISMO stream del chat activo, con una línea de contexto (corrida, familia, fidelidad, narrativa y país) para que lea la corrida con `world4_read`.
  - **Prime Radiant:** crea una pregunta con vínculo estructural (conflicto, golpe, crisis, democracia, crecimiento o deuda × ISO3) con texto y reglas de resolución precompuestos, y muestra lo que dice la corrida para ese país y año. **No lanza el pronóstico:** eso gasta el presupuesto del enjambre y se hace desde Prime Radiant.
- **Integración genérica:** el slot `world_explorer` (stage `explorer`) se pinta a todo el ancho antes del riel de etapas. Es un `kind` nuevo de `SlotRenderer`, sin ninguna rama por id de modelo. three.js y @xyflow se cargan con `next/dynamic` solo para el modelo que declara el slot.
- **Idiomas:** inglés como fuente y español completo (`world4En` / `world4Es`, forma comprobada por `tsc`). Las etiquetas de los nodos se traducen por id, con el texto del backend como respaldo.

**Revisado en el navegador** (ruta temporal con `fetch` simulado sobre los JSON de tres corridas reales, borrada al terminar):

- globo L0 y L1;
- drenaje convergiendo hacia los núcleos;
- ficha de Níger (exporta a Nigeria, Francia y China; drena hacia Francia, EE. UU. y China);
- creación de una pregunta (contra el `fetch` simulado);
- grafo y lazos;
- cambio a español.

## 6. Límites declarados

- **Los agentes no tienen palancas:** las 7 palancas actúan sobre el L0. El globo muestra la conducta clonada de 1950-2019, igual en todas las políticas, y la ficha lo dice.
- **Bandas crudas:** las bandas por país son las del ensemble crudo del L1; las recalibradas de W8 siguen viviendo en `api.query`.
- **Flujos aproximados [S]:** van sobre medianas, con la red COW de 2014 mantenida y el PIB per cápita en lugar del producto per cápita en la regla del drenaje.
- **El año de origen:** los «eventos» de 2019 son el estado observado y asimilado; por eso salen al 100 % de los miembros.
- **Textos del backend en inglés:** las etiquetas de las métricas dentro de los nodos y las unidades vienen del paquete, en inglés.
- **Lo que no se probó:**
  - una respuesta real de QUO de punta a punta (pide el backend con sesión iniciada y un modelo);
  - la creación real de la pregunta en Prime Radiant: se probó la lógica de la tool, la forma de la petición y el envío contra el `fetch` simulado.

## 7. Pendiente

- Publicar `QU01/Quasar-World4` y cambiar la unión por `git submodule add` (decisión del usuario).
- **F11:** el estimador estructural sobre el L1 completo. El explorador ya deja preguntas de Prime Radiant ligadas a World4 desde un clic.
