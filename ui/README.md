# Interfaz de Phy-World (World4)

Aplicación React (Vite + TypeScript) sobre la API JSON de `world4/server.py`.

| Vista | Qué muestra | Necesita |
|---|---|---|
| Conflicto por celdas | Mapa de las 64 818 celdas PRIO-GRID con P(evento) a 12 meses o por mes, eventos y muertes esperados; ficha de celda (pronóstico mes a mes, origen del riesgo, historia); países; validación | `--install-conflict-grid` |
| Densidad de eventos | Eventos por 10 000 km² por país (o por celda) a 1-5 años desde cualquier año 1990-2025 con las regresiones del Atlas (F13); qué factores predicen mejor; escenario sobre el L1 | Atlas y `--install-conflict-grid` |
| Sociedad ↔ World3 | World3 acoplado a 180 países: conflicto, golpes, crisis, democracia a 2100 con bandas; mapa por país | nada |
| World3 | Los escenarios 1-5 de *Limits to Growth: The 30-Year Update* | nada |
| Países (L1) | Abanico p10/p50/p90 a 1-10 años del Atlas | `--install-atlas --install-wpp --install-energy` |
| Políticas y regiones | L0 regional con posterior y evaluación robusta de una política (F6, F8) | además `--install-food` |
| Datos y fuentes | Conjuntos instalados, vistas disponibles, registro de licencias | nada |

Cuando falta algo, la vista lo dice con el comando que lo instala.

## Usar

```bash
cd ui && npm install && npm run build      # una vez: compila a ui/dist
python world4_cli.py --serve               # http://127.0.0.1:8765/
```

## Desarrollar

```bash
python world4_cli.py --serve               # la API en :8765
cd ui && npm run dev                       # http://127.0.0.1:5173/ con recarga en caliente (proxy de /api)
npm run typecheck
```

Color: los tipos de violencia usan los tres primeros colores categóricos de la paleta de referencia (validados
con todos los pares en claro y oscuro); el riesgo usa una rampa naranja de 7 pasos de un solo tono, validada como
ordinal en claro y oscuro (`--r0` … `--r6` en `src/theme.css`). Mapa base: Natural Earth 1:110m (dominio público)
vía `world-atlas`.
