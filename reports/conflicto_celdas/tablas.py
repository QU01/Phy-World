"""Tablas Markdown del informe F11 a partir de ``validacion.json`` (``world4_cli.py --validate-conflict-grid``).

    .venv/bin/python reports/conflicto_celdas/tablas.py [validacion.json]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = {"ensemble": "**Combinación**", "logistic": "Logística", "hawkes": "Hawkes",
         "bm_neighbors12": "ViEWS neighbors12", "bm_cell12": "cell12", "bm_persistence": "Persistencia",
         "bm_history": "Frecuencia histórica", "bm_zero": "ViEWS exactly zero", "bm_last": "ViEWS last historical"}
ORDER = ("ensemble", "logistic", "hawkes", "bm_neighbors12", "bm_cell12", "bm_history", "bm_persistence",
         "bm_zero", "bm_last")


def fmt(x, nd=4):
    return "–" if x is None else f"{x:.{nd}f}"


def prob_table(s, outcome, subset, block="all_h"):
    rows = s[outcome][subset]
    out = ["| Modelo | Brier | Log-loss | AUROC | AUPR | Eventos esperados / observados |", "|---|---|---|---|---|---|"]
    for m in ORDER:
        if m in rows:
            r = rows[m][block]
            if not r:
                continue
            out.append(f"| {NAMES[m]} | {fmt(r.get('brier'), 5)} | {fmt(r.get('log_loss'), 5)} | {fmt(r.get('auroc'), 3)} | "
                       f"{fmt(r.get('aupr'), 3)} | {r.get('expected', 0):.0f} / {r.get('events', 0):.0f} |")
    return "\n".join(out)


def crps_table(s, subset):
    rows = s["sb_deaths"][subset]
    out = ["| Modelo | CRPS, 1-12 meses | CRPS, h = 1 | CRPS, 2026 (candidatos) |", "|---|---|---|---|"]
    for m in ORDER:
        if m in rows:
            r = rows[m]
            out.append(f"| {NAMES[m]} | {fmt(r['all_h'].get('crps'))} | {fmt(r['h1'].get('crps'))} | "
                       f"{fmt(r['2026_candidate'].get('crps'))} |")
    return "\n".join(out)


def window_table(w, outcome, subset):
    rows = w[outcome][subset]
    out = ["| Modelo | Brier | AUROC | AUPR |", "|---|---|---|---|"]
    for m in ORDER:
        if m in rows:
            r = rows[m]
            out.append(f"| {NAMES[m]} | {fmt(r.get('brier'), 5)} | {fmt(r.get('auroc'), 3)} | {fmt(r.get('aupr'), 3)} |")
    return "\n".join(out)


def calibration_table(c, key):
    out = ["| Intervalo de p | p media | Frecuencia observada | Celda-mes |", "|---|---|---|---|"]
    edges = (0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
    for (ps, ys, n), a, b in zip(c[key], edges[:-1], edges[1:]):
        if n:
            out.append(f"| [{a}, {b}) | {ps / n:.4f} | {ys / n:.4f} | {n} |")
    return "\n".join(out)


def per_year(per, outcome="any", subset="all", models=("ensemble", "bm_neighbors12", "bm_cell12")):
    out = ["| Origen | " + " | ".join(f"AUPR {NAMES[m]}" for m in models) + " |", "|---" * (len(models) + 1) + "|"]
    origins = sorted({r["origin"] for r in per})
    for o in origins:
        vals = []
        for m in models:
            v = [r["aupr"] for r in per if r["origin"] == o and r["outcome"] == outcome and r["subset"] == subset
                 and r["model"] == m and r["aupr"] == r["aupr"]]
            vals.append(fmt(sum(v) / len(v), 3) if v else "–")
        out.append(f"| {o} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def main(path):
    v = json.loads(Path(path).read_text(encoding="utf-8"))
    s, w = v["summary"], v["window12"]
    print("## Cualquier evento, mundo, media de los horizontes 1-12\n")
    print(prob_table(s, "any", "all"))
    for o, name in (("sb", "estatal"), ("ns", "no estatal"), ("os", "unilateral")):
        print(f"\n## Violencia {name}, mundo\n")
        print(prob_table(s, o, "all"))
    print("\n## Cualquier evento, África y Oriente Medio\n")
    print(prob_table(s, "any", "ame"))
    print("\n## Cualquier evento, celdas sin historia (24 meses, celda y vecinas)\n")
    print(prob_table(s, "any", "onset"))
    print("\n## Cualquier evento, 2026 (verdad: eventos candidatos)\n")
    print(prob_table(s, "any", "all", "2026_candidate"))
    print("\n## Muertes estatales (CRPS), África y Oriente Medio\n")
    print(crps_table(s, "ame"))
    print("\n## Muertes estatales (CRPS), mundo\n")
    print(crps_table(s, "all"))
    print("\n## Al menos un evento en los 12 meses, mundo\n")
    print(window_table(w, "any", "all"))
    print("\n## Al menos un evento estatal en los 12 meses, mundo\n")
    print(window_table(w, "sb", "all"))
    print("\n## AUPR por origen (cualquier evento, mundo)\n")
    print(per_year(v["per_origin_month"]))
    print("\n## Calibración de la combinación (cualquier evento, h 1-3)\n")
    print(calibration_table(v["calibration"], "ensemble|any|h1-3"))
    print("\n## Razón de ramificación del proceso de Hawkes (hijos por evento del tipo de la fila)\n")
    print("| Origen | sb → sb | ns → ns | os → os | os → sb | sb → os |\n|---|---|---|---|---|---|")
    for o, h in v["hawkes"].items():
        b = h["branching"]
        print(f"| {o} | {b['sb']['sb']:.3f} | {b['ns']['ns']:.3f} | {b['os']['os']:.3f} | {b['sb']['os']:.3f} | "
              f"{b['os']['sb']:.3f} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE / "validacion.json")
