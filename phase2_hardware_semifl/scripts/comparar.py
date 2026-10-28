"""
Genera results/comparison/ (summary.csv, summary.json, plots/) a partir de las ejecuciones
IID y non-IID YA hechas, y (si existen) el grafico de resiliencia de la prueba de desconexion.

    python -m scripts.comparar [--results results]      (desde phase2_hardware_semifl/)
"""
import argparse
import os
import sys

from common import reporting


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--iid", default="iid")
    ap.add_argument("--noniid", default="non_iid")
    ap.add_argument("--disconnect", default="disconnect")
    ap.add_argument("--baseline", default="baseline")
    args = ap.parse_args(argv)
    r = args.results
    dirs = {"iid": os.path.join(r, args.iid), "non_iid": os.path.join(r, args.noniid)}
    faltan = [d for d in dirs.values() if not os.path.exists(os.path.join(d, "global_metrics.csv"))]
    if faltan:
        sys.exit(f"[X] faltan resultados en {faltan}. Ejecuta primero los experimentos IID y non-IID.")
    resumen, avisos = reporting.comparar(dirs, os.path.join(r, "comparison"))
    for a in avisos:
        print("[AVISO]", a)
    ancho = max(len(k) for k in next(iter(resumen.values())))
    print(f"{'indicador':<{ancho}}  {'iid':>12}  {'non_iid':>12}")
    for k in next(iter(resumen.values())):
        v = [resumen[n][k] for n in dirs]
        print(f"{k:<{ancho}}  " + "  ".join(f"{x:>12.4f}" if isinstance(x, float) else f"{str(x):>12}" for x in v))
    dd = os.path.join(r, args.disconnect)
    if os.path.exists(os.path.join(dd, "events.csv")):
        base = os.path.join(r, args.baseline)
        print("Grafico de resiliencia:", reporting.grafico_resiliencia(dd, base))
    print("Comparacion en", os.path.join(r, "comparison"))


if __name__ == "__main__":
    main()
