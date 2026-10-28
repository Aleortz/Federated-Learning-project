"""
SIMULACION LOCAL de un experimento completo (master + N workers como hilos, 127.0.0.1,
claves temporales). Usa el mismo codigo que la ejecucion distribuida, pero NO sustituye la
demostracion entre cinco computadoras.

    python -m scripts.run_local --scheme iid
    python -m scripts.run_local --scheme noniid --alpha 0.3
    python -m scripts.run_local --scheme iid --run-name disconnect --drop WORKER_3:4:offline:12

--drop WORKER_ID:RONDA:MODO:SEGUNDOS  simula la caida de un worker (modo offline|silent).
Se puede repetir. (desde phase2_hardware_semifl/)
"""
import argparse
import sys

from common import data, fl_config
from common.simulacion import simular
from worker_node.fl_worker import SimulacionCaida


def parse_drop(txt):
    try:
        w, r, modo, seg = txt.split(":")
        assert modo in ("offline", "silent")
        return w, SimulacionCaida(int(r), modo, float(seg))
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError("formato: WORKER_ID:RONDA:offline|silent:SEGUNDOS")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fl_config.agregar_argumentos(ap)
    ap.add_argument("--csv", default=data.ADULT_CSV)
    ap.add_argument("--drop", action="append", type=parse_drop, default=[])
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    try:
        cfg = fl_config.desde_args(args)
    except ValueError as e:
        sys.exit(f"[CONFIG] {e}")
    resumen, out = simular(cfg, args.csv, simulaciones=dict(args.drop), graficos=not args.no_plots)
    print(f"\nResultados en {out}")
    print({k: resumen[k] for k in ("rounds_completed", "rounds_failed", "final_test_accuracy",
                                    "final_test_f1_macro", "total_time_s", "aborted")})
    return 1 if resumen["aborted"] else 0


if __name__ == "__main__":
    sys.exit(main())
