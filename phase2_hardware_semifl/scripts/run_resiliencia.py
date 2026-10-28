"""
Experimento de resiliencia en SIMULACION LOCAL: misma configuracion con 4 clientes siempre
disponibles ("baseline") y con un cliente que se cae temporalmente ("disconnect").

    python -m scripts.run_resiliencia [--drop-worker WORKER_3] [--drop-round 4] [--drop-seconds 12]
                                      [--mode offline|silent] [--scheme iid]

La caida es simulada por el propio worker; la prueba real (tarjeta de red) se hace a mano,
ver README. Una sola ejecucion no permite atribuir causalidad: ver las limitaciones del README.
"""
import argparse
import sys

from common import data, fl_config
from common.simulacion import simular
from worker_node.fl_worker import SimulacionCaida


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fl_config.agregar_argumentos(ap)
    ap.add_argument("--csv", default=data.ADULT_CSV)
    ap.add_argument("--drop-worker", default="WORKER_3")
    ap.add_argument("--drop-round", type=int, default=4)
    ap.add_argument("--drop-seconds", type=float, default=12.0)
    ap.add_argument("--mode", choices=("offline", "silent"), default="offline")
    args = ap.parse_args(argv)
    pausa = args.round_delay if args.round_delay is not None else 2.0
    timeout = args.round_timeout if args.round_timeout is not None else 8.0
    try:
        base = fl_config.desde_args(args, run_name="baseline", round_delay=pausa, round_timeout=timeout)
        caida = fl_config.desde_args(args, run_name="disconnect", round_delay=pausa, round_timeout=timeout)
    except ValueError as e:
        sys.exit(f"[CONFIG] {e}")
    print("== referencia: 4 clientes disponibles ==")
    r0, d0 = simular(base, args.csv)
    print(f"== con caida de {args.drop_worker} (ronda {args.drop_round}, {args.mode}, {args.drop_seconds}s) ==")
    sim = {args.drop_worker: SimulacionCaida(args.drop_round, args.mode, args.drop_seconds)}
    r1, d1 = simular(caida, args.csv, simulaciones=sim)
    from common import reporting
    print("grafico:", reporting.grafico_resiliencia(d1, d0))
    for n, r in (("baseline", r0), ("disconnect", r1)):
        print(n, {k: r[k] for k in ("rounds_completed", "rounds_failed", "final_test_accuracy", "total_time_s")})


if __name__ == "__main__":
    main()
