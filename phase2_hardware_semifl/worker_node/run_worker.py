"""
Worker (cada maquina cliente): se conecta al master y participa en el experimento.

    python -m worker_node.run_worker --id WORKER_1 --master 192.168.1.50

(desde phase2_hardware_semifl/). Necesita data/adult.csv, keys/<ID>.key y la publica del
master en keys/authorized.txt. Si se pierde la conexion (p. ej. se desactiva la tarjeta de
red) el worker reintenta solo y, al volver, el master lo sincroniza y lo incorpora a la
siguiente ronda. Se detiene cuando el master envia 'fin' o con Ctrl+C.
"""
import argparse
import os
import sys

from common import config, data, reporting
from worker_node.fl_worker import FLWorker, SimulacionCaida


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True, help="ID del worker (el mismo de su clave en keys/)")
    ap.add_argument("--master", default=config.MASTER_HOST, help="IP del master (o SFL_MASTER_HOST)")
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--keydir", default=config.KEYDIR)
    ap.add_argument("--csv", default=data.ADULT_CSV)
    ap.add_argument("--seed", type=int, default=42, help="debe coincidir con el del master")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--reconnect-s", type=float, default=config.RECONNECT_S)
    ap.add_argument("--max-idle", type=float, default=None,
                    help="si el master no envia nada en N s, el worker termina (por defecto espera siempre)")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--log-dir", default="results/worker_logs")
    ap.add_argument("--sim-drop-round", type=int, default=None,
                    help="SOLO PRUEBAS: simula una caida al recibir esta ronda")
    ap.add_argument("--sim-drop-mode", choices=("offline", "silent"), default="offline")
    ap.add_argument("--sim-drop-seconds", type=float, default=15.0)
    args = ap.parse_args(argv)

    reporting.configurar_logging(os.path.join(args.log_dir, f"{args.id}.log"), args.log_level, "worker")
    sim = (SimulacionCaida(args.sim_drop_round, args.sim_drop_mode, args.sim_drop_seconds)
           if args.sim_drop_round else None)
    try:
        w = FLWorker(args.id, args.master, args.port, args.keydir, csv=args.csv, val_frac=args.val_frac,
                     test_frac=args.test_frac, seed=args.seed, reconnect_s=args.reconnect_s, simulacion=sim)
    except (OSError, ValueError, data.DatasetError) as e:
        sys.exit(f"[CONFIG] {e}")
    w.start()
    try:
        ok = w.run(args.max_idle)
    except KeyboardInterrupt:
        ok = False
    finally:
        w.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
