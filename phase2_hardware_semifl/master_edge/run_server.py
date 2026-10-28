"""
Master (maquina del servidor): ejecuta un experimento federado con workers REALES.

    python -m master_edge.run_server --scheme iid
    python -m master_edge.run_server --scheme noniid --alpha 0.3
    python -m master_edge.run_server --scheme iid --run-name disconnect --min-clients 2 --round-timeout 30

(desde phase2_hardware_semifl/). Espera a `--n-clients` workers, reparte la particion,
ejecuta las rondas y escribe results/<run-name>/. Ver README.md para el procedimiento.
"""
import argparse
import sys

from common import config, data, fl_config


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fl_config.agregar_argumentos(ap)
    ap.add_argument("--port", type=int, default=config.PORT, help="puerto TCP de escucha")
    ap.add_argument("--keydir", default=config.KEYDIR)
    ap.add_argument("--csv", default=data.ADULT_CSV, help="ruta de adult.csv")
    args = ap.parse_args(argv)
    try:
        cfg = fl_config.desde_args(args)
    except ValueError as e:
        sys.exit(f"[CONFIG] {e}")

    from master_edge.experimento import ejecutar_master
    try:
        resumen, out = ejecutar_master(cfg, args.keydir, args.port, args.csv)
    except (OSError, ValueError, data.DatasetError, TimeoutError) as e:
        sys.exit(f"[X] {e}")
    print(f"\nResultados en {out}")
    print({k: resumen[k] for k in ("rounds_completed", "rounds_failed", "final_test_accuracy",
                                    "final_test_f1_macro", "total_time_s", "aborted")})
    return 1 if resumen["aborted"] else 0


if __name__ == "__main__":
    sys.exit(main())
