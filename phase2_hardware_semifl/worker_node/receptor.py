"""
Worker: recibe su particion de datos del master por el canal seguro.

    from worker_node.receptor import recibir_particion
    p = recibir_particion(cli, timeout=60)   # cli: WorkerClient ya conectado
    p["indices"]   # filas del dataset local que le tocan a este worker
    p["hist"]      # muestras por clase

Confirma al master con un "partition_ack" (n e histograma de lo recibido). Si se le
pasa `labels` (su copia local del dataset), recalcula el histograma por su cuenta en
vez de fiarse del que anuncia el master.

Ejecutable solo (espera una particion y la imprime):

    python -m worker_node.receptor --id WORKER_1 --master 192.168.1.50   (desde phase2_hardware_semifl/)
"""
import argparse
import sys
import time

import numpy as np

from common import config, particion


def recibir_particion(cli, timeout=60, labels=None):
    """Bloquea hasta recibir la particion completa; devuelve un dict o None si vence el timeout."""
    fin = time.time() + timeout
    cabecera, trozos = None, {}
    while time.time() < fin:
        msg = cli.recv(timeout=max(0.1, fin - time.time()))
        if msg is None:
            break
        tipo = msg.get("type")
        if tipo == "partition":
            cabecera, trozos = msg, {}
        elif tipo == "partition_chunk" and cabecera is not None:
            trozos[msg["i"]] = msg["indices"]
        else:
            continue            # otro tipo de mensaje: no es de este modulo
        if cabecera is not None and len(trozos) == cabecera["chunks"]:
            indices = [x for i in range(cabecera["chunks"]) for x in trozos[i]]
            if len(indices) != cabecera["n"]:
                raise ValueError(f"particion corrupta: {len(indices)} indices de {cabecera['n']}")
            hist = cabecera["hist"]
            if labels is not None:
                hist = np.bincount(np.asarray(labels)[indices],
                                   minlength=len(cabecera["class_names"])).tolist()
            cli.send({"type": "partition_ack", "n": len(indices), "hist": hist})
            return {**{k: cabecera[k] for k in ("scheme", "alpha", "seed", "client", "class_names")},
                    "indices": indices, "n": len(indices), "hist": hist}
    return None


def main():
    from worker_node.client import WorkerClient
    ap = argparse.ArgumentParser(description="Worker: recibe su particion")
    ap.add_argument("--id", required=True)
    ap.add_argument("--master", default=config.MASTER_HOST)
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--csv", default=particion.ADULT_CSV)
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()
    try:
        cli = WorkerClient(args.id, args.master, args.port)
    except (OSError, ValueError) as e:
        sys.exit(f"[CONFIG] {e}")
    cli.start()
    if not cli.wait_connected(30):
        sys.exit("[X] sin sesion con el master")
    p = recibir_particion(cli, args.timeout, labels=particion.cargar_etiquetas_adult(args.csv))
    cli.stop()
    if not p:
        sys.exit("[X] no llego ninguna particion")
    print(particion.tabla_texto({args.id: p["hist"]}, p["class_names"],
                                f"{args.id}: particion {p['scheme']} (cliente {p['client']})"))


if __name__ == "__main__":
    main()
