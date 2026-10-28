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


class EnsambladorParticion:
    """
    Reensambla la particion a partir de los mensajes `partition` + `partition_chunk`, uno a
    uno (sirve tanto para recibir_particion() como para un bucle de mensajes mas general).
    agregar(msg) devuelve None mientras falten trozos y el dict de la particion al completarse.
    """

    def __init__(self, labels=None):
        self.labels = labels
        self.cabecera, self.trozos = None, {}

    def agregar(self, msg):
        tipo = msg.get("type")
        if tipo == "partition":
            self.cabecera, self.trozos = msg, {}
        elif tipo == "partition_chunk" and self.cabecera is not None:
            self.trozos[msg["i"]] = msg["indices"]
        else:
            return None
        c = self.cabecera
        if c is None or len(self.trozos) != c["chunks"]:
            return None
        indices = [x for i in range(c["chunks"]) for x in self.trozos[i]]
        self.cabecera, self.trozos = None, {}
        if len(indices) != c["n"]:
            raise ValueError(f"particion corrupta: {len(indices)} indices de {c['n']}")
        hist = c["hist"]
        if self.labels is not None:
            arr = np.asarray(indices, dtype=np.int64)
            if len(arr) and (arr.min() < 0 or arr.max() >= len(self.labels)):
                raise ValueError("particion con indices fuera del dataset local")
            hist = np.bincount(np.asarray(self.labels)[arr], minlength=len(c["class_names"])).tolist()
        return {**{k: c[k] for k in ("scheme", "alpha", "seed", "client", "class_names")},
                "indices": indices, "n": len(indices), "hist": hist}


def recibir_particion(cli, timeout=60, labels=None):
    """Bloquea hasta recibir la particion completa; devuelve un dict o None si vence el timeout."""
    fin = time.time() + timeout
    ens = EnsambladorParticion(labels)
    while time.time() < fin:
        msg = cli.recv(timeout=max(0.1, fin - time.time()))
        if msg is None:
            break
        p = ens.agregar(msg)
        if p is not None:
            cli.send({"type": "partition_ack", "n": p["n"], "hist": p["hist"]})
            return p
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
