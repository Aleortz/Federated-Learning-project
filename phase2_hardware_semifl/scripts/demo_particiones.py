"""
Demo/validacion de la distribucion IID vs non-IID sobre el canal seguro (un solo equipo).

Levanta un master y N workers reales (handshake + cifrado) con claves TEMPORALES, reparte
la particion IID y luego la non-IID, y muestra lo que cada WORKER recibio: tabla en consola
y grafico PNG con las muestras por clase de cada cliente.

    python -m scripts.demo_particiones [--workers 4] [--alpha 0.05] [--seed 42]
                                       (desde phase2_hardware_semifl/)
"""
import argparse
import os
import socket
import sys
import tempfile
import threading

import numpy as np
from cryptography.hazmat.primitives.asymmetric import ec

from common import particion
from common import protocol as P
from master_edge.distribuidor import distribuir, validar_acks
from master_edge.server import MasterServer
from worker_node.client import WorkerClient
from worker_node.receptor import recibir_particion

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _puerto_libre():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _claves_temporales(carpeta, ids):
    pubs = []
    for i in ids:
        k = ec.generate_private_key(ec.SECP256R1())
        with open(os.path.join(carpeta, i + ".key"), "w") as f:
            f.write(k.private_numbers().private_value.to_bytes(32, "big").hex().upper() + "\n")
        pubs.append(f"{i}={P.pub_bytes(k.public_key()).hex().upper()}")
    with open(os.path.join(carpeta, "authorized.txt"), "w") as f:
        f.write("\n".join(pubs) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=particion.ALPHA)
    ap.add_argument("--seed", type=int, default=particion.SEED)
    ap.add_argument("--csv", default=particion.ADULT_CSV)
    ap.add_argument("--png", default=os.path.join(_BASE, "results", "particiones_iid_vs_noniid.png"))
    args = ap.parse_args()

    labels = particion.cargar_etiquetas_adult(args.csv)
    clases = particion.CLASES_ADULT
    print(f"Dataset: {len(labels)} filas | clases {dict(zip(clases, np.bincount(labels).tolist()))}\n")

    ids = [f"WORKER_{i + 1}" for i in range(args.workers)]
    puerto = _puerto_libre()
    recibido = {}            # esquema -> {worker: particion}
    manifiestos = {}
    ok_global = True

    with tempfile.TemporaryDirectory() as keydir:
        _claves_temporales(keydir, ["MASTER"] + ids)
        srv = MasterServer("MASTER", keydir, puerto)
        srv.start()
        clientes = [WorkerClient(w, "127.0.0.1", puerto, "MASTER", keydir) for w in ids]
        for c in clientes:
            c.start()
        if not srv.wait_workers(len(ids), timeout=60):
            sys.exit("[X] los workers no se conectaron")

        for esquema in ("iid", "noniid"):
            print(f"\n=== Distribucion {esquema.upper()} por canal seguro ===")
            rec = {}

            def worker(c):
                rec[c.id] = recibir_particion(c, timeout=60, labels=labels)

            hilos = [threading.Thread(target=worker, args=(c,)) for c in clientes]
            for h in hilos:
                h.start()
            m, acks = distribuir(srv, esquema, labels, ids, alpha=args.alpha, seed=args.seed)
            for h in hilos:
                h.join()
            ok = validar_acks(m, ids, acks) and all(rec.get(w) for w in ids)
            # cada indice debe llegar a exactamente un worker y todos deben cubrir el pool
            todos = [i for w in ids for i in rec[w]["indices"]]
            ok &= len(todos) == len(set(todos)) == m["total_samples"]
            # los indices recibidos son los del manifiesto (se compara contra el reparto original)
            ok &= all(rec[w]["indices"] == m["clients"][i]["indices"] for i, w in enumerate(sorted(ids)))
            ok_global &= ok
            hist = {w: rec[w]["hist"] for w in ids}
            recibido[esquema], manifiestos[esquema] = hist, m
            extra = f", alpha={args.alpha}, aceptado en intento {m['intentos']}" if esquema == "noniid" else ""
            print(particion.tabla_texto(
                hist, clases, f"Muestras por clase recibidas por cada worker ({esquema}{extra})"))
            print(f"[{'OK' if ok else 'FALLO'}] acks coinciden, indices disjuntos y completos "
                  f"({m['total_samples']} muestras)")

        for c in clientes:
            c.stop()
        srv.stop()

    png = particion.grafico_comparativo(recibido, clases, args.png)
    e_iid = np.mean([particion.entropia_clases(c["pct"]) for c in manifiestos["iid"]["clients"].values()])
    e_non = np.mean([particion.entropia_clases(c["pct"]) for c in manifiestos["noniid"]["clients"].values()])
    print(f"\nEntropia media de clases: IID {e_iid:.3f} bits vs non-IID {e_non:.3f} bits")
    print(f"Grafico guardado en: {png}")
    sys.exit(0 if ok_global else 1)


if __name__ == "__main__":
    main()
