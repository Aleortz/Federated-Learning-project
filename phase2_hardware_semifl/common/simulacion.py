"""
SIMULACION LOCAL: un master y N workers como HILOS de un mismo proceso, hablando por
sockets reales (handshake + cifrado) en 127.0.0.1 con claves TEMPORALES.

Ejercita exactamente el mismo codigo que la ejecucion distribuida (Coordinador, FLWorker,
canal seguro) pero NO demuestra comunicacion entre cinco computadoras fisicas.
"""
import logging
import os
import socket
import tempfile
import threading

from cryptography.hazmat.primitives.asymmetric import ec

from common import config, data, reporting
from common import protocol as P
from master_edge.experimento import ejecutar_master
from worker_node.fl_worker import FLWorker


def puerto_libre():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def claves_temporales(carpeta, ids):
    pubs = []
    for i in ids:
        k = ec.generate_private_key(ec.SECP256R1())
        with open(os.path.join(carpeta, i + ".key"), "w") as f:
            f.write(k.private_numbers().private_value.to_bytes(32, "big").hex().upper() + "\n")
        pubs.append(f"{i}={P.pub_bytes(k.public_key()).hex().upper()}")
    with open(os.path.join(carpeta, "authorized.txt"), "w") as f:
        f.write("\n".join(pubs) + "\n")


def simular(cfg, csv=data.ADULT_CSV, simulaciones=None, reconnect_s=0.5, graficos=True, dataset=None):
    """
    Ejecuta un experimento completo en local. simulaciones: {worker_id: SimulacionCaida}.
    Devuelve (resumen, directorio_de_resultados).
    """
    simulaciones = simulaciones or {}
    ids = [f"WORKER_{i + 1}" for i in range(cfg.n_clients)]
    ds = dataset or data.cargar(csv, cfg.val_frac, cfg.test_frac, cfg.seed)
    workers, hilos, handlers = [], [], []
    with tempfile.TemporaryDirectory() as keydir:
        claves_temporales(keydir, [config.MASTER_ID] + ids)
        port = puerto_libre()

        def lanzar(coord):
            for w_id in ids:
                h = reporting.configurar_logging(os.path.join(coord.out, "logs", f"{w_id}.log"),
                                                 cfg.log_level, f"worker.{w_id}", consola=False)
                handlers.append((f"worker.{w_id}", h))
                w = FLWorker(w_id, "127.0.0.1", port, keydir, dataset=ds, reconnect_s=reconnect_s,
                             simulacion=simulaciones.get(w_id))
                w.start()
                workers.append(w)
                t = threading.Thread(target=w.run, daemon=True, name=w_id)
                t.start()
                hilos.append(t)

        try:
            return ejecutar_master(cfg, keydir, port, csv, tras_arrancar=lanzar, graficos=graficos)
        finally:
            for w in workers:
                w.stop()
            for t in hilos:
                t.join(timeout=3)
            for nombre, h in handlers:
                h.close()
                logging.getLogger(nombre).removeHandler(h)
