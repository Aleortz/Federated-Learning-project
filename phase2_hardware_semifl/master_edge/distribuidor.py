"""
Master: reparte a cada worker su particion (IID o non-IID) por el canal seguro.

Viaja solo la lista de INDICES de las filas que le tocan al worker (los datos crudos
no salen de cada equipo). Como un mensaje admite como maximo config.MAX_MSG bytes, los
indices se envian en trozos:

    master -> worker   {"type": "partition", "scheme", "alpha", "seed", "client", "n",
                        "hist", "class_names", "chunks"}
    master -> worker   {"type": "partition_chunk", "i": 0, "indices": [...]}   x chunks
    worker -> master   {"type": "partition_ack", "n": ..., "hist": ...}

Uso desde el codigo de FL:

    from master_edge.distribuidor import distribuir
    manifiesto, acks = distribuir(srv, "noniid", labels, alpha=0.05)

Ejecutable solo (espera workers reales y reparte):

    python -m master_edge.distribuidor --scheme noniid --workers 2   (desde phase2_hardware_semifl/)
"""
import argparse
import sys
import time

from common import config, particion

CHUNK_INDICES = 4000        # ~6 B por indice en JSON -> ~24 KB por mensaje (tope 65535 B)


def asignar_workers(manifiesto, worker_ids):
    """Cliente i del manifiesto -> i-esimo worker (orden alfabetico, estable entre corridas)."""
    ids = sorted(worker_ids)
    if len(ids) != manifiesto["n_clients"]:
        raise ValueError(f"el manifiesto tiene {manifiesto['n_clients']} particiones y hay {len(ids)} workers")
    return {w: manifiesto["clients"][i] | {"posicion": i} for i, w in enumerate(ids)}


def enviar_particion(srv, worker_id, manifiesto, datos, posicion):
    indices = datos["indices"]
    trozos = [indices[i:i + CHUNK_INDICES] for i in range(0, len(indices), CHUNK_INDICES)] or [[]]
    srv.send(worker_id, {
        "type": "partition", "scheme": manifiesto["scheme"], "alpha": manifiesto["alpha"],
        "seed": manifiesto["seed"], "client": posicion, "n": datos["n"], "hist": datos["hist"],
        "class_names": manifiesto["class_names"], "chunks": len(trozos),
    })
    for i, t in enumerate(trozos):
        srv.send(worker_id, {"type": "partition_chunk", "i": i, "indices": t})


def distribuir(srv, scheme, labels, worker_ids=None, alpha=particion.ALPHA, seed=particion.SEED,
               min_size=particion.MIN_SIZE, class_names=particion.CLASES_ADULT,
               subset=None, timeout_ack=30):
    """
    Genera la particion 'iid'/'noniid' y la envia a los workers conectados.
    Devuelve (manifiesto, {worker_id: ack}). Si falta un ack, ese worker no aparece.
    """
    worker_ids = sorted(worker_ids if worker_ids is not None else srv.workers())
    if not worker_ids:
        raise ConnectionError("no hay workers conectados")
    manifiesto = particion.generar_particion(scheme, labels, len(worker_ids), alpha, seed,
                                             min_size, class_names, subset)
    asignacion = asignar_workers(manifiesto, worker_ids)
    for w, datos in asignacion.items():
        enviar_particion(srv, w, manifiesto, datos, datos["posicion"])

    acks, fin = {}, time.time() + timeout_ack
    while len(acks) < len(worker_ids) and time.time() < fin:
        r = srv.recv(timeout=max(0.1, fin - time.time()))
        if r is None:
            break
        w, msg = r
        if msg.get("type") == "partition_ack":
            acks[w] = msg
    return manifiesto, acks


def validar_acks(manifiesto, worker_ids, acks):
    """True si cada worker confirmo exactamente la particion que se le envio."""
    ok = True
    for i, w in enumerate(sorted(worker_ids)):
        esperado = manifiesto["clients"][i]
        a = acks.get(w)
        bien = bool(a) and a["n"] == esperado["n"] and a["hist"] == esperado["hist"]
        ok &= bien
        if not bien:
            print(f"[X] {w}: ack ausente o distinto de lo enviado")
    return ok


def main():
    from master_edge.server import MasterServer
    ap = argparse.ArgumentParser(description="Reparte particiones IID/non-IID a los workers")
    ap.add_argument("--scheme", choices=("iid", "noniid"), default="noniid")
    ap.add_argument("--workers", type=int, required=True, help="workers a esperar")
    ap.add_argument("--alpha", type=float, default=particion.ALPHA)
    ap.add_argument("--seed", type=int, default=particion.SEED)
    ap.add_argument("--csv", default=particion.ADULT_CSV)
    ap.add_argument("--png", default="results/particion_{scheme}.png")
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()

    labels = particion.cargar_etiquetas_adult(args.csv)
    try:
        srv = MasterServer()
    except (OSError, ValueError) as e:
        sys.exit(f"[CONFIG] {e}")
    srv.start()
    if not srv.wait_workers(args.workers, timeout=args.timeout):
        srv.stop()
        sys.exit(f"[X] solo {len(srv.workers())} de {args.workers} workers conectados")
    ids = srv.workers()[:args.workers]
    m, acks = distribuir(srv, args.scheme, labels, ids, alpha=args.alpha, seed=args.seed)
    hists = {w: acks[w]["hist"] for w in ids if w in acks}
    print(particion.tabla_texto(hists, m["class_names"], f"Particion {args.scheme} recibida por los workers"))
    if validar_acks(m, ids, acks):
        print("[OK] todos los workers confirmaron su particion")
        particion.grafico_comparativo({args.scheme: hists}, m["class_names"],
                                      args.png.format(scheme=args.scheme))
    srv.stop()


if __name__ == "__main__":
    main()
