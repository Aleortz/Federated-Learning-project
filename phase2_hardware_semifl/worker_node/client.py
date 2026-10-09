"""
Worker node: cliente seguro que se conecta al master edge.

Solo se encarga de la COMUNICACION: abre una sesion autenticada y cifrada con el
master, se reconecta solo si se cae y entrega/recibe mensajes JSON. El
entrenamiento local (logistic_model.py) y la telemetria (telemetry.py) la usan asi:

    from worker_node.client import WorkerClient

    cli = WorkerClient("WORKER_1", master_host="192.168.1.50")
    cli.start()
    cli.wait_connected(timeout=30)

    msg = cli.recv(timeout=60)           # ej. {"type": "global", "round": 1, "W": [...], "b": [...]}
    cli.send({"type": "params", "round": 1, "W": W_k, "b": b_k, "n": n_k})   # arrays numpy OK

    cli.stop()

Tambien se puede ejecutar solo para probar la comunicacion:

    python -m worker_node.client --id WORKER_1 --master 192.168.1.50

Consola: texto o JSON (se envia), 1..6 (pruebas), p (estado), k (claves), q
"""
import argparse
import json
import queue
import socket
import sys
import threading
import time

from common import config
from common import protocol as P


class WorkerClient:
    def __init__(self, worker_id, master_host=config.MASTER_HOST, port=config.PORT,
                 master_id=config.MASTER_ID, keydir=config.KEYDIR, verbose=False, on_message=None):
        """
        on_message(msg)   se llama con cada mensaje autentico del master. Si es None,
                          los mensajes se encolan y se leen con recv().
        """
        self.id = worker_id
        self.priv, autorizados = P.cargar_identidad(keydir, worker_id)
        if master_id not in autorizados:
            raise ValueError(f"falta la clave publica de '{master_id}' en authorized.txt "
                             f"(python -m common.keys --addpub {master_id}=04...)")
        self.master = (master_host, port)
        self.master_id = master_id
        self.master_pub = autorizados[master_id]
        self.verbose = verbose
        self.on_message = on_message
        self.inbox = queue.Queue()
        self.ses = None
        self._conectado = threading.Event()
        self._run = False

    # ---------------- API ----------------
    def start(self):
        self._run = True
        threading.Thread(target=self._mantener, daemon=True).start()
        print(f"[WORKER] {self.id} -> {self.master_id} en {self.master[0]}:{self.master[1]} "
              f"(huella esperada {P.huella_pub(self.master_pub)})")

    def stop(self):
        self._run = False
        if self.ses:
            self.ses.cerrar()

    @property
    def connected(self):
        return self.ses is not None and self.ses.activa

    def wait_connected(self, timeout=None):
        return self._conectado.wait(timeout)

    def send(self, msg):
        """Envia un dict (con campo "type") cifrado al master. Devuelve el SEQ."""
        ses = self.ses
        if not (ses and ses.activa):
            raise ConnectionError("no hay sesion con el master")
        return ses.enviar(P.codificar(msg))

    def recv(self, timeout=None):
        """Siguiente mensaje del master (dict), o None si vence el timeout."""
        try:
            return self.inbox.get(timeout=timeout)
        except queue.Empty:
            return None

    # ---------------- interno ----------------
    def _conectar(self):
        """Sesion si todo va bien; None si el master no responde; False si el handshake falla."""
        try:
            c = socket.create_connection(self.master, timeout=5)
        except OSError:
            return None
        try:
            return P.handshake_iniciador(c, self.id, self.priv, self.master_id, self.master_pub, self.verbose)
        except P.HandshakeError as e:
            print(f"[X] {e}")
        except (OSError, ConnectionError, ValueError) as e:
            print(f"[X] Handshake interrumpido: {e}")
        c.close()
        return False

    def _mantener(self):
        avisado = False
        while self._run:
            if not self.connected:
                self._conectado.clear()
                ses = self._conectar()
                if ses:
                    self.ses = ses
                    avisado = False
                    self._conectado.set()
                    threading.Thread(target=self._rx, args=(ses,), daemon=True).start()
                elif ses is None and not avisado:
                    print(f"[..] Master {self.master[0]}:{self.master[1]} no disponible; "
                          f"reintento cada {config.RECONNECT_S} s.")
                    avisado = True
            time.sleep(config.RECONNECT_S)

    def _rx(self, ses):
        try:
            while self._run and ses.activa:
                fr = ses.leer_frame()
                if fr[0] == "ack":
                    _, seq, st, rtt = fr
                    if not st:
                        print(f"[!] El master RECHAZO nuestro paquete SEQ {seq}")
                    elif self.verbose:
                        print(f"[ACK <- {ses.peer_id}] SEQ {seq} aceptado | RTT {rtt or 0:.2f} ms")
                    continue
                _, ok, seq, x = fr
                if not ok:
                    print(f"[RECHAZADO] Paquete del master (SEQ {seq}): {x}")
                    print("[ALERTA] Intento de manipulacion en la sesion con el master.")
                    continue
                msg = P.decodificar(x)
                if self.on_message:
                    self.on_message(msg)
                else:
                    self.inbox.put(msg)
        except (OSError, ConnectionError) as e:
            if ses.activa:
                print(f"[-] Sesion con {ses.peer_id} cerrada: {e}")
        finally:
            ses.cerrar()


# =====================================================================
#  Modo prueba: consola interactiva
# =====================================================================
def _consola(cli):
    while True:
        try:
            linea = input().strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not linea:
            continue
        if linea == "q":
            return
        ses = cli.ses
        if linea == "p":
            print(f"CONECTADO (SID 0x{ses.sid:08X}, TX {ses.tx_seq}, RX {ses.rx_last})"
                  if cli.connected else "Desconectado (reintentando).")
            continue
        if not cli.connected:
            print("No hay sesion con el master todavia.")
            continue
        try:
            if linea == "k":
                print(f"CK_tx:\n{P.hx(ses.k_tx)}\nCK_rx:\n{P.hx(ses.k_rx)}")
            elif linea in ("1", "2", "3", "5", "6"):
                ses.enviar(P.MENSAJE_PRUEBA, int(linea))
            elif linea == "4":
                ses.replay()
            else:
                try:
                    msg = json.loads(linea)
                    if not (isinstance(msg, dict) and "type" in msg):
                        raise ValueError
                except ValueError:
                    msg = {"type": "text", "text": linea}
                cli.send(msg)
        except (OSError, ValueError) as e:
            print(f"[X] {e}")


def main():
    ap = argparse.ArgumentParser(description="Worker seguro (modo prueba)")
    ap.add_argument("--id", required=True, help="ID de este worker, ej. WORKER_1")
    ap.add_argument("--master", default=config.MASTER_HOST, help="IP del master")
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--master-id", default=config.MASTER_ID)
    ap.add_argument("--keydir", default=config.KEYDIR)
    ap.add_argument("-v", "--verbose", action="store_true", help="volcados hex de claves y paquetes")
    args = ap.parse_args()
    try:
        cli = WorkerClient(args.id, args.master, args.port, args.master_id, args.keydir, args.verbose,
                           on_message=lambda m: print(f"[MSG <- master] {m}"))
    except (OSError, ValueError) as e:
        sys.exit(f"[CONFIG] {e}\nGenera tu clave con: python -m common.keys --genkey {args.id}")
    cli.start()
    _consola(cli)
    cli.stop()


if __name__ == "__main__":
    main()
