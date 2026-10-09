"""
Master edge: servidor seguro al que se conectan los workers.

Solo se encarga de la COMUNICACION: autentica a cada worker, mantiene una sesion
cifrada por worker y entrega/recibe mensajes JSON. La logica de FL (agregacion,
scoring, sincronizacion con la nube) usa esta API desde sus propios modulos:

    from master_edge.server import MasterServer

    srv = MasterServer()                 # lee keys/MASTER.key y keys/authorized.txt
    srv.start()                          # escucha en segundo plano
    srv.wait_workers(3, timeout=60)      # espera a que se conecten 3 workers

    srv.broadcast({"type": "global", "round": 1, "W": W, "b": b})   # arrays numpy OK
    worker_id, msg = srv.recv(timeout=30)    # (id autenticado, dict) o None

    srv.stop()

Tambien se puede ejecutar sola para probar la comunicacion:

    python -m master_edge.server              (desde phase2_hardware_semifl/)

Consola: p (workers), k (cadenas de claves), @ID texto, @ID 1..6 (pruebas), q
"""
import argparse
import queue
import socket
import sys
import threading

from common import config
from common import protocol as P


class MasterServer:
    def __init__(self, my_id=config.MASTER_ID, keydir=config.KEYDIR, port=config.PORT,
                 verbose=False, on_message=None, on_connect=None, on_disconnect=None):
        """
        on_message(worker_id, msg)   se llama con cada mensaje autentico. Si es None,
                                     los mensajes se encolan y se leen con recv().
        on_connect(worker_id)        un worker completo el handshake.
        on_disconnect(worker_id)     se cerro la sesion de un worker.
        Los callbacks corren en el hilo de ese worker: deben ser rapidos.
        """
        self.id = my_id
        self.priv, self.autorizados = P.cargar_identidad(keydir, my_id)
        self.port = port
        self.verbose = verbose
        self.on_message = on_message
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.inbox = queue.Queue()
        self._sesiones = {}                 # worker_id -> Sesion
        self._lock = threading.Lock()
        self._cambio = threading.Condition(self._lock)
        self._run = False
        self._sock = None

    # ---------------- API ----------------
    def start(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", self.port))
        s.listen(16)
        s.settimeout(1)
        self._sock = s
        self._run = True
        threading.Thread(target=self._aceptar, daemon=True).start()
        print(f"[MASTER] {self.id} escuchando en TCP {self.port} | huella {P.huella_pub(self.priv.public_key())}")

    def stop(self):
        self._run = False
        with self._lock:
            sesiones = list(self._sesiones.values())
        for s in sesiones:
            s.cerrar()
        if self._sock:
            self._sock.close()

    def workers(self):
        """IDs de los workers conectados ahora."""
        with self._lock:
            return sorted(self._sesiones)

    def wait_workers(self, n, timeout=None):
        """Bloquea hasta que haya al menos n workers conectados. Devuelve True/False."""
        with self._cambio:
            return self._cambio.wait_for(lambda: len(self._sesiones) >= n, timeout)

    def send(self, worker_id, msg):
        """Envia un dict (con campo "type") cifrado a un worker. Devuelve el SEQ."""
        with self._lock:
            s = self._sesiones.get(worker_id)
        if not s:
            raise ConnectionError(f"'{worker_id}' no esta conectado")
        return s.enviar(P.codificar(msg))

    def broadcast(self, msg, worker_ids=None):
        """Envia el mismo mensaje a todos (o a los IDs dados). Devuelve los IDs a los que llego."""
        data = P.codificar(msg)
        with self._lock:
            destinos = {w: s for w, s in self._sesiones.items() if worker_ids is None or w in worker_ids}
        ok = []
        for w, s in destinos.items():
            try:
                s.enviar(data)
                ok.append(w)
            except OSError as e:
                print(f"[X] No se pudo enviar a {w}: {e}")
        return ok

    def recv(self, timeout=None):
        """(worker_id, msg) del siguiente mensaje, o None si vence el timeout."""
        try:
            return self.inbox.get(timeout=timeout)
        except queue.Empty:
            return None

    # ---------------- interno ----------------
    def _aceptar(self):
        while self._run:
            try:
                c, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._atender, args=(c, addr[0]), daemon=True).start()

    def _atender(self, c, ip):
        try:
            ses = P.handshake_respondedor(c, self.id, self.priv, self.autorizados, ip, self.verbose)
        except P.HandshakeError as e:
            print(f"[X] Handshake desde {ip} fallido: {e}")
            c.close()
            return
        except (OSError, ConnectionError, ValueError) as e:
            print(f"[X] Conexion desde {ip} sin handshake completo ({e}).")
            c.close()
            return
        ses.ip = ip
        with self._cambio:
            vieja = self._sesiones.get(ses.peer_id)
            self._sesiones[ses.peer_id] = ses
            self._cambio.notify_all()
        if vieja:
            print(f"[i] {ses.peer_id} se reconecto: se reemplaza su sesion anterior.")
            vieja.cerrar()
        if self.on_connect:
            self.on_connect(ses.peer_id)
        self._rx(ses)

    def _rx(self, ses):
        try:
            while self._run and ses.activa:
                fr = ses.leer_frame()
                if fr[0] == "ack":
                    _, seq, st, rtt = fr
                    if not st:
                        print(f"[!] {ses.peer_id} RECHAZO nuestro paquete SEQ {seq}")
                    elif self.verbose:
                        print(f"[ACK <- {ses.peer_id}] SEQ {seq} aceptado | RTT {rtt or 0:.2f} ms")
                    continue
                _, ok, seq, x = fr
                if not ok:
                    print(f"[RECHAZADO] Paquete de {ses.peer_id} (SEQ {seq}): {x}")
                    print(f"[ALERTA] Intento de manipulacion en la sesion con {ses.peer_id}.")
                    continue
                msg = P.decodificar(x)
                if self.on_message:
                    self.on_message(ses.peer_id, msg)
                else:
                    self.inbox.put((ses.peer_id, msg))
        except (OSError, ConnectionError) as e:
            if ses.activa:
                print(f"[-] Sesion con {ses.peer_id} cerrada: {e}")
        finally:
            ses.cerrar()
            quitado = False
            with self._cambio:
                if self._sesiones.get(ses.peer_id) is ses:
                    del self._sesiones[ses.peer_id]
                    quitado = True
                    self._cambio.notify_all()
            if quitado and self.on_disconnect:
                self.on_disconnect(ses.peer_id)


# =====================================================================
#  Modo prueba: consola interactiva
# =====================================================================
def _consola(srv):
    while True:
        try:
            linea = input().strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not linea:
            continue
        if linea == "q":
            return
        with srv._lock:
            sesiones = dict(srv._sesiones)
        if linea == "p":
            for w in sorted(srv.autorizados):
                if w == srv.id:
                    continue
                s = sesiones.get(w)
                print(f" - {w:19s} " + (f"CONECTADO desde {s.ip} (SID 0x{s.sid:08X}, TX {s.tx_seq}, RX {s.rx_last})"
                                        if s else "desconectado"))
            continue
        if linea == "k":
            for w, s in sesiones.items():
                print(f"{w}:\n  CK_tx:\n{P.hx(s.k_tx)}\n  CK_rx:\n{P.hx(s.k_rx)}")
            continue
        if not linea.startswith("@") or " " not in linea:
            print("Uso: @ID texto | @ID 1..6 | p | k | q")
            continue
        w, texto = linea[1:].split(" ", 1)
        s = sesiones.get(w)
        if not s:
            print(f"'{w}' no esta conectado.")
            continue
        try:
            if texto in ("1", "2", "3", "5", "6"):
                s.enviar(P.MENSAJE_PRUEBA, int(texto))
            elif texto == "4":
                s.replay()
            else:
                s.enviar(P.codificar({"type": "text", "text": texto}))
        except (OSError, ValueError) as e:
            print(f"[X] {e}")


def main():
    ap = argparse.ArgumentParser(description="Master edge seguro (modo prueba)")
    ap.add_argument("--id", default=config.MASTER_ID)
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--keydir", default=config.KEYDIR)
    ap.add_argument("-v", "--verbose", action="store_true", help="volcados hex de claves y paquetes")
    args = ap.parse_args()
    try:
        srv = MasterServer(args.id, args.keydir, args.port, args.verbose,
                           on_message=lambda w, m: print(f"[MSG <- {w}] {m}"))
    except (OSError, ValueError) as e:
        sys.exit(f"[CONFIG] {e}\nGenera las claves con: python -m common.keys --genkey {args.id}")
    print("Workers autorizados:")
    for w, pub in sorted(srv.autorizados.items()):
        if w != srv.id:
            print(f"  {w:19s} {P.huella_pub(pub)}")
    srv.start()
    _consola(srv)
    srv.stop()


if __name__ == "__main__":
    main()
