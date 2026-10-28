"""
Coordinador federado del master: reparte las particiones, ejecuta las rondas con FedAvg,
aplica timeouts y politica de participacion minima, sincroniza a los workers que se
reconectan y registra metricas, eventos y checkpoints.

Todo el estado lo modifica UN solo hilo (el de ejecutar()); los hilos de red del
MasterServer solo encolan eventos ("connect", "disconnect", "msg").

Protocolo de aplicacion (sobre el canal seguro, mensajes JSON con campo "type"):
  worker -> master  register         {data_fingerprint, n_features, has_partition, rounds_done}
  master -> worker  partition(+chunk) indices de SU particion (ver distribuidor.enviar_particion)
  worker -> master  partition_ack    {n, hist}
  master -> worker  round_start      {round, version, W, b, local_epochs, batch_size, lr, l2, seed}
  worker -> master  update           {round, version, n, W, b, metrics, train_time_s}
  master -> worker  sync             {round, version, phase, detail}   (tras (re)registrarse)
  master -> worker  fin              {reason}

Politica de rondas (ver README, "Politica de participacion y timeouts"):
  - Cada intento de ronda tiene un `round` unico y creciente. Una actualizacion con otro
    `round`/`version`, de un worker no esperado o duplicada, se rechaza y se registra.
  - La ronda se cierra cuando responden todos los esperados o vence `round_timeout`.
  - Si hay >= min_clients actualizaciones validas se agrega (FedAvg sobre esos clientes).
    Si no, la ronda FALLA: el modelo global no cambia, se espera `retry_interval` y se
    repite hasta `round_retries` fallos consecutivos; despues se aborta de forma controlada.
  - Un worker que se (re)conecta solo entra a la SIGUIENTE ronda que se abra.
"""
import csv
import logging
import os
import queue
import time

import numpy as np

from common import config
from common.aggregation import UpdateInvalido
from common.model import LogisticModel
from master_edge.aggregator import RondaAgregador
from master_edge.distribuidor import enviar_particion
from master_edge.server import MasterServer

log = logging.getLogger("master")

CAMPOS_GLOBALES = [
    "round", "version", "expected", "responded", "excluded", "aggregated", "n_aggregated",
    "round_timeout_s", "duration_s", "wait_s", "agg_s", "eval_s",
    "val_loss", "val_accuracy", "test_loss", "test_accuracy", "test_precision_macro",
    "test_recall_macro", "test_f1_macro", "failed_reason", "elapsed_s",
]
CAMPOS_CLIENTES = [
    "round", "client", "status", "model_version", "n_samples", "fedavg_weight",
    "local_train_loss", "local_loss_before", "local_acc_before", "local_loss_after",
    "local_acc_after", "train_time_s", "roundtrip_s", "comm_s", "detail",
]
CAMPOS_EVENTOS = ["ts", "elapsed_s", "round", "version", "event", "client", "detail"]


class _Csv:
    def __init__(self, ruta, campos):
        os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
        self.f = open(ruta, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=campos, extrasaction="ignore")
        self.w.writeheader()
        self.f.flush()

    def fila(self, **d):
        self.w.writerow(d)
        self.f.flush()

    def cerrar(self):
        self.f.close()


class Coordinador:
    def __init__(self, cfg, dataset_eval, manifiesto, out_dir, keydir=config.KEYDIR,
                 port=config.PORT, huella_datos=None):
        """
        dataset_eval: (Xval, yval, Xtest, ytest). El master NUNCA recibe datos de entrenamiento.
        manifiesto:   salida de particion.generar_particion (indices + histogramas por cliente).
        """
        self.cfg = cfg
        self.Xval, self.yval, self.Xte, self.yte = dataset_eval
        self.n_features = self.Xte.shape[1]
        self.manifiesto = manifiesto
        self.out = out_dir
        self.huella = huella_datos
        self.keydir, self.port = keydir, port
        self.q = queue.Queue()
        self.srv = None
        self.modelo = LogisticModel(self.n_features)
        self.version = 0
        self.round_id = 0
        self.listos = set()          # conectados + registrados + con particion confirmada
        self.posicion = {}           # worker -> posicion de su particion (orden alfabetico)
        self.vistos = set()          # workers que ya se registraron alguna vez
        self.desconectado_desde = {}  # worker -> instante en que se perdio
        self.fallos_seguidos = {}    # worker -> (ronda inicial perdida, instante)
        self.fase = "arranque"
        self.repartido = False
        self.t0 = time.time()
        self.abortado = None
        os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
        self.csv_glob = _Csv(os.path.join(out_dir, "global_metrics.csv"), CAMPOS_GLOBALES)
        self.csv_cli = _Csv(os.path.join(out_dir, "client_metrics.csv"), CAMPOS_CLIENTES)
        self.csv_ev = _Csv(os.path.join(out_dir, "events.csv"), CAMPOS_EVENTOS)
        self.filas_globales = []

    # ================= utilidades =================
    def _t(self):
        return time.time() - self.t0

    def evento(self, event, client="", detail="", round_id=None):
        rid = self.round_id if round_id is None else round_id
        self.csv_ev.fila(ts=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_s=round(self._t(), 3),
                         round=rid, version=self.version, event=event, client=client, detail=detail)
        log.info("[evento] %-22s ronda=%s %s %s", event, rid, client, detail)

    def _on_msg(self, w, msg):
        self.q.put(("msg", w, msg))

    def _on_connect(self, w):
        self.q.put(("connect", w, None))

    def _on_disconnect(self, w):
        self.q.put(("disconnect", w, None))

    def start(self):
        self.srv = MasterServer(keydir=self.keydir, port=self.port, on_message=self._on_msg,
                                on_connect=self._on_connect, on_disconnect=self._on_disconnect)
        self.srv.start()

    def stop(self):
        if self.srv:
            self.srv.stop()
        for c in (self.csv_glob, self.csv_cli, self.csv_ev):
            c.cerrar()

    def _enviar(self, w, msg):
        """True si se envio. Un fallo de red se trata como desconexion del worker."""
        try:
            self.srv.send(w, msg)
            return True
        except (OSError, ConnectionError, ValueError) as e:
            self.evento("send_failed", w, str(e))
            self.listos.discard(w)
            return False

    # ================= manejo de eventos de red =================
    def _procesar(self, ev):
        """Eventos que pueden llegar en cualquier fase. Devuelve el evento si es un 'update'."""
        kind, w, msg = ev
        if kind == "connect":
            self.evento("connected", w, "sesion segura establecida (pendiente de register)")
        elif kind == "disconnect":
            self.listos.discard(w)
            self.desconectado_desde[w] = time.time()
            self.evento("disconnected", w, "sesion cerrada o par caido (keepalive)")
        elif kind == "msg":
            tipo = msg.get("type")
            if tipo == "register":
                self._on_register(w, msg)
            elif tipo == "partition_ack":
                self._on_ack(w, msg)
            elif tipo == "update":
                return ev
            else:
                self.evento("unexpected_message", w, f"tipo {tipo!r}")
        return None

    def _procesar_fuera(self, ev):
        """Como _procesar, para fases sin ronda abierta: cualquier 'update' llega por definicion tarde."""
        r = self._procesar(ev)
        if r is not None:
            self.evento("stale_update_rejected", r[1], f"actualizacion de la ronda {r[2].get('round')} "
                        f"recibida con la ronda cerrada (fase {self.fase})")

    def _on_register(self, w, msg):
        if self.huella and msg.get("data_fingerprint") != self.huella:
            self.evento("register_rejected", w, "el dataset/preprocesamiento local NO coincide con el del master")
            self._enviar(w, {"type": "fin", "reason": "data_fingerprint_mismatch"})
            return
        if msg.get("n_features") != self.n_features:
            self.evento("register_rejected", w, f"n_features {msg.get('n_features')} != {self.n_features}")
            return
        reconexion = w in self.vistos
        self.vistos.add(w)
        down = time.time() - self.desconectado_desde.pop(w, time.time())
        self.evento("reconnected" if reconexion else "registered", w,
                    (f"tras {down:.1f} s sin sesion; " if reconexion else "")
                    + f"has_partition={msg.get('has_partition')} rounds_done={msg.get('rounds_done')}")
        if w not in self.posicion and self.repartido:    # llegada tardia: toma una particion libre
            libres = [i for i in range(self.cfg.n_clients) if i not in self.posicion.values()]
            if libres:
                self.posicion[w] = libres[0]
                self.evento("late_join", w, f"se le asigna la particion {libres[0]}")
                msg = {**msg, "has_partition": False}
            else:
                self.evento("register_rejected", w, "no quedan particiones libres")
        if w in self.posicion:                       # ya hay reparto hecho
            if msg.get("has_partition"):
                self.listos.add(w)
            else:
                self._enviar_particion(w)
            self._enviar(w, {"type": "sync", "round": self.round_id, "version": self.version,
                             "phase": self.fase, "detail": "participaras desde la proxima ronda que se abra"})
            self.evento("synced", w, f"modelo global v{self.version}, fase {self.fase}")

    def _on_ack(self, w, msg):
        pos = self.posicion.get(w)
        if pos is None:
            return
        esperado = self.manifiesto["clients"][pos]
        if msg.get("n") == esperado["n"] and msg.get("hist") == esperado["hist"]:
            self.listos.add(w)
            self.evento("partition_confirmed", w, f"posicion {pos}, n={esperado['n']}")
        else:
            self.evento("partition_mismatch", w, "el histograma confirmado no coincide con el enviado")

    def _enviar_particion(self, w):
        pos = self.posicion[w]
        enviar_particion(self.srv, w, self.manifiesto, self.manifiesto["clients"][pos], pos)
        self.evento("partition_sent", w, f"posicion {pos}, n={self.manifiesto['clients'][pos]['n']}")

    # ================= fase 1: conexion y reparto =================
    def esperar_workers(self):
        fin = time.time() + self.cfg.connect_timeout
        while len(self.vistos) < self.cfg.n_clients and time.time() < fin:
            try:
                self._procesar_fuera(self.q.get(timeout=0.5))
            except queue.Empty:
                pass
        if len(self.vistos) < self.cfg.n_clients and self.cfg.allow_partial_start                 and len(self.vistos) >= self.cfg.min_clients:
            self.evento("partial_start", detail=f"arranca con {len(self.vistos)} de {self.cfg.n_clients} "
                        f"workers {sorted(self.vistos)}; los demas podran incorporarse en rondas posteriores")
            return
        if len(self.vistos) < self.cfg.n_clients:
            raise TimeoutError(f"solo {len(self.vistos)} de {self.cfg.n_clients} workers se registraron "
                               f"en {self.cfg.connect_timeout:.0f} s: {sorted(self.vistos)}")

    def repartir(self):
        ids = sorted(self.vistos)[:self.cfg.n_clients]
        self.posicion = {w: i for i, w in enumerate(ids)}
        self.listos.clear()
        for w in ids:
            self._enviar_particion(w)
        fin = time.time() + self.cfg.partition_timeout
        while len(self.listos) < len(ids) and time.time() < fin:
            try:
                self._procesar_fuera(self.q.get(timeout=0.5))
            except queue.Empty:
                pass
        if len(self.listos) < len(ids):
            raise TimeoutError(f"particion sin confirmar por {sorted(set(ids) - self.listos)}")
        self.repartido = True

    # ================= evaluacion y checkpoints =================
    def evaluar(self):
        v = self.modelo.evaluar(self.Xval, self.yval, self.cfg.l2)
        t = self.modelo.evaluar(self.Xte, self.yte, self.cfg.l2)
        return v, t

    def checkpoint(self):
        ruta = os.path.join(self.out, "checkpoints", f"global_v{self.version:03d}.npz")
        np.savez(ruta, W=self.modelo.W, b=self.modelo.b, version=self.version)
        np.savez(os.path.join(self.out, "checkpoints", "latest.npz"), W=self.modelo.W,
                 b=self.modelo.b, version=self.version)

    # ================= una ronda =================
    def _fila_cliente(self, rid, w, status, agg=None, pesos=None, t_envio=None, t_resp=None, detail=""):
        fila = dict(round=rid, client=w, status=status, detail=detail, model_version=self.version)
        if agg is not None and w in agg.recibidos:
            _, _, n, m = agg.recibidos[w]
            mm = m.get("metrics", {})
            rt = (t_resp - t_envio) if (t_resp and t_envio) else None
            fila.update(n_samples=n, fedavg_weight=(pesos or {}).get(w), local_train_loss=mm.get("train_loss"),
                        local_loss_before=mm.get("loss_before"), local_acc_before=mm.get("acc_before"),
                        local_loss_after=mm.get("loss_after"), local_acc_after=mm.get("acc_after"),
                        train_time_s=m.get("train_time_s"), roundtrip_s=rt,
                        comm_s=(rt - m["train_time_s"]) if (rt is not None and m.get("train_time_s") is not None) else None)
        self.csv_cli.fila(**fila)

    def _esperar_quorum_inicial(self, deadline):
        """Antes de abrir la ronda: si hay menos de min_clients listos, espera reconexiones."""
        while len(self._esperados()) < self.cfg.min_clients and time.time() < deadline:
            try:
                self._procesar_fuera(self.q.get(timeout=0.5))
            except queue.Empty:
                pass

    def _esperados(self):
        return sorted(w for w in self.srv.workers() if w in self.listos)

    def ejecutar_ronda(self):
        """Un intento de ronda. Devuelve la fila de metricas globales (dict)."""
        cfg = self.cfg
        self.round_id += 1
        rid, t0 = self.round_id, time.time()
        self.fase = "ronda_abierta"
        self._esperar_quorum_inicial(t0 + cfg.round_timeout)
        esperados = self._esperados()
        fila = dict(round=rid, version=self.version, expected=len(esperados), round_timeout_s=cfg.round_timeout,
                    responded=0, excluded=0, aggregated=0, n_aggregated=0, failed_reason="")
        conocidos = sorted(self.posicion)

        if len(esperados) < cfg.min_clients:
            return self._cerrar_ronda_fallida(rid, t0, fila, conocidos, None, {}, {},
                                              f"solo {len(esperados)} workers listos (< min_clients={cfg.min_clients})")

        agg = RondaAgregador(rid, self.version, esperados, self.n_features)
        base = {"type": "round_start", "round": rid, "version": self.version, "W": self.modelo.W,
                "b": self.modelo.b, "local_epochs": cfg.local_epochs, "batch_size": cfg.batch_size,
                "lr": cfg.lr, "l2": cfg.l2}
        t_envio, t_resp, estado = {}, {}, {}
        self.evento("round_start", detail=f"esperados={esperados} timeout={cfg.round_timeout}s", round_id=rid)
        for w in esperados:
            msg = {**base, "seed": cfg.seed + 1000 * rid + self.posicion[w]}
            t_envio[w] = time.time()
            if not self._enviar(w, msg):
                agg.esperados.discard(w)
                estado[w] = ("send_failed", "no se pudo enviar el modelo global")

        deadline = t0 + cfg.round_timeout
        while agg.pendientes() and time.time() < deadline:
            try:
                ev = self.q.get(timeout=min(0.5, max(0.01, deadline - time.time())))
            except queue.Empty:
                continue
            kind, w, msg = ev
            if kind == "disconnect" and w in agg.pendientes():
                agg.esperados.discard(w)
                estado[w] = ("disconnected", "sesion perdida durante la ronda")
            r = self._procesar(ev)
            if r is None:
                continue
            _, w, msg = r
            try:
                agg.aceptar(w, msg)
                t_resp[w] = time.time()
            except UpdateInvalido as e:
                tarde = msg.get("round") is not None and msg.get("round") < rid
                self.evento("stale_update_rejected" if tarde else "invalid_update_rejected", w, str(e))
                if w in agg.pendientes() and not tarde:
                    estado[w] = ("invalid", str(e))
                    agg.esperados.discard(w)
        agg.cerrar()
        for w in sorted(agg.pendientes()):
            estado[w] = ("timeout", f"sin respuesta tras {cfg.round_timeout:.0f} s")
            self.evento("client_timeout", w, f"no respondio en {cfg.round_timeout:.0f} s: se excluye de la ronda", round_id=rid)
        wait_s = time.time() - t0

        validos = len(agg.recibidos)
        if validos < cfg.min_clients:
            return self._cerrar_ronda_fallida(rid, t0, fila, conocidos, agg, t_envio, t_resp,
                                              f"{validos} actualizaciones validas (< min_clients={cfg.min_clients})",
                                              estado=estado, wait_s=wait_s)
        ta = time.perf_counter()
        W, b, pesos = agg.agregar()
        self.modelo.set_params(W, b)
        self.version += 1
        agg_s = time.perf_counter() - ta
        self.checkpoint()
        self.fase = "ronda_cerrada"
        self._registrar_participacion(rid, conocidos, agg, estado, pesos, t_envio, t_resp)
        fila.update(version=self.version, responded=validos, excluded=len(conocidos) - validos,
                    aggregated=1, n_aggregated=validos, wait_s=wait_s, agg_s=agg_s)
        self.evento("round_aggregated", detail=f"clientes={sorted(agg.recibidos)} pesos="
                    + str({k: round(v, 3) for k, v in pesos.items()}) + f" -> modelo v{self.version}", round_id=rid)
        return self._finalizar_fila(fila, t0)

    def _registrar_participacion(self, rid, conocidos, agg, estado, pesos, t_envio, t_resp):
        for w in conocidos:
            if w in agg.recibidos:
                self._fila_cliente(rid, w, "ok", agg, pesos, t_envio.get(w), t_resp.get(w))
                perdida = self.fallos_seguidos.pop(w, None)
                if perdida:
                    self.evento("participation_recovered", w,
                                f"vuelve a participar en la ronda {rid}; ausente desde la ronda {perdida[0]} "
                                f"({time.time() - perdida[1]:.1f} s)", round_id=rid)
            else:
                st, det = estado.get(w, ("absent", "no estaba conectado/listo al abrir la ronda"))
                self._fila_cliente(rid, w, st, detail=det)
                self.fallos_seguidos.setdefault(w, (rid, time.time()))

    def _cerrar_ronda_fallida(self, rid, t0, fila, conocidos, agg, t_envio, t_resp, motivo, estado=None,
                              wait_s=None):
        self.fase = "ronda_fallida"
        estado = estado or {}
        vacio = agg or RondaAgregador(rid, self.version, [], self.n_features)
        self._registrar_participacion(rid, conocidos, vacio, estado, {}, t_envio, t_resp)
        fila.update(responded=len(vacio.recibidos), excluded=len(conocidos) - len(vacio.recibidos),
                    failed_reason=motivo, wait_s=wait_s if wait_s is not None else time.time() - t0)
        self.evento("round_failed", detail=motivo + "; el modelo global NO cambia", round_id=rid)
        return self._finalizar_fila(fila, t0)

    def _finalizar_fila(self, fila, t0):
        te = time.perf_counter()
        v, t = self.evaluar()
        fila.update(eval_s=time.perf_counter() - te, duration_s=time.time() - t0, elapsed_s=self._t(),
                    val_loss=v["loss"], val_accuracy=v["accuracy"], test_loss=t["loss"],
                    test_accuracy=t["accuracy"], test_precision_macro=t["precision_macro"],
                    test_recall_macro=t["recall_macro"], test_f1_macro=t["f1_macro"])
        fila["confusion"] = t["confusion"]
        self.csv_glob.fila(**fila)
        self.filas_globales.append(fila)
        log.info("ronda %d: v%d | agregada=%d (%d/%d) | test acc %.4f loss %.4f | %.1f s", fila["round"],
                 fila["version"], fila["aggregated"], fila["responded"], fila["expected"],
                 fila["test_accuracy"], fila["test_loss"], fila["duration_s"])
        return fila

    def _pausa(self, segundos):
        """Espera atendiendo eventos de red (registros, reconexiones, actualizaciones tardias)."""
        fin = time.time() + segundos
        while time.time() < fin:
            try:
                self._procesar_fuera(self.q.get(timeout=min(0.2, max(0.01, fin - time.time()))))
            except queue.Empty:
                pass

    # ================= experimento completo =================
    def ejecutar(self):
        """Conexion, reparto y rondas hasta `rounds` agregadas o aborto controlado."""
        cfg = self.cfg
        try:
            self.esperar_workers()
            self.repartir()
            self.evento("experiment_start", detail=f"{cfg.scheme}, {cfg.rounds} rondas, min_clients={cfg.min_clients}")
            hechas, fallos_seguidos = 0, 0
            while hechas < cfg.rounds:
                fila = self.ejecutar_ronda()
                if fila["aggregated"]:
                    hechas += 1
                    fallos_seguidos = 0
                    self._pausa(cfg.round_delay)
                    continue
                fallos_seguidos += 1
                if fallos_seguidos > cfg.round_retries:
                    self.abortado = (f"{fallos_seguidos} rondas fallidas seguidas (round_retries={cfg.round_retries}): "
                                     f"{fila['failed_reason']}")
                    self.evento("experiment_aborted", detail=self.abortado)
                    break
                self.evento("round_retry", detail=f"reintento {fallos_seguidos}/{cfg.round_retries} en "
                            f"{cfg.retry_interval:.1f} s")
                self._pausa(cfg.retry_interval)
        finally:
            self.fase = "finalizado"
            self.evento("experiment_end", detail=self.abortado or "completado")
            for w in self.srv.workers():
                self._enviar(w, {"type": "fin", "reason": self.abortado or "completed"})
            time.sleep(0.3)
        return self.resumen()

    def resumen(self):
        fil = self.filas_globales
        agregadas = [f for f in fil if f["aggregated"]]
        ult = fil[-1] if fil else {}
        return {
            "scheme": self.cfg.scheme, "run": self.cfg.nombre, "rounds_requested": self.cfg.rounds,
            "rounds_completed": len(agregadas), "rounds_failed": len(fil) - len(agregadas),
            "attempts": len(fil), "aborted": self.abortado,
            "final_model_version": self.version,
            "final_test_accuracy": ult.get("test_accuracy"), "final_test_loss": ult.get("test_loss"),
            "final_test_precision_macro": ult.get("test_precision_macro"),
            "final_test_recall_macro": ult.get("test_recall_macro"),
            "final_test_f1_macro": ult.get("test_f1_macro"),
            "final_confusion_matrix": ult.get("confusion"),
            "total_time_s": round(self._t(), 3),
            "mean_round_time_s": (float(np.mean([f["duration_s"] for f in fil])) if fil else None),
            "metric_definitions": "precision/recall/F1: promedio macro; metricas globales sobre el test independiente",
        }
