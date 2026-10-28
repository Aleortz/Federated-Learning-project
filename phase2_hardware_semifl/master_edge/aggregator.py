"""
Agregador de una ronda: valida cada actualizacion local y la cierra con FedAvg.

Una instancia = un intento de ronda (identificado por `round_id`, unico y creciente). Una
vez cerrada (`cerrar()`), rechaza cualquier actualizacion tardia: asi un cliente lento no
puede colar pesos de una ronda ya agregada.

Contrato de la actualizacion (mensaje "update", worker -> master):
    {"type": "update", "round": int, "version": int, "n": int, "W": [...], "b": float,
     "metrics": {...}, "train_time_s": float}
El ID del cliente NO viene en el mensaje: es el autenticado en el handshake.
"""
from common import aggregation
from common.aggregation import UpdateInvalido


class RondaCerrada(UpdateInvalido):
    pass


class RondaAgregador:
    def __init__(self, round_id, version, esperados, n_features):
        self.round_id = round_id
        self.version = version
        self.esperados = set(esperados)
        self.n_features = n_features
        self.recibidos = {}            # worker_id -> (W, b, n, msg)
        self.cerrada = False

    def aceptar(self, worker_id, msg):
        """Valida y guarda la actualizacion. Lanza UpdateInvalido si no se puede aceptar."""
        if self.cerrada:
            raise RondaCerrada(f"ronda {self.round_id} ya cerrada")
        if msg.get("round") != self.round_id:
            raise UpdateInvalido(f"ronda {msg.get('round')!r} distinta de la esperada ({self.round_id})")
        if msg.get("version") != self.version:
            raise UpdateInvalido(f"version de modelo {msg.get('version')!r} distinta de {self.version}")
        if worker_id not in self.esperados:
            raise UpdateInvalido(f"{worker_id} no participa en esta ronda")
        if worker_id in self.recibidos:
            raise UpdateInvalido(f"{worker_id} ya envio su actualizacion en esta ronda")
        n = aggregation.validar_n(msg.get("n"))
        W, b = aggregation.validar_parametros(msg.get("W"), msg.get("b"), self.n_features)
        self.recibidos[worker_id] = (W, b, n, msg)

    def pendientes(self):
        return self.esperados - set(self.recibidos)

    def completa(self):
        return not self.pendientes()

    def cerrar(self):
        self.cerrada = True

    def agregar(self):
        """FedAvg de las actualizaciones validas. Devuelve (W, b, {worker_id: peso})."""
        ids = sorted(self.recibidos)
        W, b, pesos = aggregation.fedavg([self.recibidos[w][:3] for w in ids])
        return W, b, dict(zip(ids, pesos))
