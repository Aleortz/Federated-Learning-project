"""
Configuracion del experimento federado (lo que se comparte entre master y workers).

Prioridad de valores (de menor a mayor): defaults de este archivo < archivo JSON
(--config) < variables de entorno SFL_* < argumentos de linea de comandos.

La configuracion de red/seguridad (IP, puerto, carpeta de claves) vive en
common/config.py y no se duplica aqui.
"""
import argparse
import dataclasses
import json
import os
from dataclasses import dataclass, field

ESQUEMAS = ("iid", "noniid")


@dataclass
class FLConfig:
    # ---- federacion ----
    n_clients: int = 4                 # workers esperados
    rounds: int = 10                   # rondas federadas (no confundir con epocas locales)
    min_clients: int = 2               # actualizaciones validas minimas para agregar una ronda
    allow_partial_start: bool = False  # True: si vence connect_timeout y hay >= min_clients registrados, arranca con ellos
    # ---- entrenamiento local ----
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.1
    l2: float = 1e-4
    optimizer: str = "sgd"             # solo se implementa SGD con minibatch
    loss: str = "binary_cross_entropy"
    # ---- datos ----
    seed: int = 42
    val_frac: float = 0.10
    test_frac: float = 0.20
    scheme: str = "iid"
    alpha: float = 0.3                 # concentracion Dirichlet (menor = mas heterogeneo)
    min_partition_size: int = 200
    min_js: float = 0.05               # JS media minima exigida a una particion non-IID
    # ---- tiempos (segundos) ----
    round_timeout: float = 30.0        # espera maxima de las actualizaciones de una ronda
    connect_timeout: float = 120.0     # espera maxima de los workers al inicio
    partition_timeout: float = 60.0
    round_retries: int = 1             # reintentos de una ronda sin quorum antes de abortar
    retry_interval: float = 3.0        # espera entre reintentos de ronda
    reconnect_interval: float = 4.0    # reintento de conexion del worker
    round_delay: float = 0.0           # pausa entre rondas (solo para demos; 0 = sin pausa)
    # ---- salida ----
    results_dir: str = "results"
    log_level: str = "INFO"
    run_name: str = ""                 # vacio -> igual al esquema

    def validar(self):
        err = []
        if self.scheme not in ESQUEMAS:
            err.append(f"scheme debe ser uno de {ESQUEMAS}")
        if self.n_clients < 1:
            err.append("n_clients debe ser >= 1")
        if not 1 <= self.min_clients <= self.n_clients:
            err.append("min_clients debe estar entre 1 y n_clients")
        if self.rounds < 1:
            err.append("rounds debe ser >= 1")
        if self.local_epochs < 1 or self.batch_size < 1:
            err.append("local_epochs y batch_size deben ser >= 1")
        if not self.lr > 0:
            err.append("lr debe ser > 0")
        if self.l2 < 0:
            err.append("l2 debe ser >= 0")
        if self.optimizer != "sgd":
            err.append("optimizer: solo 'sgd'")
        if self.alpha <= 0:
            err.append("alpha debe ser > 0")
        if not (0 < self.val_frac < 1 and 0 < self.test_frac < 1 and self.val_frac + self.test_frac < 0.9):
            err.append("val_frac y test_frac deben estar en (0,1) y sumar menos de 0.9")
        for campo in ("round_timeout", "connect_timeout", "partition_timeout", "retry_interval",
                      "reconnect_interval"):
            if not getattr(self, campo) > 0:
                err.append(f"{campo} debe ser > 0")
        if self.round_delay < 0:
            err.append("round_delay debe ser >= 0")
        if self.round_retries < 0:
            err.append("round_retries debe ser >= 0")
        if err:
            raise ValueError("configuracion invalida: " + "; ".join(err))
        return self

    def as_dict(self):
        return dataclasses.asdict(self)

    @property
    def nombre(self):
        return self.run_name or self.scheme


def _convertir(valor, tipo):
    if tipo is bool:
        return str(valor).lower() in ("1", "true", "yes", "si")
    return tipo(valor)


def cargar(ruta_json=None, **overrides):
    """defaults < JSON < variables SFL_<CAMPO> < overrides (los None se ignoran)."""
    base = {}
    if ruta_json:
        with open(ruta_json, encoding="utf-8") as f:
            base = json.load(f)
        desconocidos = set(base) - {f.name for f in dataclasses.fields(FLConfig)}
        if desconocidos:
            raise ValueError(f"{ruta_json}: campos desconocidos {sorted(desconocidos)}")
    campos = {f.name: f for f in dataclasses.fields(FLConfig)}
    for nombre, f in campos.items():
        env = os.environ.get("SFL_" + nombre.upper())
        if env is not None:
            base[nombre] = _convertir(env, type(getattr(FLConfig(), nombre)))
    for k, v in overrides.items():
        if v is not None:
            if k not in campos:
                raise ValueError(f"parametro desconocido '{k}'")
            base[k] = v
    return FLConfig(**base).validar()


def agregar_argumentos(ap):
    """Anade al ArgumentParser un flag --<campo> por cada campo de FLConfig."""
    ap.add_argument("--config", help="archivo JSON con parametros del experimento")
    for f in dataclasses.fields(FLConfig):
        tipo = type(getattr(FLConfig(), f.name))
        flag = "--" + f.name.replace("_", "-")
        if tipo is bool:
            ap.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction, default=None,
                            help=f"(defecto {getattr(FLConfig(), f.name)!r}; --no-... para desactivar)")
        else:
            ap.add_argument(flag, dest=f.name, type=tipo, default=None,
                            help=f"(defecto {getattr(FLConfig(), f.name)!r})")
    return ap


def desde_args(args, **extra):
    campos = {f.name for f in dataclasses.fields(FLConfig)}
    ov = {k: v for k, v in vars(args).items() if k in campos}
    ov.update(extra)
    return cargar(getattr(args, "config", None), **ov)
