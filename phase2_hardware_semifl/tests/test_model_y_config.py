import socket
import threading

import numpy as np
import pytest

from common import config, fl_config, metrics
from common import protocol as P
from common.model import LogisticModel
from worker_node.logistic_model import EntrenadorLocal


def test_metricas_con_valores_conocidos():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    m = metrics.clasificacion(y, p)
    assert m["confusion"] == [[3, 1], [2, 2]]
    assert m["accuracy"] == pytest.approx(5 / 8)
    # clase 0: P=3/5 R=3/4 ; clase 1: P=2/3 R=2/4  (promedio macro)
    assert m["precision_macro"] == pytest.approx((3 / 5 + 2 / 3) / 2)
    assert m["recall_macro"] == pytest.approx((3 / 4 + 2 / 4) / 2)
    f0, f1 = 2 * .6 * .75 / (.6 + .75), 2 * (2 / 3) * .5 / ((2 / 3) + .5)
    assert m["f1_macro"] == pytest.approx((f0 + f1) / 2)


def test_metricas_clase_ausente_no_da_nan():
    m = metrics.clasificacion(np.array([0, 0, 0]), np.array([0, 0, 0]))
    assert m["accuracy"] == 1.0 and np.isfinite(m["f1_macro"])


def test_entrenamiento_local_reduce_la_perdida(ds):
    X, y = ds.subset(ds.train_idx[:2000])
    e = EntrenadorLocal(X, y)
    W0, b0 = np.zeros(ds.n_features), 0.0
    r = e.ronda(W0, b0, epochs=3, batch_size=64, lr=0.1, l2=1e-4, seed=1)
    assert r["metrics"]["loss_after"] < r["metrics"]["loss_before"]
    assert r["metrics"]["acc_after"] > 0.7 and r["n"] == 2000 and np.all(np.isfinite(r["W"]))
    r2 = EntrenadorLocal(X, y).ronda(W0, b0, 3, 64, 0.1, 1e-4, seed=1)
    assert np.allclose(r["W"], r2["W"])                         # determinista con la misma semilla


def test_modelo_valida_parametros_recibidos():
    m = LogisticModel(3)
    with pytest.raises(ValueError, match="forma"):
        m.set_params([1, 2], 0.0)
    with pytest.raises(ValueError, match="finitos"):
        m.set_params([1, 2, np.nan], 0.0)
    with pytest.raises(ValueError):
        m.entrenar(np.zeros((0, 3)), np.zeros(0), 1, 8, 0.1)


# ---- serializacion y errores de comunicacion ----
def test_serializacion_ida_y_vuelta_con_numpy_sin_perdida():
    W = np.random.default_rng(0).normal(size=92)
    msg = P.decodificar(P.codificar({"type": "update", "W": W, "b": np.float64(0.1) / 3, "n": 7}))
    assert np.array_equal(np.array(msg["W"]), W) and msg["b"] == 0.1 / 3


def test_mensaje_de_un_modelo_cabe_en_el_limite():
    msg = {"type": "update", "round": 10, "version": 9, "n": 5699, "W": np.random.default_rng(1).normal(size=92),
           "b": 0.2, "metrics": {str(k): 0.123456789012345 for k in range(8)}, "train_time_s": 0.1}
    assert len(P.codificar(msg)) < config.MAX_MSG / 4


def test_codificar_rechaza_mensajes_invalidos_y_decodificar_tolera_basura():
    with pytest.raises(ValueError):
        P.codificar({"sin_type": 1})
    with pytest.raises(ValueError):
        P.codificar("texto")
    assert P.decodificar(b"no es json")["type"] == "text"
    assert P.decodificar(b"[1,2]")["type"] == "json"


def test_sesion_rechaza_mensaje_mayor_al_maximo():
    class S(P.Sesion):
        def __init__(self):
            self.lock_tx = threading.Lock()
    with pytest.raises(ValueError, match="maximo"):
        S().enviar(b"x" * (config.MAX_MSG + 1))


def test_recv_exact_detecta_conexion_cerrada():
    a, b = socket.socketpair()
    a.sendall(b"abc")
    a.close()
    with pytest.raises(ConnectionError):
        P.recv_exact(b, 10)                                      # mensaje incompleto
    b.close()


# ---- configuracion ----
@pytest.mark.parametrize("campos", [
    {"min_clients": 0}, {"min_clients": 5}, {"rounds": 0}, {"lr": 0}, {"scheme": "x"}, {"alpha": -1},
    {"round_timeout": 0}, {"local_epochs": 0}, {"val_frac": 0.6, "test_frac": 0.5}, {"optimizer": "adam"},
])
def test_configuracion_invalida(campos):
    with pytest.raises(ValueError, match="invalida"):
        fl_config.cargar(**campos)


def test_configuracion_env_json_y_cli(tmp_path, monkeypatch):
    j = tmp_path / "c.json"
    j.write_text('{"rounds": 7, "lr": 0.5}')
    monkeypatch.setenv("SFL_ROUNDS", "9")
    c = fl_config.cargar(str(j), lr=0.2)
    assert (c.rounds, c.lr) == (9, 0.2)                          # env > json ; cli > json
    j.write_text('{"inventado": 1}')
    with pytest.raises(ValueError, match="desconocidos"):
        fl_config.cargar(str(j))
