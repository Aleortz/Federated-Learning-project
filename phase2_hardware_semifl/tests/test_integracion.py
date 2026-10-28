import os
import tempfile
import threading
import time

import numpy as np
import pytest

from common import config, fl_config, reporting
from common.simulacion import claves_temporales, puerto_libre, simular
from master_edge.coordinador import Coordinador
from master_edge.server import MasterServer
from worker_node.client import WorkerClient
from worker_node.fl_worker import SimulacionCaida
from worker_node.receptor import EnsambladorParticion

pytestmark = pytest.mark.integration

PERMITIDOS = {
    "register": {"type", "data_fingerprint", "n_features", "has_partition", "rounds_done"},
    "partition_ack": {"type", "n", "hist"},
    "update": {"type", "round", "version", "n", "W", "b", "metrics", "train_time_s"},
}


def _cfg(tmp_path, **kw):
    base = dict(rounds=4, round_timeout=5, results_dir=str(tmp_path / "results"), connect_timeout=30,
                partition_timeout=30, log_level="WARNING")
    base.update(kw)
    return fl_config.cargar(**base)


def _csv(out, nombre):
    return reporting.leer_csv(os.path.join(out, nombre))


def _eventos(out):
    return [e["event"] for e in _csv(out, "events.csv")]


@pytest.mark.parametrize("scheme", ["iid", "noniid"])
def test_extremo_a_extremo_cuatro_clientes(tmp_path, monkeypatch, scheme):
    vistos = []
    original = Coordinador._on_msg

    def espia(self, w, msg):
        vistos.append(msg)
        return original(self, w, msg)

    monkeypatch.setattr(Coordinador, "_on_msg", espia)
    hilos_antes = {t.name for t in threading.enumerate()}
    resumen, out = simular(_cfg(tmp_path, scheme=scheme))

    ev = _eventos(out)
    assert ev.count("registered") == 4 and ev.count("partition_confirmed") == 4
    assert resumen["rounds_completed"] == 4 and resumen["rounds_failed"] == 0 and not resumen["aborted"]
    g = _csv(out, "global_metrics.csv")
    assert g[-1]["test_loss"] < 0.693 and g[-1]["test_accuracy"] > 0.75      # mejor que el modelo inicial

    # FedAvg: pesos = n_k / sum(n) y suman 1 en cada ronda
    c = [f for f in _csv(out, "client_metrics.csv") if f["status"] == "ok"]
    for r in {f["round"] for f in c}:
        fs = [f for f in c if f["round"] == r]
        tot = sum(f["n_samples"] for f in fs)
        assert sum(f["fedavg_weight"] for f in fs) == pytest.approx(1.0)
        for f in fs:
            assert f["fedavg_weight"] == pytest.approx(f["n_samples"] / tot)

    # los datos no viajan: solo mensajes y campos permitidos del worker
    assert vistos
    for m in vistos:
        assert set(m) <= PERMITIDOS[m["type"]], m["type"]
        assert len(str(m)) < config.MAX_MSG

    for f in ("config.json", "global_metrics.csv", "client_metrics.csv", "events.csv",
              "partition_statistics.csv", "summary.json", "checkpoints/latest.npz",
              "plots/accuracy_per_round.png", "plots/class_distribution.png", "logs/server.log",
              "logs/WORKER_1.log"):
        assert os.path.exists(os.path.join(out, f)), f
    time.sleep(0.5)
    assert not [t for t in threading.enumerate() if t.name.startswith("WORKER_") and t.name not in hilos_antes
                and t.is_alive()]                                             # cierre controlado


def test_timeout_de_un_cliente_no_bloquea_y_rechaza_actualizacion_tardia(tmp_path):
    t0 = time.time()
    resumen, out = simular(_cfg(tmp_path, round_timeout=1.5),
                           simulaciones={"WORKER_2": SimulacionCaida(2, "silent", 3.0)})
    ev = _eventos(out)
    assert "client_timeout" in ev and "stale_update_rejected" in ev and "participation_recovered" in ev
    assert resumen["rounds_completed"] == 4 and time.time() - t0 < 30
    g = _csv(out, "global_metrics.csv")
    assert min(f["responded"] for f in g) == 3                                # se agrego sin el cliente lento
    assert g[-1]["responded"] == 4                                            # y despues volvio


def test_desconexion_reconexion_sincronizacion_y_reincorporacion(tmp_path):
    cfg = _cfg(tmp_path, rounds=8, round_delay=0.6)
    resumen, out = simular(cfg, simulaciones={"WORKER_3": SimulacionCaida(2, "offline", 2.0)})
    ev = _eventos(out)
    for e in ("disconnected", "reconnected", "synced", "participation_recovered"):
        assert e in ev, e
    estados = [f["status"] for f in _csv(out, "client_metrics.csv") if f["client"] == "WORKER_3"]
    assert "disconnected" in estados and estados[-1] == "ok"
    assert resumen["rounds_completed"] == 8
    versiones = [f["version"] for f in _csv(out, "global_metrics.csv")]
    assert versiones == sorted(versiones)                                      # el modelo global nunca retrocede


def test_ronda_sin_quorum_falla_sin_cambiar_el_modelo_y_se_recupera(tmp_path):
    cfg = _cfg(tmp_path, min_clients=4, round_timeout=1, round_retries=6, retry_interval=0.4)
    resumen, out = simular(cfg, simulaciones={"WORKER_4": SimulacionCaida(2, "offline", 2.5)})
    g = _csv(out, "global_metrics.csv")
    fallidas = [f for f in g if not f["aggregated"]]
    assert fallidas and "round_failed" in _eventos(out)
    v_antes = [f["version"] for f in g if f["round"] == fallidas[0]["round"] - 1]
    assert fallidas[0]["version"] == (v_antes[0] if v_antes else 1)            # modelo intacto en la ronda fallida
    assert resumen["rounds_completed"] == 4 and not resumen["aborted"]


def test_sin_quorum_persistente_aborta_de_forma_controlada(tmp_path):
    cfg = _cfg(tmp_path, min_clients=4, round_timeout=1, round_retries=1, retry_interval=0.2)
    t0 = time.time()
    resumen, out = simular(cfg, simulaciones={"WORKER_4": SimulacionCaida(2, "offline", 60)})
    assert resumen["aborted"] and "experiment_aborted" in _eventos(out)
    assert resumen["rounds_completed"] < 4 and time.time() - t0 < 30           # no se queda esperando


def test_arranque_parcial_y_llegada_tardia(tmp_path):
    cfg = _cfg(tmp_path, rounds=8, round_delay=0.6, connect_timeout=2, allow_partial_start=True)
    resumen, out = simular(cfg, retrasos={"WORKER_4": 4.0})
    ev = _eventos(out)
    assert "partial_start" in ev and "late_join" in ev
    g = _csv(out, "global_metrics.csv")
    assert g[0]["expected"] == 3 and g[-1]["expected"] == 4 and resumen["rounds_completed"] == 8


def test_sin_arranque_parcial_el_servidor_exige_a_todos(tmp_path):
    cfg = _cfg(tmp_path, connect_timeout=2)
    with pytest.raises(TimeoutError, match="3 de 4"):
        simular(cfg, retrasos={"WORKER_4": 30.0})


def test_worker_no_autorizado_es_rechazado():
    with tempfile.TemporaryDirectory() as kd_master, tempfile.TemporaryDirectory() as kd_intruso:
        claves_temporales(kd_master, ["MASTER", "WORKER_1"])
        claves_temporales(kd_intruso, ["INTRUSO"])
        pub_master = open(os.path.join(kd_master, "authorized.txt")).read().splitlines()[0]
        with open(os.path.join(kd_intruso, "authorized.txt"), "a") as f:
            f.write(pub_master + "\n")
        port = puerto_libre()
        srv = MasterServer("MASTER", kd_master, port)
        srv.start()
        w = WorkerClient("INTRUSO", "127.0.0.1", port, "MASTER", kd_intruso, reconnect_s=0.2)
        w.start()
        assert not w.wait_connected(2)
        assert srv.workers() == []
        w.stop()
        srv.stop()


def test_ensamblador_rechaza_particion_corrupta_o_fuera_del_dataset():
    cab = {"type": "partition", "scheme": "iid", "alpha": None, "seed": 1, "client": 0, "n": 3,
           "hist": [2, 1], "class_names": ["a", "b"], "chunks": 1}
    labels = np.array([0, 1, 0, 1])
    e = EnsambladorParticion(labels)
    assert e.agregar(cab) is None
    with pytest.raises(ValueError, match="fuera del dataset"):
        e.agregar({"type": "partition_chunk", "i": 0, "indices": [0, 1, 99]})
    e.agregar(cab)
    with pytest.raises(ValueError, match="corrupta"):
        e.agregar({"type": "partition_chunk", "i": 0, "indices": [0, 1]})
    e.agregar(cab)
    p = e.agregar({"type": "partition_chunk", "i": 0, "indices": [0, 1, 2]})
    assert p["hist"] == [2, 1]                      # recalculado con las etiquetas locales


@pytest.mark.multimachine
@pytest.mark.skip(reason="requiere cinco computadoras fisicas: ver README, seccion 'Prueba fisica'")
def test_cinco_computadoras_fisicas():
    pass
