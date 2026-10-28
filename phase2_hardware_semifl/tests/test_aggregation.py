import numpy as np
import pytest

from common import aggregation
from common.aggregation import UpdateInvalido
from master_edge.aggregator import RondaAgregador, RondaCerrada


def test_fedavg_valores_conocidos():
    # n = 100, 300 -> pesos 0.25 y 0.75
    W, b, pesos = aggregation.fedavg([([1.0, 2.0], 1.0, 100), ([5.0, 10.0], 5.0, 300)])
    assert pesos == pytest.approx([0.25, 0.75])
    assert W == pytest.approx([0.25 * 1 + 0.75 * 5, 0.25 * 2 + 0.75 * 10])
    assert b == pytest.approx(4.0)


def test_fedavg_iguales_n_es_media_simple():
    W, b, _ = aggregation.fedavg([([0.0, 4.0], 2.0, 10), ([2.0, 0.0], 0.0, 10)])
    assert W == pytest.approx([1.0, 2.0]) and b == pytest.approx(1.0)


def test_fedavg_un_solo_cliente_devuelve_su_modelo():
    W, b, pesos = aggregation.fedavg([([1.5, -2.0], 0.3, 7)])
    assert W == pytest.approx([1.5, -2.0]) and b == pytest.approx(0.3) and pesos == [1.0]


def test_pesos_proporcionales_y_suman_uno():
    ns = [5699, 6112, 601, 1541]
    p = aggregation.pesos_fedavg(ns)
    assert sum(p) == pytest.approx(1.0)
    assert p == pytest.approx([n / sum(ns) for n in ns])


@pytest.mark.parametrize("W,b", [([1.0, np.nan], 0.0), ([1.0, np.inf], 0.0), ([1.0, 2.0], float("nan"))])
def test_rechaza_no_finitos(W, b):
    with pytest.raises(UpdateInvalido, match="finitos"):
        aggregation.fedavg([([0.0, 0.0], 0.0, 1), (W, b, 1)])


def test_rechaza_dimension_incompatible_y_n_invalido():
    with pytest.raises(UpdateInvalido, match="dimension"):
        aggregation.fedavg([([0.0, 0.0], 0.0, 1), ([0.0, 0.0, 0.0], 0.0, 1)])
    for n in (0, -3, 2.5, True, None):
        with pytest.raises(UpdateInvalido):
            aggregation.validar_n(n)
    with pytest.raises(ValueError):
        aggregation.fedavg([])


# ---- agregador de ronda: ronda, version, duplicados, tardias ----
def _msg(r=3, v=2, n=10, W=(1.0, 2.0), b=0.5):
    return {"type": "update", "round": r, "version": v, "n": n, "W": list(W), "b": b}


def _agg(esperados=("A", "B", "C")):
    return RondaAgregador(3, 2, esperados, 2)


def test_acepta_y_agrega_solo_los_recibidos():
    a = _agg()
    a.aceptar("A", _msg(n=100, W=(1, 1)))
    a.aceptar("B", _msg(n=300, W=(5, 5)))
    assert a.pendientes() == {"C"} and not a.completa()
    W, b, pesos = a.agregar()
    assert pesos == pytest.approx({"A": 0.25, "B": 0.75}) and W == pytest.approx([4.0, 4.0])


@pytest.mark.parametrize("msg,fragmento", [
    (_msg(r=2), "ronda"), (_msg(v=1), "version"), (_msg(W=(1, float("inf"))), "finitos"),
    (_msg(W=(1, 2, 3)), "dimension"), (_msg(n=0), "n de muestras"),
])
def test_rechaza_ronda_version_y_valores_invalidos(msg, fragmento):
    with pytest.raises(UpdateInvalido, match=fragmento):
        _agg().aceptar("A", msg)


def test_rechaza_duplicado_ajeno_y_ronda_cerrada():
    a = _agg()
    a.aceptar("A", _msg())
    with pytest.raises(UpdateInvalido, match="ya envio"):
        a.aceptar("A", _msg())
    with pytest.raises(UpdateInvalido, match="no participa"):
        a.aceptar("Z", _msg())
    a.cerrar()
    with pytest.raises(RondaCerrada):
        a.aceptar("B", _msg())                    # actualizacion tardia
    assert "B" not in a.recibidos
