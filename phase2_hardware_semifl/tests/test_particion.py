import numpy as np
import pytest

from common import metrics, particion


def _gen(ds, scheme, seed=42, **kw):
    return particion.generar_particion(scheme, ds.y, 4, seed=seed, pool=ds.train_idx,
                                       alpha=kw.pop("alpha", 0.3), **kw)


def _todos(m):
    return np.concatenate([np.asarray(c["indices"]) for c in m["clients"].values()])


@pytest.mark.parametrize("scheme", ["iid", "noniid"])
def test_conserva_muestras_sin_solapamiento_y_solo_train(ds, scheme):
    m = _gen(ds, scheme)
    todos = _todos(m)
    assert len(todos) == len(np.unique(todos)) == len(ds.train_idx)           # sin solapes ni perdidas
    assert set(todos) == set(ds.train_idx.tolist())
    assert not set(todos) & set(ds.val_idx.tolist())                           # val/test fuera de los clientes
    assert not set(todos) & set(ds.test_idx.tolist())
    assert all(c["n"] >= 200 for c in m["clients"].values())


@pytest.mark.parametrize("scheme", ["iid", "noniid"])
def test_reproducible_y_semilla_cambia_el_resultado(ds, scheme):
    a, b, c = _gen(ds, scheme, 1), _gen(ds, scheme, 1), _gen(ds, scheme, 2)
    assert [x["indices"] for x in a["clients"].values()] == [x["indices"] for x in b["clients"].values()]
    assert [x["indices"] for x in a["clients"].values()] != [x["indices"] for x in c["clients"].values()]


def test_iid_es_homogeneo_y_noniid_heterogeneo(ds):
    iid, non = _gen(ds, "iid"), _gen(ds, "noniid", min_js=0.05)
    assert iid["het"]["js_media"] < 0.005
    assert non["het"]["js_media"] >= 0.05
    assert non["het"]["js_media"] > 20 * iid["het"]["js_media"]
    tam = [c["n"] for c in iid["clients"].values()]
    assert max(tam) - min(tam) <= 1


def test_histograma_del_manifiesto_coincide_con_las_etiquetas(ds):
    m = _gen(ds, "noniid")
    for c in m["clients"].values():
        assert c["hist"] == np.bincount(ds.y[c["indices"]], minlength=2).tolist()


def test_noniid_falla_explicitamente_si_no_alcanza_la_heterogeneidad(ds):
    with pytest.raises(RuntimeError, match="intentos"):
        particion.dirichlet_split(ds.train_idx, ds.y, 4, 100.0, np.random.default_rng(0),
                                  min_size=200, max_tries=20, min_js=0.5)


def test_noniid_falla_si_no_hay_tamano_minimo(ds):
    with pytest.raises(RuntimeError):
        particion.generar_particion("noniid", ds.y, 4, alpha=0.001, pool=ds.train_idx, min_size=5000)


def test_esquema_invalido(ds):
    with pytest.raises(ValueError):
        particion.generar_particion("otro", ds.y, 4)


def test_validar_detecta_solape_perdida_y_fuera_de_rango():
    pool = np.arange(10)
    y = np.zeros(10, dtype=int)
    particion.validar_particion([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], pool, y)
    with pytest.raises(ValueError, match="solapamiento"):
        particion.validar_particion([[0, 1, 2, 3, 4], [4, 5, 6, 7, 8, 9]], pool, y)
    with pytest.raises(ValueError, match="pierde"):
        particion.validar_particion([[0, 1, 2], [5, 6, 7, 8, 9]], pool, y)
    with pytest.raises(ValueError, match="fuera de rango"):
        particion.validar_particion([[0, 1, 2, 3, 4], [5, 6, 7, 8, 99]], pool, y)
    with pytest.raises(ValueError, match="solo"):
        particion.validar_particion([[], list(range(10))], pool, y)


def test_guardar_y_cargar_manifiesto(ds, tmp_path):
    m = _gen(ds, "noniid")
    ruta = particion.guardar_manifiesto(m, str(tmp_path / "p.json"))
    m2 = particion.cargar_manifiesto(ruta)
    assert m2["clients"][0]["indices"] == m["clients"][0]["indices"]
    assert sorted(m2["clients"]) == [0, 1, 2, 3]
    het = particion.guardar_estadisticas_csv(m2, str(tmp_path / "stats.csv"))
    assert het["js_media"] == pytest.approx(m["het"]["js_media"])
    assert (tmp_path / "stats.csv").read_text().count("\n") == 5


def test_js_propiedades():
    assert metrics.jensen_shannon([1, 1], [1, 1]) == pytest.approx(0)
    assert metrics.jensen_shannon([1, 0], [0, 1]) == pytest.approx(1)
    assert metrics.jensen_shannon([3, 1], [1, 3]) == pytest.approx(metrics.jensen_shannon([1, 3], [3, 1]))
