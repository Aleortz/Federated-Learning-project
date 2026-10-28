"""
Un experimento completo visto desde el master: prepara el dataset de EVALUACION, la
particion, el directorio de resultados, ejecuta el coordinador y genera tablas y graficos.

El master solo conserva las filas de validacion y test (el conjunto de evaluacion global,
independiente de los clientes). Las filas de entrenamiento se descartan al construir el
Dataset y nunca se usan en el bucle federado. La particion se genera como paso de
PREPARACION del experimento (equivale a datos ya repartidos entre los clientes); a cada
worker solo se le envian sus propios INDICES por el canal seguro.
"""
import logging
import os

from common import config, data, particion, reporting
from master_edge.coordinador import Coordinador

log = logging.getLogger("master")

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ruta_resultados(cfg):
    return cfg.results_dir if os.path.isabs(cfg.results_dir) else os.path.join(_BASE, cfg.results_dir)


def preparar_particion(cfg, ds, results_dir):
    """Carga la particion guardada para (esquema, alpha, seed, n) o la genera y la guarda."""
    nombre = (f"{cfg.scheme}_n{cfg.n_clients}_seed{cfg.seed}"
              + (f"_alpha{cfg.alpha}" if cfg.scheme == "noniid" else "") + ".json")
    ruta = os.path.join(results_dir, "partitions", nombre)
    if os.path.exists(ruta):
        m = particion.cargar_manifiesto(ruta)
        pool = ds.train_idx
        parts = [m["clients"][i]["indices"] for i in range(m["n_clients"])]
        particion.validar_particion(parts, pool, ds.y, cfg.min_partition_size)      # rechaza un archivo ajeno al dataset
        log.info("particion cargada de %s", ruta)
        return m, ruta
    m = particion.generar_particion(cfg.scheme, ds.y, cfg.n_clients, alpha=cfg.alpha, seed=cfg.seed,
                                    min_size=cfg.min_partition_size, class_names=data.CLASES,
                                    pool=ds.train_idx, min_js=cfg.min_js if cfg.scheme == "noniid" else 0.0)
    particion.guardar_manifiesto(m, ruta)
    log.info("particion generada (intentos=%s) y guardada en %s", m["intentos"], ruta)
    return m, ruta


def ejecutar_master(cfg, keydir=config.KEYDIR, port=config.PORT, csv=data.ADULT_CSV, tras_arrancar=None,
                    graficos=True):
    """
    Ejecuta el experimento y devuelve (resumen, directorio_de_resultados).
    tras_arrancar(coordinador): hook opcional (lo usa la simulacion local para lanzar los workers).
    """
    cfg.validar()
    results = ruta_resultados(cfg)
    out = reporting.preparar_directorio(results, cfg.nombre)
    h = reporting.configurar_logging(os.path.join(out, "logs", "server.log"), cfg.log_level, "master")
    ds = data.cargar(csv, cfg.val_frac, cfg.test_frac, cfg.seed)
    manifiesto, ruta_part = preparar_particion(cfg, ds, results)
    het = particion.guardar_estadisticas_csv(manifiesto, os.path.join(out, "partition_statistics.csv"))
    print(particion.tabla_texto({f"cliente_{i + 1}": c["hist"] for i, c in manifiesto["clients"].items()},
                                list(data.CLASES), f"Particion {cfg.scheme}"))
    log.info("heterogeneidad (JS vs global): media %.4f, max %.4f", het["js_media"], het["js_max"])
    particion.grafico_comparativo(
        {cfg.scheme: {f"cliente_{i + 1}": c["hist"] for i, c in manifiesto["clients"].items()}},
        list(data.CLASES), os.path.join(out, "plots", "class_distribution.png"))
    huella = ds.huella()
    reporting.escribir_config(out, cfg, data_fingerprint=huella, partition_file=os.path.basename(ruta_part),
                              partition_het=het, partition_attempts=manifiesto.get("intentos"),
                              n_train=len(ds.train_idx), n_val=len(ds.val_idx), n_test=len(ds.test_idx),
                              n_features=ds.n_features, class_names=list(data.CLASES),
                              policy={"min_clients": cfg.min_clients, "round_timeout_s": cfg.round_timeout,
                                      "round_retries": cfg.round_retries})
    evaluacion = (*ds.subset(ds.val_idx), *ds.subset(ds.test_idx))
    del ds                                       # el master no conserva datos de entrenamiento
    coord = Coordinador(cfg, evaluacion, manifiesto, out, keydir, port, huella)
    try:
        coord.start()
        if tras_arrancar:
            tras_arrancar(coord)
        resumen = coord.ejecutar()
    finally:
        coord.stop()
        logging.getLogger("master").removeHandler(h)
    resumen["partition_het"] = het
    reporting.guardar_json(resumen, os.path.join(out, "summary.json"))
    if graficos:
        try:
            reporting.graficos_experimento(out, cfg.nombre)
        except Exception as e:                    # el experimento ya termino: no perder resultados por un grafico
            log.error("no se pudieron generar los graficos: %s", e)
    return resumen, out
