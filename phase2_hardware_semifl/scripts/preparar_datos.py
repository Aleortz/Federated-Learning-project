"""
Prepara el dataset (verifica data/adult.csv, imprime splits y huella) y, opcionalmente,
genera y guarda la particion IID / non-IID con sus estadisticas y grafico.

    python -m scripts.preparar_datos                       # solo verifica dataset y splits
    python -m scripts.preparar_datos --scheme iid          # + particion IID
    python -m scripts.preparar_datos --scheme noniid --alpha 0.3
    python -m scripts.preparar_datos --both
"""
import argparse
import os
import sys

from common import data, fl_config, particion, reporting
from master_edge.experimento import preparar_particion, ruta_resultados


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fl_config.agregar_argumentos(ap)
    ap.add_argument("--csv", default=data.ADULT_CSV)
    ap.add_argument("--both", action="store_true", help="genera IID y non-IID")
    args = ap.parse_args(argv)
    try:
        cfg = fl_config.desde_args(args)
        ds = data.cargar(args.csv, cfg.val_frac, cfg.test_frac, cfg.seed)
    except (ValueError, data.DatasetError) as e:
        sys.exit(f"[X] {e}")
    print(f"Dataset {args.csv}: {len(ds.y)} filas, {ds.n_features} caracteristicas, huella {ds.huella()}")
    print(f"  train {len(ds.train_idx)} | val {len(ds.val_idx)} | test {len(ds.test_idx)} (estratificados, seed {cfg.seed})")
    esquemas = ["iid", "noniid"] if args.both else ([args.scheme] if args.scheme else [])
    res = ruta_resultados(cfg)
    hists = {}
    for esq in esquemas:
        c = fl_config.cargar(args.config, **{**{k: v for k, v in vars(args).items() if k in cfg.as_dict()}, "scheme": esq})
        m, ruta = preparar_particion(c, ds, res)
        out = os.path.join(res, "partitions")
        het = particion.guardar_estadisticas_csv(m, os.path.join(out, f"{esq}_statistics.csv"))
        h = {f"cliente_{i + 1}": cl["hist"] for i, cl in m["clients"].items()}
        hists[esq] = h
        print(particion.tabla_texto(h, list(data.CLASES), f"Particion {esq}"))
        print(f"  JS media vs global: {het['js_media']:.4f} | archivo: {ruta}")
    if hists:
        png = particion.grafico_comparativo(hists, list(data.CLASES),
                                            os.path.join(res, "partitions", "partition_" + "_".join(hists) + ".png"))
        print("grafico:", png)


if __name__ == "__main__":
    main()
