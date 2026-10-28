"""
Particion de datos IID y non-IID entre clientes (workers).

Logica migrada de CHANGE_REPO/fedlab-iid-noniid (scripts/make_partitions.py):
  - IID:     permutacion aleatoria del pool + np.array_split en n_clients trozos.
  - non-IID: reparto Dirichlet por clase (Hsu, Qi y Brown, 2019). Para cada clase se
             sortea una proporcion ~ Dir(alpha) entre los clientes; alpha bajo = sesgo
             fuerte. Se reintenta hasta que todos tengan >= min_size muestras.
  - manifiesto: indices + histograma de clases por cliente.

Solo depende de numpy. Los datos nunca viajan: el master reparte INDICES (ver
master_edge/distribuidor.py) y cada worker los aplica sobre su copia local del dataset.
"""
import csv
import json
import math
import os

import numpy as np

from common import metrics

SEED = 42
ALPHA = 0.05            # 0.05 extremo, 0.1 fuerte, 0.3 moderado
MIN_SIZE = 200          # muestras minimas por cliente en el reparto non-IID
MAX_TRIES = 200

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ADULT_CSV = os.path.join(_BASE, "data", "adult.csv")
CLASES_ADULT = ("<=50K", ">50K")


def cargar_etiquetas_adult(ruta=ADULT_CSV):
    """Columna 'income' de adult.csv -> array de enteros (0: <=50K, 1: >50K)."""
    with open(ruta, newline="") as f:
        lector = csv.DictReader(f)
        y = [1 if fila["income"].strip().startswith(">") else 0 for fila in lector]
    return np.asarray(y, dtype=np.int64)


# =====================================================================
#  Algoritmos de reparto (identicos a CHANGE_REPO)
# =====================================================================
def iid_split(pool, n_clients, rng):
    return [sorted(p.tolist()) for p in np.array_split(rng.permutation(pool), n_clients)]


def dirichlet_split(pool, labels, n_clients, alpha, rng, min_size=MIN_SIZE, max_tries=MAX_TRIES,
                    min_js=0.0):
    """
    Dirichlet por clase. Menor alpha, mayor sesgo. Se reintenta (con el mismo rng, asi que
    es reproducible) hasta que todos los clientes tengan >= min_size muestras Y la
    divergencia JS media respecto de la distribucion global sea >= min_js. Si tras
    max_tries no se cumple, falla de forma explicita.
    """
    n_classes = int(labels.max()) + 1
    mejor_js = 0.0
    for intento in range(max_tries):
        parts = [[] for _ in range(n_clients)]
        for clase in range(n_classes):
            idx_clase = pool[labels[pool] == clase]
            rng.shuffle(idx_clase)
            proporciones = rng.dirichlet(np.repeat(alpha, n_clients))
            cortes = (np.cumsum(proporciones) * len(idx_clase)).astype(int)[:-1]
            for i, trozo in enumerate(np.split(idx_clase, cortes)):
                parts[i].extend(trozo.tolist())
        if min(len(p) for p in parts) < min_size:
            continue
        hists = [np.bincount(labels[np.asarray(p, dtype=np.int64)], minlength=n_classes) for p in parts]
        js = metrics.heterogeneidad(hists)["js_media"]
        mejor_js = max(mejor_js, js)
        if js >= min_js:
            return [sorted(p) for p in parts], intento + 1
    raise RuntimeError(f"no hay un reparto con >= {min_size} muestras por cliente y JS media >= "
                       f"{min_js} tras {max_tries} intentos (mejor JS {mejor_js:.4f}); "
                       f"baja alpha, baja min_js o baja min_size")


def construir_manifiesto(parts, labels, scheme, alpha, seed, class_names):
    n_classes = len(class_names)
    clientes = {}
    for i, indices in enumerate(parts):
        hist = np.bincount(labels[np.asarray(indices, dtype=np.int64)], minlength=n_classes)
        clientes[i] = {
            "n": len(indices),
            "hist": hist.tolist(),
            "pct": [round(100 * h / max(len(indices), 1), 2) for h in hist],
            "indices": indices,
        }
    return {
        "scheme": scheme,
        "alpha": alpha,
        "seed": seed,
        "n_clients": len(parts),
        "total_samples": int(sum(len(p) for p in parts)),
        "class_names": list(class_names),
        "clients": clientes,         # posicion del cliente (0..n-1) -> datos
    }


def generar_particion(scheme, labels, n_clients, alpha=ALPHA, seed=SEED, min_size=MIN_SIZE,
                      class_names=CLASES_ADULT, subset=None, pool=None, min_js=0.0):
    """
    Devuelve el manifiesto de 'iid' o 'noniid'. Determinista para (scheme, seed).
    pool: indices (sobre `labels`) que se reparten; normalmente SOLO el split de
    entrenamiento, de modo que val/test nunca llegan a ningun cliente.
    """
    if scheme not in ("iid", "noniid"):
        raise ValueError("scheme debe ser 'iid' o 'noniid'")
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    pool = np.arange(len(labels)) if pool is None else np.sort(np.asarray(pool, dtype=np.int64))
    if subset and subset < len(labels):
        pool = np.sort(rng.choice(pool, size=subset, replace=False))
    if scheme == "iid":
        parts, alpha_usado, intentos = iid_split(pool, n_clients, rng), None, 1
    else:
        parts, intentos = dirichlet_split(pool, labels, n_clients, alpha, rng, min_size, min_js=min_js)
        alpha_usado = alpha
    validar_particion(parts, pool, labels, min_size=min(min_size, len(pool) // n_clients))
    m = construir_manifiesto(parts, labels, scheme, alpha_usado, seed, class_names)
    m["intentos"] = intentos
    m["het"] = metrics.heterogeneidad([c["hist"] for c in m["clients"].values()])
    return m


def validar_particion(parts, pool, labels, min_size=1):
    """Indices validos, sin solapamientos, sin perdida de muestras y sin particiones vacias."""
    pool = np.asarray(pool, dtype=np.int64)
    todos = np.concatenate([np.asarray(p, dtype=np.int64) for p in parts]) if parts else np.array([], dtype=np.int64)
    if len(todos) and (todos.min() < 0 or todos.max() >= len(labels)):
        raise ValueError("indices fuera de rango en la particion")
    if len(np.unique(todos)) != len(todos):
        raise ValueError("solapamiento: una muestra esta asignada a varios clientes")
    if len(todos) != len(pool) or not np.array_equal(np.sort(todos), np.sort(pool)):
        raise ValueError("la particion pierde o inventa muestras respecto del pool")
    for i, p in enumerate(parts):
        if len(p) < max(min_size, 1):
            raise ValueError(f"el cliente {i} tiene solo {len(p)} muestras")


def guardar_manifiesto(m, ruta):
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(m, f)
    return ruta


def cargar_manifiesto(ruta):
    with open(ruta, encoding="utf-8") as f:
        m = json.load(f)
    m["clients"] = {int(k): v for k, v in m["clients"].items()}     # JSON convierte las claves a str
    return m


def guardar_estadisticas_csv(m, ruta):
    """Una fila por cliente: n, histograma, proporciones y JS respecto de la distribucion global."""
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    het = metrics.heterogeneidad([c["hist"] for c in m["clients"].values()])
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "cliente", "total"] + [f"n_{c}" for c in m["class_names"]]
                   + [f"pct_{c}" for c in m["class_names"]] + ["js_vs_global"])
        for i, c in m["clients"].items():
            w.writerow([m["scheme"], f"cliente_{i + 1}", c["n"], *c["hist"], *c["pct"],
                        round(het["js_por_cliente"][i], 6)])
    return het


# =====================================================================
#  Validacion: tabla en consola y grafico
# =====================================================================
def entropia_clases(pct):
    p = np.asarray(pct) / 100.0
    p = p[p > 0]
    return abs(float(-(p * np.log2(p)).sum()))      # abs: evita "-0.000"


def tabla_texto(hists, class_names, titulo="", nombres=None):
    """hists: {nombre_cliente: [muestras por clase]} -> tabla de texto (muestras y %)."""
    cab = [str(c)[:9] for c in class_names]
    ancho = max(9, *(len(c) for c in cab),
                *(len(f"{x} (100%)") for h in hists.values() for x in h)) + 2
    nombres = nombres or list(hists)
    w0 = max(len(str(n)) for n in nombres) + 2
    lin = []
    if titulo:
        lin.append(titulo)
    lin.append(f"{'cliente':<{w0}}{'total':>8}" + "".join(f"{c:>{ancho}}" for c in cab) + f"{'entropia':>10}")
    lin.append("-" * len(lin[-1]))
    ent = []
    for n in nombres:
        h = hists[n]
        tot = sum(h)
        pct = [100 * x / max(tot, 1) for x in h]
        e = entropia_clases(pct)
        ent.append(e)
        celdas = "".join(f"{f'{x} ({p:.0f}%)':>{ancho}}" for x, p in zip(h, pct))
        lin.append(f"{str(n):<{w0}}{tot:>8}{celdas}{e:>10.3f}")
    lin.append(f"entropia media {np.mean(ent):.3f} de {math.log2(len(class_names)):.3f} bits "
               f"(max = clases perfectamente balanceadas)")
    return "\n".join(lin)


def grafico_comparativo(hists_por_esquema, class_names, ruta_png):
    """hists_por_esquema: {'iid': {cliente: hist}, 'noniid': {...}} -> PNG con barras apiladas."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colores = ["#2a6fbb", "#e0752d", "#3a9a5b", "#8e5bb5", "#c2453d",
               "#8c8c3a", "#3aa0a8", "#a8648c", "#6b6b6b", "#c9a227"]
    esquemas = list(hists_por_esquema)
    fig, ejes = plt.subplots(1, len(esquemas), figsize=(6.2 * len(esquemas), 4.6),
                             sharey=True, squeeze=False)
    for ax, esq in zip(ejes[0], esquemas):
        hists = hists_por_esquema[esq]
        nombres = list(hists)
        fondo = np.zeros(len(nombres))
        for c, nombre_clase in enumerate(class_names):
            vals = np.array([hists[n][c] for n in nombres], dtype=float)
            ax.bar(nombres, vals, bottom=fondo, label=nombre_clase,
                   color=colores[c % len(colores)], edgecolor="white", linewidth=0.6)
            fondo += vals
        ax.set_title("IID" if esq == "iid" else "non-IID (Dirichlet)")
        ax.set_xlabel("cliente")
        ax.tick_params(axis="x", rotation=20)
        ax.spines[["top", "right"]].set_visible(False)
    ejes[0][0].set_ylabel("muestras asignadas")
    ejes[0][-1].legend(title="clase", frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.suptitle("Muestras por clase asignadas a cada cliente")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(ruta_png)), exist_ok=True)
    fig.savefig(ruta_png, dpi=140)
    plt.close(fig)
    return ruta_png
