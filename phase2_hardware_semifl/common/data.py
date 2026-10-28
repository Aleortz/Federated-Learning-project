"""
Dataset Adult (UCI Census Income): carga, preprocesamiento y separacion train/val/test.

Todo es determinista para (csv, seed, val_frac, test_frac): cada maquina puede
reconstruir exactamente la misma matriz sin intercambiar datos. `huella()` devuelve
un hash para que el master compruebe que todos los workers tienen el mismo dataset
y el mismo preprocesamiento.

Preprocesamiento (identico en IID y non-IID):
  - numericas (age, fnlwgt, education.num, capital.gain, capital.loss, hours.per.week):
    estandarizadas con media/desviacion del conjunto de ENTRENAMIENTO global.
  - categoricas (workclass, marital.status, occupation, relationship, race, sex,
    native.country): one-hot con las categorias del archivo ('?' es una categoria).
    'education' se omite porque 'education.num' ya la codifica.
  - etiqueta: 1 si income es '>50K', 0 si es '<=50K'.

Limitacion documentada: las estadisticas de estandarizacion salen de todo el conjunto
de entrenamiento. En un despliegue real serian constantes publicas o se calcularian de
forma federada; aqui se simplifica para que el preprocesamiento sea identico.
"""
import csv
import hashlib
import os

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ADULT_CSV = os.environ.get("SFL_DATA_CSV", os.path.join(_BASE, "data", "adult.csv"))

NUMERICAS = ("age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week")
CATEGORICAS = ("workclass", "marital.status", "occupation", "relationship", "race", "sex",
               "native.country")
CLASES = ("<=50K", ">50K")


class DatasetError(Exception):
    pass


def leer_csv(ruta=ADULT_CSV):
    if not os.path.exists(ruta):
        raise DatasetError(f"no existe el dataset '{ruta}'. Colocalo en data/adult.csv "
                           f"o define SFL_DATA_CSV")
    with open(ruta, newline="") as f:
        filas = list(csv.DictReader(f))
    faltan = [c for c in NUMERICAS + CATEGORICAS + ("income",) if filas and c not in filas[0]]
    if not filas or faltan:
        raise DatasetError(f"{ruta}: vacio o sin columnas {faltan}")
    return filas


def etiquetas(ruta=ADULT_CSV):
    return np.array([1 if f["income"].strip().startswith(">") else 0 for f in leer_csv(ruta)],
                    dtype=np.int64)


def separar(y, val_frac, test_frac, seed):
    """Indices (train, val, test) estratificados por clase. Disjuntos y exhaustivos."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        n_te, n_va = int(round(len(idx) * test_frac)), int(round(len(idx) * val_frac))
        te.append(idx[:n_te])
        va.append(idx[n_te:n_te + n_va])
        tr.append(idx[n_te + n_va:])
    return tuple(np.sort(np.concatenate(p)) for p in (tr, va, te))


class Dataset:
    """X (n, d) float64, y (n,) int64 y los indices de cada split sobre las filas del CSV."""

    def __init__(self, X, y, train_idx, val_idx, test_idx, feature_names, seed):
        self.X, self.y = X, y
        self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx
        self.feature_names = feature_names
        self.seed = seed

    @property
    def n_features(self):
        return self.X.shape[1]

    def subset(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        return self.X[idx], self.y[idx]

    def huella(self):
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.X.round(9)).tobytes())
        h.update(self.y.tobytes())
        for a in (self.train_idx, self.val_idx, self.test_idx):
            h.update(a.tobytes())
        return h.hexdigest()[:16]


def cargar(ruta=ADULT_CSV, val_frac=0.10, test_frac=0.20, seed=42):
    filas = leer_csv(ruta)
    y = np.array([1 if f["income"].strip().startswith(">") else 0 for f in filas], dtype=np.int64)
    train_idx, val_idx, test_idx = separar(y, val_frac, test_frac, seed)

    num = np.array([[float(f[c]) for c in NUMERICAS] for f in filas])
    media, desv = num[train_idx].mean(0), num[train_idx].std(0)
    desv[desv == 0] = 1.0
    num = (num - media) / desv

    columnas, nombres = [num], list(NUMERICAS)
    for c in CATEGORICAS:
        cats = sorted({f[c].strip() for f in filas})
        valores = np.array([f[c].strip() for f in filas])
        columnas.append(np.stack([(valores == k).astype(np.float64) for k in cats], axis=1))
        nombres += [f"{c}={k}" for k in cats]
    X = np.concatenate(columnas, axis=1)
    return Dataset(X, y, train_idx, val_idx, test_idx, nombres, seed)
