"""
Metricas de clasificacion y de heterogeneidad (solo numpy).

Precision, Recall y F1 se calculan con promedio MACRO (media simple entre clases),
que es la convencion usada en CHANGE_REPO. Las metricas globales se calculan siempre
evaluando el modelo global sobre el conjunto de test independiente, no promediando
metricas locales.
"""
import numpy as np


def matriz_confusion(y_true, y_pred, n_clases=2):
    m = np.zeros((n_clases, n_clases), dtype=np.int64)
    np.add.at(m, (np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)), 1)
    return m


def clasificacion(y_true, y_pred, n_clases=2):
    """accuracy + precision/recall/F1 macro + matriz de confusion."""
    cm = matriz_confusion(y_true, y_pred, n_clases)
    tp = np.diag(cm).astype(float)
    pred, real = cm.sum(0).astype(float), cm.sum(1).astype(float)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    rec = np.divide(tp, real, out=np.zeros_like(tp), where=real > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "precision_macro": float(prec.mean()),
        "recall_macro": float(rec.mean()),
        "f1_macro": float(f1.mean()),
        "confusion": cm.tolist(),
    }


def jensen_shannon(p, q):
    """Divergencia de Jensen-Shannon en bits (0 = iguales, 1 = soportes disjuntos)."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def heterogeneidad(hists):
    """
    hists: lista de histogramas de clases por cliente. Devuelve la JS de cada cliente
    respecto de la distribucion global (agregada) y su media ponderada por muestras.
    Es la medida cuantitativa de non-IID-ness usada en todo el proyecto.
    """
    h = np.asarray(hists, dtype=float)
    glob = h.sum(0)
    js = [jensen_shannon(fila, glob) if fila.sum() > 0 else 0.0 for fila in h]
    pesos = h.sum(1) / h.sum()
    return {"js_por_cliente": js, "js_media": float(np.dot(js, pesos)), "js_max": float(max(js))}
