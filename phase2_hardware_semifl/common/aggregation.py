"""
FedAvg y validacion de actualizaciones locales.

    w_{t+1} = sum_{k in S_t} (n_k / sum_{j in S_t} n_j) * w_k

S_t son SOLO los clientes con actualizacion valida en la ronda t. Si un cliente no
responde, el resto se reponderan (los pesos normalizados siguen sumando 1).
"""
import math

import numpy as np


class UpdateInvalido(ValueError):
    """Una actualizacion local no cumple el contrato (forma, finitud, n, ronda...)."""


def validar_parametros(W, b, n_features):
    """Devuelve (W ndarray float64, b float) o lanza UpdateInvalido."""
    try:
        W = np.asarray(W, dtype=np.float64)
        b = float(b)
    except (TypeError, ValueError) as e:
        raise UpdateInvalido(f"parametros no numericos: {e}")
    if W.shape != (n_features,):
        raise UpdateInvalido(f"dimension incompatible: W {W.shape}, se esperaba ({n_features},)")
    if not (np.all(np.isfinite(W)) and math.isfinite(b)):
        raise UpdateInvalido("parametros no finitos (NaN/Inf)")
    return W, b


def validar_n(n):
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise UpdateInvalido(f"n de muestras invalido: {n!r}")
    return n


def pesos_fedavg(ns):
    """n_k / sum(n) para cada cliente; suman 1 dentro de la tolerancia numerica."""
    ns = [validar_n(n) for n in ns]
    if not ns:
        raise ValueError("no hay actualizaciones que agregar")
    total = sum(ns)
    w = [n / total for n in ns]
    assert abs(sum(w) - 1.0) < 1e-9
    return w


def fedavg(updates):
    """
    updates: lista de (W, b, n). Devuelve (W_global, b_global, pesos_normalizados).
    Exige formas identicas, valores finitos y n > 0.
    """
    if not updates:
        raise ValueError("no hay actualizaciones que agregar")
    n_features = np.asarray(updates[0][0]).shape[0] if np.ndim(updates[0][0]) else -1
    validos = [(*validar_parametros(W, b, n_features), validar_n(n)) for W, b, n in updates]
    pesos = pesos_fedavg([n for _, _, n in validos])
    W = sum(p * w for p, (w, _, _) in zip(pesos, validos))
    b = sum(p * bb for p, (_, bb, _) in zip(pesos, validos))
    return W, float(b), pesos
