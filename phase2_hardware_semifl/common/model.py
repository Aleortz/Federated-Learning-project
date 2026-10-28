"""
Regresion logistica binaria en numpy (el modelo de la phase 1, sin dependencias de ML).

Parametros: pesos W (n_features,) y sesgo b (escalar). Son los unicos datos del modelo
que viajan por la red: W + b ocupan ~100 floats (~2-3 KB en JSON), dentro del limite
de un mensaje del canal seguro.
"""
import numpy as np

from common import metrics


def sigmoide(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))        # estable para |z| grande


class LogisticModel:
    def __init__(self, n_features, W=None, b=0.0):
        self.n_features = n_features
        self.W = np.zeros(n_features) if W is None else np.asarray(W, dtype=np.float64).copy()
        self.b = float(b)

    # ---- parametros ----
    def get_params(self):
        return {"W": self.W.copy(), "b": self.b}

    def set_params(self, W, b):
        W = np.asarray(W, dtype=np.float64)
        if W.shape != (self.n_features,):
            raise ValueError(f"W tiene forma {W.shape}, se esperaba ({self.n_features},)")
        if not (np.all(np.isfinite(W)) and np.isfinite(b)):
            raise ValueError("parametros no finitos")
        self.W, self.b = W.copy(), float(b)

    # ---- inferencia ----
    def proba(self, X):
        return sigmoide(X @ self.W + self.b)

    def predict(self, X):
        return (self.proba(X) >= 0.5).astype(np.int64)

    def loss(self, X, y, l2=0.0):
        p = np.clip(self.proba(X), 1e-12, 1 - 1e-12)
        ce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return float(ce + 0.5 * l2 * np.dot(self.W, self.W))

    def evaluar(self, X, y, l2=0.0):
        m = metrics.clasificacion(y, self.predict(X))
        m["loss"] = self.loss(X, y, l2)
        return m

    # ---- entrenamiento local ----
    def entrenar(self, X, y, epochs, batch_size, lr, l2=0.0, seed=0):
        """SGD con minibatch sobre el modelo actual. Devuelve la perdida media del entrenamiento."""
        if len(y) == 0:
            raise ValueError("no hay muestras locales para entrenar")
        rng = np.random.default_rng(seed)
        total, vistas = 0.0, 0
        for _ in range(epochs):
            orden = rng.permutation(len(y))
            for ini in range(0, len(y), batch_size):
                idx = orden[ini:ini + batch_size]
                xb, yb = X[idx], y[idx]
                p = sigmoide(xb @ self.W + self.b)
                err = p - yb
                self.W -= lr * (xb.T @ err / len(idx) + l2 * self.W)
                self.b -= lr * float(err.mean())
                pc = np.clip(p, 1e-12, 1 - 1e-12)
                total += float(-(yb * np.log(pc) + (1 - yb) * np.log(1 - pc)).sum())
                vistas += len(idx)
        return total / vistas
