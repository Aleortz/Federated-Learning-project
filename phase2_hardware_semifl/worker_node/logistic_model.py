"""
Entrenamiento local de un worker: aplica el modelo global recibido, entrena con SUS datos
y devuelve la actualizacion (W, b) junto con las metricas locales.

El modelo (common/model.py) es el mismo en master y workers. Las metricas "antes" se
calculan con el modelo global recibido sobre los datos locales: miden cuanto se ajusta el
modelo global a la distribucion de este cliente (util para ver el efecto del non-IID).
"""
import time

from common.model import LogisticModel


class EntrenadorLocal:
    def __init__(self, X, y):
        if len(y) == 0:
            raise ValueError("el worker no tiene muestras locales")
        self.X, self.y = X, y
        self.modelo = LogisticModel(X.shape[1])

    @property
    def n(self):
        return len(self.y)

    def ronda(self, W, b, epochs, batch_size, lr, l2, seed):
        self.modelo.set_params(W, b)              # valida forma y finitud del modelo recibido
        antes = self.modelo.evaluar(self.X, self.y, l2)
        t0 = time.perf_counter()
        train_loss = self.modelo.entrenar(self.X, self.y, epochs, batch_size, lr, l2, seed)
        t = time.perf_counter() - t0
        despues = self.modelo.evaluar(self.X, self.y, l2)
        p = self.modelo.get_params()
        return {
            "n": self.n, "W": p["W"], "b": p["b"], "train_time_s": t,
            "metrics": {
                "train_loss": train_loss,
                "loss_before": antes["loss"], "acc_before": antes["accuracy"],
                "loss_after": despues["loss"], "acc_after": despues["accuracy"],
                "f1_after": despues["f1_macro"],
            },
        }
