"""
Resultados de un experimento: directorios, config.json, logging, graficos y comparacion.

Todo lo que se grafica o tabula se LEE de los CSV/JSON que escribe el coordinador
(global_metrics.csv, client_metrics.csv, events.csv, summary.json); nada se escribe a mano.
"""
import csv
import json
import logging
import os
import platform
import shutil
import sys
import time

import numpy as np

CAMPOS_COMPATIBLES = ("seed", "rounds", "local_epochs", "batch_size", "lr", "l2", "n_clients",
                      "val_frac", "test_frac", "optimizer", "loss")
COLORES = {"iid": "#2a6fbb", "non_iid": "#e0752d", "noniid": "#e0752d", "disconnect": "#c2453d",
           "baseline": "#2a6fbb"}


# ---------------------------------------------------------------- directorios y config
def preparar_directorio(results_dir, nombre):
    """
    results/<nombre>/ vacio y listo. Si ya existia con contenido, se MUEVE (no se borra) a
    results/_archive/<nombre>_<fecha> para no perder ejecuciones anteriores.
    """
    out = os.path.join(results_dir, nombre)
    if os.path.isdir(out) and os.listdir(out):
        destino = os.path.join(results_dir, "_archive", f"{nombre}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        shutil.move(out, destino)
    for sub in ("logs", "checkpoints", "plots"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    return out


def entorno():
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__}
    for mod in ("cryptography", "matplotlib"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "no instalado"
    return info


def guardar_json(obj, ruta):
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
    return ruta


def escribir_config(out, cfg, **extra):
    return guardar_json({"config": cfg.as_dict(), "environment": entorno(),
                         "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), **extra},
                        os.path.join(out, "config.json"))


def configurar_logging(logfile=None, level="INFO", nombre="", consola=True):
    """Handler a archivo (y consola) para el logger `nombre` ('' = raiz). Idempotente por archivo."""
    lg = logging.getLogger(nombre)
    lg.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    if consola and not any(getattr(h, "_sfl_consola", False) for h in lg.handlers):
        h = logging.StreamHandler()
        h._sfl_consola = True
        h.setFormatter(fmt)
        lg.addHandler(h)
    if logfile:
        os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)
        h = logging.FileHandler(logfile, encoding="utf-8")
        h.setFormatter(fmt)
        lg.addHandler(h)
        return h
    return None


# ---------------------------------------------------------------- lectura
def leer_csv(ruta):
    with open(ruta, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    out = []
    for fila in filas:
        d = {}
        for k, v in fila.items():
            try:
                d[k] = float(v) if v not in ("", None) else None
            except ValueError:
                d[k] = v
        out.append(d)
    return out


def leer_json(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})
    return plt


def _guardar(fig, ruta):
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return ruta


def _serie(filas, campo, solo_agregadas=False):
    fs = [f for f in filas if not solo_agregadas or f["aggregated"]]
    return [f["round"] for f in fs], [f[campo] for f in fs]


# ---------------------------------------------------------------- graficos por experimento
def graficos_experimento(out, titulo=None):
    """plots/ de UNA ejecucion: accuracy, perdida, participantes y tiempo por ronda."""
    plt = _plt()
    g = leer_csv(os.path.join(out, "global_metrics.csv"))
    nombre = titulo or os.path.basename(out.rstrip("/\\"))
    rutas = []
    for campo, etiqueta, fn in (("test_accuracy", "Accuracy (test global)", "accuracy_per_round.png"),
                                ("test_loss", "Perdida (test global)", "loss_per_round.png")):
        fig, ax = plt.subplots(figsize=(6, 4))
        x, y = _serie(g, campo)
        ax.plot(x, y, marker="o", label="test", color=COLORES.get(nombre, "#2a6fbb"))
        xv, yv = _serie(g, "val_accuracy" if "acc" in campo else "val_loss")
        ax.plot(xv, yv, marker=".", ls="--", color="#888", label="validacion")
        fallidas = [f["round"] for f in g if not f["aggregated"]]
        for r in fallidas:
            ax.axvline(r, color="#c2453d", alpha=.25, lw=6)
        ax.set(xlabel="ronda (intento)", ylabel=etiqueta, title=f"{nombre}: {etiqueta} por ronda")
        ax.legend(frameon=False)
        rutas.append(_guardar(fig, os.path.join(out, "plots", fn)))
    rutas.append(grafico_participantes(out, g, nombre))
    return rutas


def grafico_participantes(out, g=None, nombre=None):
    """Clientes que participaron por ronda (barras) y duracion de la ronda (linea)."""
    plt = _plt()
    g = g or leer_csv(os.path.join(out, "global_metrics.csv"))
    nombre = nombre or os.path.basename(out.rstrip("/\\"))
    r = [f["round"] for f in g]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    colores = ["#3a9a5b" if f["aggregated"] else "#c2453d" for f in g]
    a1.bar(r, [f["responded"] for f in g], color=colores)
    a1.plot(r, [f["expected"] for f in g], "k_", ms=14, label="esperados al abrir la ronda")
    a1.set(ylabel="clientes con actualizacion valida", title=f"{nombre}: participacion por ronda "
           "(verde = agregada, rojo = fallida)")
    a1.legend(frameon=False, loc="lower right")
    a2.plot(r, [f["duration_s"] for f in g], marker="o", color="#2a6fbb")
    a2.axhline(g[0]["round_timeout_s"], color="#c2453d", ls=":", label="round_timeout")
    a2.set(xlabel="ronda (intento)", ylabel="duracion de la ronda (s)")
    a2.legend(frameon=False)
    return _guardar(fig, os.path.join(out, "plots", "participants_and_time_per_round.png"))


def grafico_resiliencia(dir_desconexion, dir_base=None):
    """Compara la ejecucion con desconexion contra la de referencia sin desconexion (si existe)."""
    plt = _plt()
    g = leer_csv(os.path.join(dir_desconexion, "global_metrics.csv"))
    ev = leer_csv(os.path.join(dir_desconexion, "events.csv"))
    fig, axs = plt.subplots(3, 1, figsize=(7.5, 8.5), sharex=True)
    r = [f["round"] for f in g]
    axs[0].step(r, [f["responded"] for f in g], where="mid", color="#c2453d", label="con desconexion")
    axs[0].set(ylabel="clientes agregados", title="Prueba de desconexion: participacion, tiempo y accuracy")
    axs[1].plot(r, [f["duration_s"] for f in g], marker="o", color="#c2453d", label="con desconexion")
    axs[1].set(ylabel="duracion de la ronda (s)")
    axs[2].plot(r, [f["test_accuracy"] for f in g], marker="o", color="#c2453d", label="con desconexion")
    axs[2].set(xlabel="ronda (intento)", ylabel="accuracy (test)")
    if dir_base and os.path.exists(os.path.join(dir_base, "global_metrics.csv")):
        b = leer_csv(os.path.join(dir_base, "global_metrics.csv"))
        rb = [f["round"] for f in b]
        axs[0].step(rb, [f["responded"] for f in b], where="mid", color="#2a6fbb", ls="--", label="referencia (4 clientes)")
        axs[1].plot(rb, [f["duration_s"] for f in b], marker=".", ls="--", color="#2a6fbb", label="referencia")
        axs[2].plot(rb, [f["test_accuracy"] for f in b], marker=".", ls="--", color="#2a6fbb", label="referencia")
    marcas = {"client_timeout": "timeout", "disconnected": "desconexion", "reconnected": "reconexion",
              "participation_recovered": "vuelve a participar"}
    visto = set()
    for e in ev:
        if e["event"] in marcas:
            for ax in axs:
                ax.axvline(e["round"], color="#999", alpha=.4, lw=1)
            etq = marcas[e["event"]]
            axs[0].annotate(etq, (e["round"], axs[0].get_ylim()[1]), rotation=90, va="top", ha="right",
                            fontsize=7, color="#555") if (e["event"], e["round"]) not in visto else None
            visto.add((e["event"], e["round"]))
    axs[0].legend(frameon=False, loc="lower right")
    return _guardar(fig, os.path.join(dir_desconexion, "plots", "disconnect_resilience.png"))


# ---------------------------------------------------------------- comparacion IID vs non-IID
def _resumen_ejecucion(out):
    g = leer_csv(os.path.join(out, "global_metrics.csv"))
    c = leer_csv(os.path.join(out, "client_metrics.csv"))
    s = leer_json(os.path.join(out, "summary.json")) if os.path.exists(os.path.join(out, "summary.json")) else {}
    ok = [f for f in c if f["status"] == "ok" and f["local_acc_before"] is not None]
    por_ronda = {}
    for f in ok:
        por_ronda.setdefault(f["round"], []).append(f["local_acc_before"])
    var = [float(np.std(v)) for v in por_ronda.values() if len(v) > 1]
    ult = por_ronda[max(por_ronda)] if por_ronda else []
    ult_g = [f for f in g if f["aggregated"]][-1] if any(f["aggregated"] for f in g) else g[-1]
    return {
        "Accuracy final": ult_g["test_accuracy"], "Precision final (macro)": ult_g["test_precision_macro"],
        "Recall final (macro)": ult_g["test_recall_macro"], "F1-score final (macro)": ult_g["test_f1_macro"],
        "Perdida final (test)": ult_g["test_loss"],
        "Rondas completadas": s.get("rounds_completed", sum(1 for f in g if f["aggregated"])),
        "Rondas fallidas": s.get("rounds_failed", sum(1 for f in g if not f["aggregated"])),
        "Tiempo total (s)": s.get("total_time_s", g[-1]["elapsed_s"]),
        "Tiempo medio por ronda (s)": float(np.mean([f["duration_s"] for f in g])),
        "Variabilidad entre clientes (std acc local del modelo global, ultima ronda)": float(np.std(ult)) if ult else None,
        "Variabilidad entre clientes (std acc local, media de rondas)": float(np.mean(var)) if var else None,
    }


def comparar(dirs, out_dir):
    """
    dirs: {"iid": ruta, "non_iid": ruta}. Escribe comparison/summary.csv, summary.json y los
    graficos de evolucion y rendimiento final. Devuelve (resumen, avisos_de_compatibilidad).
    """
    plt = _plt()
    cfgs = {k: leer_json(os.path.join(d, "config.json"))["config"] for k, d in dirs.items()}
    avisos = []
    for campo in CAMPOS_COMPATIBLES:
        vals = {k: c.get(campo) for k, c in cfgs.items()}
        if len({json.dumps(v) for v in vals.values()}) > 1:
            avisos.append(f"configuraciones NO comparables: '{campo}' difiere {vals}")
    resumen = {k: _resumen_ejecucion(d) for k, d in dirs.items()}
    indicadores = list(next(iter(resumen.values())))
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["indicador", *dirs])
        for ind in indicadores:
            w.writerow([ind, *[resumen[k][ind] for k in dirs]])
    guardar_json({"runs": resumen, "configs": cfgs, "compatibility_warnings": avisos},
                 os.path.join(out_dir, "summary.json"))

    glob = {k: leer_csv(os.path.join(d, "global_metrics.csv")) for k, d in dirs.items()}
    for campo, etq, fn in (("test_accuracy", "Accuracy (test global)", "accuracy_per_round.png"),
                           ("test_loss", "Perdida (test global)", "loss_per_round.png"),
                           ("test_f1_macro", "F1 macro (test global)", "f1_per_round.png")):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for k, g in glob.items():
            x, y = _serie(g, campo)
            ax.plot(x, y, marker="o", label=k, color=COLORES.get(k))
        ax.set(xlabel="ronda federada", ylabel=etq, title=f"IID vs non-IID: {etq} por ronda")
        ax.legend(frameon=False)
        _guardar(fig, os.path.join(out_dir, "plots", fn))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    mets = ["Accuracy final", "Precision final (macro)", "Recall final (macro)", "F1-score final (macro)"]
    ancho = 0.8 / len(dirs)
    for i, k in enumerate(dirs):
        vals = [resumen[k][m] for m in mets]
        barras = ax.bar(np.arange(len(mets)) + i * ancho, vals, ancho, label=k, color=COLORES.get(k))
        for b, v in zip(barras, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(len(mets)) + ancho * (len(dirs) - 1) / 2,
                  ["Accuracy", "Precision", "Recall", "F1"])
    ax.set(ylabel="valor (test global)", ylim=(0, 1.05), title="Rendimiento final: IID vs non-IID")
    ax.legend(frameon=False)
    _guardar(fig, os.path.join(out_dir, "plots", "final_performance.png"))

    # distribucion de clases (3 y comparativa de particiones)
    hists, nombres = {}, None
    for k, d in dirs.items():
        ruta = os.path.join(d, "partition_statistics.csv")
        if os.path.exists(ruta):
            filas = leer_csv(ruta)
            nombres = [c[2:] for c in filas[0] if c.startswith("n_")]
            hists["iid" if "non" not in k else "noniid"] = {
                f["cliente"]: [int(f[f"n_{n}"]) for n in nombres] for f in filas}
    if hists:
        from common import particion
        particion.grafico_comparativo(hists, nombres, os.path.join(out_dir, "plots", "partition_comparison.png"))
    return resumen, avisos
