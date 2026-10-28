#!/usr/bin/env bash
# Lanzador (Linux/macOS). Equivalente a fl.ps1. Uso: scripts/fl.sh <comando> [args]
#   setup | data | partitions | server <iid|noniid> | client <ID> <IP> | test | iid | noniid | disconnect | compare | all
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."          # phase2_hardware_semifl/, desde cualquier directorio
PY=python3; [ -x .venv/bin/python ] && PY=.venv/bin/python
cmd="${1:-help}"; shift || true
case "$cmd" in
  setup)      python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt ;;
  data)       $PY -m scripts.preparar_datos "$@" ;;
  partitions) $PY -m scripts.preparar_datos --both "$@" ;;
  server)     s="$1"; shift; name="$s"; [ "$s" = noniid ] && name=non_iid
              $PY -m master_edge.run_server --scheme "$s" --run-name "$name" "$@" ;;
  client)     id="$1"; ip="$2"; shift 2; $PY -m worker_node.run_worker --id "$id" --master "$ip" "$@" ;;
  test)       $PY -m pytest -q "$@" ;;
  iid)        $PY -m scripts.run_local --scheme iid --run-name iid "$@" ;;
  noniid)     $PY -m scripts.run_local --scheme noniid --run-name non_iid "$@" ;;
  disconnect) $PY -m scripts.run_resiliencia "$@" ;;
  compare)    $PY -m scripts.comparar "$@" ;;
  all)        $PY -m pytest -q; $PY -m scripts.run_local --scheme iid --run-name iid
              $PY -m scripts.run_local --scheme noniid --run-name non_iid; $PY -m scripts.run_resiliencia; $PY -m scripts.comparar ;;
  *)          sed -n 2,4p "$0" ;;
esac
