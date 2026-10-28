#!/usr/bin/env bash
# Prepara un worker Linux: entorno virtual + dependencias. Luego: scripts/fl.sh client WORKER_1 <IP_MASTER>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
echo "Listo. Genera tu clave: .venv/bin/python -m common.keys --genkey WORKER_1"
