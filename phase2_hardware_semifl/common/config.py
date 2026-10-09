"""
Configuracion de la capa de comunicacion segura (master <-> workers).
Se puede sobrescribir por linea de comandos o con variables de entorno.
"""
import os

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # phase2_hardware_semifl/

MASTER_ID = os.environ.get("SFL_MASTER_ID", "MASTER")
MASTER_HOST = os.environ.get("SFL_MASTER_HOST", "127.0.0.1")   # IPv4 de la PC del master
PORT = int(os.environ.get("SFL_PORT", "8080"))

# Carpeta con <ID>.key (privada, NUNCA a git) y authorized.txt (publicas)
KEYDIR = os.environ.get("SFL_KEYDIR", os.path.join(_BASE, "keys"))

# Tamano maximo del plaintext de un mensaje. El campo de longitud del paquete es
# uint16, asi que el tope fisico es 65535 bytes. Los parametros de la regresion
# logistica de la phase 1 (103 coef + intercepto) ocupan unos 2-3 KB en JSON.
MAX_MSG = 65535

# Segundos entre reintentos de conexion del worker
RECONNECT_S = 4
