#!/usr/bin/env python3
"""
Atacante para las pruebas de seguridad del canal master <-> worker.
Se ejecuta desde phase2_hardware_semifl/:  python scripts/atacante.py ...

Ataques contra el MASTER:
    python scripts/atacante.py <IP_master> no-autorizado
        Se presenta con un ID que no esta en authorized.txt. Debe ser rechazado
        de inmediato (status = 0), antes de cualquier calculo.

    python scripts/atacante.py <IP_master> suplantar <ID_valido>
        Usa el ID de un cliente autorizado pero NO tiene su clave privada; manda
        una firma inventada. El master debe rechazarlo.

    python scripts/atacante.py <IP_master> miembro <ID_que_finge> <ID_cuya_clave_posee>
        Atacante INTERNO: tiene la clave privada valida de un miembro
        (keys/<ID_cuya_clave_posee>.key) y la usa para fingir ser OTRO.
        Debe ser rechazado: robar una clave solo permite ser ESE dispositivo.

    python scripts/atacante.py <IP_master> inyectar
        Manda una trama de datos forjada sin handshake. Debe descartarse.

Ataque contra los WORKERS:
    python scripts/atacante.py 0.0.0.0 servidor-falso
        Escucha en el puerto 8080 haciendose pasar por MASTER (MITM / master
        impostor). Apunta un worker a la IP de esta maquina: debe detectar que la
        firma no es la del master real y no entregarle nada.

En cada caso, mira la consola del master / worker: debe aparecer una [ALERTA].
"""
import os
import socket
import sys

from cryptography.hazmat.primitives.asymmetric import ec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import protocol as P  # noqa: E402

TIMEOUT = 20


def recibir(sock, n):
    datos = b""
    while len(datos) < n:
        try:
            t = sock.recv(n - len(datos))
        except (socket.timeout, OSError):
            break
        if not t:
            break
        datos += t
    return datos


def conectar(ip):
    s = socket.create_connection((ip, P.PUERTO), timeout=TIMEOUT)
    s.settimeout(TIMEOUT)
    return s


def construir_hello(dev_id):
    # dh_pub valido para que la prueba llegue a la autenticacion
    _, y = P.dh_par()
    return P.HELLO.pack(P.MAGIC, P.id_bytes(dev_id), os.urandom(16), y)


def no_autorizado(ip):
    dev_id = "WORKER_Hacker"
    print(f"[D] Conectando a {ip}:{P.PUERTO} como '{dev_id}' (NO esta en la lista)...")
    s = conectar(ip)
    s.sendall(construir_hello(dev_id))
    r = recibir(s, P.HELLOREPLY.size)
    s.close()
    if r and r[0] == 0:
        print("[OK] El master RECHAZO al dispositivo no autorizado (status = 0).")
    elif not r:
        print("[OK] El master cerro la conexion sin dar sesion.")
    else:
        print(f"[!!] Respuesta inesperada (primer byte = {r[0]}). No deberia aceptar.")


def suplantar(ip, dev_id):
    print(f"[D] Conectando a {ip}:{P.PUERTO} como '{dev_id}', pero SIN su clave privada...")
    s = conectar(ip)
    s.sendall(construir_hello(dev_id))
    r = recibir(s, P.HELLOREPLY.size)
    if not r or r[0] != 1:
        print("[OK] El master rechazo el intento ya en el primer mensaje.")
        s.close()
        return
    print("[D] El master respondio y firmo su parte (normal: el ID si existe).")
    print("[D] Ahora habria que firmar con la clave privada de ese worker. Mando una firma inventada...")
    s.sendall(os.urandom(64))
    resultado = recibir(s, 1)
    s.close()
    if resultado in (b"\x00", b""):
        print("[OK] El master RECHAZO la autenticacion: sin la clave privada no se puede suplantar.")
    else:
        print(f"[!!] El master acepto la sesion (resultado = {resultado.hex()}). NO deberia pasar.")


def miembro(ip, dev_id, id_clave):
    priv = P.cargar_priv(os.path.join(P.KEYDIR_DEFECTO, id_clave + ".key"))
    print(f"[D] Atacante INTERNO: tengo la clave valida de '{id_clave}' y la uso para fingir ser '{dev_id}'...")
    s = conectar(ip)
    h = construir_hello(dev_id)
    s.sendall(h)
    r = recibir(s, P.HELLOREPLY.size)
    if len(r) != P.HELLOREPLY.size or r[0] != 1:
        print("[OK] El master rechazo el intento ya en el primer mensaje.")
        s.close()
        return
    _, rid_b, rnonce, rpub, _ = P.HELLOREPLY.unpack(r)
    T = P.transcript(h, rid_b, rnonce, rpub)
    s.sendall(P.firmar(priv, b"P1-INIT", T))
    resultado = recibir(s, 1)
    s.close()
    if resultado in (b"\x00", b""):
        print(f"[OK] El master RECHAZO la firma: la clave de '{id_clave}' no sirve para ser '{dev_id}'.")
    elif dev_id == id_clave:
        print(f"[OK] (control) Sesion legitima aceptada: '{id_clave}' firmo como si mismo.")
    else:
        print(f"[!!] El master acepto la sesion (resultado = {resultado.hex()}). NO deberia pasar.")


def inyectar(ip):
    print(f"[D] Enviando a {ip}:{P.PUERTO} una trama de datos FORJADA, sin handshake...")
    s = conectar(ip)
    header = P.HEADER.pack(P.id_bytes("WORKER_1"), 0xDEADBEEF, 1, os.urandom(12), 24)
    paquete = bytes([P.FRAME_DATA]) + header + os.urandom(24) + os.urandom(16)
    s.sendall(paquete.ljust(P.HELLO.size, b"\0"))   # relleno: el master lo lee como un Hello
    resp = recibir(s, 16)
    s.close()
    if not resp:
        print("[OK] El master descarto la trama forjada y cerro la conexion.")
    else:
        print(f"[!!] Respuesta inesperada: {resp.hex()}")


def servidor_falso(ip_escucha, server_id=P.config.MASTER_ID):
    clave_falsa = ec.generate_private_key(ec.SECP256R1())
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((ip_escucha, P.PUERTO))
    s.listen(4)
    print(f"[D] Master IMPOSTOR escuchando en {ip_escucha}:{P.PUERTO} como '{server_id}'. Ctrl+C para salir.")
    while True:
        c, addr = s.accept()
        c.settimeout(TIMEOUT)
        hello = recibir(c, P.HELLO.size)
        if len(hello) != P.HELLO.size:
            c.close()
            continue
        _, cid_b, _, _ = P.HELLO.unpack(hello)
        print(f"[D] Worker '{P.id_str(cid_b)}' ({addr[0]}) cayo en el impostor. Respondo con MI clave...")
        _, y = P.dh_par()
        nonce = os.urandom(16)
        rid = P.id_bytes(server_id)
        T = P.transcript(hello, rid, nonce, y)
        c.sendall(P.HELLOREPLY.pack(1, rid, nonce, y, P.firmar(clave_falsa, b"P1-RESP", T)))
        fin = recibir(c, 64)
        c.close()
        if len(fin) == 64:
            print("[!!] El worker acepto la firma falsa y siguio el handshake. NO deberia pasar.")
        else:
            print("[OK] El worker detecto la firma invalida y corto: no hay MITM posible.")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    ip, modo = sys.argv[1], sys.argv[2]
    try:
        if modo == "no-autorizado":
            no_autorizado(ip)
        elif modo == "suplantar" and len(sys.argv) >= 4:
            suplantar(ip, sys.argv[3])
        elif modo == "miembro" and len(sys.argv) >= 5:
            miembro(ip, sys.argv[3], sys.argv[4])
        elif modo == "inyectar":
            inyectar(ip)
        elif modo == "servidor-falso":
            servidor_falso(ip, sys.argv[3] if len(sys.argv) >= 4 else P.config.MASTER_ID)
        else:
            print(__doc__)
            sys.exit(1)
    except KeyboardInterrupt:
        pass
    except OSError as e:
        sys.exit(f"[X] {e}")


if __name__ == "__main__":
    main()
