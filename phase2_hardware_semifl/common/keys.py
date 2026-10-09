"""
Gestion de claves ECDSA P-256 (identidad de cada nodo). Desde phase2_hardware_semifl/:

    python -m common.keys --genkey MASTER            # en la PC del master
    python -m common.keys --genkey WORKER_1          # cada worker, en su PC

    python -m common.keys --addpub WORKER_2=04AB...  # agrega la PUBLICA de otro
    python -m common.keys --list                     # IDs y huellas

Cada uno genera SOLO su propia clave; la privada nunca sale de su equipo.

Archivos (en keys/):
    <ID>.key         clave PRIVADA (64 hex). NUNCA se sube a GitHub (.gitignore).
    authorized.txt   ID=clave publica. No es secreta.

Quien necesita que:
    master   MASTER.key + publicas de TODOS los workers
    worker   <su ID>.key + publica del MASTER (y la suya)
"""
import argparse
import os
import sys

from cryptography.hazmat.primitives.asymmetric import ec

from common import protocol as P


def leer_pubs(keydir):
    ruta = os.path.join(keydir, "authorized.txt")
    out = {}
    if os.path.exists(ruta):
        with open(ruta) as f:
            for linea in f:
                linea = linea.strip()
                if linea and not linea.startswith("#") and "=" in linea:
                    pid, _, hexa = linea.partition("=")
                    out[pid.strip()] = hexa.strip().upper()
    return out


def escribir_pubs(keydir, pubs):
    os.makedirs(keydir, exist_ok=True)
    with open(os.path.join(keydir, "authorized.txt"), "w") as f:
        f.write("# ID=clave publica ECDSA P-256 (no es secreta)\n")
        for pid in sorted(pubs):
            f.write(f"{pid}={pubs[pid]}\n")


def main():
    ap = argparse.ArgumentParser(description="Gestion de claves")
    ap.add_argument("--genkey", nargs="+", metavar="ID", help="genera TU par de claves")
    ap.add_argument("--addpub", action="append", metavar="ID=PUBLICA_HEX", help="agrega la publica de otro (repetible)")
    ap.add_argument("--list", action="store_true", help="lista IDs y huellas")
    ap.add_argument("--force", action="store_true", help="con --genkey: sobrescribe una clave existente")
    ap.add_argument("--keydir", default=P.KEYDIR_DEFECTO)
    args = ap.parse_args()
    if not (args.genkey or args.addpub or args.list):
        ap.print_help()
        return

    kd = args.keydir
    pubs = leer_pubs(kd)

    for pid in args.genkey or []:
        if not P.id_valido(pid):
            sys.exit(f"ID invalido '{pid}' (1 a 19 caracteres).")
        ruta = os.path.join(kd, pid + ".key")
        if os.path.exists(ruta) and not args.force:
            sys.exit(f"{ruta} ya existe. Para regenerarla usa --force (y avisa a los demas de tu nueva publica).")
        os.makedirs(kd, exist_ok=True)
        k = ec.generate_private_key(ec.SECP256R1())
        with open(ruta, "w") as f:
            f.write(k.private_numbers().private_value.to_bytes(32, "big").hex().upper() + "\n")
        pubs[pid] = P.pub_bytes(k.public_key()).hex().upper()
        print(f"[+] {pid}: privada en {ruta} (SECRETA).")
        print(f"    Comparte SOLO esta linea con los demas:\n    python -m common.keys --addpub {pid}={pubs[pid]}\n")

    for item in args.addpub or []:
        pid, _, hexa = item.partition("=")
        pid, hexa = pid.strip(), hexa.strip().upper()
        if not P.id_valido(pid) or not hexa:
            sys.exit(f"Formato invalido '{item}'. Usa ID=CLAVE_PUBLICA_HEX.")
        try:
            pub = P.pub_desde_hex(hexa)
        except ValueError as e:
            sys.exit(f"{pid}: {e} (¿se copio mal?).")
        propia = os.path.join(kd, pid + ".key")
        if os.path.exists(propia) and P.pub_bytes(P.cargar_priv(propia).public_key()) != P.pub_bytes(pub):
            sys.exit(f"{pid}: esa publica no corresponde a tu {pid}.key.")
        if pid in pubs and pubs[pid] != hexa:
            print(f"[!] {pid} ya tenia otra clave publica: se reemplaza.")
        pubs[pid] = hexa
        print(f"[+] Agregada la publica de {pid} (huella {P.huella_pub(pub)}).")

    if args.genkey or args.addpub:
        escribir_pubs(kd, pubs)

    if args.list:
        print(f"authorized.txt en {kd}:")
        for pid in sorted(pubs):
            priv = " (tengo su privada)" if os.path.exists(os.path.join(kd, pid + ".key")) else ""
            print(f"  {pid:19s} {P.huella_pub(P.pub_desde_hex(pubs[pid]))}{priv}")


if __name__ == "__main__":
    main()
