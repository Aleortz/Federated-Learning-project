"""
Canal seguro master <-> worker para el FL semi-centralizado.

Protocolo (heredado del Proyecto 1 de Criptografia, topologia cliente-servidor):
  - Diffie-Hellman MODP 2048 (RFC 3526) efimero en cada sesion.
  - Autenticacion mutua con firmas ECDSA P-256 sobre el transcript del handshake.
    Cada worker tiene grabada (pinned) la clave publica del master -> sin MITM.
  - HKDF-SHA256 -> una cadena de claves por direccion + Session ID.
  - Cadena de claves por mensaje (MK_n = HKDF(CK_n), CK_(n+1) = HKDF(CK_n)).
  - AES-256-GCM con la cabecera completa como AAD; SEQ creciente contra replay.

El master es siempre el respondedor y cada worker el iniciador. Encima del canal
viajan mensajes JSON (dict con un campo "type"); ver codificar()/decodificar().
"""
import json
import os
import socket
import struct
import threading
import time

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature, encode_dss_signature
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from common import config

MAGIC = b"SFL1"                 # va dentro del transcript: separa este protocolo de otros
PUERTO = config.PORT
ID_LEN = 20                     # IDs de 1 a 19 caracteres
MAX_CIPHER = config.MAX_MSG     # bytes de plaintext por mensaje (tope fisico: uint16)
DH_LEN = 256
DH_PRIV_BYTES = 32
DH_G = 2
DH_P = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74"
    "020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F1437"
    "4FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF05"
    "98DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB"
    "9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF695581718"
    "3995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF", 16)

HELLO = struct.Struct("<4s20s16s256s")              # magic, id, nonce, dh_pub
HELLOREPLY = struct.Struct("<B20s16s256s64s")       # status, id, nonce, dh_pub, firma (r||s)
HEADER = struct.Struct("<20sII12sH")                # sender_id, sid, seq, nonce, cipher_len
ACK = struct.Struct("<IB")                          # seq, status
FRAME_DATA, FRAME_ACK = 0x01, 0x02
MENSAJE_PRUEBA = '{"type": "test", "text": "mensaje de prueba de seguridad"}'

TIMEOUT_HS = 20                 # el DH de 2048 bits en un ESP32 tarda unos segundos
CHUNK = 64

KEYDIR_DEFECTO = config.KEYDIR


# =====================================================================
#  Utilidades
# =====================================================================
def hx(b):
    s = b.hex().upper()
    return "\n".join(s[i:i + CHUNK] for i in range(0, len(s), CHUNK))


def id_bytes(s):
    return s.encode().ljust(ID_LEN, b"\0")[:ID_LEN]


def id_str(b):
    return b.split(b"\0")[0].decode(errors="replace")


def id_valido(s):
    return 1 <= len(s.encode()) <= ID_LEN - 1


def recv_exact(sock, n):
    """Lee exactamente n bytes o lanza ConnectionError / OSError."""
    buf = bytearray()
    while len(buf) < n:
        t = sock.recv(n - len(buf))
        if not t:
            raise ConnectionError("conexion cerrada por el otro extremo")
        buf += t
    return bytes(buf)


def sha256(b):
    h = hashes.Hash(hashes.SHA256())
    h.update(b)
    return h.finalize()


# =====================================================================
#  Firmas, claves e HKDF (identico al .ino)
# =====================================================================
def firmar(priv, etiqueta, T):
    """ECDSA P-256 sobre SHA256(etiqueta || T); devuelve r||s (64 bytes)."""
    r, s = decode_dss_signature(priv.sign(etiqueta + T, ec.ECDSA(hashes.SHA256())))
    return r.to_bytes(32, "big") + s.to_bytes(32, "big")


def verificar(pub, etiqueta, T, sig):
    """Verifica con la clave publica GRABADA (nunca con una recibida por la red)."""
    if len(sig) != 64:
        return False
    der = encode_dss_signature(int.from_bytes(sig[:32], "big"), int.from_bytes(sig[32:], "big"))
    try:
        pub.verify(der, etiqueta + T, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


def pub_bytes(pub):
    return pub.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)


def huella_pub(pub):
    return sha256(pub_bytes(pub))[:4].hex().upper()


def hkdf(salt, ikm, info, length=32):
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(ikm)


def msg_key(ck):       # MK_n = HKDF(CK_n, "P1-MSG")
    return hkdf(None, ck, b"P1-MSG")


def next_chain(ck):    # CK_(n+1) = HKDF(CK_n, "P1-CHAIN")
    return hkdf(None, ck, b"P1-CHAIN")


def dh_par():
    priv = int.from_bytes(os.urandom(DH_PRIV_BYTES), "big")
    return priv, pow(DH_G, priv, DH_P).to_bytes(DH_LEN, "big")


def dh_secreto(priv, peer_pub):
    y = int.from_bytes(peer_pub, "big")
    if not (2 <= y < DH_P - 1):           # rechaza 0, 1, p-1 (valores degenerados)
        raise HandshakeError("clave publica DH degenerada")
    return pow(y, priv, DH_P).to_bytes(DH_LEN, "big")


def transcript(hello, rid_b, rnonce, rpub):
    # T = SHA256(Hello || ID_R || N_R || Y_R)
    return sha256(hello + rid_b + rnonce + rpub)


def derivar(z, n_i, n_r, T):
    okm = hkdf(n_i + n_r, z, b"P1-KEYS" + T, 68)
    return okm[:32], okm[32:64], struct.unpack("<I", okm[64:68])[0]


# =====================================================================
#  Archivos de claves
# =====================================================================
def cargar_priv(ruta):
    with open(ruta) as f:
        hexa = f.read().strip()
    if len(hexa) != 64:
        raise ValueError(f"{ruta}: la clave privada debe tener 64 caracteres hexadecimales")
    return ec.derive_private_key(int(hexa, 16), ec.SECP256R1())


def pub_desde_hex(hexa):
    hexa = hexa.strip()
    if len(hexa) != 130 or not hexa.upper().startswith("04"):
        raise ValueError("una clave publica debe tener 130 caracteres hex y empezar con 04")
    return ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), bytes.fromhex(hexa))


def cargar_autorizados(ruta):
    """authorized.txt -> dict ID -> clave publica."""
    out = {}
    with open(ruta) as f:
        for n, linea in enumerate(f, 1):
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            pid, _, hexa = linea.partition("=")
            try:
                out[pid.strip()] = pub_desde_hex(hexa)
            except Exception as e:
                raise ValueError(f"{ruta}, linea {n}: clave publica invalida para '{pid.strip()}' ({e})")
    return out


def cargar_identidad(keydir, my_id):
    """Carga MI clave privada y la lista de autorizados, y comprueba que coinciden."""
    priv = cargar_priv(os.path.join(keydir, my_id + ".key"))
    autorizados = cargar_autorizados(os.path.join(keydir, "authorized.txt"))
    if my_id not in autorizados:
        raise ValueError(f"'{my_id}' no esta en {keydir}/authorized.txt")
    if pub_bytes(priv.public_key()) != pub_bytes(autorizados[my_id]):
        raise ValueError(f"{my_id}.key no corresponde a la clave publica de '{my_id}' en authorized.txt")
    return priv, autorizados


# =====================================================================
#  Handshake
# =====================================================================
class HandshakeError(Exception):
    pass


def handshake_iniciador(sock, my_id, priv, peer_id, peer_pub, verbose=False):
    """Cliente -> servidor. Devuelve una Sesion o lanza HandshakeError."""
    sock.settimeout(TIMEOUT_HS)
    t0 = time.perf_counter()
    x, y = dh_par()
    nonce = os.urandom(16)
    hello = HELLO.pack(MAGIC, id_bytes(my_id), nonce, y)
    sock.sendall(hello)
    status, rid_b, rnonce, rpub, rsig = HELLOREPLY.unpack(recv_exact(sock, HELLOREPLY.size))
    if status != 1:
        raise HandshakeError(f"{peer_id} me rechazo (mi ID no esta en su authorized.txt)")
    if id_str(rid_b) != peer_id:
        raise HandshakeError(f"respondio con otra identidad ('{id_str(rid_b)}')")
    T = transcript(hello, rid_b, rnonce, rpub)
    if not verificar(peer_pub, b"P1-RESP", T, rsig):
        raise HandshakeError(f"[ALERTA] {peer_id} NO demostro ser quien dice (firma invalida): posible MITM")
    sock.sendall(firmar(priv, b"P1-INIT", T))
    if recv_exact(sock, 1) != b"\x01":
        raise HandshakeError(f"{peer_id} rechazo mi firma (¿mi clave publica no coincide en su authorized.txt?)")
    z = dh_secreto(x, rpub)
    k_i2r, k_r2i, sid = derivar(z, nonce, rnonce, T)
    sock.settimeout(None)
    return Sesion(sock, my_id, peer_id, k_tx=k_i2r, k_rx=k_r2i, sid=sid, rol="iniciador",
                  z=z, dh_pub=y, t_hs=time.perf_counter() - t0, verbose=verbose)


def handshake_respondedor(sock, my_id, priv, autorizados, addr="?", verbose=False):
    """Servidor <- cliente. Devuelve una Sesion o lanza HandshakeError."""
    sock.settimeout(TIMEOUT_HS)
    t0 = time.perf_counter()
    hello = recv_exact(sock, HELLO.size)
    magic, rid_b, rnonce, rpub = HELLO.unpack(hello)
    if magic != MAGIC:
        raise HandshakeError(f"[ALERTA] datos sin protocolo valido desde {addr}: descartado")
    rid = id_str(rid_b)
    if rid not in autorizados or rid == my_id:
        sock.sendall(HELLOREPLY.pack(0, b"", b"", b"", b""))
        raise HandshakeError(f"[ALERTA] dispositivo NO AUTORIZADO '{rid}' ({addr}): rechazado")
    x, y = dh_par()
    nonce = os.urandom(16)
    my_b = id_bytes(my_id)
    T = transcript(hello, my_b, nonce, y)
    sock.sendall(HELLOREPLY.pack(1, my_b, nonce, y, firmar(priv, b"P1-RESP", T)))
    try:
        fin = recv_exact(sock, 64)
    except (OSError, ConnectionError):
        raise HandshakeError(f"[ALERTA] '{rid}' ({addr}) abandono el handshake sin autenticarse")
    if not verificar(autorizados[rid], b"P1-INIT", T, fin):
        sock.sendall(b"\x00")
        raise HandshakeError(f"[ALERTA] suplantacion detectada: '{rid}' ({addr}) no posee la clave "
                             f"privada de ese dispositivo (firma invalida)")
    try:
        z = dh_secreto(x, rpub)
    except HandshakeError:
        sock.sendall(b"\x00")
        raise
    sock.sendall(b"\x01")
    k_i2r, k_r2i, sid = derivar(z, rnonce, nonce, T)
    sock.settimeout(None)
    return Sesion(sock, my_id, rid, k_tx=k_r2i, k_rx=k_i2r, sid=sid, rol="respondedor",
                  z=z, dh_pub=y, t_hs=time.perf_counter() - t0, verbose=verbose)


# =====================================================================
#  Sesion establecida: envio / recepcion
# =====================================================================
class Sesion:
    def __init__(self, sock, my_id, peer_id, k_tx, k_rx, sid, rol, z, dh_pub, t_hs, verbose):
        self.sock = sock
        self.my_id = my_id
        self.peer_id = peer_id
        self.k_tx, self.k_rx, self.sid = k_tx, k_rx, sid
        self.rol = rol
        self.verbose = verbose
        self.tx_seq = 0
        self.rx_last = 0
        self.ultimo_frame = None
        self.pend = {}                       # seq -> instante de envio (para el RTT)
        self.activa = True
        self.lock_tx = threading.Lock()      # envio y k_tx se tocan desde varios hilos
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError:
            pass
        print(f"[+] Sesion segura con {peer_id} | SID 0x{sid:08X} | handshake {t_hs * 1000:.1f} ms")
        if verbose:
            print("Mi clave publica DH (256 bytes):\n" + hx(dh_pub))
            print("Secreto compartido DH (z):\n" + hx(z))
            print("CK_tx inicial:\n" + hx(k_tx))
            print("CK_rx inicial:\n" + hx(k_rx))

    def cerrar(self):
        self.activa = False
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass

    # ---------- envio ----------
    def enviar(self, texto, modo=1):
        """modo 1 normal; 2 altera ciphertext; 3 altera TAG; 5 falsifica ID_S; 6 clave forjada."""
        data = texto.encode() if isinstance(texto, str) else bytes(texto)
        if len(data) > MAX_CIPHER:
            # recortar un JSON lo corromperia: mejor fallar de forma explicita
            raise ValueError(f"mensaje de {len(data)} B: el maximo es {MAX_CIPHER} B")
        with self.lock_tx:
            self.tx_seq += 1
            seq = self.tx_seq
            nonce = os.urandom(8) + struct.pack("<I", seq)
            sender = self.my_id
            mk = msg_key(self.k_tx)
            clave = mk
            if modo == 6:
                clave = os.urandom(32)
                print(f"[TEST -> {self.peer_id}] Inyectando paquete forjado (clave que no es la de la sesion)")
            if modo == 5:
                sender = "Intruso"
                print(f"[TEST -> {self.peer_id}] Falsificando el remitente (ID_S)")
            hdr = HEADER.pack(id_bytes(sender), self.sid, seq, nonce, len(data))
            ctag = AESGCM(clave).encrypt(nonce, data, hdr)
            ct, tag = ctag[:-16], ctag[-16:]
            if modo == 2:
                ct = bytes([ct[0] ^ 0xFF]) + ct[1:] if ct else ct
                print(f"[TEST -> {self.peer_id}] Alterando 1 byte del ciphertext")
            if modo == 3:
                tag = bytes([tag[0] ^ 0xFF]) + tag[1:]
                print(f"[TEST -> {self.peer_id}] Alterando 1 byte del TAG")
            frame = bytes([FRAME_DATA]) + hdr + ct + tag
            if self.verbose:
                print(f"\n>>> ENVIANDO a {self.peer_id} (SEQ {seq}, {len(data)} B) <<<")
                print("Plaintext:\n" + data.decode(errors="replace"))
                print("CK_tx (estado de la cadena):\n" + hx(self.k_tx))
                print("MK (clave de este mensaje):\n" + hx(clave))
                print("Ciphertext:\n" + hx(ct))
            self.pend[seq] = time.perf_counter()
            self.sock.sendall(frame)
            # La cadena avanza SOLO con mensajes normales. Los de prueba (2, 3, 5, 6)
            # los rechaza el receptor sin avanzar su CK_rx; si el emisor avanzara
            # su CK_tx, la sesion quedaria desincronizada para siempre.
            if modo == 1:
                self.ultimo_frame = frame
                self.k_tx = next_chain(self.k_tx)
                if self.verbose:
                    print("[CADENA] Nuevo CK_tx:\n" + hx(self.k_tx))
        return seq

    def replay(self):
        if not self.ultimo_frame:
            print(f"[TEST -> {self.peer_id}] Primero envia un mensaje normal.")
            return
        print(f"[TEST -> {self.peer_id}] REPLAY: retransmitiendo el ultimo paquete aceptado, identico.")
        with self.lock_tx:
            self.sock.sendall(self.ultimo_frame)

    # ---------- recepcion ----------
    def leer_frame(self):
        """
        Bloquea hasta leer una trama completa.
        Devuelve ("data", ok, seq, plaintext_o_motivo) o ("ack", seq, status, rtt_ms).
        Lanza ConnectionError/OSError si la conexion se cae o la trama es invalida.
        """
        tipo = recv_exact(self.sock, 1)[0]
        if tipo == FRAME_ACK:
            seq, status = ACK.unpack(recv_exact(self.sock, ACK.size))
            t = self.pend.pop(seq, None)
            rtt = (time.perf_counter() - t) * 1000 if t else None
            return ("ack", seq, status, rtt)
        if tipo != FRAME_DATA:
            raise ConnectionError(f"trama desconocida 0x{tipo:02X}")
        hdr = recv_exact(self.sock, HEADER.size)
        sender_b, sid, seq, nonce, clen = HEADER.unpack(hdr)
        if clen > MAX_CIPHER:
            raise ConnectionError(f"longitud invalida ({clen} B)")
        ct = recv_exact(self.sock, clen)
        tag = recv_exact(self.sock, 16)
        sender = id_str(sender_b)
        mk = msg_key(self.k_rx)
        if self.verbose:
            print(f"\n<<< TRAMA de {sender} (SEQ {seq}) >>>")
            print("Ciphertext:\n" + hx(ct))
            print("CK_rx (estado de la cadena):\n" + hx(self.k_rx))
            print("MK (clave de este mensaje):\n" + hx(mk))
        motivo, pt = None, None
        if sender != self.peer_id:
            motivo = "Remitente (ID_S) no coincide con la sesion"
        elif sid != self.sid:
            motivo = "Session ID (SID) invalido"
        elif seq <= self.rx_last:
            motivo = "REPLAY detectado (SEQ repetido o antiguo)"
        else:
            try:
                pt = AESGCM(mk).decrypt(nonce, ct + tag, hdr)
            except Exception:
                motivo = "Fallo de autenticacion GCM (ciphertext, cabecera o TAG alterados, o paquete forjado)"
        with self.lock_tx:
            self.sock.sendall(bytes([FRAME_ACK]) + ACK.pack(seq, 0 if motivo else 1))
        if motivo:
            return ("data", False, seq, motivo)
        self.rx_last = seq
        self.k_rx = next_chain(self.k_rx)      # avanza SOLO tras autenticar el mensaje
        if self.verbose:
            print("[CADENA] Nuevo CK_rx:\n" + hx(self.k_rx))
        return ("data", True, seq, pt)


# =====================================================================
#  Mensajes de aplicacion (JSON)
# =====================================================================
def _a_json(o):
    """Permite enviar arrays y escalares de numpy sin que el modulo dependa de numpy."""
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} no es serializable a JSON")


def codificar(msg):
    """dict -> bytes JSON compactos. Los floats se serializan sin perder precision."""
    if not isinstance(msg, dict) or "type" not in msg:
        raise ValueError('un mensaje debe ser un dict con un campo "type"')
    return json.dumps(msg, default=_a_json, separators=(",", ":"), ensure_ascii=False).encode()


def decodificar(data):
    """bytes -> dict. Si no es un JSON con "type", lo entrega como {"type": "text", "text": ...}."""
    texto = data.decode(errors="replace")
    try:
        msg = json.loads(texto)
    except ValueError:
        return {"type": "text", "text": texto}
    if isinstance(msg, dict) and "type" in msg:
        return msg
    return {"type": "json", "data": msg}
