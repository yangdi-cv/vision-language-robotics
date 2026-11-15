"""Small deadline-bounded Unix-socket transport; no physics or policy logic."""

import json
import socket


def call(path, op, timeout=120, **payload):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(str(path))
        s.sendall((json.dumps({"op": op, **payload}) + "\n").encode())
        f = s.makefile("rb")
        line = f.readline(8 * 1024 * 1024)
        if not line:
            raise RuntimeError("Simulator closed connection")
        r = json.loads(line)
        if not r.get("ok"):
            raise RuntimeError(r.get("error", "Simulator error"))
        return r
