"""Bounded newline-JSON transports, preserving existing worker wire format."""

from __future__ import annotations

import json
import io
import os
import selectors
import signal
import socket
import subprocess
import time
from typing import Any

MAX_FRAME = 4 * 1024 * 1024


def decode_object(line: bytes) -> dict[str, Any]:
    try:
        value = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Malformed JSON protocol frame") from exc
    if not isinstance(value, dict):
        raise ValueError("Protocol frame must be an object")
    return value


def call(path: Any, op: str, timeout: float = 120, **payload: Any) -> dict[str, Any]:
    if not isinstance(op, str) or not op or "op" in payload or timeout <= 0:
        raise ValueError("Invalid RPC request")
    request = (json.dumps({"op": op, **payload}, allow_nan=False) + "\n").encode()
    if len(request) > MAX_FRAME:
        raise ValueError("RPC request exceeds frame limit")
    deadline = time.monotonic() + timeout
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(path))
        connection.sendall(request)
        frame = bytearray()
        while b"\n" not in frame:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("RPC response deadline exceeded")
            connection.settimeout(remaining)
            chunk = connection.recv(min(65536, MAX_FRAME + 1 - len(frame)))
            if not chunk:
                raise RuntimeError("Simulator closed connection before newline")
            frame.extend(chunk)
            if len(frame) > MAX_FRAME:
                raise ValueError("RPC response exceeds frame limit")
        response = decode_object(bytes(frame).split(b"\n", 1)[0])
        if response.get("ok") is not True:
            raise RuntimeError(str(response.get("error", "Simulator error")))
        return response


def stop_process(process: subprocess.Popen[Any], grace: float = 2) -> None:
    """Only signal a group if this process owns it; never signal the caller's group."""
    if process.poll() is not None:
        return

    def send(sig: signal.Signals) -> None:
        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, sig)
            else:
                process.send_signal(sig)
        except ProcessLookupError:
            pass

    send(signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        send(signal.SIGKILL)
        process.wait(timeout=grace)


class JsonWorker:
    def __init__(self, process: subprocess.Popen[Any]):
        if process.stdout is None:
            raise ValueError("Worker stdout must be piped")
        self.process = process
        self.buffer = b""

    def read(self, timeout: float) -> dict[str, Any]:
        assert self.process.stdout is not None
        if timeout <= 0:
            raise ValueError("Worker timeout must be positive")
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout.fileno(), selectors.EVENT_READ)
            while True:
                if b"\n" in self.buffer:
                    line, self.buffer = self.buffer.split(b"\n", 1)
                    if len(line) > MAX_FRAME:
                        raise ValueError("Worker response exceeds frame limit")
                    return decode_object(line)
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("Worker response deadline exceeded")
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError(f"Worker closed output, exit={self.process.poll()}")
                self.buffer += chunk
                if len(self.buffer) > MAX_FRAME and b"\n" not in self.buffer:
                    raise ValueError("Worker response exceeds frame limit")

    def close(self) -> None:
        process = self.process
        if process.poll() is None and process.stdin is not None:
            try:
                stop = '{"stop":true}\n'
                process.stdin.write(
                    stop if isinstance(process.stdin, io.TextIOBase) else stop.encode()
                )
                process.stdin.flush()
                process.wait(timeout=2)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                stop_process(process)
        else:
            stop_process(process)
        for stream in (process.stdin, process.stdout):
            if stream:
                stream.close()


def read_json_line(process: subprocess.Popen[Any], timeout: float = 90) -> dict[str, Any]:
    client = getattr(process, "_json_client", None)
    if client is None:
        client = JsonWorker(process)
        setattr(process, "_json_client", client)
    try:
        result: dict[str, Any] = client.read(timeout)
        return result
    except Exception:
        client.close()
        raise


def close_json_worker(process: subprocess.Popen[Any]) -> None:
    client = getattr(process, "_json_client", None) or JsonWorker(process)
    client.close()
