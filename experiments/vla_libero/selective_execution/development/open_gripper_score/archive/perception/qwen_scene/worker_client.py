"""Deadline-aware newline JSON transport for isolated model workers."""

import json, os, selectors, signal, time


class JsonWorker:
    def __init__(self, process):
        self.process = process
        self.buffer = b""

    def read(self, timeout):
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as sel:
            sel.register(self.process.stdout.fileno(), selectors.EVENT_READ)
            while True:
                if b"\n" in self.buffer:
                    line, self.buffer = self.buffer.split(b"\n", 1)
                    try:
                        return json.loads(line)
                    except (ValueError, UnicodeError) as exc:
                        raise RuntimeError("Worker returned malformed JSON") from exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Worker response exceeded {timeout}s")
                if not sel.select(remaining):
                    raise TimeoutError(f"Worker response exceeded {timeout}s")
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError(
                        f"Worker closed output, exit={self.process.poll()}"
                    )
                self.buffer += chunk
                if len(self.buffer) > 4 * 1024 * 1024:
                    raise RuntimeError("Worker response exceeds4MiB")

    def close(self):
        p = self.process
        if p.poll() is None:
            try:
                p.stdin.write('{"stop":true}\n')
                p.stdin.flush()
                p.wait(timeout=2)
            except (
                BrokenPipeError,
                OSError,
                TimeoutError,
                __import__("subprocess").TimeoutExpired,
            ):
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    p.wait(timeout=2)
                except __import__("subprocess").TimeoutExpired:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    p.wait(timeout=2)
        for stream in (p.stdin, p.stdout):
            if stream:
                stream.close()


def read_json_line(process, timeout=90):
    if not hasattr(process, "_json_client"):
        process._json_client = JsonWorker(process)
    try:
        return process._json_client.read(timeout)
    except Exception:
        process._json_client.close()
        raise


def close_json_worker(process):
    if not hasattr(process, "_json_client"):
        process._json_client = JsonWorker(process)
    process._json_client.close()
