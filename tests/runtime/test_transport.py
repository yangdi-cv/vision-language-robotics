import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from vla_runtime.transport import JsonWorker, call, decode_object, read_json_line, stop_process


class WorkerTransport(unittest.TestCase):
    def spawn(self, code, session=True):
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=session,
        )
        self.addCleanup(stop_process, process)
        self.addCleanup(lambda: process.stdin.close() if not process.stdin.closed else None)
        self.addCleanup(lambda: process.stdout.close() if not process.stdout.closed else None)
        return process

    def test_multiple_frames_remain_buffered(self):
        process = self.spawn('print(\'{"a":1}\\n{"a":2}\',flush=True)')
        client = JsonWorker(process)
        self.assertEqual(client.read(2), {"a": 1})
        self.assertEqual(client.read(2), {"a": 2})

    def test_split_frame(self):
        process = self.spawn(
            'import sys,time;sys.stdout.write(\'{"ok":\');sys.stdout.flush();time.sleep(.05);print("true}")'
        )
        self.assertTrue(read_json_line(process, 2)["ok"])

    def test_timeout_stops_worker_and_does_not_signal_parent_group(self):
        process = self.spawn("import time;time.sleep(10)", session=False)
        with self.assertRaises(TimeoutError):
            read_json_line(process, 0.05)
        self.assertIsNotNone(process.poll())
        self.assertEqual(os.getpgrp(), os.getpgid(os.getpid()))

    def test_malformed_frame_stops_worker(self):
        process = self.spawn('import time;print("not json",flush=True);time.sleep(10)')
        with self.assertRaises(ValueError):
            read_json_line(process, 2)
        self.assertIsNotNone(process.poll())

    def test_eof_is_not_a_response(self):
        process = self.spawn("pass")
        with self.assertRaises(RuntimeError):
            read_json_line(process, 2)

    def test_close_accepts_stop_message(self):
        process = self.spawn(
            'import sys,json; print("{}",flush=True); assert json.loads(sys.stdin.readline())["stop"]'
        )
        client = JsonWorker(process)
        client.read(2)
        client.close()
        self.assertEqual(process.returncode, 0)

    def test_nonfinite_and_nonobject_frames(self):
        for data in (b"[]", b'{"x":NaN}', b'{"x":Infinity}', b"\xff"):
            with self.assertRaises(ValueError):
                decode_object(data)


class RPCTransport(unittest.TestCase):
    def exchange(self, response, timeout=1, delay=0):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "rpc.sock")
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(path)
                server.listen(1)

                def respond():
                    connection, _ = server.accept()
                    with connection:
                        connection.recv(4096)
                        time.sleep(delay)
                        try:
                            connection.sendall(response)
                        except BrokenPipeError:
                            pass

                thread = threading.Thread(target=respond)
                thread.start()
                try:
                    return call(path, "state", timeout=timeout)
                finally:
                    thread.join(timeout=2)

    def test_rpc_success(self):
        self.assertEqual(self.exchange(b'{"ok":true,"value":3}\n')["value"], 3)

    def test_rpc_failure_and_bad_framing(self):
        for data in (b'{"ok":false,"error":"bad state"}\n', b'{"ok":"true"}\n', b"{}"):
            with self.assertRaises(RuntimeError):
                self.exchange(data)

    def test_rpc_total_deadline(self):
        with self.assertRaises(TimeoutError):
            self.exchange(b'{"ok":true}\n', timeout=0.02, delay=0.1)

    def test_rpc_rejects_nan_request_before_connect(self):
        with self.assertRaises(ValueError):
            call("/missing", "goal", value=float("nan"))
