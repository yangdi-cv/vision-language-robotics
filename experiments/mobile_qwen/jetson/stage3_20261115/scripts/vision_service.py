"""Localhost-only serial vision service. Start only under the documented memory limit."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
import signal
import subprocess
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from vla_runtime.records import Journal, atomic_json, error_record, logger
from vla_runtime.schemas import grounding


def main() -> int:
    from PIL import Image

    def terminate(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)

    stage = Path(__file__).resolve().parents[1]
    model_stage = stage.parent / "stage2_20261114"
    output = Path(os.environ.get("VLA_SERVICE_OUT", str(stage / "service_run")))
    output.mkdir(parents=True, exist_ok=False)
    requests = Journal(output / "requests.jsonl")
    log = logger(__name__)
    tegra = runtime = server = None
    status = 1
    with (output / "tegrastats.log").open("w") as tegra_log:
        try:
            # Named module loading avoids mutating sys.path for a retained model backend.
            spec = importlib.util.spec_from_file_location(
                "nf4_backend", model_stage / "scripts/runtime.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            tegra = subprocess.Popen(
                ["/usr/bin/tegrastats", "--interval", "100"],
                stdout=tegra_log,
                stderr=subprocess.STDOUT,
            )
            runtime = module.Runtime(
                model_stage / "nf4_2b",
                precision="nf4",
                adapter=model_stage / "adapter_2b",
                resolution=512,
            )
            atomic_json(
                output / "load.json",
                {"seconds": runtime.load_seconds, "after_load": runtime.after_load},
            )
            if len(Path("/proc/swaps").read_text().splitlines()) != 1:
                raise RuntimeError("Stable measurement requires swap disabled by administrator")
            runtime.samples.clear()
            runtime.samples.append({"time_unix": time.time(), **module.memory()})
            prompt = json.loads((stage / "input_manifest.json").read_text())["prompts"]["handle"]

            class Handler(BaseHTTPRequestHandler):
                timeout = 10

                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"ready":true}')

                def do_POST(self):
                    query = {}
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        if not 0 < length <= 2 * 1024 * 1024:
                            raise ValueError("Invalid request length")
                        raw_request = self.rfile.read(length)
                        if len(raw_request) != length:
                            raise ValueError("Truncated HTTP request")
                        query = json.loads(raw_request)
                        if not isinstance(query, dict) or query.get("mode") != "lora":
                            raise ValueError("Only the validated LoRA endpoint is enabled")
                        data = base64.b64decode(query["image_base64"], validate=True)
                        start = time.perf_counter()
                        with Image.open(io.BytesIO(data)) as image:
                            if image.width * image.height > 4096 * 4096:
                                raise ValueError("Image dimensions exceed limit")
                            result = runtime.generate(image.convert("RGB"), prompt, 40)
                        raw = result["raw"].strip()
                        if raw.startswith("```"):
                            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                        value = json.loads(raw)
                        value = value[0] if isinstance(value, list) else value
                        point = grounding({"ok": True, "point_2d": value["point_2d"]})["point_2d"]
                        result.update(
                            ok=True,
                            point_2d=point,
                            mode="lora",
                            server_wall_seconds=time.perf_counter() - start,
                            inference={
                                "adapter_enabled": True,
                                "model": "Qwen3-VL-2B NF4",
                                "device": "Jetson (power mode must be recorded separately)",
                                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                "input_png_sha256": hashlib.sha256(data).hexdigest(),
                            },
                        )
                    except Exception as exc:
                        log.exception("Vision request failed")
                        result = {"ok": False, **error_record(exc)}
                    result["phase"] = (
                        query.get("phase", "case") if isinstance(query, dict) else "invalid_request"
                    )
                    requests.append("request", **result)
                    payload = json.dumps(result, allow_nan=False).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

                def log_message(self, format, *args):
                    log.info(format, *args)

            server = HTTPServer(
                ("127.0.0.1", int(os.environ.get("VLA_SERVICE_PORT", "18888"))), Handler
            )
            log.info("Vision service ready")
            status = 0
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            log.exception("Vision service failed")
            atomic_json(output / "failure.json", error_record(exc))
            status = 1
        finally:
            try:
                if server:
                    server.server_close()
                if runtime:
                    report = runtime.report()
                    samples = report["stable_samples"]
                    report["global_swap_io_pages"] = (
                        {
                            key: max(x[key] for x in samples) - min(x[key] for x in samples)
                            for key in ["pswpin", "pswpout"]
                        }
                        if samples
                        else {}
                    )
                    report["valid_without_swap"] = (
                        bool(samples)
                        and report["stable_max_swap_bytes"] == 0
                        and not any(report["global_swap_io_pages"].values())
                    )
                    atomic_json(output / "resources.json", report)
            finally:
                requests.close()
                if tegra:
                    from vla_runtime.transport import stop_process

                    stop_process(tegra)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
