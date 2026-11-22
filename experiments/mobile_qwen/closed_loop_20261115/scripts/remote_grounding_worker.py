"""JSON-lines adapter for the SSH-forwarded vision service (no model loading)."""

from __future__ import annotations

import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

from vla_runtime.config import load_config
from vla_runtime.records import error_record, logger
from vla_runtime.schemas import grounding


def main() -> int:
    config = load_config()
    url = f"http://127.0.0.1:{int(config.remote('local_port'))}"
    log = logger(__name__)
    with urllib.request.urlopen(url, timeout=10) as response:
        if json.load(response).get("ready") is not True:
            raise RuntimeError("Vision service is not ready")
    print(json.dumps({"ready": True, "device": "remote Jetson via SSH"}), flush=True)
    for line in sys.stdin:
        try:
            if len(line) > 16384:
                raise ValueError("Worker request too large")
            request = json.loads(line)
            if request.get("stop") is True:
                break
            path = Path(request["image"])
            if not 0 < path.stat().st_size <= 1024 * 1024:
                raise ValueError("Image must be between 1 byte and 1 MiB")
            payload = {
                "mode": request["mode"],
                "image_base64": base64.b64encode(path.read_bytes()).decode(),
            }
            start = time.monotonic()
            query = urllib.request.Request(
                url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(
                query, timeout=config.limit("worker_response_seconds")
            ) as response:
                raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError("HTTP response too large")
            result = json.loads(raw)
            grounding(result)
            result["client_wall_seconds"] = time.monotonic() - start
        except Exception as exc:
            log.exception("Remote grounding failed")
            result = {"ok": False, **error_record(exc)}
        print(json.dumps(result, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
