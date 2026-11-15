"""One inference at a time, reading only the latest camera frame."""

import argparse
import json
import time
from pathlib import Path
from engine import SceneEngine
from frame_transport import read_latest_image


def publish(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--max-results", type=int, default=0)
    args = parser.parse_args()
    status = args.session / "worker_status.json"
    publish(status, {"state": "loading"})
    try:
        engine = SceneEngine()
        publish(status, {"state": "ready"})
        previous = None
        count = 0
        while not (args.session / "stop").exists():
            try:
                image, meta = read_latest_image(args.session)
                frame_id = meta["frame_id"]
                captured_ns = int(meta["captured_ns"])
                rotation = int(meta["rotation"])
            except (FileNotFoundError, OSError, KeyError):
                time.sleep(0.02)
                continue
            if frame_id == previous:
                time.sleep(0.02)
                continue
            if time.monotonic_ns() - captured_ns > 1_000_000_000:
                publish(status, {"state": "waiting_camera"})
                time.sleep(0.1)
                continue
            previous = frame_id
            publish(status, {"state": "inferencing", "frame_id": frame_id})
            picked_ns = time.monotonic_ns()
            try:
                language = json.loads((args.session / "language.json").read_text()).get(
                    "language", "zh"
                )
            except (OSError, ValueError):
                language = "zh"
            result, directory = engine.analyze(
                image,
                args.session / "analyses",
                rotation,
                source=f"live frame {frame_id}",
                language=language,
                pixels_already_rotated=True,
            )
            result.update(
                frame_id=frame_id,
                captured_monotonic_ns=captured_ns,
                picked_monotonic_ns=picked_ns,
                completed_monotonic_ns=time.monotonic_ns(),
                input_age_ms=(picked_ns - captured_ns) / 1e6,
                output_path=str(directory),
            )
            publish(directory / "result.json", result)
            publish(args.session / "latest_result.json", result)
            count += 1
            print(
                f"result {count}: frame={frame_id}, inference={result['inference_seconds']:.2f}s, input age={result['input_age_ms']:.1f}ms",
                flush=True,
            )
            if args.max_results and count >= args.max_results:
                break
        publish(status, {"state": "stopped", "results": count})
    except Exception as exc:
        publish(status, {"state": "error", "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
