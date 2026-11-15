"""Latest-frame visual overlays, separate from Qwen and robot control."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from continuous_worker import publish
from vision_engine import VisionEngine
from frame_transport import read_latest_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('session', type=Path)
    parser.add_argument('--max-results', type=int, default=0)
    args = parser.parse_args()
    status_path = args.session / 'vision_status.json'
    publish(status_path, {'state': 'loading'})
    try:
        engine = VisionEngine()
        previous = None
        count = 0
        with (args.session/'vision_timing.csv').open('w',newline='') as log:
            writer = csv.writer(log)
            writer.writerow(['frame_id','captured_ns','completed_ns','inference_seconds','input_age_ms','boxes'])
            while not (args.session/'stop').exists():
                start = time.monotonic()
                try:
                    image,meta=read_latest_image(args.session)
                    frame_id=meta['frame_id'];captured_ns=int(meta['captured_ns'])
                except (FileNotFoundError, OSError, KeyError):
                    time.sleep(.02)
                    continue
                if frame_id == previous or time.monotonic_ns()-captured_ns > 1_000_000_000:
                    time.sleep(.02)
                    continue
                previous = frame_id
                picked_ns = time.monotonic_ns()
                detected, depth_color, raw_depth, stats = engine.analyze(image)
                stats.update(frame_id=frame_id, captured_ns=captured_ns, completed_ns=time.monotonic_ns(),
                             input_age_ms=(picked_ns-captured_ns)/1e6)
                with (args.session/'vision.tmp').open('wb') as f:
                    np.savez_compressed(f, detection=detected, depth_color=depth_color, raw_depth=raw_depth,
                                        metadata=np.array(json.dumps(stats)))
                (args.session/'vision.tmp').replace(args.session/'latest_vision.npz')
                publish(status_path, {'state':'running','frame_id':frame_id})
                writer.writerow([frame_id,captured_ns,stats['completed_ns'],stats['inference_seconds'],stats['input_age_ms'],len(stats['objects'])])
                log.flush()
                count += 1
                if args.max_results and count >= args.max_results:
                    break
                time.sleep(max(0,.1-(time.monotonic()-start)))
        publish(status_path, {'state':'stopped','results':count})
    except Exception as exc:
        publish(status_path, {'state':'error','message':str(exc)})
        raise


if __name__ == '__main__':
    main()
