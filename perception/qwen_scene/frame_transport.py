"""One shared RGB frame with locked metadata; readers never accumulate a queue."""

import json, struct, fcntl
from pathlib import Path
from multiprocessing import shared_memory, resource_tracker
import numpy as np
from PIL import Image

HEADER = struct.Struct("<QQIII")
OFFSET = 64


class SharedFrame:
    def __init__(self, session, create=False, capacity=1920 * 1080 * 3):
        self.session = Path(session)
        self.owner = create
        self.lock = (self.session / "frame.lock").open("a+b")
        if create:
            self.memory = shared_memory.SharedMemory(
                create=True, size=OFFSET + capacity
            )
            self.memory.buf[:OFFSET] = bytes(OFFSET)
            temp = self.session / "frame_transport.tmp"
            temp.write_text(
                json.dumps({"name": self.memory.name, "capacity": capacity})
            )
            temp.replace(self.session / "frame_transport.json")
        else:
            d = json.loads((self.session / "frame_transport.json").read_text())
            self.memory = shared_memory.SharedMemory(name=d["name"])
            resource_tracker.unregister(self.memory._name, "shared_memory")

    def write(self, rgb, frame_id, captured_ns, rotation=270):
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w, c = rgb.shape
        if c != 3 or rgb.nbytes > len(self.memory.buf) - OFFSET:
            raise ValueError("Unsupported frame shape")
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        try:
            self.memory.buf[OFFSET : OFFSET + rgb.nbytes] = rgb.tobytes()
            HEADER.pack_into(
                self.memory.buf, 0, frame_id + 1, captured_ns, rotation, h, w
            )
        finally:
            fcntl.flock(self.lock, fcntl.LOCK_UN)

    def read(self):
        fcntl.flock(self.lock, fcntl.LOCK_SH)
        try:
            seq, stamp, rotation, h, w = HEADER.unpack_from(self.memory.buf)
            if not seq:
                raise FileNotFoundError("Waiting for first frame")
            rgb = (
                np.frombuffer(
                    self.memory.buf, dtype=np.uint8, count=h * w * 3, offset=OFFSET
                )
                .reshape(h, w, 3)
                .copy()
            )
        finally:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
        return Image.fromarray(rgb), dict(
            frame_id=str(seq - 1), captured_ns=stamp, rotation=rotation
        )

    def close(self):
        self.memory.close()
        self.lock.close()
        if self.owner:
            try:
                self.memory.unlink()
            except FileNotFoundError:
                pass


_readers = {}


def read_latest_image(session):
    session = Path(session)
    if (session / "frame_transport.json").exists():
        key = str(session.resolve())
        if key not in _readers:
            _readers[key] = SharedFrame(session)
        return _readers[key].read()
    with Image.open(session / "latest.png") as image:
        return image.convert("RGB"), {
            key: image.info[key] for key in ("frame_id", "captured_ns", "rotation")
        }
