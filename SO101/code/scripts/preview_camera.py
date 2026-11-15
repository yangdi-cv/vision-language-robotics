"""Preview the SO-ARM101 USB camera on Windows. Press Q or Esc to quit."""

import sys
import os
import tkinter as tk

import cv2
from PIL import Image, ImageTk
from camera_view import DEFAULT_ROTATION, rotate_frame


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Preview a SO-ARM101 USB camera")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=DEFAULT_ROTATION,
                        help="Clockwise image rotation in degrees")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
    camera = cv2.VideoCapture(args.camera_index, backend)
    if not camera.isOpened():
        raise SystemExit(
            f"Cannot open camera {args.camera_index}. Close other camera apps and try again."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    root = tk.Tk()
    root.title(f"SO-ARM101 camera {args.camera_index} - {args.rotation} deg CW - Q to quit")
    frame_view = tk.Label(root)
    frame_view.pack()
    closed = False

    def close(_event=None) -> None:
        nonlocal closed
        if not closed:
            closed = True
            camera.release()
            root.destroy()

    def update_frame() -> None:
        if closed:
            return
        ok, frame = camera.read()
        if not ok:
            print("Camera stopped returning frames.", file=sys.stderr)
            close()
            return
        rgb = cv2.cvtColor(rotate_frame(frame, args.rotation), cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        frame_view.configure(image=photo)
        frame_view.image = photo
        root.after(30, update_frame)

    root.bind("q", close)
    root.bind("<Escape>", close)
    root.protocol("WM_DELETE_WINDOW", close)
    root.after(0, update_frame)
    if args.self_test:
        root.after(1000, close)

    print(f"Camera {args.camera_index} preview is open. Press Q or Esc to quit.")
    try:
        root.mainloop()
    finally:
        if not closed:
            close()


if __name__ == "__main__":
    main()
