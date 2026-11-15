"""Shared camera orientation for preview and recorded observations."""
import cv2

DEFAULT_ROTATION = 270


def rotate_frame(frame, clockwise_degrees=DEFAULT_ROTATION):
    if clockwise_degrees == 0:
        return frame
    rotations = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if clockwise_degrees not in rotations:
        raise ValueError("Rotation must be 0, 90, 180 or 270 clockwise degrees")
    return cv2.rotate(frame, rotations[clockwise_degrees])
