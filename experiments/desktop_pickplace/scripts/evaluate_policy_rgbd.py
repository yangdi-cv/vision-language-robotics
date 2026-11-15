"""Compatibility entry for RGB-D coordinate-conditioned policy evaluation."""

from pathlib import Path
import runpy, sys

if "--target-position" not in sys.argv:
    sys.argv.append("--target-position")
runpy.run_path(str(Path(__file__).with_name("evaluate_policy.py")), run_name="__main__")
