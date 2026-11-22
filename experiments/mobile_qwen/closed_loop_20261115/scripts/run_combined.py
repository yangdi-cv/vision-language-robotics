"""Compatibility entrypoint: shared orchestration with policy handover enabled."""

import sys
from run_cases import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--task", "combined"]
    raise SystemExit(main())
