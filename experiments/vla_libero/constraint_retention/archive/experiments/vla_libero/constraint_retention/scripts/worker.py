"""Reuse the established worker; no new model, adapter, or prompting here."""
from pathlib import Path
import runpy

if __name__=='__main__':
    source=Path(__file__).resolve().parents[2]/'selective_execution/scripts/worker.py'
    runpy.run_path(str(source),run_name='__main__')
