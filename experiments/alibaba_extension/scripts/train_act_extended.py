"""Compatibility CLI for the shared ALOHA implementation."""
from pathlib import Path
import runpy,sys
p=Path(__file__).resolve().parents[3]/'experiments/alibaba_pc_revision/scripts'
sys.path.insert(0,str(p))
runpy.run_path(str(p/'public_act_trainer.py'),run_name='__main__')
