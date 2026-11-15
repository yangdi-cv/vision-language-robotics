"""Reuse the unchanged prior physics rescore, redirect only result root."""
import importlib.util
from pathlib import Path
S=Path(__file__).resolve().parents[1];V=S.parent
spec=importlib.util.spec_from_file_location('original_physics_audit',V/'skill_bootstrap/scripts/audit_policy.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
if __name__=='__main__':
 audit.S=S
 audit.main('development',('base','cag15'),'PHYSICS_AUDIT.json')
