"""Reuse the existing verified Nav2 bridge; stop by parking/unfolding, not reset."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'experiments/mobile_vla/scripts'))
import run_nav2 as nav
original=nav.call
# --nav-only normally requests stop. Here it transfers ownership to the arm driver.
def call(path,op,**kwargs):return original(path,'dock_for_arm' if op=='stop' else op,**kwargs)
nav.call=call
if '--nav-only' not in sys.argv:sys.argv.append('--nav-only')
if '--wall-clock' not in sys.argv:sys.argv.append('--wall-clock')
nav.main()
