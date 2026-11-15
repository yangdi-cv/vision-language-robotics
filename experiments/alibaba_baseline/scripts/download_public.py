"""Fetch the exact public inputs used in this experiment; no robot interface."""
from pathlib import Path
from huggingface_hub import snapshot_download
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'experiments/alibaba_baseline/data'
snapshot_download('lerobot/act_aloha_sim_transfer_cube_human', revision='ba73b2766f1371cdc133ca4efb97eb090d744625', local_dir=DATA/'official_act')
snapshot_download('lerobot/aloha_sim_transfer_cube_human', repo_type='dataset', revision='8f81f044cbc65d9c1efbe5a302e8f9f4c16a86ed', local_dir=DATA/'aloha_transfer', allow_patterns=['meta/*','data/*/*.parquet','videos/*/*/*.mp4','README.md'])
