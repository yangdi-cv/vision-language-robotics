from huggingface_hub import snapshot_download
from pathlib import Path
import json
p=snapshot_download('Qwen/Qwen3-VL-2B-Instruct',revision='89644892e4d85e24eaac8bacfd4f463576704203',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','LICENSE','README.md'],max_workers=2)
Path(__file__).resolve().parents[1].joinpath('MODEL_2B.json').write_text(json.dumps({'id':'Qwen/Qwen3-VL-2B-Instruct','revision':'89644892e4d85e24eaac8bacfd4f463576704203','snapshot':p},indent=2))
print(p,flush=True)
