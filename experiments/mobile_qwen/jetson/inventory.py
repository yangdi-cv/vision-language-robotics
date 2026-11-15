"""Read-only inventory; does not import/load models or install dependencies."""
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

def command(argv):
    p = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {'returncode':p.returncode,'output':p.stdout.strip()}

def read(path):
    p=Path(path)
    return p.read_text().strip('\x00\n') if p.exists() else None

result={
 'captured_unix':time.time(), 'architecture':platform.machine(),'python':sys.version,'executable':sys.executable,
 'model':read('/proc/device-tree/model'),'l4t':read('/etc/nv_tegra_release'),
 'meminfo':read('/proc/meminfo'),
 'jetpack':command(['dpkg-query','-W','nvidia-jetpack','nvidia-l4t-core']),
 'cuda':json.loads(read('/usr/local/cuda/version.json') or '{}'),
 'disk':command(['df','-h','/']),
 'packages':{x:{'installed':importlib.util.find_spec(x) is not None} for x in ['torch','transformers','bitsandbytes','peft']},
 'cuda_tensor_test':'not run: torch not installed; package installation requires next phase',
}
print(json.dumps(result,ensure_ascii=False,indent=2))
