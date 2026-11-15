import json
from pathlib import Path
import torch
import torchvision
import bitsandbytes as bnb
import transformers
import peft
r={'torch':torch.__version__,'torchvision':torchvision.__version__, 'bitsandbytes':bnb.__version__, 'transformers':transformers.__version__,'peft':peft.__version__,'cuda_available':torch.cuda.is_available()}
assert r['cuda_available']
r['gpu']=torch.cuda.get_device_name(); r['capability']=torch.cuda.get_device_capability()
x=torch.randn(128,128,device='cuda',dtype=torch.float16); y=x@x
assert torch.isfinite(y).all()
layer=bnb.nn.Linear4bit(128,128,bias=False,compute_dtype=torch.float16,quant_type='nf4',compress_statistics=True).to('cuda')
z=layer(x[:2]); torch.cuda.synchronize()
assert torch.isfinite(z).all()
r['nf4_shape']=list(z.shape);r['passed']=True
Path('/home/csip/ali-jetson/deployment_20261114/stack_result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
