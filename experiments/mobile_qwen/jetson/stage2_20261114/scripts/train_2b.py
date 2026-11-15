"""Run the unchanged 4B training recipe with only the model source changed."""
from pathlib import Path
import hashlib
import json
import sys
from runtime import Runtime,Q,STAGE
sys.path.insert(0,str(Q/'affordance'))
import train_lora as original
model=json.loads((STAGE/'MODEL_2B.json').read_text())
h=hashlib.sha256()
with (Path(model['snapshot'])/'model.safetensors').open('rb') as stream:
    for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
assert h.hexdigest()=='7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0', 'Model SHA256 mismatch'
model['weight_sha256']=h.hexdigest()
original.SceneEngine=lambda:Runtime(model['snapshot'],revision=model['revision'])
# Record the exact training implementation and reject any split/selection mismatch.
old=json.loads((Q/'affordance/lora_v1/protocol.json').read_text())
out=STAGE/'lora_2b'
sys.argv=['train_2b.py',str(Q/'affordance/data_train_v1'),str(out),'--steps','600','--accum','4','--lr','0.0001','--eval-every','150','--val-inits','13-16']
original.main()
new=json.loads((out/'protocol.json').read_text())
assert new==old, 'Training recipe/split diverged from 4B'
(out/'MODEL_SOURCE.json').write_text(json.dumps({**model,'recipe_sha256':hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),'same_protocol_as_4b':True},indent=2))
