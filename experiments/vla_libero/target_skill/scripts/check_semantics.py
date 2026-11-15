"""Check the pinned upstream runtime contraction, not a presumed A@B formula."""
import json
from pathlib import Path
import jax,jax.numpy as jnp
import numpy as np
from openpi.models import lora
S=Path(__file__).resolve().parents[1]
def main():
 rng=np.random.default_rng(6);n,h,d,r=3,4,5,2
 x=rng.normal(size=(1,2,n,h)).astype(np.float32);w=rng.normal(size=(n,h,d)).astype(np.float32);a=rng.normal(size=(n,h,r)).astype(np.float32);b=rng.normal(size=(n,r,d)).astype(np.float32)
 module=lora.Einsum(shape=w.shape,lora_config=lora.LoRAConfig(rank=r,alpha=r))
 params={'params':{'w':jnp.array(w),'lora_a':jnp.array(a),'lora_b':jnp.array(b)}}
 actual=np.asarray(module.apply(params,'BTNH,NHD->BTD',jnp.array(x)))
 naive=np.einsum('btnh,nhd->btd',x,w+np.matmul(a,b))
 corrected=np.einsum('btnh,nhd->btd',x,w+np.matmul(a,b.sum(axis=0,keepdims=True)))
 result=dict(equation='BTNH,NHD->BTD',naive_max_abs=float(abs(actual-naive).max()),corrected_max_abs=float(abs(actual-corrected).max()),passed=bool(np.allclose(actual,corrected,atol=2e-5,rtol=1e-5)),source='pinned openpi.models.lora.Einsum.apply, float32')
 assert result['passed'] and result['naive_max_abs']>.01,result
 (S/'SEMANTIC_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
