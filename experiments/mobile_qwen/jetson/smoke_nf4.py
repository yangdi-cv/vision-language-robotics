"""Tiny software-stack/serialization probe, not a Qwen benchmark.
Run --prepare on PC; copy output directory and run --verify on Jetson.
Both invocations must be memory-limited systemd jobs.
"""
import argparse
import json
from pathlib import Path
import time

p=argparse.ArgumentParser()
p.add_argument('mode', choices=['prepare','verify'])
p.add_argument('directory', type=Path)
a=p.parse_args()
import torch
import torchvision
import bitsandbytes as bnb
from transformers import LlamaConfig,LlamaForCausalLM,BitsAndBytesConfig
from peft import LoraConfig,PeftModel,get_peft_model

assert torch.cuda.is_available(), 'CUDA unavailable'
torch.manual_seed(123)
result={'torch':torch.__version__,'torchvision':torchvision.__version__, 'bitsandbytes':bnb.__version__,
        'gpu':torch.cuda.get_device_name(), 'capability':torch.cuda.get_device_capability(),
        'compiled_cuda':torch.version.cuda,'mode':a.mode}
# Independent CUDA and vision extension probes.
x=torch.randn(64,64,device='cuda',dtype=torch.float16)
assert torch.isfinite(x@x).all()
boxes=torch.tensor([[0.,0.,1.,1.],[0.,0.,1.,1.]],device='cuda')
assert torchvision.ops.nms(boxes,torch.tensor([.9,.8],device='cuda'),.5).numel()==1
layer=bnb.nn.Linear4bit(128,128,bias=False,quant_type='nf4',compress_statistics=True,compute_dtype=torch.float16)
layer=layer.to('cuda')
y=layer(torch.randn(2,128,device='cuda',dtype=torch.float16))
assert torch.isfinite(y).all()
result['nf4_matmul_shape']=list(y.shape)
del layer,x,y
ids=torch.tensor([[1,5,8,12,2]],device='cuda')
if a.mode=='prepare':
    a.directory.mkdir(parents=True,exist_ok=False)
    cfg=LlamaConfig(vocab_size=256,hidden_size=128,intermediate_size=256,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=64)
    tiny=LlamaForCausalLM(cfg).half()
    tiny.save_pretrained(a.directory/'fp16_source')
    del tiny
    model=LlamaForCausalLM.from_pretrained(a.directory/'fp16_source',device_map={'':'cuda:0'},quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16),attn_implementation='eager')
    model.eval()
    with torch.no_grad(): base=model(ids).logits.float().cpu()
    model.save_pretrained(a.directory/'nf4')
    model=get_peft_model(model,LoraConfig(r=4,lora_alpha=8,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    with torch.no_grad():
        for name,param in model.named_parameters():
            if 'lora_A' in name: param.fill_(.03)
            if 'lora_B' in name: param.fill_(.02)
    model.eval()
    with torch.no_grad(): adapted=model(ids).logits.float().cpu()
    model.save_pretrained(a.directory/'adapter')
    assert (adapted-base).abs().max().item()>1e-5
    torch.save({'base':base,'adapted':adapted},a.directory/'reference.pt')
    result['adapter_output_change']=float((adapted-base).abs().max())
else:
    model=LlamaForCausalLM.from_pretrained(a.directory/'nf4',device_map={'':'cuda:0'},attn_implementation='eager')
    assert any(isinstance(m,bnb.nn.Linear4bit) for m in model.modules()), 'NF4 was not restored'
    ref=torch.load(a.directory/'reference.pt',map_location='cpu',weights_only=True)
    model.eval()
    with torch.no_grad(): base=model(ids).logits.float().cpu()
    model=PeftModel.from_pretrained(model,a.directory/'adapter')
    model.eval()
    with torch.no_grad(): adapted=model(ids).logits.float().cpu()
    result['base_max_abs_error']=float((base-ref['base']).abs().max())
    result['adapter_max_abs_error']=float((adapted-ref['adapted']).abs().max())
    result['adapter_output_change']=float((adapted-base).abs().max())
    assert result['base_max_abs_error']<.03 and result['adapter_max_abs_error']<.03
    assert result['adapter_output_change']>1e-5
result['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
result['passed']=True
(a.directory/(a.mode+'_result.json')).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
