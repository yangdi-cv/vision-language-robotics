"""Create a separate NF4 checkpoint without changing the source model or adapter."""
from pathlib import Path
import argparse,json,time
import torch,bitsandbytes as bnb
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration,BitsAndBytesConfig
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
model_id='Qwen/Qwen3-VL-4B-Instruct';rev='ebb281ec70b05090aa6165b016eac8ec08e71b17';t=time.perf_counter()
q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,llm_int8_skip_modules=['lm_head'])
m=Qwen3VLForConditionalGeneration.from_pretrained(model_id,revision=rev,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda:0'},quantization_config=q,attn_implementation='sdpa').eval()
n=sum(isinstance(v,bnb.nn.Linear4bit) for v in m.modules());assert n>0
m.save_pretrained(a.out,safe_serialization=True);AutoProcessor.from_pretrained(model_id,revision=rev,local_files_only=True).save_pretrained(a.out)
(a.out/'conversion.json').write_text(json.dumps({'model':model_id,'revision':rev,'quantized_linear_count':n,'torch':torch.__version__,'bitsandbytes':bnb.__version__,'seconds':time.perf_counter()-t,'adapter_merged':False,'quantization':'NF4, double quantization, BF16 compute; language and vision linear layers except lm_head'},indent=2));print('QUANTIZED',n,flush=True)
