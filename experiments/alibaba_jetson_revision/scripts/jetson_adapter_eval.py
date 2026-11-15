"""Evaluate the project's own 4B goal adapter on the original held-out cases on Jetson."""
from pathlib import Path
import argparse,json,time,os,resource
from unittest.mock import patch
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
def save(n,v):(a.out/n).write_text(json.dumps(v,ensure_ascii=False,indent=2))
save('status.json',{'phase':'imports'})
configuration=os.environ.get('PYTORCH_CUDA_ALLOC_CONF','backend:native')
if 'max_split_size_mb' not in configuration and 'cudaMallocAsync' not in configuration:configuration+=',max_split_size_mb:512'
os.environ['PYTORCH_CUDA_ALLOC_CONF']=configuration
import torch,bitsandbytes as bnb
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
# Evict only this checkpoint's clean page-cache pages before reserving device memory.
for checkpoint in a.model.glob('*.safetensors'):
 fd=os.open(checkpoint,os.O_RDONLY)
 try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
 finally:os.close(fd)
pool=[torch.empty(800*1024*1024,dtype=torch.uint8,device='cuda') for _ in range(2)]+[torch.empty(64*1024*1024,dtype=torch.uint8,device='cuda') for _ in range(18)]
for block in pool:block.zero_()
del block;torch.cuda.synchronize();del pool
save('status.json',{'phase':'loading'});start=time.perf_counter()
try:
 from transformers import core_model_loading
except ImportError:
 core_model_loading=None
original_copy=core_model_loading._materialize_copy if core_model_loading is not None else None
def traced_copy(tensor,device,dtype):
 try:return original_copy(tensor,device,dtype)
 except BaseException:
  print('FAILED_ALLOCATION',getattr(tensor,'shape',None),str(dtype),'allocated',torch.cuda.memory_allocated(),'reserved',torch.cuda.memory_reserved(),flush=True)
  raise
if core_model_loading is not None:core_model_loading._materialize_copy=traced_copy
processor=AutoProcessor.from_pretrained(a.model,local_files_only=True)
with patch('transformers.modeling_utils.caching_allocator_warmup',lambda *args,**kwargs:None,create=True):
 model=Qwen3VLForConditionalGeneration.from_pretrained(a.model,dtype=torch.bfloat16,device_map={'':0},local_files_only=True).eval()
base_quantized=sum(isinstance(v,bnb.nn.Linear4bit) for v in model.modules());assert base_quantized>0
model=PeftModel.from_pretrained(model,a.adapter,is_trainable=False).eval()
layers=[v for v in model.modules() if hasattr(v,'lora_B')];assert layers and any(float(v.lora_B['default'].weight.detach().abs().sum())>0 for v in layers)
save('runtime.json',{'model_path':str(a.model),'adapter':str(a.adapter),'loading_s':time.perf_counter()-start,'quantized_linear_count':base_quantized,'lora_layers':len(layers),'adapter_merged':False,'active_adapters':model.active_adapters,'cuda_devices':sorted({str(v.device) for v in model.parameters()}),'hidden_size':model.config.text_config.hidden_size,'allocator_configuration':os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),'torch':torch.__version__,'bitsandbytes':bnb.__version__})
cases=json.loads((a.data/'cases.json').read_text());rows=[]
for c in cases:
 prompt=f"""桌面自主抓放任务，指令：{c['instruction']}
先看图：目标颜色是否存在，两个盒子分别是画面左侧和右侧。只有目标和目的盒子均明确，且不是否定指令时才能执行。
缺少目标、指令歧义或否定操作时，should_execute=false，target_color和destination均为null。
只输出JSON：{{"target_color":"red"或"blue"或"green"或null,"destination":"left"或"right"或null,"should_execute":true或false}}。"""
 image=Image.open(a.data/c['image']).convert('RGB')
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
 batch=processor.apply_chat_template([{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device)
 with torch.inference_mode():output=model.generate(**batch,max_new_tokens=64,do_sample=False)
 torch.cuda.synchronize();elapsed=time.perf_counter()-t;tokens=output[0][len(batch.input_ids[0]):];text=processor.decode(tokens,skip_special_tokens=True)
 try:pred=json.loads(text[text.index('{'):text.rindex('}')+1])
 except Exception:pred=None
 expected={k:c[k] for k in ['target_color','destination','should_execute']}
 correct=bool(pred is not None and all(pred.get(k)==v for k,v in expected.items()))
 item={'id':c['id'],'seconds':elapsed,'tokens':len(tokens),'raw':text,'parsed':pred,'expected':expected,'correct':correct,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'warmup':len(rows)==0}
 rows.append(item);save('results.json',rows);save('status.json',{'phase':'running','completed':len(rows)});print(json.dumps(item,ensure_ascii=False),flush=True)
save('status.json',{'phase':'complete','completed':len(rows),'correct':sum(x['correct'] for x in rows),'robot_commands_sent':0})
