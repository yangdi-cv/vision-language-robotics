"""Pinned Qwen BF16/NF4 runtime for the one-factor deployment comparison."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import resource
import threading
import time
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

Q = Path(__file__).resolve().parents[3]
STAGE = Path(__file__).resolve().parents[1]


def cgroup_path():
    path=Path('/proc/self/cgroup').read_text().strip().split('0::')[-1].strip()
    return Path('/sys/fs/cgroup') / path.lstrip('/')


def memory():
    cg=cgroup_path()
    def number(name):
        try: return int((cg/name).read_text())
        except (OSError,ValueError): return None
    status={}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:','VmHWM:','VmSwap:')):
            k,v=line.split(':',1); status[k]=int(v.split()[0])*1024
    vm={k:int(v) for k,v in (x.split() for x in Path('/proc/vmstat').read_text().splitlines()) if k in ('pswpin','pswpout')}
    return {**status,**vm,'cgroup_memory_current':number('memory.current'),'cgroup_memory_peak':number('memory.peak'),
            'cgroup_swap_current':number('memory.swap.current'),'cgroup_swap_peak':number('memory.swap.peak')}


class Runtime:
    def __init__(self, source, revision=None, precision='bf16', adapter=None, resolution=512):
        assert torch.cuda.is_available()
        self.torch=torch
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        self.resolution=resolution
        self.source=str(source)
        self.file_cache_note='OS page cache not dropped; first load in this process, not guaranteed cold disk cache'
        self.precision=precision
        self.runtime_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        started=time.perf_counter()
        self.before_load=memory()
        self.processor=AutoProcessor.from_pretrained(source,revision=revision,local_files_only=True)
        kwargs=dict(revision=revision,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
        config=json.loads((Path(source)/'config.json').read_text()) if Path(source).is_dir() else {}
        if precision=='nf4' and not config.get('quantization_config'):
            kwargs['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,llm_int8_skip_modules=['visual','lm_head'])
        self.model=Qwen3VLForConditionalGeneration.from_pretrained(source,**kwargs).eval()
        self.quantized_module_names=[n for n,m in self.model.named_modules() if m.__class__.__name__=='Linear4bit']
        if precision=='nf4':
            assert self.quantized_module_names, 'Expected restored NF4 modules'
            assert not any('visual' in n for n in self.quantized_module_names), 'Visual tower precision changed'
        self.adapter=str(adapter) if adapter else None
        if adapter:
            from peft import PeftModel
            self.model=PeftModel.from_pretrained(self.model,str(adapter)).eval()
        torch.cuda.synchronize()
        self.load_seconds=time.perf_counter()-started
        self.after_load=memory()
        self.stop=threading.Event()
        self.samples=[]
        def sample():
            while not self.stop.wait(.1): self.samples.append({'time_unix':time.time(),**memory()})
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start()

    def generate(self,images,prompt,max_new_tokens=40):
        from PIL import Image
        images=images if isinstance(images,list) else [images]
        images=[im.resize((self.resolution,self.resolution),Image.Resampling.BICUBIC) if im.size!=(self.resolution,self.resolution) else im for im in images]
        messages=[{'role':'user','content':[*[{'type':'image','image':im} for im in images],{'type':'text','text':prompt}]}]
        inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda:0')
        torch.cuda.synchronize(); started=time.perf_counter()
        with torch.inference_mode(): out=self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False)
        torch.cuda.synchronize()
        tokens=out[0,inputs['input_ids'].shape[-1]:]
        m=memory();self.samples.append({'time_unix':time.time(),**m})
        return {'raw':self.processor.decode(tokens,skip_special_tokens=True),'seconds':time.perf_counter()-started,
                'metadata':{'output_token_ids':tokens.tolist(),'input_token_sha256':hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest(),
                            'pixel_values_sha256':hashlib.sha256(inputs['pixel_values'].cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),
                            'image_grid_thw':inputs['image_grid_thw'].cpu().tolist(),'resolution':self.resolution,
                            'input_tokens':int(inputs['input_ids'].shape[-1]),'output_tokens':int(tokens.numel()),'memory':m}}

    def report(self):
        self.stop.set();self.thread.join(timeout=1)
        return {'source':self.source,'runtime_source_sha256':self.runtime_source_sha256,'quantized_module_names':self.quantized_module_names,'adapter':self.adapter,'precision':self.precision,'resolution':self.resolution,
                'load_seconds':self.load_seconds,'file_cache_note':self.file_cache_note,'before_load':self.before_load,'after_load':self.after_load,
                'stable_samples':self.samples,'stable_max_swap_bytes':max([x.get('cgroup_swap_current') or 0 for x in self.samples]+[0]),
                'peak_cuda_allocated':torch.cuda.max_memory_allocated(),'peak_cuda_reserved':torch.cuda.max_memory_reserved(),
                'versions':{x:importlib.metadata.version(x) for x in ['torch','torchvision','transformers','bitsandbytes','peft','pillow']}}
