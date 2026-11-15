"""Fresh-process serialization/reload check for saved mixed language-INT8 Qwen."""
import json,time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from rgbd_protocol import prompt
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/'experiments/alibaba_extension';path=TASK/'runs/profile_int8_cache/quantized_checkpoint';data=TASK/'data/rgbd_cases'
model=Qwen3VLForConditionalGeneration.from_pretrained(path,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True).eval();processor=AutoProcessor.from_pretrained(path,local_files_only=True);case=json.loads((data/'cases.json').read_text())[0]
messages=[{'role':'user','content':[{'type':'image','image':Image.open(data/case['image']).convert('RGB')},{'type':'text','text':prompt(case)}]}]
inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=120,do_sample=False)
raw=processor.decode(output[0][inputs['input_ids'].shape[-1]:],skip_special_tokens=True);prediction=json.loads(raw.strip().removeprefix('```json').removesuffix('```').strip());assert prediction['target_visible'] is True and prediction['selected_skill']=='transfer_red_cube'
result={'fresh_process_reloaded':True,'quantized_linear_modules':sum(type(m).__name__=='Linear8bitLt' for m in model.modules()),'prediction':prediction,'base_weight_download_not_used_for_reload':True,'notes':'Single-case saved model compatibility smoke test; no latency comparison'}
(TASK/'runs/profile_int8_cache/reload_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
