import sys,json
from pathlib import Path
import torch
from PIL import Image
from peft import PeftModel
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/'experiments/alibaba_baseline';sys.path.insert(0,str(ROOT/'perception/qwen_scene'))
from engine import SceneEngine
engine=SceneEngine();model=PeftModel.from_pretrained(engine.model,TASK/'runs/qwen_lora_001/adapter').eval()
image=Image.open(TASK/'data/qwen_frames/frame_0180.png').convert('RGB')
prompt=(TASK/'runs/qwen_instruction_001/0180_blue_case_prompt.txt').read_text()
inputs=engine.processor.apply_chat_template([{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=128,do_sample=False)
raw=engine.processor.decode(generated[0][inputs['input_ids'].shape[-1]:],skip_special_tokens=True)
print(raw)
(TASK/'runs/qwen_lora_001/reload_check.json').write_text(json.dumps({'adapter_loaded_from_disk':True,'image':'frame_0180.png','raw_output':raw},ensure_ascii=False,indent=2))

answer=json.loads(raw.strip());assert type(answer['target_visible']) is bool and answer['next_subtask'] in ('locate','observe') and isinstance(answer['reason'],str)
print('Saved adapter reload and structured inference passed')
