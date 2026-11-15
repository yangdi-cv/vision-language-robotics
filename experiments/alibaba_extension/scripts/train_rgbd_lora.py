"""Scene-disjoint synthetic target localization/skill LoRA, no physical motion."""
import argparse,json,time,random,csv,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image
from peft import LoraConfig,get_peft_model,TaskType
from rgbd_protocol import prompt,answer
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'perception/qwen_scene'));from engine import SceneEngine
TASK=ROOT/'experiments/alibaba_extension';p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=240);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
random.seed(20261013);torch.manual_seed(20261013);torch.set_num_threads(8);cases=json.loads((TASK/'data/rgbd_cases/cases.json').read_text());train=[c for c in cases if c['scene']<18]
engine=SceneEngine();model=get_peft_model(engine.model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=16,lora_dropout=.05,target_modules=r'.*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$'));model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
assert not any(p.requires_grad for name,p in model.named_parameters() if '.visual.' in name)
optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5);started=time.perf_counter()
def sample(case):
 image=Image.open(TASK/'data/rgbd_cases'/case['image']).convert('RGB');messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt(case)}]}]
 prefix=engine.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt');n=prefix['input_ids'].shape[-1]
 messages.append({'role':'assistant','content':[{'type':'text','text':json.dumps(answer(case),ensure_ascii=False)}]});full=engine.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,return_dict=True,return_tensors='pt')
 assert torch.equal(full['input_ids'][:,:n],prefix['input_ids']);labels=full['input_ids'].clone();labels[:,:n]=-100;full['labels']=labels;return full.to('cuda')
with (a.out/'training.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['step','loss','seconds'])
 for step in range(1,a.steps+1):
  model.train();optimizer.zero_grad(set_to_none=True);loss=model(**sample(random.choice(train))).loss
  assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
  if step%20==0:w.writerow([step,float(loss),time.perf_counter()-started]);f.flush();print('LoRA',step,float(loss),flush=True)
model.save_pretrained(a.out/'adapter');engine.processor.save_pretrained(a.out/'adapter')
config={'steps':a.steps,'lr':5e-5,'r':8,'alpha':16,'dropout':.05,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'train_scenes':list(range(18)),'validation_scenes':list(range(18,24)),'test_scenes':list(range(24,30)),'train_cases':72,'validation_cases':24,'test_cases':24,'labels':'paired material difference renderer labels, not VLM labels','wall_seconds':time.perf_counter()-started,'vision_frozen':True,'notes':'Synthetic single-cube scene adaptation; no evidence of real-world generalization; base LoRA pilot not reused'}
(a.out/'training_config.json').write_text(json.dumps(config,indent=2));print(json.dumps(config,indent=2),flush=True)
