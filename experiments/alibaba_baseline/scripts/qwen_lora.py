"""Bounded LoRA pilot with frame-disjoint splits; one-video limitations explicit."""
import csv,json,random,sys,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
from peft import LoraConfig,get_peft_model,TaskType
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/'experiments/alibaba_baseline'
sys.path.insert(0,str(ROOT/'perception/qwen_scene'))
from engine import SceneEngine,MODEL_ID,REVISION
out=TASK/'runs/qwen_lora_001';out.mkdir(parents=True,exist_ok=False)
random.seed(2026);torch.manual_seed(2026);torch.set_num_threads(8)
cases={r['id']:r for r in json.loads((TASK/'data/qwen_cases.json').read_text())};split=json.loads((TASK/'data/qwen_split.json').read_text())
engine=SceneEngine();model=engine.model
lora=LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=16,lora_dropout=.05,target_modules=r'.*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$')
model=get_peft_model(model,lora)
model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
assert trainable>0 and not any(p.requires_grad for name,p in model.named_parameters() if '.visual.' in name)
print('Trainable LoRA parameters:',trainable,flush=True)

def prompt(case):
 return f'''你是机械臂的高层视觉观察模块。当前指令：{case['instruction']}
只根据本图可见证据判断指令指定的目标是否可见。不能因指令提及目标就假定它存在。
仅输出JSON：{{"target_visible":true或false,"next_subtask":"locate"或"observe","reason":"简短中文证据"}}。
可见时下一子任务为locate；不可见时为observe。不要输出运动指令或假定三维距离。'''

def preprocess(case,training):
 image=Image.open(TASK/'data'/case['image']).convert('RGB')
 messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt(case)}]}]
 prefix=engine.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
 if not training:return prefix
 answer=json.dumps({'target_visible':case['target_visible'],'next_subtask':case['expected_next_subtask'],'reason':'画面中可见指令指定目标。' if case['target_visible'] else '画面中没有可见的指令目标。'},ensure_ascii=False)
 messages.append({'role':'assistant','content':[{'type':'text','text':answer}]})
 full=engine.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,return_dict=True,return_tensors='pt')
 length=prefix['input_ids'].shape[-1]
 assert torch.equal(prefix['input_ids'],full['input_ids'][:,:length]),'Chat prefix mismatch'
 labels=full['input_ids'].clone();labels[:,:length]=-100;full['labels']=labels
 return full

train=[preprocess(cases[name],True) for name in split['train']]
optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5)
config={'base_model':MODEL_ID,'revision':REVISION,'r':8,'alpha':16,'target_modules':lora.target_modules,'updates':120,'gradient_accumulation':2,'learning_rate':5e-5,'seed':2026,'trainable_parameters':trainable,'train_cases':len(split['train']),'validation_cases':len(split['validation']),'test_cases':len(split['test']),'split_file':'data/qwen_split.json','labels':'provisional visual review, not Qwen predictions','limitations':'All frames in one correlated video; not multi-scene generalization.'}
(out/'training_config.json').write_text(json.dumps(config,indent=2));started=time.perf_counter()
with (out/'training.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['update','wall_seconds','cross_entropy'])
 for update in range(1,121):
  model.train();optimizer.zero_grad(set_to_none=True);losses=[]
  for _ in range(2):
   sample={k:v.to('cuda') for k,v in random.choice(train).items()}
   result=model(**sample);loss=result.loss
   assert torch.isfinite(loss);(loss/2).backward();losses.append(float(loss.item()))
  torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);optimizer.step()
  writer.writerow([update,time.perf_counter()-started,float(np.mean(losses))]);f.flush()
  if update%10==0:print('update',update,'loss',round(float(np.mean(losses)),4),'seconds',round(time.perf_counter()-started,1),flush=True)
model.save_pretrained(out/'adapter');engine.processor.save_pretrained(out/'adapter');model.eval()
model.gradient_checkpointing_disable();model.config.use_cache=True
results=[]
for split_name in ('validation','test'):
 for name in split[split_name]:
  case=cases[name];inputs={k:v.to('cuda') for k,v in preprocess(case,False).items()}
  torch.cuda.synchronize();tick=time.perf_counter()
  with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=128,do_sample=False)
  torch.cuda.synchronize();seconds=time.perf_counter()-tick
  raw=engine.processor.decode(output[0][inputs['input_ids'].shape[-1]:],skip_special_tokens=True)
  row={'id':name,'split':split_name,'raw':raw,'seconds':seconds,'valid':False,'correct':False}
  try:
   text=raw.strip()
   if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
   answer=json.loads(text);assert type(answer['target_visible']) is bool
   row.update(valid=True,correct=answer['target_visible']==case['target_visible'],subtask_correct=answer['next_subtask']==case['expected_next_subtask'])
  except (ValueError,KeyError,AssertionError,TypeError):pass
  results.append(row)
  with (out/'predictions.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
base={r['id']:r for r in [json.loads(line) for line in (TASK/'runs/qwen_instruction_001/predictions.jsonl').read_text().splitlines()]}
summary={'updates':120,'wall_seconds':time.perf_counter()-started,'splits':{},'notes':'Pilot tuning on one video and provisional labels. Changes are not evidence of generalization.'}
for name in ('validation','test'):
 rows=[r for r in results if r['split']==name]
 summary['splits'][name]={'cases':len(rows),'base_correct':sum(base[r['id']]['presence_correct'] for r in rows),'lora_correct':sum(r['correct'] for r in rows),'schema_valid':sum(r['valid'] for r in rows)}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
