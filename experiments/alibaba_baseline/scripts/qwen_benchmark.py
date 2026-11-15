"""Instruction-conditioned observation evaluation; never sends robot actions."""
import json,time,sys,csv
from pathlib import Path
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/'experiments/alibaba_baseline'
sys.path.insert(0,str(ROOT/'perception/qwen_scene'))
from engine import SceneEngine,MODEL_ID,REVISION
out=TASK/'runs/qwen_instruction_001';out.mkdir(parents=True,exist_ok=False)
cases=json.loads((TASK/'data/qwen_cases.json').read_text());engine=SceneEngine();records=[]
(out/'config.json').write_text(json.dumps({'model':MODEL_ID,'revision':REVISION,'do_sample':False,'max_new_tokens':256,'case_count':len(cases),'reference_labels':'provisional independent visual review'},indent=2))
for case in cases:
 image=Image.open(TASK/'data'/case['image']).convert('RGB')
 prompt=f'''你是机械臂的高层视觉观察模块。当前指令：{case['instruction']}
只根据本图可见证据判断指令指定的目标是否可见。不能因指令提及目标就假定它存在。
仅输出JSON：{{"target_visible":true或false,"next_subtask":"locate"或"observe","reason":"简短中文证据"}}。
可见时下一子任务为locate；不可见时为observe。不要输出运动指令或假定三维距离。'''
 messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
 torch.cuda.synchronize();start=time.perf_counter()
 inputs=engine.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda:0')
 with torch.inference_mode():generated=engine.model.generate(**inputs,max_new_tokens=256,do_sample=False)
 torch.cuda.synchronize();seconds=time.perf_counter()-start
 raw=engine.processor.decode(generated[0][inputs['input_ids'].shape[-1]:],skip_special_tokens=True)
 row={**case,'raw_output':raw,'seconds':seconds,'valid':False,'prediction':None}
 try:
  text=raw.strip()
  if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
  prediction=json.loads(text)
  assert type(prediction['target_visible']) is bool and prediction['next_subtask'] in ('locate','observe') and isinstance(prediction['reason'],str)
  row.update(valid=True,prediction=prediction,presence_correct=prediction['target_visible']==case['target_visible'],subtask_correct=prediction['next_subtask']==case['expected_next_subtask'])
 except (ValueError,KeyError,AssertionError,TypeError):row.update(presence_correct=False,subtask_correct=False)
 records.append(row)
 with (out/'predictions.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 (out/(case['id']+'_prompt.txt')).write_text(prompt)
 print(f"{len(records)}/{len(cases)} {case['id']} valid={row['valid']} correct={row['presence_correct']} {seconds:.2f}s",flush=True)
import numpy as np
metrics={'cases':len(records),'schema_valid_fraction':float(np.mean([r['valid'] for r in records])),'provisional_presence_accuracy':float(np.mean([r['presence_correct'] for r in records])),'provisional_subtask_accuracy':float(np.mean([r['subtask_correct'] for r in records])),'seconds_p50':float(np.median([r['seconds'] for r in records])),'seconds_p95':float(np.percentile([r['seconds'] for r in records],95)),'groups':{},'notes':'One episode, provisional visual-review labels; not formal generalization accuracy.'}
for group in sorted({r['query_group'] for r in records}):
 selected=[r for r in records if r['query_group']==group]
 positive=[r for r in selected if r['target_visible']];negative=[r for r in selected if not r['target_visible']]
 metrics['groups'][group]={'cases':len(selected),'correct':sum(r['presence_correct'] for r in selected),'positive_recall':float(np.mean([r['presence_correct'] for r in positive])) if positive else None,'negative_recall':float(np.mean([r['presence_correct'] for r in negative])) if negative else None}
(out/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False))
print(json.dumps(metrics,indent=2))
