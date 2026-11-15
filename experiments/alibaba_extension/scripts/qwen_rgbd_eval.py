"""Evaluate independent RGB-D simulator cases with true masks and optional language INT8."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
from rgbd_protocol import prompt as make_prompt
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration,BitsAndBytesConfig
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'perception/qwen_scene'));from engine import MODEL_ID,REVISION
TASK=ROOT/'experiments/alibaba_extension';p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--precision',choices=['bf16','int8'],default='bf16');p.add_argument('--save-model',action='store_true');p.add_argument('--adapter',type=Path);p.add_argument('--heldout-only',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--disable-kv-cache',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(8);started=time.perf_counter();processor=AutoProcessor.from_pretrained(MODEL_ID,revision=REVISION,local_files_only=True)
kwargs={'dtype':torch.bfloat16,'device_map':{'':'cuda:0'},'attn_implementation':'sdpa'}
if a.precision=='int8':kwargs['quantization_config']=BitsAndBytesConfig(load_in_8bit=True,llm_int8_skip_modules=['visual','lm_head'])
model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID,revision=REVISION,local_files_only=True,**kwargs).eval();loadtime=time.perf_counter()-started
if a.adapter:
 from peft import PeftModel
 model=PeftModel.from_pretrained(model,a.adapter).eval()
baseline_memory=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();rows=[];data=TASK/'data/rgbd_cases';cases=json.loads((data/'cases.json').read_text())
if a.heldout_only:cases=[c for c in cases if c['scene']>=18]
if a.limit:cases=cases[:a.limit]
def iou(pred,ref):
 x1=max(pred[0],ref[0]);y1=max(pred[1],ref[1]);x2=min(pred[2],ref[2]);y2=min(pred[3],ref[3]);inter=max(0,x2-x1)*max(0,y2-y1);union=(pred[2]-pred[0])*(pred[3]-pred[1])+(ref[2]-ref[0])*(ref[3]-ref[1])-inter
 return inter/union if union>0 else 0
with (a.out/'predictions.jsonl').open('w') as log:
 for case in cases:
  image=Image.open(data/case['image']).convert('RGB')
  prompt=make_prompt(case)
  messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
  torch.cuda.synchronize();tick=time.perf_counter();inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
  with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=120,do_sample=False,use_cache=not a.disable_kv_cache)
  torch.cuda.synchronize();elapsed=time.perf_counter()-tick;raw=processor.decode(output[0][inputs['input_ids'].shape[-1]:],skip_special_tokens=True)
  row={'case':case,'prompt':prompt,'raw':raw,'seconds':elapsed,'schema_valid':False,'presence_correct':False,'skill_correct':False,'iou':None,'predicted_box_depth_m':None,'depth_abs_error_m':None,'generated_tokens':int(output.shape[1]-inputs['input_ids'].shape[1])}
  try:
   text=raw.strip()
   if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
   goal=json.loads(text);assert type(goal['target_visible']) is bool and goal['selected_skill'] in ('transfer_red_cube','observe')
   box=goal['bbox_1000'];assert box is None or (isinstance(box,list) and len(box)==4 and all(isinstance(v,(int,float)) and 0<=v<=1000 for v in box) and box[0]<box[2] and box[1]<box[3])
   assert goal['target_visible']==(box is not None)
   row.update(schema_valid=True,prediction=goal,presence_correct=goal['target_visible']==case['target_visible'],skill_correct=goal['selected_skill']==case['expected_skill'])
   if box is not None:
    pixels=[int(box[0]*640/1000),int(box[1]*480/1000),int(box[2]*640/1000),int(box[3]*480/1000)];row['predicted_box_pixels']=pixels
    if case['target_visible']:
     row['iou']=iou(pixels,case['bbox_xyxy_pixels']);depth=np.load(data/f"depth_{case['scene']:03d}.npy");cx=(pixels[0]+pixels[2])//2;cy=(pixels[1]+pixels[3])//2;patch=depth[max(0,cy-2):min(480,cy+3),max(0,cx-2):min(640,cx+3)];row['predicted_box_depth_m']=float(np.median(patch));row['depth_abs_error_m']=abs(row['predicted_box_depth_m']-case['depth_m'])
    if case['query_color']=='red' and case['scene']<6:
     draw=ImageDraw.Draw(image);draw.rectangle(pixels,outline='yellow',width=2)
     if case['bbox_xyxy_pixels']:draw.rectangle(case['bbox_xyxy_pixels'],outline='lime',width=2)
     image.save(a.out/f"overlay_{case['id']}.png")
  except (ValueError,TypeError,KeyError,AssertionError) as error:row['error']=str(error)
  log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush();rows.append(row)
  if len(rows)%20==0:print(a.precision,'cases',len(rows),flush=True)
pos=[r for r in rows if r['case']['target_visible']];detected=[r for r in pos if r['iou'] is not None];times=[r['seconds'] for r in rows[1:]]
metrics={'precision':a.precision,'use_cache':not a.disable_kv_cache,'adapter':str(a.adapter) if a.adapter else None,'cases':len(rows),'schema_valid':sum(r['schema_valid'] for r in rows),'presence_correct':sum(r['presence_correct'] for r in rows),'skill_correct':sum(r['skill_correct'] for r in rows),'positive_cases':len(pos),'positive_correct':sum(r['presence_correct'] for r in pos),'bbox_iou_ge_05_over_all_positive':sum(r['iou'] is not None and r['iou']>=.5 for r in pos),'mean_iou_detected_positive':float(np.mean([r['iou'] for r in detected])) if detected else None,'depth_mae_detected_positive_m':float(np.mean([r['depth_abs_error_m'] for r in detected])) if detected else None,'negation_cases':sum(r['case']['negated'] for r in rows),'negation_skill_correct':sum(r['skill_correct'] for r in rows if r['case']['negated']),'seconds_p50_excluding_first':float(np.median(times)),'seconds_p95_excluding_first':float(np.percentile(times,95)),'model_load_seconds':loadtime,'allocated_after_load_gib':baseline_memory/1024**3,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'model_memory_footprint_gib':model.get_memory_footprint()/1024**3,'gpu':torch.cuda.get_device_name(),'base_revision':REVISION,'quantized_visual_linear_modules':sum(type(module).__name__=='Linear8bitLt' for name,module in model.named_modules() if '.visual.' in name),'quantized_linear_modules':sum(type(module).__name__=='Linear8bitLt' for module in model.modules()),'notes':'Synthetic independent scenes, single cube. RGB to Qwen, metric simulated depth sampled by predicted box center. No robot motion or real depth calibration; timing may include concurrent GPU workload unless explicitly recorded otherwise.'}
(a.out/'metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2),flush=True)
if a.save_model:model.save_pretrained(a.out/'quantized_checkpoint',safe_serialization=True);processor.save_pretrained(a.out/'quantized_checkpoint')
