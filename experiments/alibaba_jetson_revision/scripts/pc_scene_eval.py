"""PC 4B scene observations on the exact saved Jetson images; not a compression comparison."""
from pathlib import Path
import json,time,torch,argparse
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
R=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=R/'runs/jetson');parser.add_argument('--out',type=Path,default=R/'runs/pc_scene');args=parser.parse_args()
O=args.out;O.mkdir(parents=True,exist_ok=False)
model_id='Qwen/Qwen3-VL-4B-Instruct';revision='ebb281ec70b05090aa6165b016eac8ec08e71b17'
p=AutoProcessor.from_pretrained(model_id,revision=revision,local_files_only=True)
m=Qwen3VLForConditionalGeneration.from_pretrained(model_id,revision=revision,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
prompt='Observe only this image. Return a compact JSON object with keys visible_objects (short English list), gripper_visible (boolean), target_visible (boolean or null). The target is a dark blue plastic earbud case. Do not infer objects outside the image. Do not output commands.'
rows=[]
for folder,expected in [('scene',True),('scene_absent',False),('scene_shifted',True),('scene_partial',True)]:
 for index in [5,10]:
  image=args.data/folder/f'frame_{index:03}.jpg'
  b=p.apply_chat_template([{'role':'user','content':[{'type':'image','image':str(image)},{'type':'text','text':prompt}]}],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda:0')
  with torch.inference_mode():v=m.generate(**b,max_new_tokens=64,do_sample=False)
  tokens=v[0][len(b.input_ids[0]):];text=p.decode(tokens,skip_special_tokens=True)
  try:parsed=json.loads(text[text.index('{'):text.rindex('}')+1])
  except Exception:parsed=None
  rows.append({'image':str(image),'expected_target_visible':expected,'raw':text,'parsed':parsed,'tokens':len(tokens),'target_visibility_correct':parsed.get('target_visible') is expected if parsed else False})
  (O/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print(rows[-1],flush=True)
(O/'runtime.json').write_text(json.dumps({'model':model_id,'revision':revision,'dtype':'BF16','latency_not_compared':'PC GPU shared with ACT training; accuracy examples only','scene_conditions':4,'frames_per_scene':2},indent=2))
