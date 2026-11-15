"""Persistent local Qwen process for a constrained skill-selection interface."""
import sys,json,time,contextlib
from pathlib import Path
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'perception/qwen_scene'))
from engine import SceneEngine
with contextlib.redirect_stdout(sys.stderr):engine=SceneEngine()
print(json.dumps({'ready':True}),flush=True)
for line in sys.stdin:
 request=json.loads(line)
 if request.get('stop'):break
 image=Image.open(request['image']).convert('RGB')
 prompt=f'''当前指令：{request['instruction']}
这是双臂桌面机器人场景。仅根据图中可见证据选择目标和技能。
唯一可执行技能是transfer_red_cube：抓起红色方块并交给另一只机械臂。
如果指令要求红色方块且它可见，选择transfer_red_cube；否则选择observe，不要替换成其他目标。
只输出JSON：{{"target_visible":true或false,"selected_skill":"transfer_red_cube"或"observe","reason":"中文证据"}}。
不要输出关节目标、距离或假定未出现的物体存在。'''
 messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
 torch.cuda.synchronize();started=time.perf_counter()
 inputs=engine.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
 with torch.inference_mode():output=engine.model.generate(**inputs,max_new_tokens=128,do_sample=False)
 torch.cuda.synchronize();raw=engine.processor.decode(output[0][inputs['input_ids'].shape[-1]:],skip_special_tokens=True)
 result={'raw':raw,'seconds':time.perf_counter()-started,'valid':False,'goal':None,'prompt':prompt}
 try:
  text=raw.strip()
  if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
  goal=json.loads(text);assert type(goal['target_visible']) is bool and goal['selected_skill'] in ('transfer_red_cube','observe')
  result.update(valid=True,goal=goal)
 except (ValueError,KeyError,AssertionError,TypeError):pass
 print(json.dumps(result,ensure_ascii=False),flush=True)
