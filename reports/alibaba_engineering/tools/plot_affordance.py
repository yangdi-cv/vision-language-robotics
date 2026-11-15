"""Draw the first matched saved test example; no inference or sample selection by score."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parents[1];R=D.parents[1];Q=R/'experiments/mobile_qwen/affordance'
lora=json.loads((Q/'eval_lora_v1_test.json').read_text());row=next(x for x in lora if x['query']=='handle')
base=next(x for x in json.loads((Q/'eval_base_v2.json').read_text()) if x['id']==row['id'] and x['query']=='handle')
label=next(json.loads(x) for x in (Q/'data_v1/labels.jsonl').read_text().splitlines() if json.loads(x)['id']==row['id'])
image=Image.open(Q/'data_v1/rgb'/f"{row['id']}.png");fig,axs=plt.subplots(1,3,figsize=(10,3.4))
for ax,name,pred in zip(axs,['Input','Base','LoRA'],[None,base,row]):
 ax.imshow(image);ax.set_title(name);ax.axis('off')
 if pred:
  uv=np.array(pred['parsed'])/1000*512;gt=label['handle']['uv'];ax.scatter(*gt,marker='+',color='white',s=100,linewidths=1.7);ax.scatter(*uv,facecolors='none',edgecolors='white',s=110,linewidths=1.5)
fig.tight_layout();out=D/'figures/ch06/s17_affordance_example.png';out.parent.mkdir(exist_ok=True);fig.savefig(out,dpi=160,facecolor='white');plt.close(fig);print(row['id'])
