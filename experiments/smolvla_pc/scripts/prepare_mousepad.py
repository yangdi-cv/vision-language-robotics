"""Audit raw demonstrations; trim only leading/trailing idle; retain originals."""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3];DATA=ROOT/'datasets/earbud_mousepad';OUT=DATA/'prepared_v1';OUT.mkdir(exist_ok=True)
summary=[]
for p in sorted(DATA.glob('episode_0[123]')):
 rows=[json.loads(s) for s in (p/'trajectory.jsonl').read_text().splitlines()];q=np.array([r['state'] for r in rows]);a=np.array([r['action'] for r in rows])
 # Five-frame changes suppress tiny servo noise; preserve 2 seconds on both ends.
 motion=np.max(abs(q[10:,:5]-q[:-10,:5]),axis=1);active=np.flatnonzero(motion>5)+10
 active=active[active>30]  # Ignore startup alignment; retain 2s vertical context.
 start=max(0,int(active[0])-20) if len(active) else 0;end=min(len(rows),int(active[-1])+21) if len(active) else len(rows)
 kept=rows[start:end]
 (OUT/(p.name+'.json')).write_text(json.dumps({'source':str(p.relative_to(ROOT)),'start':start,'end_exclusive':end,'rows':kept},ensure_ascii=False))
 canvas=Image.new('RGB',(4*320,3*265),'#182333');d=ImageDraw.Draw(canvas)
 indices=np.linspace(start,end-1,12).astype(int)
 for k,i in enumerate(indices):
  im=Image.open(p/rows[i]['image']).convert('RGB').resize((320,240));x=k%4*320;y=k//4*265;canvas.paste(im,(x,y));d.text((x+5,y+242),f"{p.name} frame {i} t={(rows[i]['state_ns']-rows[0]['state_ns'])/1e9:.1f}s",fill='white')
 canvas.save(OUT/(p.name+'_review.jpg'))
 lag=np.array([(r['sent_ns']-r['image_ns'])/1e6 for r in rows]);state_image=np.array([(r['image_ns']-r['state_ns'])/1e6 for r in rows])
 info={'episode':p.name,'raw_frames':len(rows),'keep_start':start,'keep_end_exclusive':end,'kept_frames':len(kept),'state_image_receipt_delta_ms_median':float(np.median(state_image)),'image_to_sent_ms_median':float(np.median(lag)),'gripper_min_max':[float(q[:,5].min()),float(q[:,5].max())],'raw_sha256':hashlib.sha256((p/'trajectory.jsonl').read_bytes()).hexdigest(),'success_label':'operator_reported; contact sheets require review'}
 summary.append(info)
(OUT/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,indent=2))
